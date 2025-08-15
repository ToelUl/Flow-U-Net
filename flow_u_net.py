import math
from typing import Optional, Sequence, Tuple
import unittest
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
import logging

torch.set_float32_matmul_precision('high')
sympy_interp_logger = logging.getLogger("torch.utils._sympy.interp")
sympy_interp_logger.setLevel(logging.ERROR)


# ==============================================================================
# Helper Function
# ==============================================================================
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Applies a learned affine transformation to the input tensor.

    This function is used for Adaptive LayerNorm (adaLN), where the input
    tensor `x` is modulated by a shift and a scale vector derived from an
    embedding (e.g., a time embedding).

    Args:
        x: The input tensor. 4D (B, C, H, W).
        shift: The shift vector, broadcastable to `x`.
        scale: The scale vector, broadcastable to `x`.

    Returns:
        The modulated tensor.
    """
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


# ==============================================================================
# Custom Normalization Layers
# ==============================================================================
class RMSNorm2d(nn.Module):
    """A 2D Root Mean Square Layer Normalization module.

    This normalization is applied across the channel dimension of a 4D tensor.

    Attributes:
        eps: A small value added to the denominator for numerical stability.
        elementwise_affine: If True, this module has learnable per-channel
            affine parameters.
        scale: The learnable per-channel scaling parameter.
    """
    def __init__(self, num_features: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """Initializes the RMSNorm2d module.

        Args:
            num_features: The number of channels in the input tensor.
            eps: A value added to the denominator for numerical stability.
            elementwise_affine: A boolean value that when set to True, this
                module has learnable per-channel affine parameters.
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.scale = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter('scale', None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RMSNorm2d.

        Args:
            x: The input tensor of shape (B, C, H, W).

        Returns:
            The normalized tensor.
        """
        norm_factor = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x_norm = x * norm_factor
        if self.elementwise_affine:
            scale_reshaped = self.scale.view(1, -1, 1, 1)
            return x_norm * scale_reshaped
        return x_norm


# ==============================================================================
# Core Modules
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    """Creates sinusoidal positional embeddings.

    Attributes:
        embedding_dim: The dimension of the embedding.
        max_period: The maximum period of the sinusoidal functions.
        factor: A scaling factor for the input before embedding.
    """
    def __init__(self, embedding_dim: int, max_period: int = 10000, factor: float = 1.0):
        """Initializes the SinusoidalPosEmb module.

        Args:
            embedding_dim: The desired dimension of the embedding.
            max_period: The maximum period for the sinusoidal functions.
            factor: A multiplicative factor applied to the input tensor.
        """
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
        # Ensure embedding_dim is even
        self.max_period = max_period
        self.embedding_dim = embedding_dim if embedding_dim % 2 == 0 else embedding_dim + 1
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        """Generates sinusoidal embeddings for the input tensor.

        Args:
            x: A 1D tensor of values to be embedded.

        Returns:
            A 2D tensor of shape (len(x), embedding_dim) containing the embeddings.
        """
        args = x.float().unsqueeze(1) * self.freqs.unsqueeze(0) * self.factor
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLPEmbedder(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) embedder for input features.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp_emb_in_layer = nn.Linear(in_dim, 4 * in_dim, bias=True)
        self.mlp_emb_act = nn.SiLU()
        self.mlp_emb_out_layer = nn.Linear(4 * in_dim, in_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.mlp_emb_in_layer(x)
        h = self.mlp_emb_act(h)
        h = self.mlp_emb_out_layer(h)
        return h


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    This function is a core component of Rotary Position Embedding. It splits the
    last dimension of the input tensor into two halves, negates the second half,
    and then concatenates them in a swapped order.

    Args:
        x (torch.Tensor): The input tensor. Can be of any shape, but the last
            dimension must be even.

    Returns:
        torch.Tensor: The tensor with half of its last dimension values rotated.
    """
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_rotated_pairs = torch.cat(
        [-x_reshaped[..., 1:], x_reshaped[..., :1]], dim=-1
    )
    return x_rotated_pairs.flatten(start_dim=-2)


class RoPE(nn.Module):
    """Implements the Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even for 1D RoPE, but got {dim}.")
        self.dim = dim
        self.base = base
        freqs = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (self.base**freqs)
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        device = q.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos_vals = freqs.cos()
        sin_vals = freqs.sin()
        cos = cos_vals.repeat_interleave(2, dim=-1).unsqueeze(0)
        sin = sin_vals.repeat_interleave(2, dim=-1).unsqueeze(0)
        rotated_q = q * cos + rotate_half(q) * sin
        rotated_k = k * cos + rotate_half(k) * sin
        return rotated_q, rotated_k


class RoPEMixed(nn.Module):
    """Implements the mixed-axis Rotary Position Embedding (RoPE-Mixed)."""
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, but got {dim}")
        self.freqs = nn.Parameter(torch.randn(dim // 2, 2))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        H, W, D = q.shape[-3:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=self.freqs.dtype),
            torch.arange(W, device=device, dtype=self.freqs.dtype),
            indexing="ij",
        )
        positions_2d = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        theta_y = self.freqs[:, 0]
        theta_x = self.freqs[:, 1]
        angles = torch.einsum("n,d->nd", positions_2d[:, 0], theta_y) + torch.einsum(
            "n,d->nd", positions_2d[:, 1], theta_x
        )
        cos_vals = angles.cos()
        sin_vals = angles.sin()
        cos = cos_vals.repeat_interleave(2, dim=-1).reshape(H, W, D)
        sin = sin_vals.repeat_interleave(2, dim=-1).reshape(H, W, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        rotated_q = q * cos + rotate_half(q) * sin
        rotated_k = k * cos + rotate_half(k) * sin
        return rotated_q, rotated_k

@torch.compile
class ConditionalMSAWithRoPE(nn.Module):
    """A Multi-Head Self-Attention block for fusing conditional embeddings with RoPE."""
    def __init__(self, dim: int, num_heads: int, seq_len: int, qkv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Dimension ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = dim // num_heads
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.seq_expand_proj = nn.Linear(seq_len, dim,)
        self.qkv_projection = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rope = RoPE(dim=self.head_dim)
        self.seq_combine_proj = nn.Linear(dim, 1, bias=False)

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(embeddings, dim=1)
        B, S, D = x.shape
        x = self.seq_expand_proj(x.permute(0, 2, 1))
        x = self.norm(x)
        qkv = self.qkv_projection(x.permute(0, 2, 1))
        qkv = qkv.reshape(B, self.dim, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_orig_shape, k_orig_shape = q.shape, k.shape
        q_reshaped = q.reshape(B * self.num_heads, self.dim, self.head_dim)
        k_reshaped = k.reshape(B * self.num_heads, self.dim, self.head_dim)
        q_rot, k_rot = self.rope(q_reshaped, k_reshaped)
        q = q_rot.view(q_orig_shape)
        k = k_rot.view(k_orig_shape)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.contiguous().view(B, D, self.dim)
        processed_sequence = self.seq_combine_proj(attn_output).squeeze(-1)
        return processed_sequence

@torch.compile
class MSAWithRoPE(nn.Module):
    """A Multi-Head Self-Attention module integrated with RoPE-Mixed."""
    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 qkv_bias: bool = False,
                 padding_mode: str = "circular",
                 ):
        super().__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})."
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim % 2 == 0, f"Head dimension ({self.head_dim}) must be even for RoPE-Mixed."
        self.qkv_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=3, padding=1, bias=qkv_bias,
            padding_mode=padding_mode, groups=channels
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.rope = RoPEMixed(dim=self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv_projection(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(B, self.num_heads, self.head_dim, H, W), qkv)
        q_for_rope = self.q_norm(q.permute(0, 1, 3, 4, 2))
        k_for_rope = self.k_norm(k.permute(0, 1, 3, 4, 2))
        q_rotated, k_rotated = self.rope(q_for_rope, k_for_rope)
        q_attn = q_rotated.view(B, self.num_heads, N, self.head_dim)
        k_attn = k_rotated.view(B, self.num_heads, N, self.head_dim)
        v_attn = v.contiguous().view(B, self.num_heads, N, self.head_dim)
        out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + out


class Mlp(nn.Module):
    """Multi-Layer Perceptron for spatial features."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU(),
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels * 4
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=bias)
        self.act = act_layer
        self.drop1 = nn.Dropout(drop)
        self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, 1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.pw_conv2(x)
        return x

@torch.compile(
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    }
)
class ResnetBlock(nn.Module):
    """
    A unified ResNet-like block that can optionally include a self-attention mechanism.
    """
    def __init__(
            self,
            channels: int,
            emb_dim: int,
            attention: bool = False,
            num_heads: int = 8,
            dropout: float = 0.0,
            padding_mode: str = "circular",
    ):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.use_attention = attention

        self.norm1 = RMSNorm2d(num_features=channels, eps=1e-6, elementwise_affine=False)

        if self.use_attention:
            self.attn_or_conv = MSAWithRoPE(
                channels=self.channels,
                num_heads=num_heads,
                padding_mode=padding_mode,
            )
        else:
            self.attn_or_conv = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3, padding=1,
                padding_mode=padding_mode,
                groups=channels
            )

        self.norm2 = RMSNorm2d(num_features=channels, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            channels,
            act_layer=nn.GELU(approximate="tanh"),
            drop=dropout,
        )

        modulation_dim = channels * 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, modulation_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        shift1, scale1, gate1, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=1)
        h_pre = self.norm1(x)
        h_pre = modulate(h_pre, shift1, scale1)
        h_processed = self.attn_or_conv(h_pre)
        h = x + gate1.unsqueeze(-1).unsqueeze(-1) * h_processed
        h_mlp = self.norm2(h)
        h_mlp = modulate(h_mlp, shift_mlp, scale_mlp)
        h_mlp = self.mlp(h_mlp)
        return h + gate_mlp.unsqueeze(-1).unsqueeze(-1) * h_mlp


class ConditionalChannelProjection(nn.Module):
    """Conditional channel projection layer."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            emb_dim: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = RMSNorm2d(num_features=in_channels, eps=1e-6, elementwise_affine=False)
        modulation_dim = in_channels * 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, modulation_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.SELU()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.act(self.conv(x))


class FinalLayer(nn.Module):
    """Final layer of the U-Net"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            emb_dim: int,
            padding_mode: str = "circular",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = RMSNorm2d(num_features=in_channels, eps=1e-6, elementwise_affine=False)
        modulation_dim = in_channels * 2
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * emb_dim, modulation_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        self.act = nn.SELU()
        self.conv_out = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=1, padding_mode=padding_mode
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.conv_out(self.act(x))


class ConditionalDownsample(nn.Module):
    """Conditional downsampling layer using depthwise convolution and pixel unshuffle."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        scale_factor: int = 2,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, padding_mode=padding_mode, groups=in_channels
        )
        self.channel_proj = ConditionalChannelProjection(
            in_channels, out_channels // (scale_factor ** 2), emb_dim
        )
        self.pixel_shuffle = nn.PixelUnshuffle(scale_factor)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        return self.pixel_shuffle(self.channel_proj(self.dw_conv(x), emb))


class ConditionalUpsample(nn.Module):
    """Conditional upsampling layer using pixel shuffle and channel projection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.channel_proj = ConditionalChannelProjection(
            in_channels, out_channels * (scale_factor ** 2), emb_dim
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        return self.act(self.pixel_shuffle(self.channel_proj(x, emb)))


def lecun_init(module: nn.Module) -> None:
    """
    Initializes the weights of convolutional layers using LeCun normal initialization.
    """
    if isinstance(module, (nn.Conv2d,)):
        fan_in = init._calculate_correct_fan(module.weight, mode="fan_in")
        var = 1.0 / math.sqrt(fan_in)
        init.normal_(module.weight, 0.0, var)     # LeCun normal
        if module.bias is not None:
            init.zeros_(module.bias)


@torch.compile(
    options={
        "shape_padding": True,
    }
)
class FlowUNet(nn.Module):
    """A U-Net model with a Diffusion Transformer (DiT) like architecture.

    This model implements a U-Net structure where the traditional ResNet blocks
    are replaced with blocks inspired by Diffusion Transformers. These blocks
    use adaptive layer normalization (adaLN) to condition on time and other
    embeddings, and can optionally replace convolutions with multi-head
    self-attention (`MSAWithRoPE`) at deeper levels of the network. It supports
    multiple conditional inputs, which are fused with the time embedding using
    another attention layer before being passed to the network blocks.

    Attributes:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        model_channels (int): The base number of channels in the model.
        num_levels (int): The number of downsampling/upsampling levels.
        start_attn_level (int): The level at which to start using attention blocks.
        num_blocks (int): Number of ResnetBlocks per level.
        emb_dim (int): The dimension of the conditional embeddings.
        num_conditions (int): The number of expected conditional inputs.
    """
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            model_channels: int = 32,
            downsample_out_ch_mult: Sequence[int] = (2, 2, 4,),
            start_attn_level: int = 0,
            num_blocks: int = 1,
            dropout: float = 0.1,
            num_heads: int = 8,
            num_conditions: int = 1,
            emb_dim: Optional[int] = None,
            padding_mode: str = "circular",
    ):
        """Initializes the FlowUNet model.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels in the output tensor.
            model_channels: The base number of channels for the first convolution.
            downsample_out_ch_mult: A sequence of multipliers for the number of
                channels at each downsampling level and the bottleneck.
            start_attn_level: The downsampling level (0-indexed) at which to
                start using self-attention blocks instead of convolutions.
            num_blocks: The number of `ResnetBlock`s per resolution level.
            dropout: The dropout rate used in the MLP of the `ResnetBlock`.
            num_heads: The number of heads for the self-attention blocks.
            num_conditions: The number of conditional inputs the model expects,
                excluding the time embedding.
            emb_dim: The dimension for the time and conditional embeddings. If
                None, defaults to `model_channels`.
            padding_mode: The padding mode for all convolutions.

        Raises:
            ValueError: If `downsample_out_ch_mult` has fewer than two elements,
                or if `start_attn_level` or `num_blocks` are not positive.
        """
        super().__init__()
        if len(downsample_out_ch_mult) < 2:
            raise ValueError("downsample_out_ch_mult must have at least two elements.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.downsample_out_ch_mult = downsample_out_ch_mult
        self.num_levels = len(downsample_out_ch_mult) - 1
        self.encoder_channel_mults = downsample_out_ch_mult[:-1]
        self.bottleneck_channel_mult = downsample_out_ch_mult[-1]
        if start_attn_level < 0:
            raise ValueError(f"start_attn_level must be positive, got {start_attn_level}.")
        self.start_attn_level = start_attn_level
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.padding_mode = padding_mode

        emb_dim = emb_dim or model_channels
        if emb_dim % 2 != 0: emb_dim += 1
        self.emb_dim = emb_dim
        self.time_embedder = SinusoidalPosEmb(self.emb_dim, factor=1000)
        self.time_mlp_embedders = MLPEmbedder(self.emb_dim)

        self.num_conditions = num_conditions
        if self.num_conditions > 0:
            self.cond_pos_embedders = nn.ModuleList([SinusoidalPosEmb(self.emb_dim) for _ in range(num_conditions)])
            self.cond_mlp_embedders = nn.ModuleList([MLPEmbedder(self.emb_dim) for _ in range(num_conditions)])
            self.mlp_msa = ConditionalMSAWithRoPE(self.emb_dim, num_heads, num_conditions + 1)

        resnet_block_args = {
            "emb_dim": self.emb_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "padding_mode": padding_mode,
        }
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1, padding_mode=padding_mode)

        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        skip_channels = []
        ch_schedule = [model_channels] + [model_channels * m for m in self.encoder_channel_mults]
        bottleneck_ch = model_channels * self.bottleneck_channel_mult

        for i in range(self.num_levels):
            current_level_channels = ch_schedule[i]

            if i < self.num_levels - 1:
                next_level_channels = ch_schedule[i + 1]
            else:
                next_level_channels = bottleneck_ch

            use_attention = i >= self.start_attn_level

            level_blocks = nn.ModuleList([
                ResnetBlock(channels=current_level_channels, attention=use_attention, **resnet_block_args)
                for _ in range(num_blocks)
            ])
            self.down_blocks.append(level_blocks)

            skip_channels.append(current_level_channels)
            self.down_samplers.append(ConditionalDownsample(
                in_channels=current_level_channels, out_channels=next_level_channels,
                emb_dim=self.emb_dim, padding_mode=padding_mode,
            ))

        use_middle_attention = self.num_levels >= self.start_attn_level
        self.middle_blocks = nn.ModuleList([
            ResnetBlock(channels=bottleneck_ch, attention=use_middle_attention, **resnet_block_args),
            ResnetBlock(channels=bottleneck_ch, attention=use_middle_attention, **resnet_block_args)
        ])
        ch = bottleneck_ch

        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.up_proj_convs = nn.ModuleList()
        self.skip_norm = nn.ModuleList()

        for i in reversed(range(self.num_levels)):
            target_ch = skip_channels.pop()
            self.up_samplers.append(
                ConditionalUpsample(
                    in_channels=ch, out_channels=target_ch, emb_dim=self.emb_dim,
                    scale_factor=2,
                )
            )
            self.skip_norm.append(
                nn.Sequential(
                    RMSNorm2d(num_features=target_ch, eps=1e-6),
                )
            )
            self.up_proj_convs.append(
                nn.Sequential(
                    RMSNorm2d(num_features=target_ch, eps=1e-6),
                    nn.Conv2d(
                        target_ch, target_ch,
                        1,
                    ),
                )
            )

            use_attention = i >= self.start_attn_level

            level_blocks = nn.ModuleList([
                ResnetBlock(channels=target_ch, attention=use_attention, **resnet_block_args)
                for _ in range(num_blocks)
            ])
            self.up_blocks.append(level_blocks)
            ch = target_ch

        self.final_layer = FinalLayer(
            in_channels=ch, out_channels=out_channels,
            emb_dim=self.emb_dim, padding_mode=padding_mode,
        )

        self.apply(lecun_init)

    def forward(self, x: Tensor, time: Tensor, conditions: Optional[Sequence[Tensor]] = None, **keywords) -> Tensor:
        B, C, H, W = x.shape
        min_size = 2 ** self.num_levels
        if H % min_size != 0 or W % min_size != 0:
            raise ValueError(f"Input H/W ({H}/{W}) must be divisible by the total downsampling factor {min_size}")

        if time.ndim != 1 or time.shape[0] != B:
            time = time.flatten().expand(B)
        t_emb = self.time_mlp_embedders(self.time_embedder(time.to(x.device)))

        if self.num_conditions > 0:
            if conditions is None or len(conditions) != self.num_conditions:
                raise ValueError(
                    f"Expected {self.num_conditions} conditions, but got {len(conditions) if conditions else 0}")
            cond_sequence = [t_emb]
            for cond_index, cond_tensor in enumerate(conditions):
                if cond_tensor.ndim != 1 or cond_tensor.shape[0] != B:
                    cond_tensor = cond_tensor.flatten().expand(B)
                c_emb = self.cond_pos_embedders[cond_index](cond_tensor.to(x.device))
                c_emb = self.cond_mlp_embedders[cond_index](c_emb)
                cond_sequence.append(c_emb)
            final_emb = self.mlp_msa(cond_sequence)
        else:
            final_emb = t_emb

        skips, h = [], self.conv_in(x)
        for i in range(self.num_levels):
            for block in self.down_blocks[i]:
                h = block(h, final_emb)
            skips.append(h)
            h = self.down_samplers[i](h, final_emb)

        for block in self.middle_blocks:
            h = block(h, final_emb)

        for i in range(self.num_levels):
            h = self.up_samplers[i](h, final_emb)
            skip_h = self.skip_norm[i](skips.pop())
            h = self.up_proj_convs[i](h * skip_h)
            for block in self.up_blocks[i]:
                h = block(h, final_emb)

        return self.final_layer(h, final_emb)


# ==============================================================================
# Test Suite
# ==============================================================================

class TestHelperFunctions(unittest.TestCase):
    def test_modulate_4d(self):
        x = torch.randn(2, 16, 8, 8)
        shift, scale = torch.randn(2, 16), torch.randn(2, 16)
        output = modulate(x, shift, scale)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.allclose(output, x))

class TestCoreModules(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.channels, self.height, self.width = 2, 32, 16, 16
        self.emb_dim, self.seq_len = 64, 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_rms_norm_2d(self):
        norm_affine = RMSNorm2d(num_features=self.channels).to(self.device)
        norm_no_affine = RMSNorm2d(num_features=self.channels, elementwise_affine=False).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output_affine = norm_affine(x)
        output_no_affine = norm_no_affine(x)
        self.assertEqual(output_affine.shape, x.shape)
        self.assertEqual(output_no_affine.shape, x.shape)
        self.assertIsNotNone(norm_affine.scale)
        self.assertIsNone(norm_no_affine.scale)

    def test_sinusoidal_pos_emb(self):
        emb = SinusoidalPosEmb(self.emb_dim - 1).to(self.device)
        self.assertEqual(emb.embedding_dim, self.emb_dim)
        x = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        output = emb(x)
        self.assertEqual(output.shape, (self.batch_size, self.emb_dim))
        with self.assertRaises(ValueError): SinusoidalPosEmb(0)

    def test_rotary_embedding_1d(self):
        rope = RoPE(dim=self.emb_dim).to(self.device)
        q = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        k = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        q_rot, k_rot = rope(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertFalse(torch.allclose(q_rot, q))
        with self.assertRaises(ValueError): RoPE(dim=31)

    def test_msa_with_rope(self):
        msa = MSAWithRoPE(channels=self.channels, num_heads=4).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = msa(x)
        self.assertEqual(output.shape, x.shape)
        with self.assertRaises(AssertionError): MSAWithRoPE(channels=30, num_heads=4)
        with self.assertRaises(AssertionError): MSAWithRoPE(channels=32, num_heads=5)

    def test_mlp(self):
        mlp = Mlp(in_channels=self.channels, out_channels=self.channels * 2).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.channels * 2, self.height, self.width))

class TestBuildingBlocks(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.channels, self.height, self.width = 2, 32, 16, 16
        self.emb_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_unified_resnet_block(self):
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        emb = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        block_with_attn = ResnetBlock(
            channels=self.channels, emb_dim=self.emb_dim, attention=True, num_heads=4
        ).to(self.device)
        output_attn = block_with_attn(x, emb)
        self.assertEqual(output_attn.shape, x.shape)
        self.assertTrue(block_with_attn.use_attention)
        self.assertIsInstance(block_with_attn.attn_or_conv, MSAWithRoPE)
        block_without_attn = ResnetBlock(
            channels=self.channels, emb_dim=self.emb_dim, attention=False
        ).to(self.device)
        output_no_attn = block_without_attn(x, emb)
        self.assertEqual(output_no_attn.shape, x.shape)
        self.assertFalse(block_without_attn.use_attention)
        self.assertIsInstance(block_without_attn.attn_or_conv, nn.Conv2d)

    def test_conditional_downsample(self):
        down = ConditionalDownsample(
            in_channels=self.channels, out_channels=self.channels * 2, emb_dim=self.emb_dim
        ).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = down(x, torch.randn(self.batch_size, self.emb_dim).to(self.device))
        self.assertEqual(output.shape, (self.batch_size, self.channels * 2, self.height // 2, self.width // 2))

    def test_conditional_upsample(self):
        up = ConditionalUpsample(
            in_channels=self.channels, out_channels=self.channels // 2, scale_factor=2, emb_dim=self.emb_dim
        ).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = up(x, torch.randn(self.batch_size, self.emb_dim).to(self.device))
        self.assertEqual(output.shape, (self.batch_size, self.channels // 2, self.height * 2, self.width * 2))

class TestFlowUNet(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.in_channels, self.out_channels = 2, 3, 3
        self.height, self.width = 32, 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_config = {
            "in_channels": self.in_channels, "out_channels": self.out_channels,
            "model_channels": 32, "downsample_out_ch_mult": (1, 2, 4, 8),
            "num_blocks": 2, "num_heads": 4,
        }

    def test_forward_pass_without_conditions(self):
        model = FlowUNet(**self.base_config, num_conditions=0).to(self.device)
        x = torch.randn(self.batch_size, self.in_channels, self.height, self.width).to(self.device)
        time = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        output = model(x, time)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))

    def test_forward_pass_with_conditions(self):
        num_cond = 2
        model = FlowUNet(**self.base_config, num_conditions=num_cond).to(self.device)
        x = torch.randn(self.batch_size, self.in_channels, self.height, self.width).to(self.device)
        time = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        conditions = [torch.rand(self.batch_size).to(self.device) for _ in range(num_cond)]
        output = model(x, time, conditions=conditions)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))
        with self.assertRaisesRegex(ValueError, "Expected 2 conditions, but got 1"):
            model(x, time, conditions=[conditions[0]])

    def test_architecture_based_on_start_attn_level(self):
        num_levels = len(self.base_config["downsample_out_ch_mult"]) - 1
        self.assertEqual(num_levels, 3)
        test_cases = [
            {"desc": "All levels use attention", "start_attn_level": 0},
            {"desc": "Attention starts at level 1", "start_attn_level": 1},
            {"desc": "Attention starts at level 2", "start_attn_level": 2},
            {"desc": "Only bottleneck uses attention", "start_attn_level": num_levels},
        ]
        for case in test_cases:
            with self.subTest(desc=case["desc"]):
                model = FlowUNet(**self.base_config, start_attn_level=case["start_attn_level"])
                for level_idx, level_blocks in enumerate(model.down_blocks):
                    should_use_attention = level_idx >= case["start_attn_level"]
                    for block in level_blocks:
                        self.assertIsInstance(block, ResnetBlock)
                        self.assertEqual(block.use_attention, should_use_attention)
                for block in model.middle_blocks:
                    self.assertIsInstance(block, ResnetBlock)
                    self.assertTrue(block.use_attention, "Middle blocks should always use attention")
                for i, level_blocks in enumerate(model.up_blocks):
                    level_idx = num_levels - 1 - i
                    should_use_attention = level_idx >= case["start_attn_level"]
                    for block in level_blocks:
                        self.assertIsInstance(block, ResnetBlock)
                        self.assertEqual(block.use_attention, should_use_attention)

    def test_input_size_error_handling(self):
        model = FlowUNet(**self.base_config).to(self.device)
        invalid_size = self.height - 1
        x = torch.randn(self.batch_size, self.in_channels, invalid_size, invalid_size).to(self.device)
        time = torch.rand(self.batch_size).to(self.device)
        with self.assertRaisesRegex(ValueError, "must be divisible by the total downsampling factor"):
            model(x, time)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)