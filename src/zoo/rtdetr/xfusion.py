import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math
import torch.nn.init as init


class WindowMultiHeadAttention(nn.Module):
    """Multi-head attention with local windows (Swin-like, single level)"""

    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pair-wise relative position indices for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self._reset_params()
    def _reset_params(self):
        init.xavier_uniform_(self.qkv.weight)
        init.constant_(self.qkv.bias, 0)
        init.xavier_uniform_(self.proj.weight)
        init.constant_(self.proj.bias, 0)
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) - input tokens
            H, W: spatial dimensions
        Returns:
            (B, H*W, C) - output tokens
        """
        B, N, C = x.shape
        assert N == H * W, f"Input sequence length {N} doesn't match H*W = {H * W}"

        # Reshape to spatial format for window partitioning
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        H_pad, W_pad = H + pad_b, W + pad_r

        # Partition windows
        x_windows = self.window_partition(x, self.window_size)  # (num_windows*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (num_windows*B, window_size*window_size, C)

        # Multi-head attention in windows
        attn_windows = self.window_attention(x_windows)  # (num_windows*B, window_size*window_size, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # (B, H_pad, W_pad, C)

        # Remove padding if added
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # Reshape back to sequence format
        x = x.view(B, H * W, C)

        return x

    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
            window_size: window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size: Window size
            H: Height of image
            W: Width of image
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Window based multi-head self attention with relative position bias.
        Args:
            x: input features with shape of (num_windows*B, N, C)
        Returns:
            output features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape  # B_ = num_windows * B

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


# For backward compatibility, create an alias
MultiHeadAttention = WindowMultiHeadAttention


class MLPI(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPI, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.reset_parameters()

    def forward(self, x):
        return self.linear(x)

    def reset_parameters(self):
        init.constant_(self.linear.weight, 0)
        init.constant_(self.linear.bias, 0)


class MultiScaleFusionBlock(nn.Module):
    """Multi-scale fusion using up/down sampling and 1x1 convolutions"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 1x1 convolutions for feature fusion at each scale
        self.conv_1x1_scale1 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True)
        self.conv_1x1_scale2 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True)
        self.conv_1x1_scale3 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True)

        # Window-based multi-head attention for each scale
        # Adjust window sizes for each scale (smaller windows for smaller feature maps)
        window_size_1 = min(4, min(16, 16))  # For scale 1 (highest resolution)
        window_size_2 = min(4, min(8, 8))  # For scale 2 (middle resolution)
        window_size_3 = min(4, min(4, 4))  # For scale 3 (lowest resolution)

        self.mha_scale1 = WindowMultiHeadAttention(dim, window_size_1, num_heads, dropout=dropout)
        self.mha_scale2 = WindowMultiHeadAttention(dim, window_size_2, num_heads, dropout=dropout)
        self.mha_scale3 = WindowMultiHeadAttention(dim, window_size_3, num_heads, dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward networks
        self.ffn1 = MLPI(dim, dim)

        self.norm3 = nn.LayerNorm(dim)



    def _upsample_2x(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Upsample feature map by 2x using bilinear interpolation"""
        return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _downsample_2x(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample feature map by 2x using average pooling"""
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    def _reshape_to_spatial(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reshape from (B, H*W, C) to (B, C, H, W)"""
        B, _, C = x.shape
        return x.transpose(1, 2).reshape(B, C, h, w)

    def _reshape_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from (B, C, H, W) to (B, H*W, C)"""
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).transpose(1, 2)
    @staticmethod
    def split_multiscale_tensor(x: torch.Tensor, w1: int, h1: int, w2: int, h2: int, w3: int, h3: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split input tensor into three scales
        Args:
            x: (B, w1*h1 + w2*h2 + w3*h3, C)
        Returns:
            A1: (B, w1*h1, C)
            A2: (B, w2*h2, C)
            A3: (B, w3*h3, C)
        """
        n1, n2, n3 = w1 * h1, w2 * h2, w3 * h3

        A1 = x[:, :n1, :]
        A2 = x[:, n1:n1 + n2, :]
        A3 = x[:, n1 + n2:n1 + n2 + n3, :]

        return A1, A2, A3
    def forward(self, x: torch.Tensor, shapes) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        (w1, h1) , (w2 , h2) , (w3, h3) = shapes
        shortcut = x
        x = self.norm1(x)
        A1, A2, A3 = self.split_multiscale_tensor(x,w1,h1,w2,h2,w3,h3)
        
        # Convert to spatial format (B, C, H, W)
        A1_spatial = self._reshape_to_spatial(A1, h1, w1)  # (B, C, h1, w1)
        A2_spatial = self._reshape_to_spatial(A2, h2, w2)  # (B, C, h2, w2)
        A3_spatial = self._reshape_to_spatial(A3, h3, w3)  # (B, C, h3, w3)

        # === Scale 1 Fusion (highest resolution) ===
        # A2 -> A1: upsample by 2x
        A2_to_A1 = self._upsample_2x(A2_spatial, h1, w1)
        # A3 -> A1: upsample by 4x
        A3_to_A1 = self._upsample_2x(A3_spatial, h2, w2)  # First 2x
        A3_to_A1 = self._upsample_2x(A3_to_A1, h1, w1)  # Second 2x

        # Concatenate and fuse with 1x1 conv
        A1_concat = torch.cat([A1_spatial, A2_to_A1, A3_to_A1], dim=1)  # (B, 3C, h1, w1)
        A1_fused = self.conv_1x1_scale1(A1_concat)  # (B, C, h1, w1)

        # === Scale 2 Fusion (middle resolution) ===
        # A1 -> A2: downsample by 2x
        A1_to_A2 = self._downsample_2x(A1_spatial)
        # A3 -> A2: upsample by 2x
        A3_to_A2 = self._upsample_2x(A3_spatial, h2, w2)

        # Concatenate and fuse with 1x1 conv
        A2_concat = torch.cat([A1_to_A2, A2_spatial, A3_to_A2], dim=1)  # (B, 3C, h2, w2)
        A2_fused = self.conv_1x1_scale2(A2_concat)  # (B, C, h2, w2)

        # === Scale 3 Fusion (lowest resolution) ===
        # A1 -> A3: downsample by 4x
        A1_to_A3 = self._downsample_2x(A1_spatial)  # First 2x
        A1_to_A3 = self._downsample_2x(A1_to_A3)  # Second 2x
        # A2 -> A3: downsample by 2x
        A2_to_A3 = self._downsample_2x(A2_spatial)

        # Concatenate and fuse with 1x1 conv
        A3_concat = torch.cat([A1_to_A3, A2_to_A3, A3_spatial], dim=1)  # (B, 3C, h3, w3)
        A3_fused = self.conv_1x1_scale3(A3_concat)  # (B, C, h3, w3)

        # Convert back to sequence format
        A1_fused_seq = self._reshape_to_sequence(A1_fused)  # (B, h1*w1, C)
        A2_fused_seq = self._reshape_to_sequence(A2_fused)  # (B, h2*w2, C)
        A3_fused_seq = self._reshape_to_sequence(A3_fused)  # (B, h3*w3, C)

        # Add residual connections
        # A1_fused_seq = A1 + A1_fused_seq
        # A2_fused_seq = A2 + A2_fused_seq
        # A3_fused_seq = A3 + A3_fused_seq

        # Layer normalization
        # A1_fused_seq = self.norm1_1(A1_fused_seq)
        # A2_fused_seq = self.norm1_2(A2_fused_seq)
        # A3_fused_seq = self.norm1_3(A3_fused_seq)

        # === Multi-head attention with local windows for each scale ===
        A1_attn = self.mha_scale1(A1_fused_seq, h1, w1)
        A2_attn = self.mha_scale2(A2_fused_seq, h2, w2)
        A3_attn = self.mha_scale3(A3_fused_seq, h3, w3)

        # Residual connection
        # A1_attn = A1_fused_seq + A1_attn
        # A2_attn = A2_fused_seq + A2_attn
        # A3_attn = A3_fused_seq + A3_attn
        x = torch.cat([A1_attn, A2_attn, A3_attn], dim=1)
        x = shortcut + x
        x = x + self.ffn1(self.norm3(x))

        # Layer normalization
        # A1_attn = self.norm2_1(A1_attn)
        # A2_attn = self.norm2_2(A2_attn)
        # A3_attn = self.norm2_3(A3_attn)

        # === Feed-forward networks ===
        # A1_out = A1_attn + self.ffn1(A1_attn)
        # A2_out = A2_attn + self.ffn2(A2_attn)
        # A3_out = A3_attn + self.ffn3(A3_attn)

        # Final layer normalization
        # A1_out = self.norm3_1(A1_out)
        # A2_out = self.norm3_2(A2_out)
        # A3_out = self.norm3_3(A3_out)

        return x


class MultiScaleFusion(nn.Module):
    """Multi-layer multi-scale fusion network"""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MultiScaleFusionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor,shapes) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, shapes)
        return x





def concatenate_multiscale_tensors(A1: torch.Tensor, A2: torch.Tensor, A3: torch.Tensor) -> torch.Tensor:
    """
    Concatenate three scale tensors back into single tensor
    Args:
        A1: (B, w1*h1, C)
        A2: (B, w2*h2, C)
        A3: (B, w3*h3, C)
    Returns:
        x: (B, w1*h1 + w2*h2 + w3*h3, C)
    """
    return torch.cat([A1, A2, A3], dim=1)

