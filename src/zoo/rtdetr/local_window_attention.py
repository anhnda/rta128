import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LocalWindowMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_radius):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_radius = window_radius
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, H, W, C]
        Returns:
            Output tensor of shape [B, H, W, C]
        """
        B, H, W, C = x.shape
        k = self.window_radius
        window_size = (2 * k + 1) ** 2
        
        # Pad input to handle boundary conditions
        x_padded = F.pad(x, (0, 0, k, k, k, k), mode='reflect')
        
        # Use unfold to extract all local windows efficiently
        # unfold(dimension, size, step) extracts sliding windows
        windows = x_padded.unfold(1, 2*k+1, 1).unfold(2, 2*k+1, 1)
        # Shape: [B, H, W, C, 2k+1, 2k+1]
        
        # Reshape to [B, H, W, C, window_size] where window_size = (2k+1)^2
        windows = windows.reshape(B, H, W, C, window_size)
        
        # Transpose to [B, H, W, window_size, C] for easier processing
        windows = windows.transpose(-2, -1)  # [B, H, W, window_size, C]
        
        # Extract queries (center pixels)
        center_idx = window_size // 2  # Center index in flattened window
        queries = windows[:, :, :, center_idx:center_idx+1, :]  # [B, H, W, 1, C]
        
        # Keys and values are all pixels in each window
        keys = windows    # [B, H, W, window_size, C]
        values = windows  # [B, H, W, window_size, C]
        
        # Reshape for batch processing: treat H*W as batch dimension
        queries = queries.reshape(B * H * W, 1, C)
        keys = keys.reshape(B * H * W, window_size, C)
        values = values.reshape(B * H * W, window_size, C)
        
        # Apply multi-head attention
        output = self.multi_head_attention(queries, keys, values)  # [B*H*W, 1, C]
        
        # Reshape back to [B, H, W, C]
        output = output.reshape(B, H, W, C)
        
        return output
    
    def multi_head_attention(self, query, keys, values):
        """
        Args:
            query: [B*H*W, 1, C]
            keys: [B*H*W, window_size, C]
            values: [B*H*W, window_size, C]
        Returns:
            [B*H*W, 1, C]
        """
        batch_size, N, C = keys.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [B*H*W, 1, C]
        K = self.k_proj(keys)   # [B*H*W, N, C]
        V = self.v_proj(values) # [B*H*W, N, C]
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B*H*W, num_heads, 1, head_dim]
        K = K.reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*H*W, num_heads, N, head_dim]
        V = V.reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*H*W, num_heads, N, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B*H*W, num_heads, 1, N]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B*H*W, num_heads, 1, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, 1, C)  # [B*H*W, 1, C]
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


# Alternative implementation using im2col-style approach
class LocalWindowMultiHeadAttentionFast(nn.Module):
    def __init__(self, embed_dim, num_heads, window_radius):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_radius = window_radius
        self.head_dim = embed_dim // num_heads
        self.window_size = (2 * window_radius + 1) ** 2
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, H, W, C]
        Returns:
            Output tensor of shape [B, H, W, C]
        """
        B, H, W, C = x.shape
        k = self.window_radius
        
        # Pad input
        x_padded = F.pad(x, (0, 0, k, k, k, k), mode='reflect')
        
        # Convert to [B, C, H_pad, W_pad] for conv operations
        x_padded = x_padded.permute(0, 3, 1, 2)  # [B, C, H_pad, W_pad]
        
        # Use unfold to extract patches - this is equivalent to im2col
        patches = F.unfold(x_padded, kernel_size=2*k+1, stride=1)  # [B, C*window_size, H*W]
        patches = patches.transpose(1, 2)  # [B, H*W, C*window_size]
        patches = patches.reshape(B, H*W, self.window_size, C)  # [B, H*W, window_size, C]
        
        # Extract center pixels as queries
        center_idx = self.window_size // 2
        queries = patches[:, :, center_idx, :]  # [B, H*W, C]
        
        # All patches are keys and values
        keys = patches.reshape(B * H * W, self.window_size, C)
        values = keys
        queries = queries.reshape(B * H * W, C).unsqueeze(1)  # [B*H*W, 1, C]
        
        # Single QKV projection for efficiency
        qkv = self.qkv_proj(torch.cat([queries, keys, values], dim=1))  # [B*H*W, 1+2*window_size, 3*C]
        qkv = qkv.reshape(B * H * W, 1 + 2 * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*H*W, num_heads, 1+2*window_size, head_dim]
        
        q = qkv[0, :, :, :1, :]  # [B*H*W, num_heads, 1, head_dim]
        k = qkv[1, :, :, 1:1+self.window_size, :]  # [B*H*W, num_heads, window_size, head_dim]
        v = qkv[2, :, :, 1+self.window_size:, :]  # [B*H*W, num_heads, window_size, head_dim]
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B*H*W, num_heads, 1, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B * H * W, C)
        output = self.out_proj(attn_output)
        
        # Reshape back to [B, H, W, C]
        output = output.reshape(B, H, W, C)
        
        return output


# Example usage and test
if __name__ == "__main__":
    # Parameters
    B, H, W, C = 2, 32, 32, 128
    num_heads = 8
    window_radius = 3
    
    # Create models
    model1 = LocalWindowMultiHeadAttention(
        embed_dim=C,
        num_heads=num_heads,
        window_radius=window_radius
    )
    
    model2 = LocalWindowMultiHeadAttentionFast(
        embed_dim=C,
        num_heads=num_heads,
        window_radius=window_radius
    )
    
    # Create input
    x = torch.randn(B, H, W, C)
    print(f"Input shape: {x.shape}")
    
    # Test both implementations
    with torch.no_grad():
        output1 = model1(x)
        output2 = model2(x)
    
    print(f"Output1 shape: {output1.shape}")
    print(f"Output2 shape: {output2.shape}")
    
    # Verify shapes
    assert output1.shape == (B, H, W, C), f"Expected {(B, H, W, C)}, got {output1.shape}"
    assert output2.shape == (B, H, W, C), f"Expected {(B, H, W, C)}, got {output2.shape}"
    print("✓ Both implementations produce correct output shape")
    
    # Benchmark
    import time
    
    # Warmup
    for _ in range(5):
        _ = model1(x)
        _ = model2(x)
    
    # Time model1
    start = time.time()
    for _ in range(10):
        _ = model1(x)
    time1 = time.time() - start
    
    # Time model2
    start = time.time()
    for _ in range(10):
        _ = model2(x)
    time2 = time.time() - start
    
    print(f"Unfold approach: {time1:.4f}s")
    print(f"Im2col approach: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")