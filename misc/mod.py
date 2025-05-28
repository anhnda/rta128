import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import xformers.ops as xops

# Custom Multihead Attention (xformers-based or fallback)
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections - separate Q, K, V for clarity
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key) 
        v = self.v_linear(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use standard scaled dot-product attention instead of xformers for exact match
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_scores += attn_mask
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final output projection
        output = self.out_linear(attn_output)
        
        return output, attn_weights

# Function to copy weights from nn.MultiheadAttention to CustomMultiheadAttention
def copy_weights(pytorch_mha, custom_mha):
    with torch.no_grad():
        # Get the device and dtype from the custom model
        device = next(custom_mha.parameters()).device
        dtype = next(custom_mha.parameters()).dtype
        
        # PyTorch MHA uses in_proj_weight which concatenates Q, K, V weights
        # Shape: [3*embed_dim, embed_dim] where rows are [Q_weights; K_weights; V_weights]
        embed_dim = custom_mha.embed_dim
        
        # Split the concatenated weight matrix
        q_weight = pytorch_mha.in_proj_weight[:embed_dim].to(device=device, dtype=dtype)
        k_weight = pytorch_mha.in_proj_weight[embed_dim:2*embed_dim].to(device=device, dtype=dtype)
        v_weight = pytorch_mha.in_proj_weight[2*embed_dim:].to(device=device, dtype=dtype)
        
        # Split the concatenated bias vector
        q_bias = pytorch_mha.in_proj_bias[:embed_dim].to(device=device, dtype=dtype)
        k_bias = pytorch_mha.in_proj_bias[embed_dim:2*embed_dim].to(device=device, dtype=dtype)
        v_bias = pytorch_mha.in_proj_bias[2*embed_dim:].to(device=device, dtype=dtype)
        
        # Copy to separate linear layers
        custom_mha.q_linear.weight.copy_(q_weight)
        custom_mha.k_linear.weight.copy_(k_weight)
        custom_mha.v_linear.weight.copy_(v_weight)
        
        custom_mha.q_linear.bias.copy_(q_bias)
        custom_mha.k_linear.bias.copy_(k_bias)
        custom_mha.v_linear.bias.copy_(v_bias)
        
        # Copy output projection weights
        custom_mha.out_linear.weight.copy_(pytorch_mha.out_proj.weight.to(device=device, dtype=dtype))
        custom_mha.out_linear.bias.copy_(pytorch_mha.out_proj.bias.to(device=device, dtype=dtype))

# Alternative version using xformers (if you want to keep using it)
class CustomMultiheadAttentionXFormers(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttentionXFormers, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections - separate Q, K, V for clarity
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key) 
        v = self.v_linear(value)
        
        # Reshape for multi-head attention - xformers expects [B, N, H, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # xformers memory-efficient attention
        attn_output = xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            p=self.dropout if self.training else 0.0,
            op=None
        )
        
        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Final output projection
        output = self.out_linear(attn_output)
        
        return output, None

# Parameters
embed_dim = 512  # head_dim = 512 // 8 = 64
num_heads = 8
batch_size = 2
seq_len = 10
dropout = 0.0

# Input tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)  # For reproducibility
query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
key = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
value = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

# Initialize models
torch.manual_seed(42)  # Ensure same initialization
pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
custom_mha_xf = CustomMultiheadAttentionXFormers(embed_dim, num_heads, dropout=dropout)

# Move models to CUDA
pytorch_mha = pytorch_mha.to(device)
custom_mha = custom_mha.to(device)
custom_mha_xf = custom_mha_xf.to(device)

# Copy weights
copy_weights(pytorch_mha, custom_mha)
copy_weights(pytorch_mha, custom_mha_xf)

# Set to evaluation mode
pytorch_mha.eval()
custom_mha.eval()
custom_mha_xf.eval()

# Forward pass
with torch.no_grad():
    pytorch_output, pytorch_attn = pytorch_mha(query, key, value, need_weights=True, average_attn_weights=False)
    custom_output, custom_attn = custom_mha(query, key, value)
    custom_output_xf, _ = custom_mha_xf(query, key, value)

# Debug shapes and values
print("=== Standard Implementation ===")
print("PyTorch output sum:", torch.sum(pytorch_output).item())
print("Custom output sum:", torch.sum(custom_output).item())
print("PyTorch output shape:", pytorch_output.shape)
print("Custom output shape:", custom_output.shape)

# Compare outputs
output_diff = torch.abs(pytorch_output - custom_output).max().item()
mean_diff = torch.abs(pytorch_output - custom_output).mean().item()

print(f"Maximum difference in output: {output_diff:.8f}")
print(f"Mean difference in output: {mean_diff:.8f}")

# Compare attention weights if available
if custom_attn is not None:
    attn_diff = torch.abs(pytorch_attn - custom_attn).max().item()
    print(f"Maximum difference in attention weights: {attn_diff:.8f}")

print("\n=== XFormers Implementation ===")
print("XFormers output sum:", torch.sum(custom_output_xf).item())
output_diff_xf = torch.abs(pytorch_output - custom_output_xf).max().item()
mean_diff_xf = torch.abs(pytorch_output - custom_output_xf).mean().item()
print(f"Maximum difference in output (XFormers): {output_diff_xf:.8f}")
print(f"Mean difference in output (XFormers): {mean_diff_xf:.8f}")

# Check for equivalence
tolerance = 1e-5
print(f"\n=== Results (tolerance: {tolerance}) ===")
if output_diff < tolerance:
    print("✓ Standard implementation: Outputs are equivalent")
else:
    print(f"✗ Standard implementation: Outputs differ. Max diff: {output_diff:.6f}")
    
if output_diff_xf < tolerance:
    print("✓ XFormers implementation: Outputs are equivalent")
else:
    print(f"✗ XFormers implementation: Outputs differ. Max diff: {output_diff_xf:.6f}")

if output_diff > tolerance:
    print("\nDebugging info:")
    print("First few elements comparison:")
    print("PyTorch:", pytorch_output[0, 0, :5])
    print("Custom:  ", custom_output[0, 0, :5])
    print("Diff:    ", (pytorch_output - custom_output)[0, 0, :5])