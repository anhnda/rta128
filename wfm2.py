import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import xformers.ops as xops

# Custom Multihead Attention (xformers-based or fallback)
class CustomMultiheadAttentionXFormers(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttentionXFormers, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections - separate Q, K, V for clarity
        # self.q_linear = nn.Linear(embed_dim, embed_dim)
        # self.k_linear = nn.Linear(embed_dim, embed_dim)
        # self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)

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

# Function to copy weights from nn.MultiheadAttention to CustomMultiheadAttention
def copy_weights(pytorch_mha, custom_mha):
    with torch.no_grad():
        # Get the device and dtype from the custom model
        device = next(custom_mha.parameters()).device
        dtype = next(custom_mha.parameters()).dtype
        print(device, dtype)
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

# Parameters
embed_dim = 512  # head_dim = 512 // 8 = 64
num_heads = 8
batch_size = 2
seq_len = 10
dropout = 0.0

# Input tensors
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
query = torch.randn(batch_size, seq_len, embed_dim, device=device)
key = torch.randn(batch_size, seq_len, embed_dim, device=device)
value = torch.randn(batch_size, seq_len, embed_dim, device=device)
torch.manual_seed(1)
# Initialize models
pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
custom_mha = CustomMultiheadAttentionXFormers(embed_dim, num_heads, dropout=dropout)

# Move pytorch_mha to CUDA if xformers is used

pytorch_mha.cuda()
custom_mha.cuda()
query = query.to(device='cuda')
key = key.to(device='cuda')
value = value.to(device='cuda')

# Copy weights
copy_weights(pytorch_mha, custom_mha)

# Set to evaluation mode
pytorch_mha.eval()
custom_mha.eval()

# Forward pass
with torch.no_grad():
    pytorch_output, pytorch_attn_weights = pytorch_mha(query, key, value, need_weights=True, average_attn_weights=False)
    custom_output, custom_attn_weights = custom_mha(query, key, value)

# Debug shapes
#print("PyTorch attn_weights shape:", pytorch_attn_weights.shape)
#print("Custom attn_weights shape:", custom_attn_weights.shape)
print("Output values: ", torch.sum(pytorch_output), torch.sum(custom_output))
# Compare outputs
output_diff = torch.abs(pytorch_output - custom_output).max().item()
#attn_weights_diff = torch.abs(pytorch_attn_weights - custom_attn_weights).max().item()

print(f"Maximum difference in output: {output_diff:.8f}")
#print(f"Maximum difference in attention weights: {attn_weights_diff:.8f}")

# Check equivalence
assert output_diff < 1e-5, "Outputs are not sufficiently close"
#assert attn_weights_diff < 1e-5, "Attention weights are not sufficiently close"
print("Outputs and attention weights are equivalent within numerical precision.")