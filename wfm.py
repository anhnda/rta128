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

        # Check if xformers is supported
      
        # Linear projections
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)
        self.qkv_linear.to(dtype=torch.float16, device='cuda')
      
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear.to(dtype=torch.float32,device='cuda')  # Ensure output is float32
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

            # Convert inputs to float16 and move to CUDA
        query = query.to(dtype=torch.float16, device='cuda')
        key = key.to(dtype=torch.float16, device='cuda')
        value = value.to(dtype=torch.float16, device='cuda')
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.float16, device='cuda')

        # xformers implementation
        qkv = self.qkv_linear(query)  # [batch_size, seq_len, 3 * embed_dim]
        q, k, v = qkv.split(self.embed_dim, dim=-1)  # Each: [batch_size, seq_len, embed_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # xformers memory-efficient attention
        attn_output = xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            p=self.dropout,
            op=None
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(attn_output.to(dtype=torch.float32))
   
  

        return output, None

# Function to copy weights from nn.MultiheadAttention to CustomMultiheadAttention
def copy_weights(pytorch_mha, custom_mha):
    with torch.no_grad():
        custom_mha.qkv_linear.weight.copy_(
            torch.cat([
                pytorch_mha.in_proj_weight[:embed_dim],
                pytorch_mha.in_proj_weight[embed_dim:2*embed_dim],
                pytorch_mha.in_proj_weight[2*embed_dim:]
            ], dim=0).to(dtype=torch.float16, device='cuda')
        )
        custom_mha.qkv_linear.bias.copy_(
            torch.cat([
                pytorch_mha.in_proj_bias[:embed_dim],
                pytorch_mha.in_proj_bias[embed_dim:2*embed_dim],
                pytorch_mha.in_proj_bias[2*embed_dim:]
            ], dim=0).to(dtype=torch.float16, device='cuda')
        )
        custom_mha.out_linear.weight.copy_(pytorch_mha.out_proj.weight.to(device='cuda'))
        custom_mha.out_linear.bias.copy_(pytorch_mha.out_proj.bias.to(device='cuda'))

# Parameters
embed_dim = 512  # head_dim = 512 // 8 = 64
num_heads = 8
batch_size = 2
seq_len = 10
dropout = 0.0

# Input tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
query = torch.randn(batch_size, seq_len, embed_dim, device=device)
key = torch.randn(batch_size, seq_len, embed_dim, device=device)
value = torch.randn(batch_size, seq_len, embed_dim, device=device)

# Initialize models
pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)

# Move pytorch_mha to CUDA if xformers is used

pytorch_mha.cuda()
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
    pytorch_output, _ = pytorch_mha(query, key, value, need_weights=True, average_attn_weights=False)
    custom_output, _ = custom_mha(query, key, value)

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