import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Custom Multihead Attention (xformer-like)
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(attn_output)
        attn_weights = torch.mean(attn_weights, dim=1)
        return output, attn_weights

# Function to copy weights
def copy_weights(pytorch_mha, custom_mha):
    with torch.no_grad():
        custom_mha.q_linear.weight.copy_(pytorch_mha.in_proj_weight[:embed_dim])
        custom_mha.k_linear.weight.copy_(pytorch_mha.in_proj_weight[embed_dim:2*embed_dim])
        custom_mha.v_linear.weight.copy_(pytorch_mha.in_proj_weight[2*embed_dim:])
        custom_mha.q_linear.bias.copy_(pytorch_mha.in_proj_bias[:embed_dim])
        custom_mha.k_linear.bias.copy_(pytorch_mha.in_proj_bias[embed_dim:2*embed_dim])
        custom_mha.v_linear.bias.copy_(pytorch_mha.in_proj_bias[2*embed_dim:])
        custom_mha.out_linear.weight.copy_(pytorch_mha.out_proj.weight)
        custom_mha.out_linear.bias.copy_(pytorch_mha.out_proj.bias)

# Parameters
embed_dim = 64
num_heads = 8
batch_size = 20
seq_len = 10
dropout = 0.0

# Input tensors
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Initialize models
pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)

# Copy weights
copy_weights(pytorch_mha, custom_mha)

# Set to evaluation mode
pytorch_mha.eval()
custom_mha.eval()

# Forward pass
with torch.no_grad():
    pytorch_output, pytorch_attn_weights = pytorch_mha(query, key, value)
    custom_output, custom_attn_weights = custom_mha(query, key, value)

# Debug shapes
print("PyTorch attn_weights shape:", pytorch_attn_weights.shape)
print("Custom attn_weights shape:", custom_attn_weights.shape)

# Compare outputs
output_diff = torch.abs(pytorch_output - custom_output).max().item()
print(output_diff)
print(pytorch_attn_weights.shape)
print(custom_attn_weights.shape)
attn_weights_diff = torch.abs(pytorch_attn_weights - custom_attn_weights).max().item()

print(f"Maximum difference in output: {output_diff:.8f}")
print(f"Maximum difference in attention weights: {attn_weights_diff:.8f}")

# Check equivalence
assert output_diff < 1e-6, "Outputs are not sufficiently close"
assert attn_weights_diff < 1e-6, "Attention weights are not sufficiently close"
print("Outputs and attention weights are equivalent within numerical precision.")