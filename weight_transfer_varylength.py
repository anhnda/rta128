import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import xformers.ops as xops


import torch
import torch.nn.functional as F
from xformers.ops import fmha


def collate_xformer(batch):
    """
    Collate function using xformers BlockDiagonalMask for efficient attention
    without padding. Concatenates all sequences and uses block diagonal masking.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.concat([b for b in batch]).unsqueeze(0).to(device, dtype=torch.float32)
    indices = torch.concat([torch.arange(b.shape[0]) for b in batch]).to(device)
    seqlens = [b.shape[0] for b in batch]  # e.g., [3, 5, 4]
    batch_sizes = [1 for _ in batch]
    block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens, device=device)
    block_diag._batch_sizes = batch_sizes
    return {
        'inputs': inputs,
        'indices': indices,
        'attn_bias': block_diag
    }

def collate_padding_pytorch(batch):
    """
    Collate function using traditional padding approach.
    Pads all sequences to the maximum length in the batch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract inputs and targets separately
    inputs_list = [b for b in batch]
    
    # Find max sequence length
    max_len = max(inp.shape[0] for inp in inputs_list)
    batch_size = len(batch)
    embed_dim = inputs_list[0].shape[1]
    
    # Initialize padded tensors
    padded_inputs = torch.zeros(batch_size, max_len, embed_dim, 
                               dtype=torch.float32, device=device)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    
    # Fill in the data and create mask
    seqlens = []
    for i, inp in enumerate(inputs_list):
        seq_len = inp.shape[0]
        seqlens.append(seq_len)
        padded_inputs[i, :seq_len] = inp
        attention_mask[i, :seq_len] = True
    
    return {
        'inputs': padded_inputs,
        'attention_mask': attention_mask,
        'seqlens': seqlens
    }

def collate_padding_pytorch_causal(batch):
    """
    Collate function with causal (lower triangular) attention mask for autoregressive models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inputs_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    
    max_len = max(inp.shape[0] for inp in inputs_list)
    batch_size = len(batch)
    embed_dim = inputs_list[0].shape[1]
    
    padded_inputs = torch.zeros(batch_size, max_len, embed_dim, 
                               dtype=torch.float32, device=device)
    padded_targets = torch.zeros(batch_size, max_len, embed_dim, 
                                dtype=torch.float32, device=device)
    
    # Create causal mask: combines padding mask with causal (lower triangular) mask
    # Shape: (batch_size, max_len, max_len)
    causal_mask = torch.tril(torch.ones(max_len, max_len, device=device)).bool()
    attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply padding mask
    seqlens = []
    for i, (inp, tgt) in enumerate(zip(inputs_list, targets_list)):
        seq_len = inp.shape[0]
        seqlens.append(seq_len)
        padded_inputs[i, :seq_len] = inp
        padded_targets[i, :seq_len] = tgt
        # Zero out attention for padded positions
        attention_mask[i, seq_len:, :] = False  # Can't attend to padding
        attention_mask[i, :, seq_len:] = False  # Padding can't attend
    
    return {
        'inputs': padded_inputs,
        'targets': padded_targets,
        'attention_mask': attention_mask,
        'seqlens': seqlens
    }

# Example usage and comparison
# if __name__ == "__main__":
#     print("=== xformers collation ===")
#     xformer_batch = collate_xformer(batch)
#     print(f"Inputs shape: {xformer_batch['inputs'].shape}")
#     print(f"Targets shape: {xformer_batch['targets'].shape}")
#     print(f"Sequence lengths: {xformer_batch['attn_bias'].q_seqinfo.seqstart_py}")
    
#     print("\n=== Padding collation ===")
#     padding_batch = collate_padding_pytorch(batch)
#     print(f"Inputs shape: {padding_batch['inputs'].shape}")
#     print(f"Targets shape: {padding_batch['targets'].shape}")
#     print(f"Attention mask shape: {padding_batch['attention_mask'].shape}")
#     print(f"Sequence lengths: {padding_batch['seqlens']}")
    
#     print("\n=== Memory comparison ===")
#     xformer_memory = xformer_batch['inputs'].numel() * 2  # float32 = 2 bytes
#     padding_memory = padding_batch['inputs'].numel() * 2
#     print(f"xformers memory: {xformer_memory} bytes")
#     print(f"Padding memory: {padding_memory} bytes")
#     print(f"Memory overhead: {(padding_memory / xformer_memory - 1) * 100:.1f}%")
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

    def forward(self, query, key, value, attn_bias=None):
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
            attn_bias=attn_bias,
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
# Sample batch data

# Input tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)  # For reproducibility
S1 = 300
S2 = 500
S3 = 200
batch = [
    torch.randn(S1, embed_dim, dtype=torch.float32, device='cuda'),
    torch.randn(S2, embed_dim, dtype=torch.float32, device='cuda'),
    torch.randn(S3, embed_dim, dtype=torch.float32, device='cuda')
]


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

xformer_inp = collate_xformer(batch)
pytorch_inp = collate_padding_pytorch(batch)
query_x = key_x = value_x = xformer_inp['inputs'].to(device)
query_q = key_q = value_q = pytorch_inp['inputs'].to(device)
attention_x = xformer_inp['attn_bias']
attention_q = ~pytorch_inp['attention_mask']
# Forward pass
import time
start_time = time.time()
with torch.no_grad():

    for _ in range(100): 
        pytorch_output, pytorch_attn = pytorch_mha(query_q, key_q, value_q, key_padding_mask = attention_q, need_weights=True, average_attn_weights=False)
torch.cuda.synchronize()  # Ensure GPU operations complete

pytorch_time = time.time() - start_time
print(f"Pytorch time: {pytorch_time:.4f} seconds")

start_time = time.time()
with torch.no_grad():
    for _ in range(100): 
        custom_output_xf, _ = custom_mha_xf(query_x, key_x, value_x, attn_bias=attention_x)
torch.cuda.synchronize()  # Ensure GPU operations complete
xformer_time = time.time() - start_time
print(f"xFormers time: {xformer_time:.4f} seconds")

pytorch_output[0,S1:,:] = 0
pytorch_output[2,S3:,:] = 0
# Debug shapes and values
print("=== Standard Implementation ===")
print("PyTorch output sum:", torch.sum(pytorch_output).item())
print("PyTorch output shape:", pytorch_output.shape)




print("\n=== XFormers Implementation ===")
print("XFormers output sum:", torch.sum(custom_output_xf).item())
print("XFormers output shape:", custom_output_xf.shape)

# output_diff_xf = torch.abs(pytorch_output - custom_output_xf).max().item()
# mean_diff_xf = torch.abs(pytorch_output - custom_output_xf).mean().item()
# print(f"Maximum difference in output (XFormers): {output_diff_xf:.8f}")
# print(f"Mean difference in output (XFormers): {mean_diff_xf:.8f}")

# # Check for equivalence
# tolerance = 1e-5
# if output_diff_xf < tolerance:
#     print("✓ XFormers implementation: Outputs are equivalent")
# else:
#     print(f"✗ XFormers implementation: Outputs differ. Max diff: {output_diff_xf:.6f}")

