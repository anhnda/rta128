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



# Alternative version using xformers (if you want to keep using it)
class CustomMultiheadAttentionXFormers(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, pytorch_mha=None):
        super(CustomMultiheadAttentionXFormers, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections 
        if pytorch_mha is not None:
            self.qkv_linear_weight = pytorch_mha.in_proj_weight.reshape((3,embed_dim, embed_dim)).permute(0,2,1)
            self.qkv_linear_bias = pytorch_mha.in_proj_bias.reshape((3,embed_dim))
            self.out_linear = pytorch_mha.out_proj
        else:
            self.qkv_linear_weight = nn.Parameter(torch.zeros((3,embed_dim,embed_dim)))
            self.qkv_linear_bias = nn.Parameter(torch.zeros((3,embed_dim)))
            self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_bias=None):
        batch_size, seq_len, _ = query.size()
        qkv = torch.cat((query, key, value),dim=0)
        qkv_projected = torch.bmm(qkv, self.qkv_linear_weight) 
        qkv_projected += self.qkv_linear_bias.unsqueeze(1)
        

        
        # Reshape for multi-head attention - xformers expects [B, N, H, D]
        q = qkv_projected[0].view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = qkv_projected[1].view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = qkv_projected[2].view(batch_size, seq_len, self.num_heads, self.head_dim)

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
S1 = 200
S2 = 300
S3 = 150
batch = [
    torch.randn(S1, embed_dim, dtype=torch.float32, device='cuda'),
    torch.randn(S2, embed_dim, dtype=torch.float32, device='cuda'),
    torch.randn(S3, embed_dim, dtype=torch.float32, device='cuda')
]


# Initialize models
torch.manual_seed(42)  # Ensure same initialization
pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
pytorch_mha = pytorch_mha.to(device)
# custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
custom_mha_xf = CustomMultiheadAttentionXFormers(embed_dim, num_heads, dropout=dropout, pytorch_mha=pytorch_mha)
custom_mha_xf = custom_mha_xf.to(device)

# Copy weights
# copy_weights(pytorch_mha, custom_mha_xf)

# Set to evaluation mode
pytorch_mha.eval()
# custom_mha.eval()
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
