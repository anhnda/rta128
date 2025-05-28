import torch
import functools
from xformers.ops import fmha
from xformers.ops import memory_efficient_attention as mea

# Example batch with variable-length sequences
batch = [
    (torch.randn(3, 512, dtype=torch.float16, device='cuda'), torch.randn(3, 512, dtype=torch.float16, device='cuda')),
    (torch.randn(5, 512, dtype=torch.float16, device='cuda'), torch.randn(5, 512, dtype=torch.float16, device='cuda')),
    (torch.randn(4, 512, dtype=torch.float16, device='cuda'), torch.randn(4, 512, dtype=torch.float16, device='cuda'))
]

# Collate function to prepare inputs and mask
def collate_xformer(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.concat([b[0] for b in batch]).unsqueeze(0).to(device, dtype=torch.float16)
    targets = torch.concat([b[1] for b in batch]).unsqueeze(0).to(device, dtype=torch.float16)
    indices = torch.concat([torch.arange(b[0].shape[0]) for b in batch]).to(device)
    seqlens = [b[0].shape[0] for b in batch]  # e.g., [3, 5, 4]
    batch_sizes = [1 for _ in batch]
    block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens, device=device)
    block_diag._batch_sizes = batch_sizes
    return {
        'inputs': inputs,
        'targets': targets,
        'indices': indices,
        'attn_bias': block_diag
    }

# Define attention block
class MyAttentionBlock(torch.nn.Module):
    def __init__(self, attn_fn, dim, num_heads, format='bshd'):
        super().__init__()
        self.attn_fn = attn_fn
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)
        self.permute = torch.nn.Identity() if format == 'bshd' else functools.partial(torch.transpose, dim0=1, dim1=2)

    def reshape_and_permute(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return self.permute(x)

    def forward(self, x_in, attn_bias=None):
        batch_size = x_in.size(0)
        qkv = self.qkv(x_in)
        q, k, v = qkv.chunk(3, -1)
        q = self.reshape_and_permute(q, batch_size)
        k = self.reshape_and_permute(k, batch_size)
        v = self.reshape_and_permute(v, batch_size)
        x = self.attn_fn(q, k, v, attn_bias=attn_bias)
        x = self.permute(x).reshape(batch_size, -1, self.dim)
        x = self.proj(x)
        return x

# Set up attention function
mea_eval = lambda q, k, v, attn_bias=None: mea(q, k, v, attn_bias=attn_bias)
block_fn = functools.partial(MyAttentionBlock, attn_fn=mea_eval, format='bshd')

# Process batch
batch_data = collate_xformer(batch)
inputs = batch_data['inputs']
attn_bias = batch_data['attn_bias']

# Initialize model
model = block_fn(dim=512, num_heads=8).to(device='cuda', dtype=torch.float16)

# Forward pass
output = model(inputs, attn_bias=attn_bias)
print(output.shape)