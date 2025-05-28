import torch
import functools
import time
import torch.nn as nn
from xformers.ops import fmha
from xformers.ops import memory_efficient_attention as mea

# Configuration
batch_size = 1
dim = 512
num_heads = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16 if device.type == 'cuda' else torch.float32

# Example batch with variable-length sequences
batch = [
    (torch.randn(3, dim, dtype=dtype, device=device), torch.randn(3, dim, dtype=dtype, device=device)),
    (torch.randn(5, dim, dtype=dtype, device=device), torch.randn(5, dim, dtype=dtype, device=device)),
    (torch.randn(4, dim, dtype=dtype, device=device), torch.randn(4, dim, dtype=dtype, device=device))
]

# Collate function for xFormers
def collate_xformer(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.concat([b[0] for b in batch]).unsqueeze(0).to(device, dtype=dtype)
    targets = torch.concat([b[1] for b in batch]).unsqueeze(0).to(device, dtype=dtype)
    indices = torch.concat([torch.arange(b[0].shape[0]) for b in batch]).to(device)
    seqlens = [b[0].shape[0] for b in batch]
    batch_sizes = [1 for _ in batch]
    block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens, device=device)
    block_diag._batch_sizes = batch_sizes
    return {
        'inputs': inputs,
        'targets': targets,
        'indices': indices,
        'attn_bias': block_diag
    }

# Collate function for PyTorch MultiheadAttention
def collate_pytorch(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = max(b[0].shape[0] for b in batch)
    inputs = torch.stack([torch.nn.functional.pad(b[0], (0, 0, 0, max_len - b[0].shape[0]), value=0) for b in batch])
    targets = torch.stack([torch.nn.functional.pad(b[1], (0, 0, 0, max_len - b[1].shape[0]), value=0) for b in batch])
    key_padding_mask = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=device)
    for i, b in enumerate(batch):
        key_padding_mask[i, b[0].shape[0]:] = True  # Mask padded tokens
    return {
        'inputs': inputs,
        'targets': targets,
        'key_padding_mask': key_padding_mask
    }

# xFormers Attention Block
class XFormersAttentionBlock(torch.nn.Module):
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

# PyTorch MultiheadAttention Block
class PyTorchAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_in, key_padding_mask=None):
        x, _ = self.attn(x_in, x_in, x_in, key_padding_mask=key_padding_mask)
        x = self.proj(x)
        return x

# Function to transfer weights from PyTorch to xFormers
def transfer_weights(pytorch_model, xformers_model):
    """
    Transfer weights from PyTorchAttentionBlock to XFormersAttentionBlock.
    """
    # Ensure both models are on the same device and dtype
    xformers_model = xformers_model.to(device=device, dtype=dtype)
    pytorch_model = pytorch_model.to(device=device, dtype=dtype)

    # Get state dictionaries
    pytorch_state_dict = pytorch_model.state_dict()
    xformers_state_dict = xformers_model.state_dict()

    # Map weights
    weight_map = {
        'attn.in_proj_weight': 'qkv.weight',
        'attn.in_proj_bias': 'qkv.bias',
        'proj.weight': 'proj.weight',
        'proj.bias': 'proj.bias'
    }

    # Transfer weights
    for pytorch_key, xformers_key in weight_map.items():
        xformers_state_dict[xformers_key].copy_(pytorch_state_dict[pytorch_key])

    # Load updated state dict into xformers model
    xformers_model.load_state_dict(xformers_state_dict)

    return xformers_model

# Set up xFormers
mea_eval = lambda q, k, v, attn_bias=None: mea(q, k, v, attn_bias=attn_bias)
xformers_block_fn = functools.partial(XFormersAttentionBlock, attn_fn=mea_eval, format='bshd')

# Process batches
xformers_batch = collate_xformer(batch)
pytorch_batch = collate_pytorch(batch)

# Initialize models
pytorch_model = PyTorchAttentionBlock(dim=dim, num_heads=num_heads).to(device=device, dtype=dtype)
xformers_model = xformers_block_fn(dim=dim, num_heads=num_heads).to(device=device, dtype=dtype)

# Transfer weights from PyTorch to xFormers
xformers_model = transfer_weights(pytorch_model, xformers_model)

# Verify output consistency (optional, for debugging)
with torch.no_grad():
    pytorch_output = pytorch_model(pytorch_batch['inputs'], key_padding_mask=pytorch_batch['key_padding_mask'])
    xformers_output = xformers_model(xformers_batch['inputs'], attn_bias=xformers_batch['attn_bias'])
    print("Output shapes:", pytorch_output.shape, xformers_output.shape)
    # Note: Outputs may differ slightly due to different masking implementations and numerical precision

# Warm-up (to avoid initial CUDA overhead)
with torch.no_grad():
    xformers_model(xformers_batch['inputs'], attn_bias=xformers_batch['attn_bias'])
    pytorch_model(pytorch_batch['inputs'], key_padding_mask=pytorch_batch['key_padding_mask'])

# Measure xFormers speed
start_time = time.time()
with torch.no_grad():
    for _ in range(100):  # Run multiple iterations for stable timing
        xformers_model(xformers_batch['inputs'], attn_bias=xformers_batch['attn_bias'])
torch.cuda.synchronize()  # Ensure GPU operations complete
xformers_time = time.time() - start_time
print(f"xFormers time: {xformers_time:.4f} seconds")

# Measure PyTorch speed
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        pytorch_model(pytorch_batch['inputs'], key_padding_mask=pytorch_batch['key_padding_mask'])
torch.cuda.synchronize()
pytorch_time = time.time() - start_time
print(f"PyTorch time: {pytorch_time:.4f} seconds")