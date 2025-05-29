import torch
B, N, C = 3, 300, 256
# Input
x = torch.randn(B, N, C)
lengths = torch.tensor([3, 2, 4])

# Build a mask
range_tensor = torch.arange(N).unsqueeze(0).expand(B, N)  # (B, N)
mask = range_tensor < lengths.unsqueeze(1)                # (B, N)

# Masking
# Flatten x and mask
x_flat = x.reshape(-1, C)  # (B*N, C)
mask_flat = mask.reshape(-1)  # (B*N,)
result = x_flat[mask_flat]  # shape: (sum(lengths), C)
