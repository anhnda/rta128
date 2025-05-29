import torch

import xformers.ops as xops
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
def xformer_flattern_subseq(x,seq_lens, device):
    B, N = x.shape[0], x.shape[1]
    block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seq_lens, device=device)
    batch_sizes = [1 for _ in range(B)]
    block_diag._batch_sizes = batch_sizes

    seq_lens = torch.tensor(seq_lens, dtype=int)
    x_flatten = torch.flatten(x,0,1)[(torch.arange(N).expand(B, N) < seq_lens.unsqueeze(1)).reshape(-1)].to(device)
    return x_flatten, block_diag
def demo():
    B, N, H, C = 3, 300, 8, 32
    x = torch.randn((B,N,H,C))
    seq = [200,250,150]
    x_flatten, attn_bias = xformer_flattern_subseq(x,seq, device='cuda')
    print(x_flatten.shape)
    # print(result)

if __name__ == "__main__":
    demo()