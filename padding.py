import torch
import torch.nn.functional as F
from xformers.ops import fmha

# Sample batch data
batch = [
    (torch.randn(3, 512, dtype=torch.float32, device='cuda'), torch.randn(3, 512, dtype=torch.float32, device='cuda')),
    (torch.randn(5, 512, dtype=torch.float32, device='cuda'), torch.randn(5, 512, dtype=torch.float32, device='cuda')),
    (torch.randn(4, 512, dtype=torch.float32, device='cuda'), torch.randn(4, 512, dtype=torch.float32, device='cuda'))
]

def collate_xformer(batch):
    """
    Collate function using xformers BlockDiagonalMask for efficient attention
    without padding. Concatenates all sequences and uses block diagonal masking.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.concat([b[0] for b in batch]).unsqueeze(0).to(device, dtype=torch.float32)
    targets = torch.concat([b[1] for b in batch]).unsqueeze(0).to(device, dtype=torch.float32)
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

def collate_padding_pytorch(batch):
    """
    Collate function using traditional padding approach.
    Pads all sequences to the maximum length in the batch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract inputs and targets separately
    inputs_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    
    # Find max sequence length
    max_len = max(inp.shape[0] for inp in inputs_list)
    batch_size = len(batch)
    embed_dim = inputs_list[0].shape[1]
    
    # Initialize padded tensors
    padded_inputs = torch.zeros(batch_size, max_len, embed_dim, 
                               dtype=torch.float32, device=device)
    padded_targets = torch.zeros(batch_size, max_len, embed_dim, 
                                dtype=torch.float32, device=device)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    
    # Fill in the data and create mask
    seqlens = []
    for i, (inp, tgt) in enumerate(zip(inputs_list, targets_list)):
        seq_len = inp.shape[0]
        seqlens.append(seq_len)
        padded_inputs[i, :seq_len] = inp
        padded_targets[i, :seq_len] = tgt
        attention_mask[i, :seq_len] = True
    
    return {
        'inputs': padded_inputs,
        'targets': padded_targets,
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
if __name__ == "__main__":
    print("=== xformers collation ===")
    xformer_batch = collate_xformer(batch)
    print(f"Inputs shape: {xformer_batch['inputs'].shape}")
    print(f"Targets shape: {xformer_batch['targets'].shape}")
    print(f"Sequence lengths: {xformer_batch['attn_bias'].q_seqinfo.seqstart_py}")
    
    print("\n=== Padding collation ===")
    padding_batch = collate_padding_pytorch(batch)
    print(f"Inputs shape: {padding_batch['inputs'].shape}")
    print(f"Targets shape: {padding_batch['targets'].shape}")
    print(f"Attention mask shape: {padding_batch['attention_mask'].shape}")
    print(f"Sequence lengths: {padding_batch['seqlens']}")
    
    print("\n=== Memory comparison ===")
    xformer_memory = xformer_batch['inputs'].numel() * 2  # float32 = 2 bytes
    padding_memory = padding_batch['inputs'].numel() * 2
    print(f"xformers memory: {xformer_memory} bytes")
    print(f"Padding memory: {padding_memory} bytes")
    print(f"Memory overhead: {(padding_memory / xformer_memory - 1) * 100:.1f}%")