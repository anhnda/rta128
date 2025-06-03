"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torch

def get_k(ar, offset=20, lag=10, alpha=1, beta=0.4, gamma=0.5):
    N = len(ar)
    best_score = -np.inf
    scores = []
    for k in range(offset, N-offset-lag):
        A1 = ar[:k]
        A2 = ar[k+lag:]

        # Average slope of A1 using linear fit (fast & stable)
        x1 = np.arange(len(A1))
        slope_A1 = np.polyfit(x1, A1, deg=1)[0]

        # Variance of A2
        var_A2 = np.var(A2)
        var_A1 = np.var(A1)
        # Objective score
        score = alpha * abs(slope_A1) + beta * var_A1 - gamma * var_A2
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k + lag



def get_k_tensor(ar, offset=20, lag=10, alpha=1, beta=0.4, gamma=0.5):
    """
    Vectorized version that processes batch of sequences.

    Args:
        ar: torch.Tensor of shape (batch_size, length)
        offset: int, offset from start/end
        lag: int, lag between segments
        alpha, beta, gamma: float, scoring coefficients

    Returns:
        torch.Tensor of shape (batch_size,) containing best k+lag for each sequence
    """
    batch_size, N = ar.shape
    device = ar.device

    # Valid k range
    k_start = offset
    k_end = N - offset - lag
    k_range = torch.arange(k_start, k_end, device=device)
    num_k = len(k_range)

    if num_k <= 0:
        raise ValueError("Invalid parameters: no valid k values")

    # Expand dimensions for vectorized computation
    # ar: (batch_size, length) -> (batch_size, 1, length)
    # k_range: (num_k,) -> (1, num_k, 1)
    ar_expanded = ar.unsqueeze(1)  # (batch_size, 1, length)
    k_expanded = k_range.unsqueeze(0).unsqueeze(2)  # (1, num_k, 1)

    # Create masks for A1 and A2 segments
    indices = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, length)

    # A1 mask: indices < k
    mask_A1 = indices < k_expanded  # (1, num_k, length)

    # A2 mask: indices >= k + lag
    mask_A2 = indices >= (k_expanded + lag)  # (1, num_k, length)

    # Broadcast masks to match ar dimensions
    mask_A1 = mask_A1.expand(batch_size, num_k, N)  # (batch_size, num_k, length)
    mask_A2 = mask_A2.expand(batch_size, num_k, N)  # (batch_size, num_k, length)

    # Extract A1 and A2 segments using masks
    ar_broadcast = ar_expanded.expand(batch_size, num_k, N)  # (batch_size, num_k, length)

    # Calculate slopes for A1 segments
    # For each A1 segment, we need to fit a line y = slope * x + intercept
    # Using the formula: slope = (n*Σ(xy) - Σ(x)Σ(y)) / (n*Σ(x²) - (Σ(x))²)

    # Get lengths of A1 segments
    A1_lengths = mask_A1.sum(dim=2, keepdim=True)  # (batch_size, num_k, 1)

    # Create x coordinates for each segment
    x_coords = torch.arange(N, device=device, dtype=torch.float32).expand(batch_size, num_k, N)

    # Apply masks and calculate sums
    A1_vals = torch.where(mask_A1, ar_broadcast, torch.tensor(0.0, device=device))
    A1_x = torch.where(mask_A1, x_coords, torch.tensor(0.0, device=device))

    # Adjust x coordinates to start from 0 for each segment
    k_start_vals = k_range.unsqueeze(0).expand(batch_size, num_k)  # (batch_size, num_k)
    A1_x_adjusted = A1_x - k_start_vals.unsqueeze(2) * mask_A1.float()
    A1_x_adjusted = torch.where(mask_A1, A1_x_adjusted, torch.tensor(0.0, device=device))

    # Calculate slope using vectorized linear regression
    n = A1_lengths.squeeze(2)  # (batch_size, num_k)
    sum_x = A1_x_adjusted.sum(dim=2)  # (batch_size, num_k)
    sum_y = A1_vals.sum(dim=2)  # (batch_size, num_k)
    sum_xy = (A1_x_adjusted * A1_vals).sum(dim=2)  # (batch_size, num_k)
    sum_x2 = (A1_x_adjusted ** 2).sum(dim=2)  # (batch_size, num_k)

    # slope = (n*Σ(xy) - Σ(x)Σ(y)) / (n*Σ(x²) - (Σ(x))²)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x ** 2

    # Avoid division by zero
    denominator = torch.where(denominator.abs() < 1e-10, torch.tensor(1e-10, device=device), denominator)
    slope_A1 = numerator / denominator  # (batch_size, num_k)

    # Calculate variances for A1 and A2
    A1_counts = mask_A1.sum(dim=2)  # (batch_size, num_k)
    A2_counts = mask_A2.sum(dim=2)  # (batch_size, num_k)

    # A1 variance
    A1_mean = A1_vals.sum(dim=2) / torch.clamp(A1_counts, min=1)  # (batch_size, num_k)
    A1_vals_centered = A1_vals - A1_mean.unsqueeze(2) * mask_A1.float()
    A1_var = (A1_vals_centered ** 2 * mask_A1.float()).sum(dim=2) / torch.clamp(A1_counts, min=1)

    # A2 variance
    A2_vals = torch.where(mask_A2, ar_broadcast, torch.tensor(0.0, device=device))
    A2_mean = A2_vals.sum(dim=2) / torch.clamp(A2_counts, min=1)  # (batch_size, num_k)
    A2_vals_centered = A2_vals - A2_mean.unsqueeze(2) * mask_A2.float()
    A2_var = (A2_vals_centered ** 2 * mask_A2.float()).sum(dim=2) / torch.clamp(A2_counts, min=1)

    # Calculate scores
    scores = alpha * slope_A1.abs() + beta * A1_var - gamma * A2_var  # (batch_size, num_k)

    # Find best k for each batch
    best_indices = scores.argmax(dim=1)  # (batch_size,)
    best_k = k_range[best_indices]  # (batch_size,)

    return best_k + lag
def get_k_tensor_constrained(ar, sub_seq, offset=20, lag=10, alpha=1, beta=0.4, gamma=0.5):
    """
    Vectorized version that processes batch of sequences with per-sequence constraints.

    Args:
        ar: torch.Tensor of shape (batch_size, length)
        sub_seq: torch.Tensor of shape (batch_size,) containing max value for best_k+lag for each sequence
        offset: int, offset from start/end
        lag: int, lag between segments
        alpha, beta, gamma: float, scoring coefficients

    Returns:
        torch.Tensor of shape (batch_size,) containing best k+lag for each sequence,
        constrained by sub_seq values
    """
    batch_size, N = ar.shape
    device = ar.device

    # Ensure sub_seq is on the same device
    if type(sub_seq) == list:
        sub_seq = torch.tensor(sub_seq, device=device)

    # Calculate per-sequence k ranges based on sub_seq constraints
    # For each sequence i: k + lag <= sub_seq[i], so k <= sub_seq[i] - lag
    k_start = offset
    k_end_global = N - offset - lag
    k_end_per_seq = sub_seq - lag  # (batch_size,)

    # Apply global constraint to per-sequence constraints
    k_end_per_seq = torch.clamp(k_end_per_seq, max=k_end_global)

    # Find the maximum possible k_end across all sequences to create a common range
    max_k_end = k_end_per_seq.max().item()

    if max_k_end <= k_start:
        raise ValueError("Invalid parameters: no valid k values for any sequence")

    # Create full k range
    k_range = torch.arange(k_start, max_k_end, device=device)
    num_k = len(k_range)

    if num_k <= 0:
        raise ValueError("Invalid parameters: no valid k values")

    # Create per-sequence masks for valid k values
    # k_range: (num_k,) -> (1, num_k)
    # k_end_per_seq: (batch_size,) -> (batch_size, 1)
    k_range_expanded = k_range.unsqueeze(0)  # (1, num_k)
    k_end_expanded = k_end_per_seq.unsqueeze(1)  # (batch_size, 1)

    # Valid k mask: k < k_end_per_seq for each sequence
    valid_k_mask = k_range_expanded < k_end_expanded  # (batch_size, num_k)

    # Expand dimensions for vectorized computation
    ar_expanded = ar.unsqueeze(1)  # (batch_size, 1, length)
    k_expanded = k_range.unsqueeze(0).unsqueeze(2)  # (1, num_k, 1)

    # Create masks for A1 and A2 segments
    indices = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, length)

    # A1 mask: indices < k
    mask_A1 = indices < k_expanded  # (1, num_k, length)

    # A2 mask: indices >= k + lag
    mask_A2 = indices >= (k_expanded + lag)  # (1, num_k, length)

    # Broadcast masks to match ar dimensions
    mask_A1 = mask_A1.expand(batch_size, num_k, N)  # (batch_size, num_k, length)
    mask_A2 = mask_A2.expand(batch_size, num_k, N)  # (batch_size, num_k, length)

    # Extract A1 and A2 segments using masks
    ar_broadcast = ar_expanded.expand(batch_size, num_k, N)  # (batch_size, num_k, length)

    # Calculate slopes for A1 segments
    # Get lengths of A1 segments
    A1_lengths = mask_A1.sum(dim=2, keepdim=True)  # (batch_size, num_k, 1)

    # Create x coordinates for each segment
    x_coords = torch.arange(N, device=device, dtype=torch.float32).expand(batch_size, num_k, N)

    # Apply masks and calculate sums
    A1_vals = torch.where(mask_A1, ar_broadcast, torch.tensor(0.0, device=device))
    A1_x = torch.where(mask_A1, x_coords, torch.tensor(0.0, device=device))

    # Adjust x coordinates to start from 0 for each segment
    k_start_vals = k_range.unsqueeze(0).expand(batch_size, num_k)  # (batch_size, num_k)
    A1_x_adjusted = A1_x - k_start_vals.unsqueeze(2) * mask_A1.float()
    A1_x_adjusted = torch.where(mask_A1, A1_x_adjusted, torch.tensor(0.0, device=device))

    # Calculate slope using vectorized linear regression
    n = A1_lengths.squeeze(2)  # (batch_size, num_k)
    sum_x = A1_x_adjusted.sum(dim=2)  # (batch_size, num_k)
    sum_y = A1_vals.sum(dim=2)  # (batch_size, num_k)
    sum_xy = (A1_x_adjusted * A1_vals).sum(dim=2)  # (batch_size, num_k)
    sum_x2 = (A1_x_adjusted ** 2).sum(dim=2)  # (batch_size, num_k)

    # slope = (n*Σ(xy) - Σ(x)Σ(y)) / (n*Σ(x²) - (Σ(x))²)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x ** 2

    # Avoid division by zero
    denominator = torch.where(denominator.abs() < 1e-10, torch.tensor(1e-10, device=device), denominator)
    slope_A1 = numerator / denominator  # (batch_size, num_k)

    # Calculate variances for A1 and A2
    A1_counts = mask_A1.sum(dim=2)  # (batch_size, num_k)
    A2_counts = mask_A2.sum(dim=2)  # (batch_size, num_k)

    # A1 variance
    A1_mean = A1_vals.sum(dim=2) / torch.clamp(A1_counts, min=1)  # (batch_size, num_k)
    A1_vals_centered = A1_vals - A1_mean.unsqueeze(2) * mask_A1.float()
    A1_var = (A1_vals_centered ** 2 * mask_A1.float()).sum(dim=2) / torch.clamp(A1_counts, min=1)

    # A2 variance
    A2_vals = torch.where(mask_A2, ar_broadcast, torch.tensor(0.0, device=device))
    A2_mean = A2_vals.sum(dim=2) / torch.clamp(A2_counts, min=1)  # (batch_size, num_k)
    A2_vals_centered = A2_vals - A2_mean.unsqueeze(2) * mask_A2.float()
    A2_var = (A2_vals_centered ** 2 * mask_A2.float()).sum(dim=2) / torch.clamp(A2_counts, min=1)

    # Calculate scores
    scores = alpha * slope_A1.abs() + beta * A1_var - gamma * A2_var  # (batch_size, num_k)

    # Apply valid k mask - set invalid k scores to very negative values
    scores = torch.where(valid_k_mask, scores, torch.tensor(-1e10, device=device))

    # Find best k for each batch (only among valid k values)
    best_indices = scores.argmax(dim=1)  # (batch_size,)
    best_k = k_range[best_indices]  # (batch_size,)

    # Ensure the result respects the sub_seq constraint
    result = best_k + lag
    result = torch.clamp(result, max=sub_seq)

    return result
def pad_to_M(x, M, value=0):
    """
    Pad a tensor of shape [B, N, C] to [B, M, C] along the N dimension.
    """
    _, N, _ = x.shape
    if M < N:
        raise ValueError(f"Target length M={M} must be >= current length N={N}")

    pad_len = M - N
    # Pad format: (last_dim_pad_left, last_dim_pad_right, ..., 2nd_dim_pad_left, 2nd_dim_pad_right, 1st_dim_pad_left, 1st_dim_pad_right)
    padded = F.pad(x, (0, 0, 0, pad_len), value=value)  # pad only along N dimension
    return padded

def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


import math 
def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


