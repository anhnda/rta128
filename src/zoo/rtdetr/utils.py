"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from xformers.ops import fmha
def convert_padded_maxlen(x, seq_len, device=None, value=0):
    if device is None:
        device = x.device
    if x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape[0] == 1:
        x = x.squeeze(0)

    # Determine max length to pad to
    max_len = max(seq_len)

    # Split into subsequences
    subseqs = torch.split(x, seq_len)  # List of [b1, C], [b2, C], ...

    # Pad each subsequence to max_len
    padded_subseqs = []
    for s in subseqs:
        pad_len = max_len - s.size(0)
        padded = F.pad(s, (0, 0, 0, pad_len), value=value)  # pad (left, right, top, bottom)
        padded_subseqs.append(padded)

    # Stack into [B, max_len, C]
    result = torch.stack(padded_subseqs)
    return result.to(device)
def convert_padded_M(x, seq_len, device=None, M=300, value=0):
    if device is None:
        device = x.device
    if x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape[0] == 1:
        x = x.squeeze(0)
    # Split into subsequences
    subseqs = torch.split(x, seq_len)  # List of [b1, C], [b2, C], ...

    # Pad each subsequence manually to length M
    padded_subseqs = []
    for s in subseqs:
        pad_len = M - s.size(0)
        padded = F.pad(s, (0, 0, 0, pad_len), value=value)  # Pad rows (top, bottom), not cols
        padded_subseqs.append(padded)

    # Stack into final tensor
    result = torch.stack(padded_subseqs)  # [B, M, C]
    return result.to(device)
def xformer_flattern_subseq(x,seq_lens, batch_size=-1, device=None, with_mask=False):
    B, N = x.shape[0], x.shape[1]
    if device is None:
        device = x.device
    seq_lens_org = seq_lens
    if B != 1:
        seq_lens = torch.tensor(seq_lens, dtype=int)
        x_flatten = torch.flatten(x,0,1)[(torch.arange(N).expand(B, N) < seq_lens.unsqueeze(1)).reshape(-1)]
        x_flatten = x_flatten.unsqueeze(0).to(device)

    else:
        x_flatten = x
        B = batch_size
    if not with_mask:
        block_diag = None
    else:
        block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seq_lens_org, device=device)
        batch_sizes = [1 for _ in range(B)]
        block_diag._batch_sizes = batch_sizes
        block_diag.to(device)
    return x_flatten, block_diag
def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights,seq_lens=None):
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

def deformable_attention_core_func_df(valuess, value_spatial_shapes, sampling_locationss, attention_weights, seq_lens=None):
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
    sampling_value_list = []
    assert seq_lens is not None
    sampling_locationss = sampling_locationss.split(seq_lens, dim=1)
    for i in range(len(seq_lens)):
        value = valuess[i].unsqueeze(0)
        bs, _, n_head, c = value.shape
        sampling_locations = sampling_locationss[i]
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        split_shape = [h * w for h, w in value_spatial_shapes]
        value_list = value.split(split_shape, dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_listi = []
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
            sampling_value_listi.append(sampling_value_l_)
        sampling_value_listi = torch.stack(sampling_value_listi, dim=-2).flatten(-2)
        sampling_value_list.append(sampling_value_listi)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    # attention_w_shape = attention_weights.shape
    bs, Len_q, n_head, n_levels, n_points = attention_weights.shape
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    sampling_value_list = torch.cat(sampling_value_list, dim=-2)
    output = (sampling_value_list *
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


