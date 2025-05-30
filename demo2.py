from xformers.ops import memory_efficient_attention

possible_ops = [
    "cutlass",
    "flash",
    "triton",
    "math",   # always available, pure PyTorch fallback
    None      # default selection
]

for op in possible_ops:
    try:
        print(f"Trying op = {op}")
        out = memory_efficient_attention(q, k, v, attn_bias=None, op=op)
        print(f"✅ {op} succeeded, output shape: {out.shape}")
    except Exception as e:
        print(f"❌ {op} failed: {e}")
