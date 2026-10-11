import math


def block_flops(d, d_ff, top_k):
    # 4*d*d + top_k*3*d*d_ff
    return 4 * d * d + top_k * 3 * d * d_ff


def block_params(d, d_ff, n_experts):
    # 4*d*d + n_experts*3*d*d_ff
    return 4 * d * d + n_experts * 3 * d * d_ff


def smelt_match(d, n_blocks, n_experts, top_k, ff_mult, kv_heads, head_dim, loop_fraction=0.5):
    lam = 1.0 + loop_fraction

    # 1. Total block applications in the looped model
    applications = round(n_blocks * lam)

    # 2. Narrow the hidden width: d / sqrt(lambda), truncated to int, rounded down to multiple of 8
    d_raw = int(d / math.sqrt(lam))
    d_new = (d_raw // 8) * 8
    d_ff_new = int(ff_mult * d_new)

    # Base model parameters & FLOPs
    d_ff_base = int(ff_mult * d)
    base_params = n_blocks * block_params(d, d_ff_base, n_experts)
    base_flops = n_blocks * block_flops(d, d_ff_base, top_k)

    # 3. Find experts_new: smallest integer >= 1 such that total looped block params >= base_params
    experts_new = 1
    while True:
        looped_params = n_blocks * block_params(d_new, d_ff_new, experts_new)
        if looped_params >= base_params:
            break
        experts_new += 1

    # 4. Narrow KV heads: kv_heads / lambda, rounded to nearest integer, min 1
    kv_heads_new = max(1, round(kv_heads / lam))

    # 5. Compute metrics & ratios
    looped_flops = applications * block_flops(d_new, d_ff_new, top_k)

    # KV cache elements per token across all applications
    base_kv = n_blocks * (2 * kv_heads * head_dim)
    looped_kv = applications * (2 * kv_heads_new * head_dim)

    flops_ratio = round(looped_flops / base_flops, 4)
    params_ratio = round(looped_params / base_params, 4)
    kv_ratio = round(looped_kv / base_kv, 4)

    return {
        "d_new": d_new,
        "d_ff_new": d_ff_new,
        "experts_new": experts_new,
        "kv_heads_new": kv_heads_new,
        "applications": applications,
        "flops_ratio": flops_ratio,
        "params_ratio": params_ratio,
        "kv_ratio": kv_ratio,
    }