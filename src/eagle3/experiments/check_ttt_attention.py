"""
Experiment: the dot-product TTT attention is exact (article §VIII.2).

`DraftAttention.forward_ttt` never builds Figure 6's mask: it scores the native
step with a causal matmul and every simulated step with one dot product per
position (the diagonal). This check rebuilds the same computation the slow way —
all steps' keys concatenated, Figure 6's mask built explicitly, standard
scaled-dot-product attention — and compares outputs at every step.

Article result: max absolute difference 1.19e-7 at every step (float32 rounding).

Run:  uv run eagle3-check-ttt
"""

import argparse

import torch
import torch.nn.functional as F

from eagle3.model import DraftAttention
from eagle3.utils import header, row, set_seed


def figure6_mask(T, step, device=None):
    """Explicit Figure 6 mask for queries of `step` over keys of steps 0..step.

    Returns [T, (step+1)*T] bool: causal block for the native step, then an
    identity (diagonal) block for every simulated step.
    """
    mask = torch.zeros(T, (step + 1) * T, dtype=torch.bool, device=device)
    mask[:, :T] = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    for r in range(1, step + 1):
        mask[:, r * T:(r + 1) * T] = torch.eye(T, dtype=torch.bool, device=device)
    return mask


@torch.no_grad()
def reference_ttt(attn, inputs, positions):
    """Slow reference: full attention over every step's keys with the explicit mask."""
    ks, vs, outs = [], [], []
    for s, (x2k, pos) in enumerate(zip(inputs, positions)):
        q, k, v = attn.qkv(x2k, pos)
        ks.append(k)
        vs.append(v)
        mask = figure6_mask(q.shape[2], s, q.device)
        o = F.scaled_dot_product_attention(q, torch.cat(ks, 2), torch.cat(vs, 2), attn_mask=mask)
        outs.append(attn.out(o))
    return outs


@torch.no_grad()
def run(steps=7, batch=2, seq_len=16, k=64, n_heads=4, n_kv_heads=2, seed=0):
    """Compare forward_ttt with the explicit-mask reference. Returns per-step max |diff|."""
    set_seed(seed)
    attn = DraftAttention(k, n_heads, n_kv_heads, rope_theta=500000.0)
    inputs = [torch.randn(batch, seq_len, 2 * k) for _ in range(steps)]
    positions = [torch.arange(seq_len) + s for s in range(steps)]      # pos + s, as in ttt_loss

    ttt, fast = {"k": [], "v": []}, []
    for x2k, pos in zip(inputs, positions):
        o, ttt = attn.forward_ttt(x2k, pos, ttt)
        fast.append(o)
    slow = reference_ttt(attn, inputs, positions)

    diffs = [(a - b).abs().max().item() for a, b in zip(fast, slow)]
    header("§VIII.2  forward_ttt (dot products) vs explicit Figure 6 mask")
    for s, d in enumerate(diffs):
        row(f"step {s} ({'native' if s == 0 else 'simulated'}) max |diff|", f"{d:.3e}")
    row("max over all steps", f"{max(diffs):.3e}")
    return {"max_abs_diff_per_step": diffs, "max_abs_diff": max(diffs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=7, help="TTT steps to unroll")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(steps=a.steps, seq_len=a.seq_len, seed=a.seed)


if __name__ == "__main__":
    main()
