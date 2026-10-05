"""
Experiment: tensor-shape walkthrough on the toy target (article §IX.7).

Feeds a two-token prefix (standing in for "How can") through the target, fuses
the low/mid/high taps into g, and runs one draft catch-up step — printing every
intermediate shape.

Article result (8 layers, k = 64, untrained full-vocab draft):

    tap_layers(8)            -> (2, 4, 5)
    [l; m; h]   lmh          -> (1, 2, 192)      # [B, T, 3k]
    g = fc(lmh)              -> (1, 2, 64)       # [B, T, k]
    [e; g]      attention in -> (1, 2, 128)      # [B, T, 2k]
    a = layer(e, g)          -> (1, 2, 64)       # [B, T, k]
    draft K/V cache          -> (1, 4, 2, 16)    # [B, heads, T, head_dim]
    draft logits             -> (1, 2, 96)       # [B, T, V_d]

Run:  uv run eagle3-shapes
"""

import argparse

import torch
from transformers import DynamicCache

from eagle3.features import tap_layers, target_forward
from eagle3.model import Eagle3Draft
from eagle3.toy import build_toy_target
from eagle3.utils import header, row, set_seed


@torch.no_grad()
def run(target_seed=0, prefix=(11, 42)):
    """Print and return the shapes of every tensor in one prefill + draft step."""
    target = build_toy_target(seed=target_seed)
    set_seed(0)
    draft = Eagle3Draft.from_target(target)                  # full vocabulary, untrained

    prefix_ids = torch.tensor([list(prefix)])                # "How can"
    taps = tap_layers(target.config.num_hidden_layers)
    logits, lmh, _ = target_forward(target, prefix_ids, DynamicCache())
    pending = logits[0, -1].argmax().item()                  # the target's "I"
    g = draft.fuse(lmh)

    # Pair each g with the NEXT token's embedding: (g_how, e_can), (g_can, e_I).
    nxt = torch.tensor([list(prefix[1:]) + [pending]])
    e = draft.embed(nxt)
    attn_in = draft.layer._attn_in(e, g)
    a, d_cache = draft.layer(e, g, torch.arange(len(prefix)), None)
    d_logits = draft.logits(a)

    shapes = {
        "tap_layers": taps,
        "lmh [B, T, 3k]": tuple(lmh.shape),
        "g [B, T, k]": tuple(g.shape),
        "[e; g] attention in [B, T, 2k]": tuple(attn_in.shape),
        "a [B, T, k]": tuple(a.shape),
        "draft K/V cache [B, heads, T, head_dim]": tuple(d_cache["k"].shape),
        "draft logits [B, T, V_d]": tuple(d_logits.shape),
    }
    header("§IX.7  Tensor shapes on the toy target, 2-token prefix")
    for name, shp in shapes.items():
        row(name, shp)
    return {name: list(shp) for name, shp in shapes.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(target_seed=a.target_seed)


if __name__ == "__main__":
    main()
