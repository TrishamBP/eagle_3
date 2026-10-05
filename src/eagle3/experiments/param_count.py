"""
Experiment: draft model size at LLaMA-3.1-8B scale (article §VIII.7).

Instantiates Eagle3Draft on PyTorch's `meta` device — shapes only, no memory —
with k = 4096, 32 heads, 8 KV heads, MLP width 14,336, a 128,256-token target
vocabulary and a 32,000-token draft vocabulary, then counts parameters.

Article result: fc 50.3M, attention 67.1M, MLP 176.2M, lm_head 131.1M,
trainable total 424.7M, frozen embedding 525.3M.

Run:  uv run eagle3-param-count
"""

import argparse

import torch

from eagle3.model import Eagle3Draft
from eagle3.utils import header, row


def _count(module):
    return sum(p.numel() for p in module.parameters())


def run(k=4096, n_heads=32, n_kv_heads=8, inter=14336, vocab=128256, draft_vocab=32000):
    """Count draft parameters per component. Returns a dict of counts (in parameters)."""
    with torch.device("meta"):
        draft = Eagle3Draft(k, n_heads, n_kv_heads, inter, torch.empty(vocab, k),
                            draft_vocab_ids=torch.arange(draft_vocab))
    layer = draft.layer
    counts = {
        "fc (fusion 3k->k)": _count(draft.fc),
        "attention (Q/K/V from 2k, O)": _count(layer.attn),
        "mlp (SwiGLU)": _count(layer.gate) + _count(layer.up) + _count(layer.down),
        "lm_head (draft vocab)": _count(draft.lm_head),
        "norms": _count(layer.input_norm) + _count(layer.hidden_norm)
                 + _count(layer.post_norm) + _count(draft.norm),
        "trainable total": sum(p.numel() for p in draft.parameters() if p.requires_grad),
        "embedding (frozen copy)": _count(draft.embed),
    }

    header(f"§VIII.7  Draft parameters at k={k}, heads={n_heads}/{n_kv_heads}, "
           f"MLP={inter}, V={vocab}, V_d={draft_vocab}")
    for name, n in counts.items():
        row(name, f"{n / 1e6:8.1f}M")
    row("trainable share of an 8B target", f"{counts['trainable total'] / 8e9:8.1%}")
    return {name: n for name, n in counts.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4096, help="hidden size")
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--inter", type=int, default=14336, help="MLP intermediate width")
    ap.add_argument("--vocab", type=int, default=128256, help="target vocabulary")
    ap.add_argument("--draft-vocab", type=int, default=32000, help="draft vocabulary")
    a = ap.parse_args(argv)
    run(a.k, a.n_heads, a.n_kv_heads, a.inter, a.vocab, a.draft_vocab)


if __name__ == "__main__":
    main()
