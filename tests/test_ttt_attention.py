"""forward_ttt (dot-product diagonal trick) must equal attention with Figure 6's explicit mask."""

import torch

from eagle3.experiments.check_ttt_attention import figure6_mask, run


def test_figure6_mask_structure():
    m = figure6_mask(T=3, step=2)
    # Native block: lower-triangular; each simulated block: identity.
    assert torch.equal(m[:, :3], torch.ones(3, 3, dtype=torch.bool).tril())
    assert torch.equal(m[:, 3:6], torch.eye(3, dtype=torch.bool))
    assert torch.equal(m[:, 6:9], torch.eye(3, dtype=torch.bool))


def test_forward_ttt_matches_explicit_mask():
    r = run(steps=7, seq_len=12)
    assert r["max_abs_diff"] < 1e-5          # article: 1.19e-7, float32 rounding
