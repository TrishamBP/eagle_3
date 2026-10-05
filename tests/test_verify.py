"""verify() must be lossless: emitted tokens follow the target distribution p."""

import torch

from eagle3.experiments.check_verify import run
from eagle3.sampling import probs, verify


def test_verify_distribution_matches_target():
    r = run(trials=20_000)                   # article: 200K trials, TV 0.0024
    assert r["tv_emitted_vs_target"] < 0.02


def test_greedy_verify_accepts_only_argmax_matches():
    V = 5
    p = probs(torch.tensor([[0.0, 3.0, 0.0, 0.0, 0.0],      # target argmax at pos 0: token 1
                            [0.0, 0.0, 0.0, 3.0, 0.0],      # pos 1: token 3
                            [4.0, 0.0, 0.0, 0.0, 0.0]]), 0)  # bonus: token 0
    draft_tokens = torch.tensor([1, 2])                      # first matches, second does not
    q = torch.nn.functional.one_hot(draft_tokens, V).float()
    n_acc, nxt = verify(draft_tokens, q, p)
    assert n_acc == 1 and nxt == 3                           # replacement = target argmax


def test_all_accepted_samples_bonus():
    p = probs(torch.tensor([[5.0, 0.0], [0.0, 5.0]]), 0)
    n_acc, nxt = verify(torch.tensor([0]), p[:1], p)
    assert n_acc == 1 and nxt == 1
