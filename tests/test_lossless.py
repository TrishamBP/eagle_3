"""Greedy EAGLE-3 must reproduce plain greedy decoding token for token — for ANY draft."""

import torch

from eagle3.engine import autoregressive_generate, generate
from eagle3.experiments.common import make_prompts
from eagle3.training import make_self_generated_dataset, train_draft


def test_greedy_lossless_untrained_draft(target, draft):
    for prompt in make_prompts(target.config.vocab_size, n_prompts=3, prompt_len=4, seed=7):
        out, accepted = generate(target, draft, prompt, max_new_tokens=24, n_draft=4)
        ref = autoregressive_generate(target, prompt, max_new_tokens=24)
        assert out == ref
        assert all(0 <= a <= 4 for a in accepted)


def test_greedy_lossless_briefly_trained_draft(target, draft):
    # A few training steps change what the draft proposes, never what is emitted.
    data = make_self_generated_dataset(target, n_seqs=8, seq_len=32, seed=0)
    train_draft(target, draft, data, train_steps=10, batch_size=4, log_every=0)
    prompt = make_prompts(target.config.vocab_size, n_prompts=1, seed=11)[0]
    out, _ = generate(target, draft, prompt, max_new_tokens=24, n_draft=3)
    assert out == autoregressive_generate(target, prompt, max_new_tokens=24)


def test_sampling_mode_returns_requested_length(target, draft):
    torch.manual_seed(0)
    prompt = make_prompts(target.config.vocab_size, n_prompts=1, seed=3)[0]
    out, accepted = generate(target, draft, prompt, max_new_tokens=20, n_draft=4, temperature=1.0)
    assert len(out) == 20 and len(accepted) >= 1


def test_ttt_loss_is_finite_and_decreases(target, draft):
    data = make_self_generated_dataset(target, n_seqs=8, seq_len=32, seed=1)
    hist = train_draft(target, draft, data, train_steps=30, batch_size=8, log_every=1)
    assert all(torch.isfinite(torch.tensor(hist["loss"])))
    assert hist["loss"][-1] < hist["loss"][0]
