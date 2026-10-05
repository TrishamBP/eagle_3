"""
Experiment: speculative verification preserves the target distribution (article §X.6).

Draws a random target distribution p and a random draft distribution q over 8
tokens. Each trial samples one draft token d ~ q and runs `verify`. The token
the procedure emits at that position is d if accepted, otherwise the
replacement sampled from norm(max(0, p - q)). Over many trials, the emitted
tokens must be distributed exactly as p.

Article result: 200,000 trials, total-variation distance 0.0024 (sampling noise).
Expected noise floor ~ sqrt(V / trials) / 2, i.e. roughly 0.002-0.003 here.

Run:  uv run eagle3-check-verify        (takes ~1 minute on CPU: a Python loop)
"""

import argparse

import torch

from eagle3.metrics import tv_distance
from eagle3.sampling import sample, verify
from eagle3.utils import header, row, set_seed


@torch.no_grad()
def run(trials=200_000, vocab=8, sharpness=2.0, seed=0):
    """Return the TV distance between emitted-token frequencies and p."""
    set_seed(seed)
    p = torch.softmax(torch.randn(vocab) * sharpness, -1)            # target
    q = torch.softmax(torch.randn(vocab) * sharpness, -1)            # draft
    p_rows = torch.stack([p, p])                                     # [n+1, V] with n = 1 draft

    counts = torch.zeros(vocab)
    n_accepted = 0
    for _ in range(trials):
        d = sample(q).item()
        n_acc, nxt = verify(torch.tensor([d]), q[None], p_rows)
        emitted = d if n_acc == 1 else nxt                           # token at the draft position
        counts[emitted] += 1
        n_accepted += n_acc

    empirical = counts / trials
    tv = tv_distance(empirical, p)
    header(f"§X.6  verify() output distribution vs target p ({trials:,} trials, V={vocab})")
    row("TV(p, q)  (how different the draft is)", f"{tv_distance(p, q):.4f}")
    row("acceptance rate", f"{n_accepted / trials:.4f}")
    row("expected acceptance  sum min(p, q)", f"{torch.minimum(p, q).sum().item():.4f}")
    row("TV(emitted, p)", f"{tv:.4f}")
    return {"trials": trials, "tv_emitted_vs_target": tv,
            "tv_draft_vs_target": tv_distance(p, q),
            "acceptance_rate": n_accepted / trials}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(trials=a.trials, vocab=a.vocab, seed=a.seed)


if __name__ == "__main__":
    main()
