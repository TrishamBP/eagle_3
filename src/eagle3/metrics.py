"""
Metrics used by the experiments (article §XIII).

    accepted_per_cycle   mean number of draft tokens accepted per cycle
    tokens_per_pass      mean tokens committed per target forward pass (= accepted + 1)
    n_alpha              acceptance rate with n self-estimated inputs (chain draft)
    tv_distance          total-variation distance between two distributions
"""

import torch


def accepted_per_cycle(accepted):
    """Mean accepted draft tokens per drafting-verification cycle."""
    return sum(accepted) / max(len(accepted), 1)


def tokens_per_pass(accepted):
    """Mean tokens produced per target forward pass.

    Each cycle commits the pending target token plus the accepted drafts, so
    this is accepted_per_cycle + 1. (The paper's tau counts per cycle too.)
    """
    return accepted_per_cycle(accepted) + 1.0


def n_alpha(accepted, n_draft):
    """Acceptance rate n-alpha from per-cycle accepted counts (chain draft).

    n-alpha = P(draft token n+1 accepted | drafts 1..n accepted), i.e. the
    draft's input already contains n of its own outputs a. From per-cycle
    counts:  alpha_n = #(cycles with n_acc > n) / #(cycles with n_acc >= n).

    Returns a list of length n_draft (NaN where no cycle reached position n).
    """
    out = []
    for n in range(n_draft):
        reached = sum(a >= n for a in accepted)
        passed = sum(a > n for a in accepted)
        out.append(passed / reached if reached else float("nan"))
    return out


def tv_distance(p, q):
    """Total-variation distance 0.5 * sum |p - q| between two [V] distributions."""
    return 0.5 * (torch.as_tensor(p).float() - torch.as_tensor(q).float()).abs().sum().item()
