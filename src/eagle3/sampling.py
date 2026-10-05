"""
Probabilities, sampling and lossless speculative verification (article §X).

Verification is unchanged from vanilla speculative sampling (paper §2.1):
EAGLE-3 only changes how the draft is produced. Each draft token d_i is
accepted with probability min(1, p_i(d_i) / q_i(d_i)), front to back; the first
rejection resamples from norm(max(0, p_i - q_i)) and discards the rest.
"""

import torch
import torch.nn.functional as F


def probs(logits, temperature):
    """Next-token distribution. temperature == 0 means greedy (one-hot argmax).

    logits: [..., V] -> [..., V] float32 probabilities.

    Using a one-hot at T=0 lets the same `verify` rule handle greedy decoding:
    it then accepts a draft token iff it equals the target's argmax.
    """
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits.float() / temperature, -1)


def sample(p):
    """Draw one token per row from probabilities p: [..., V] -> [...]."""
    return torch.multinomial(p, 1).squeeze(-1)


def verify(draft_tokens, q, p):
    """Speculative sampling acceptance, front to back (paper §2.1).

    draft_tokens: [n]       draft tokens d_1..d_n
    q:            [n, V]    draft distributions each d_i was sampled from
    p:            [n+1, V]  target distributions at the same positions, plus one more
    Returns (number of accepted draft tokens, next token from the target's distribution).

    Notes:
        * q_i(d_i) > 0 always, because d_i was sampled from q_i.
        * After a rejection the residual is never all zeros: p_i(d_i) < q_i(d_i)
          and both sum to 1, so p_i must exceed q_i somewhere else.
        * If every draft is accepted, a bonus token is sampled from p[-1].
    """
    for i, t in enumerate(draft_tokens.tolist()):
        if torch.rand(()) < torch.clamp(p[i, t] / q[i, t], max=1.0):
            continue                                                 # accept d_i
        residual = torch.clamp(p[i] - q[i], min=0.0)                 # reject: resample
        return i, sample(residual / residual.sum()).item()
    return len(draft_tokens), sample(p[-1]).item()                   # all accepted: bonus
