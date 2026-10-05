"""
DraftDecoderLayer — the single LLaMA-style decoder layer that IS the EAGLE-3 draft.

It differs from a normal decoder layer in having two input streams (official
code, [Code]):

    e  the token embedding (of the NEXT token, see §V.6)
    x  the feature stream: g at real positions, a at drafted positions

Three details that are easy to miss:
    * each stream has its own RMSNorm before the two are concatenated to 2k;
    * the residual stream is x only — e enters through attention but is never
      added to the residual, so the output a stays a k-dim feature;
    * the MLP is a standard SwiGLU over k and never sees e directly.

Article: §VIII.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eagle3.model.attention import DraftAttention
from eagle3.model.layers import RMSNorm


class DraftDecoderLayer(nn.Module):
    """One LLaMA-style decoder layer over two input streams: e (token) and x (g or a).

    Args:
        k:          hidden size.
        n_heads:    attention query heads.
        n_kv_heads: attention key/value heads.
        inter:      SwiGLU intermediate width.
        rope_theta: RoPE base for the draft's attention.
        eps:        RMSNorm epsilon.
    """

    def __init__(self, k, n_heads, n_kv_heads, inter, rope_theta, eps):
        super().__init__()
        self.input_norm = RMSNorm(k, eps)        # normalises the token embedding e
        self.hidden_norm = RMSNorm(k, eps)       # normalises the feature stream x
        self.attn = DraftAttention(k, n_heads, n_kv_heads, rope_theta)
        self.post_norm = RMSNorm(k, eps)
        self.gate = nn.Linear(k, inter, bias=False)
        self.up = nn.Linear(k, inter, bias=False)
        self.down = nn.Linear(inter, k, bias=False)

    def _attn_in(self, e, x):
        """Concatenate the two normalised streams: [B, T, k] x 2 -> [B, T, 2k]."""
        return torch.cat([self.input_norm(e), self.hidden_norm(x)], -1)   # [B, T, 2k]

    def _finish(self, x, attn_out):
        """Residual + SwiGLU MLP. The residual stream is x, never e."""
        x = x + attn_out                         # residual stream is x, not e
        h = self.post_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def forward(self, e, x, pos, cache=None):
        """Inference path.

        e, x: [B, T, k]; pos: [T]; cache: draft K/V or None.
        Returns (a [B, T, k], updated cache).
        """
        a, cache = self.attn(self._attn_in(e, x), pos, cache)
        return self._finish(x, a), cache

    def forward_ttt(self, e, x, pos, ttt):
        """Training-time-test path (Figure 6 attention).

        e, x: [B, T, k]; pos: [T]; ttt: K/V of earlier TTT steps.
        Returns (a [B, T, k], updated ttt state).
        """
        a, ttt = self.attn.forward_ttt(self._attn_in(e, x), pos, ttt)
        return self._finish(x, a), ttt
