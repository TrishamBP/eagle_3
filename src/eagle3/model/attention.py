"""
DraftAttention — the only context-mixing component of the EAGLE-3 draft model.

Paper §3.2: apart from self-attention, nothing in the draft's decoder layer
interacts with context, so self-attention is the only part training-time test
has to modify. This class therefore has two forward paths:

    forward      ordinary causal attention with a KV cache  (inference, §XI)
    forward_ttt  Figure 6 training-time-test attention      (training, §XII.8)

Article: §VIII.2 (code), §VII.5 (the mask it implements).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eagle3.model.layers import rope


class DraftAttention(nn.Module):
    """Self-attention whose Q/K/V projections read the 2k-wide [e ; x] input.

    The paper says "concat, then an FC layer down to k". The official code has
    no separate FC: q/k/v_proj take the 2k input directly. [Code]

    Args:
        k:          hidden size of the target (and of the draft's residual stream).
        n_heads:    number of query heads.
        n_kv_heads: number of key/value heads (grouped-query attention if < n_heads).
        rope_theta: RoPE base used for the draft's own positions.
    """

    def __init__(self, k, n_heads, n_kv_heads, rope_theta):
        super().__init__()
        self.h, self.kvh, self.d = n_heads, n_kv_heads, k // n_heads
        self.theta = rope_theta
        # 2k inputs: [norm(e) ; norm(x)] concatenated by DraftDecoderLayer._attn_in.
        self.q_proj = nn.Linear(2 * k, n_heads * self.d, bias=False)
        self.k_proj = nn.Linear(2 * k, n_kv_heads * self.d, bias=False)
        self.v_proj = nn.Linear(2 * k, n_kv_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, k, bias=False)

    def qkv(self, x2k, pos):
        """Project and rotate.

        x2k: [B, T, 2k], pos: [T]  ->  q, k, v each [B, n_heads, T, head_dim].
        K/V heads are repeated up to n_heads so later code can treat all heads alike.
        """
        B, T, _ = x2k.shape
        q = self.q_proj(x2k).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x2k).view(B, T, self.kvh, self.d).transpose(1, 2)
        v = self.v_proj(x2k).view(B, T, self.kvh, self.d).transpose(1, 2)
        q, k = rope(q, pos, self.theta), rope(k, pos, self.theta)
        rep = self.h // self.kvh                                    # grouped-query attention
        return q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

    def out(self, o):
        """Merge heads and project back: [B, H, T, d] -> [B, T, k]."""
        B, H, T, d = o.shape
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * d))

    def forward(self, x2k, pos, cache=None):
        """Inference: causal attention over cached + new positions.

        Args:
            x2k:   [B, T, 2k] inputs for the T new positions.
            pos:   [T] their absolute positions.
            cache: None or {"k": [B, H, S_old, d], "v": [B, H, S_old, d]}.

        Returns:
            (output [B, T, k], new cache covering S_old + T positions).
        """
        q, k, v = self.qkv(x2k, pos)
        if cache is not None:
            k = torch.cat([cache["k"], k], 2)
            v = torch.cat([cache["v"], v], 2)
        T, S = q.shape[2], k.shape[2]
        # New query i sits at absolute index S-T+i, so it may see keys 0..S-T+i.
        mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o), {"k": k, "v": v}

    def forward_ttt(self, x2k, pos, ttt):
        """Training-time test attention (paper §3.2, Figure 6).

        ttt["k"][0]: keys of the native step, one per position, causal.
        ttt["k"][s], s >= 1: keys of simulated step s, visible only to the
        query at the SAME position (the diagonal). Diagonal scores are vector
        dot products, not a matmul.

        Args:
            x2k: [B, T, 2k] inputs of the current step (all T training positions).
            pos: [T] positions for this step (pos + s at step s).
            ttt: {"k": [...], "v": [...]} keys/values of all previous steps;
                 empty lists at the native step.

        Returns:
            (output [B, T, k], ttt state with this step's K/V appended).
        """
        q, k, v = self.qkv(x2k, pos)
        ks, vs = ttt["k"] + [k], ttt["v"] + [v]
        T, scale = q.shape[2], 1.0 / math.sqrt(self.d)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        # Native-step block: a full T x T causal score matrix.
        s0 = (q @ ks[0].transpose(2, 3) * scale).masked_fill(~causal, float("-inf"))
        # Simulated-step blocks: only the diagonal survives the mask, so one
        # dot product per position replaces a mostly-masked T x T matmul.
        diag = [(q * ki).sum(-1, keepdim=True) * scale for ki in ks[1:]]   # [B,H,T,1] each
        # One softmax across [native block | diag_1 | diag_2 | ...].
        w = torch.softmax(torch.cat([s0] + diag, -1).float(), -1).to(q.dtype)
        o = w[..., :T] @ vs[0]
        for i, vi in enumerate(vs[1:]):
            o = o + w[..., T + i: T + i + 1] * vi                    # weight_i * v at same position
        return self.out(o), {"k": ks, "v": vs}
