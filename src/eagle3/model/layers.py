"""
Building blocks shared by the draft model: RMSNorm and rotary position embedding.

Both follow the LLaMA conventions, because the EAGLE-3 draft is "one LLaMA-style
decoder layer" (paper §3.2) and the targets used in the paper are LLaMA-family.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (LLaMA style): no mean subtraction, no bias.

    Computed in float32 for stability, then cast back to the input dtype.

    Args:
        dim: size of the last dimension being normalised (the hidden size k).
        eps: added to the mean square before the reciprocal square root.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """x: [..., dim] -> [..., dim]."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope(x, pos, theta):
    """Rotary position embedding, "rotate-half" layout (as in HuggingFace LLaMA).

    Args:
        x:     [B, H, T, d] queries or keys, d = head dimension (must be even).
        pos:   [T] integer positions of the T tokens (need not start at 0 —
               cached decoding and training-time test both pass offsets).
        theta: RoPE base, e.g. 500000.0 for LLaMA-3.x.

    Returns:
        [B, H, T, d], x rotated by position-dependent angles.
    """
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[:, None] * inv[None, :]                       # [T, d/2]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin
