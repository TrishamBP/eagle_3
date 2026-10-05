"""
eagle3.model — the EAGLE-3 draft model, split by component.

    layers.py     RMSNorm and rotary position embedding (shared building blocks)
    attention.py  DraftAttention: Q/K/V read the 2k-wide [e ; x] input
    decoder.py    DraftDecoderLayer: one LLaMA-style layer over two input streams
    draft.py      Eagle3Draft: fusion FC + frozen embedding + layer + draft LM head
"""

from eagle3.model.attention import DraftAttention
from eagle3.model.decoder import DraftDecoderLayer
from eagle3.model.draft import Eagle3Draft, load_draft, save_draft
from eagle3.model.layers import RMSNorm, rope

__all__ = [
    "DraftAttention",
    "DraftDecoderLayer",
    "Eagle3Draft",
    "RMSNorm",
    "load_draft",
    "rope",
    "save_draft",
]
