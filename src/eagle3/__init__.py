"""
eagle3 — a minimal, readable EAGLE-3 reconstruction in PyTorch.

Companion code for the article "EAGLE-3: Engineering Implementation of LLM
Inference Acceleration" (paper: arXiv:2503.01840). Not the official
implementation (github.com/SafeAILab/EAGLE). Where the paper is silent, the
layout follows the official code at commit cb7e084.

Module map (concern -> file -> article section):

    model/layers.py      RMSNorm, RoPE building blocks          Appendix C
    model/attention.py   draft self-attention (+ TTT attention)  §VIII.2, §VII.5
    model/decoder.py     two-stream draft decoder layer          §VIII.1
    model/draft.py       the Eagle3Draft module                  §VIII.4-VIII.6
    features.py          low/mid/high target feature taps        §IX
    sampling.py          probabilities + lossless verification  §X
    engine.py            EAGLE-3 chain-draft decoding loop       §XI
    training/ttt.py      training-time test loss                 §XII.8
    training/data.py     self-generated data + draft vocabulary  §XII.2, §VIII.6
    training/trainer.py  AdamW + grad-clip training loop         §XII.5-XII.7
    toy.py               the small test target used in §XI.6
    metrics.py           acceptance length, n-alpha, TV distance §XIII
    experiments/         one runnable script per reported result
"""

from eagle3.engine import autoregressive_generate, generate
from eagle3.features import tap_layers, target_forward
from eagle3.model import DraftAttention, DraftDecoderLayer, Eagle3Draft, RMSNorm, rope
from eagle3.sampling import probs, sample, verify
from eagle3.training.ttt import shift_left, ttt_loss

__all__ = [
    "DraftAttention",
    "DraftDecoderLayer",
    "Eagle3Draft",
    "RMSNorm",
    "autoregressive_generate",
    "generate",
    "probs",
    "rope",
    "sample",
    "shift_left",
    "tap_layers",
    "target_forward",
    "ttt_loss",
    "verify",
]

__version__ = "0.1.0"
