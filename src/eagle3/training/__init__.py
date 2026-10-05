"""
eagle3.training — everything needed to train an EAGLE-3 draft.

    ttt.py      training-time test loss (7 unrolled steps, 0.8**s weights)  §XII.8
    data.py     target-generated training sequences + draft vocabulary     §XII.2, §VIII.6
    trainer.py  AdamW(0.9, 0.95) + grad-clip 0.5 optimisation loop         §XII.5-XII.7
"""

from eagle3.training.data import ToyDataset, make_self_generated_dataset, top_draft_vocab
from eagle3.training.trainer import train_draft
from eagle3.training.ttt import shift_left, ttt_loss

__all__ = [
    "ToyDataset",
    "make_self_generated_dataset",
    "shift_left",
    "top_draft_vocab",
    "train_draft",
    "ttt_loss",
]
