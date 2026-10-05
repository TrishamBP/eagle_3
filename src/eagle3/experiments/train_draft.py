"""
Experiment: train the toy EAGLE-3 draft with training-time test (article §XI.6, §XII.8).

Pipeline:
    1. build the toy target (8 layers, k=64, vocab 96, sharpened head)
    2. let it generate 64 training sequences itself (paper §4: target-regenerated data)
    3. keep the 64 most frequent response tokens as the draft vocabulary
    4. train 800 steps of ttt_loss (7 TTT steps, 0.8**s weights) with AdamW + clip 0.5
    5. save the draft to runs/draft_ttt{N}.pt

The checkpoint is what `eagle3-eval` and `eagle3-ablation` load.

Run:  uv run eagle3-train                    (7 TTT steps, article default)
      uv run eagle3-train --ttt-steps 1      (no training-time test, for the ablation)
"""

import argparse

from eagle3.experiments.common import DEFAULT_TTT_STEPS, DEFAULTS, train_and_save
from eagle3.toy import build_toy_target
from eagle3.utils import header, row


def run(ttt_steps=DEFAULT_TTT_STEPS, **opts):
    """Train and save a draft. Returns the training info dict (final loss, checkpoint path)."""
    o = {**DEFAULTS, **opts}
    header(f"§XI.6 / §XII.8  Train the toy draft — {ttt_steps} TTT step(s), "
           f"{o['train_steps']} optimiser steps")
    target = build_toy_target(seed=o["target_seed"])
    _, info = train_and_save(target, ttt_steps, **o)
    row("final weighted loss", f"{info['final_loss']:.4f}")
    row("checkpoint", info["checkpoint"])
    return {"ttt_steps": ttt_steps, "final_loss": info["final_loss"],
            "checkpoint": info["checkpoint"]}


def add_training_args(ap):
    """CLI flags shared by every script that may train a draft."""
    d = DEFAULTS
    ap.add_argument("--train-steps", type=int, default=d["train_steps"], help="optimiser steps (article: 800)")
    ap.add_argument("--batch-size", type=int, default=d["batch_size"], help="sequences per step (my default)")
    ap.add_argument("--lr", type=float, default=d["lr"], help="AdamW LR (paper: 5e-5 at scale; toy default 1e-3)")
    ap.add_argument("--n-seqs", type=int, default=d["n_seqs"], help="self-generated sequences (article: 64)")
    ap.add_argument("--seq-len", type=int, default=d["seq_len"], help="tokens per sequence (my default)")
    ap.add_argument("--draft-vocab", type=int, default=d["draft_vocab"], help="draft vocabulary size (article: 64)")
    ap.add_argument("--target-seed", type=int, default=d["target_seed"])
    ap.add_argument("--data-seed", type=int, default=d["data_seed"])
    ap.add_argument("--draft-seed", type=int, default=d["draft_seed"])
    ap.add_argument("--out-dir", default=d["out_dir"], help="where checkpoints/JSON go")


def training_opts(a):
    """Convert parsed CLI args (from add_training_args) into DEFAULTS-style options."""
    return dict(train_steps=a.train_steps, batch_size=a.batch_size, lr=a.lr, n_seqs=a.n_seqs,
                seq_len=a.seq_len, draft_vocab=a.draft_vocab, target_seed=a.target_seed,
                data_seed=a.data_seed, draft_seed=a.draft_seed, out_dir=a.out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttt-steps", type=int, default=DEFAULT_TTT_STEPS,
                    help="training-time-test steps (official: 7; 1 = disabled)")
    add_training_args(ap)
    a = ap.parse_args(argv)
    run(ttt_steps=a.ttt_steps, **training_opts(a))


if __name__ == "__main__":
    main()
