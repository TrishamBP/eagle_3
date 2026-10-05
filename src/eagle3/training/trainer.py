"""
Draft training loop (article §XII.5-XII.8).

Paper §4: AdamW with betas (0.9, 0.95), learning rate 5e-5, gradient clipping 0.5.
Only the draft is trained; the target stays frozen.

The paper's 5e-5 is for ~500K real conversations over many epochs. On the toy
target with 800 steps that learning rate barely moves the draft, so the toy
experiments default to 1e-3 (my choice — the article does not state the toy LR).
"""

import torch

from eagle3.training.ttt import ttt_loss


def train_draft(target, draft, dataset, train_steps=800, batch_size=8, lr=1e-3,
                ttt_steps=7, decay=0.8, clip=0.5, seed=0, log_every=100):
    """Train `draft` in place with the training-time test loss.

    Args:
        target:      frozen target.
        draft:       Eagle3Draft to optimise.
        dataset:     ToyDataset (or anything with .batch(batch_size, generator)).
        train_steps: optimiser steps (article toy run: 800).
        batch_size:  sequences per step (my default: 8).
        lr:          AdamW learning rate (paper: 5e-5; toy default: 1e-3).
        ttt_steps:   unrolled TTT steps (official: 7; 1 = no training-time test).
        decay:       per-step loss weight base (official: 0.8).
        clip:        gradient-norm clip (paper: 0.5).
        seed:        mini-batch sampling seed.
        log_every:   print the loss every this many steps (0 = silent).

    Returns:
        History dict: {"loss": [...], "per_step": [[...], ...]} per logged step.
    """
    draft.train()
    params = [p for p in draft.parameters() if p.requires_grad]     # embedding is frozen
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))       # paper §4
    g = torch.Generator().manual_seed(seed)
    history = {"step": [], "loss": [], "per_step": []}

    for step in range(1, train_steps + 1):
        input_ids, loss_mask = dataset.batch(batch_size, generator=g)
        loss, per_step = ttt_loss(target, draft, input_ids, loss_mask,
                                  steps=ttt_steps, decay=decay)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)                 # paper §4
        opt.step()

        if log_every and (step % log_every == 0 or step == 1):
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["per_step"].append(per_step)
            steps_str = " ".join(f"{v:.3f}" for v in per_step)
            print(f"  step {step:>5}  loss {loss.item():.4f}  per-TTT-step [{steps_str}]")

    draft.eval()
    return history
