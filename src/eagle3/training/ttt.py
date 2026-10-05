"""
Training-time test (TTT) loss — paper §3.2, article §VII and §XII.8.

The draft is unrolled for `steps` steps over every training position in
parallel. Step 0 (native) sees the target's real fused features g; every
later step feeds the draft's OWN output a back in as the feature stream,
which is exactly the input mix it will see at inference time.

Official recipe [Code]: 7 steps, step s weighted 0.8**s, soft cross-entropy
against the target's distribution renormalised over the draft vocabulary,
masked to positions whose target top-1 token is inside the draft vocabulary.

Note what feeds back: the draft's FEATURE a. The token embeddings stay ground
truth (the training sequence, shifted one more position per step).
"""

import torch
import torch.nn.functional as F

from eagle3.features import tap_layers


def shift_left(t):
    """t[:, i] <- t[:, i+1], zero-padded at the end (official `padding(left=False)`).

    Works for [B, T] ids/masks and [B, T, V] logits alike.
    """
    return torch.cat([t[:, 1:], torch.zeros_like(t[:, :1])], 1)


def ttt_loss(target, draft, input_ids, loss_mask, steps=7, decay=0.8):
    """Training-time test loss for one batch.

    Official recipe: 7 steps, step s weighted 0.8**s, soft cross-entropy against
    the target's distribution restricted to the draft vocabulary.

    Args:
        target:    frozen target model.
        draft:     Eagle3Draft being trained.
        input_ids: [B, T] training sequences (target-generated, §XII.2).
        loss_mask: [B, T] 1 where a position should be supervised
                   (assistant-response tokens in the official pipeline).
        steps:     number of unrolled TTT steps (1 = no training-time test).
        decay:     per-step loss weight base.

    Returns:
        (total weighted loss [scalar tensor, differentiable], per-step losses list[float]).
    """
    # The target pass is frozen and gradient-free: it only supplies features and soft labels.
    with torch.no_grad():
        out = target(input_ids=input_ids, output_hidden_states=True)
        taps = tap_layers(target.config.num_hidden_layers)
        lmh = torch.cat([out.hidden_states[i] for i in taps], -1)
        tgt = out.logits.float()

    x = draft.fuse(lmh)                                      # native step input: g
    pos = torch.arange(input_ids.shape[1], device=input_ids.device)
    ids, tgt = shift_left(input_ids), shift_left(tgt)        # position i: e_{t_{i+1}}, predict t_{i+2}
    mask = loss_mask.float()                                 # (official: not shifted before step 0)
    ttt, total, per_step = {"k": [], "v": []}, 0.0, []

    for s in range(steps):
        x, ttt = draft.layer.forward_ttt(draft.embed(ids), x, pos + s, ttt)
        logp = F.log_softmax(draft.logits(x).float(), -1)                     # [B, T, Vd]
        p_t = F.softmax(tgt[..., draft.d2t], -1)                              # target on draft vocab
        m = mask * draft.in_draft[tgt.argmax(-1)].float()                     # skip out-of-vocab targets
        loss = -(m[..., None] * p_t * logp).sum(-1).mean()
        total = total + decay ** s * loss
        per_step.append(loss.item())
        ids, tgt, mask = shift_left(ids), shift_left(tgt), shift_left(mask)  # x stays: a feeds back
    return total, per_step
