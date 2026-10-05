"""
Shared plumbing for the toy experiments: defaults, prompts, train-or-load, evaluation.

Keeping this in one place guarantees that `train_draft`, `eval_lossless` and
`ablation_ttt_steps` all use the same target, data, draft vocabulary and seeds.
"""

from pathlib import Path

import torch

from eagle3.engine import autoregressive_generate, generate
from eagle3.metrics import accepted_per_cycle, n_alpha, tokens_per_pass
from eagle3.model import Eagle3Draft, load_draft, save_draft
from eagle3.training import make_self_generated_dataset, top_draft_vocab, train_draft
from eagle3.utils import ensure_dir, set_seed

# Values marked "article" come from §XI.6 / §XII.8; the rest are my defaults.
DEFAULTS = dict(
    target_seed=0,       # toy target weight init
    data_seed=0,         # self-generated data
    n_seqs=64,           # article: 64 target-generated sequences
    seq_len=64,          # my default
    prompt_len=4,        # my default
    data_temperature=1.0,  # my default
    draft_vocab=64,      # article: 64-token draft vocabulary
    train_steps=800,     # article: 800 steps
    batch_size=8,        # my default
    lr=1e-3,             # my default (paper's 5e-5 is for real-scale training)
    draft_seed=0,        # draft weight init + mini-batch order
    out_dir="runs",
)

# Training-time-test steps (article / official code: 7). Kept out of DEFAULTS
# because it is always passed explicitly and also names the checkpoint file.
DEFAULT_TTT_STEPS = 7


def checkpoint_path(out_dir, ttt_steps):
    """Where the draft trained with `ttt_steps` TTT steps is stored."""
    return Path(out_dir) / f"draft_ttt{ttt_steps}.pt"


def build_draft(target, vocab_ids, seed):
    """Fresh (untrained) draft for `target` with a fixed init seed."""
    set_seed(seed)
    return Eagle3Draft.from_target(target, draft_vocab_ids=vocab_ids)


def train_and_save(target, ttt_steps, **opts):
    """Generate data, pick the draft vocab, train a draft, save it. Returns (draft, info)."""
    o = {**DEFAULTS, **opts}
    data = make_self_generated_dataset(target, n_seqs=o["n_seqs"], seq_len=o["seq_len"],
                                       prompt_len=o["prompt_len"],
                                       temperature=o["data_temperature"], seed=o["data_seed"])
    vocab_ids = top_draft_vocab(data.input_ids, data.loss_mask,
                                target.config.vocab_size, o["draft_vocab"])
    draft = build_draft(target, vocab_ids, o["draft_seed"])
    history = train_draft(target, draft, data, train_steps=o["train_steps"],
                          batch_size=o["batch_size"], lr=o["lr"], ttt_steps=ttt_steps,
                          seed=o["draft_seed"])
    info = {"ttt_steps": ttt_steps, "options": o,
            "final_loss": history["loss"][-1] if history["loss"] else None}
    path = checkpoint_path(ensure_dir(o["out_dir"]), ttt_steps)
    save_draft(draft, path, extra=info)
    info["checkpoint"] = str(path)
    return draft, info


def load_or_train(target, ttt_steps, retrain=False, **opts):
    """Load runs/draft_ttt{N}.pt if it exists (and retrain is False), else train it."""
    o = {**DEFAULTS, **opts}
    path = checkpoint_path(o["out_dir"], ttt_steps)
    if path.exists() and not retrain:
        print(f"  loading trained draft from {path}")
        draft, _ = load_draft(target, path)
        return draft.eval()
    print(f"  no checkpoint at {path} — training a draft with {ttt_steps} TTT step(s)")
    draft, _ = train_and_save(target, ttt_steps, **o)
    return draft


def make_prompts(vocab_size, n_prompts=8, prompt_len=4, seed=1234):
    """Random [1, prompt_len] prompts (toy target has no tokenizer)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (1, prompt_len), generator=g) for _ in range(n_prompts)]


def evaluate(target, draft, prompts, temperature=0.0, n_draft=4, max_new_tokens=64,
             seed=0, check_lossless=False):
    """Run EAGLE-3 decoding over `prompts` and aggregate acceptance statistics.

    If check_lossless (only meaningful at temperature 0), also runs plain greedy
    decoding of the target and checks the outputs are identical token for token.
    """
    accepted, identical = [], []
    for i, prompt in enumerate(prompts):
        set_seed(seed + i)
        out, acc = generate(target, draft, prompt, max_new_tokens=max_new_tokens,
                            n_draft=n_draft, temperature=temperature)
        accepted += acc
        if check_lossless:
            ref = autoregressive_generate(target, prompt, max_new_tokens=max_new_tokens)
            identical.append(out == ref)
    result = {
        "temperature": temperature,
        "cycles": len(accepted),
        "accepted_per_cycle": accepted_per_cycle(accepted),
        "tokens_per_target_pass": tokens_per_pass(accepted),
        "n_alpha": n_alpha(accepted, n_draft),
    }
    if check_lossless:
        result["identical_to_greedy_target"] = all(identical)
        result["identical_count"] = f"{sum(identical)}/{len(identical)}"
    return result
