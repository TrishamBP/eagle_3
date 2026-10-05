"""
Experiment: training-time test on vs off (article §XII.8, "a negative result").

Trains two drafts on the same toy data, vocabulary and seeds — one with 7 TTT
steps, one with 1 step (no training-time test) — and compares per-position
acceptance n-alpha and drafts accepted per cycle at temperature 0.

Article result: nearly identical — 0-alpha ~0.32-0.34, 1-alpha ~0.40-0.44,
0.88 vs 0.90 drafts per cycle. The toy target's outputs fall into repetitive
loops, which makes later positions easy for ANY draft, so this setup cannot
test the paper's Figure 7 claim. Reproducing that needs a real target and data.
It is included so the negative result is reproducible too.

Run:  uv run eagle3-ablation      (reuses runs/draft_ttt7.pt / draft_ttt1.pt if present)
"""

import argparse

from eagle3.experiments.common import DEFAULTS, evaluate, load_or_train, make_prompts
from eagle3.experiments.train_draft import add_training_args, training_opts
from eagle3.toy import build_toy_target
from eagle3.utils import header, row


def run(step_counts=(7, 1), n_prompts=8, prompt_len=4, max_new_tokens=64, n_draft=4,
        prompt_seed=1234, retrain=False, **opts):
    """Train/load one draft per TTT step count and evaluate each greedily."""
    o = {**DEFAULTS, **opts}
    header(f"§XII.8  Ablation: TTT steps {list(step_counts)} on the toy target")
    target = build_toy_target(seed=o["target_seed"])
    prompts = make_prompts(target.config.vocab_size, n_prompts, prompt_len, prompt_seed)

    results = {}
    for steps in step_counts:
        draft = load_or_train(target, steps, retrain=retrain, **o)
        results[f"ttt_{steps}"] = evaluate(target, draft, prompts, temperature=0.0,
                                           n_draft=n_draft, max_new_tokens=max_new_tokens)

    print()
    for name, r in results.items():
        alphas = " ".join(f"{x:.2f}" for x in r["n_alpha"])
        row(f"{name}: n-alpha (0-a, 1-a, ...)", alphas)
        row(f"{name}: drafts accepted per cycle", f"{r['accepted_per_cycle']:.2f}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step-counts", type=int, nargs="+", default=[7, 1],
                    help="TTT step counts to compare")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--n-draft", type=int, default=4)
    ap.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    add_training_args(ap)
    a = ap.parse_args(argv)
    run(step_counts=tuple(a.step_counts), n_prompts=a.n_prompts, max_new_tokens=a.max_new_tokens,
        n_draft=a.n_draft, retrain=a.retrain, **training_opts(a))


if __name__ == "__main__":
    main()
