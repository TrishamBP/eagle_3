"""
Experiment: losslessness and acceptance of EAGLE-3 decoding (article §XI.6).

For an UNTRAINED and a TRAINED draft (same draft vocabulary):

    * temperature 0: EAGLE-3 output must equal plain greedy decoding of the
      target, token for token — whatever the draft proposes;
    * temperature 0 and 1: drafts accepted per cycle.

Article result:

    Greedy, untrained draft   identical output; 0.00 drafts accepted per cycle
    Greedy, trained draft     identical output; 0.74 drafts accepted per cycle (1.74 tokens/pass)
    Temperature 1             0.73 -> 1.55 drafts accepted per cycle (untrained -> trained)

These are CORRECTNESS tests, not performance results: a 64-wide toy target on
CPU says nothing about GPU speedups. Exact acceptance numbers depend on seeds
and on settings the article doesn't state (see README) — the identical-output
checks must hold for every seed.

Run:  uv run eagle3-eval          (trains runs/draft_ttt7.pt first if it is missing)
"""

import argparse

from eagle3.experiments.common import (DEFAULT_TTT_STEPS, DEFAULTS, build_draft, evaluate,
                                       load_or_train, make_prompts)
from eagle3.experiments.train_draft import add_training_args, training_opts
from eagle3.toy import build_toy_target
from eagle3.utils import header, row


def _report(name, r):
    line = f"{r['accepted_per_cycle']:.2f} drafts/cycle ({r['tokens_per_target_pass']:.2f} tokens/pass)"
    if "identical_to_greedy_target" in r:
        line = f"identical={r['identical_to_greedy_target']} [{r['identical_count']}]; " + line
    row(name, line)


def run(n_prompts=8, prompt_len=4, max_new_tokens=64, n_draft=4, prompt_seed=1234,
        retrain=False, **opts):
    """Evaluate untrained vs trained drafts. Returns nested result dicts."""
    o = {**DEFAULTS, **opts}
    header("§XI.6  EAGLE-3 vs plain decoding on the toy target")
    target = build_toy_target(seed=o["target_seed"])
    trained = load_or_train(target, DEFAULT_TTT_STEPS, retrain=retrain, **o)
    untrained = build_draft(target, trained.d2t.clone(), seed=o["draft_seed"])

    prompts = make_prompts(target.config.vocab_size, n_prompts, prompt_len, prompt_seed)
    kw = dict(n_draft=n_draft, max_new_tokens=max_new_tokens)
    results = {}
    for name, draft in (("untrained", untrained), ("trained", trained)):
        results[name] = {
            "greedy": evaluate(target, draft, prompts, temperature=0.0, check_lossless=True, **kw),
            "temperature_1": evaluate(target, draft, prompts, temperature=1.0, **kw),
        }

    print()
    for name in ("untrained", "trained"):
        _report(f"greedy, {name} draft", results[name]["greedy"])
    for name in ("untrained", "trained"):
        _report(f"temperature 1, {name} draft", results[name]["temperature_1"])
    alphas = " ".join(f"{x:.2f}" for x in results["trained"]["greedy"]["n_alpha"])
    row("trained draft n-alpha at T=0 (0-a, 1-a, ...)", alphas)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--n-draft", type=int, default=4, help="draft chain length per cycle")
    ap.add_argument("--prompt-seed", type=int, default=1234)
    ap.add_argument("--retrain", action="store_true", help="ignore an existing checkpoint")
    add_training_args(ap)
    a = ap.parse_args(argv)
    run(n_prompts=a.n_prompts, max_new_tokens=a.max_new_tokens, n_draft=a.n_draft,
        prompt_seed=a.prompt_seed, retrain=a.retrain, **training_opts(a))


if __name__ == "__main__":
    main()
