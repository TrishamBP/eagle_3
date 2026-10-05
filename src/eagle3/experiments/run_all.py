"""
Run every experiment in the article's order and write runs/summary.json.

    1. param_count          §VIII.7
    2. shapes               §IX.7
    3. check_ttt_attention  §VIII.2
    4. check_verify         §X.6
    5. train_draft          §XI.6 / §XII.8   (always retrains, 7 TTT steps)
    6. eval_lossless        §XI.6            (uses the draft from step 5)
    7. ablation_ttt_steps   §XII.8           (reuses step 5's draft, trains the 1-step one)

Run:  uv run eagle3-all             full settings (a few minutes on a laptop CPU)
      uv run eagle3-all --quick     smoke run: fewer trials/steps, results in runs/quick/
"""

import argparse
import time
from pathlib import Path

from eagle3.experiments import (ablation_ttt_steps, check_ttt_attention, check_verify,
                                eval_lossless, param_count, shapes, train_draft)
from eagle3.utils import header, save_json


def run(quick=False, out_dir="runs"):
    """Run all experiments; returns and saves the combined results."""
    out_dir = str(Path(out_dir) / "quick") if quick else out_dir
    train_opts = {"out_dir": out_dir}
    if quick:
        train_opts["train_steps"] = 100
    trials = 20_000 if quick else 200_000

    t0 = time.time()
    summary = {"quick": quick}
    summary["param_count"] = param_count.run()
    summary["shapes"] = shapes.run()
    summary["check_ttt_attention"] = check_ttt_attention.run()
    summary["check_verify"] = check_verify.run(trials=trials)
    summary["train_draft"] = train_draft.run(ttt_steps=7, **train_opts)
    summary["eval_lossless"] = eval_lossless.run(**train_opts)
    summary["ablation_ttt_steps"] = ablation_ttt_steps.run(**train_opts)
    summary["seconds"] = round(time.time() - t0, 1)

    path = save_json(summary, Path(out_dir) / "summary.json")
    header(f"Done in {summary['seconds']}s — summary written to {path}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke run with reduced settings")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args(argv)
    run(quick=a.quick, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
