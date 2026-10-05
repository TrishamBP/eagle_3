"""
eagle3.experiments — one runnable script per result reported in the article.

Each module exposes:
    run(**options) -> dict   the experiment itself, returns its numbers
    main(argv=None)          argparse CLI wrapper (registered in pyproject.toml)

    param_count          §VIII.7   draft size at LLaMA-3.1-8B scale (meta device)
    shapes               §IX.7     tensor-shape walkthrough on the toy target
    check_ttt_attention  §VIII.2   dot-product TTT attention == explicit Figure 6 mask
    check_verify         §X.6      verify() output distribution == target p
    train_draft          §XI.6     train the toy draft with training-time test
    eval_lossless        §XI.6     greedy EAGLE-3 == greedy target; acceptance stats
    ablation_ttt_steps   §XII.8    7 TTT steps vs 1 step (the negative result)
    run_all              all of the above, summary written to runs/summary.json
"""
