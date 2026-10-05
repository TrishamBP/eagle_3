# eagle_three — EAGLE-3 Speculative Decoding from Scratch in PyTorch

A small, runnable reconstruction of **EAGLE-3** ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)).
It covers the draft model, multi-layer feature fusion, training-time test, lossless verification and a
chain-draft inference engine. It also includes **one script for each correctness experiment** reported in the companion article:

> **EAGLE-3: Engineering Implementation of LLM Inference Acceleration**
> `/engineering/eagle-3-speculative-decoding-llm-inference-acceleration/`
> (source: `../_implementations/eagle-3-speculative-decoding-llm-inference-acceleration.md`)

This is **not** the official implementation ([SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)).
Where the paper is silent, the layout follows the official code at commit `cb7e084`.

---

## What this project does — and doesn't

| Does | Doesn't |
| --- | --- |
| Rebuild the EAGLE-3 draft (two-stream decoder layer, fusion FC, frozen embedding, reduced-vocab LM head) | Reproduce the paper's GPU **speedups** (3.0x–6.5x). Nothing here measures wall-clock speed |
| Implement Figure 6 training-time-test attention with the diagonal dot-product trick | Use EAGLE-2's dynamic draft **tree**. This is a **chain** draft |
| Verify losslessness: greedy EAGLE-3 output == greedy target output, token for token | Support batching (batch size 1 only) |
| Re-run every test in article §XI.6 on a toy LLaMA target, on CPU | Load official EAGLE-3 checkpoints (module names differ, no key mapping) |

The model / engine / loss code is moved **logic-unchanged** from the article's Appendix C listing. It is split into files and commented.
The one code change: new tensors in `generate` and `ttt_loss` are created on the input's device, so a GPU target also works.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (`pip install uv`, or see the uv docs)
- Python 3.10+ (uv will fetch 3.11 automatically from `.python-version`)
- CPU is enough for every experiment. No GPU is needed.

The article's runs used **PyTorch 2.14.1 (CPU)** and **Transformers 5.18.0**. `pyproject.toml` only sets lower bounds
(`torch>=2.4`, `transformers>=4.46`). To pin the article's exact versions:

```bash
uv add "torch==2.14.1" "transformers==5.18.0"
```

---

## Install

```bash
cd eagle_three
uv sync --group dev      # creates .venv, installs torch, transformers, pytest and this package
```

## Quickstart

```bash
uv run pytest            # fast CPU test-suite (correctness checks in miniature)
uv run eagle3-all        # every experiment, in article order -> runs/summary.json
uv run eagle3-all --quick   # smoke run with reduced settings -> runs/quick/summary.json
```

All outputs (checkpoints, JSON) go to `runs/` (git-ignored). Run the commands from inside `eagle_three/`.

---

## Experiments

Each command maps to one result in the article. Every script also runs as
`uv run python -m eagle3.experiments.<module>` and accepts `--help`.

### 1. Draft size at LLaMA-3.1-8B scale — §VIII.7

```bash
uv run eagle3-param-count
```

Builds the draft on PyTorch's `meta` device, which allocates no memory. The settings are $k = 4096$, 32/8 heads, MLP 14,336, $V = 128{,}256$ and $V_d = 32{,}000$.

| Component | Article |
| --- | --- |
| `fc` (fusion 3k → k) | 50.3M |
| Attention (Q/K/V from 2k, O) | 67.1M |
| MLP (SwiGLU) | 176.2M |
| `lm_head` | 131.1M |
| **Trainable total** | **424.7M** |
| Embedding (frozen copy) | 525.3M |

These counts are fully determined by the shapes, so they should match exactly.

### 2. Tensor-shape walkthrough — §IX.7

```bash
uv run eagle3-shapes
```

The toy target (8 layers, $k = 64$) gets a 2-token prefix standing in for "How can". Expected:
`lmh (1, 2, 192)`, `g (1, 2, 64)`, attention input `(1, 2, 128)`, `a (1, 2, 64)`,
draft K/V `(1, 4, 2, 16)`, draft logits `(1, 2, 96)`. These should match exactly.

### 3. Training-time-test attention is exact — §VIII.2

```bash
uv run eagle3-check-ttt
```

Compares `DraftAttention.forward_ttt` with plain attention over all steps' keys using Figure 6's mask built
explicitly. `forward_ttt` uses a causal matmul for the native step and one dot product per position for each simulated step.
**Article: max |diff| 1.19e-7 at every step.** Expect ~1e-7 (float32 rounding). The exact digits depend on the seed.

### 4. Verification preserves the target distribution — §X.6

```bash
uv run eagle3-check-verify            # ~1 minute: 200K Python-level trials
```

Random $p$ (target) and $q$ (draft) over 8 tokens. Over many draft → `verify` trials, the emitted token must be distributed as $p$.
**Article: TV distance 0.0024 over 200K trials.** Expect roughly 0.002–0.003. That's the sampling-noise floor, so the exact value moves with the seed.

### 5. Train the toy draft — §XI.6, §XII.8

```bash
uv run eagle3-train                   # 7 TTT steps (official) -> runs/draft_ttt7.pt
uv run eagle3-train --ttt-steps 1     # no training-time test   -> runs/draft_ttt1.pt
```

The steps:

1. Build the toy target.
2. Let it generate **64** sequences itself (paper §4: target-regenerated data).
3. Keep the **64** most frequent response tokens as the draft vocabulary.
4. Train **800** steps of `ttt_loss`: 7 steps weighted $0.8^s$, AdamW $\beta = (0.9, 0.95)$, gradient clip 0.5.

### 6. Losslessness and acceptance — §XI.6

```bash
uv run eagle3-eval                    # trains runs/draft_ttt7.pt first if missing
```

| Test | Article |
| --- | --- |
| Greedy, **untrained** draft | Identical to plain greedy target; 0.00 drafts accepted / cycle |
| Greedy, **trained** draft | Identical; 0.74 drafts accepted / cycle (1.74 tokens per target pass) |
| Temperature 1, untrained → trained | 0.73 → 1.55 drafts accepted / cycle |

**The "identical" rows must hold for every seed and setting.** That is the losslessness guarantee.
The acceptance numbers depend on seeds and on the settings listed below, so you should see the same *pattern* (trained > untrained), not the same digits.

### 7. Ablation: 7 TTT steps vs 1 — §XII.8

```bash
uv run eagle3-ablation
```

**Article (a negative result):** both drafts are nearly identical, with 0-α ≈ 0.32–0.34, 1-α ≈ 0.40–0.44, and 0.88 vs 0.90 drafts accepted per cycle.
The toy target falls into repetitive loops, so later positions are easy for any draft. This setup therefore **cannot** test the paper's Figure 7 claim.
That needs a real target and real data. The script exists so the negative result can be reproduced too.

---

## Settings the article does not state (my defaults)

The article specifies the toy architecture, 64 sequences, a 64-token draft vocabulary, 800 steps, 7 TTT steps, the $0.8^s$ weights,
AdamW betas and clip 0.5. Everything else is my choice. Each one is a CLI flag:

| Setting | Default | Flag |
| --- | --- | --- |
| Toy MLP width | 128 | (in `toy.py`) |
| Seeds (target / data / draft / prompts) | 0 / 0 / 0 / 1234 | `--target-seed`, `--data-seed`, `--draft-seed`, `--prompt-seed` |
| Training sequence length | 64 | `--seq-len` |
| Random prompt length | 4 | (in `common.py`) |
| Data sampling temperature | 1.0 | (in `common.py`) |
| Batch size | 8 | `--batch-size` |
| Learning rate | **1e-3** (paper: 5e-5 at real scale) | `--lr` |
| Eval prompts / new tokens / chain length | 8 / 64 / 4 | `--n-prompts`, `--max-new-tokens`, `--n-draft` |

---

## Project map

```
eagle_three/
├── pyproject.toml                 uv project + console commands
├── src/eagle3/
│   ├── model/
│   │   ├── layers.py              RMSNorm, RoPE                             App. C
│   │   ├── attention.py           DraftAttention: forward + forward_ttt     §VIII.2, §VII.5
│   │   ├── decoder.py             two-stream DraftDecoderLayer              §VIII.1
│   │   └── draft.py               Eagle3Draft, save_draft / load_draft      §VIII.4–6
│   ├── features.py                tap_layers (2, L/2, L-3), target_forward  §IX
│   ├── sampling.py                probs, sample, verify                     §X
│   ├── engine.py                  generate (EAGLE-3), autoregressive ref    §XI
│   ├── metrics.py                 acceptance / n-α / TV distance            §XIII
│   ├── training/
│   │   ├── ttt.py                 shift_left, ttt_loss                      §XII.8
│   │   ├── data.py                self-generated data, top-N draft vocab    §XII.2, §VIII.6
│   │   └── trainer.py             AdamW + clip training loop                §XII.5–7
│   ├── toy.py                     the toy LLaMA target                      §XI.6
│   ├── utils.py                   seeding, JSON, printing
│   └── experiments/               one script per article result (see above)
└── tests/                         pytest versions of the checks
```

How one EAGLE-3 cycle maps onto the code (`engine.generate`):

```text
prefill   target_forward(prompt)        -> logits, [l;m;h]   -> pending ("I"), g = draft.fuse(...)
loop
  1 catch-up  draft.layer(e_next, g)     real features for new committed positions
  2 draft     draft.logits(a) -> d_j     then (a, e_dj) -> a'  ...   k draft tokens
  3 verify    target_forward([pending, d_1..d_k]) -> p ; verify(d, q, p) -> n_acc, next token
  4 commit    crop target KV, keep g for accepted positions, drop speculative draft K/V
```

---

## Using a real HuggingFace target

The library works with any HuggingFace LLaMA-style causal LM:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eagle3 import Eagle3Draft, generate, ttt_loss

path = "meta-llama/Llama-3.1-8B-Instruct"            # needs a GPU with enough memory
tok = AutoTokenizer.from_pretrained(path)
target = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
target.requires_grad_(False)

draft = Eagle3Draft.from_target(target, draft_vocab_ids=None,   # or your top-32K token ids
                                rope_theta=500000.0).cuda().to(torch.bfloat16)

# ... train the draft: loss, per_step = ttt_loss(target, draft, input_ids, loss_mask) ...

prompt_ids = tok("How can I", return_tensors="pt").input_ids.cuda()
out, accepted = generate(target, draft, prompt_ids, max_new_tokens=256, n_draft=4,
                         temperature=0.0, eos_token_id=tok.eos_token_id)
print(tok.decode(out))
```

Caveats:

- A useful draft needs the paper's training scale: ShareGPT + UltraChat-200K with target-regenerated responses, many epochs.
- An untrained draft is still **lossless**. It is just slow, because nothing gets accepted.
- This code cannot load official EAGLE-3 checkpoints.

---

## Attribution

- Paper: Li, Wei, Zhang, Zhang — *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, arXiv:2503.01840.
- Official code: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), consulted at commit `cb7e084`. It is the source of the layer taps,
  the missing separate FC, the frozen embedding, the reduced-vocab LM head, 7 TTT steps and the $0.8^s$ soft cross-entropy.
- Reconstruction, toy harness and tests: Trisham Patil.
