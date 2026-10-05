"""
EAGLE-3 inference engine (article §XI) and the vanilla autoregressive baseline.

One drafting-verification cycle of `generate`:

    1. catch-up  draft over (g_i, e_{t_{i+1}}) for new real positions  -> a at last position
    2. draft     a -> LM head -> d1;  (a, e_d1) -> a' -> d2;  ...       -> d1..dk, q1..qk
    3. verify    target([pending, d1..dk])  -> p, new [l;m;h]
                 accept front to back       -> n_acc, next pending
    4. commit    ids += pending + d1..d_n_acc
                 crop target KV to the committed length
                 keep g only for committed positions; drop the draft's speculative K/V

SIMPLIFIED compared with the paper / a production engine: batch size 1, a chain
draft instead of EAGLE-2's dynamic tree, no CUDA graphs, Python-level verify.
The only change from the article listing is that new tensors are created on
`prompt_ids.device`, so the same code also runs with a GPU target.
"""

import torch
from transformers import DynamicCache

from eagle3.features import target_forward
from eagle3.sampling import probs, sample, verify


@torch.no_grad()
def generate(target, draft, prompt_ids, max_new_tokens=64, n_draft=4,
             temperature=0.0, eos_token_id=None):
    """EAGLE-3 decoding, batch size 1. SIMPLIFIED: a chain draft, not EAGLE-2's tree.

    Args:
        target:         frozen HuggingFace LLaMA-style causal LM.
        draft:          Eagle3Draft built for this target.
        prompt_ids:     [1, P] prompt token ids.
        max_new_tokens: number of tokens to generate.
        n_draft:        draft chain length k per cycle.
        temperature:    0 = greedy; >0 = sampling (still lossless).
        eos_token_id:   optional stop token.

    Returns (generated token ids, accepted-draft count per cycle).
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    ids = prompt_ids[0].tolist()              # committed tokens t_0 .. t_{n-1}

    # Prefill: the target reads the prompt and samples the next token ("How can" -> "I").
    logits, lmh, t_cache = target_forward(target, prompt_ids, DynamicCache())
    pending = sample(probs(logits[0, -1], temperature)).item()     # target token, not a draft
    new_g = draft.fuse(lmh)                   # g for positions the draft has not seen yet
    d_cache, d_len = None, 0                  # draft K/V, only for positions with REAL g
    accepted = []

    while len(ids) - prompt_len < max_new_tokens:
        n = len(ids)

        # 1) Catch the draft up: positions d_len..n-1 get (g_i, e_{t_{i+1}}).
        #    Embeddings are of the NEXT token (one-step shift, §V.6); the last
        #    real position pairs with the pending target token.
        nxt = torch.tensor([ids[d_len + 1:] + [pending]], device=device)
        x, d_cache = draft.layer(draft.embed(nxt), new_g,
                                 torch.arange(d_len, n, device=device), d_cache)
        d_len, committed = n, d_cache         # snapshot: K/V built from real g only
        a = x[:, -1:]                         # a at position n-1, e.g. a_I

        # 2) Draft n_draft tokens: each step feeds back (a, e_{draft token}).
        d_tokens, q = [], []
        for j in range(n_draft):
            qj = draft.to_target_vocab(probs(draft.logits(a)[0, -1], temperature))
            dj = sample(qj).item()
            d_tokens.append(dj)
            q.append(qj)
            if j + 1 < n_draft:
                pos = torch.tensor([n + j], device=device)
                # a stands in for the missing g of the token just drafted (§VI.6).
                a, d_cache = draft.layer(draft.embed(torch.tensor([[dj]], device=device)),
                                         a, pos, d_cache)

        # 3) Verify: one target pass over [pending, d_1..d_k].
        block = torch.tensor([[pending] + d_tokens], device=device)
        logits, lmh, t_cache = target_forward(target, block, t_cache)
        p = probs(logits[0], temperature)                            # [k+1, V]
        n_acc, nxt_tok = verify(torch.tensor(d_tokens), torch.stack(q), p)

        # 4) Commit, then roll both caches back to what is now real.
        ids += [pending] + d_tokens[:n_acc]
        pending = nxt_tok
        drop = t_cache.get_seq_length() - len(ids)
        if drop > 0:
            t_cache.crop(-drop)                                      # drop rejected positions
        new_g = draft.fuse(lmh[:, : 1 + n_acc])                      # g for the kept positions
        # Drafted positions were built from a, not g: discard them even if accepted,
        # because the verification pass just produced their real features.
        d_cache = {key: val[:, :, :d_len] for key, val in committed.items()}
        accepted.append(n_acc)
        if eos_token_id is not None and eos_token_id in ids[n:] + [pending]:
            break

    out = (ids + [pending])[prompt_len:prompt_len + max_new_tokens]
    if eos_token_id is not None and eos_token_id in out:
        out = out[: out.index(eos_token_id) + 1]
    return out, accepted


@torch.no_grad()
def autoregressive_generate(target, prompt_ids, max_new_tokens=64, temperature=0.0,
                            eos_token_id=None):
    """Vanilla autoregressive decoding: the reference EAGLE-3 must match.

    Deliberately recomputes the full sequence every step (no KV cache), so the
    reference shares no cache logic with `generate` and cannot hide its bugs.
    Only meant for small/toy targets.

    Returns the list of generated token ids.
    """
    ids = prompt_ids
    out = []
    for _ in range(max_new_tokens):
        logits = target(input_ids=ids).logits[0, -1]
        tok = sample(probs(logits, temperature)).item()
        out.append(tok)
        if eos_token_id is not None and tok == eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[tok]], device=ids.device)], 1)
    return out
