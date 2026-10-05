"""
Multi-layer feature taps on the frozen target model (article §IX).

EAGLE-3 reads three hidden-state sequences from the target — low (l), middle (m)
and high (h) — and concatenates them per position into [l ; m ; h] (3k wide).
The paper does not say which layers; the official code uses the hidden states
ENTERING decoder layers 2, L//2 and L-3 (layers 2, 16, 29 on a 32-layer target).

`target_forward` is the only place the target model is touched, during both
prefill and verification. No extra target pass is needed:
`output_hidden_states=True` exposes states the forward pass computes anyway.
"""

import torch


def tap_layers(num_layers):
    """Official choice: hidden states entering decoder layers 2, L//2 and L-3.

    HuggingFace convention: hidden_states[0] is the embedding output and
    hidden_states[i] is the input to decoder layer i.
    """
    return (2, num_layers // 2, num_layers - 3)


@torch.no_grad()
def target_forward(target, input_ids, cache=None):
    """One target pass -> (logits, [l; m; h] per position, cache).

    Args:
        target:    HuggingFace causal LM (LLaMA-style), frozen.
        input_ids: [B, T] new tokens (the whole prompt at prefill, or the
                   verification block [pending, d_1..d_k]).
        cache:     transformers Cache (e.g. DynamicCache) or None.

    Returns:
        logits [B, T, V], lmh [B, T, 3k], updated cache.

    HuggingFace convention: hidden_states[i] is the input to decoder layer i.
    """
    out = target(input_ids=input_ids, past_key_values=cache,
                 use_cache=True, output_hidden_states=True)
    taps = tap_layers(target.config.num_hidden_layers)
    lmh = torch.cat([out.hidden_states[i] for i in taps], -1)       # [B, T, 3k]
    return out.logits, lmh, out.past_key_values
