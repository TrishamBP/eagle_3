"""
The small test target used for every experiment in the article (§XI.6).

A randomly initialised HuggingFace `LlamaForCausalLM`:

    8 layers, hidden size k = 64, 4 heads, 2 KV heads, vocabulary 96

Its LM-head weights are scaled by 8 ("sharpened"), so next-token distributions
look more like a real LM's than a near-uniform random model's.

From the article: layers / k / heads / KV heads / vocab / x8 sharpening.
My defaults (not stated in the article): intermediate_size=128,
max_position_embeddings=1024, seed=0.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

TOY_CONFIG = dict(
    vocab_size=96,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=1024,
    tie_word_embeddings=False,
)


def build_toy_target(seed=0, sharpen=8.0, device="cpu"):
    """Build the frozen toy LLaMA target.

    Args:
        seed:    weight-init seed. Every experiment uses the same seed so they
                 all talk about the same target.
        sharpen: multiplier on the LM-head weights (article: 8).
        device:  where to put the model.

    Returns:
        LlamaForCausalLM in eval mode with requires_grad=False everywhere.
    """
    torch.manual_seed(seed)
    target = LlamaForCausalLM(LlamaConfig(**TOY_CONFIG))
    with torch.no_grad():
        target.lm_head.weight.mul_(sharpen)
    target.requires_grad_(False)
    return target.eval().to(device)
