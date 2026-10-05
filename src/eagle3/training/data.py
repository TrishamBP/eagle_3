"""
Training data for the draft (article §XII.2 and §VIII.6).

Target-generated data [Paper]: the paper regenerates the responses with the
target model itself, so the draft learns the target's own output distribution —
exactly what verification will test it against. For the toy target we do the
same: start from short random prompts and let the target sample a continuation.

Reduced draft vocabulary [Code]: the official recipe keeps the N most frequent
tokens in the training responses (N = 32,000 for LLaMA-3.1-8B; 64 in the
article's toy run) and maps them back to target ids via `d2t`.
"""

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from eagle3.sampling import probs


@dataclass
class ToyDataset:
    """Self-generated training sequences.

    input_ids: [N, T] prompt + target continuation.
    loss_mask: [N, T] 1 on target-generated positions, 0 on the random prompt
               (the analogue of "assistant-response tokens only").
    """

    input_ids: torch.Tensor
    loss_mask: torch.Tensor

    def batch(self, batch_size, generator=None):
        """Random mini-batch (with replacement across calls): ([B, T] ids, [B, T] mask)."""
        idx = torch.randint(0, self.input_ids.shape[0], (batch_size,), generator=generator)
        return self.input_ids[idx], self.loss_mask[idx]


@torch.no_grad()
def make_self_generated_dataset(target, n_seqs=64, seq_len=64, prompt_len=4,
                                temperature=1.0, seed=0):
    """Let the target generate its own training sequences (paper §4, data regeneration).

    Args:
        target:      the frozen target model.
        n_seqs:      number of sequences (article toy run: 64).
        seq_len:     total length per sequence, prompt included (my default: 64).
        prompt_len:  random prompt tokens at the start (my default: 4).
        temperature: sampling temperature for the continuation (my default: 1.0,
                     so the data covers more than one greedy path).
        seed:        RNG seed for prompts and sampling.

    Returns:
        ToyDataset with input_ids / loss_mask of shape [n_seqs, seq_len].
    """
    g = torch.Generator().manual_seed(seed)
    device = next(target.parameters()).device
    vocab = target.config.vocab_size

    ids = torch.randint(0, vocab, (n_seqs, prompt_len), generator=g).to(device)
    cache = DynamicCache()
    out = target(input_ids=ids, past_key_values=cache, use_cache=True)
    cache, logits = out.past_key_values, out.logits[:, -1]
    seq = [ids]
    for _ in range(seq_len - prompt_len):
        p = probs(logits, temperature)                               # [N, V]
        nxt = torch.multinomial(p.cpu(), 1, generator=g).to(device)  # [N, 1]
        seq.append(nxt)
        out = target(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]

    input_ids = torch.cat(seq, 1)
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[:, prompt_len:] = 1
    return ToyDataset(input_ids=input_ids, loss_mask=loss_mask)


def top_draft_vocab(input_ids, loss_mask, vocab_size, n):
    """Pick the n most frequent tokens in the supervised (response) positions.

    Args:
        input_ids:  [N, T] training sequences.
        loss_mask:  [N, T] which positions count as responses.
        vocab_size: target vocabulary size V.
        n:          draft vocabulary size V_d (clamped to V).

    Returns:
        [V_d] sorted target token ids, to pass as `draft_vocab_ids`.
    """
    tokens = input_ids[loss_mask.bool()].flatten().cpu()
    counts = torch.bincount(tokens, minlength=vocab_size)
    return counts.topk(min(n, vocab_size)).indices.sort().values
