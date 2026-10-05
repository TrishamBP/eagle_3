"""
Eagle3Draft — the complete EAGLE-3 draft model, plus checkpoint helpers.

Components (article §VIII.4-VIII.6):

    fc        Linear(3k -> k): fuses [l ; m ; h] target features into g   [Paper]
    embed     a FROZEN copy of the target's embedding table                [Code]
    layer     one DraftDecoderLayer                                        [Paper]
    norm      final RMSNorm before the LM head                             [Code]
    lm_head   DRAFT-OWNED, trained, over a reduced draft vocabulary        [Code]

The reduced vocabulary is not lossy: a draft that can never propose a rare
token just gets that position rejected, and the target emits it itself (§VIII.6).
"""

import torch
import torch.nn as nn

from eagle3.model.decoder import DraftDecoderLayer
from eagle3.model.layers import RMSNorm


class Eagle3Draft(nn.Module):
    """EAGLE-3 draft model.

    Args:
        k:               target hidden size.
        n_heads:         draft attention heads (official code mirrors the target).
        n_kv_heads:      draft key/value heads.
        inter:           SwiGLU intermediate width.
        target_embed:    [V, k] target embedding weight; copied and frozen.
        draft_vocab_ids: target token ids that make up the draft vocabulary
                         (None = full target vocabulary).
        rope_theta:      RoPE base (500000.0 = LLaMA-3.x).
        eps:             RMSNorm epsilon.

    Buffers:
        d2t       [V_d]  draft id -> target id (absolute ids; the official code
                         stores offsets, which is equivalent).
        in_draft  [V]    bool, is this target id inside the draft vocabulary?
    """

    def __init__(self, k, n_heads, n_kv_heads, inter, target_embed,
                 draft_vocab_ids=None, rope_theta=500000.0, eps=1e-5):
        super().__init__()
        vocab = target_embed.shape[0]
        self.vocab = vocab
        self.fc = nn.Linear(3 * k, k, bias=False)                    # [l; m; h] -> g
        self.embed = nn.Embedding.from_pretrained(target_embed.detach().clone(), freeze=True)
        self.layer = DraftDecoderLayer(k, n_heads, n_kv_heads, inter, rope_theta, eps)
        self.norm = RMSNorm(k, eps)
        if draft_vocab_ids is None:
            draft_vocab_ids = torch.arange(vocab)
        ids = torch.as_tensor(draft_vocab_ids).long().sort().values
        self.register_buffer("d2t", ids)                             # draft id -> target id
        in_draft = torch.zeros(vocab, dtype=torch.bool)
        in_draft[ids] = True
        self.register_buffer("in_draft", in_draft)                   # target id in draft vocab?
        self.lm_head = nn.Linear(k, len(ids), bias=False)            # draft-owned head

    @classmethod
    def from_target(cls, target, draft_vocab_ids=None, rope_theta=500000.0, eps=1e-5):
        """Build a draft sized to a HuggingFace LLaMA-style target (article §XI.6 usage).

        Head counts and MLP width are copied from the target config, and the
        embedding table is copied from `target.model.embed_tokens`.
        """
        c = target.config
        return cls(c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
                   c.intermediate_size, target.model.embed_tokens.weight,
                   draft_vocab_ids=draft_vocab_ids, rope_theta=rope_theta, eps=eps)

    def fuse(self, lmh):
        """[B, T, 3k] concatenated target features -> fused feature g [B, T, k]."""
        return self.fc(lmh)

    def logits(self, a):
        """Draft output a [B, T, k] -> logits over the draft vocabulary [B, T, Vd]."""
        return self.lm_head(self.norm(a))

    def to_target_vocab(self, draft_probs):
        """Scatter draft-vocab probabilities [..., Vd] into the target vocab [..., V].

        Needed so that verification compares p (target) and q (draft) over the
        same support. Tokens outside the draft vocabulary get probability 0.
        """
        out = draft_probs.new_zeros(*draft_probs.shape[:-1], self.vocab)
        return out.scatter(-1, self.d2t.expand_as(draft_probs), draft_probs)


# ----------------------------------------------------------------- checkpoints
def save_draft(draft, path, extra=None):
    """Save a draft's weights (+ optional metadata such as training config) to `path`."""
    torch.save({"state_dict": draft.state_dict(), "extra": extra or {}}, path)


def load_draft(target, path, rope_theta=500000.0, map_location="cpu"):
    """Rebuild a draft for `target` from a checkpoint written by `save_draft`.

    The draft vocabulary is recovered from the saved `d2t` buffer, so the
    reconstructed module has the same LM-head shape as the saved one.

    Returns (draft, extra metadata dict).
    """
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["state_dict"]
    draft = Eagle3Draft.from_target(target, draft_vocab_ids=state["d2t"], rope_theta=rope_theta)
    draft.load_state_dict(state)
    return draft, ckpt.get("extra", {})
