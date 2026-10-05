"""Reduced draft vocabulary: selection, d2t/in_draft buffers, scatter back to target vocab."""

import torch

from eagle3.model import Eagle3Draft, load_draft, save_draft
from eagle3.training import top_draft_vocab


def test_top_draft_vocab_picks_most_frequent_response_tokens():
    ids = torch.tensor([[9, 9, 1, 1, 1, 2]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])      # the two 9s are prompt, not response
    vocab = top_draft_vocab(ids, mask, vocab_size=10, n=2)
    assert vocab.tolist() == [1, 2]


def test_buffers_and_scatter():
    embed = torch.randn(10, 8)
    draft = Eagle3Draft(8, 2, 1, 16, embed, draft_vocab_ids=[7, 2, 5])
    assert draft.d2t.tolist() == [2, 5, 7]                   # sorted absolute ids
    assert draft.in_draft.nonzero().flatten().tolist() == [2, 5, 7]
    full = draft.to_target_vocab(torch.tensor([0.2, 0.3, 0.5]))
    assert full.shape == (10,)
    assert torch.allclose(full[[2, 5, 7]], torch.tensor([0.2, 0.3, 0.5]))
    assert torch.isclose(full.sum(), torch.tensor(1.0))
    assert not draft.embed.weight.requires_grad              # frozen copied embedding


def test_checkpoint_roundtrip(target, tmp_path):
    torch.manual_seed(0)
    draft = Eagle3Draft.from_target(target, draft_vocab_ids=torch.arange(10, 42))
    path = tmp_path / "d.pt"
    save_draft(draft, path, extra={"note": "test"})
    loaded, extra = load_draft(target, path)
    assert extra == {"note": "test"}
    assert torch.equal(loaded.d2t, draft.d2t)
    for (n1, p1), (n2, p2) in zip(draft.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
