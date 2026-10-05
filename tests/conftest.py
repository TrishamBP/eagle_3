"""Shared pytest fixtures: the toy target and an untrained draft (CPU, seconds to build)."""

import pytest
import torch

from eagle3.model import Eagle3Draft
from eagle3.toy import build_toy_target


@pytest.fixture(scope="session")
def target():
    """The article's toy LLaMA target (8 layers, k=64, vocab 96, sharpened head)."""
    return build_toy_target(seed=0)


@pytest.fixture()
def draft(target):
    """A fresh, untrained draft over a 64-token draft vocabulary."""
    torch.manual_seed(0)
    return Eagle3Draft.from_target(target, draft_vocab_ids=torch.arange(64)).eval()
