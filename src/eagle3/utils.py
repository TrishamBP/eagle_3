"""Small helpers shared by the experiments: seeding, output paths, JSON, printing."""

import json
import random
from pathlib import Path

import torch


def set_seed(seed):
    """Seed Python and PyTorch RNGs (sampling, init, multinomial, torch.rand)."""
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    """Create `path` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    """Write `obj` as pretty JSON to `path`, creating the parent directory."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))
    return path


def header(title):
    """Print a section header so multi-experiment logs stay readable."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def row(label, value):
    """Print one aligned `label : value` result line."""
    print(f"  {label:<48} {value}")
