"""Shape checks matching the article's §IX.7 walkthrough and §VIII.7 parameter table."""

from eagle3.experiments import param_count, shapes
from eagle3.features import tap_layers


def test_tap_layers_official_choice():
    # Hidden states entering layers 2, L//2, L-3.
    assert tap_layers(8) == (2, 4, 5)
    assert tap_layers(32) == (2, 16, 29)


def test_shape_walkthrough_matches_article():
    s = shapes.run()
    assert s["lmh [B, T, 3k]"] == [1, 2, 192]
    assert s["g [B, T, k]"] == [1, 2, 64]
    assert s["[e; g] attention in [B, T, 2k]"] == [1, 2, 128]
    assert s["a [B, T, k]"] == [1, 2, 64]
    assert s["draft K/V cache [B, heads, T, head_dim]"] == [1, 4, 2, 16]
    assert s["draft logits [B, T, V_d]"] == [1, 2, 96]


def test_param_count_at_8b_scale():
    c = param_count.run()
    assert round(c["fc (fusion 3k->k)"] / 1e6, 1) == 50.3
    assert round(c["attention (Q/K/V from 2k, O)"] / 1e6, 1) == 67.1
    assert round(c["mlp (SwiGLU)"] / 1e6, 1) == 176.2
    assert round(c["lm_head (draft vocab)"] / 1e6, 1) == 131.1
    assert round(c["trainable total"] / 1e6, 1) == 424.7
    assert round(c["embedding (frozen copy)"] / 1e6, 1) == 525.3
