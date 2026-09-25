"""Tests for the VRAM budget.

The point of the module is that for this model the intuition is wrong: the
memory goes on the vocabulary, not on the weights.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import budget  # noqa: E402

GEMMA = budget.CONFIGS["google/gemma-3-270m"]


def test_the_parameter_count_matches_the_name():
    p = budget.parameters(GEMMA)
    assert 260e6 < p["total"] < 275e6
    assert p["embedding"] / p["total"] > 0.6


def test_the_logits_dominate_at_a_reasonable_batch():
    b = budget.budget(GEMMA, batch=8, block=1024)
    logits = b["logits"] + b["logits upcast"]
    weights = b["parameters"] + b["gradients"] + b["optimizer"] + b["master weights"]
    assert logits > 2 * weights


def test_the_fixed_cost_is_independent_of_the_batch():
    keys = ("parameters", "gradients", "optimizer", "master weights")
    small = budget.budget(GEMMA, 1, 512)
    large = budget.budget(GEMMA, 32, 4096)
    assert sum(small[k] for k in keys) == sum(large[k] for k in keys)


def test_memory_grows_with_the_product_of_batch_and_block():
    """1 x 2048 and 2 x 1024 are the same number of tokens and the same cost,
    which is what makes gradient accumulation a free way to shrink the peak.
    """
    a = sum(budget.budget(GEMMA, 1, 2048).values())
    b = sum(budget.budget(GEMMA, 2, 1024).values())
    assert abs(a - b) / a < 0.01


def test_a_small_vocabulary_changes_the_shape_of_the_answer():
    """pythia-160m has 50,304 entries against gemma's 262,144, so the same
    batch costs a fifth of the logits. The advice differs by model.
    """
    g = budget.budget(GEMMA, 8, 1024)
    p = budget.budget(budget.CONFIGS["EleutherAI/pythia-160m"], 8, 1024)
    assert g["logits"] / p["logits"] > 4
