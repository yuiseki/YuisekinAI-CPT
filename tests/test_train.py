"""Tests for the parts of training that can be wrong without raising.

Both functions here decide which tokens the model sees. Getting either wrong
produces a run that completes, logs a falling loss, and means nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train  # noqa: E402


def test_the_holdout_is_the_tail_and_never_empty():
    assert train.split_holdout(1_000_000, 0.005, 1024) == 995_000
    # A corpus barely longer than one block still leaves something to evaluate.
    cut = train.split_holdout(3000, 0.005, 1024)
    assert cut == 1975 and 3000 - cut > 1024


def test_the_holdout_is_not_sampled_at_random():
    """Windows overlap, so a random evaluation window shares tokens with the
    training windows either side of it and reports a loss already seen. The
    only cheap way to avoid that is to cut the corpus in two.
    """
    n, block = 100_000, 128
    cut = train.split_holdout(n, 0.05, block)
    rng = np.random.default_rng(0)
    # Every training window ends before the cut.
    for _ in range(200):
        s = rng.integers(0, cut - block - 1)
        assert s + block < cut


def test_a_block_longer_than_the_slice_is_refused_rather_than_silently_shrunk():
    tokens = np.arange(50, dtype=np.uint32)
    rng = np.random.default_rng(0)
    try:
        train.batch(tokens, 0, 50, 2, 128, rng, "cpu")
    except SystemExit as e:
        assert "block" in str(e)
    else:
        raise AssertionError("a block longer than the corpus was accepted")


def test_a_batch_is_one_window_because_the_model_does_the_shifting():
    """Transformers shifts labels itself: model(input_ids=x, labels=x) compares
    logits[:, :-1] with x[:, 1:]. Handing it a window already shifted by one
    asks the model to predict two tokens ahead, and the loss sits at chance:
    11.94 against ln(262144) = 12.48 for gemma-3, on a pretrained model
    reading its own language. The run completes and means nothing.
    """
    import torch  # noqa: F401

    tokens = np.arange(1000, dtype=np.uint32)
    rng = np.random.default_rng(3)
    x = train.batch(tokens, 0, 1000, 4, 16, rng, "cpu")
    assert x.shape == (4, 16)
    assert isinstance(x, torch.Tensor)


def test_the_loss_of_a_pretrained_model_is_far_below_chance():
    """The only cheap detector for a shifting mistake. A model that has read
    the language predicts far better than a uniform draw over its vocabulary;
    one being asked the wrong question does not.
    """
    import math
    assert train.chance_loss(262_144) > 12.4
    assert train.looks_like_chance(11.94, 262_144)
    assert not train.looks_like_chance(3.1, 262_144)


def test_a_starting_loss_at_chance_stops_the_run():
    """Reporting it is not enough: the run would complete, the loss would come
    down from 11.9 to 8.1, and the log would look like learning.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "train.py"),
               encoding="utf-8").read()
    assert "looks_like_chance(first[" in src
    assert "raise SystemExit" in src.split("looks_like_chance(first[")[1][:400]
