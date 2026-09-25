"""Tests for turning a dataset into a flat array of token ids.

The shape matters more than it looks. A window sampled from the array can
straddle two documents, so what sits between them is what the model learns
about where an article ends.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import corpus  # noqa: E402


class FakeTokenizer:
    """One id per character, so the test can read the output by eye."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def test_documents_are_separated_by_the_end_of_text_id():
    out = list(corpus.encode(["ab", "cd"], FakeTokenizer(), eos_id=0, batch=8))
    assert out == [[97, 98, 0], [99, 100, 0]]


def test_batching_does_not_change_what_is_written():
    docs = ["a", "bb", "ccc", "dddd", "e"]
    one = list(corpus.encode(docs, FakeTokenizer(), eos_id=0, batch=1))
    many = list(corpus.encode(docs, FakeTokenizer(), eos_id=0, batch=3))
    assert one == many


def test_the_file_reads_back_as_the_tokens_that_went_in():
    ids = [[1, 2, 3], [4, 5], [6]]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.bin")
        n = corpus.write(iter(ids), path)
        back = np.fromfile(path, dtype=corpus.DTYPE)
    assert n == 6
    assert back.tolist() == [1, 2, 3, 4, 5, 6]


def test_a_flush_boundary_does_not_lose_or_reorder_tokens():
    """The buffer is flushed by size, and an off-by-one there is silent."""
    ids = [[i] for i in range(1000)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.bin")
        n = corpus.write(iter(ids), path, flush_every=7)
        back = np.fromfile(path, dtype=corpus.DTYPE)
    assert n == 1000
    assert back.tolist() == list(range(1000))


def test_uint16_would_not_hold_the_vocabularies_in_play():
    """gemma-3 has 262,144 entries and OLMo 2 has 100,352."""
    assert np.iinfo(corpus.DTYPE).max >= 262_144


def test_the_manifest_sits_beside_the_binary():
    assert corpus.manifest_path("data/ja.bin") == "data/ja.json"


def test_the_builder_leaves_without_finalising_the_interpreter():
    """A streaming dataset stopped early by --limit leaves an HTTP thread
    mid-retry, and interpreter finalisation with that thread alive aborts with
    PyGILState_Release after the file is already written. Exiting directly is
    the fix, and this test is here so that removing it is a deliberate act.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "corpus.py"),
               encoding="utf-8").read()
    assert "os._exit(0)" in src
