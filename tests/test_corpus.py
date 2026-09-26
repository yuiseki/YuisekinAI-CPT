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


def test_where_selects_rows_by_column():
    assert corpus.parse_where(["form=ja"]) == {"form": {"ja"}}
    assert corpus.parse_where(["a=1", "b=2"]) == {"a": {"1"}, "b": {"2"}}
    assert corpus.parse_where(None) == {}


def test_a_comma_means_any_of():
    """One column, several acceptable values.

    The two directions of a containment are two predicates saying one fact,
    and a run that wants both cannot say so with two flags: those would read
    as a conjunction and keep nothing.
    """
    assert corpus.parse_where(["predicate=sfWithin,sfContains"]) == {
        "predicate": {"sfWithin", "sfContains"}}


def test_where_without_a_value_is_refused():
    """--where form ja, which would otherwise select nothing and say nothing."""
    import pytest
    with pytest.raises(SystemExit):
        corpus.parse_where(["form"])


def test_the_limit_counts_what_is_kept_not_what_is_read(monkeypatch):
    """A filter matching the tail of a table returning an empty corpus.

    The rows of geo-triples-tokyo23's cpt table are sorted by form, so every
    ja row sits after 126,208 ntriples rows. Counting rows read rather than
    rows kept would make a small --limit produce nothing and look like a
    broken filter.
    """
    rows = ([{"text": f"n{i}", "form": "ntriples"} for i in range(5)] +
            [{"text": f"j{i}", "form": "ja"} for i in range(5)])
    monkeypatch.setitem(sys.modules, "datasets",
                        type(sys)("datasets"))
    sys.modules["datasets"].load_dataset = lambda *a, **k: rows
    got = list(corpus.documents("d", "cpt", "train", "text", 2,
                                {"form": {"ja"}}))
    assert got == ["j0", "j1"]
