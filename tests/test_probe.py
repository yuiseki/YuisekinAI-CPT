"""Tests for the probe, including the two ways it has already been wrong."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import probe  # noqa: E402


def test_either_form_of_the_prefecture_name_counts():
    assert probe.correct("愛媛県です", "愛媛県")
    assert probe.correct("愛媛", "愛媛")
    assert probe.correct("松山市は愛媛にあります", "愛媛県")
    assert not probe.correct("東京都", "愛媛県")
    assert not probe.correct("", "愛媛県")


def test_hokkaido_is_not_trimmed_into_something_else():
    """The suffix rule strips 県府都道, and 北海道 ends in 道."""
    assert probe.bare("北海道") == "北海"
    assert probe.correct("北海道", "北海道")


def test_a_prompt_that_invites_a_blurb_is_a_separate_form():
    """"{city}は、" measured how the model talks, not what it knows: it scored
    1.6% where a question scored 6.4% on the same model. It is kept so the
    difference can be shown, and it is not a default.
    """
    assert "質問" in probe.prompt("松山市", "qa")
    assert probe.prompt("松山市", "plain") == "松山市は、"


def test_the_bands_are_population_because_sitelinks_separated_nothing():
    assert probe.band(326) == "<20k"
    assert probe.band(50_000) == "20k-100k"
    assert probe.band(500_000) == "100k+"
    assert probe.band(None) == "unknown"


def test_chance_is_reported_beside_the_score():
    s = probe.score([(500_000, True), (500_000, False), (326, False)])
    assert s["n"] == 3 and s["correct"] == 1
    assert abs(s["chance"] - 1 / 47) < 1e-9
    assert s["bands"]["100k+"]["n"] == 2
    assert s["bands"]["<20k"]["accuracy"] == 0.0


def test_the_answer_key_holds_municipalities_and_not_neighbourhoods():
    """長沼町 Q11653218 is a block of Tokyo with one sitelink. It was in the
    first answer key, and the 35B was marked wrong for answering 北海道.
    """
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "jp_municipalities.json")
    rows = json.load(open(path, encoding="utf-8"))
    assert len(rows) > 1000
    assert all(r.get("pref_ja") for r in rows)
    assert "Q11653218" not in {r["qid"] for r in rows}
    names = [r["ja"] for r in rows]
    assert len(names) == len(set(names)), "a name with two answers is in the set"
