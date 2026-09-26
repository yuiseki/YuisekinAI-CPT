"""The corpus's wording, and the one thing the register is there for."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import phrasings  # noqa: E402

PROBE = [
    {"child_id": "abr-muni-403423", "child_ja": "篠栗町", "parent_ja": "福岡県",
     "split": "train"},
    {"child_id": "abr-muni-011011", "child_ja": "札幌市中央区",
     "parent_ja": "北海道", "split": "train"},
    {"child_id": "abr-muni-016942", "child_ja": "羅臼町", "parent_ja": "北海道",
     "split": "eval"},
]
REGISTER = [
    {"lg_code": "403423", "pref": "福岡県", "county": "糟屋郡", "name": "篠栗町"},
    {"lg_code": "011011", "pref": "北海道", "county": None,
     "name": "札幌市中央区"},
    {"lg_code": "016942", "pref": "北海道", "county": "目梨郡", "name": "羅臼町"},
]


def facts(**kw):
    return phrasings.facts("probe", rows=PROBE, register_rows=REGISTER, **kw)


def test_a_town_address_carries_the_county():
    """福岡県篠栗町 is how people write it, not what the address is.

    The 郡 is the reason the register is joined at all. A municipality's own
    name leaves it out, so a corpus built from names alone teaches 846 of the
    1,807 addresses in an abbreviated form and calls it a register.
    """
    assert phrasings.address("福岡県", "糟屋郡", "篠栗町") == "福岡県糟屋郡篠栗町"


def test_a_city_has_nothing_between_the_prefecture_and_itself():
    assert phrasings.address("北海道", None, "札幌市") == "北海道札幌市"
    assert phrasings.address("北海道", None, "札幌市中央区") == "北海道札幌市中央区"


def test_the_split_is_the_probes_split():
    """A held-out municipality must not reach the corpus by another door.

    The register has every municipality in the country and the probe holds
    one in ten out. Joining them the other way round, or forgetting the
    filter, puts the held-out facts back in and the eval half then measures
    recall like the other one.
    """
    trained = {f["child_id"] for f in facts()}
    assert "abr-muni-016942" not in trained
    assert trained == {"abr-muni-403423", "abr-muni-011011"}


def test_a_municipality_the_register_does_not_have_stops_the_build():
    """Rather than quietly writing a smaller corpus.

    The probe would still ask about it afterwards, and a fact that was never
    written reads exactly like a fact the model failed to learn.
    """
    with pytest.raises(SystemExit):
        phrasings.facts("probe", rows=PROBE, register_rows=REGISTER[1:])


def test_every_template_is_filled():
    """A stray placeholder would train the model on the word address.

    format() leaves nothing behind when every name is given, so what this
    catches is a template referring to a key the fact does not carry.
    """
    for f in facts():
        for text in phrasings.phrasings(f, len(phrasings.PHRASE)):
            assert "{" not in text and "}" not in text, text
            assert f["child"] in text or f["address"] in text, text


def test_the_eight_ways_are_eight_different_sentences():
    """Two templates that render the same are one template counted twice."""
    for f in facts():
        said = phrasings.phrasings(f, len(phrasings.PHRASE))
        assert len(set(said)) == len(said), said


def test_half_of_them_open_the_way_the_cloze_probe_asks():
    """The cloze probe opens 「{child}は」 and lets the model finish.

    If no sentence in the corpus started that way the probe would be asking
    in a shape the corpus never uses, and a model could hold the fact and
    score nothing. If they all did, the model could learn the one sentence.
    """
    f = facts()[0]
    said = phrasings.phrasings(f, len(phrasings.PHRASE))
    opens = [s for s in said if s.startswith(f["child"] + "は")]
    assert 2 <= len(opens) < len(said), said


def test_the_notebook_says_the_same_eight_things():
    """The notebook is a flattened copy, and a copy drifts.

    Each notebook in notebooks/ carries its own PHRASE table so that it runs
    on Colab with no checkout. If one ever disagrees with the module, the
    corpus that run trains on is not the one the tests here are about, and
    nothing would say so.

    Every notebook, not the newest: an older one that has been left behind by
    a change to the module is exactly the case worth catching, and the fix is
    to decide whether it is a record of a run that happened, in which case it
    should not be regenerated and this test should name it as such, or a
    notebook nobody has run yet.
    """
    import glob
    import json

    found = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "*.ipynb")))
    if not found:
        pytest.skip("no notebook built; run tools/mk_notebook.py")
    for path in found:
        check_notebook_phrasings(path)


def check_notebook_phrasings(path):
    import json

    nb = json.load(open(path, encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    scope = {}
    body = code[code.index("PHRASE = ["):]
    exec(body[:body.index("\n\n\ndef address")], scope)
    assert scope["PHRASE"] == phrasings.PHRASE, path
