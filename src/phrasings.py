#!/usr/bin/env python3
"""Eight ways to say one containment, for the government registers.

A fact met in eight contexts is remembered in far fewer passes than the same
fact met eight times in one. Two runs against the OpenStreetMap corpus raised
the exposure instead and the score followed, 9.2% at thirty passes and 23.3%
at a hundred, but the control loss began to rise and every sentence still had
the same shape. A model shown "X市は〈県〉に含まれる。" a hundred times learns
the sentence and a prior over prefecture names; the gradients for one template
mostly cancel.

What the registers add over the OpenStreetMap corpus is the address. The
Address Base Registry is an address register, so it carries the 郡 that a
town's address needs and a municipality's own name leaves out: 篠栗町 is in
福岡県糟屋郡, and 福岡県篠栗町 is how people write it rather than what it is.
846 of the 1,807 municipalities have one. The fact set still comes from the
probe, so the split is the probe's split and nothing held out is written.

    from phrasings import facts, phrasings
    for f in facts("yuiseki/geo-triples-jp-gov"):
        for text in phrasings(f, 8):
            ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Where the 郡 comes from. The probe carries the fact and the two names; this
# carries how the name is written down in an address.
REGISTER = "yuiseki/jp-admin-2026-09"
REGISTER_CONFIG = "municipalities"

# Four open with the child's name and a は, which is how the cloze probe
# asks, so the question is not a shape the model has never seen. The other
# four do not, which is what stops it learning the sentence instead of the
# fact.
#
# The last two are the register's own content rather than a sentence about
# it. An address is what this source is for, and a model that has read
# 福岡県糟屋郡篠栗町 has been told where 篠栗町 is by the only means an
# address has.
PHRASE = [
    "{child}は{parent}に含まれる。",
    "{parent}は{child}を含む。",
    "{child}は{parent}にある。",
    "{child}の所在する都道府県は{parent}である。",
    "{parent}の市区町村のひとつが{child}である。",
    "{child}は{parent}の市区町村である。",
    "住所：{address}",
    "{address}",
]


def address(pref, county, name):
    """福岡県糟屋郡篠栗町, and 北海道札幌市中央区 with no county to place.

    The name of a designated city's ward already carries its city, so the
    three parts are all there is: nothing else goes between a prefecture and
    a municipality in a Japanese address.
    """
    return f"{pref}{county or ''}{name}"


def phrasings(fact, n):
    """The first n ways of saying one fact.

    n rather than all of them, so that a run can vary the number and see what
    the variety is worth rather than assuming it.
    """
    return [t.format(child=fact["child"], parent=fact["parent"],
                     address=fact["address"])
            for t in PHRASE[:n]]


# How many times a fact is written, by how many tokens its subject takes.
#
# Run 1 learnt 1.7% of the six-token names wrongly and 60.3% of the two-token
# ones. The rate falls monotonically in between, and it is not capacity: 1,632
# facts is about 9,000 bits. A short name gives the model one or two places to
# hang a fact on, and one of them is the 市 or 町 it shares with 800 others.
#
# Two other things were measured and are deliberately not used here. The
# frequency of the name's rarest token in Japanese Wikipedia predicts the same
# failures slightly less well (131 of the 188 against 153) and costs a
# frequency table the builder would have to carry. The size of the answering
# prefecture predicts them independently and much more weakly, and evening it
# out would mean repeating whole prefectures, 4.6 times the corpus rather than
# 1.6.
def repeats(name_tokens):
    """How many times to write each phrasing of a fact with this subject.

    Written as a comparison rather than a lookup so that a name shorter than
    any in this dataset gets the most exposure rather than the least, which a
    table with no entry for it would have given.
    """
    if name_tokens <= 2:
        return 3
    if name_tokens == 3:
        return 2
    return 1


def code_of(child_id):
    """abr-muni-403423 -> 403423, the local government code the register uses."""
    return child_id.rsplit("-", 1)[-1]


def facts(dataset, revision=None, split="train", register=REGISTER,
          register_revision=None, rows=None, register_rows=None):
    """One fact per municipality, with its name, its prefecture and its address.

    The fact set is the probe's, filtered to one side of its split, so the
    corpus cannot contain a municipality the evaluation holds out. The
    register is joined on the code only to spell the address; it adds no
    facts and cannot add a municipality the probe does not have.

    rows and register_rows are for the tests, which have no network.
    """
    if rows is None:
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(dataset, "probe", split="train",
                                              revision=revision)]
    if register_rows is None:
        from datasets import load_dataset
        register_rows = load_dataset(
            register, REGISTER_CONFIG, split="train",
            revision=register_revision).remove_columns(["geometry"])
    by_code = {r["lg_code"]: r for r in register_rows}

    out = []
    for r in rows:
        if split and r.get("split") != split:
            continue
        if not (r.get("child_ja") and r.get("parent_ja")):
            continue
        m = by_code.get(code_of(r["child_id"]))
        if m is None:
            # Loudly. A silent skip here would drop facts from the corpus and
            # leave the probe still asking about them, which reads afterwards
            # as a model that failed to learn them.
            raise SystemExit(f"{r['child_id']} is not in {register}; the two "
                             f"are not the same vintage")
        out.append({
            "child_id": r["child_id"],
            "child": r["child_ja"],
            "parent": r["parent_ja"],
            "address": address(m["pref"], m["county"], m["name"]),
        })
    out.sort(key=lambda f: f["child_id"])
    return out
