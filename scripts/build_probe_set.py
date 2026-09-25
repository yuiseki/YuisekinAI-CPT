#!/usr/bin/env python3
"""Build the answer key: Japanese municipalities and their prefecture.

From wikidata-gazetteer, by class rather than by the last character of the
name. Filtering on 市町村 as characters let in 町丁, the neighbourhood, and
長沼町 Q11653218 is a block of Tokyo with one sitelink rather than the town in
Hokkaido. The 35B was marked wrong for answering correctly.

Names that belong to more than one municipality are dropped: the question has
no single answer, and a model cannot be scored on it either way.

    python3 scripts/build_probe_set.py \
        --places /path/to/wikidata/parquet/places.parquet \
        --names  /path/to/wikidata/parquet/names.parquet \
        --out data/jp_municipalities.json
"""
import argparse
import collections
import json
import sys

# 市, 町, 村 of Japan. Q5327369 町丁 and Q486972 human settlement are what the
# character filter used to let in.
MUNICIPALITY = {"Q494721", "Q1059478", "Q4174776"}
PREFECTURE = "Q50337"
JAPAN = "Q17"


def labels(table, lang):
    import pyarrow.compute as pc
    f = table.filter(pc.and_(pc.equal(table.column("lang"), lang),
                             pc.equal(table.column("kind"), "label")))
    out = {}
    for q, n in zip(f.column("qid").to_pylist(), f.column("name").to_pylist()):
        out.setdefault(q, n)
    return out


def main():
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t = pq.read_table(a.places, columns=["qid", "country", "instance_of",
                                         "located_in", "sitelinks", "population"])
    jp = t.filter(pc.equal(t.column("country"), JAPAN))
    cols = {c: jp.column(c).to_pylist()
            for c in ("qid", "instance_of", "located_in", "sitelinks", "population")}
    prefectures = {q for q, i in zip(cols["qid"], cols["instance_of"])
                   if i and PREFECTURE in i}

    n = pq.read_table(a.names, columns=["qid", "lang", "name", "kind"])
    ja, en = labels(n, "ja"), labels(n, "en")

    rows = []
    for q, inst, loc, sl, pop in zip(cols["qid"], cols["instance_of"],
                                     cols["located_in"], cols["sitelinks"],
                                     cols["population"]):
        if not inst or not MUNICIPALITY.intersection(inst):
            continue
        parents = [p for p in (loc or []) if p in prefectures]
        if len(parents) != 1 or q not in ja:
            continue
        rows.append({"qid": q, "ja": ja[q], "en": en.get(q),
                     "pref_qid": parents[0], "pref_ja": ja.get(parents[0]),
                     "pref_en": en.get(parents[0]), "sitelinks": sl,
                     "population": int(pop) if pop else None})

    duplicated = {k for k, v in collections.Counter(r["ja"] for r in rows).items()
                  if v > 1}
    rows = [r for r in rows if r["ja"] not in duplicated and r["pref_ja"]]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{len(rows):,} municipalities  (dropped {len(duplicated)} ambiguous names)")
    print("  prefectures found:", len(prefectures))
    by = collections.Counter(r["ja"][-1] for r in rows)
    print("  by class:", dict(by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
