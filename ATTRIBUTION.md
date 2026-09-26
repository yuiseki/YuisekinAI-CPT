# Attribution and licence

The code in this repository is MIT. See LICENSE. It does not cover the data
it reads, and that data does not cover it.

Nothing here redistributes a corpus. Every dataset is fetched from the Hub at
run time, and the two files checked in under `data/` are described below.

## What the runs read

| what | where | licence |
|---|---|---|
| the facts and the addresses | [`yuiseki/geo-triples-jp-gov`](https://huggingface.co/datasets/yuiseki/geo-triples-jp-gov), [`yuiseki/jp-admin-2026-09`](https://huggingface.co/datasets/yuiseki/jp-admin-2026-09) | CC BY 4.0 |
| the earlier corpora | [`yuiseki/geo-triples-tokyo23`](https://huggingface.co/datasets/yuiseki/geo-triples-tokyo23), [`yuiseki/geo-triples-japan`](https://huggingface.co/datasets/yuiseki/geo-triples-japan) | ODbL-1.0 |
| the control corpus | [`yuiseki/wikipedia-geotagged`](https://huggingface.co/datasets/yuiseki/wikipedia-geotagged) | CC BY-SA 4.0 |
| the base models | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base), [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) | Apache-2.0, Gemma Terms of Use |

The CC BY 4.0 half is Japan's own registers:

> 「アドレス・ベース・レジストリ」（デジタル庁）
> 「令和2年国勢調査 小地域（町丁・字等別）境界データ」（総務省統計局）

The ODbL half is OpenStreetMap:

> (c) OpenStreetMap contributors, available under the Open Database License.
> https://www.openstreetmap.org/copyright

A corpus built from an ODbL dataset is a Derivative Database and carries
ODbL. That is one of the reasons the jp-gov work exists separately: share-
alike is contagious and attribution is not.

## The two files under data/

`data/jp_municipalities.json` is the answer key of the older generalisation
probe: 1,134 Japanese municipalities and their prefecture, with a Wikidata
item id, a sitelink count and a population for each. It is built by
`scripts/build_probe_set.py` from
[`yuiseki/wikidata-gazetteer`](https://huggingface.co/datasets/yuiseki/wikidata-gazetteer),
which comes from Wikidata. Wikidata's data is CC0, so this file is CC0.

`data/dry/*.json` are three ten-line manifests naming a dataset, a config and
a split for the dry run. They contain no data.

## Models published from this code

[`yuiseki/qwen3-0.6b-jp-gov-v0.1`](https://huggingface.co/yuiseki/qwen3-0.6b-jp-gov-v0.1),
Apache-2.0 after its base model, with its own card stating what went into it.
Whether weights are a derivative of their training text is unsettled; the card
names the sources so a reader can judge.
