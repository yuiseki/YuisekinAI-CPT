# Natural Earth admin-2 as a second corpus: what to know before building it

`ne-admin2` is defined in YuisekinGeoSPARQL's `builder/sources.py` and served
by no manifest yet. 3,224 counties, the United States and nothing else,
public domain, joined to a state by the `REGION` column. As a rung it is the
exact counterpart of municipality-in-prefecture, in English, and the licence
is looser than anything used so far.

Two things were measured before proposing that it be built.

## Neither base model knows it the way llm-jp knows Japan

Chance is 2.0%: 51 states.

    llm-jp-3-440m   「X County is in the state of」   30.0%
    llm-jp-3-440m   three-shot question               26.0%

Against 91% for Japanese municipalities asked as a question. So for llm-jp
this is a corpus of new knowledge rather than new phrasings, which by the
measurement in `tokenizer_question.md` means it wants many passes rather than
few. That is a prediction this corpus can test: run 3 and run 4 say six epochs
for a corpus that is 93% phrasing work, and this one should want more.

## Half the names have no answer

    3,224 counties, 1,957 distinct "name type" strings
      unique to one state   1,533   (78% of names)
      rows with a name that is not unique   1,691   (52% of rows)

      Washington County   30 states
      Jefferson County    25
      Franklin County     24
      Lincoln County      23

Japanese municipality names are almost all unique, so the question "which
prefecture is 松山市 in" has an answer. "Which state is Washington County in"
does not. Half of this dataset cannot be asked, and more importantly cannot be
written: a corpus containing both "Washington County is in Maryland" and
"Washington County is in Oregon" is 30 contradictions rather than 30 facts.

This is the same class of error as the one `src/probe.py` already carries a
docstring about, where filtering Wikidata by the last character of a name let
in 町丁 and asked which prefecture a Tokyo city block is in.

## So the corpus has to disambiguate, and there are three ways

Use only the 1,533 unique names. Smallest and simplest, and it throws away the
half of the country that is named after presidents.

Name the county with its state in the subject and ask something else of it.
That changes the rung being taught and is a different experiment.

Carry the FIPS code or the full "Washington County, Maryland" form in the
sentence. This is what people actually write, it keeps all 3,224, and it makes
the subject longer, which interacts with the name-length question rather than
being independent of it.

None of these is obviously right. What is clear is that the 52% cannot be fed
in as they stand.

## Postscript: Qwen does not know them either

    model             「X County is in the state of」   three-shot question
    llm-jp-3-440m                    30.0%                     26.0%
    Qwen3-0.6B-Base                  17.5%                     27.0%

Chance is 2.0%, and both figures are against an answer key that has no answer
for half the questions, so read them as a floor rather than as a score. Neither
model holds this rung. The English-pretrained model is not better at the
American one, which is worth remembering before assuming a corpus is easy for
a model because of what language it is in.
