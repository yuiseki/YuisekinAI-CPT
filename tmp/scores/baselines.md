# What these models score before anyone trains them

Measured on this machine with `tools/baseline.py`, which runs the notebook's
own cells up to the training heading with MODEL rebound, so the questions,
the sampling and the scoring are the same ones the runs use.

                          llm-jp-3-440m   Qwen3-0.6B-Base   qwen3-0.6b-jp-gov
                              (before)        (before)        v0.1.1 (after)
    cloze train                 31.0%            3.3%              93.5%
    cloze train no-leak         25.2%            2.3%              90.2%
    cloze eval                  33.7%            4.7%              28.5%
    cloze eval no-leak          28.5%            3.8%              24.1%
    qa train                    90.5%            3.8%              73.8%
    qa train no-leak            91.8%            1.8%              75.2%
    qa eval                     91.9%           10.5%              18.6%
    qa eval no-leak             91.1%            3.8%              19.6%

Chance is 2.1%. The Qwen before-column is from run 1 on Colab; a repeat on
this machine's CPU is in `baseline_qwen3_06b.json`.

## The 91% was checked before it was believed

A base model that will not stop talking scores well under a rule that looks
for the answer anywhere in what it said, so the qa figure was rescored three
ways over 200 questions with `tools/verify_qa.py`:

    substring anywhere    182/200   91.0%     <- what the probe reports
    first line exactly    182/200   91.0%
    names one and right   182/200   91.0%

All three agree exactly. The raw answers are a single prefecture and then a
question the model invents for itself:

    留萌市   want 北海道 | 北海道 \n \n Q: 由利本荘市はどの都道府県にありますか。 \n A:

The third scorer first read 86.5%, and that was a bug in the scorer: 京都 is a
substring of 東京都, so every correct 東京都 counted as naming two prefectures.
Nine answers, all of them Tokyo. Fixed to match full names.

## What this means for the project

llm-jp-3-440m knows the municipality-to-prefecture map at 91% before any
training. Qwen3-0.6B-Base knows it at 3.8%, and sixty epochs of continued
pretraining brought it to 75.2% on the half of the facts the corpus states and
19.6% on the half it does not. A smaller model, freely licensed under the same
Apache 2.0, starts far above where the trained one finished.

Two further things fall out of the table.

The gap between cloze and qa runs opposite ways in the two models. llm-jp
answers 91% of the questions and completes 31% of the sentences; the trained
Qwen completes 93.5% of the sentences and answers 75.2% of the questions.
Knowing a fact and being able to state it in a given form are separable, and
each model is short of a different one. This is the same thing run 2 saw when
the exposure schedule moved qa by 17 points and cloze by 4.

A comparison of the two models does not isolate the tokenizer. llm-jp was
pretrained on Japanese; Qwen was not, to anything like the same degree. Any
difference between them carries both. The tokenizer question has to be asked
inside one model: after training llm-jp, do the names that llm-jp cuts into
one or two tokens fail the way Qwen's two-token names failed? If they do, short
names are hard for a reason no tokenizer design fixes. If they do not, the
tokenizer is where the fix lives.
