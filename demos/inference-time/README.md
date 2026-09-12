# Practical demo: one policy, five inference strategies

**Created by Shivam Bharadwaj** · [Course home](../../README.md) · [Part III](../../chapters/41-from-policy-learning-to-inference-time-planning/README.md)

Compare single sampling, self-consistency, best-of-N, sampled beam search, and best-first
search on the same arithmetic tasks. The proposal policy remains frozen. Change the
scoring signal or budget and inspect which trajectories the system selects.

<p align="center"><img src="../../assets/diagrams/inference-time-planning.svg" width="760" alt="Frozen proposals feed a scored search loop, followed by independent final evaluation"></p>

## Run it

The CLI uses only the Python standard library, with Python 3.10 or newer. From the full
repository root:

```bash
python -m rl_course.inference_demo --output /tmp/rl-inference-demo.html --json /tmp/rl-inference-results.json
```

Open the generated HTML file in your browser. On Windows, choose paths such as
`--output inference-demo.html --json inference-results.json` instead of `/tmp/...`.
No model downloads, API keys, server, or browser network requests are needed.

A recorded [interactive HTML report](demo.html) and its [raw results](results.json) are
included. GitHub displays HTML source rather than running the demo in a repository file
view. Download the HTML and open it locally, or generate your own report with the command.
Browser selectors filter recorded runs; they do not recompute experiments or run an LLM.

## Change the experiment

```bash
python -m rl_course.inference_demo --tasks 200 --budgets 12 24 48 96 --seeds 7 19 41 --depth 4 --score-cost 3 --output /tmp/rl-inference-cost-study.html
```

| Setting | Meaning |
|---|---|
| `--tasks` | Number of unique generated operand tuples |
| `--budgets` | Maximum work units allowed per task and strategy |
| `--seeds` | Generation seeds, not training seeds |
| `--depth` | Number of intermediate addition decisions |
| `--score-cost` | Work units charged per scorer query |
| `--output` | Standalone interactive report destination |
| `--json` | Raw seed-level aggregates and first-task expansion logs |

Use [Notebook 13](../../notebooks/13_budgeted_reasoning_search.ipynb) to vary beam width
and branching or inspect Python-level results. The CLI uses width two and branching two.

## Task and policy contract

ArithmeticTrace-v1 requires left-to-right addition. For `(2,3,4,5)`, the valid trace is
`(5,9,14)`. The frozen categorical policy computes a local sum from the claimed predecessor,
then samples an offset from `(-1,0,+1)` with probabilities `(0.1,0.4,0.5)`.

This deliberately makes the modal local proposal wrong. Arithmetic is built into the
proposal interface; the experiment does not establish learned arithmetic ability or
open-domain LLM reasoning. Its purpose is to isolate candidate selection, search,
evaluator failure, and resource accounting in a system learners can inspect completely.

The exact process scorer measures the fraction of locally correct steps. The misleading
scorer rewards `+1` errors. Neither is a learned value model. Search receives the chosen
score, while a separate final evaluator checks the actual sum and cumulative ground truth
after selection. Unit tests check that inference does not call the final evaluator.

## Cost and selection rules

A proposed step costs one unit. Each score query costs the configured amount. Final
benchmark truth checks are outside the inference budget and are unavailable for selection.
These teaching units are not measured FLOPs, tokens, latency, or financial costs.

Best-of-N scores only complete candidates; tree methods score prefixes. Query lengths
can therefore differ despite identical abstract query costs. Self-consistency votes
without querying the scorer. Single sampling leaves extra allowance unused.

Beam search retains two prefixes per depth after sampling two children per parent.
Best-first expands the highest-scoring prefix, breaking ties by greater depth and then
insertion order. Both charge duplicate samples. Both have finite configured trees and
can exhaust them before spending a large allowance. If no completed candidate exists,
the system abstains and counts the task as unsuccessful.

## Recorded observations

The included report uses 100 unique tasks with operands in `[10,30)`, task seed 2026,
depth three, generation seeds 7/19/41, and scorer cost one. These are measurements of
this specific constructed experiment, not general algorithm rankings.

| Strategy / signal | Budget cap | Selected answer correct | Valid process | Mean units used |
|---|---:|---:|---:|---:|
| Single sampling | 48 | 22.7% | 8.3% | 3.0 |
| Self-consistency | 48 | 9.0% | 3.0% | 48.0 |
| Best-of-N / exact process | 48 | 53.3% | 52.7% | 48.0 |
| Beam / exact process | 48 | 35.3% | 32.7% | 20.0 |
| Best-first / exact process | 48 | 33.7% | 29.3% | 28.0 |
| Best-of-N / misleading | 48 | 0.0% | 0.0% | 48.0 |

More samples can strengthen a systematic wrong majority or help a scorer select its own
preferred mistakes. Final answers can also be correct through compensating intermediate
errors. Compare oracle candidate opportunity, selected success, valid process, completion,
and cost together before explaining any apparent gain.

## Exercises and next step

1. Find a budget where a tree method has no complete candidate. Explain its spending trace.
2. Keep candidate samples fixed and change only the scorer. Identify the selected error.
3. Raise scorer cost to three and repeat the strategy comparison.
4. Use [Notebook 14](../../notebooks/14_learning_from_search.ipynb) to fit a separate student
   from accepted search traces. Identify exactly where learning begins.

Use the [Part III assessment](../../chapters/PART_III_EXERCISES.md) for worked calculation
answers and a complete evaluation rubric. Code is [MIT](../../LICENSE-MIT); educational
text, visualizations, and recorded reports use [CC BY 4.0](../../LICENSE-CC-BY-4.0).
