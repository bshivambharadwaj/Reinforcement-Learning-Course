<a id="topic-46"></a>

# 46. Budget-Aware Reasoning

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 45](45-search-over-reasoning-trajectories.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 47 →](47-learning-from-search.md)

**Go deeper:** [Chapter 46](../chapters/46-budget-aware-reasoning/README.md)

Inference compute has opportunity cost. Expanding, scoring, retrying, and stopping should be compared under an explicit resource model. Real systems may count tokens, tool calls, latency, or several resources separately.

**Analogy:** A dispatcher should not spend an hour optimizing an already adequate two-minute route.

## Work through the idea

Under independent attempts, the next oracle-success increment after n attempts is p(1−p)^n. At p=0.2, the sixth attempt adds about 0.06554. This diminishing-return calculation assumes reliable identification of success and does not solve general adaptive search.

## Try it

[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Why report consumed resources as well as the allowed budget?

<details><summary>Answer</summary>

Methods can stop early or exhaust their configured trees. An unused allowance is not computation that actually occurred.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 46](../chapters/46-budget-aware-reasoning/README.md).

---

[← Topic 45](45-search-over-reasoning-trajectories.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 47 →](47-learning-from-search.md)
