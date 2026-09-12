<a id="topic-48"></a>

# 48. Evaluating Inference-Time Reasoning

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 47](47-learning-from-search.md) | [Course home](../README.md) | [Quick reads](README.md) | [Part III exercises →](../chapters/PART_III_EXERCISES.md)

**Go deeper:** [Chapter 48](../chapters/48-evaluating-inference-time-reasoning/README.md)

Evaluate selected answers, candidate opportunities, intermediate validity, abstentions, and actual cost. Hold task distributions and policy checkpoints fixed when isolating an inference strategy's effect.

**Analogy:** A courier's best possible proposed route is not the same as the route actually dispatched and completed.

## Work through the idea

If 60 of 100 tasks are solved and 80 receive any completed answer, overall success is 60% while accuracy conditional on completion is 75%. Omitting abstentions changes the claim.

## Try it

[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Do three generation seeds measure variation across trained models?

<details><summary>Answer</summary>

No. They vary sampling from one frozen policy. Training robustness requires independently trained checkpoints or another explicitly designed evaluation.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 48](../chapters/48-evaluating-inference-time-reasoning/README.md).

---

[← Topic 47](47-learning-from-search.md) | [Course home](../README.md) | [Quick reads](README.md) | [Part III exercises →](../chapters/PART_III_EXERCISES.md)
