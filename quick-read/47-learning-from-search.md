<a id="topic-47"></a>

# 47. Learning from Search

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 46](46-budget-aware-reasoning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 48 →](48-evaluating-inference-time-reasoning.md)

**Go deeper:** [Chapter 47](../chapters/47-learning-from-search/README.md)

A separate training stage can imitate verified trajectories discovered by search. Teacher generation, filtering, student fitting, and held-out inference evaluation should be recorded independently.

**Analogy:** A courier learns a simpler route manual from costly planning sessions, then uses it on later deliveries.

## Work through the idea

Ten valid three-step teacher traces yield offset counts [0,30,0]. Unit pseudocounts produce student probabilities [1/33,31/33,1/33]. This lab fits a categorical policy, not a language model, and its update is supervised imitation rather than PPO or GRPO.

## Try it

[Notebook 14](../notebooks/14_learning_from_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Why can outcome-only filtering teach invalid intermediate behavior?

<details><summary>Answer</summary>

A correct final answer can contain compensating errors. Accepting the whole trace reinforces those actions in a student that cannot model the correction structure.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 47](../chapters/47-learning-from-search/README.md).

---

[← Topic 46](46-budget-aware-reasoning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 48 →](48-evaluating-inference-time-reasoning.md)
