<a id="topic-42"></a>

# 42. Reasoning as a Sequential Decision Problem

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 41](41-from-policy-learning-to-inference-time-planning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 43 →](43-sampling-and-candidate-selection.md)

**Go deeper:** [Chapter 42](../chapters/42-reasoning-as-a-sequential-decision-problem/README.md)

Reasoning can be modeled as a sequence of decisions, but its state and action definitions must match the task. A token prefix, a tool transcript, and a search frontier carry different information.

**Analogy:** A courier's next road choice is different from the dispatcher's decision to inspect another route. The second is a computation-allocation action.

## Work through the idea

For operands (1,1,5), trace (3,7) ends at the correct total but contains two compensating arithmetic errors. Final correctness and valid intermediate steps are distinct metrics.

## Try it

[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Why should remaining budget be part of a search controller's state?

<details><summary>Answer</summary>

It changes which future computations are affordable and whether an unfinished branch can still produce a complete answer.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 42](../chapters/42-reasoning-as-a-sequential-decision-problem/README.md).

---

[← Topic 41](41-from-policy-learning-to-inference-time-planning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 43 →](43-sampling-and-candidate-selection.md)
