<a id="topic-41"></a>

# 41. From Policy Learning to Inference-Time Planning

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 40](40-multimodal-rl-and-rl-for-ai-agents.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 42 →](42-reasoning-as-a-sequential-decision-problem.md)

**Go deeper:** [Chapter 41](../chapters/41-from-policy-learning-to-inference-time-planning/README.md)

A frozen policy can produce better selected answers when a system generates alternatives, evaluates them, and allocates inference computation. The weights stay fixed; the search procedure changes which candidate is returned.

**Analogy:** A courier can consult the same route manual several times and compare routes without rewriting the manual. Revising it later is a separate learning stage.

## Work through the idea

Four independent attempts with success probability 0.2 contain at least one success with probability 1−0.8⁴=0.5904. That is an oracle opportunity, not the probability that an imperfect selector returns the correct candidate.

## Try it

[Notebook 12](../notebooks/12_sampling_and_selection.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Does generating several candidates and comparing them make inference GRPO?

<details><summary>Answer</summary>

No. GRPO updates a policy using a specified training objective. Candidate comparison alone is selection or search.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 41](../chapters/41-from-policy-learning-to-inference-time-planning/README.md).

---

[← Topic 40](40-multimodal-rl-and-rl-for-ai-agents.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 42 →](42-reasoning-as-a-sequential-decision-problem.md)
