<a id="topic-43"></a>

# 43. Sampling and Candidate Selection

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 42](42-reasoning-as-a-sequential-decision-problem.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 44 →](44-verifiers-process-rewards-and-value-models.md)

**Go deeper:** [Chapter 43](../chapters/43-sampling-and-candidate-selection/README.md)

Sampling creates candidates; selection determines which one is returned. Best-of-N chooses a scorer's favorite, self-consistency votes over final answers, and oracle pass rate asks whether any candidate is correct.

**Analogy:** Several route suggestions can be ranked by popularity, by a predictor, or by verified delivery success. These rules use different evidence.

## Work through the idea

With depth 3, score cost 1, and budget 12, best-of-N scores three candidates while self-consistency samples four. Equal budget caps do not imply equal candidate counts. A systematic wrong answer can dominate a larger vote.

## Try it

[Notebook 12](../notebooks/12_sampling_and_selection.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Does a higher oracle pass rate prove a better deployed selector?

<details><summary>Answer</summary>

No. A correct answer can exist in the pool while the selector consistently chooses an incorrect one.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 43](../chapters/43-sampling-and-candidate-selection/README.md).

---

[← Topic 42](42-reasoning-as-a-sequential-decision-problem.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 44 →](44-verifiers-process-rewards-and-value-models.md)
