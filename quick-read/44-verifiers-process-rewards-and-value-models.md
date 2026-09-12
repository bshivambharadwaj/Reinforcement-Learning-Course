<a id="topic-44"></a>

# 44. Verifiers, Process Rewards, and Value Models

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 43](43-sampling-and-candidate-selection.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 45 →](45-search-over-reasoning-trajectories.md)

**Go deeper:** [Chapter 44](../chapters/44-verifiers-process-rewards-and-value-models/README.md)

A final verifier checks an outcome, a process scorer inspects intermediate steps, and a value model predicts a future quantity under a defined continuation policy. Their scores are not interchangeable.

**Analogy:** A delivery receipt, a traffic-rule check, and an estimated arrival probability answer three different questions.

## Work through the idea

If 10% of candidates are correct, a verifier with 90% true positives and 10% false positives accepts a set that is only 50% correct under those rates. Prevalence and selection pressure matter.

## Try it

[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Is the fraction of correct steps a calibrated success probability?

<details><summary>Answer</summary>

No. It describes the observed prefix and does not define future success under a policy, search strategy, or remaining budget.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 44](../chapters/44-verifiers-process-rewards-and-value-models/README.md).

---

[← Topic 43](43-sampling-and-candidate-selection.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 45 →](45-search-over-reasoning-trajectories.md)
