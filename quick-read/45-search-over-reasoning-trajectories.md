<a id="topic-45"></a>

# 45. Search over Reasoning Trajectories

**Quick read · Part III** · **Created by Shivam Bharadwaj**

[← Topic 44](44-verifiers-process-rewards-and-value-models.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 46 →](46-budget-aware-reasoning.md)

**Go deeper:** [Chapter 45](../chapters/45-search-over-reasoning-trajectories/README.md)

Search can score partial trajectories and decide which to expand. Beam search retains a bounded frontier at each depth; best-first uses a priority queue. Both can lose useful branches when the ranking signal is wrong.

**Analogy:** The courier compares junctions before completing every route, but a misleading map can discard the only successful path.

## Work through the idea

The demo's sampled beam with width 2, branching 2, and depth 3 makes 2+4+4=10 proposals. With one scoring unit per proposal, it spends 20 units even if the cap is 48. A fixed tree can be exhausted.

## Try it

[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb) provides the related experiment.
The [practical demo](../demos/inference-time/README.md) compares five strategies under frozen
proposal probabilities and explicit scoring costs. It is a controlled arithmetic simulator,
not an LLM benchmark.

## Knowledge check

Is the demo's priority-queue search MCTS?

<details><summary>Answer</summary>

No. It lacks MCTS visit/value backups and exploration statistics. It is explicitly best-first search with a fixed prefix scorer.

</details>

For the derivation, assumptions, further exercises, and primary references,
[continue to Chapter 45](../chapters/45-search-over-reasoning-trajectories/README.md).

---

[← Topic 44](44-verifiers-process-rewards-and-value-models.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 46 →](46-budget-aware-reasoning.md)
