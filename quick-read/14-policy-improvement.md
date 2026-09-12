**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 13](13-policy-evaluation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 15 →](15-policy-iteration.md)

**Go deeper:** [Chapter 14: Policy Improvement](../chapters/14-policy-improvement/README.md)

---

<a id="topic-14"></a>

# 14. Policy Improvement

> **Why this algorithm exists:** A value estimate becomes useful for control when it helps choose a better action. Policy improvement turns an evaluator into a decision rule through one-step lookahead.

Suppose we know V<sup>π</sup>. We can improve the policy by choosing actions that look better according to one-step lookahead:

<p align="center">
  <img src="../assets/equations/equation-27.svg" alt="\pi&#x27;(s)=\arg\max_a\sum_{s&#x27;,r}p(s&#x27;,r|s,a)[r+\gamma V^{\pi}(s&#x27;)]." width="760">
</p>

The policy improvement theorem explains why a policy constructed greedily with respect to the current value function is at least as good as the original policy under the usual assumptions.

This creates the evaluation–improvement loop at the heart of many RL algorithms.


### Why the improvement is justified

For each state, calculate the expected reward plus discounted V<sup>π</sup> for every action. The largest of these numbers is at least their average under π. Consequently, choosing a maximizing action gives Q<sup>π</sup>(s,π&#x27;(s))≥V<sup>π</sup>(s).

With exact evaluation in the standard discounted setting, repeatedly using those improved choices yields V<sup>π&#x27;</sup>(s)≥V<sup>π</sup>(s) for every state. Approximate learned values can misrank actions, so this guarantee does not automatically transfer to neural-network training.

**Analogy:** improve the courier's instructions one junction at a time using an accurate assessment of the existing plan. When actions tie, use a consistent tie-breaking rule to avoid unnecessary policy changes.

---

**Continue in depth:** [Read Chapter 14](../chapters/14-policy-improvement/README.md)

[← Topic 13](13-policy-evaluation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 15 →](15-policy-iteration.md)
