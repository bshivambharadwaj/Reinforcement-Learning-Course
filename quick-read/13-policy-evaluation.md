**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 12](12-dynamic-programming.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 14 →](14-policy-improvement.md)

**Go deeper:** [Chapter 13: Policy Evaluation](../chapters/13-policy-evaluation/README.md)

---

<a id="topic-13"></a>

# 13. Policy Evaluation

> **Why this algorithm exists:** Before improving a policy, ask how well it already performs. Policy evaluation separates that question from action selection, making the later improvement step interpretable.

Policy evaluation estimates V<sup>π</sup> for a fixed policy.

An iterative update is

<p align="center">
  <img src="../assets/equations/equation-26.svg" alt="V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s&#x27;,r}p(s&#x27;,r|s,a)[r+\gamma V_k(s&#x27;)]." width="760">
</p>

Repeated Bellman expectation backups converge to V<sup>π</sup> under standard finite discounted-MDP assumptions.

This gives us the first half of a powerful pattern:

**evaluate the current behavior before improving it.**


### How to evaluate a fixed policy

1. Initialize each nonterminal value, often to zero; keep terminal values at zero.
2. For every state, compute the expectation backup using the fixed policy.
3. Repeat until the largest value change is below a chosen tolerance.

If the first trip segment costs 1 and leads deterministically to a state currently valued at 8, its target with γ=0.9 is 6.2. Repeated sweeps propagate downstream rewards backward through the state space.

**Analogy:** estimate how well the courier's existing route-selection rule works before changing that rule. Evaluation changes the value estimates; it does not change the policy. A small numerical tolerance gives approximate evaluation, not exact equality.

---

**Continue in depth:** [Read Chapter 13](../chapters/13-policy-evaluation/README.md)

[← Topic 12](12-dynamic-programming.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 14 →](14-policy-improvement.md)
