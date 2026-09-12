**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 14](14-policy-improvement.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 16 →](16-value-iteration.md)

**Go deeper:** [Chapter 15: Policy Iteration](../chapters/15-policy-iteration/README.md)

---

<a id="topic-15"></a>

# 15. Policy Iteration

> **Why this algorithm exists:** A single greedy improvement changes the policy whose value you need. Policy iteration closes this loop by alternating evaluation and improvement until the policy stabilizes.

Policy iteration alternates:

<p align="center">
  <img src="../assets/diagrams/policy-iteration.svg" alt="Evaluate a policy, then improve it" width="760">
</p>

It exposes a recurring RL pattern: **prediction + control**.

Prediction estimates how good behavior is. Control changes behavior to make it better.


### A complete planning loop

```text
Choose an initial policy.
Repeat:
    Evaluate that policy to obtain V.
    At each state, choose an action maximizing expected reward + gamma * V(next).
    If the policy did not change, stop.
```

**Analogy:** a delivery manager measures the current routing plan, revises it using those measurements, then measures the revised plan. Evaluation and improvement solve different subproblems and help each other.

For a finite discounted MDP, exact policy iteration with consistent tie handling reaches an optimal policy. The main cost is evaluation, especially when there are many states. Using only a few evaluation sweeps gives a modified policy-iteration approach.

---

**Continue in depth:** [Read Chapter 15](../chapters/15-policy-iteration/README.md)

[← Topic 14](14-policy-improvement.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 16 →](16-value-iteration.md)
