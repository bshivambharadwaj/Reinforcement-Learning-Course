**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 20](20-q-learning-and-exploration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 22 →](22-deep-q-networks.md)

**Go deeper:** [Chapter 21: Function Approximation](../chapters/21-function-approximation/README.md)

---

<a id="topic-21"></a>

# 21. Function Approximation

> **Why this method exists:** A table cannot reuse what it learned about one observation at another. Function approximation shares parameters across states; the benefit is generalization and the risk is interference between estimates.

Tabular methods store a separate value for each state or state-action pair. That becomes impossible when spaces are huge or continuous.

Instead, approximate:

<p align="center">
  <img src="../assets/equations/equation-37.svg" alt="\hat V(s;w),\qquad \hat Q(s,a;w),\qquad \pi_\theta(a|s)." width="760">
</p>

Neural networks can generalize across similar states, but approximation also introduces instability: changing parameters for one sample can change predictions for many others.

This transition—from tables to learned representations—is what makes **deep reinforcement learning** possible.


### Generalization is useful and risky

A table memorizes each junction separately. A function approximator can learn that low battery is risky across many junctions, including ones rarely visited. Features may be hand-designed, linear, or learned by a network.

**Analogy:** replace a notebook of individual addresses with a rule that generalizes across neighborhoods. An incorrect rule can also spread mistakes widely.

The **deadly triad** refers to function approximation, bootstrapping, and off-policy learning together: this combination can make value learning diverge. Deep RL therefore needs care with targets, data distributions, and optimization. Increasing network size does not by itself solve these problems.


> **Failure Mode — the deadly triad:** function approximation, bootstrapping, and off-policy learning can interact to destabilize value estimates. Parameter sharing spreads errors, bootstrap targets reuse them, and the training distribution may differ from the target policy's distribution. The combination is a warning about possible instability, not a statement that every DQN run must diverge.

> **Engineering Note:** monitor value magnitudes and actual return together. A falling regression loss alone cannot establish that the policy improved.


**Read the source:** [Sutton & Barto, Chapter 11](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 21](../chapters/21-function-approximation/README.md)

[← Topic 20](20-q-learning-and-exploration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 22 →](22-deep-q-networks.md)
