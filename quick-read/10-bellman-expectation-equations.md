**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 9](09-advantage-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 11 →](11-bellman-optimality-equations.md)

**Go deeper:** [Chapter 10: Bellman Expectation Equations](../chapters/10-bellman-expectation-equations/README.md)

---

<a id="topic-10"></a>

# 10. Bellman Expectation Equations

The Bellman equation introduces one of RL's deepest ideas: **a long-horizon value can be decomposed recursively**.

For state value:

<p align="center">
  <img src="../assets/equations/equation-19.svg" alt="V^{\pi}(s)=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma V^{\pi}(S_{t+1})\mid S_t=s\right]." width="760">
</p>

Expanded over actions and transitions:

<p align="center">
  <img src="../assets/equations/equation-20.svg" alt="V^{\pi}(s)=\sum_a\pi(a|s)\sum_{s&#x27;,r}p(s&#x27;,r|s,a)\left[r+\gamma V^{\pi}(s&#x27;)\right]." width="760">
</p>

Conceptually:

<p align="center">
  <img src="../assets/diagrams/bellman-decomposition.svg" alt="A long future, one step at a time" width="760">
</p>

This recursive structure is the foundation of dynamic programming and temporal-difference learning.


### One-step lookahead, with numbers

Suppose an action gives reward −1, then reaches a state of value 8 with probability 0.75 or value 4 with probability 0.25. Its expected one-step target is

<p align="center">
  <img src="../assets/equations/equation-21.svg" alt="-1+0.9[0.75(8)+0.25(4)]=5.3." width="760">
</p>

For V<sup>π</sup>, also average these action-specific targets using the policy's action probabilities. For Q<sup>π</sup>, fix the first action and average subsequent actions under the policy:

<p align="center">
  <img src="../assets/equations/equation-22.svg" alt="Q^{\pi}(s,a)=\sum_{s&#x27;,r}p(s&#x27;,r|s,a)
\left[r+\gamma\sum_{a&#x27;}\pi(a&#x27;|s&#x27;)Q^{\pi}(s&#x27;,a&#x27;)\right]." width="760">
</p>

**Analogy:** the remaining travel time equals time to the next junction plus remaining travel time from there. Bellman equations express this consistency for expected discounted reward.

---

**Continue in depth:** [Read Chapter 10](../chapters/10-bellman-expectation-equations/README.md)

[← Topic 9](09-advantage-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 11 →](11-bellman-optimality-equations.md)
