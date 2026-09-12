**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 11](11-bellman-optimality-equations.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 13 →](13-policy-evaluation.md)

**Go deeper:** [Chapter 12: Dynamic Programming](../chapters/12-dynamic-programming/README.md)

---

<a id="topic-12"></a>

# 12. Dynamic Programming

> **Why this algorithm exists:** If the environment model is known, sampling every outcome wastes information. Dynamic programming uses that model to propagate future consequences backward. Its price is enumerating states and outcomes; an unknown or inaccurate map removes this advantage.

Dynamic Programming (DP) solves MDPs when the environment model is known.

The key operation is a **Bellman backup**:

<p align="center">
  <img src="../assets/diagrams/dynamic-programming.svg" alt="A Bellman backup with a known model" width="760">
</p>

DP is rarely the final solution for huge modern environments because exact dynamics and full state sweeps are often unavailable. But it provides the conceptual blueprint for much of RL.

**[Try it in Notebook 01 → MDPs and Dynamic Programming](../notebooks/01_mdp_dynamic_programming.ipynb)**


### What having a model means

A model tells us the probabilities and rewards for each possible transition. DP uses this information to calculate expected targets instead of estimating them from sampled trips. Here, “programming” means organizing a recursive optimization, not a particular programming language.

A full backup enumerates outcomes, computes reward plus discounted next-state value, and combines them by averaging or maximizing as appropriate. A sweep applies backups across states.

**Analogy:** if you have a complete road map with reliable travel-time distributions, you can compare routes at your desk. Without that map, you must learn from journeys. DP assumes the model is available; learning a model from experience belongs to model-based RL.


**Read the source:** [Sutton & Barto, Chapter 4](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 12](../chapters/12-dynamic-programming/README.md)

[← Topic 11](11-bellman-optimality-equations.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 13 →](13-policy-evaluation.md)
