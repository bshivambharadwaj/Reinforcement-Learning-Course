**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 17](17-monte-carlo-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 19 →](19-sarsa.md)

**Go deeper:** [Chapter 18: Temporal-Difference Learning](../chapters/18-temporal-difference-learning/README.md)

---

<a id="topic-18"></a>

# 18. Temporal-Difference Learning

> **Why this algorithm exists:** Waiting for a whole delivery is unnecessary when the next state already has a useful value estimate. TD learns after each transition by bootstrapping; imperfect estimates now influence later estimates.

Temporal-Difference (TD) learning combines ideas from Monte Carlo and dynamic programming.

TD(0) uses

<p align="center">
  <img src="../assets/equations/equation-30.svg" alt="V(S_t)\leftarrow V(S_t)+\alpha\underbrace{[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]}_{\text{TD error}}." width="760">
</p>

The target

<p align="center">
  <img src="../assets/equations/equation-31.svg" alt="R_{t+1}+\gamma V(S_{t+1})" width="760">
</p>

contains an existing estimate. This is called **bootstrapping**.

<p align="center">
  <img src="../assets/diagrams/td-learning.svg" alt="TD combines sampling and bootstrapping" width="760">
</p>

TD can learn before an episode ends and is one of the most important ideas in RL.

**[Try it in Notebook 02 → Monte Carlo versus TD](../notebooks/02_mc_vs_td.ipynb)**


### A numerical TD update

Suppose V(s)=5, the next reward is −1, V(s&#x27;)=8, γ=0.9, and α=0.1. The target is 6.2, the TD error is 1.2, and the updated value is 5.12.

**Analogy:** update your expected arrival time after reaching the next junction, using your current estimate for the remaining journey. You do not need to finish the trip first.

| Method | Uses sampled transitions? | Bootstraps? | Needs the full episode for its standard target? |
|---|---|---|---|
| DP | No; uses the model | Yes | No |
| Monte Carlo | Yes | No | Yes |
| TD(0) | Yes | Yes | No |

At a true terminal transition, the TD target is just the reward. A time-limit cutoff may still require bootstrapping if it merely interrupts a continuing task.


**Read the source:** [Sutton & Barto, Chapter 6](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 18](../chapters/18-temporal-difference-learning/README.md)

[← Topic 17](17-monte-carlo-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 19 →](19-sarsa.md)
