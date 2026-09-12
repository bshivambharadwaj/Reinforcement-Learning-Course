**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 28](28-advantage-actor-critic.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 30 →](30-trust-region-policy-optimization.md)

**Go deeper:** [Chapter 29: Generalized Advantage Estimation](../chapters/29-generalized-advantage-estimation/README.md)

---

<a id="topic-29"></a>

# 29. Generalized Advantage Estimation

> **Why this method exists:** One-step advantages depend strongly on the critic; long returns can be noisy. GAE blends multi-step TD residuals so lambda controls how far observed rewards carry credit before bootstrapping.

**[Try it in Notebook 06 → PPO](../notebooks/06_ppo.ipynb)**

Generalized Advantage Estimation (GAE) balances bias and variance by combining multi-step TD residuals:

<p align="center">
  <img src="../assets/equations/equation-54.svg" alt="\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}." width="760">
</p>

where

<p align="center">
  <img src="../assets/equations/equation-55.svg" alt="\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)." width="760">
</p>

λ controls the trade-off between shorter, lower-variance estimates and longer, typically lower-bias estimates.

GAE is especially important because it is commonly paired with PPO.


### Compute GAE backward through a rollout

For a rollout ending at index T−1, compute

<p align="center">
  <img src="../assets/equations/equation-56.svg" alt="\delta_t=R_{t+1}+\gamma(1-d_t)V(S_{t+1})-V(S_t)," width="760">
</p>

<p align="center">
  <img src="../assets/equations/equation-57.svg" alt="\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}." width="760">
</p>

Here d<sub>t</sub> marks true termination and c<sub>t</sub> is zero at an episode boundary or the end of the available rollout, and one otherwise. Initialize the unavailable next advantage to zero. A time-limit truncation can bootstrap from its final observation while still stopping the recursion across the reset.

**Analogy:** combine the coach's immediate feedback with feedback from later checkpoints. At λ=0, only one TD residual is used. At λ=1, a complete episode telescopes to Monte Carlo return minus the current value estimate. A truncated rollout retains its final bootstrap term.


**Knowledge check:** at lambda=1, does every truncated rollout become a complete Monte Carlo return? No. A finite rollout that stops before true termination retains its endpoint bootstrap contribution.


**Read the source:** [Schulman et al.: GAE](https://arxiv.org/abs/1506.02438).

---

**Continue in depth:** [Read Chapter 29](../chapters/29-generalized-advantage-estimation/README.md)

[← Topic 28](28-advantage-actor-critic.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 30 →](30-trust-region-policy-optimization.md)
