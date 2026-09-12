**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 29](29-generalized-advantage-estimation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 31 →](31-proximal-policy-optimization.md)

**Go deeper:** [Chapter 30: Trust Region Policy Optimization](../chapters/30-trust-region-policy-optimization/README.md)

---

<a id="topic-30"></a>

# 30. Trust Region Policy Optimization

> **Why this algorithm exists:** A promising sampled gradient can move action probabilities far enough to invalidate the local improvement estimate. TRPO constrains distributional change; its practical approximations are not an unconditional performance guarantee.

Large policy updates can destroy useful behavior. Trust Region Policy Optimization (TRPO) formalizes the idea that policy improvement should occur within a constrained region.

Conceptually:

<p align="center">
  <img src="../assets/equations/equation-58.svg" alt="\max_\theta \; \text{surrogate improvement}" width="760">
</p>

subject to a constraint on the divergence between old and new policies, commonly expressed using KL divergence.

TRPO is important less because it is always the default implementation today and more because it establishes the principle of **controlled policy updates** that motivates PPO.


### Measure policy change in probability space

A small change in neural-network weights can still cause a large change in action probabilities. TRPO therefore measures change between action distributions, using an average KL-divergence constraint under states visited by the old policy:

<p align="center">
  <img src="../assets/equations/equation-59.svg" alt="\mathbb{E}_{s\sim\pi_{old}}
[D_{KL}(\pi_{old}(\cdot|s)\Vert\pi_\theta(\cdot|s))]\leq\delta." width="760">
</p>

**Analogy:** improve a courier's routing policy while limiting how much its actual decisions change at familiar junctions. This is more behaviorally meaningful than only limiting the size of parameter changes.

Practical TRPO uses approximations to solve the constrained update, including curvature information and a line search. The theoretical motivation should not be read as a blanket guarantee of improvement in every sampled implementation.


**Read the source:** [Schulman et al.: TRPO](https://arxiv.org/abs/1502.05477).

---

**Continue in depth:** [Read Chapter 30](../chapters/30-trust-region-policy-optimization/README.md)

[← Topic 29](29-generalized-advantage-estimation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 31 →](31-proximal-policy-optimization.md)
