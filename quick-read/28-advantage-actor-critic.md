**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 27](27-actor-critic-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 29 →](29-generalized-advantage-estimation.md)

**Go deeper:** [Chapter 28: Advantage Actor–Critic](../chapters/28-advantage-actor-critic/README.md)

---

<a id="topic-28"></a>

# 28. Advantage Actor–Critic

> **Why this algorithm exists:** Raw success can reflect an easy state rather than a good action. An advantage signal compares an action with the state baseline, focusing the actor on whether it did better than expected.

Rather than weighting policy updates by raw return, advantage actor–critic methods estimate

<p align="center">
  <img src="../assets/equations/equation-52.svg" alt="A(s,a)=Q(s,a)-V(s)." width="760">
</p>

A one-step estimate is the TD error:

<p align="center">
  <img src="../assets/equations/equation-53.svg" alt="\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)." width="760">
</p>

The actor learns whether the selected action performed better than expected, while the critic learns the baseline expectation.

This separation improves learning efficiency and reduces policy-gradient variance.


### Relative feedback at every step

Suppose the critic predicts value 5, but reward plus discounted next value is 6.2. The estimated advantage is +1.2, so the sampled action receives a positive policy-gradient weight. The critic also moves its prediction toward 6.2.

With the exact V<sup>π</sup>, the expected one-step TD error conditioned on (s,a) equals A<sup>π</sup>(s,a). With an approximate critic, it can be biased. A multi-step target uses more observed rewards before bootstrapping and offers another trade-off.

**A2C** usually refers to a synchronous advantage actor–critic implementation that collects batches from several environments before updating. **A3C** uses asynchronous workers. The broader actor–critic idea does not require either execution pattern.


**Read the source:** [Mnih et al.: asynchronous actor-critic](https://arxiv.org/abs/1602.01783).

---

**Continue in depth:** [Read Chapter 28](../chapters/28-advantage-actor-critic/README.md)

[← Topic 27](27-actor-critic-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 29 →](29-generalized-advantage-estimation.md)
