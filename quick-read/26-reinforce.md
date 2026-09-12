**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 25](25-policy-gradient-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 27 →](27-actor-critic-methods.md)

**Go deeper:** [Chapter 26: REINFORCE](../chapters/26-reinforce/README.md)

---

<a id="topic-26"></a>

# 26. REINFORCE

> **Why this algorithm exists:** REINFORCE provides a direct Monte Carlo policy-gradient estimator without a learned critic. Its simplicity exposes the score-function idea, but complete-return weights can be noisy.

REINFORCE is a Monte Carlo policy-gradient algorithm:

<p align="center">
  <img src="../assets/equations/equation-48.svg" alt="\theta\leftarrow\theta+\alpha\gamma^{t} G_t\nabla_\theta\log\pi_\theta(A_t|S_t)." width="760">
</p>

Intuitively:

- positive return weights reinforce sampled actions;
- negative return weights discourage sampled actions;
- a baseline makes the weights relative to expected performance.

Its weakness is high variance. A baseline can reduce variance:

<p align="center">
  <img src="../assets/equations/equation-49.svg" alt="G_t-b(S_t)." width="760">
</p>

Choosing b(S<sub>t</sub>)=V(S<sub>t</sub>) leads naturally toward advantage-based actor–critic methods.

**[Try it in Notebook 05 → Policy Gradients](../notebooks/05_policy_gradient.ipynb)**


### A complete REINFORCE update

For a finite episode and the discounted start-state objective, one estimator is

<p align="center">
  <img src="../assets/equations/equation-50.svg" alt="\hat g=\sum_{t=0}^{T-1}\gamma^{t}
\nabla_\theta\log\pi_\theta(A_t|S_t)[G_t-b(S_t)]." width="760">
</p>

Collect an episode, compute its returns backward, evaluate log probabilities, and ascend along ĝ. When using a loss-minimizing optimizer, minimize the negative of the corresponding weighted log-probability sum. Treat the return and baseline weights as fixed in the actor gradient.

**Analogy:** after a delivery, reinforce decisions according to how much better or worse the outcome was than expected. With a baseline, a positive residual reinforces a sampled action and a negative residual discourages it. Without a baseline, a low but positive return still contributes a positive weight; “poor return” alone does not imply a negative update.

A state-only baseline preserves the expected gradient because ∑<sub>a</sub>π(a&#124;s)∇log π(a&#124;s)=0. It can reduce variance without supplying a new reward objective.


> **Common Misconception:** any number subtracted from a return is a safe baseline. The baseline must not depend on the sampled action in a way that changes the expected score-function estimator. A learned state baseline also needs careful gradient separation; see the lagged baseline in Notebook 05.


**Read the source:** [Williams: REINFORCE](https://doi.org/10.1007/BF00992696).

---

**Continue in depth:** [Read Chapter 26](../chapters/26-reinforce/README.md)

[← Topic 25](25-policy-gradient-methods.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 27 →](27-actor-critic-methods.md)
