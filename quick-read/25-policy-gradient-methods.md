**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 24](24-target-networks.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 26 →](26-reinforce.md)

**Go deeper:** [Chapter 25: Policy Gradient Methods](../chapters/25-policy-gradient-methods/README.md)

---

<a id="topic-25"></a>

# 25. Policy Gradient Methods

> **Why this algorithm exists:** A greedy maximum over Q is awkward for some action spaces and does not directly represent a stochastic decision rule. Policy gradients optimize the policy distribution itself; the challenge shifts to noisy credit assignment.

**[Try it in Notebook 05 → Policy Gradients](../notebooks/05_policy_gradient.ipynb)**

Instead of learning values and deriving a policy, policy-gradient methods optimize policy parameters directly:

<p align="center">
  <img src="../assets/equations/equation-45.svg" alt="\pi_\theta(a|s)." width="760">
</p>

The objective is

<p align="center">
  <img src="../assets/equations/equation-46.svg" alt="J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]." width="760">
</p>

The policy-gradient theorem gives a gradient of the form

<p align="center">
  <img src="../assets/equations/equation-47.svg" alt="\nabla_\theta J(\theta)\propto
\mathbb{E}\left[\nabla_\theta\log\pi_\theta(A_t|S_t)Q^{\pi}(S_t,A_t)\right]." width="760">
</p>

This formulation naturally supports stochastic policies and large or continuous action spaces.


### Why the log probability appears

The likelihood-ratio identity ∇<sub>θ</sub>p<sub>θ</sub>=p<sub>θ</sub>∇<sub>θ</sub>log p<sub>θ</sub> lets us estimate how changing action probabilities changes expected return. We need differentiable policy probabilities, but we do not need to differentiate through the environment's transition dynamics.

**Analogy:** increase the chance of choosing a route when evidence says it produces better outcomes. The update changes the probability distribution over actions, not the action that already occurred.

The expectation in a policy-gradient formula must use the appropriate policy-induced state distribution. For the discounted episodic objective defined earlier, an explicit trajectory estimator includes γ<sup>t</sup>G<sub>t</sub> at time t; equivalent theorem statements may absorb discount weights into a discounted visitation distribution. This convention matters when translating notation into code.


**Read the source:** [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 25](../chapters/25-policy-gradient-methods/README.md)

[← Topic 24](24-target-networks.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 26 →](26-reinforce.md)
