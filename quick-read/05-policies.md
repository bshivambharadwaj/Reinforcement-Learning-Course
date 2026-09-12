**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 4](04-states-actions-rewards-and-transitions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 6 →](06-returns-and-discounting.md)

**Go deeper:** [Chapter 5: Policies](../chapters/05-policies/README.md)

---

<a id="topic-5"></a>

# 5. Policies

A policy describes the agent's behavior.

A stochastic policy is

<p align="center">
  <img src="../assets/equations/equation-07.svg" alt="\pi(a|s)=P(A_t=a|S_t=s)." width="760">
</p>

A deterministic policy maps states directly to actions:

<p align="center">
  <img src="../assets/equations/equation-08.svg" alt="a=\mu(s)." width="760">
</p>

RL tries to find a policy that performs well according to the chosen objective. Importantly, the policy is not necessarily a lookup table. In deep RL it is often a neural network parameterized by θ:

<p align="center">
  <img src="../assets/equations/equation-09.svg" alt="\pi_\theta(a|s)." width="760">
</p>

For language models, token generation itself can be interpreted as a stochastic policy over the vocabulary conditioned on context.


### A policy is a decision rule, not a route

A fixed route says “east, east, north.” A policy says “at this junction, with this battery level, choose east with probability 0.8 and north with probability 0.2.” It can react when an unexpected event changes the state.

For discrete actions, probabilities satisfy ∑<sub>a</sub>π(a&#124;s)=1 and π(a&#124;s)≥0. A neural policy commonly outputs logits, then uses softmax to obtain these probabilities. For continuous controls, it might output a Gaussian distribution over steering angles instead.

**Behavior policy** means the policy collecting experience. **Target policy** means the policy being evaluated or improved. Keeping these roles separate will explain the difference between SARSA and Q-learning.

---

**Continue in depth:** [Read Chapter 5](../chapters/05-policies/README.md)

[← Topic 4](04-states-actions-rewards-and-transitions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 6 →](06-returns-and-discounting.md)
