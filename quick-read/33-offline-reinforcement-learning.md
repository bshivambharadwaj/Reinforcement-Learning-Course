**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 32](32-entropy-regularization.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 34 →](34-model-based-reinforcement-learning.md)

**Go deeper:** [Chapter 33: Offline Reinforcement Learning](../chapters/33-offline-reinforcement-learning/README.md)

---

<a id="topic-33"></a>

# 33. Offline Reinforcement Learning

> **Why this method exists:** Some tasks supply only logged interactions. Offline RL tries to improve behavior without collecting corrections, making unsupported actions and distribution shift central concerns.

Traditional RL assumes the agent can interact with the environment. Offline RL instead learns from a fixed dataset:

<p align="center">
  <img src="../assets/equations/equation-64.svg" alt="\mathcal{D}=\{(s,a,r,s&#x27;)\}." width="760">
</p>

The major challenge is **distribution shift**. The learned policy may choose actions poorly represented in the dataset, where value estimates can be unreliable.

Offline RL is especially relevant when interaction is expensive, dangerous, slow, or ethically constrained—for example robotics, healthcare research, and expensive real-world systems.


### Why a fixed dataset changes the problem

Imagine learning routes only from another courier's logs. If those logs never include a narrow alley, the learner has no direct evidence about it. A maximization over estimated Q-values might nevertheless favor that alley because of an optimistic approximation error. There is no new interaction to correct the mistake during offline training.

Behavior cloning simply imitates logged actions. Offline RL uses reward and temporal structure to seek better decisions, while needing protection against unsupported actions. Conservative value estimates and policies kept close to the data are two broad approaches.

**Off-policy is not the same as offline:** off-policy Q-learning can continue collecting new experience; offline RL holds the training dataset fixed. Evaluating a new policy from logs also requires coverage assumptions and careful uncertainty assessment.


> **Failure Mode — distribution shift:** a large predicted value for an action absent from the logs is unsupported optimism, not evidence. Report coverage limitations and evaluate outside the training dataset only with an appropriate evaluation design. [Offline RL tutorial](https://arxiv.org/abs/2005.01643).

---

**Continue in depth:** [Read Chapter 33](../chapters/33-offline-reinforcement-learning/README.md)

[← Topic 32](32-entropy-regularization.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 34 →](34-model-based-reinforcement-learning.md)
