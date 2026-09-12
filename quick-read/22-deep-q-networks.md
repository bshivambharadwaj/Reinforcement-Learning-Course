**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 21](21-function-approximation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 23 →](23-experience-replay.md)

**Go deeper:** [Chapter 22: Deep Q-Networks](../chapters/22-deep-q-networks/README.md)

---

<a id="topic-22"></a>

# 22. Deep Q-Networks

> **Why this algorithm exists:** Images and large state spaces make a separate Q entry for every observation impractical. DQN replaces the table with a neural estimator; replay and a delayed target help train it from correlated, changing data.

**[Try it in Notebook 04 → DQN](../notebooks/04_dqn.ipynb)**

A Deep Q-Network (DQN) approximates

<p align="center">
  <img src="../assets/equations/equation-38.svg" alt="Q(s,a;\theta)" width="760">
</p>

with a neural network.

A typical squared TD objective is

<p align="center">
  <img src="../assets/equations/equation-39.svg" alt="L(\theta)=\mathbb{E}\left[(y-Q(s,a;\theta))^{2}\right]" width="760">
</p>

with target

<p align="center">
  <img src="../assets/equations/equation-40.svg" alt="y=r+\gamma\max_{a&#x27;}Q(s&#x27;,a&#x27;;\theta^{-})." width="760">
</p>

DQN demonstrated that value-based RL could learn useful policies directly from high-dimensional observations when combined with stabilization techniques.


### From the Q-table to a network

For a small discrete action set, the network takes a state and outputs one Q-value per action. The selected action's prediction is trained toward a detached target:

<p align="center">
  <img src="../assets/equations/equation-41.svg" alt="y=r+\gamma(1-d)\max_{a&#x27;}Q(s&#x27;,a&#x27;;\theta^{-})," width="760">
</p>

where d=1 means true termination. Detaching means that the target is treated as constant when differentiating the loss. A Huber loss is also commonly used to reduce sensitivity to large TD errors.

**Analogy:** the network is a shared route-rating system; replay supplies past journeys and a target network supplies a more stable estimate of what comes next. Standard DQN fits discrete actions naturally because it can enumerate them; maximizing over arbitrary continuous actions requires additional machinery.


**Read the source:** [Mnih et al.: DQN](https://doi.org/10.1038/nature14236).

---

**Continue in depth:** [Read Chapter 22](../chapters/22-deep-q-networks/README.md)

[← Topic 21](21-function-approximation.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 23 →](23-experience-replay.md)
