**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 23](23-experience-replay.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 25 →](25-policy-gradient-methods.md)

**Go deeper:** [Chapter 24: Target Networks](../chapters/24-target-networks/README.md)

---

<a id="topic-24"></a>

# 24. Target Networks

> **Why this method exists:** The bootstrap target depends on a value function that is also being updated. A target network slows that feedback loop. Holding the measuring stick still briefly helps, but cannot repair a wrong reward, terminal mask, or unstable optimization setting.

If the same rapidly changing network predicts both the current Q-value and its bootstrap target, the target itself moves during optimization.

DQN therefore maintains a target network θ<sup>−</sup>:

<p align="center">
  <img src="../assets/equations/equation-43.svg" alt="y=r+\gamma\max_{a&#x27;}Q(s&#x27;,a&#x27;;\theta^{-})." width="760">
</p>

The target parameters are periodically copied or slowly updated from the online network.

Together, **experience replay + target networks** address two major sources of instability in deep Q-learning.

**[Try it in Notebook 04 → DQN](../notebooks/04_dqn.ipynb)**


### Hold the measuring stick steady

For a hard update, periodically set θ<sup>−</sup>←θ. A soft update instead uses

<p align="center">
  <img src="../assets/equations/equation-44.svg" alt="\theta^{-}\leftarrow(1-\tau)\theta^{-}+\tau\theta," width="760">
</p>

where a small τ∈(0,1] makes the target change slowly.

**Analogy:** it is easier to aim at a target that pauses between movements. The target network is not a second independently optimized critic in standard DQN; it follows the online network.

A minimal DQN loop is: collect transitions, store them, sample a minibatch, construct masked detached targets, update the online network, and occasionally update the target network. These techniques improve stability but do not guarantee convergence with nonlinear approximation.


**Knowledge check:** should a true terminal target include the target network's next-state value? No. Its future contribution is zero. For an external time cutoff, bootstrapping may still be appropriate if the task continues and you retain the final observation.


**Read the source:** [Mnih et al.: DQN](https://doi.org/10.1038/nature14236).

---

**Continue in depth:** [Read Chapter 24](../chapters/24-target-networks/README.md)

[← Topic 23](23-experience-replay.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 25 →](25-policy-gradient-methods.md)
