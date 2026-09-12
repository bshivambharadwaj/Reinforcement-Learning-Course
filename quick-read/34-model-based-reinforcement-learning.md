**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 33](33-offline-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 35 →](35-multi-agent-reinforcement-learning.md)

**Go deeper:** [Chapter 34: Model-Based Reinforcement Learning](../chapters/34-model-based-reinforcement-learning/README.md)

---

<a id="topic-34"></a>

# 34. Model-Based Reinforcement Learning

> **Why this method exists:** Real interaction can cost far more than computation. A dynamics model lets an agent rehearse or plan, while requiring checks against errors that a planner may exploit.

Model-free RL learns values or policies without explicitly learning environment dynamics.

Model-based RL learns or uses a model such as

<p align="center">
  <img src="../assets/equations/equation-65.svg" alt="\hat P(s&#x27;|s,a),\qquad \hat R(s,a)" width="760">
</p>

and then plans through predicted futures.

<p align="center">
  <img src="../assets/diagrams/world-model.svg" alt="Learn in the world; plan with a model" width="760">
</p>

The attraction is data efficiency and planning capability. The danger is **model error**: planning can exploit inaccuracies in the learned model.


### Planning with a learned simulator

**Analogy:** practice deliveries inside a simulator before spending battery on real roads. A Dyna-style learner combines updates from real transitions with updates from simulated transitions. Model predictive control instead plans a short action sequence, executes its first action, observes the real outcome, and replans.

<p align="center">
  <img src="../assets/diagrams/model-based-loop.svg" alt="Close the loop with real feedback" width="760">
</p>

Errors compound over long imagined trajectories. A planner may discover actions that look excellent in the model but fail in reality. Shorter planning horizons, frequent replanning, and model uncertainty estimates can help. A world model can predict latent representations rather than raw pixels; model-based RL is defined by using predicted dynamics for planning or learning.


**Read the source:** [Sutton & Barto, Chapter 8](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 34](../chapters/34-model-based-reinforcement-learning/README.md)

[← Topic 33](33-offline-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 35 →](35-multi-agent-reinforcement-learning.md)
