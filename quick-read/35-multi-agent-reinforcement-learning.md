**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 34](34-model-based-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 36 →](36-reinforcement-learning-from-human-feedback.md)

**Go deeper:** [Chapter 35: Multi-Agent Reinforcement Learning](../chapters/35-multi-agent-reinforcement-learning/README.md)

---

<a id="topic-35"></a>

# 35. Multi-Agent Reinforcement Learning

> **Why this method exists:** When other decision-makers adapt, a single stationary-environment assumption can fail. Multi-agent methods address coordination, competition, and training information shared across learners.

Many environments contain multiple learning agents.

Agent i may have policy

<p align="center">
  <img src="../assets/equations/equation-66.svg" alt="\pi_i(a_i|o_i)" width="760">
</p>

while rewards can be cooperative, competitive, or mixed.

Challenges include:

- non-stationarity because other agents are learning too;
- credit assignment;
- communication;
- coordination;
- partial observability;
- equilibrium behavior.

Multi-agent RL provides useful conceptual tools for systems where multiple autonomous AI agents cooperate or compete, although practical LLM multi-agent systems are not automatically MARL systems unless learning through interaction is actually involved.


### When everyone else's behavior changes

Two delivery robots sharing a narrow corridor may cooperate to finish quickly, compete for priority, or receive a mix of individual and team rewards. From one robot's viewpoint, transitions depend on the other robot's actions. If the other robot is learning, that effective environment changes over training.

**Analogy:** learning to play doubles while your partner is also changing their strategy. Good individual actions may require coordination to become good team actions.

In **centralized training with decentralized execution**, training can use joint information while each deployed agent acts using its own observation. Shared rewards create a credit-assignment problem: team success alone does not reveal whose action helped. Multiple scripted or prompted agents exchanging messages are not by themselves an RL training algorithm.

---

**Continue in depth:** [Read Chapter 35](../chapters/35-multi-agent-reinforcement-learning/README.md)

[← Topic 34](34-model-based-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 36 →](36-reinforcement-learning-from-human-feedback.md)
