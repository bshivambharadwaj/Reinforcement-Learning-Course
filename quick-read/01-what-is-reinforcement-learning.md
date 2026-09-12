**Quick read** · **Created by Shivam Bharadwaj**

[← Quick reads](README.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 2 →](02-the-agent-environment-interaction.md)

**Go deeper:** [Chapter 1: What Is Reinforcement Learning?](../chapters/01-what-is-reinforcement-learning/README.md)

---

<a id="topic-1"></a>

# 1. What Is Reinforcement Learning?

Reinforcement Learning (RL) studies how an **agent learns to make decisions through interaction** with an environment.

Unlike supervised learning, the agent is not normally given the correct action for every situation. It acts, observes consequences, receives a reward signal, and gradually learns behavior that maximizes long-term return.

The basic interaction is:

<p align="center">
  <img src="../assets/diagrams/agent-environment.svg" alt="Act, observe, and learn" width="760">
</p>

The central challenge is that an action can affect not only the immediate reward but also the states—and therefore rewards—the agent encounters later.

A common objective is

<p align="center">
  <img src="../assets/equations/equation-01.svg" alt="J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^{t} R_{t+1}\right]." width="760">
</p>

RL is therefore fundamentally about **sequential decision making under uncertainty**.

**Modern connection:** language-model agents choosing tools, robots choosing movements, recommendation systems choosing content, and reasoning models choosing intermediate steps can all be viewed through this lens when an appropriate state, action, and feedback process can be defined.


### Intuition: learning to deliver a parcel

Imagine a delivery robot that must reach a house. Nobody labels the best move at every junction. The robot tries routes, pays a small cost for travel, and earns a reward when it delivers the parcel. A shortcut may save time but risk a costly collision. Learning means using experience to choose routes with better **total consequences**.

| Learning setting | Feedback | Delivery analogy |
|---|---|---|
| Supervised learning | Correct target for each training example | Copy routes labeled by an expert |
| Unsupervised learning | Structure in unlabeled data | Group streets by their visual appearance |
| Reinforcement learning | Rewards resulting from actions | Try routes and learn which deliveries work well |

Two difficulties make RL distinctive: **delayed credit assignment** (which earlier turn caused success?) and **action-dependent data** (the chosen route determines what the robot sees next). A reward is feedback, not an instruction identifying the correct action.

---

**Continue in depth:** [Read Chapter 1](../chapters/01-what-is-reinforcement-learning/README.md)

[← Quick reads](README.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 2 →](02-the-agent-environment-interaction.md)
