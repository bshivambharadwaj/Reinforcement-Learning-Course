**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 1](01-what-is-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 3 →](03-markov-decision-processes.md)

**Go deeper:** [Chapter 2: The Agent–Environment Interaction](../chapters/02-the-agent-environment-interaction/README.md)

---

<a id="topic-2"></a>

# 2. The Agent–Environment Interaction

At time step t:

1. the agent observes state S<sub>t</sub>;
2. it selects action A<sub>t</sub>;
3. the environment transitions;
4. the agent receives reward R<sub>t+1</sub> and next state S<sub>t+1</sub>.

A trajectory can be written as

<p align="center">
  <img src="../assets/equations/equation-02.svg" alt="\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots)." width="760">
</p>

An **episode** is a trajectory that terminates. Chess and many games are episodic. Server resource management can instead be a continuing task.

The abstraction matters because RL algorithms operate on these interactions rather than on isolated input-label pairs.


### Follow one interaction

At a junction, the robot observes its location and battery level, chooses “go east,” spends one unit of energy, and reaches another junction. That gives one transition (s,a,r,s&#x27;). An episode might run from parcel pickup until delivery or failure.

<p align="center">
  <img src="../assets/diagrams/interaction-loop.svg" alt="From one step to the next episode" width="760">
</p>

A **time step** is one interaction; an **episode** contains many steps; a **training run** usually contains many episodes. During evaluation, we normally freeze the learned parameters to measure the behavior we have learned.

---

**Continue in depth:** [Read Chapter 2](../chapters/02-the-agent-environment-interaction/README.md)

[← Topic 1](01-what-is-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 3 →](03-markov-decision-processes.md)
