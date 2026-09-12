**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 3](03-markov-decision-processes.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 5 →](05-policies.md)

**Go deeper:** [Chapter 4: States, Actions, Rewards and Transitions](../chapters/04-states-actions-rewards-and-transitions/README.md)

---

<a id="topic-4"></a>

# 4. States, Actions, Rewards and Transitions

These four objects define the decision problem.

### State
A representation of the information available for decision making.

### Action
A choice available to the agent. Actions can be discrete, continuous, structured, or even textual.

### Reward
A scalar feedback signal describing immediate desirability:

<p align="center">
  <img src="../assets/equations/equation-05.svg" alt="R_{t+1} \in \mathbb{R}." width="760">
</p>

Reward is **not the same thing as the objective behavior itself**. Poorly designed rewards can produce unintended strategies—often called reward hacking or specification gaming.

### Transition
The environment dynamics:

<p align="center">
  <img src="../assets/equations/equation-06.svg" alt="P(s&#x27;|s,a)." width="760">
</p>

Some environments are deterministic; many practical environments are stochastic.


### Specify the delivery task before choosing an algorithm

| Component | Example | Why the choice matters |
|---|---|---|
| State | Position, battery, parcel status | Missing information can make decisions ambiguous |
| Action | North, south, east, west, recharge | Determines what the agent can control |
| Reward | −1 per ordinary move, +10 on delivery | Defines the trade-offs being optimized |
| Transition | A move succeeds with probability 0.9 | Captures uncertainty in consequences |
| Termination | Delivery or battery exhaustion | Defines when future reward stops |

Here, delivery reward replaces the ordinary movement cost on the final step. State such conventions explicitly: small ambiguities change the optimization problem.

**Analogy:** rewarding a courier only for distance traveled could teach endless driving. Reward should reflect successful delivery and relevant costs. The environment can also forbid impossible actions; a penalty and a hard action constraint are different mechanisms.

---

**Continue in depth:** [Read Chapter 4](../chapters/04-states-actions-rewards-and-transitions/README.md)

[← Topic 3](03-markov-decision-processes.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 5 →](05-policies.md)
