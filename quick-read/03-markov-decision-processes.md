**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 2](02-the-agent-environment-interaction.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 4 →](04-states-actions-rewards-and-transitions.md)

**Go deeper:** [Chapter 3: Markov Decision Processes](../chapters/03-markov-decision-processes/README.md)

---

<a id="topic-3"></a>

# 3. Markov Decision Processes

**[Try it in Notebook 01 → MDPs and Dynamic Programming](../notebooks/01_mdp_dynamic_programming.ipynb)**

A Markov Decision Process (MDP) provides the standard mathematical model for RL:

<p align="center">
  <img src="../assets/equations/equation-03.svg" alt="\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)." width="760">
</p>

Where:

- 𝒮 — state space;
- 𝒜 — action space;
- P(s&#x27;&#124;s,a) — transition dynamics;
- R — reward specification;
- γ∈[0,1] — discount factor.

The **Markov property** says that the current state contains the information needed to predict the future given the action:

<p align="center">
  <img src="../assets/equations/equation-04.svg" alt="P(S_{t+1}|S_t,A_t,S_{t-1},\ldots)=P(S_{t+1}|S_t,A_t)." width="760">
</p>

This does not mean the real world has no history. It means that a well-designed state representation should summarize the relevant history.

For an AI agent, a state might include conversation context, tool results, memory, and task progress rather than a single physical observation.


### What makes a state sufficient?

Location alone may be insufficient: the same route can be feasible with a full battery and impossible with an empty one. Including battery charge makes the state more informative. If traffic depends on time of day, the clock may matter too.

An **observation** is what the agent can sense; a **state** is a sufficient description for predicting transitions and rewards. A camera image may hide a vehicle behind a wall. This leads to a **partially observable MDP (POMDP)**, where an agent can use observation history, memory, or a belief distribution over possible states.

For finite discounted problems we normally assume 0≤γ&lt;1 and bounded rewards. Undiscounted episodic problems can use γ=1 when termination and integrability make returns well-defined. An initial-state distribution ρ<sub>0</sub> specifies where episodes begin; together with the policy it determines the expected course objective.

---

**Continue in depth:** [Read Chapter 3](../chapters/03-markov-decision-processes/README.md)

[← Topic 2](02-the-agent-environment-interaction.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 4 →](04-states-actions-rewards-and-transitions.md)
