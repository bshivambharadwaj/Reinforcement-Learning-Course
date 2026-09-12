**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 6](06-returns-and-discounting.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 8 →](08-action-value-functions.md)

**Go deeper:** [Chapter 7: State-Value Functions](../chapters/07-state-value-functions/README.md)

---

<a id="topic-7"></a>

# 7. State-Value Functions

The state-value function answers:

> How good is it to be in state s while following policy π?

<p align="center">
  <img src="../assets/equations/equation-13.svg" alt="V^{\pi}(s)=\mathbb{E}_{\pi}[G_t|S_t=s]." width="760">
</p>

Values compress potentially long future trajectories into a single expectation. They are therefore powerful tools for evaluating decisions without explicitly enumerating every future outcome at decision time.


### Value averages over possible futures

Suppose following the current policy from a junction produces return 10 half the time and 2 half the time. Then V<sup>π</sup>(s)=6. A value estimate of 6 does not promise that any single delivery will earn exactly 6.

**Analogy:** a city's average travel time describes what to expect, not the duration of every trip. Likewise, state value depends on both the environment and the policy: a skilled robot and an inexperienced robot can have different values for the same junction.

We conventionally set the value of a terminal state to zero because no future rewards remain after entry. The reward for reaching it belongs to the transition into that state.

---

**Continue in depth:** [Read Chapter 7](../chapters/07-state-value-functions/README.md)

[← Topic 6](06-returns-and-discounting.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 8 →](08-action-value-functions.md)
