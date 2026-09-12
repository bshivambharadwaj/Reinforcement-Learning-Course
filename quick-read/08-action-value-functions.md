**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 7](07-state-value-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 9 →](09-advantage-functions.md)

**Go deeper:** [Chapter 8: Action-Value Functions](../chapters/08-action-value-functions/README.md)

---

<a id="topic-8"></a>

# 8. Action-Value Functions

The action-value function answers:

> How good is taking action a in state s, then following π?

<p align="center">
  <img src="../assets/equations/equation-14.svg" alt="Q^{\pi}(s,a)=\mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]." width="760">
</p>

If Q is known, action selection can be straightforward:

<p align="center">
  <img src="../assets/equations/equation-15.svg" alt="\pi(s)=\arg\max_a Q(s,a)" width="760">
</p>

for a greedy deterministic policy.

Q-learning and DQN build directly on this idea.


### Compare options at one junction

Suppose Q<sup>π</sup>(s,east)=8 and Q<sup>π</sup>(s,north)=3. These values include the first chosen action and all subsequent behavior under π.

The connection to state value is

<p align="center">
  <img src="../assets/equations/equation-16.svg" alt="V^{\pi}(s)=\sum_a\pi(a|s)Q^{\pi}(s,a)." width="760">
</p>

If the policy chooses each action equally often, V<sup>π</sup>(s)=5.5. A greedy improvement would choose east. Greedy behavior with respect to an inaccurate estimate is not necessarily optimal, and greedifying Q<sup>π</sup> once need not produce the globally optimal policy.

**Analogy:** V rates your overall prospects at a junction; Q rates the individual roads available there.

---

**Continue in depth:** [Read Chapter 8](../chapters/08-action-value-functions/README.md)

[← Topic 7](07-state-value-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 9 →](09-advantage-functions.md)
