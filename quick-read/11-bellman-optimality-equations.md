**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 10](10-bellman-expectation-equations.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 12 →](12-dynamic-programming.md)

**Go deeper:** [Chapter 11: Bellman Optimality Equations](../chapters/11-bellman-optimality-equations/README.md)

---

<a id="topic-11"></a>

# 11. Bellman Optimality Equations

For the optimal value function:

<p align="center">
  <img src="../assets/equations/equation-23.svg" alt="V^{*}(s)=\max_a \mathbb{E}\left[R_{t+1}+\gamma V^{*}(S_{t+1})|S_t=s,A_t=a\right]." width="760">
</p>

For action values:

<p align="center">
  <img src="../assets/equations/equation-24.svg" alt="Q^{*}(s,a)=\mathbb{E}\left[R_{t+1}+\gamma\max_{a&#x27;}Q^{*}(S_{t+1},a&#x27;)\right]." width="760">
</p>

Once Q<sup>∗</sup> is known, an optimal action can be selected greedily:

<p align="center">
  <img src="../assets/equations/equation-25.svg" alt="a^{*}=\arg\max_a Q^{*}(s,a)." width="760">
</p>

The difference from the expectation equation is crucial: evaluation asks what happens **under a policy**; optimality asks what happens when future choices are optimal.


### Average the uncertainty; maximize the choice

The agent controls its action but cannot choose which random outcome occurs. That is why the maximum goes over actions while an expectation remains over next states and rewards.

<p align="center">
  <img src="../assets/diagrams/bellman-optimality.svg" alt="Average outcomes, then choose an action" width="760">
</p>

**Analogy:** choose the route with the best expected travel result, not the route whose luckiest possible traffic conditions are best. For finite discounted MDPs, Bellman optimality has a unique value-function solution, although several actions can tie for optimality.

---

**Continue in depth:** [Read Chapter 11](../chapters/11-bellman-optimality-equations/README.md)

[← Topic 10](10-bellman-expectation-equations.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 12 →](12-dynamic-programming.md)
