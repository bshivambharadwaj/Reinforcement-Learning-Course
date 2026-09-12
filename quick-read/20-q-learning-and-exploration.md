**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 19](19-sarsa.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 21 →](21-function-approximation.md)

**Go deeper:** [Chapter 20: Q-Learning and Exploration](../chapters/20-q-learning-and-exploration/README.md)

---

<a id="topic-20"></a>

# 20. Q-Learning and Exploration

> **Why this algorithm exists:** Sometimes we want to learn a greedy deployment policy while collecting data with an exploratory behavior policy. Q-learning separates those roles through a maximum over next-action values. This is not a guarantee that it is safer or better than SARSA during training.

Q-learning uses the target

<p align="center">
  <img src="../assets/equations/equation-34.svg" alt="R_{t+1}+\gamma\max_{a&#x27;}Q(S_{t+1},a&#x27;)." width="760">
</p>

and update

<p align="center">
  <img src="../assets/equations/equation-35.svg" alt="Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma\max_{a&#x27;}Q(S_{t+1},a&#x27;)-Q(S_t,A_t)\right]." width="760">
</p>

It is **off-policy**: the behavior policy can explore while the update targets a greedy policy.

A common exploration strategy is ε-greedy:

<p align="center">
  <img src="../assets/equations/equation-36.svg" alt="A_t=\begin{cases}
\text{random action} &amp; \text{with probability }\epsilon\\
\arg\max_aQ(S_t,a) &amp; \text{otherwise.}
\end{cases}" width="760">
</p>

This introduces the **exploration–exploitation trade-off**: use what we know, or gather information that may improve future decisions?

**[Try it in Notebook 03 → SARSA and Q-Learning](../notebooks/03_q_learning.ipynb)**


### Q-learning learns a greedy target while exploring

For the same next-state values 8 and 2, Q-learning's target is −1+0.9max (8,2)=6.2, even if behavior selects the second action next. This is the precise contrast with [SARSA's 0.8 target in Topic 19](19-sarsa.md).

**Analogy:** test unfamiliar restaurants to gather information while keeping a separate estimate of the best known dining choice. With n actions and uniform random exploration, the unique greedy action is selected with probability 1−ε+ε/n.

<p align="center">
  <img src="../assets/diagrams/q-learning.svg" alt="Explore with behavior; learn a greedy target" width="760">
</p>

The diagram's next-Q term is zero at termination. Tabular convergence requires sufficient state-action coverage and suitable decreasing step sizes; it is not guaranteed just by using the update equation. Exploration that decays too quickly can leave useful actions undiscovered.


> **Failure Mode — maximization bias:** even unbiased noisy action estimates can produce an optimistic maximum. The action selected as “best” is often the one with a favorable error. Double Q-learning separates action selection and evaluation across estimators; it addresses this mechanism without promising uniformly better returns. [Original Double Q-learning paper](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

**Knowledge check:** why can Q-learning have a good greedy policy and poor exploratory training returns? Its target and behavior policies differ. Compare both in [Notebook 08](../notebooks/08_one_problem_many_algorithms.ipynb).


**Read the source:** [Watkins & Dayan: Q-learning](https://doi.org/10.1007/BF00992698).

---

**Continue in depth:** [Read Chapter 20](../chapters/20-q-learning-and-exploration/README.md)

[← Topic 19](19-sarsa.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 21 →](21-function-approximation.md)
