**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 18](18-temporal-difference-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 20 →](20-q-learning-and-exploration.md)

**Go deeper:** [Chapter 19: SARSA](../chapters/19-sarsa/README.md)

---

<a id="topic-19"></a>

# 19. SARSA

> **Why this algorithm exists:** An exploratory courier actually takes occasional risky turns. SARSA evaluates that behavior, including the sampled next action, so its learned values reflect the cost of exploration.

**[Try it in Notebook 03 → SARSA and Q-Learning](../notebooks/03_q_learning.ipynb)**

SARSA is an **on-policy** TD control algorithm. Its name comes from the transition tuple:

<p align="center">
  <img src="../assets/equations/equation-32.svg" alt="S_t,A_t,R_{t+1},S_{t+1},A_{t+1}." width="760">
</p>

Update:

<p align="center">
  <img src="../assets/equations/equation-33.svg" alt="Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]." width="760">
</p>

Because the next action is sampled from the behavior policy, the learned action values reflect the policy actually being followed, including its exploration behavior.


### SARSA learns about the behavior you actually use

Suppose the next state has Q-values 8 and 2, but exploration selects the second action. With reward −1 and γ=0.9, SARSA's target is −1+0.9(2)=0.8.

**Analogy:** a courier accounts for their own occasional navigation mistakes when choosing how close to a dangerous road edge to travel. In a cliff-walking environment, an exploratory SARSA policy can favor a safer route because its values include the cost of exploratory moves.

Choose the next action before forming the update, then actually execute that same action on the next step. Resampling it afterward would break the intended transition sequence. If the episode truly terminates, omit the next-action value entirely.


**Read the source:** [Sutton & Barto, Chapter 6](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 19](../chapters/19-sarsa/README.md)

[← Topic 18](18-temporal-difference-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 20 →](20-q-learning-and-exploration.md)
