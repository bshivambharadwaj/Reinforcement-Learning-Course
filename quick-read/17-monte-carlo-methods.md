**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 16](16-value-iteration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 18 →](18-temporal-difference-learning.md)

**Go deeper:** [Chapter 17: Monte Carlo Methods](../chapters/17-monte-carlo-methods/README.md)

---

<a id="topic-17"></a>

# 17. Monte Carlo Methods

> **Why this algorithm exists:** When you cannot enumerate dynamics but can finish episodes, observed returns provide learning targets. Monte Carlo removes the need for a transition model or bootstrap estimate, at the cost of waiting for outcomes and coping with return variance.

**[Try it in Notebook 02 → Monte Carlo versus TD](../notebooks/02_mc_vs_td.ipynb)**

What if the transition model is unknown?

Monte Carlo (MC) methods learn from complete sampled episodes. After observing a return G<sub>t</sub>, a value estimate can be updated toward it:

<p align="center">
  <img src="../assets/equations/equation-29.svg" alt="V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]." width="760">
</p>

MC methods:

- do not require a model;
- learn from experience;
- use actual sampled returns;
- typically wait until enough future rewards are observed, often the end of an episode.

Their estimates can have high variance because the full sampled return is noisy.


### Learn from completed trips

If visits to a junction have produced returns 4, 8, and 6, their sample mean is 6. Using α=1/N(s) recovers an incremental sample mean, where N(s) counts included returns for that state. A constant learning rate instead keeps adapting to recent data.

**First-visit MC** uses the first occurrence of a state in each episode; **every-visit MC** uses every occurrence. Repeated visits within one episode are correlated, so they should not be treated as independent evidence.

**Analogy:** judge a route only after completing the delivery. You observe the whole outcome, but traffic and later choices can make that outcome noisy. For a fixed policy and suitable sampling, MC targets avoid bootstrap bias, although finite-sample estimates remain uncertain.


**Read the source:** [Sutton & Barto, Chapter 5](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 17](../chapters/17-monte-carlo-methods/README.md)

[← Topic 16](16-value-iteration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 18 →](18-temporal-difference-learning.md)
