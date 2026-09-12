**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 15](15-policy-iteration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 17 →](17-monte-carlo-methods.md)

**Go deeper:** [Chapter 16: Value Iteration](../chapters/16-value-iteration/README.md)

---

<a id="topic-16"></a>

# 16. Value Iteration

> **Why this algorithm exists:** Fully evaluating every intermediate policy can be expensive. Value iteration interleaves local greedy improvement with value backups instead of waiting for a complete policy-evaluation solve.

Value iteration combines truncated evaluation and improvement into a single optimality update:

<p align="center">
  <img src="../assets/equations/equation-28.svg" alt="V_{k+1}(s)=\max_a\sum_{s&#x27;,r}p(s&#x27;,r|s,a)[r+\gamma V_k(s&#x27;)]." width="760">
</p>

Once values converge, derive the greedy policy.

Policy iteration performs more explicit evaluation between improvements; value iteration performs frequent improvement with shorter evaluation. Both are manifestations of **generalized policy iteration**.

**[Try it in Notebook 01 → MDPs and Dynamic Programming](../notebooks/01_mdp_dynamic_programming.ipynb)**


### Compare the two DP algorithms

| Method | Work before improving decisions | Stopping idea |
|---|---|---|
| Policy iteration | Evaluate the current policy, then improve it | Policy stops changing |
| Value iteration | Apply one optimality backup per state per sweep | Values change by less than a tolerance |

**Analogy:** policy iteration fully reviews a route plan before revising it; value iteration keeps revising short estimates as information spreads through the map. Value iteration does not require a separately stored policy during the value updates.

After stopping, extract a greedy policy using reward plus discounted next-state value. For finite discounted MDPs, the Bellman optimality operator contracts maximum value error by at most γ each application, explaining convergence. Stopping at a tolerance produces an approximation.


**Read the source:** [Sutton & Barto, Chapter 4](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 16](../chapters/16-value-iteration/README.md)

[← Topic 15](15-policy-iteration.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 17 →](17-monte-carlo-methods.md)
