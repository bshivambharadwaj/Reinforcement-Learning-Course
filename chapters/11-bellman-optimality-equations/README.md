# 11. Bellman Optimality Equations

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 10](../10-bellman-expectation-equations/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 12 →](../12-dynamic-programming/README.md)

[Quick-read Topic 11](../../README.md#topic-11) · [Notation](../NOTATION.md)

- [11.1 Optimality changes the action operator](#section-11-1)
- [11.2 Define optimal values and the operator](#section-11-2)
- [11.3 Why max and expectation cannot be casually swapped](#section-11-3)
- [11.4 Work through a one-step optimal backup](#section-11-4)
- [11.5 Prove the optimal operator is a contraction](#section-11-5)
- [11.6 Greedy policies and approximate-value guarantees](#section-11-6)
- [11.7 Ties, finite horizons, and model error](#section-11-7)
- [11.8 Lab: separate numerical convergence from policy quality](#section-11-8)
- [11.9 From optimal backups to modern search](#section-11-9)
- [11.10 Problems, worked answers, and sources](#section-11-10)

---

<p align="center">
  <img src="../../assets/diagrams/bellman-optimality.svg" width="760" alt="Optimality replaces a fixed action rule with a best-action comparison." />
</p>

*Optimality replaces a fixed action rule with a best-action comparison.*

<a id="section-11-1"></a>

## 11.1 Optimality changes the action operator

Bellman expectation evaluates a specified policy. Bellman optimality asks which available action yields the best immediate-plus-future outcome. Environment uncertainty is still averaged; action choice is optimized. Confusing these two operations changes the problem.

**Prerequisites:** Chapters 8 and 10. **Targets:** define the optimal Bellman operator, prove discounted contraction, explain why maximizing inside environmental uncertainty can grant impossible information, and relate approximate values to greedy policy quality.

The courier can choose a road before traffic is realized. It cannot normally choose a different road separately for every hidden traffic outcome after seeing that outcome. Correct operator order respects when information becomes available.

<a id="section-11-2"></a>

## 11.2 Define optimal values and the operator

For a finite discounted MDP, V_star(s) is the maximum achievable expected return from s. The optimal Bellman operator is:

```text
(T_star v)(s) = max_a sum_(s',r) p(s',r|s,a) [r + gamma v(s')]
V_star = T_star V_star
```

The corresponding action-value equation is Q_star(s,a)=E[r+gamma max_b Q_star(s',b)|s,a], with zero continuation at true termination. The maximum over b belongs to the next decision, after s' is observed.

Existence of a deterministic stationary optimum holds for the finite discounted fully observed expected-return setting. Other settings require their own assumptions and policy classes. Finite-horizon optimal policies can be represented as stationary only after including time in the state.

<a id="section-11-3"></a>

## 11.3 Why max and expectation cannot be casually swapped

Suppose two actions have outcomes determined by an unseen fair coin. Action left pays 10 on heads and 0 on tails; right pays 0 on heads and 10 on tails. Each action has expected reward 5, so max_a E[R|a]=5.

If one incorrectly computes E[max_a R(a,coin)], the answer is 10. That calculation grants the agent access to the coin before it chooses. It solves a different information structure.

The general inequality E[max X_a]≥max E[X_a] reflects this advantage of outcome-dependent selection. Equality can occur, but it is not guaranteed. In an MDP the action maximum must sit at the decision point where its conditioning information is available.

<a id="section-11-4"></a>

## 11.4 Work through a one-step optimal backup

At state A, safe ends with reward 3. Risky costs −1 and reaches B with probability 0.8 or terminal failure with probability 0.2. If candidate v(B)=6 and gamma=0.9, the risky backup is −1+0.9×0.8×6=3.32. The optimal backup selects risky and assigns 3.32 to A.

If the correct V_star(B) were only 4, the risky value would be 1.88 and safe would be optimal. Greedy action choice depends on continuation estimates, not only the immediate reward or action name.

A backup from a guess v does not certify that its greedy action is globally optimal. Iteration or another solution method must resolve the continuation estimates under the intended model.

<a id="section-11-5"></a>

## 11.5 Prove the optimal operator is a contraction

For any finite action vectors x and y, |max_a x_a−max_a y_a|≤max_a |x_a−y_a|. Apply this inequality to the two sets of action backups from v and w. Their reward terms cancel, and each expected continuation difference is at most gamma||v−w||_infinity.

Therefore ||T_star v−T_star w||_infinity≤gamma||v−w||_infinity. With gamma<1, the fixed point is unique and synchronous value iteration converges from any finite vector.

The maximum is nonlinear but nonexpansive in this norm. The proof does not require differentiability of argmax. It does require a well-defined finite discounted MDP and exact backups, which differ from neural fitted updates on a changing dataset.

<a id="section-11-6"></a>

## 11.6 Greedy policies and approximate-value guarantees

Let pi_v be greedy with respect to one-step backups using v, and assume ||v−V_star||_infinity≤epsilon. A standard discounted bound is ||V_star−V_pi_v||_infinity≤2gamma epsilon/(1−gamma).

To see the mechanism, use T_pi_v v=T_star v and insert these equal terms between T_star V_star and T_pi_v V_pi_v. Contraction bounds one difference by gamma epsilon and the other by gamma||v−V_pi_v||. Triangle inequality adds another epsilon plus the unknown policy-value gap; rearrangement yields the bound.

This is conservative and requires a uniform value approximation to V_star, not merely a low regression loss on visited states. It helps explain why a small residual needs stronger tolerances when gamma is near one.

<a id="section-11-7"></a>

## 11.7 Ties, finite horizons, and model error

Several actions can be optimal. A deterministic tie rule makes output reproducible but does not make its selected policy uniquely correct. Random tie handling can alter displayed routes without changing value.

For a finite horizon, backward induction replaces an infinite fixed-point iteration. Each backup uses next-time values already solved, so no discount contraction is needed merely to ensure termination of that finite computation.

Model error is a separate issue. The optimal policy for an approximate simulator may exploit its inaccuracies. Exact optimization amplifies the simulator's preferences just as reward optimization can amplify an evaluator's blind spots. Solver convergence is not evidence of real-world optimality.

<a id="section-11-8"></a>

## 11.8 Lab: separate numerical convergence from policy quality

Use [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) to compare optimal values with shortest-path calculations. Track both the change in values and changes in greedy actions. In a near-tie state, policy labels can change after the values already appear visually stable.

Construct the hidden-coin example as a one-step environment and calculate both operator orders. The correct learner without coin observation cannot average above 5 in expectation. If an implementation does, inspect whether the observation or evaluator leaked the coin.

For [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb), explain why its DP optimum is a model-informed baseline rather than a competitor with zero sample complexity under the same information budget.

<a id="section-11-9"></a>

## 11.9 From optimal backups to modern search

Reasoning systems sometimes sample several completed candidates and select the best verified answer. That selection occurs after paying the cost of generating and evaluating those candidates. It is not identical to an agent choosing the best unknown outcome without sampling.

Similarly, a tool agent can gather information before selecting an action if the task permits it and accounts for its cost. Changing what can be observed before acting changes the optimal decision problem.

The conceptual bridge is controlled optimization under an information structure. Bellman optimality is precise about when actions are selected and which uncertainty remains; modern evaluation should be equally explicit about sampling budgets and verifier access.

<a id="section-11-10"></a>

## 11.10 Problems, worked answers, and sources

**A.** Find the candidate value v(B) at which safe and risky tie in the numerical example. **B.** If epsilon=0.01 and gamma=0.9, compute the greedy-policy bound. **C.** Explain why best-of-eight answer accuracy cannot be compared fairly with one-sample accuracy without reporting the different budgets.

<details><summary>Worked answers</summary>

A: solve −1+0.72v(B)=3, so v(B)=50/9≈5.5556. B: 2×0.9×0.01/0.1=0.18. C: best-of-eight consumes eight generations and a selection mechanism, giving additional opportunities and information. Both metrics can be useful, but they represent different systems. Reporting only the higher score hides the resource and evaluator advantages.

</details>

**Read:** [Sutton and Barto, Chapters 3–4](http://incompleteideas.net/book/the-book-2nd.html). Chapters 12–16 turn these operators into planning algorithms.

---

[← Chapter 10](../10-bellman-expectation-equations/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 12 →](../12-dynamic-programming/README.md)
