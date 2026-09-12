# 16. Value Iteration

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 15](../15-policy-iteration/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 17 →](../17-monte-carlo-methods/README.md)

[Quick-read Topic 16](../../quick-read/16-value-iteration.md) · [Notation](../NOTATION.md)

- [16.1 Plan with repeated optimal backups](#section-16-1)
- [16.2 Specify initialization and boundaries](#section-16-2)
- [16.3 Trace propagation along a chain](#section-16-3)
- [16.4 Derive convergence and residual bounds](#section-16-4)
- [16.5 Extract the policy with a full backup](#section-16-5)
- [16.6 Make stopping criteria meaningful](#section-16-6)
- [16.7 Understand optimistic initialization](#section-16-7)
- [16.8 Diagnose errors using invariants](#section-16-8)
- [16.9 Laboratory and search connection](#section-16-9)
- [16.10 Problems, worked answers, and reading](#section-16-10)

---

<a id="section-16-1"></a>

## 16.1 Plan with repeated optimal backups

Value iteration updates values toward optimality without fully evaluating an intermediate policy. At each state, it asks which action looks best under the current continuation estimates.

**Prerequisites:** Chapters 11–12. **Targets:** trace information propagation, derive a stopping certificate, and extract and verify a policy.

Think of reward information moving backward across a road network. A destination first improves its direct predecessors; later sweeps carry that news farther away. Update order controls how quickly that news travels within a sweep.

<a id="section-16-2"></a>

## 16.2 Specify initialization and boundaries

```text
v = any finite vector, with terminal values fixed at zero
repeat:
    v_next(s) = max_a [r(s,a) + gamma * P(s,a) dot v]
    residual = max_s abs(v_next(s) − v(s))
    v = v_next
```

For infinite-horizon discounted finite MDPs, gamma<1 guarantees convergence. In a finite-horizon problem, use backward induction with explicit time rather than stopping based on a discounted stationary fixed point.

<a id="section-16-3"></a>

## 16.3 Trace propagation along a chain

Let A→B→C→terminal be the only available route. The first two transitions pay 0; the final transition pays 1. With gamma=0.9 and zero initialization, synchronous sweeps give:

| Sweep | A | B | C |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 0.9 | 1 |
| 3 | 0.81 | 0.9 | 1 |

Updating C, then B, then A in place produces the final vector in one sweep for this acyclic example. The two methods did not perform identical intermediate computations.

<a id="section-16-4"></a>

## 16.4 Derive convergence and residual bounds

Contraction gives ||v_k−V_star||≤gamma^k||v_0−V_star||. The geometric rate slows as gamma approaches one. A computable residual d=||T_star v−v|| also yields ||v−V_star||≤d/(1−gamma).

If d_k=||v_(k+1)−v_k||, this residual directly certifies v_k. The newly computed v_(k+1) has error at most gamma d_k/(1−gamma) by contraction. Indexing the certificate matters when a program reports the updated vector.

### Derive an iteration budget from bounded rewards

Starting from zero with |r|≤R_max, the initial value error is at most R_max/(1−gamma). Contraction yields ||v_k−V_star||∞≤gamma^k R_max/(1−gamma). To guarantee error at most epsilon, choose k satisfying the same geometric inequality used for a discounted return tail.

At gamma=0.9, R_max=1, and epsilon=0.01, k=66 suffices under exact synchronous backups. This is a worst-case bound. A finite acyclic chain can converge exactly after its maximum path length, so the bound should not replace observed residuals when those are cheaply available.

<a id="section-16-5"></a>

## 16.5 Extract the policy with a full backup

Choose actions maximizing r(s,a)+gamma P(s,a) dot v. Do not choose a successor merely because it has the largest value: actions may differ in immediate costs, probabilities, and termination.

If v approximates V_star uniformly within epsilon, the extracted policy has value loss at most 2gamma epsilon/(1−gamma). This bound can be loose; evaluate the extracted policy exactly when a small model makes that affordable.

### A greedy policy can be correct before values converge

In a one-state problem, action A gives reward 1 and self-loops; action B gives reward 0 and self-loops. With any common candidate continuation v and gamma=0.9, A's backup exceeds B's by 1. The greedy action is correct immediately, while zero-initialized value iteration approaches the true value 10 over many sweeps.

This explains why action-map stability can coexist with significant value error. It can be useful if the goal is control, but it is not an all-purpose stopping proof: in other models, continuation errors can still reverse a close ranking at an unexamined state.

<a id="section-16-6"></a>

## 16.6 Make stopping criteria meaningful

“The action map stopped changing” is not a value convergence certificate. Values may continue changing while the best action remains stable. Conversely, nearly tied optimal actions may keep changing even when values are accurate.

Choose whether the objective is accurate values, a sufficiently good policy, or limited compute, and report the corresponding evidence. For a desired value error eta, the old-vector residual criterion d≤(1−gamma)eta is sufficient under exact backups.

<a id="section-16-7"></a>

## 16.7 Understand optimistic initialization

Starting from a componentwise upper bound can preserve an upper-bounding sequence when backups are applied correctly. For bounded absolute rewards R_max, the constant R_max/(1−gamma) bounds discounted values above, with terminal boundaries handled separately.

Optimism in a planner is an initialization strategy, not exploration of an unknown environment. In Q-learning, optimistic estimates can influence which data are gathered; exact DP already has the transition model and does not need to discover its rows.

### Establish monotone lower and upper sequences

If v_0≤T_star v_0, monotonicity implies v_0≤v_1≤v_2≤…≤V_star when v_0 is also a lower bound. Similarly, an upper initialization with v_0≥T_star v_0 produces a decreasing upper sequence.

For rewards in [−R_max,R_max], constant vectors ±R_max/(1−gamma) provide crude bounds under compatible absorbing-state conventions. Running both sequences gives a bracket for each value. The gap can certify convergence without knowing V_star, but both sequences still rely on the correct model and exact backups.

<a id="section-16-8"></a>

## 16.8 Diagnose errors using invariants

Verify terminal values, probability sums, legal action masks, and the Bellman residual computed independently of the update function. A duplicated bug in an update and its test can produce a false pass.

Use a hand-solvable chain and the two-state policy-iteration example as separate checks. Compare rewards counted by a rollout with rewards counted by the planner. Off-by-one discounting often survives visual inspection of a smooth convergence curve.

<a id="section-16-9"></a>

## 16.9 Laboratory and search connection

Use [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) for the tabular experiment, and [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) to interpret the DP reference against sampled learners.

For reasoning search, deeper lookahead similarly propagates terminal feedback backward. However, approximate transition generation, heuristic leaf scores, and a restricted search tree make it a different computation. Record search breadth and depth when comparing it with a policy that gets only one rollout.

<a id="section-16-10"></a>

## 16.10 Problems, worked answers, and reading

1. With gamma=0.9 and d_k=0.002, bound the error of v_k and v_(k+1).
2. Add an immediate terminal action worth 0.85 at A in the chain. Which action is optimal?
3. Why is a stable greedy action map insufficient to certify exact values?

<details><summary>Worked answers</summary>

1. Old vector: at most 0.02. New vector: at most 0.018. These are uniform exact-model bounds.
2. The route is worth 0.81, so take 0.85 immediately. The maximum successor value alone would miss this comparison.
3. The same action can maximize a family of inaccurate vectors. Action ranking need not change as those vectors approach the fixed point.

</details>

Read [Sutton and Barto, Chapters 4.4–4.5](http://incompleteideas.net/book/the-book-2nd.html). Compare synchronous and asynchronous algorithms by backup count as well as wall time.

---

[← Chapter 15](../15-policy-iteration/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 17 →](../17-monte-carlo-methods/README.md)
