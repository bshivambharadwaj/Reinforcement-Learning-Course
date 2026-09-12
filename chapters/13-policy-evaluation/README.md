# 13. Policy Evaluation

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 12](../12-dynamic-programming/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 14 →](../14-policy-improvement/README.md)

[Quick-read Topic 13](../../quick-read/13-policy-evaluation.md) · [Notation](../NOTATION.md)

- [13.1 Evaluation asks a conditional question](#section-13-1)
- [13.2 Construct policy-induced quantities](#section-13-2)
- [13.3 Solve a coupled example](#section-13-3)
- [13.4 Compare direct and iterative solutions](#section-13-4)
- [13.5 Derive an error certificate](#section-13-5)
- [13.6 Interpret state weighting](#section-13-6)
- [13.7 Know when the proof fails](#section-13-7)
- [13.8 Distinguish three evaluation protocols](#section-13-8)
- [13.9 Laboratory investigation](#section-13-9)
- [13.10 Problems, worked answers, and reading](#section-13-10)

---

<a id="section-13-1"></a>

## 13.1 Evaluation asks a conditional question

Policy evaluation asks: if the courier follows this policy from now on, what return should we expect? It does not ask whether the policy is optimal. Keeping the policy fixed makes the problem linear in tabular finite MDPs.

**Prerequisites:** Chapters 7, 10, and 12. **Targets:** solve an evaluation system, interpret convergence tolerances, and distinguish exact-model evaluation from empirical policy assessment.

The distinction remains useful in modern agents: estimating how well the current tool policy behaves is separate from deciding how to change it.

<a id="section-13-2"></a>

## 13.2 Construct policy-induced quantities

Average over actions using the policy probabilities:

```text
r_pi(s) = sum_a pi(a|s) r(s,a)
P_pi(s,s') = sum_a pi(a|s) P(s'|s,a)
v_pi = r_pi + gamma P_pi v_pi
```

For masked terminal continuation, the transient-state matrix can be substochastic. Its missing mass represents termination. Alternatively include an absorbing state with value zero. Do not mix the two conventions by counting terminal rewards twice.

### Derive the transient-state convention explicitly

Partition a fixed policy's transition matrix into nonterminal states and a terminal absorbing state. The nonterminal block P_N can have row sums below one because some probability mass exits to termination. Since terminal value is zero, nonterminal values satisfy v_N=r_N+gamma P_N v_N.

For one state that pays 2, then terminates with probability 0.25 and returns to itself otherwise, v=2+0.75gamma v. At gamma=0.9 this is 80/13≈6.1538. At gamma=1 it is 8, still finite because expected episode length is four. This example shows why undiscounted episodic evaluation can work even though the discounted contraction argument does not apply unchanged.

<a id="section-13-3"></a>

## 13.3 Solve a coupled example

State A gives reward 1 and moves to B. At B, the fixed policy produces expected reward 2 and transitions equally to A and termination. Let gamma=0.5.

The equations are v_A=1+0.5v_B and v_B=2+0.25v_A. Substitute the second into the first: 0.875v_A=2. Therefore v_A=16/7 and v_B=18/7.

The term 0.25v_A combines both the discount 0.5 and transition probability 0.5. Forgetting either factor changes the model, not merely the numerical precision.

<a id="section-13-4"></a>

## 13.4 Compare direct and iterative solutions

Direct evaluation solves (I−gamma P_pi)v=r_pi. Numerically, use a linear solver rather than explicitly computing an inverse. Dense factorization is approximately O(S³), so iterative methods can be preferable for large sparse problems.

Starting from zero, synchronous evaluation of the example gives [1,2], then [2,2.25], then [2.125,2.5]. These are partial estimates, not complete policy evaluations. At gamma<1, repeated backups converge to the unique fixed point.

<a id="section-13-5"></a>

## 13.5 Derive an error certificate

Let d=||T_pi v−v||_infinity. Using v_pi=T_pi v_pi and contraction:

```text
||v_pi−v|| ≤ ||T_pi v_pi−T_pi v|| + ||T_pi v−v||
           ≤ gamma ||v_pi−v|| + d
therefore ||v_pi−v|| ≤ d/(1−gamma)
```

This is a full-model, all-state certificate. A sampled residual on one minibatch is not the same quantity. At gamma=0.99, residual 0.01 allows value error up to 1; the discount strongly affects useful stopping tolerances.

### Compare update size with residual after an in-place sweep

In synchronous evaluation, the full update difference T_pi v−v is directly a Bellman residual of the old vector. In an in-place sweep, each state uses a mixture of old and newly updated entries. The largest assignment change during that sweep is not automatically the full residual of either the initial or final vector.

After an in-place sweep, independently recompute T_pi v using the completed vector if you want the standard residual certificate. This extra pass costs computation but removes ambiguity. A stopping tolerance should be attached to a precisely defined measured quantity, not simply to a variable named `delta`.

<a id="section-13-6"></a>

## 13.6 Interpret state weighting

A vector v_pi describes every modeled state. A reported scalar performance J(pi)=sum_s rho(s)v_pi(s) uses a chosen initial-state distribution. A policy can improve average performance while worsening a rarely initialized state.

For deployment, include the relevant initial states and subgroup summaries. An agent benchmark that samples only easy tasks can produce accurate evaluation of the wrong population. Changing rho changes the summary even when the policy and state values are unchanged.

<a id="section-13-7"></a>

## 13.7 Know when the proof fails

At gamma=1, a continuing positive-reward self-loop has infinite return; the discounted inverse argument no longer applies. Episodic undiscounted evaluation can still work under suitable transience conditions, but requires a separate argument.

With function approximation, evaluation includes projection onto a restricted class and a data distribution. The exact tabular contraction proof does not automatically survive that projection. With a changing policy, the target itself also moves.

<a id="section-13-8"></a>

## 13.8 Distinguish three evaluation protocols

| Protocol | Information | Main uncertainty |
|---|---|---|
| Exact model evaluation | Full P and r | Model correctness |
| Fresh rollouts | Ability to run the policy | Sampling error and test coverage |
| Logged-data evaluation | Another policy's trajectories | Coverage, weighting, and model assumptions |

For fresh rollouts, report independent episodes and uncertainty rather than treating correlated timesteps as independent trials. Logged-data evaluation needs additional assumptions developed in Chapter 33.

### Estimate a confidence interval under a stated sampling model

For independent evaluation returns G_1,…,G_n with sample standard deviation s, the estimated standard error of the mean is s/sqrt(n). A large-sample approximate 95% interval is mean±1.96s/sqrt(n), subject to the approximation and sampling assumptions.

If n=100, mean=3, and s=2, the interval is approximately [2.608,3.392]. Heavy-tailed outcomes, small samples, dependent episodes, and repeated model selection can make this approximation unreliable. Always report the raw sample count and grouping unit so readers know what the uncertainty calculation treats as independent.

<a id="section-13-9"></a>

## 13.9 Laboratory investigation

Use [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) to compare a direct solution with repeated backups. Plot the actual maximum error and the residual-based upper bound on the same iterations. Repeat for gamma=0.5 and 0.95.

In [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb), distinguish exact policy value from sampled test return. Their difference should be interpreted through finite evaluation samples and implementation consistency, not automatically as a learning failure.

<a id="section-13-10"></a>

## 13.10 Problems, worked answers, and reading

1. Compute J for the example when rho(A)=0.75 and rho(B)=0.25.
2. At gamma=0.8, what full residual suffices to certify value error at most 0.05?
3. If a batch residual is zero, what additional evidence would be needed for the tabular certificate?

<details><summary>Worked answers</summary>

1. J=(0.75×16+0.25×18)/7=33/14, approximately 2.3571.
2. Require d≤(1−0.8)×0.05=0.01.
3. The residual must use the correct model and cover every relevant state, with the norm computed over the full state set. Exact zeros on a sampled subset leave unobserved states unconstrained.

</details>

Read [Sutton and Barto, Chapter 4.1](http://incompleteideas.net/book/the-book-2nd.html). Reconstruct the contraction argument without assuming the desired fixed point error is already small.

---

[← Chapter 12](../12-dynamic-programming/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 14 →](../14-policy-improvement/README.md)
