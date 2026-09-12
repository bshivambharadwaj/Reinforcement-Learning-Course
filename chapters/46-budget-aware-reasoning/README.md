# 46. Budget-Aware Reasoning

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 45](../45-search-over-reasoning-trajectories/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 47 →](../47-learning-from-search/README.md)

[Quick read](../../quick-read/46-budget-aware-reasoning.md) · [Notation](../NOTATION.md)

- [46.1 Treat computation as a resource with opportunity cost](#section-46-1)
- [46.2 Specify the resource model](#section-46-2)
- [46.3 Derive a one-more-attempt threshold](#section-46-3)
- [46.4 Derive diminishing returns for oracle success](#section-46-4)
- [46.5 Define budget-conditioned value](#section-46-5)
- [46.6 Allocate across multiple tasks](#section-46-6)
- [46.7 Distinguish early stopping from selective reporting](#section-46-7)
- [46.8 Read a success-versus-cost frontier](#section-46-8)
- [46.9 Laboratory and demo](#section-46-9)
- [46.10 Exercises, answers, and reading](#section-46-10)

---

<a id="section-46-1"></a>

## 46.1 Treat computation as a resource with opportunity cost

An inference system must decide not only what answer to propose, but how much additional work an uncertain task deserves. More computation has value only if it can improve a relevant outcome enough to justify its cost.

**Prerequisites:** Chapters 6, 30, and 41–45. **Learning targets:** distinguish fixed budgets from adaptive allocation, derive a simple stopping rule, and design resource-aware comparisons.

A courier dispatcher should not spend an hour optimizing an already adequate two-minute route. The useful next computation depends on uncertainty, stakes, and available alternatives.

<a id="section-46-2"></a>

## 46.2 Specify the resource model

A resource budget can count generated tokens, model forward passes, verifier calls, tool actions, wall time, energy, or money. These units are not generally interchangeable. Batching and caching can reduce wall time without reducing the logical number of candidates.

The simulator uses proposal steps plus weighted score calls. A whole-trace score and a prefix score each count as one query under this declared abstraction, even though their real computational work could differ. Vary score cost to test sensitivity rather than presenting teaching units as hardware measurements.

<a id="section-46-3"></a>

## 46.3 Derive a one-more-attempt threshold

Suppose a trustworthy mechanism confirms whether an independent new attempt succeeds with probability p. The value of obtaining success is U, no success is currently available, and the extra attempt costs c. Under this simple one-step utility model, trying is worthwhile when pU>c.

With p=0.2, U=10, and c=1, expected gain is 1. If a confirmed success is already available and there is no reward for additional quality, another attempt adds cost without utility. Real systems usually lack exact p and perfect confirmation, so this threshold illustrates the decision rather than solving it generally.

<a id="section-46-4"></a>

## 46.4 Derive diminishing returns for oracle success

Under independent attempts, oracle success after n samples is 1−(1−p)^n. The increment from sample n+1 is p(1−p)^n, decreasing with n for 0<p<1.

At p=0.2, the first sample adds 0.2 opportunity and the sixth adds 0.2×0.8^5≈0.06554. If sample cost is fixed and utility linear in oracle success, a threshold on this increment yields a simple allocation rule. An imperfect selector or correlated candidates can change the shape entirely.

<a id="section-46-5"></a>

## 46.5 Define budget-conditioned value

A computation value V(m,b) depends on current search memory m and remaining budget b. Stopping returns the utility of the current selected answer. Continuing chooses an affordable computation and averages over the information it may produce.

This formulation makes uncertainty reduction valuable: scoring a candidate can change which answer is returned even without generating a new one. It also makes information useless when there is no remaining opportunity to act on it. The demo uses predefined policies; an optimal metareasoning solver is a future extension.

<a id="section-46-6"></a>

## 46.6 Allocate across multiple tasks

With a shared total budget, spending more on one task reduces resources available to others. A policy assigning every task the same maximum may waste compute on easy cases and underserve hard but solvable ones.

To evaluate adaptive allocation, define the task distribution and total budget before testing. Report per-task budgets and subgroup outcomes. Using a hidden correctness label to identify hard tasks grants oracle information; a deployed allocator must use observable features or calibrated uncertainty estimates.

<a id="section-46-7"></a>

## 46.7 Distinguish early stopping from selective reporting

An early-stopping rule can use an available score or verification result, with its query cost recorded. Stopping the evaluation experiment after a favorable streak of tasks is a separate statistical decision that can bias reported performance.

Freeze the per-task stopping rule before final testing, and evaluate the full planned task set. Report abstentions and failures. A system that returns fewer answers may improve conditional precision without improving overall completion; both quantities matter.

<a id="section-46-8"></a>

## 46.8 Read a success-versus-cost frontier

Plot verified success against actual average resource use and show the configured caps separately. A strategy leaving budget unused should appear at its actual cost, not at an invented expenditure equal to the cap.

If one method uses fewer proposal steps but many expensive verifier calls, its apparent advantage depends on the cost model. A useful report includes separate cost components and sensitivity experiments rather than only one composite score chosen after observing results.

<a id="section-46-9"></a>

## 46.9 Laboratory and demo

In [Notebook 13](../../notebooks/13_budgeted_reasoning_search.ipynb), repeat the comparison with score costs one and three. Keep tasks, policy, generation seeds, and maximum units fixed. Explain whether changed rankings reflect fewer candidates, fewer completed branches, or different selection behavior.

The [demo CLI](../../demos/inference-time/README.md) exposes `--budgets`, `--depth`, and `--score-cost`. Its report filters precomputed runs interactively; changing the numerical experiment requires rerunning the CLI, not merely changing a browser selector.

<a id="section-46-10"></a>

## 46.10 Exercises, answers, and reading

1. With p=0.25, what oracle-success increment does the third attempt add?
2. At depth 3 and B=24, how many candidates can best-of-N score when scoring costs 1 versus 3?
3. Why should a benchmark report actual cost in addition to budget caps?

<details><summary>Worked answers</summary>

1. 0.25×0.75²=0.140625, assuming independent attempts.
2. Six versus four complete scored candidates.
3. Strategies can stop early, exhaust their configured trees, or be unable to spend a remainder. Equal caps do not imply equal consumed resources.

</details>

Read [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) for empirical allocation questions. The simple stopping calculations here depend on explicitly stated oracle and independence assumptions.

---

[← Chapter 45](../45-search-over-reasoning-trajectories/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 47 →](../47-learning-from-search/README.md)
