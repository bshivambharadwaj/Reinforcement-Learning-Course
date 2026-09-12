# 15. Policy Iteration

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 14](../14-policy-improvement/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 16 →](../16-value-iteration/README.md)

[Quick-read Topic 15](../../quick-read/15-policy-iteration.md) · [Notation](../NOTATION.md)

- [15.1 Alternate understanding and revision](#section-15-1)
- [15.2 Write the algorithm precisely](#section-15-2)
- [15.3 Trace a two-state problem](#section-15-3)
- [15.4 Explain finite termination](#section-15-4)
- [15.5 Separate theorem from complexity](#section-15-5)
- [15.6 Modified policy iteration](#section-15-6)
- [15.7 Handle numerical ties deliberately](#section-15-7)
- [15.8 Compare against another solver](#section-15-8)
- [15.9 Laboratory and conceptual bridge](#section-15-9)
- [15.10 Problems, worked answers, and reading](#section-15-10)

---

<p align="center">
  <img src="../../assets/diagrams/policy-iteration.svg" width="760" alt="Policy evaluation and improvement form distinct stages." />
</p>

*Policy evaluation and improvement form distinct stages.*

<a id="section-15-1"></a>

## 15.1 Alternate understanding and revision

Policy iteration alternates a complete answer to “how good is this policy?” with a revision toward better actions. The separation makes it a useful reference algorithm even when large-scale methods use approximate versions.

**Prerequisites:** Chapters 13–14. **Targets:** trace policy iteration, establish its finite tabular stopping argument, and compare evaluation accuracy against planning cost.

The courier analogy is a route review cycle: calculate the consequences of the current manual, update the manual, then recalculate consequences. Repeatedly editing the manual without reevaluating it is a different procedure.

<a id="section-15-2"></a>

## 15.2 Write the algorithm precisely

```text
initialize a deterministic legal policy pi
repeat:
    v = solve policy evaluation for pi
    for every state s:
        choose a maximizer of r(s,a) + gamma P(s,a) dot v
        retain pi(s) if it is already a maximizer
    if no action changed: return pi, v
    otherwise replace pi with the improved policy
```

Finite discounted assumptions make exact evaluation well-defined. Floating-point implementations need explicit tolerances; an “unchanged” decision based on loose tolerances is an approximate stopping claim.

<a id="section-15-3"></a>

## 15.3 Trace a two-state problem

At A, exit pays 1 and terminates; go pays 0 and reaches B. At B, exit pays 4 and terminates; loop pays 1 and returns to B. Let gamma=0.5 and start with exit at A, loop at B.

Evaluation gives V_A=1 and V_B=2. At A, go has value 0+0.5×2=1, a tie, so retain exit. At B, exit has value 4, so switch to exit. Reevaluation gives [1,4]. Now go at A has value 2, so switch. The final values are [2,4].

One improvement round was insufficient because the improved downstream decision had to propagate back to A through reevaluation.

<a id="section-15-4"></a>

## 15.4 Explain finite termination

There are finitely many deterministic policies. Exact evaluation and greedy improvement never decrease the value vector. With stable tie handling, any changed action has a strict old-value improvement at some state, yielding a strictly better value somewhere after reevaluation.

Thus the process cannot revisit a previous policy indefinitely. When it stops, the evaluated policy is greedy with respect to its own values, so those values satisfy the optimal Bellman equation. Uniqueness identifies them as V_star.

### Why revisiting a policy contradicts strict progress

With exact evaluation, a deterministic policy determines a unique discounted value vector. If a changed action is strictly better under the old values and ties retain the old action, policy improvement produces a value vector that is no worse everywhere and strictly better somewhere.

Suppose a later iteration revisited an earlier policy. Its value would equal that policy's earlier value, contradicting the chain of componentwise nondecrease with at least one strict improvement along the cycle. This argument relies on exact comparisons and a finite deterministic-policy set. Approximate evaluation and numerical tolerances require an approximate stopping interpretation instead.

<a id="section-15-5"></a>

## 15.5 Separate theorem from complexity

Finite termination does not mean a small worst-case number of iterations. A crude bound based on the number of deterministic policies can be exponential in the number of states. In practice, policy iteration often uses few improvement rounds but spends substantial work evaluating each policy.

Report Bellman operations or linear-solver work, not only outer iterations. Comparing ten value-iteration sweeps to ten full policy-iteration evaluations as equal compute would be misleading.

<a id="section-15-6"></a>

## 15.6 Modified policy iteration

Modified policy iteration performs a limited number of evaluation sweeps before improving. It interpolates between more thorough evaluation and rapid revision. The current vector is then only an approximation to V_pi.

This can save work, but the exact theorem's premise must be reconsidered when discussing each intermediate policy. A small change in the value vector does not necessarily certify accurate evaluation under the newly changed policy. Document the evaluation sweep count and stopping rule.

### Inspect a modified-policy-iteration step

Suppose evaluation is stopped after k Bellman expectation sweeps from a previous estimate v. The new vector is T_pi^k v, with error at most gamma^k||v−V_pi||∞ under exact tabular backups. Increasing k reduces this evaluation error geometrically for the fixed current policy.

But after improvement changes pi, the target value V_pi changes too. A small residual for the previous policy does not certify the new one. Record the policy identity alongside every residual. This prevents a log from presenting an accurately evaluated old policy as evidence that a newly updated policy has already been evaluated.

<a id="section-15-7"></a>

## 15.7 Handle numerical ties deliberately

Use an action-change tolerance relative to the scale of rewards and numerical solve accuracy. If the best backup improves on the old action by less than the tolerance, retaining the action prevents oscillation.

This choice introduces approximation. Log the final optimal Bellman residual so readers can assess what the tolerance implies. Very large reward magnitudes may need different numerical tolerances even when positive scaling leaves the optimal policy unchanged.

<a id="section-15-8"></a>

## 15.8 Compare against another solver

In a small MDP, run policy iteration and value iteration independently. Compare final values, not just action labels: two different actions can be exactly tied. Check each reported policy by a fresh evaluation solve.

Deliberately initialize an unfavorable policy and a near-optimal one. Measure total backups and evaluations, then ask whether the observed speed difference is due to initialization or algorithm structure. Include unsuccessful or slow cases in the report.

### Compare algorithms using work that actually occurred

For a dense model, count state-action-successor terms processed during improvement and evaluation. A policy-evaluation solve has a different cost structure from a sweep, so record solve time and matrix size as well as sweep counts.

Use the same final policy-value error or residual target when comparing value iteration and policy iteration. Equal outer iteration counts do not represent equal accuracy or compute. Finally evaluate the extracted policies with a separate routine; matching internal stopping flags is weaker evidence than matching independently calculated returns.

<a id="section-15-9"></a>

## 15.9 Laboratory and conceptual bridge

[Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) supplies the planning setting. A useful extension is an audit table with one row per outer iteration: policy, evaluated values, changed actions, and final residual.

Actor–critic resembles this separation at a broad level: a critic evaluates and an actor changes decisions. It is not exact policy iteration. Shared parameters, stochastic gradients, evolving data, and incomplete evaluation remove the simple finite-policy proof.

<a id="section-15-10"></a>

## 15.10 Problems, worked answers, and reading

1. In the example, what happens if ties at A are resolved in favor of go in the first round?
2. Why does a stable policy under exact improvement satisfy optimality?
3. A policy has a tiny evaluation residual but a large optimality residual. Interpret this.

<details><summary>Worked answers</summary>

1. Both actions change in the first improvement. Reevaluation gives [2,4], already optimal. Stable tie handling changes the path, not the optimal value.
2. Evaluation gives v=T_pi v and stability gives T_pi v=T_star v. Therefore v is the unique optimal fixed point in the discounted setting.
3. Its values are accurately evaluated, but some alternative actions are much better. Evaluation accuracy is not policy quality.

</details>

Read [Sutton and Barto, Chapters 4.3 and 4.6](http://incompleteideas.net/book/the-book-2nd.html), including generalized policy iteration.

---

[← Chapter 14](../14-policy-improvement/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 16 →](../16-value-iteration/README.md)
