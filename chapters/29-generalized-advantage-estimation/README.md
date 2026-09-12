# 29. Generalized Advantage Estimation

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 28](../28-advantage-actor-critic/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 30 →](../30-trust-region-policy-optimization/README.md)

[Quick-read Topic 29](../../quick-read/29-generalized-advantage-estimation.md) · [Notation](../NOTATION.md)

- [29.1 Blend short and long credit assignment](#section-29-1)
- [29.2 Define the estimator](#section-29-2)
- [29.3 Derive the weighted residual sum](#section-29-3)
- [29.4 Calculate a terminal two-step example](#section-29-4)
- [29.5 Explain the two-mask boundary case](#section-29-5)
- [29.6 Interpret bias and variance precisely](#section-29-6)
- [29.7 Construct value targets consistently](#section-29-7)
- [29.8 Verify recursion before optimization](#section-29-8)
- [29.9 Laboratory and reasoning connection](#section-29-9)
- [29.10 Problems, worked answers, and reading](#section-29-10)

---

<a id="section-29-1"></a>

## 29.1 Blend short and long credit assignment

Generalized Advantage Estimation (GAE) accumulates discounted TD residuals. Its lambda parameter controls how strongly distant residuals contribute, offering a practical tradeoff between bootstrap dependence and sampling variability.

**Prerequisites:** Chapters 9, 18, and 27–28. **Targets:** derive the residual recursion, distinguish two boundary masks, and calculate GAE by hand.

The courier's judgment about an early turn can incorporate several later surprises, with progressively smaller weight. Lambda controls how far that chain of surprises reaches.

<a id="section-29-2"></a>

## 29.2 Define the estimator

```text
delta_t = r_t + gamma * bootstrap_mask_t * V(s_(t+1)) − V(s_t)
A_hat_t = delta_t + gamma * lambda * trace_mask_t * A_hat_(t+1)
```

The bootstrap mask answers whether continuation value belongs in the one-step target. The trace mask answers whether the next stored residual belongs to the same trajectory segment. These questions differ at collection cutoffs and autoresets.

<a id="section-29-3"></a>

## 29.3 Derive the weighted residual sum

Repeated substitution gives A_hat_t=delta_t+(gamma lambda)delta_(t+1)+(gamma lambda)²delta_(t+2)+… until the trace boundary. This is equivalent to a particular exponentially weighted mixture of multi-step advantage estimators under matching boundary conventions.

At lambda=0, GAE is one-step TD. At lambda=1 over a complete episode, TD residuals telescope to G_t−V(s_t). At a nonterminal finite rollout, the endpoint value remains in that telescoping expression.

### Prove the lambda-one telescoping identity

Over a segment of length n without an internal boundary, sum_(k=0 to n−1)gamma^k delta_(t+k) expands into discounted rewards plus value terms. Each intermediate positive continuation term cancels the next residual's negative current-value term.

The result is sum_k gamma^k r_(t+k)+gamma^n V(s_(t+n))−V(s_t). With true terminal continuation zero, this becomes the observed return minus current value. With a nonterminal endpoint, it remains a bootstrapped return-minus-value estimator. Lambda below one changes the cancellation weights and prevents this simple full telescoping.

<a id="section-29-4"></a>

## 29.4 Calculate a terminal two-step example

Let rewards be [1,2], values [0.5,1], gamma=0.9, lambda=0.8, and the second transition terminate. Then delta_0=1+0.9×1−0.5=1.4 and delta_1=2−1=1.

A_hat_1=1 and A_hat_0=1.4+0.72×1=2.12. The corresponding value targets V+A_hat are [2.62,2]. At lambda=1, the first advantage becomes 2.3, matching the full return 2.8 minus 0.5.

<a id="section-29-5"></a>

## 29.5 Explain the two-mask boundary case

At an external timeout of a continuing task, bootstrap from the valid final observation, so the bootstrap mask can be one. If the next stored observation comes from a reset episode, the trace mask must be zero.

For a true task deadline represented as termination, both are zero. At the end of a collected rollout without a reset, initialize the unobserved future advantage to zero while retaining the endpoint value in the last delta. This is a truncated estimator, not a claim that future residuals are truly zero.

<a id="section-29-6"></a>

## 29.6 Interpret bias and variance precisely

With an exact V_pi, each appropriately conditioned TD residual supports correct advantage estimation; with approximate values, lambda changes how errors enter. Complete Monte Carlo returns remove successor bootstrap error but can have higher variance.

The common bias–variance description is a design guide, not a universal monotonic theorem for every finite batch, critic, and environment. Gamma also changes the objective or weighting convention; it should not be treated as merely another harmless variance knob.

### Track the coefficient on each critic error

Write V_hat=V_pi+e. For an unbroken n-step GAE segment, the value-error contribution is −e_t+gamma(1−lambda)sum_(j=1 to n−1)(gamma lambda)^(j−1)e_(t+j)+gamma^n lambda^(n−1)e_(t+n).

This follows by collecting each state's positive continuation error and negative current error across adjacent weighted residuals. At lambda=1, intermediate errors cancel, leaving −e_t+gamma^n e_endpoint. At lambda=0, the expression reduces to the one-step error gamma e_(t+1)−e_t, interpreting the n=1 boundary case separately.

The derivation explains the role of lambda more precisely than saying it “smooths advantages.” It changes how errors from interior and endpoint critic estimates enter the signal.

<a id="section-29-7"></a>

## 29.7 Construct value targets consistently

Many implementations use V_old+A_hat as the critic target, frozen for the update batch. Others use separately computed returns. State which target is used, especially when applying advantage normalization for the actor.

Do not train the critic toward normalized actor advantages plus values unless that is deliberately the chosen algorithm. Normalizing advantages for policy optimization should not silently redefine the reward scale of value regression.

<a id="section-29-8"></a>

## 29.8 Verify recursion before optimization

Use a hand-built rollout containing a nonterminal step, a true termination, and a separate timeout. Compare the backward recursion with an explicit weighted sum on each segment.

Check that no reward from a reset episode influences the previous episode's advantage. Freeze values while computing the batch, detach targets, and verify that changing the timeout endpoint value affects only the intended preceding segment.

### Use a direct-sum implementation as a small oracle

For a short stored segment, compute each advantage by explicitly summing all later TD residuals with powers of gamma lambda until the appropriate trace boundary. Compare this slow O(T²) calculation with the fast O(T) backward recursion.

Use nonuniform values and rewards so an indexing error cannot cancel accidentally. Include a true termination and a timeout with a nonzero final value. This is an independent formulation of the same estimator, suitable for checking the optimized recursion without repeating its exact implementation.

<a id="section-29-9"></a>

## 29.9 Laboratory and reasoning connection

[Notebook 06](../../notebooks/06_ppo.ipynb) is the direct GAE/PPO lab. Compare lambda settings under the same rollout and interaction budgets; include critic quality as a possible explanation for differences.

Group-relative reasoning methods can avoid a learned value critic, but they do not make credit assignment disappear. GAE distributes temporal residuals; group methods compare sampled completions or traces. Their baselines and data structures answer different questions.

<a id="section-29-10"></a>

## 29.10 Problems, worked answers, and reading

1. Recalculate the first advantage in the example at lambda=0 and lambda=0.5.
2. At a timeout, r=1, V(s)=2, V(final)=4, gamma=0.9. What is delta?
3. Why should a normalized actor advantage not automatically become the critic target?

<details><summary>Worked answers</summary>

1. At zero: 1.4. At 0.5: 1.4+0.45×1=1.85.
2. With valid continuing-task bootstrapping, delta=1+3.6−2=2.6. The trace still stops if the next stored step belongs to a reset episode.
3. Normalization changes scale and offset using batch statistics, while the critic is intended to predict return in the task's reward units.

</details>

Read [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), especially its distinction between discounting and lambda-based estimation.

---

[← Chapter 28](../28-advantage-actor-critic/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 30 →](../30-trust-region-policy-optimization/README.md)
