# 31. Proximal Policy Optimization

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 30](../30-trust-region-policy-optimization/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 32 →](../32-entropy-regularization/README.md)

[Quick-read Topic 31](../../README.md#topic-31) · [Notation](../NOTATION.md)

- [31.1 Reuse a recent rollout with a restrained surrogate](#section-31-1)
- [31.2 Define the clipped actor objective](#section-31-2)
- [31.3 Work through both advantage signs](#section-31-3)
- [31.4 Explain why the minimum matters](#section-31-4)
- [31.5 Freeze the correct old policy information](#section-31-5)
- [31.6 Build the full training cycle](#section-31-6)
- [31.7 Interpret KL and clipping diagnostics](#section-31-7)
- [31.8 Handle token and trajectory weighting](#section-31-8)
- [31.9 Laboratory investigation](#section-31-9)
- [31.10 Problems, worked answers, and reading](#section-31-10)

---

<p align="center">
  <img src="../../assets/diagrams/ppo-training.svg" width="760" alt="PPO alternates fresh rollout collection with limited reuse of that batch." />
</p>

*PPO alternates fresh rollout collection with limited reuse of that batch.*

<a id="section-31-1"></a>

## 31.1 Reuse a recent rollout with a restrained surrogate

Proximal Policy Optimization (PPO) commonly uses a clipped probability-ratio objective to discourage some excessively large policy updates. It is popular partly because it works with ordinary minibatch optimizers.

**Prerequisites:** Chapters 25–30. **Targets:** calculate the sign-sensitive clipped objective, manage old-policy snapshots, and explain why clipping is not a hard KL bound.

The courier can learn several times from one recent batch of journeys, but the old route probabilities remain the reference for judging how far those updates have moved.

<a id="section-31-2"></a>

## 31.2 Define the clipped actor objective

```text
ratio = exp(log_pi_new(a|s) − log_pi_old(a|s))
L_clip = mean(min(ratio * A,
                  clip(ratio, 1−epsilon, 1+epsilon) * A))
```

Maximize this actor objective, or minimize its negative. Advantages and old log probabilities are fixed for the rollout batch. Critic and entropy losses are additional terms with separately chosen coefficients.

<a id="section-31-3"></a>

## 31.3 Work through both advantage signs

Let epsilon=0.2. For A=2 and ratio=1.5, the unclipped term is 3 and clipped term is 2.4, so the objective uses 2.4. Further increasing that sampled action's probability receives no direct benefit from this term.

For A=−2 and ratio=0.5, terms are −1 and −1.6, so the minimum is −1.6. Further decreasing that action receives no direct benefit. For A=−2 and ratio=1.5, the minimum is −3: moving in the harmful direction remains penalized.

<a id="section-31-4"></a>

## 31.4 Explain why the minimum matters

Clipping the ratio alone and multiplying by A would suppress some penalties as well as some benefits. Taking the minimum yields a pessimistic surrogate relative to the unclipped sample term.

This is not a projection of the policy into a permitted set. Shared parameters and other samples can still change a clipped action's probability. Unobserved actions and states can move without appearing directly in the batch objective.

<a id="section-31-5"></a>

## 31.5 Freeze the correct old policy information

Collect trajectories under pi_old and store each sampled action's old log probability. During several update epochs, keep those numbers unchanged. After the collection/update cycle, gather a fresh rollout using the updated policy.

Recomputing “old” probabilities after every optimizer step collapses the meaning of the ratio. In LLM alignment, a separate fixed reference model may define a KL anchor; it is not the rollout-generating old policy unless explicitly identical at that stage.

<a id="section-31-6"></a>

## 31.6 Build the full training cycle

```text
collect a rollout under the current policy snapshot
compute fixed returns and advantages with correct boundaries
for a limited number of epochs:
    sample rollout minibatches
    optimize clipped actor + value loss − entropy bonus
    monitor policy divergence and other diagnostics
collect fresh data
```

Record rollout size, minibatch size, epochs, clip range, learning rates, value loss, and advantage normalization. These choices determine actual data reuse and optimization pressure.

<a id="section-31-7"></a>

## 31.7 Interpret KL and clipping diagnostics

Clip fraction measures how many sampled ratios fall outside the clip interval under a defined convention. It is not the fraction of all policy changes constrained. Approximate KL estimates can be noisy and depend on the estimator and sampling distribution.

Early stopping based on KL is an additional practical control, not part of the algebraic clipping guarantee. Check actual return and task outcomes alongside surrogate metrics. A rising surrogate on a reused batch can coexist with worsening performance.

<a id="section-31-8"></a>

## 31.8 Handle token and trajectory weighting

For language policies, token ratios, sequence ratios, response masks, and loss reductions are distinct choices. A sum over tokens gives longer responses more terms; a per-response average changes weighting. Padding and prompt tokens must not accidentally contribute to response-action losses.

The exact implementation matters more than the label “PPO.” State how terminal response rewards, intermediate penalties, and value estimates produce each token's advantage before interpreting an alignment result.

<a id="section-31-9"></a>

## 31.9 Laboratory investigation

[Notebook 06](../../notebooks/06_ppo.ipynb) is the direct PPO implementation. Verify the three numerical clipping cases, then compare one versus several optimization epochs with a fixed interaction budget.

Use [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) for cross-method comparison. Report both environment interactions and gradient updates; PPO's reuse can change compute even when every learner receives the same number of transitions.

<a id="section-31-10"></a>

## 31.10 Problems, worked answers, and reading

1. With epsilon=0.2, A=3, and ratio=0.7, calculate L for one sample.
2. With A=−3 and ratio=1.1, calculate it again.
3. Why can an individual ratio move beyond the clipping interval after optimization?

<details><summary>Worked answers</summary>

1. min(2.1,2.4)=2.1. Decreasing a positive-advantage action remains penalized.
2. Both terms are −3.3, since 1.1 is inside the interval.
3. Clipping changes an objective term; it does not enforce a hard parameter or probability constraint. Gradients from other samples and losses can move the same policy outputs.

</details>

Read [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), distinguishing its proposed variants and empirical observations from trust-region guarantees.

---

[← Chapter 30](../30-trust-region-policy-optimization/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 32 →](../32-entropy-regularization/README.md)
