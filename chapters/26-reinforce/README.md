# 26. REINFORCE

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 25](../25-policy-gradient-methods/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 27 →](../27-actor-critic-methods/README.md)

[Quick-read Topic 26](../../README.md#topic-26) · [Notation](../NOTATION.md)

- [26.1 Turn the policy-gradient identity into an algorithm](#section-26-1)
- [26.2 Specify the objective before the loss](#section-26-2)
- [26.3 Calculate reward-to-go backward](#section-26-3)
- [26.4 Add a state-value baseline](#section-26-4)
- [26.5 Analyze a baseline that removes variance](#section-26-5)
- [26.6 Treat normalization as an algorithmic choice](#section-26-6)
- [26.7 Keep the collection policy fixed within a batch](#section-26-7)
- [26.8 Diagnose sparse-reward failure](#section-26-8)
- [26.9 Laboratory investigation](#section-26-9)
- [26.10 Problems, worked answers, and reading](#section-26-10)

---

<a id="section-26-1"></a>

## 26.1 Turn the policy-gradient identity into an algorithm

REINFORCE samples complete trajectories and weights each sampled action's log probability by its observed return, optionally subtracting a baseline. It is a direct way to see how outcome feedback changes a stochastic policy.

**Prerequisites:** Chapters 17 and 25. **Targets:** implement reward-to-go correctly, analyze variance reduction, and distinguish a Monte Carlo actor update from bootstrapped actor–critic.

The courier waits for delivery outcomes, then increases the probability of choices associated with better-than-expected results. Waiting makes the target conceptually simple but can be statistically expensive.

<a id="section-26-2"></a>

## 26.2 Specify the objective before the loss

For an undiscounted episodic objective, the actor loss sums −log pi(A_t|S_t)×(G_t−b(S_t)) over timesteps. For the start-state discounted objective in Chapter 25, include the corresponding gamma^t weighting.

Batch aggregation also matters. Averaging trajectory sums gives equal episode weighting; averaging over all timesteps can reweight variable-length episodes through the denominator. State whether the optimization targets episode return or a different normalization convention.

<a id="section-26-3"></a>

## 26.3 Calculate reward-to-go backward

```text
running_return = 0
for t from final decision down to first:
    running_return = reward[t] + gamma * running_return
    returns[t] = running_return
```

For rewards [−1,−1,5] and gamma=0.9, the returns are [2.15,3.5,5]. A full-episode weight of 2.15 on every action would include irrelevant earlier rewards for later decisions and generally have different variance from reward-to-go.

<a id="section-26-4"></a>

## 26.4 Add a state-value baseline

Train b_w(s) toward MC returns with a regression loss. The actor uses detached G_t−b_w(S_t). A state-only baseline can reduce variance without changing the expected gradient under the score-function assumptions.

The variance-minimizing scalar baseline is not always the plain mean return: score magnitudes affect the optimum. The value baseline is a useful and interpretable approximation, not a universal exact variance minimizer.

<a id="section-26-5"></a>

## 26.5 Analyze a baseline that removes variance

For the one-step two-action problem paying 4 or 0 at p=0.5, the score is +0.5 or −0.5. Without a baseline, reward×score is 2 or 0, each with probability 0.5; its mean is 1 and variance is 1.

With baseline 2, both outcomes give gradient estimate 1: (4−2)×0.5=1 and (0−2)×(−0.5)=1. Variance is zero in this special example. Larger tasks usually do not permit such a perfect scalar cancellation.

<a id="section-26-6"></a>

## 26.6 Treat normalization as an algorithmic choice

Subtracting a batch mean and dividing by a batch standard deviation can improve numerical conditioning. It also creates sample-dependent weights and changes finite-batch behavior. With a single trajectory or a constant-return batch, careless normalization can erase the signal or divide by zero.

Report whether returns or advantages are normalized, over which axes, and with what small denominator offset. Do not describe normalization as a theorem that always preserves the exact finite-sample estimator.

<a id="section-26-7"></a>

## 26.7 Keep the collection policy fixed within a batch

Gather trajectories under one policy snapshot, compute the corresponding log probabilities, then update. If log probabilities are recomputed after parameters change, the samples are no longer on-policy for the new parameters.

Repeatedly optimizing the same trajectories without correction is not vanilla REINFORCE. Increasing the number of epochs can improve the batch objective while moving the policy away from the distribution that made the estimator valid.

<a id="section-26-8"></a>

## 26.8 Diagnose sparse-reward failure

If every sampled trajectory receives zero reward and the baseline is zero, the batch supplies no return-based policy gradient. Increasing optimizer steps on that same batch cannot invent evidence of success.

Investigate exploration, curriculum, reward design, and episode length. Any shaping must be described separately from the task metric. For reasoning, a terminal verifier may make successful traces too rare; process feedback changes which intermediate behavior receives signal.

<a id="section-26-9"></a>

## 26.9 Laboratory investigation

Use [Notebook 05](../../notebooks/05_policy_gradient.ipynb). Compare no baseline and a learned state baseline at the same interaction budget, using several seeds. Estimate gradient variability on a frozen policy with repeated independent batches to isolate variance reduction from changing learning dynamics.

The comparison in [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) provides a common task context. Explain both final return and variability; one lucky seed is not evidence of an inherently better estimator.

<a id="section-26-10"></a>

## 26.10 Problems, worked answers, and reading

1. For rewards [1,0,2] and gamma=0.5, calculate all returns.
2. In the one-step variance example, what is the gradient estimate for each action with baseline 0?
3. Why is a zero-reward batch insufficient to learn a successful unseen sequence through this update alone?

<details><summary>Worked answers</summary>

1. [1.5,1,2]. For the discounted start-state objective, action-score terms additionally carry [1,0.5,0.25].
2. Rewarding action: 2. Other action: 0. Their expected value is 1.
3. All return weights are zero, so the sampled score terms have zero coefficients. Additional data or an altered source of training signal is needed.

</details>

Read Williams's [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696) and [Sutton and Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html).

---

[← Chapter 25](../25-policy-gradient-methods/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 27 →](../27-actor-critic-methods/README.md)
