# 25. Policy Gradient Methods

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 24](../24-target-networks/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 26 →](../26-reinforce/README.md)

[Quick-read Topic 25](../../README.md#topic-25) · [Notation](../NOTATION.md)

- [25.1 Optimize decisions directly](#section-25-1)
- [25.2 Define a finite-episode objective](#section-25-2)
- [25.3 Derive the likelihood-ratio identity](#section-25-3)
- [25.4 Use causality to replace the full return](#section-25-4)
- [25.5 Calculate a one-step gradient](#section-25-5)
- [25.6 Prove baseline cancellation](#section-25-6)
- [25.7 Distinguish gradient ascent from loss minimization](#section-25-7)
- [25.8 Know the limitations of the estimator](#section-25-8)
- [25.9 Laboratory and token policies](#section-25-9)
- [25.10 Problems, worked answers, and reading](#section-25-10)

---

<a id="section-25-1"></a>

## 25.1 Optimize decisions directly

Policy-gradient methods adjust a parameterized action distribution to increase expected return. They do not require differentiating the environment's transition function or taking an explicit maximum over all actions.

**Prerequisites:** Chapters 5–6 and 9; expectations and differentiation. **Targets:** derive the score-function estimator, explain reward-to-go and baselines, and keep the optimization objective explicit.

The courier changes the probabilities in its route-selection rule by comparing the outcomes of sampled routes. It need not know how traffic probabilities differentiate with respect to the policy parameters.

<a id="section-25-2"></a>

## 25.2 Define a finite-episode objective

For a trajectory tau sampled under pi_theta, let J(theta)=E[G_0]. Assume the environment and initial distribution do not depend on theta, and the policy distribution has suitable differentiability and support.

The trajectory probability factors into the initial distribution, policy action probabilities, and environment transitions. Only the policy factors contribute to the score derivative. If the environment itself depends on theta, those additional derivative terms cannot be discarded.

<a id="section-25-3"></a>

## 25.3 Derive the likelihood-ratio identity

```text
gradient J = sum_tau P_theta(tau) G_0(tau) gradient log P_theta(tau)
           = E[ G_0 * sum_t gradient log pi_theta(A_t|S_t) ]
```

The first line follows from gradient P=P gradient log P under conditions allowing differentiation through the expectation. It gives an unbiased Monte Carlo estimator for the stated objective, though its variance can be large.

<a id="section-25-4"></a>

## 25.4 Use causality to replace the full return

Rewards received before action A_t cannot be influenced by that action. Their expected product with its conditional score is zero. For J=E[sum_k gamma^k R_(k+1)], the reward-to-go expression is:

```text
gradient J = E[sum_t gamma^t G_t * gradient log pi(A_t|S_t)]
```

The outer gamma^t matters for this precise start-state discounted objective. Implementations sometimes use an undiscounted episodic objective or another state-weighting convention; document the convention rather than quietly dropping a factor in a proof.

<a id="section-25-5"></a>

## 25.5 Calculate a one-step gradient

Two terminal actions pay 4 and 0. Let p=sigmoid(z) be the probability of the first action. Then J=4p and dJ/dz=4p(1−p). At p=0.25, the exact derivative is 0.75.

The score for choosing the first action is 1−p=0.75; for the second it is −p=−0.25. Averaging reward times score gives 0.25×4×0.75+0.75×0×(−0.25)=0.75, matching direct differentiation.

<a id="section-25-6"></a>

## 25.6 Prove baseline cancellation

For an action-independent b(s), E_(a~pi)[b(s) gradient log pi(a|s)]=b(s) gradient sum_a pi(a|s)=0. Subtracting such a baseline preserves the expected score estimator.

The baseline should be treated as constant in the actor loss. If it depends on the sampled action, the cancellation generally fails. A baseline estimated from a batch that includes the same action outcome can introduce subtle finite-sample dependence; do not assume every normalization is exactly unbiased.

<a id="section-25-7"></a>

## 25.7 Distinguish gradient ascent from loss minimization

Optimizers usually minimize, so the sampled actor loss is the negative weighted log probability. Detach the return or advantage weight when it is intended as a score-function coefficient.

```text
actor_loss = −mean(log_probability_of_sampled_action * detached_weight)
```

Increasing probability for positive weights and decreasing it for negative weights is a useful one-sample sign check. For continuous policies, include the correct transformed density if actions are squashed or otherwise transformed.

<a id="section-25-8"></a>

## 25.8 Know the limitations of the estimator

An unbiased gradient estimate need not have low variance or produce monotonic improvement at a finite step size. Rare high returns can dominate; long trajectories add many noisy score terms; approximate critics can bias weights.

Fresh on-policy data are central to the basic derivation. Reusing old samples requires a justified correction or a different objective. A high training surrogate after repeated optimization on one batch does not certify better expected return.

<a id="section-25-9"></a>

## 25.9 Laboratory and token policies

[Notebook 05](../../notebooks/05_policy_gradient.ipynb) provides the policy-gradient lab. Verify the one-step analytical example before interpreting a longer-horizon learning curve.

A language model is a policy over tokens conditioned on a prefix. Sequence log probability is the sum of token log probabilities, so the same score identity applies. Prompt masks, response boundaries, sampling policy, and reward definition determine which terms actually belong in the estimator.

<a id="section-25-10"></a>

## 25.10 Problems, worked answers, and reading

1. In the one-step example at p=0.5, calculate the exact derivative.
2. At p=0.25, use baseline b=1 and verify the expected derivative remains 0.75.
3. Why does a baseline equal to the sampled reward generally destroy this estimator?

<details><summary>Worked answers</summary>

1. 4×0.5×0.5=1.
2. 0.25×(4−1)×0.75+0.75×(0−1)×(−0.25)=0.5625+0.1875=0.75.
3. Every weight becomes zero. This baseline depends on the sampled outcome/action and does not meet the action-independent cancellation condition.

</details>

Read [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) and [Sutton and Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html).

---

[← Chapter 24](../24-target-networks/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 26 →](../26-reinforce/README.md)
