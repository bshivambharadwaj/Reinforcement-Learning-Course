# 6. Returns and Discounting

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 5](../05-policies/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 7 →](../07-state-value-functions/README.md)

[Quick-read Topic 6](../../README.md#topic-6) · [Notation](../NOTATION.md)

- [6.1 Choose the objective before calculating it](#section-6-1)
- [6.2 Finite and infinite return definitions](#section-6-2)
- [6.3 Derive the recursive identity](#section-6-3)
- [6.4 Work through timing and truncation](#section-6-4)
- [6.5 Effective horizon and a quantitative tail bound](#section-6-5)
- [6.6 Return variability is not just reward variability](#section-6-6)
- [6.7 Distinguish discounting, risk, and reward shaping](#section-6-7)
- [6.8 Lab: isolate objective changes from estimator changes](#section-6-8)
- [6.9 Return in language-model post-training](#section-6-9)
- [6.10 Problems, worked answers, and sources](#section-6-10)

---

<a id="section-6-1"></a>

## 6.1 Choose the objective before calculating it

Return converts a sequence of rewards into a quantity a policy can optimize. Different conventions can prefer different policies even when the environment produces identical trajectories. Discounted return, finite-horizon total reward, and long-run average reward are related formulations, not interchangeable names.

**Prerequisites:** Chapters 2–5 and geometric series. **Targets:** derive the recursive return identity, quantify discounting's effective horizon and variance in a simple model, distinguish terminal and truncated returns, and explain the consequences of reward rescaling.

The delivery analogy is a contract that specifies whether earlier delivery is worth more, whether every trip ends by a deadline, and whether the objective concerns one trip or a continuing fleet. Gamma belongs to this contract, not merely to a list of optimizer settings.

<a id="section-6-2"></a>

## 6.2 Finite and infinite return definitions

For an episode ending at T, define G_t as R_(t+1)+gamma R_(t+2)+...+gamma^(T−t−1)R_T. The terminal continuation G_T is zero. For a continuing discounted task extend the sum indefinitely.

If rewards are bounded in absolute value by R_max and 0≤gamma<1, absolute return is bounded by R_max/(1−gamma). This makes the discounted infinite sum well defined. At gamma=1, bounded individual rewards alone do not bound an infinite sum.

A finite horizon permits gamma=1 because only finitely many terms appear. An average-reward objective instead considers reward per unit time under suitable limiting conditions. It can prefer policies differently from a particular discounted objective, especially when transient behavior matters.

<a id="section-6-3"></a>

## 6.3 Derive the recursive identity

Separate the first term from the discounted sum and factor gamma from the rest:

```text
G_t = R_(t+1) + gamma [R_(t+2) + gamma R_(t+3) + ...]
    = R_(t+1) + gamma G_(t+1)
```

This is an identity about the realized reward sequence, before taking any expectation. Bellman equations later condition its expectation on a state or state-action pair. Monte Carlo uses observed G_t; TD replaces the unknown continuation with an estimate.

Compute returns backward through a finite episode using a running accumulator initialized to zero at true termination. This costs linear time in episode length. Starting the accumulator from a value estimate at an external truncation yields a bootstrapped target, not a fully observed Monte Carlo return.

<a id="section-6-4"></a>

## 6.4 Work through timing and truncation

For rewards [−1,−1,+10] and gamma=0.9, the returns are G_2=10, G_1=8, and G_0=6.2. Notice that a later return can be larger because earlier costs are excluded. The first reward of each G_t receives no discount.

Suppose collection stops after the first two rewards while the task continues. With endpoint value estimate 10, the truncated target from the start is −1−0.9+0.81×10=6.2. It matches the complete return in this deterministic example because the endpoint estimate happens to be exact.

If the endpoint value is 7, the target becomes 3.77. This difference is bootstrap error, not arithmetic error. If the task actually terminated after the two −1 rewards, the correct return is −1.9 with no endpoint value at all.

<a id="section-6-5"></a>

## 6.5 Effective horizon and a quantitative tail bound

The total discounted weight is 1/(1−gamma), often used as an informal effective-horizon scale. It is not a hard cutoff. Rewards arbitrarily far away retain nonzero weight when gamma>0.

If rewards are bounded by R_max, ignoring all rewards after H steps creates an absolute tail error at most R_max gamma^H/(1−gamma). This follows by bounding each discarded term and summing the remaining geometric series. To choose H for an error tolerance, solve this inequality rather than treating 1/(1−gamma) as exact episode length.

For gamma=0.9, R_max=1, and H=50, the bound is about 0.0515. It is conservative because it assumes every future reward has the worst possible magnitude and aligned sign. Higher gamma makes both long-term effects and error propagation more significant.

<a id="section-6-6"></a>

## 6.6 Return variability is not just reward variability

As a deliberately simplified model, suppose continuing rewards are independent with common variance sigma². Then the variance of discounted return is sigma² sum_k gamma^(2k)=sigma²/(1−gamma²). The squared discounts appear because variance scales quadratically under multiplication.

Real RL rewards are usually correlated through states and actions, so covariance terms also contribute. A policy can change both the mean reward sequence and its correlation structure. Do not use the independent-reward formula as a general RL variance estimator.

Higher return variance can make Monte Carlo estimation expensive. Lowering gamma can reduce some variance while changing the objective. That is not a free statistical improvement when the intended task values distant consequences.

<a id="section-6-7"></a>

## 6.7 Distinguish discounting, risk, and reward shaping

Discounting weights time, while risk-sensitive objectives depend on distributions beyond the mean. A policy with rare catastrophic outcomes can still have high expected discounted return. A constraint on failure probability or a different risk objective adds information that gamma alone does not express.

Likewise, reward shaping changes feedback timing or incentives depending on its form. Potential-based shaping has specific invariance conditions; an arbitrary bonus for intermediate events may change which policy is optimal. Chapter 4 derives the telescoping case.

> **Failure Mode:** tuning gamma until training becomes easy can quietly redefine the task. Report its value and explain whether it encodes the desired objective or is an approximation made for learning stability.

<a id="section-6-8"></a>

## 6.8 Lab: isolate objective changes from estimator changes

Use the return-to-go function in [Notebook 05](../../notebooks/05_policy_gradient.ipynb). Check the sequence [−1,−1,10] by hand before training. Add tests for an empty continuation, a one-step episode, and gamma=1 on a finite reward sequence.

Then compare two gamma values on the same saved trajectories. This isolates changed weighting from changed exploration. A second experiment can retrain policies under each objective, but the comparison now includes both changed targets and changed data collection.

For the fixed-horizon [comparison lab](../../notebooks/08_one_problem_many_algorithms.ipynb), explain why gamma=1 is well defined and why all learners receive the same number of transitions. Do not import its return scale into another environment without checking reward conventions.

<a id="section-6-9"></a>

## 6.9 Return in language-model post-training

A terminal answer score can be combined with per-token regularization costs. If gamma=1, all valid generated tokens can receive credit from the same terminal outcome, while intermediate costs contribute from their respective positions onward. Prompt and padding tokens are not sampled completion actions.

Length-dependent costs and EOS handling can strongly affect the resulting return. If EOS receives an unintended extra reward, or padding is counted as generated length, the learner may exploit formatting rather than improve task quality.

Outcome return also does not label the correctness of individual reasoning steps. A correct answer can follow an invalid trace. The course's reasoning lab explicitly separates final-answer and intermediate-step evaluation to make this limitation visible.

<a id="section-6-10"></a>

## 6.10 Problems, worked answers, and sources

**A.** Calculate every return for [2,0,−1] with gamma=0.5. **B.** Bound the ignored tail after H=10 with R_max=2 and gamma=0.8. **C.** Why is lowering gamma not merely a variance-reduction technique?

<details><summary>Worked answers</summary>

A: G_2=−1, G_1=−0.5, G_0=1.75. B: the bound is 2×0.8^10/0.2≈1.07374. C: it changes the relative weight of distant rewards and can change the optimal policy. Any reduction in estimator variance must be evaluated alongside that changed objective. Under independent unit-variance rewards, gamma=0.5 gives return variance 4/3, but correlated task rewards need covariance terms too.

</details>

**Read:** [Sutton and Barto, Chapters 3 and 12](http://incompleteideas.net/book/the-book-2nd.html). Continue to Chapter 7 to take conditional expectations of return.

---

[← Chapter 5](../05-policies/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 7 →](../07-state-value-functions/README.md)
