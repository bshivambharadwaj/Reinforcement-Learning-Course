# 27. Actor–Critic Methods

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 26](../26-reinforce/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 28 →](../28-advantage-actor-critic/README.md)

[Quick-read Topic 27](../../quick-read/27-actor-critic-methods.md) · [Notation](../NOTATION.md)

- [27.1 Couple a decision-maker with an evaluator](#section-27-1)
- [27.2 Define a one-step advantage estimate](#section-27-2)
- [27.3 Calculate both updates](#section-27-3)
- [27.4 Decompose critic error in the advantage](#section-27-4)
- [27.5 Choose separate losses](#section-27-5)
- [27.6 Understand timescale intuition](#section-27-6)
- [27.7 Distinguish on-policy and off-policy variants](#section-27-7)
- [27.8 Diagnose actor and critic separately](#section-27-8)
- [27.9 Laboratory and modern connection](#section-27-9)
- [27.10 Problems, worked answers, and reading](#section-27-10)

---

<p align="center">
  <img src="../../assets/diagrams/actor-critic.svg" width="760" alt="The actor chooses actions and the critic supplies a learning signal." />
</p>

*The actor chooses actions and the critic supplies a learning signal.*

<a id="section-27-1"></a>

## 27.1 Couple a decision-maker with an evaluator

An actor chooses actions through pi_theta. A critic estimates values used to judge those actions. Their learning processes interact: the actor changes which data arrive, and critic errors influence the actor's updates.

**Prerequisites:** Chapters 9, 18, and 25–26. **Targets:** derive a one-step actor–critic update, identify bootstrap bias, and separate actor and critic optimization.

The courier has a route-selection rule and an internal travel-value estimator. Improving the estimator helps only if it evaluates the policy and task the courier actually follows.

<a id="section-27-2"></a>

## 27.2 Define a one-step advantage estimate

For a state-value critic, delta=r+gamma m V_w(s')−V_w(s), where m masks true termination. If V_w=V_pi exactly, E[delta|s,a]=A_pi(s,a).

Use delta as a detached weight in an actor score update and as an error signal for a semi-gradient critic update. With an inaccurate critic, its conditional expectation differs from the true advantage. The estimator is convenient, not automatically unbiased at every training step.

<a id="section-27-3"></a>

## 27.3 Calculate both updates

Suppose r=1, gamma=0.9, V(s)=2, and V(s')=3. Then delta=1.7. The sampled action should increase in probability under an ascent step because the observed target exceeds the current baseline.

With a tabular critic step size 0.1, V(s) becomes 2.17. If the sampled action has probability p=0.25 in a binary logistic policy, its logit score is 0.75 and its unscaled actor direction is 1.275. Discounted-objective timestep weights may also apply.

<a id="section-27-4"></a>

## 27.4 Decompose critic error in the advantage

Write V_w=V_pi+e. For a nonterminal transition:

```text
E[delta | s,a] = A_pi(s,a)
                + gamma E[e(S') | s,a] − e(s)
```

The state-only term −e(s) cancels in the expected action score at s, but successor error can depend on a and bias the actor direction. This explains why “a baseline may be inaccurate” does not justify every bootstrapped advantage estimate.

### Construct an actor bias from successor-value error

Two actions at s both pay zero and lead deterministically to different terminal-on-next-step states with the same true continuation value 5. Their true advantages under a uniform policy are zero. Suppose the critic estimates those successors as 7 and 3.

With gamma=0.9, the difference between their expected TD-based actor weights is 0.9×(7−3)=3.6, despite equal true action values. A state-only baseline at s cannot cancel that action-dependent difference. The actor can therefore prefer one action due entirely to the critic's uneven successor errors.

<a id="section-27-5"></a>

## 27.5 Choose separate losses

The actor minimizes negative log probability times detached advantage. The critic minimizes a regression loss toward a detached return target. Entropy can be added to the actor objective with an explicitly reported coefficient.

If actor and critic share an encoder, the critic loss also changes policy features. Loss coefficients then affect policy behavior indirectly. Separate optimizers do not by themselves isolate parameters that are intentionally shared.

### Separate update paths in a shared network

Consider a shared feature encoder with a policy head and value head. The actor loss should send gradients through the policy log probability and shared encoder; the detached advantage should not create a second actor gradient through the value head. The critic loss can independently send gradients through the value head and shared encoder.

Inspect gradient norms by parameter group after each loss separately. A zero policy-head gradient from the critic loss is expected with separate heads, while a nonzero shared-encoder gradient is intentional. This diagnostic makes shared representation interference concrete instead of attributing every effect to an abstract “unstable critic.”

<a id="section-27-6"></a>

## 27.6 Understand timescale intuition

The critic ideally tracks the actor's current value function sufficiently well for useful updates. A rapidly moving actor can outrun its critic; an overly aggressive critic can destabilize shared representations.

Classical two-timescale convergence arguments require specific conditions on models and step sizes. A practical choice of two learning rates does not establish those conditions for a deep network. Use the intuition to design diagnostics, not to claim a theorem that has not been shown.

<a id="section-27-7"></a>

## 27.7 Distinguish on-policy and off-policy variants

On-policy actor–critic learns from the current actor's trajectories. Off-policy actor–critic introduces additional machinery to handle another data distribution; deterministic policy gradients and importance-corrected methods have their own objectives and assumptions.

Do not mix an arbitrary replay buffer into an on-policy implementation merely because the critic resembles TD learning. The actor expectation, not just the critic target, must match the intended method.

<a id="section-27-8"></a>

## 27.8 Diagnose actor and critic separately

Track return, entropy, advantage statistics, critic error on held-out rollouts, and policy change. A critic's explained variance can help describe fit but is unstable when target variance is tiny and does not certify unbiased actor updates.

Probe a known state-action pair with an analytical advantage. If the actor moves in the wrong direction, inspect loss signs and masks before tuning learning rates. If signs are correct but returns worsen, investigate approximation and distribution change.

### Build a frozen-actor critic audit

Freeze the policy and gather independent evaluation trajectories. Fit or evaluate the critic against empirical returns on these trajectories, then compare its state-wise errors with the actor's subsequent update weights.

If critic accuracy deteriorates mainly after actor updates resume, policy nonstationarity is a plausible explanation. If it remains poor under a frozen actor, investigate representation, coverage, target boundaries, and optimization. This controlled separation is more informative than changing both actor and critic learning rates simultaneously.

<a id="section-27-9"></a>

## 27.9 Laboratory and modern connection

[Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) includes actor–critic in the common environment. [Notebook 06](../../notebooks/06_ppo.ipynb) develops a richer actor–critic training loop.

In LLM post-training, a value head predicts expected future reward from a response prefix. A reward model scores an outcome or preference; it is not automatically a value function. Conflating these two roles can produce an invalid advantage target.

<a id="section-27-10"></a>

## 27.10 Problems, worked answers, and reading

1. Recompute delta in the example if the transition terminates.
2. If e(s)=2 and every successor has e(s')=2 with gamma=0.9, what is the conditional delta error?
3. Why does action-independent baseline cancellation not remove arbitrary successor-value error?

<details><summary>Worked answers</summary>

1. delta=1−2=−1; the continuation value is excluded.
2. 0.9×2−2=−0.2. This constant conditional shift cancels in the expected action score at that state.
3. Different actions can lead to successors with different errors, making the remaining term action-dependent and therefore capable of changing the expected gradient.

</details>

Read [Sutton and Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html) and [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

---

[← Chapter 26](../26-reinforce/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 28 →](../28-advantage-actor-critic/README.md)
