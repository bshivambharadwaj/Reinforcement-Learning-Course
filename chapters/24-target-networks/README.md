# 24. Target Networks

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 23](../23-experience-replay/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 25 →](../25-policy-gradient-methods/README.md)

[Quick-read Topic 24](../../README.md#topic-24) · [Notation](../NOTATION.md)

- [24.1 Slow down the regression reference](#section-24-1)
- [24.2 Define hard synchronization](#section-24-2)
- [24.3 Define soft synchronization](#section-24-3)
- [24.4 Derive the lag of an exponential average](#section-24-4)
- [24.5 Understand what remains coupled](#section-24-5)
- [24.6 Separate four different reference objects](#section-24-6)
- [24.7 Avoid accidental parameter sharing](#section-24-7)
- [24.8 Design a synchronization ablation](#section-24-8)
- [24.9 Laboratory and broader interpretation](#section-24-9)
- [24.10 Problems, worked answers, and reading](#section-24-10)

---

<a id="section-24-1"></a>

## 24.1 Slow down the regression reference

A target network supplies continuation estimates that move more slowly than the online network. It reduces immediate feedback between the prediction being fitted and the target used to fit it.

**Prerequisites:** Chapters 18 and 22. **Targets:** implement hard and soft updates, analyze their lag, and distinguish stabilization from a convergence theorem.

A courier revises its route estimates using a periodically published map instead of a map that redraws itself after every individual observation. The published map can stabilize planning while also becoming stale.

<a id="section-24-2"></a>

## 24.2 Define hard synchronization

Maintain online parameters theta and target parameters theta_bar. Initialize the target as a copy. Every C optimizer steps, copy theta into theta_bar. Between copies, target parameters remain fixed.

The unit matters: C environment steps and C optimizer steps differ when multiple updates occur per interaction. Document both the synchronization interval and the update-to-data ratio so another learner can reproduce the actual target age.

<a id="section-24-3"></a>

## 24.3 Define soft synchronization

A Polyak-style update uses theta_bar←(1−tau)theta_bar+tau theta, with 0<tau≤1. Here tau is the weight on the new online parameters; some implementations use the complementary coefficient, so read the formula rather than relying on a parameter name.

With tau=0.1, old target 2, and online value 6 for one scalar parameter, the updated target is 2.4. This is parameter averaging, not a direct average of Q predictions in a nonlinear network.

<a id="section-24-4"></a>

## 24.4 Derive the lag of an exponential average

If the online parameter stays fixed at theta_star, after k updates the target error is (1−tau)^k(theta_bar_0−theta_star). The half-life is log(0.5)/log(1−tau) update steps.

For tau=0.1, the half-life is about 6.58 updates. For small tau, it is approximately 0.693/tau. If the online parameters keep changing, this fixed-target calculation describes smoothing scale rather than the full tracking error.

<a id="section-24-5"></a>

## 24.5 Understand what remains coupled

Even with a frozen target network, batches share data, online parameters generalize across states, and the behavior policy changes future replay content. At synchronization, the target can shift abruptly.

Target networks reduce one source of rapid movement. They do not guarantee contraction of the neural training process, eliminate off-policy instability, or prevent reward-model exploitation. A stable loss curve can still represent a poor policy.

<a id="section-24-6"></a>

## 24.6 Separate four different reference objects

| Object | Purpose | Typical update |
|---|---|---|
| DQN target network | Bootstrap regression values | Periodic or smoothed copy |
| PPO old policy | Record the rollout-generating distribution | Snapshot each rollout cycle |
| Alignment reference policy | Anchor a preference/KL objective | Often fixed during a stage |
| Best checkpoint | Preserve an evaluated candidate | Selection by validation rule |

These objects cannot be interchanged merely because each is a copy of a model. Their mathematical roles determine when copying is correct.

<a id="section-24-7"></a>

## 24.7 Avoid accidental parameter sharing

A target must have independent parameter storage. Assigning a second variable to the same module does not create a frozen copy. Disable target gradients and update it only through the documented synchronization mechanism.

Also consider non-parameter state such as normalization statistics. A network in training mode can change buffers even when gradients are disabled. Verify which state is copied and whether stochastic layers are intended in target computation.

<a id="section-24-8"></a>

## 24.8 Design a synchronization ablation

Compare several C or tau values under identical environment and optimizer budgets. Log target-online prediction differences on a fixed probe set, TD errors, and independent returns.

Very frequent copying can increase feedback; very slow copying can make targets obsolete. These are hypotheses to test, not guaranteed monotonic effects. Report the full curve and seed spread instead of selecting only the best interval after seeing test results.

<a id="section-24-9"></a>

## 24.9 Laboratory and broader interpretation

[Notebook 04](../../notebooks/04_dqn.ipynb) is the direct target-network lab. Add a probe-state plot as an extension: online Q and target Q over training at the same states.

In modern post-training, frozen models also provide baselines and reference likelihoods, but Chapter 38's reference defines the objective itself. Changing it changes more than target smoothness. This distinction prevents a common confusion between DQN stabilization and alignment regularization.

<a id="section-24-10"></a>

## 24.10 Problems, worked answers, and reading

1. Starting at target 0 with a fixed online parameter 8 and tau=0.25, calculate two updates.
2. If C=100 optimizer steps and there are four updates per environment step, what is the approximate interaction interval?
3. Why can no-gradient mode alone fail to freeze all model state?

<details><summary>Worked answers</summary>

1. First target 2; second 0.75×2+0.25×8=3.5.
2. About 25 environment steps, assuming that update ratio is constant.
3. Stateful layers may update buffers during forward passes, and shared storage may be modified by another optimizer. Parameter gradients are only one mutation path.

</details>

Read [DQN](https://doi.org/10.1038/nature14236) for hard targets and [DDPG](https://arxiv.org/abs/1509.02971) for an influential use of soft target updates.

---

[← Chapter 23](../23-experience-replay/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 25 →](../25-policy-gradient-methods/README.md)
