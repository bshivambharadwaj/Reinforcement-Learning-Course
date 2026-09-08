# 5. Policies

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 4](../04-states-actions-rewards-and-transitions/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 6 →](../06-returns-and-discounting/README.md)

[Quick-read Topic 5](../../README.md#topic-5) · [Notation](../NOTATION.md)

- [5.1 Separate a decision rule from its optimizer](#section-5-1)
- [5.2 Conditional distributions and legal actions](#section-5-2)
- [5.3 Softmax makes a differentiable discrete policy](#section-5-3)
- [5.4 Work through a probability and gradient example](#section-5-4)
- [5.5 Behavior and target policies answer different questions](#section-5-5)
- [5.6 Old policy and reference policy have distinct roles](#section-5-6)
- [5.7 What deterministic optimal-policy results do and do not say](#section-5-7)
- [5.8 Lab: inspect distributions instead of only returns](#section-5-8)
- [5.9 Policies over tokens, tools, and trajectories](#section-5-9)
- [5.10 Problems, worked answers, and sources](#section-5-10)

---

<a id="section-5-1"></a>

## 5.1 Separate a decision rule from its optimizer

A policy is a conditional decision distribution. An optimizer changes the parameters of that distribution. Evaluation measures the behavior produced by fixed parameters. Keeping these roles separate prevents confusion between what the agent does now and what its training procedure is trying to learn.

**Prerequisites:** Chapters 1–4 and elementary probability. **Targets:** compare deterministic and stochastic policies, derive softmax score derivatives, reason about policy support, and distinguish behavior, target, old, and reference policies.

A route is one realized sequence. A routing policy must also say what to do after a blocked road or unexpected battery reading. The same policy can produce different routes because of its own random choices, environment randomness, or different starting conditions.

<a id="section-5-2"></a>

## 5.2 Conditional distributions and legal actions

For each state s, pi(a|s) is nonnegative and sums to one over legal actions. In a deterministic policy all mass falls on one action. In a finite-horizon problem, the rule may depend on remaining time, either explicitly through pi_t or through time-augmented state.

With hidden state, write pi(a|h) for a policy conditioned on observation history h. Calling an observation-only rule pi(a|s) is a notational convenience that must not erase the underlying partial-observation assumption.

Action masks change the distribution. If a softmax assigns mass to three tools and one becomes unavailable, normalize over the remaining legal actions before sampling and logging probabilities. Keeping an unmasked probability in the loss describes a different policy from the one that actually generated the data.

<a id="section-5-3"></a>

## 5.3 Softmax makes a differentiable discrete policy

Given logits z_a, define pi_a=exp(z_a)/sum_b exp(z_b). Adding the same constant to every logit does not change pi, so logits are not uniquely identified. For numerical stability subtract the maximum logit before exponentiating or use a stable library implementation.

The log probability is z_a−log(sum_b exp(z_b)). Differentiating with respect to z_j gives:

```text
d log pi(a) / d z_j = indicator(j=a) − pi(j)
```

This derivative is useful because increasing the sampled action's log probability necessarily redistributes probability mass. There is no independent knob that increases every action probability simultaneously. The score vector sums to zero across its logit components, consistent with the invariance to a common shift.

<a id="section-5-4"></a>

## 5.4 Work through a probability and gradient example

For two logits [log 3, 0], the policy probabilities are [0.75, 0.25]. If action 0 is sampled, its log-score gradient is [0.25, −0.25]. A positive advantage pushes its logit up relative to the alternative; a negative advantage reverses that direction.

For logits [0,0], both actions have probability 0.5. Replacing sampling by argmax with a fixed tie rule always selects one action. That deployment rule is deterministic even though the trained distribution is uniform.

This illustrates why greedy evaluation and stochastic evaluation can disagree. Neither is inherently the correct metric for every application. State the intended deployment mode and measure it; if both are reported, label them separately.

<a id="section-5-5"></a>

## 5.5 Behavior and target policies answer different questions

The behavior policy mu generates data. A target policy pi is the policy whose value or improvement an algorithm concerns. On-policy methods align these roles in the relevant estimator; off-policy methods handle a difference between them under appropriate assumptions or objectives.

At a state, if mu chooses an action with probability 0.5 and pi chooses it with probability 0.8, its one-step importance ratio is 1.6. If mu never chooses an action to which pi assigns positive mass, ordinary importance weighting cannot recover that action's unseen outcomes: the support condition fails.

Ratios can also have high variance even when support exists. A tiny behavior probability creates a very large weight. Off-policy does not mean that arbitrary data can be safely reused without checking what the estimator requires.

<a id="section-5-6"></a>

## 5.6 Old policy and reference policy have distinct roles

In PPO, the old policy is the collector of the current rollout. Its cached action log probabilities define the denominator of the update ratio. They stay fixed during reuse of that batch and are refreshed when a new rollout is collected.

In regularized post-training, a reference policy is often a frozen starting model. It defines the behavior that a KL penalty discourages departing from. It may remain fixed for many PPO iterations while the old policy changes repeatedly.

In fixed-pair DPO, the reference enters completion-likelihood comparisons. These uses share the notion of a distribution over actions or responses but serve different mathematical purposes. Naming every frozen model `old_model` invites mistakes that shape checks cannot detect.

<a id="section-5-7"></a>

## 5.7 What deterministic optimal-policy results do and do not say

For a finite discounted fully observed MDP with ordinary expected return, a deterministic stationary optimal policy exists. This is an existence statement under those conditions, not a claim that stochastic policies are useless during learning.

Exploration, partial observation, games, constrained objectives, and entropy-regularized objectives can make stochastic rules practically or mathematically important. Finite-horizon tasks may need time-dependent rules unless time is included in the state. In a symmetric adversarial game, predictable deterministic behavior can be exploitable.

> **Common Misconception:** because a deterministic optimum exists for one formal setting, training should always immediately collapse to a single action. Existence does not supply the information needed to identify that action from limited experience.

<a id="section-5-8"></a>

## 5.8 Lab: inspect distributions instead of only returns

In [Notebook 05](../../notebooks/05_policy_gradient.ipynb), inspect action probabilities before and after a policy update. In [Notebook 06](../../notebooks/06_ppo.ipynb), locate the cached old log probabilities and explain why gradients must not update them during an epoch.

Add a diagnostic that prints probabilities for the same fixed observation at each training checkpoint. Compare entropy and task performance. Then evaluate both sampling and greedy action selection using frozen parameters. A change in the diagnostic state is not a global measure of policy drift; include several representative states.

For a controlled test, mask one action and verify that stored log probabilities match the normalized legal-action distribution. Do not deliberately deploy invalid actions to test a mask when a tiny simulator can test the same invariant.

<a id="section-5-9"></a>

## 5.9 Policies over tokens, tools, and trajectories

A causal language model supplies next-token probabilities. A complete response probability is a product of conditional token probabilities; the log probability is a sum. A tool-selection policy may instead choose among a small structured vocabulary. Both are policies, but their action spaces and training costs differ.

A sequence-level preference objective can act on the summed log probability without directly providing a label for each token. That does not mean each token was equally responsible for the preference. The probability factorization and the credit-assignment problem are separate ideas.

For continuous robot actions, a policy might output a Gaussian distribution. Clipping sampled values at action bounds changes the executed distribution; transformations such as squashing require their own probability accounting. We return to this representation issue in policy gradients.

<a id="section-5-10"></a>

## 5.10 Problems, worked answers, and sources

**A.** For probabilities [0.2,0.5,0.3], give the log-score gradient when action 1 is selected. **B.** If mu(a|s)=0.1 and pi(a|s)=0.4, calculate the ratio and explain its support assumption. **C.** Explain why updating a reference model every PPO minibatch changes the intended regularized objective.

<details><summary>Worked answers</summary>

A: [−0.2,0.5,−0.3]. B: the ratio is 4; target-positive actions must have positive behavior probability for ordinary importance weighting. C: the penalty would measure distance to a moving distribution instead of the designated fixed anchor. PPO's old-policy denominator has a separate role and also must remain fixed for the current batch.

</details>

**Read:** [Sutton and Barto, Chapters 3 and 13](http://incompleteideas.net/book/the-book-2nd.html); [PPO](https://arxiv.org/abs/1707.06347); [DPO](https://arxiv.org/abs/2305.18290).

---

[← Chapter 4](../04-states-actions-rewards-and-transitions/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 6 →](../06-returns-and-discounting/README.md)
