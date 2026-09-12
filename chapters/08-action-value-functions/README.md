# 8. Action-Value Functions

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 7](../07-state-value-functions/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 9 →](../09-advantage-functions/README.md)

[Quick-read Topic 8](../../quick-read/08-action-value-functions.md) · [Notation](../NOTATION.md)

- [8.1 Why condition on the first action?](#section-8-1)
- [8.2 Two equivalent decompositions](#section-8-2)
- [8.3 Calculate a stochastic action value](#section-8-3)
- [8.4 A matrix view exposes dimensions](#section-8-4)
- [8.5 Q_pi and Q_star are different objects](#section-8-5)
- [8.6 Greedy decisions and estimation error](#section-8-6)
- [8.7 Maximization turns noise into optimism](#section-8-7)
- [8.8 Lab: make the target policy visible](#section-8-8)
- [8.9 Action values for tool choices](#section-8-9)
- [8.10 Problems, worked answers, and sources](#section-8-10)

---

<a id="section-8-1"></a>

## 8.1 Why condition on the first action?

A state value averages over the policy's actions. To compare choices at a junction, condition on the first action explicitly. Q_pi(s,a) is the expected return from taking a in s and then following pi. Only the first action is forced; the continuation still belongs to pi.

**Prerequisites:** Chapters 5–7. **Targets:** derive the V/Q relationship, calculate a stochastic one-step action value, distinguish Q_pi from Q_star, and understand when maximizing a learned Q estimate is meaningful.

For the courier, Q answers “what happens if I take this road now and use my current routing rule afterward?” It is not necessarily the value of committing to that road forever or of acting optimally at every later junction.

<a id="section-8-2"></a>

## 8.2 Two equivalent decompositions

One form averages a next-state value after the first transition:

```text
Q_pi(s,a) = sum_(s',r) p(s',r|s,a) [r + gamma V_pi(s')]
V_pi(s)   = sum_a pi(a|s) Q_pi(s,a)
```

Substituting the second identity into the first gives the Bellman expectation equation for Q, with an additional average over the next action under pi. The distinction between forcing the current action and averaging future actions is essential.

Both identities assume compatible state, reward, policy, and horizon conventions. In a finite-horizon model, the continuation uses the next time index. In a true terminal next state the continuation is zero regardless of any arbitrary array entry assigned to it.

<a id="section-8-3"></a>

## 8.3 Calculate a stochastic action value

At a junction, action inspect costs −1. With probability 0.75 it leads to state B with V_pi(B)=8; otherwise it leads to C with V_pi(C)=0. With gamma=0.9, Q_pi(s,inspect)=−1+0.9×(0.75×8+0.25×0)=4.4.

Action leave ends the task for reward 3, so Q_pi(s,leave)=3. A policy choosing inspect with probability 0.4 has state value 0.4×4.4+0.6×3=3.56.

The policy average cannot exceed the largest of these exact Q_pi values or fall below the smallest. If code produces such a result, check action probabilities, tensor axes, or whether values from different policies were mixed.

<a id="section-8-4"></a>

## 8.4 A matrix view exposes dimensions

In a finite problem store Q as an array of shape states × actions. A deterministic greedy policy selects an argmax along the action axis, separately for every state. Taking a maximum over the whole matrix would select one state-action pair, not a policy.

When constructing targets, a batch of B states and A actions gives a B×A predicted array. Gathering selected actions should produce B values, not a B×B broadcasted matrix. Many deep-RL losses accept a broadcast silently, so assert target and prediction shapes before reducing.

Illegal actions also need consistent treatment. They should not win a maximum merely because their untrained values default to zero while legal actions have negative returns. Mask them during action selection and target construction according to the task contract.

<a id="section-8-5"></a>

## 8.5 Q_pi and Q_star are different objects

Q_star(s,a) takes action a and then uses an optimal continuation. Q_pi uses the designated pi. A greedy maximum of Q_pi compares first actions with that policy's continuation; it is not generally Q_star.

Suppose two routes lead to a later junction where the current policy makes a poor choice. Improving that later choice can increase the earlier action values too. One round of greedy action selection does not necessarily solve every downstream consequence in an arbitrary MDP.

Exact policy iteration alternates evaluation and improvement to address this dependency. Q-learning instead uses a greedy bootstrap target intended to learn optimal action values under suitable tabular sampling and step-size conditions. A symbol named `q` does not tell you which object it estimates.

### Work through a misleading early action ranking

At A, `stop` pays 2 and terminates; `continue` pays 0 and reaches B. At B, the current policy chooses an action paying 1, although another action pays 10. With gamma=0.9, Q_pi(A,continue)=0.9, so a first-action comparison under the old continuation favors stopping.

After improving the policy at B, continuing is worth 9 and becomes preferable. The difference is not a contradiction: Q_pi and Q_new condition on different future behavior. A one-pass greedy change using stale continuation values can miss an upstream improvement that becomes visible after reevaluation.

<a id="section-8-6"></a>

## 8.6 Greedy decisions and estimation error

Assume every estimated action value at a fixed state is within epsilon of the true Q values. Let a_hat maximize the estimates and a_star maximize the true values. Then the true value gap Q(a_star)−Q(a_hat) is at most 2epsilon: one epsilon can inflate the selected action and another can deflate the true best action.

This is a local ranking bound. It does not automatically bound total policy regret when the Q estimates concern a different continuation or errors alter future visitation. It does explain why small value errors can flip a near tie while leaving a large-margin decision unchanged.

Estimate uncertainty and action-value margins together. A nearly tied choice may be fragile even when average squared error is small.

### Prove the action-gap robustness criterion

Let the best true action have value q_1 and the runner-up q_2, with gap Delta=q_1−q_2>0. If every action estimate has absolute error at most epsilon, then the estimated best action's value is at least q_1−epsilon, while any competitor's is at most q_2+epsilon.

Thus Delta>2epsilon guarantees the true best action remains the unique estimated maximizer. When Delta=2epsilon, a tie can occur; when it is smaller, reversal is possible. For values [4,3] and epsilon=0.4, ranking is protected. At epsilon=0.6, estimates [3.4,3.6] reverse it.

This explains why value error alone does not fully describe control error: the same error magnitude can be harmless in a large-gap state and decisive near a tie.

<a id="section-8-7"></a>

## 8.7 Maximization turns noise into optimism

If several action estimates contain zero-mean noise, their maximum tends to select a favorable error. For an elementary illustration, two independent estimates each equal +1 or −1 with equal probability have mean zero individually. Their maximum equals −1 only when both are −1, so its expectation is 0.75×1+0.25×(−1)=0.5.

This does not require a biased estimator for each action. Selection creates the effect. When the selected maximum becomes a bootstrap target, optimism can propagate backward through value learning.

Double Q-style methods separate selection from evaluation to address this mechanism. They do not imply that all optimism is harmful or that the numerical example exactly describes correlated neural-network errors.

### Extend maximization bias to more actions

For m independent action estimates that are each +1 or −1 with equal probability around a true value of zero, the maximum is −1 only if all m estimates are −1. Therefore E[max]=1−2×2^(−m)=1−2^(1−m).

For m=2 the bias is 0.5; for m=4 it is 0.875. More candidates can increase optimistic selection even when each candidate estimator remains unbiased. Correlated errors change this calculation, so action count alone is not a complete bias model. The same controlled construction helps explain why searching more model-scored responses can exploit evaluator noise.

<a id="section-8-8"></a>

## 8.8 Lab: make the target policy visible

In [Notebook 03](../../notebooks/03_q_learning.ipynb), locate the SARSA and Q-learning bootstrap terms. For the same next-state Q row, calculate both targets using a sampled next action that differs from the greedy action. Explain which continuation each target describes.

In [Notebook 04](../../notebooks/04_dqn.ipynb), inspect the gather operation and terminal mask. Add a fixed toy batch with distinct action values so a wrong axis or broadcast produces an unmistakable result.

For a deeper extension, simulate the two-estimator maximization example and compare independent selection/evaluation with using the same estimate for both. Report the mean selected target, not just the fraction of correct argmax decisions.

<a id="section-8-9"></a>

## 8.9 Action values for tool choices

A tool-selection Q value can include the immediate call cost and the expected value of the information or result it returns. A search call that completes nothing immediately can still be worthwhile if its response enables a better later decision.

Large structured action spaces make explicit maximization difficult: choosing a tool and arbitrary text arguments is not a tiny fixed action table. Policy-based methods or hierarchical action models can be more practical, which motivates the transition from value methods to policy gradients.

The mathematical distinction remains useful even when Q is not explicitly trained: action value asks about one decision plus a continuation policy, while a terminal response score only evaluates an outcome already produced.

<a id="section-8-10"></a>

## 8.10 Problems, worked answers, and sources

**A.** In the inspect example, change V_pi(B) from 8 to 4. Which first action is better? **B.** If all action errors are at most 0.3, bound the one-step ranking loss. **C.** Why must an illegal action be masked in a negative-return task?

<details><summary>Worked answers</summary>

A: inspect becomes −1+0.9×0.75×4=1.7, below leave's 3. B: at most 0.6 under the fixed-state, same-Q-object assumptions. C: an untrained illegal action with zero output can appear better than all legal negative values. Selecting or bootstrapping from it would optimize a nonexistent action. Masking must affect both executed policy probabilities and any target maximum consistently.

</details>

**Read:** [Sutton and Barto, Chapters 3 and 6](http://incompleteideas.net/book/the-book-2nd.html); [Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

---

[← Chapter 7](../07-state-value-functions/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 9 →](../09-advantage-functions/README.md)
