# 32. Entropy Regularization

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 31](../31-proximal-policy-optimization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 33 →](../33-offline-reinforcement-learning/README.md)

[Quick-read Topic 32](../../quick-read/32-entropy-regularization.md) · [Notation](../NOTATION.md)

- [32.1 Reward a distribution, not only an action](#section-32-1)
- [32.2 Define discrete entropy](#section-32-2)
- [32.3 Solve a one-state regularized choice](#section-32-3)
- [32.4 Calculate the temperature effect](#section-32-4)
- [32.5 Distinguish entropy from reference KL](#section-32-5)
- [32.6 Derive the reference-regularized optimum](#section-32-6)
- [32.7 Recognize continuous-action differences](#section-32-7)
- [32.8 Separate diversity from safe exploration](#section-32-8)
- [32.9 Laboratory and modern connection](#section-32-9)
- [32.10 Problems, worked answers, and reading](#section-32-10)

---

<a id="section-32-1"></a>

## 32.1 Reward a distribution, not only an action

Entropy regularization encourages a policy to retain multiple plausible actions. In maximum-entropy RL, entropy is part of the objective; in other algorithms, an entropy bonus is a practical addition to the actor loss.

**Prerequisites:** Chapters 5 and 25–31. **Targets:** derive a soft optimal policy, distinguish entropy from reference KL, and analyze the effects of coefficient scale.

The courier keeps alternative routes available instead of committing immediately to one uncertain estimate. Diversity can help exploration, but random behavior is not automatically good behavior.

<a id="section-32-2"></a>

## 32.2 Define discrete entropy

For a finite action distribution, H(pi(.|s))=−sum_a pi(a|s)log pi(a|s), with 0 log 0 interpreted as zero. Using natural logarithms measures entropy in nats.

Entropy is maximized by the uniform distribution over legal actions, with value log A. Action masks change the maximum by changing A. Comparing raw entropy across states with different legal-action counts therefore needs care.

<a id="section-32-3"></a>

## 32.3 Solve a one-state regularized choice

Maximize sum_a pi(a)Q(a)+alpha H(pi), subject to probabilities summing to one and alpha>0. Differentiating the Lagrangian gives Q(a)−alpha(log pi(a)+1)+lambda=0.

Rearranging and normalizing yields pi(a)=exp(Q(a)/alpha)/sum_b exp(Q(b)/alpha). The optimized value is alpha log sum_a exp(Q(a)/alpha). This derivation assumes a finite action set and unconstrained probability choices beyond normalization.

### Bound the soft maximum's approximation error

Let q_max=max_a Q(a). Factor exp(q_max/alpha) from the log-sum-exp. Since the remaining sum lies between 1 and A, q_max≤alpha log sum_a exp(Q(a)/alpha)≤q_max+alpha log A.

The bound shows how temperature and action count control the gap between a hard maximum and the entropy-regularized soft value. A larger action set can increase the bonus even if the best task value is unchanged. With variable legal-action sets, this can influence preferences among states when entropy is accumulated over time.

<a id="section-32-4"></a>

## 32.4 Calculate the temperature effect

For Q=[2,0] and alpha=1, the better action receives exp(2)/(exp(2)+1), approximately 0.8808. With alpha=2, it receives exp(1)/(exp(1)+1), approximately 0.7311.

Larger alpha makes the distribution flatter. As alpha approaches zero, mass concentrates on maximizers, with ties requiring interpretation. The regularized optimum can sacrifice expected task reward in exchange for entropy because that is what its objective requests.

<a id="section-32-5"></a>

## 32.5 Distinguish entropy from reference KL

For a uniform reference u over A actions, D_KL(pi||u)=log A−H(pi). Thus maximizing entropy is equivalent to minimizing KL to uniform up to a constant at a fixed action set.

For a nonuniform reference pi_ref, the KL penalty also contains −E_pi[log pi_ref(a)]. It favors actions the reference considers likely, not merely a spread-out distribution. In LLM alignment, the reference encodes learned language behavior and is far from uniform.

<a id="section-32-6"></a>

## 32.6 Derive the reference-regularized optimum

For max_pi E_pi[Q]−beta D_KL(pi||pi_ref), the analogous Lagrange calculation gives pi_star(a) proportional to pi_ref(a)exp(Q(a)/beta), assuming adequate reference support.

An action with zero reference probability cannot receive positive mass at finite KL under this direction. This is a support restriction, not just a numerical issue. Chapter 38 uses this form to connect reward optimization with preference learning.

### Calculate a nonuniform-reference optimum

Let Q=[2,0], reference probabilities [0.1,0.9], and beta=1. The regularized optimum is proportional to [0.1exp(2),0.9], giving probability about 0.4509 to the higher-reward action. The strong reference preference can outweigh the task-value advantage at this coefficient.

With a uniform reference the same Q and coefficient give probability about 0.8808. This numerical contrast shows that reference-KL regularization retains information from a particular prior policy, while entropy alone favors uniformity. Changing the reference changes the objective even when beta is unchanged.

<a id="section-32-7"></a>

## 32.7 Recognize continuous-action differences

Differential entropy depends on coordinates and can be negative. A change of units or an action transformation changes its value through a Jacobian term. Squashed Gaussian policies need the transformed log density for correct entropy-related calculations.

The discrete bound 0≤H≤log A does not apply to arbitrary continuous distributions. Avoid carrying that intuition into a continuous-action implementation without specifying its density and support.

<a id="section-32-8"></a>

## 32.8 Separate diversity from safe exploration

High entropy can spread probability onto harmful or irrelevant actions. Low entropy can be appropriate once evidence strongly favors one action. An entropy curve should be interpreted with action quality, task stage, and constraints.

Reward scaling also matters: multiplying all rewards by ten while holding alpha fixed weakens regularization relative to task reward. Report coefficients together with reward units and normalization conventions.

### Distinguish state-wise diversity from useful trajectory coverage

A policy can have high entropy at many irrelevant states yet almost never visit a crucial branch. Conversely, low entropy at routine states can coexist with deliberate exploration at an uncertain junction.

Measure visitation and task-relevant action diversity alongside mean entropy. If an entropy bonus increases random movement but decreases completed deliveries, it may be optimizing its added objective without improving useful data collection. A coefficient schedule is a design choice whose effects need evidence; decreasing entropy over training is not a universal success criterion.

<a id="section-32-9"></a>

## 32.9 Laboratory and modern connection

Use [Notebook 06](../../notebooks/06_ppo.ipynb) to vary the entropy coefficient while holding other settings fixed. Compare return, entropy, and action coverage; predict whether diversity persists after useful behavior has been learned.

[Notebook 07](../../notebooks/07_preference_and_grpo.ipynb) helps contrast a reference-anchored preference objective with generic action diversity. A full soft actor–critic implementation is outside the shipped notebook set.

<a id="section-32-10"></a>

## 32.10 Problems, worked answers, and reading

1. What is the maximum entropy for four legal actions?
2. With equal Q values and reference probabilities [0.9,0.1], what policy maximizes the reference-KL objective?
3. Why does doubling rewards change behavior if alpha stays fixed?

<details><summary>Worked answers</summary>

1. log 4, approximately 1.3863 nats.
2. The reference itself: [0.9,0.1]. Equal exponential reward factors cancel during normalization.
3. The reward-to-regularization ratio increases, making the objective favor higher-value actions more sharply. Scaling alpha equally would preserve the one-state softmax ratios.

</details>

Read [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) and the regularized-policy derivation in [DPO](https://arxiv.org/abs/2305.18290).

---

[← Chapter 31](../31-proximal-policy-optimization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 33 →](../33-offline-reinforcement-learning/README.md)
