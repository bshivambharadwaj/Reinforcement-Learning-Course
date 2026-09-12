# 28. Advantage Actor–Critic

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 27](../27-actor-critic-methods/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 29 →](../29-generalized-advantage-estimation/README.md)

[Quick-read Topic 28](../../quick-read/28-advantage-actor-critic.md) · [Notation](../NOTATION.md)

- [28.1 Build a rollout-based actor–critic learner](#section-28-1)
- [28.2 Define an n-step bootstrapped return](#section-28-2)
- [28.3 Work a three-step rollout](#section-28-3)
- [28.4 Define the combined loss](#section-28-4)
- [28.5 Handle multiple environments](#section-28-5)
- [28.6 Contrast A2C and A3C](#section-28-6)
- [28.7 Examine batch-size tradeoffs](#section-28-7)
- [28.8 Diagnose policy collapse](#section-28-8)
- [28.9 Laboratory investigation](#section-28-9)
- [28.10 Problems, worked answers, and reading](#section-28-10)

---

<a id="section-28-1"></a>

## 28.1 Build a rollout-based actor–critic learner

Advantage actor–critic combines policy updates with estimated advantages from short rollouts. Synchronous A2C gathers a batch before updating; asynchronous A3C uses workers whose updates can be based on different parameter versions.

**Prerequisites:** Chapters 26–27. **Targets:** construct n-step rollout targets, handle vectorized boundaries, and distinguish synchronous batching from asynchronous learning.

Several couriers can gather route experience at once. Their observations broaden a batch, but each journey still has its own termination and cutoff boundaries.

<a id="section-28-2"></a>

## 28.2 Define an n-step bootstrapped return

At a nonterminal rollout boundary, initialize the backward return with V(s_T). At a true terminal boundary, initialize it with zero. Then walk backward, applying G_t=r_t+gamma G_(t+1) until an episode boundary requires its own treatment.

The advantage estimate is G_t−V(s_t). Longer rollouts include more observed rewards and rely on a more distant bootstrap estimate. They also delay updates and can increase return variability.

### Derive how an endpoint error reaches earlier targets

If a nonterminal rollout's final value estimate has error e, its contribution to the n-step return target at the first decision is gamma^n e. Every earlier step multiplies the endpoint contribution by another gamma.

At gamma=1, no discount attenuates that error within the finite rollout. Longer rollouts may still reduce reliance on endpoint estimates by reaching true termination more often. This is a distinct mechanism from discount attenuation, so finite-horizon gamma=1 experiments should explain why changing rollout length affects their targets.

<a id="section-28-3"></a>

## 28.3 Work a three-step rollout

Take rewards [1,0,2], gamma=0.9, and a nonterminal endpoint value 3. Backward targets are G_2=4.7, G_1=4.23, and G_0=4.807.

If current values are [2,2,2], the advantages are [2.807,2.23,2.7]. If the final transition truly terminates, the targets instead become [2.62,1.8,2]. One boundary flag changes every earlier target in the rollout.

<a id="section-28-4"></a>

## 28.4 Define the combined loss

```text
total_loss = actor_loss + c_value * critic_loss − c_entropy * entropy
actor_loss = −mean(log_prob * detached_advantage)
critic_loss = mean((value − detached_return_target)²)
```

The sign on entropy encourages diversity when minimizing this loss. Coefficients are scale-dependent: multiplying rewards changes advantage and value targets without necessarily changing entropy. Report the precise reductions and coefficients used.

<a id="section-28-5"></a>

## 28.5 Handle multiple environments

Store rollout tensors with explicit time and environment axes, commonly [T,N,...]. Each environment has independent episode boundaries. Flattening to [T×N,...] is convenient only after temporal returns or advantages have been computed correctly.

If an environment resets within a rollout, do not let the next episode's rewards flow backward into the previous one. At a timeout, bootstrap from the final observation when appropriate, but do not continue the trace through the reset state.

### Work through a reset inside one vectorized stream

One worker collects rewards [1,5,2,3], where the second transition terminates and the last two rewards belong to a new episode. With gamma=1 and both episodes completed, correct returns are [6,5,5,3]. An unmasked backward accumulator gives [11,10,5,3], leaking the later episode into the earlier targets.

Other workers may have different boundary locations. A [T,N] mask must apply independently to each worker; a single boundary flag for a whole timestep can incorrectly cut valid traces or preserve invalid ones. A four-step hand-built batch can test this more clearly than a long rollout printout.

<a id="section-28-6"></a>

## 28.6 Contrast A2C and A3C

Synchronous A2C waits for a collection batch and applies an update to a shared policy. A3C workers collect and compute gradients asynchronously, introducing parameter staleness along with parallel experience.

Both use actor–critic ideas, but their systems behavior differs. Calling any multi-environment actor–critic “A3C” is inaccurate. Wall-time advantages depend on hardware and environment speed; sample efficiency must be measured separately.

<a id="section-28-7"></a>

## 28.7 Examine batch-size tradeoffs

Increasing N adds parallel trajectories; increasing T extends temporal context before bootstrapping. Equal products T×N do not imply identical estimators. Longer T can change bootstrap bias and temporal correlation, while more environments can broaden start-state coverage.

Compare these settings at equal transition budgets and report update counts. A large batch can reduce gradient noise while producing fewer policy revisions for the same amount of experience.

### Equal batch size does not imply equal credit assignment

Compare T=32,N=4 with T=8,N=16. Both collect 128 transitions per batch, but the first can observe longer within-worker reward chains before bootstrapping. The second can cover more independent initial states in parallel.

For a reward delayed by twelve decisions, the short rollout often places that reward beyond its endpoint bootstrap. For a highly variable initial-state distribution, more workers can improve batch diversity. Predict which property matters in the chosen environment, then compare both target error and return at the same total transition budget.

<a id="section-28-8"></a>

## 28.8 Diagnose policy collapse

If entropy falls quickly and returns remain low, the actor may have committed before gathering useful evidence. Check advantage signs, reward scale, action masks, and exploration conditions before assuming an entropy coefficient alone will solve it.

If critic loss dominates shared gradients, inspect coefficient scales and representation interference. A small scalar actor loss is not necessarily a small actor gradient; log norms or targeted diagnostics when the scalar losses are hard to interpret.

<a id="section-28-9"></a>

## 28.9 Laboratory investigation

Use the actor–critic component of [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) and the rollout bookkeeping in [Notebook 06](../../notebooks/06_ppo.ipynb). A dedicated asynchronous A3C implementation is a proposed extension, not a shipped notebook feature.

Design an experiment that holds T×N fixed while varying T and N. Predict which change affects endpoint bootstrap dependence, then compare value error and return rather than reporting only wall time.

<a id="section-28-10"></a>

## 28.10 Problems, worked answers, and reading

1. In the worked rollout, replace the endpoint value 3 with 0 without changing termination. How much does G_0 change?
2. Why must returns be computed before arbitrarily shuffling time indices?
3. With N=8 and T=16, how many transitions are collected per complete rollout batch?

<details><summary>Worked answers</summary>

1. It decreases by gamma³×3=2.187, from 4.807 to 2.62. Zeroing a valid bootstrap creates this target difference even if the boundary is not terminal.
2. Return recursion depends on temporal successors and episode boundaries. A random ordering destroys that structure.
3. 128, assuming all environments produce one transition per collection step. Resets do not necessarily change this count, but their boundaries still matter.

</details>

Read [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), separating the estimator from the asynchronous execution architecture.

---

[← Chapter 27](../27-actor-critic-methods/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 29 →](../29-generalized-advantage-estimation/README.md)
