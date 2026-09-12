# 34. Model-Based Reinforcement Learning

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 33](../33-offline-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 35 →](../35-multi-agent-reinforcement-learning/README.md)

[Quick-read Topic 34](../../quick-read/34-model-based-reinforcement-learning.md) · [Notation](../NOTATION.md)

- [34.1 Learn or use a model of consequences](#section-34-1)
- [34.2 Separate three components](#section-34-2)
- [34.3 Describe Dyna-style learning](#section-34-3)
- [34.4 Explain model-predictive control](#section-34-4)
- [34.5 Bound a simple compounding-error scenario](#section-34-5)
- [34.6 Work a planner-exploitation example](#section-34-6)
- [34.7 Use uncertainty without overstating it](#section-34-7)
- [34.8 Diagnose reward and dynamics separately](#section-34-8)
- [34.9 Laboratory extension and agent search](#section-34-9)
- [34.10 Problems, worked answers, and reading](#section-34-10)

---

<p align="center">
  <img src="../../assets/diagrams/model-based-loop.svg" width="760" alt="A model supports imagined consequences and planning." />
</p>

*A model supports imagined consequences and planning.*

<a id="section-34-1"></a>

## 34.1 Learn or use a model of consequences

Model-based RL uses an environment model to support decisions or learning. The model may be known, learned from data, or used only locally for short-horizon planning. A good one-step predictor is useful, but not sufficient to guarantee a good planner.

**Prerequisites:** Chapters 3, 12, 20–23. **Targets:** distinguish planning architectures, quantify error accumulation, and design a model-exploitation test.

The courier rehearses routes on an internal map before driving. Rehearsal can save real journeys, but a map error repeated through many imagined steps can make an impossible shortcut look excellent.

<a id="section-34-2"></a>

## 34.2 Separate three components

Specify the dynamics model, reward model, and decision procedure. A model can predict next-state distributions, latent states, or value-relevant quantities rather than raw observations. Planning may use tree search, trajectory optimization, or synthetic transitions for a value learner.

These choices solve different subproblems. A world model that generates plausible images is not necessarily accurate about the consequences of the agent's actions or the task reward.

<a id="section-34-3"></a>

## 34.3 Describe Dyna-style learning

Collect a real transition, update the value learner, update the model, then perform additional value updates on simulated transitions from the model. The synthetic updates reuse learned environmental structure.

Report real and model-generated transitions separately. Ten thousand imagined transitions are not ten thousand independent observations of the environment. If the model is wrong, repeated planning can amplify its errors rather than average them away.

<a id="section-34-4"></a>

## 34.4 Explain model-predictive control

At the current state, optimize a candidate action sequence over a finite horizon, execute only its first action, observe the real next state, then replan. Replanning limits open-loop commitment to model errors.

The planning horizon, candidate count, dynamics uncertainty, and terminal value estimate determine behavior. A short horizon can be myopic; a long one can exploit inaccuracies. Equal environment interactions do not imply equal decision-time compute across planners.

<a id="section-34-5"></a>

## 34.5 Bound a simple compounding-error scenario

Suppose a coupling gives per-step probability at most epsilon that a model and real transition disagree while their states still match. Over H steps, the probability of any mismatch is at most H epsilon by a union bound, capped at one.

With nonnegative rewards bounded by R_max, this crude argument can yield an undiscounted return-difference bound of order R_max epsilon H² when rewards agree on matched transitions. This is not a universal bound for any learned model; it states explicit assumptions and illustrates horizon amplification.

### Derive a finite-horizon coupling bound step by step

Assume model and real trajectories begin together, use the same policy, and can be coupled so that each matched-state transition disagrees with probability at most epsilon. Assume each reward lies in [0,R_max] and matched transitions have equal rewards.

By time t+1, mismatch probability is at most (t+1)epsilon, capped at one. The expected reward difference at that transition is therefore at most R_max min(1,(t+1)epsilon). Summing over H transitions gives a return-difference bound no larger than R_max epsilon H(H+1)/2, and trivially no larger than HR_max.

For signed rewards in [−R_max,R_max], the maximum per-transition difference becomes 2R_max. This derivation exposes the norm and reward-range conventions hidden by an informal O(H²epsilon) statement.

<a id="section-34-6"></a>

## 34.6 Work a planner-exploitation example

Two routes have true returns 5 and 1. A learned model predicts 4.8 and 7 because the second route enters an unfamiliar state. Planning chooses the second route and obtains 1.

Average prediction error on the logged first route can be tiny while decision quality is poor. Optimization preferentially searches for high predicted values, including positive model errors. Evaluate model accuracy on planner-selected trajectories, not only random held-out transitions.

### Separate random prediction error from optimizer-selected error

Suppose every candidate plan has true return 5, while the learned model predicts 5 plus independent noise. Randomly evaluating model accuracy can show zero mean error. Selecting the largest predicted plan instead preferentially selects a positive error.

Measure prediction calibration on the plans the optimizer actually chooses, including repeated optimization rounds. Increasing search budget can worsen this selected error even with an unchanged model. A planner should therefore be evaluated as a model-plus-search system, not by combining unrelated claims about average model accuracy and search thoroughness.

<a id="section-34-7"></a>

## 34.7 Use uncertainty without overstating it

Ensembles, probabilistic models, and conservative penalties can help identify or discourage uncertain plans. Ensemble disagreement is a model-dependent signal, not a guarantee that all members are wrong in different ways.

Compare predicted uncertainty with actual rollout errors on held-out tasks. Shared architecture, training data, and objective can produce confidently shared mistakes. A penalty coefficient also changes the decision tradeoff and should be included in the experimental specification.

<a id="section-34-8"></a>

## 34.8 Diagnose reward and dynamics separately

If predicted states are accurate but planned behavior is undesirable, inspect the reward model. If predicted rewards are correct only on unrealistic imagined states, inspect dynamics support. If both are accurate locally but planning fails, inspect horizon and optimization.

Use ablations with known dynamics and learned rewards, then learned dynamics and known rewards when a small simulator permits it. This isolates failure sources more effectively than changing every model component at once.

<a id="section-34-9"></a>

## 34.9 Laboratory extension and agent search

The course's DP reference in [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) uses a known model; it is not a learned world-model implementation. A proposed extension learns transition counts from a restricted dataset and compares planned versus actual policy values.

For tool agents, executing a tool in a sandbox can provide a real transition rather than an imagined one. For reasoning agents, internal text simulation is a learned prediction. Distinguish these sources of evidence when counting search and judging reliability.

### Design a model-learning budget that can be compared fairly

Train a model from a fixed number of real transitions, then vary the number of imagined updates while keeping that dataset fixed. Report real interaction count, model-query count, and actual evaluated policy return separately.

If imagined updates first help and then hurt, inspect whether the planner is exploiting unsupported transitions or simply overfitting a limited task distribution. An oracle-model version of the same experiment can isolate model error from planning/optimization limitations. This is a proposed extension to the known-model reference, not a claim that the current notebook already trains a world model.

<a id="section-34-10"></a>

## 34.10 Problems, worked answers, and reading

1. With epsilon=0.02 and H=10, what union-bound mismatch probability is available?
2. Why does executing only the first planned action help with model error?
3. What does low one-step error fail to certify about optimized trajectories?

<details><summary>Worked answers</summary>

1. At most 0.2 under the stated coupling assumptions. This can be loose and does not itself give a task-success guarantee.
2. Real observations replace imagined intermediate states before the next plan, reducing open-loop error accumulation. Model and optimization errors can still affect each chosen first action.
3. The planner may visit unsupported states or select unusually optimistic errors, so its induced distribution can differ sharply from the validation distribution.

</details>

Read [Sutton and Barto, Chapter 8](http://incompleteideas.net/book/the-book-2nd.html) and [Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253).

---

[← Chapter 33](../33-offline-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 35 →](../35-multi-agent-reinforcement-learning/README.md)
