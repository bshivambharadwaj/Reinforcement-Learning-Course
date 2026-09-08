# 3. Markov Decision Processes

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 2](../02-the-agent-environment-interaction/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 4 →](../04-states-actions-rewards-and-transitions/README.md)

[Quick-read Topic 3](../../README.md#topic-3) · [Notation](../NOTATION.md)

- [3.1 What the Markov model promises](#section-3-1)
- [3.2 Specify all components and their types](#section-3-2)
- [3.3 Construct a three-state delivery MDP](#section-3-3)
- [3.4 Derive the induced Markov reward process](#section-3-4)
- [3.5 Time and hidden state change the model](#section-3-5)
- [3.6 When is state compression legitimate?](#section-3-6)
- [3.7 Audit a model before solving it](#section-3-7)
- [3.8 Lab: a solver is also a diagnostic instrument](#section-3-8)
- [3.9 Connect the abstraction to language and agents](#section-3-9)
- [3.10 Problems, worked answers, and sources](#section-3-10)

---

<p align="center">
  <img src="../../assets/diagrams/agent-environment.svg" width="760" alt="The interface separates policy choices from environment outcomes." />
</p>

*The interface separates policy choices from environment outcomes.*

<a id="section-3-1"></a>

## 3.1 What the Markov model promises

A Markov decision process is a contract about conditional predictions, not a requirement that the world be simple. Once the current state and action are known, additional past information does not change the model's distribution of the next state and reward. This permits reusable value functions and Bellman recursions.

**Prerequisites:** Chapters 1–2, conditional probability, and basic matrix notation. **Targets:** construct a finite MDP, test whether a representation discards relevant information, separate modeling assumptions from algorithmic guarantees, and solve a small policy value problem.

The delivery analogy is a dispatch screen. If the screen contains location, destination, battery, and relevant road conditions, yesterday's route may add no predictive information. If it omits battery, identical screen states can imply different chances of completing the next movement.

<a id="section-3-2"></a>

## 3.2 Specify all components and their types

A discounted finite MDP can be specified by a state set S, available actions A(s), joint transition/reward kernel p(s',r|s,a), initial distribution rho, and discount gamma. Many presentations replace the joint kernel with P(s'|s,a) and expected reward R(s,a), which is sufficient for expected-return Bellman equations but not for every risk-sensitive question.

For each legal state-action pair, outcome probabilities are nonnegative and sum to one. Initial-state probabilities also sum to one. Rewards can be deterministic or random. A terminal state can be represented as absorbing with zero reward thereafter; its value is then zero.

Do not confuse stochastic policy choices with stochastic dynamics. A deterministic environment can still generate different trajectories under a stochastic policy. Conversely, a deterministic policy need not produce the same outcome every time.

<a id="section-3-3"></a>

## 3.3 Construct a three-state delivery MDP

Let A be the depot, B the delivery street, and T terminal. At A, action go costs −1 and reaches B deterministically; action cancel ends the task with reward 0. At B, action deliver ends with reward +4. Action wait returns to B with reward −1. Use gamma=0.9 and start at A.

The stationary policy pi takes go at A and deliver at B. Its equations are V(T)=0, V(B)=4, and V(A)=−1+0.9×4=2.6. Cancellation has Q(A,cancel)=0, so go is preferred under this policy's continuation.

If the policy instead always waits at B, then V(B)=−1+0.9V(B), giving V(B)=−10 and V(A)=−10. The same environment has radically different values under different policies. The model does not contain a unique value until an objective and policy or optimization criterion are specified.

<a id="section-3-4"></a>

## 3.4 Derive the induced Markov reward process

Fixing pi removes the action-selection decision by averaging it into the dynamics:

```text
P_pi(s,s') = sum_a pi(a|s) P(s'|s,a)
r_pi(s)   = sum_a pi(a|s) R(s,a)
V_pi      = r_pi + gamma P_pi V_pi
```

For gamma<1 and finite states, the inverse representation is V_pi=(I−gamma P_pi)^(-1)r_pi. A stochastic P_pi has spectral radius at most one, so discounting makes the geometric series sum of gamma^k P_pi^k converge. That series interpretation explains the inverse as accumulated expected future rewards rather than an arbitrary linear-algebra trick.

In numerical code solve the linear system rather than explicitly constructing an inverse. At gamma=1, the discounted argument no longer applies; episodic/transience conditions or finite-horizon backward recursion must replace it.

<a id="section-3-5"></a>

## 3.5 Time and hidden state change the model

In a finite-horizon task, augment state with remaining time or use time-indexed values V_t. A policy can then depend on the budget. A route worth trying with ten steps left may be impossible with one step left even at the same location.

Partial observation is different from stochasticity. Suppose a hidden road condition is chosen once per episode and persists. The current camera image does not reveal it, but an earlier failed movement does. A policy using only the image cannot generally represent all useful behavior. A history-dependent policy or a belief distribution over hidden state is an appropriate conceptual alternative.

Treating observations as states can still be a practical approximation. Name it as such. A convergence theorem for a fully observed tabular MDP cannot automatically certify the resulting learner.

<a id="section-3-6"></a>

## 3.6 When is state compression legitimate?

Suppose a mapping groups original states into abstract states. A strong sufficient condition for exact abstraction is that original states in the same group have identical expected immediate rewards and identical probabilities of transitioning into every abstract group for each corresponding action.

Why? If the abstract next-state distribution and reward agree, a Bellman backup on any value function constant within each group agrees too. Repeated backups therefore preserve that consistency. This is a sufficient structural argument, not a claim that every useful learned representation must satisfy exact equality.

For a counterexample, group two depot states with different battery levels while a long movement succeeds only from the charged state. The abstract action has no single history-independent success probability unless the hidden-state mixture is fixed in a way the policy cannot change. Different visitation histories can break that assumption.

<a id="section-3-7"></a>

## 3.7 Audit a model before solving it

Check dimensions, probability normalization, reward timing, absorbing terminal behavior, and reachability from rho. A state unreachable from the initial distribution can have a large value without affecting the start-state objective. It can still matter if the initial distribution later changes.

Compare sampled transition frequencies with the specified kernel using independent random draws. This checks whether the simulator implements the model being solved. Exact planning results are not meaningful if the simulator used for evaluation has different terminal conventions or rewards.

> **Failure Mode:** an optimizer can solve the wrong MDP perfectly. Modeling errors are not automatically corrected by more Bellman iterations, more training seeds, or a deeper network.

<a id="section-3-8"></a>

## 3.8 Lab: a solver is also a diagnostic instrument

In [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb), inspect the transition tensor and reward array before running value iteration. The notebook compares iterative policy evaluation with an independent linear solve. Explain which assumptions make that comparison legitimate.

Extend the grid with a two-level battery state. First solve the augmented MDP. Then deliberately collapse the battery feature and identify states for which the collapsed transition distribution is history dependent. The deliverable is a concrete counterexample, not simply a lower average score.

For [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb), explain why time is in DeliveryLine-v1's state and why backward DP gives an exact oracle there. The sampled learners do not receive that oracle during training.

<a id="section-3-9"></a>

## 3.9 Connect the abstraction to language and agents

A complete generated prefix can function as state for a token-generation model when subsequent token probabilities depend on that prefix and fixed model parameters. Truncating history or hiding tool state can destroy that sufficiency for an external agent task, even if the language model itself accepts the shorter context.

An agent's environment may include a filesystem, browser state, external service, and task budget. Its observation is only the accessible part. Defining an MDP over the complete world is conceptually possible but does not grant the policy access to that state.

The practical question is which information supports good decisions under available compute and memory. Good modeling makes the approximation visible and supplies tests for its consequences.

<a id="section-3-10"></a>

## 3.10 Problems, worked answers, and sources

**A.** In the three-state MDP, choose deliver at B with probability 0.5 and wait otherwise. Derive V(B) and V(A) when go is always chosen at A.

**B.** Why does setting gamma=1 make the always-wait policy problematic? Contrast this with a six-step finite task.

**C.** Give a counterexample to the claim that two states with the same immediate reward can always be merged.

<details><summary>Worked answers</summary>

A: V(B)=0.5×4+0.5×(−1+0.9V(B))=1.5+0.45V(B). Thus V(B)=30/11≈2.7273 and V(A)=−1+0.9×30/11=16/11≈1.4545. B: undiscounted repeated −1 rewards sum to negative infinity; a fixed finite horizon bounds the number of terms and permits backward recursion. C: identical immediate rewards can lead to different future state groups, such as one route reaching a rewarding terminal and another entering a costly loop.

</details>

**Read:** [Sutton and Barto, Chapter 3](http://incompleteideas.net/book/the-book-2nd.html). Chapters 10–16 here derive and apply the associated Bellman operators.

---

[← Chapter 2](../02-the-agent-environment-interaction/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 4 →](../04-states-actions-rewards-and-transitions/README.md)
