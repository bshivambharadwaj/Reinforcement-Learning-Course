# 4. States, Actions, Rewards and Transitions

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 3](../03-markov-decision-processes/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 5 →](../05-policies/README.md)

[Quick-read Topic 4](../../quick-read/04-states-actions-rewards-and-transitions.md) · [Notation](../NOTATION.md)

- [4.1 The components are design decisions](#section-4-1)
- [4.2 State variables and information leakage](#section-4-2)
- [4.3 Actions and their time scale](#section-4-3)
- [4.4 Joint outcomes versus expected rewards](#section-4-4)
- [4.5 Reward scale and reward shifts](#section-4-5)
- [4.6 Derive potential-based shaping](#section-4-6)
- [4.7 Failure cases that reveal specification errors](#section-4-7)
- [4.8 Lab: preserve the task while changing feedback timing](#section-4-8)
- [4.9 Apply the design questions to post-training](#section-4-9)
- [4.10 Problems, worked answers, and sources](#section-4-10)

---

<p align="center">
  <img src="../../assets/diagrams/delivery-robot.svg" width="760" alt="A concrete delivery task makes state, action, and reward choices inspectable." />
</p>

*A concrete delivery task makes state, action, and reward choices inspectable.*

<a id="section-4-1"></a>

## 4.1 The components are design decisions

States, actions, rewards, and transitions specify what problem an algorithm will solve. They are not neutral labels added after choosing PPO. A delivery task with road segments as actions differs from one with motor commands: the control rate, uncertainty, and credit horizon change together.

**Prerequisites:** Chapters 2–3. **Targets:** identify a sufficient state representation, compare action granularities, reason about reward transformations, and build a task specification that separates physical dynamics from learning objectives.

The recurring test is implementation agreement. Two engineers given your specification should generate the same legal actions, rewards, and terminal events for the same state and random outcome. An ambiguous reward rule can make an apparently reproducible experiment irreproducible at the task level.

<a id="section-4-2"></a>

## 4.2 State variables and information leakage

List variables that affect next outcomes, then distinguish those actually observed. Position, battery, destination, and remaining deadline can all matter. Including the destination is legitimate if it is part of the task input; including the simulator's future success flag is not.

Normalize continuous features for numerical conditioning without discarding their meaning. A battery fraction and a raw pixel intensity need not have comparable scales. Document units and coordinate conventions so evaluation uses the same representation as training.

If a representation is lossy, explain what histories it aliases and whether the aliasing affects optimal actions. A compact representation can be adequate for a particular routing policy while failing to be a complete Markov state. Avoid presenting empirical adequacy as a proof of sufficiency.

<a id="section-4-3"></a>

## 4.3 Actions and their time scale

An action can be discrete, continuous, or structured. Structured tool calls combine a tool identifier with arguments. A policy may generate all parts jointly or delegate argument construction to another module; specify which part is learned.

Actions can also take different amounts of time. If a macro-action lasts k elementary steps, its discounted continuation uses gamma^k and its immediate macro reward accumulates those k rewards with their internal discounts. Treating every tool call as an equal-duration step is a modeling choice, not an identity.

For a two-step macro action with rewards −1 and +4, gamma=0.9, and next-state value 3, its target is −1+0.9×4+0.9²×3=5.03. Using a single gamma on the continuation would incorrectly yield 5.3 under the same elementary-step objective.

### Random-duration actions require an expectation over duration

For an action lasting a random number K of elementary steps, define its discounted accumulated reward as R_macro=sum_(j=0 to K−1)gamma^j R_(t+j+1). Its backup is E[R_macro+gamma^K V(S_next)|s,a]. Duration, reward, and successor may be correlated, so replacing gamma^K by gamma raised to the mean duration is generally invalid.

For K equal to 1 or 3 with equal probability and gamma=0.5, E[gamma^K]=0.3125, while gamma^E[K]=0.25. Even with zero internal rewards and a constant continuation value 8, these produce 2.5 versus 2. The difference comes from applying a nonlinear function before versus after averaging.

<a id="section-4-4"></a>

## 4.4 Joint outcomes versus expected rewards

The joint kernel p(s',r|s,a) captures correlation between rewards and next states. For expected-return Bellman updates, one can sum the joint outcomes directly or use a conditional mean reward representation consistently.

Imagine a risky movement: with probability 0.8 it reaches B and yields +2, otherwise it reaches failure and yields −8. Its expected immediate reward is zero. That does not make it equivalent to a deterministic zero-reward move if the future states differ or if the objective is risk sensitive.

When constructing an environment, sample correlated outcomes together. Sampling next state and reward independently from their marginals can invent impossible combinations such as a success reward paired with a failure state. Matching marginal means is insufficient to reproduce the trajectory distribution.

<a id="section-4-5"></a>

## 4.5 Reward scale and reward shifts

Multiplying every reward by a positive constant preserves expected-return policy rankings when the rest of the objective is unchanged. It can still change optimization behavior because gradients, value losses, clipping thresholds, and regularization coefficients have scales.

Adding a constant per step is more subtle. In an infinite discounted continuing task, adding c shifts all policy values by c/(1−gamma), provided the shift applies at every step under the same continuing convention. In a fixed-horizon task it similarly adds a policy-independent sum if all trajectories have the same length.

With policy-dependent episode lengths and no rewards after termination, a per-step constant changes the incentive to stop. A +1 living bonus can make stalling attractive. Do not apply the continuing-task invariance result to variable-length episodes without checking its assumptions.

### Clarify when adding a constant preserves rankings

With exactly H rewarded transitions for every policy, adding c per transition adds c(1−gamma^H)/(1−gamma) for gamma≠1, or cH for gamma=1. This is independent of action choices and therefore preserves rankings. If episodes can terminate early and the addition stops at termination, policies receive different numbers of added terms.

An absorbing-state representation does not automatically resolve this ambiguity. Adding c to every absorbing-state reward continues paying after nominal completion; adding it only to preterminal transitions does not. These implement different objectives even if both are described informally as “adding a living reward.” Write the rule at the transition level.

<a id="section-4-6"></a>

## 4.6 Derive potential-based shaping

Let the shaping addition be F_t=gamma Phi(s_(t+1))−Phi(s_t). Its discounted sum through T transitions telescopes:

```text
sum_(t=0..T-1) gamma^t F_t = −Phi(s_0) + gamma^T Phi(s_T)
```

If terminal potential is zero in a finite episodic task, the change is a start-dependent constant rather than a route-dependent bonus. For bounded potential in an infinite discounted task, the endpoint term vanishes as T grows. Under these conditions the return ordering is preserved.

Boundary conventions matter. A nonzero terminal potential can introduce a length- or terminal-state-dependent change. A hand-written reward for every apparent progress event is not automatically potential-based; reversible progress can be farmed repeatedly if the reverse movement is not accounted for.

### Show why a shaping loop cannot create free discounted reward

Consider A→B→A with Phi(A)=0, Phi(B)=2, and gamma=0.9. The first shaping reward is 1.8; the second is −2. Their discounted sum is 1.8+0.9×(−2)=0, matching the telescoping formula because the start and end potentials are both zero.

If only positive progress is paid and the reverse move is unpenalized, the loop earns 1.8 instead. That is a different reward function and can encourage repeated movement. A good shaping audit enumerates reversible transitions and checks complete discounted cycles, including their boundary potentials, rather than checking isolated positive rewards.

<a id="section-4-7"></a>

## 4.7 Failure cases that reveal specification errors

Rewarding the robot whenever it enters a street may encourage looping between two streets. Charging a movement cost after already replacing it with a terminal reward can double-count the final step. Forgetting to expose time can make identical observations require different actions near a deadline.

For tools, an invalid call needs a defined outcome: an error observation, cost, possible state change, and boundary flag. Silently resampling until a valid action is drawn changes the actual behavior policy and its probability. If the actor loss records the pre-resampling probability, its likelihood is wrong.

> **Engineering Note:** document invalid-action handling next to the action space. External masks, rejection with cost, and constrained parameterizations are different mechanisms with different training distributions.

<a id="section-4-8"></a>

## 4.8 Lab: preserve the task while changing feedback timing

[Notebook 11](../../notebooks/11_tool_agent_capstone.ipynb) compares verified reward with potential shaping. Gamma is one, the initial potential is zero, and terminal potential is zero; the total shaping contribution cancels. Explain why learning can still follow different paths even when total returns agree.

Extend the capstone with different tool costs and a strict action budget. Report task success and cost separately. Then add a naive +0.1 bonus for every task-read call and test whether rereading becomes attractive. The hypothesis is that the naive bonus changes the objective, whereas correctly bounded potential shaping does not.

Do not describe the capstone as learning arithmetic. Its tools calculate; its policy selects operations. That boundary is precisely what makes the comparison interpretable.

<a id="section-4-9"></a>

## 4.9 Apply the design questions to post-training

For a language model, token count can become a cost, EOS an action, and the response an episode. A per-token bonus may favor length; a per-token penalty may favor premature stopping. These are objective choices that need independent quality evaluation.

A preference score and a KL penalty also live on particular scales. Rescaling reward without rescaling its reference penalty changes the regularized optimum. A training implementation that normalizes advantages may hide some scale effects in the actor but not eliminate all effects on critics or other loss terms.

In multimodal tasks, action coordinates and observation preprocessing must agree. A correct click in normalized coordinates can become a wrong click after an undocumented resize. The learning algorithm cannot repair a systematically inconsistent action interface by theorem alone.

<a id="section-4-10"></a>

## 4.10 Problems, worked answers, and sources

**A.** Verify the macro-action target when the continuation value is 0 instead of 3. **B.** For gamma=0.9, Phi(s_0)=2 and Phi(s_T)=0, calculate the total discounted shaping addition. **C.** Construct a two-policy variable-length example where adding +1 per transition changes the preferred policy.

<details><summary>Worked answers</summary>

A: −1+0.9×4=2.6. B: −2, regardless of the intermediate potentials, assuming the shaping identity is applied at every transition. C: let policy A terminate immediately for reward 1, while policy B takes three transitions and totals 0 with gamma=1. Before the shift A wins 1>0; afterward A totals 2 while B totals 3, so B wins. Fixed-horizon or continuing-task invariance assumptions do not hold.

</details>

**Read:** [Ng, Harada, and Russell: policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf); [Sutton and Barto, Chapter 3](http://incompleteideas.net/book/the-book-2nd.html).

---

[← Chapter 3](../03-markov-decision-processes/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 5 →](../05-policies/README.md)
