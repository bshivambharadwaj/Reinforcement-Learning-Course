# 2. The Agent–Environment Interaction

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 1](../01-what-is-reinforcement-learning/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 3 →](../03-markov-decision-processes/README.md)

[Quick-read Topic 2](../../README.md#topic-2) · [Notation](../NOTATION.md)

- [2.1 Define the interaction contract before the learner](#section-2-1)
- [2.2 Time indexing and what a transition contains](#section-2-2)
- [2.3 Episodes, rollouts, and continuing tasks](#section-2-3)
- [2.4 Factor the probability of a trajectory](#section-2-4)
- [2.5 Work through a two-step trace](#section-2-5)
- [2.6 Implement a boundary-aware collection loop](#section-2-6)
- [2.7 A numerical reset bug and its consequence](#section-2-7)
- [2.8 Experiment: invariants before performance](#section-2-8)
- [2.9 Transfer the contract to token and tool interaction](#section-2-9)
- [2.10 Problems, worked answers, and sources](#section-2-10)

---

<p align="center">
  <img src="../../assets/diagrams/interaction-loop.svg" width="760" alt="Actions produce feedback that becomes the next decision context." />
</p>

*Actions produce feedback that becomes the next decision context.*

<a id="section-2-1"></a>

## 2.1 Define the interaction contract before the learner

The agent–environment boundary determines what the learner controls. A delivery controller may choose wheel velocities, road segments, or complete routes. Those are different action spaces and time scales, even if all systems operate the same robot. An experiment is underspecified until the boundary, decision interval, observations, and termination rule are fixed.

**Prerequisite:** Chapter 1. **Learning targets:** write an unambiguous transition record, distinguish an episode from a rollout, factor a trajectory distribution, and diagnose reset-related target errors. The central engineering artifact is a task contract that someone else can implement without guessing when rewards arrive.

Think of the interface as an agreement between a pilot and a flight simulator. The pilot selects a control; the simulator advances the world and returns feedback. If both sides disagree about whether a crash has already ended the flight, the learning data become inconsistent.

<a id="section-2-2"></a>

## 2.2 Time indexing and what a transition contains

Use the convention that action A_t is selected from information available at time t. The environment then returns reward R_(t+1) and next observation O_(t+1). A full-state task may expose S_t directly; a partially observed task generally does not.

```text
record_t = (observation_t, action_t, reward_(t+1),
            final_observation_(t+1), terminated, truncated)
```

The reward belongs to the transition caused by the stored action. Shifting rewards one position can silently teach the wrong action. If a terminal transition delivers the parcel and pays +10, that reward remains part of the episode even though future reward after termination is zero.

For off-policy analysis, also retain the behavior action probability or enough information to reconstruct it. A dataset of states and rewards alone cannot generally identify which policy generated its actions.

<a id="section-2-3"></a>

## 2.3 Episodes, rollouts, and continuing tasks

An episode is a task-defined interaction ending in termination. A rollout is a collected segment; it can contain several episodes or stop halfway through one. A continuing task, such as resource allocation, may have no natural terminal event at all.

Suppose a courier has a ten-action deadline. If expiration defines task failure, remaining time belongs in the state and the deadline is true termination. If a data collector stops after ten actions merely to build a batch while the task could continue, that boundary is a truncation. The same numeric time limit can represent either situation; the meaning comes from the task contract.

Auto-reset environments complicate storage: the next observation returned to a caller may already belong to a new task. Preserve the previous task's final observation when it is needed for bootstrapping. A reset observation is not evidence about the interrupted episode's future.

<a id="section-2-4"></a>

## 2.4 Factor the probability of a trajectory

For a finite fully observed Markov task with initial distribution rho and policy pi, a trajectory likelihood factors into initial-state, action, and transition terms:

```text
p_pi(trajectory) = rho(s_0)
                  × product_t pi(a_t | s_t) p(s_(t+1), r_(t+1) | s_t, a_t)
```

This is the chain rule plus the conditional-independence assumptions of the model. For a history-dependent policy, replace s_t in the policy term with the available history. The environment term need not be differentiable for a score-function policy gradient: if its parameters do not depend on the policy parameters, differentiating the log likelihood leaves only policy terms.

That observation explains why reinforcement learning can train through nondifferentiable tools. It does not imply that arbitrary hidden environmental dependencies can be ignored. Parameter-dependent reset rules or simulators require a more careful derivative.

<a id="section-2-5"></a>

## 2.5 Work through a two-step trace

Let the initial state be A. The policy chooses right with probability 0.6. That action reaches B with probability 0.8 and otherwise ends in failure. At B, the policy chooses deliver with probability 0.5, and delivery then succeeds deterministically. The probability of the particular successful action/state sequence is 0.6 × 0.8 × 0.5 = 0.24.

If its rewards are −1 and +5 with gamma=0.9, that successful trajectory has return −1 + 0.9×5 = 3.5. Multiplying 0.24×3.5 gives its contribution to expected return, not the whole expected return. The remaining trajectories must also be included.

This distinction prevents a frequent mistake: reporting a high-return successful trace as the expected performance of the policy. A trajectory probability is also different from the probability of success, because several distinct trajectories may succeed.

<a id="section-2-6"></a>

## 2.6 Implement a boundary-aware collection loop

```text
observation = reset()
repeat:
    action, behavior_log_probability = policy(observation)
    next_observation, reward, terminated, truncated, info = step(action)
    final_observation = recover_final_observation_if_auto_reset(info, next_observation)
    store(observation, action, reward, final_observation, terminated, truncated)
    if terminated or truncated:
        observation = reset_or_use_auto_reset_observation()
    else:
        observation = next_observation
```

This is interface pseudocode; reset conventions differ among environments. Test it with a one-step task, a two-step task, and an externally truncated task. Each test should check the actual stored observation, not only the number of records.

In vectorized collection, masks are per environment. A single global `done` variable can erase or mix trajectories when one worker terminates before the others. Episode statistics and gradient targets need their own boundary handling.

<a id="section-2-7"></a>

## 2.7 A numerical reset bug and its consequence

Consider a truncated transition with reward 1, gamma=0.9, and value 4 at the final observation. Its continuing-task bootstrap target is 1+0.9×4 = 4.6. If an auto-reset observation has value 10 and is used accidentally, the target becomes 10. If the transition was genuine termination, the target would instead be 1.

All three computations can execute without a shape error. Only the task semantics determine which one is correct. Plotting a loss will not reveal the mistake reliably because a network can fit incorrect targets.

> **Failure Mode:** the replay buffer or rollout may be numerically valid but causally invalid. Trace one stored episode end-to-end and compare it with the simulator's event log before scaling collection.

<a id="section-2-8"></a>

## 2.8 Experiment: invariants before performance

Inspect [Notebook 06](../../notebooks/06_ppo.ipynb), especially the separate bootstrap and recursion masks. It implements vectorized rollouts; the simpler [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) makes terminal values explicit.

Design an invariant table: terminal next-value contribution is zero; a reset never contributes to the previous task's return; a rollout-end truncation can retain a final bootstrap; stored actions reproduce the rewards in a deterministic environment. Check these before comparing learning curves.

Then deliberately substitute reset observations at truncated boundaries and measure value error on a tiny task with a known answer. Restore the correct collector afterward. The point is to explain a failure mechanism, not to search for a hyperparameter that hides it.

<a id="section-2-9"></a>

## 2.9 Transfer the contract to token and tool interaction

For token generation, a token is an action and the prefix becomes the next observation. EOS may terminate one response, but a tool-using task can continue after that response. A tool result changes the next context without necessarily being a policy-generated token whose log probability belongs in the actor loss.

An agent trace should therefore distinguish generated tokens, tool calls, tool outputs, and environment resets. Masking every token in a concatenated transcript as if the policy generated it gives an incorrect objective. A task may include a decision to stop, so the stopping rule itself can be learned.

The transfer is structural: both robot trajectories and tool transcripts require correctly attributed actions, rewards, and boundaries. Their raw data formats differ substantially.

<a id="section-2-10"></a>

## 2.10 Problems, worked answers, and sources

**Problem A — trace accounting.** In the two-step example, delivery at B is chosen with probability 0.75 instead of 0.5. Calculate the probability and return of the specified successful trace. Does its return change?

**Problem B — target semantics.** A true terminal transition has reward −3 and a mistakenly stored next value of 100. With gamma=0.95, what should its target be? What target would the bug produce?

**Problem C — design.** Specify an episode boundary for an agent that can retry a failed calculator call. Explain whether a failed call ends the episode.

<details><summary>Worked answers</summary>

A: the trace probability becomes 0.6×0.8×0.75=0.36. Its realized return remains 3.5; only its probability changes. B: the correct target is −3. The erroneous bootstrap gives −3+95=92. C: a call failure should not terminate a retry-capable task unless the contract says so; return an error observation, charge its defined cost, and retain the remaining budget. Submission or exhausted task budget can terminate the task.

</details>

**Read:** Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 3 ([author's book page](http://incompleteideas.net/book/the-book-2nd.html)). For policy-gradient use of trajectory probabilities, continue to Chapter 25 here.

---

[← Chapter 1](../01-what-is-reinforcement-learning/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 3 →](../03-markov-decision-processes/README.md)
