# 1. What Is Reinforcement Learning?

**Beginner** · **Created by Shivam Bharadwaj**

[Chapters](../README.md) | [Quick-read Topic 1](../../quick-read/01-what-is-reinforcement-learning.md) | [Practical track](../../README.md#practical-track)

Reinforcement learning studies how to improve decisions using feedback about their
consequences. The hard part is that a decision can change both the outcome and what the
learner gets to observe next. This chapter builds that idea carefully before introducing
particular learning algorithms.

**Before starting:** you should be comfortable with basic arithmetic and weighted averages.
No neural-network knowledge is needed here. The full course's mathematics prerequisites
become useful in later chapters.

**By the end, you should be able to:** define an RL task, distinguish observations from
hidden state, calculate a discounted return, explain exploration and credit assignment,
and identify what would make a language-model or tool system an RL learner.

## Chapter navigation

- [1.1 Why decisions need a time horizon](#section-1-1)
- [1.2 What kind of feedback teaches the learner?](#section-1-2)
- [1.3 Agent, environment, state, observation, and action](#section-1-3)
- [1.4 Rewards, objectives, and constraints](#section-1-4)
- [1.5 From individual rewards to return](#section-1-5)
- [1.6 A policy is a decision rule](#section-1-6)
- [1.7 Exploration versus exploitation](#section-1-7)
- [1.8 Delayed feedback and credit assignment](#section-1-8)
- [1.9 Work through a complete decision problem](#section-1-9)
- [1.10 From the delivery robot to reasoning agents](#section-1-10)
- [Practice and readiness check](#practice)

<a id="section-1-1"></a>

## 1.1 Why decisions need a time horizon

Suppose a delivery robot approaches two roads. One road is longer but reliable. The other
is shorter but can lead to a failed delivery. Choosing the cheapest immediate move does
not settle which road is better: the robot must consider what each choice makes possible
afterward.

This separates a **local decision** from a **sequential decision problem**. A local rule
can prefer an action based on its immediate result. A sequential rule considers the
later situations created by that action as well.

Reading a task specification is another example. It may cost time and produce no completed
work immediately, but it can make a later action much more likely to succeed. Information
gathering can therefore have long-term value.

**Analogy:** choosing what to study tonight is not just choosing the easiest exercise.
An exercise that builds a missing skill may improve tomorrow's performance more than
one that gives an easy success now.

Not every feedback problem requires a long-horizon RL formulation. If a choice affects
only its immediate payoff and does not change future decision opportunities, a bandit
model may be sufficient. If the correct output is already labeled and the goal is to
imitate those labels, supervised learning may be a more direct fit.

**Check your understanding:** why can an action with an immediate cost still be useful?

<details>
<summary>Explain the answer</summary>

Its effect on later states or available information can increase future reward by more
than the immediate cost. The relevant comparison is the total objective over the task,
not just the next reward.

</details>

<a id="section-1-2"></a>

## 1.2 What kind of feedback teaches the learner?

Consider three ways to teach the delivery robot:

| Setup | Data supplied | Learning question |
|---|---|---|
| Supervised imitation | Situations paired with an expert's chosen moves | Which action resembles the expert's action here? |
| Unsupervised representation learning | Observations without action or reward labels | What useful structure exists in these observations? |
| Reinforcement learning | Actions, resulting observations, and rewards | Which behavior produces better long-term outcomes? |

A reward does not usually identify the correct action. If a robot receives −1 after
moving, that alone does not tell it whether another move would have been better. Both
routes might cost −1 now, while their future risks differ.

This creates a **counterfactual gap**: the robot observes the consequences of the action
it took, not all the consequences of actions it did not take. Repeated experience,
exploration, or a model can help close that gap, but a single reward does not close it.

The learner's behavior also influences its dataset. A robot that always uses the safe
road may gather excellent evidence about that road and almost none about the shortcut.
This differs from a simple supervised setup where a fixed dataset is handed to the learner.

There are useful combinations. A policy can first imitate demonstrations and then improve
using rewards. A representation can be pretrained without rewards and reused by an RL
policy. These stages have different objectives even when they share the same network.

> **Common Misconception:** RL always means collecting fresh experience during every update.
> Offline RL learns from a fixed interaction dataset. What matters is the decision problem
> and learning objective, not whether a live environment is connected at that moment.

**Check your understanding:** a dataset contains expert actions and rewards. Is every
training procedure on that dataset automatically RL?

<details>
<summary>Explain the answer</summary>

No. A procedure that only fits expert action labels is imitation learning. A procedure
using rewards and transition structure to optimize sequential performance may be offline
RL. The presence of reward columns does not by itself determine the training objective.

</details>

<a id="section-1-3"></a>

## 1.3 Agent, environment, state, observation, and action

The **agent** chooses actions. The **environment** determines what happens as a result
and supplies feedback. This boundary is a modeling choice: decide which decisions are
learned and which behavior belongs to the surrounding system.

<p align="center">
  <img src="../../assets/diagrams/agent-environment.svg" width="760" alt="The agent selects an action; the environment returns a reward and a new observation">
</p>

| Object | Meaning | Delivery example |
|---|---|---|
| State | Information describing the environment for predicting subsequent outcomes | Position, remaining battery, destination, and relevant traffic conditions |
| Observation | Information actually available to the agent | Camera image, battery reading, and a destination instruction |
| Action | A choice available to the agent | Move left, move right, inspect a road, or stop |
| Transition | How the situation changes after an action | Moving to another junction or remaining blocked |
| Reward | Scalar feedback associated with a transition or outcome | Movement cost or successful-delivery reward |
| Episode | A task instance with a specified ending | One delivery ending in success, failure, or a task-defined deadline |

State and observation need not be identical. A camera image may show a road without
revealing whether it is blocked farther ahead. Two identical observations can therefore
hide different situations. History, memory, or a belief about hidden conditions may help.

A state representation is sufficient for a Markov model when the information retained
is enough for the relevant next-outcome distribution, without needing the omitted past.
Simply naming a vector `state` does not make it sufficient. Topic 3 develops this idea
formally; here, the practical question is **what information would change the consequence
of the same action?**

Time can matter too. If a delivery must finish within six moves, the same position with
five moves left is different from that position with one move left. Hiding the remaining
budget may make an otherwise simple task partially observed.

> **Engineering Note:** write down exactly what the policy sees. Do not accidentally give
> it the future outcome, the correct action, or hidden evaluation information as an input.

**Check your understanding:** is a screenshot necessarily a complete state for a browser agent?

<details>
<summary>Explain the answer</summary>

No. It may omit pending requests, hidden application state, earlier observations, or
information outside the viewport. Whether a representation is sufficient depends on
the task and environment dynamics, not its size or modality.

</details>

<a id="section-1-4"></a>

## 1.4 Rewards, objectives, and constraints

A **reward** is a signal. An **objective** specifies how those signals are combined and
averaged. A **constraint** specifies a restriction on acceptable behavior or outcomes.
These ideas are related, but they are not interchangeable.

For a delivery task, a possible reward design is:

- ordinary move: −1;
- successful delivery: +10 on the final transition;
- failed delivery: −6 on the final transition.

Here the terminal reward replaces the ordinary movement cost on that final transition.
Writing this convention explicitly prevents accidentally counting the same cost twice.
These are teaching values, not a claim that all delivery problems use this reward.

The objective might maximize expected discounted return over deliveries. The environment
might also forbid entering particular roads. A large failure penalty affects incentives,
but does not logically forbid a risky action. An allowed-action rule can enforce that
restriction independently of what the policy prefers.

Rewards also encode choices made by the task designer. If the designer rewards distance
traveled, a robot might drive in circles. If the intended goal is successful delivery,
distance is at best a proxy and may be a poor one.

**Analogy:** measuring study effort by pages highlighted can encourage highlighting
instead of understanding. Optimizing a measurement is not the same as satisfying the
intention behind it.

> **Failure Mode — reward hacking:** the learner finds behavior that scores well under
> the supplied reward while failing the intended task. The correct response includes
> examining the reward and evaluator, not only blaming the optimizer.

**Check your understanding:** an agent receives a high score for printing “task complete.”
What additional evidence would establish completion?

<details>
<summary>Explain the answer</summary>

An independent check of the task's actual output or environment state. Self-reported
completion is insufficient when the agent can produce that report without doing the work.
The tool-agent capstone demonstrates this distinction with a trained policy.

</details>

### Expected reward can hide an unacceptable outcome

Consider two delivery routes. Reliable always returns 4. Risky returns 10 with probability 0.9 and −20 with probability 0.1. Their expected returns are 4 and 7, so an expected-return objective prefers risky. The calculation is correct even if the customer finds a 10% severe-failure rate unacceptable. That disagreement reveals a missing objective requirement, not a mistake in averaging.

Write two separate quantities: expected return and probability of severe failure. Requiring the latter to be at most 0.01 excludes risky in this example. Replacing the constraint with a large penalty creates a tradeoff whose result depends on the coefficient. A constraint and a preference for lower risk are different specifications.

**Check:** if risky's success probability is p, its expected return is 30p−20. It beats reliable when p>0.8, but the 1% failure constraint requires p≥0.99. These thresholds answer different questions about the same route.

<a id="section-1-5"></a>

## 1.5 From individual rewards to return

The **return** combines rewards over time. In a finite episode with discount factor gamma,
the return from the start is:

```text
return = first reward
       + gamma × second reward
       + gamma² × third reward
       + ...
```

For a successful safe-route delivery with rewards **[−1, −1, +10]** and gamma **0.9**:

```text
return = −1 + 0.9 × (−1) + 0.9² × 10
       = −1 − 0.9 + 8.1
       = 6.2
```

The undiscounted total is 8. These are two different quantities, not two answers to
the same calculation. Gamma changes how the objective weights reward timing.

For a finite task, gamma=1 is possible: no discount is applied. In an unbounded continuing
task, discounting is one way to make an infinite reward sum well behaved under suitable
conditions. Other formulations exist; “RL must always use gamma less than one” is too broad.

Discounting is not a risk penalty. It weights time. A policy can still prefer an uncertain
shortcut under a discounted objective if its expected return is high enough.

A **realized return** describes one observed episode. An **expected return** averages
over the possible outcomes under a policy and environment. Comparing algorithms from one
lucky trip confuses these two levels.

**Check your understanding:** compute the return for [−1, +10] at gamma=0.9.

<details>
<summary>Show the calculation</summary>

The return is −1 + 0.9 × 10 = 8. The first reward is not discounted. The final +10 is
received one transition later than the first reward, so it receives one factor of gamma.

</details>

<a id="section-1-6"></a>

## 1.6 A policy is a decision rule

A **policy** maps available information to action choices. It is not necessarily a
precomputed route or a single sequence of moves.

A deterministic policy might always take the safe road at a particular junction. A
stochastic policy might choose the safe road with probability 0.8 and the shortcut with
probability 0.2. Its action probabilities must be nonnegative and sum to one over the
available choices.

The policy must also specify behavior at later decision points. If a road is blocked,
what does it do then? A route describes one possible sequence; a policy describes how
to respond to the situations it encounters.

Policies can be represented by tables, simple formulas, decision trees, or neural networks.
“Neural” is a representation choice, not part of the definition of RL. The early course
notebooks use small representations to expose the learning mechanism clearly.

Distinguish the **policy** from the **learning algorithm**. The policy makes decisions
with its current parameters. The algorithm changes those parameters using data. During
evaluation, we normally freeze learning so that the measurements describe a fixed policy.

| Question | Object being discussed |
|---|---|
| Which action should be sampled now? | Policy |
| How should new experience change the decision rule? | Learning algorithm |
| How well does the current rule perform? | Evaluation procedure |

**Check your understanding:** if a stochastic policy is evaluated with greedy actions
instead, are you necessarily measuring the same behavior?

<details>
<summary>Explain the answer</summary>

No. Replacing sampling with the highest-probability action changes the decision rule.
Both evaluations may be useful, but the evaluation mode must be stated. The same issue
appears when a language model is sampled during training but decoded greedily at test time.

</details>

<a id="section-1-7"></a>

## 1.7 Exploration versus exploitation

**Exploitation** chooses according to what the learner currently believes works best.
**Exploration** collects information that could change that belief. A policy that never
tries an uncertain road cannot learn its outcomes directly from its own trips.

Consider estimated action values of 6 for the safe road and 4 for the shortcut. A greedy
rule selects the safe road. But the estimate of 4 might come from one unlucky shortcut
attempt, while the estimate of 6 comes from hundreds of safe trips. Estimates and confidence
in estimates are different pieces of information.

A simple epsilon-greedy rule is:

```text
with probability epsilon:
    choose uniformly among available actions
otherwise:
    choose an action with the largest estimated value
```

With two actions, one uniquely greedy action, and epsilon=0.1, the greedy action has
probability **0.9 + 0.1/2 = 0.95**. The exploratory branch can choose the greedy action too.
The non-greedy action has probability 0.05.

This rule is useful for understanding exploration, but it does not reason explicitly about
which experiment would reduce uncertainty most. It can waste actions on choices already
known to be poor. Other exploration methods make different trade-offs.

Exploration also affects the cost of data collection. A learned greedy route may be good
while the exploratory training behavior still makes expensive mistakes. Later, SARSA and
Q-learning make this distinction especially clear.

> **Common Misconception:** exploration is always harmless randomness. Its cost depends
> on the environment. A simulator and a physical robot can have very different acceptable
> exploration budgets and allowed actions.

**Check your understanding:** with four actions and epsilon=0.2, what is the probability
of the unique greedy action under uniform epsilon-greedy exploration?

<details>
<summary>Show the calculation</summary>

It is 0.8 + 0.2/4 = 0.85. Each of the three other actions receives probability 0.05.
These probabilities sum to one.

</details>

### Separate uncertainty about the world from randomness in the world

Suppose road A has a known 50% traffic-delay probability. Road B has an unknown delay probability because it has been used only twice. Both can produce unpredictable trips, but only the second contains uncertainty that additional observations can reduce about the probability itself.

Exploration has value when its observations improve later choices. If the courier has only one remaining trip, spending that trip solely to learn a route can be wasteful. With many future trips, the same information may repay its cost. The horizon therefore affects information-gathering decisions as well as movement decisions.

An experiment should record how often each route was tried, not just its estimated mean. Two identical estimated means based on two and two thousand independent observations support different confidence levels. A neural network's numerical confidence is not automatically an uncertainty estimate calibrated to those sample counts.

<a id="section-1-8"></a>

## 1.8 Delayed feedback and credit assignment

The robot receives +10 after delivery, but success may depend on a route decision made
several steps earlier. **Temporal credit assignment** asks how later outcomes should
influence earlier decisions.

Giving all credit to the final movement ignores the choices that made that movement
possible. Giving equal credit to every earlier action can also be misleading: some
actions may have been unnecessary or actively harmful.

Different algorithm families approach the problem differently:

| Idea | How information travels backward | Later course topic |
|---|---|---|
| Monte Carlo return | Use the observed rewards after an earlier decision | [Topic 17](../../quick-read/17-monte-carlo-methods.md) |
| TD learning | Combine an immediate reward with the next state's current value estimate | [Topic 18](../../quick-read/18-temporal-difference-learning.md) |
| Policy gradients | Weight sampled action log probabilities by a return or advantage signal | [Topic 25](../../quick-read/25-policy-gradient-methods.md) |
| GAE | Combine several horizons of TD residuals | [Topic 29](../../quick-read/29-generalized-advantage-estimation.md) |

You do not need those update equations yet. The connection to retain is that future
feedback must somehow change decisions made before the feedback arrived.

Delayed reward and sparse reward are related but distinct. A signal can be delayed yet
arrive reliably at the end of every episode. Sparse reward means informative outcomes
are infrequent in the learner's experience. If every attempt receives the same failure
score, there may be little information for the optimizer to rank actions or trajectories.

**Analogy:** a student receives a final exam score but no feedback on intermediate practice.
It is difficult to know which study habits helped. Intermediate feedback can be useful,
but only if it measures something that supports the intended learning outcome.

**Check your understanding:** does a correct final answer prove every preceding reasoning
step was correct or necessary?

<details>
<summary>Explain the answer</summary>

No. The system may make compensating mistakes, guess, or produce a plausible trace that
does not determine the answer. Outcome correctness and intermediate-step correctness
must be measured separately when that distinction matters.

</details>

<a id="section-1-9"></a>

## 1.9 Work through a complete decision problem

Now specify enough detail to compare two routes numerically. This is a small **route-choice
example**, not a full navigation learner. Only the initial route selection is a decision;
the remaining route transitions follow the specified outcome branch.

| Task element | Definition |
|---|---|
| Start | A junction with the destination known |
| Available actions | Choose the safe route or choose the shortcut |
| Safe outcome | Always succeeds, rewards [−1, −1, +10] |
| Shortcut success | Probability 0.7, rewards [−1, +10] |
| Shortcut failure | Probability 0.3, rewards [−1, −6] |
| Ending | Delivery succeeds or fails; no future task rewards |
| Objective | Maximize expected discounted return, gamma=0.9 |

The probabilities here are part of the specified environment, not estimates inferred
from a handful of runs. As before, the final +10 or −6 replaces an ordinary move cost
on that transition.

**Step 1: calculate the return of every possible outcome.**

| Outcome | Calculation | Return |
|---|---|---|
| Safe success | −1 − 0.9 + 0.81 × 10 | 6.2 |
| Shortcut success | −1 + 0.9 × 10 | 8.0 |
| Shortcut failure | −1 + 0.9 × (−6) | −6.4 |

**Step 2: average over the environment's uncertainty.**

```text
safe expected return     = 6.2
shortcut expected return = 0.7 × 8.0 + 0.3 × (−6.4)
                         = 5.6 − 1.92
                         = 3.68
```

Under this objective and these assumptions, choosing the safe route is better. It is
not better merely because it is called “safe.” The comparison follows from the rewards,
timing, and probabilities we specified.

**Step 3: evaluate a stochastic route policy.**

If the policy chooses the safe route 80% of the time, its expected return is:

```text
0.8 × 6.2 + 0.2 × 3.68 = 5.696
```

The outer average is over the policy's action choice. The shortcut value already contains
an inner average over environment outcomes. Separating these sources of randomness will
help when reading Bellman expectation equations later.

**Step 4: change one assumption.**

If shortcut success rises to 0.9, its expected return becomes 0.9 × 8 + 0.1 × (−6.4) = 6.56.
The expected-return preference changes. A separate reliability requirement could still
rule out the shortcut; that requirement is additional to maximizing this scalar objective.

**Check your understanding:** after one successful shortcut trip, can you conclude that
the shortcut is better than the safe route?

<details>
<summary>Explain the answer</summary>

No. You observed return 8 on one trip, but that does not remove the failure branch.
Choosing between policies requires the relevant outcome distribution or an estimate of
it, with attention to uncertainty. The best observed episode is not the expected result.

</details>

### Evaluate a policy as a repeatable procedure

Freeze the learned route rule and evaluate it on new initial conditions generated by a declared rule. Keep the training seed, environment seed, and evaluation seed conceptually separate. Record the number of evaluation episodes before looking at the result; stopping evaluation immediately after a successful streak biases the report.

For binary completion, ten successes out of ten demonstrate those ten outcomes. They do not establish perfect reliability. A stronger report includes the task distribution, sample count, failures, and uncertainty. If the task distribution changes from familiar streets to unfamiliar buildings, a new evaluation question has been introduced.

This is the first recurring course discipline: specify the target behavior, identify the evidence, and state what that evidence leaves unresolved. A polished trajectory is useful for debugging and explanation; an evaluation protocol supports a claim about performance.

<a id="section-1-10"></a>

## 1.10 From the delivery robot to reasoning agents

The interface changes in modern AI systems, but the same questions remain: what can the
policy observe, what does it choose, what feedback arrives, and how does that feedback
change subsequent behavior?

| Delivery concept | Language-model or tool-agent counterpart |
|---|---|
| Current situation | Prompt, generated prefix, and available tool observations |
| Next movement | Next token, structured tool choice, or argument selection |
| Route | Generated response or multi-turn tool trajectory |
| Successful delivery | Checked answer, passed task tests, or preference feedback |
| Movement cost | Tool calls, latency, tokens, or another explicit resource measure |
| Routing policy | A distribution over tokens or tool actions |

Autoregressive token generation can be represented as sequential action selection:
each token changes the prefix used to choose the next token. This interpretation does
not mean ordinary next-token supervised training is automatically RL. The training
objective and feedback determine that distinction.

In a PPO-based RLHF setup, generated responses receive reward-model feedback and the
policy is updated with an RL objective. DPO takes a different route: it learns from
preference pairs using a direct objective related to KL-regularized reward optimization.
In GRPO-style reasoning experiments, several attempts at the same prompt can supply
group-relative feedback without a separately trained value model.

Do not collapse these into one generic “RL algorithm.” **RLHF describes a feedback
setup; PPO describes an optimizer; DPO describes a direct preference objective.**
Their relationship becomes clearer once policies, returns, and credit assignment are
familiar. The quick-read course develops those distinctions in Topics 36–39.

A tool-using agent adds another layer: an action can execute a tool, and the tool result
becomes a new observation. A system that always follows a fixed tool sequence is not
necessarily learning. In the course capstone, the routing policy is actually updated
from outcomes, while the arithmetic tools themselves remain ordinary Python functions.

> **Failure Mode — optimizing the evaluator:** if an agent can earn reward without
> achieving the actual task, learning may amplify that shortcut. Always distinguish
> what the training evaluator scores from what a separate task check verifies.

**Check your understanding:** what is missing from the statement “this system uses tools,
therefore it uses reinforcement learning”?

<details>
<summary>Explain the answer</summary>

Evidence that a policy is learned or improved through an appropriate reward-driven
interaction objective. Tool use alone describes behavior or an interface, not a training
method. A prompted or scripted agent can use tools without any RL update.

</details>

<a id="practice"></a>

## Practice and readiness check

Write a task contract for a system you know. Include:

1. The agent/environment boundary and the decisions being learned.
2. What the agent observes and what remains hidden.
3. The available actions and any hard restrictions.
4. The reward for each relevant outcome and how reward timing matters.
5. The episode boundary and the evaluation procedure.
6. One behavior that could exploit the reward without satisfying the intended task.

Then explain why the task is sequential, whether fresh interaction is available, and
which measurement would establish improvement. There is no need to pick PPO or another
algorithm before settling those questions.

**Notebook connections:**

- [Notebook 01: MDPs and dynamic programming](../../notebooks/01_mdp_dynamic_programming.ipynb) makes a decision model explicit. Return after studying the planning topics if its equations are unfamiliar.
- [Notebook 08: One Problem, Many Algorithms](../../notebooks/08_one_problem_many_algorithms.ipynb) shows why prediction, planning, and control require different comparisons.
- [Notebook 11: Tool-Agent Capstone](../../notebooks/11_tool_agent_capstone.ipynb) is a later application of the same task-contract and evaluator questions.

**Further reading:** [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 1](http://incompleteideas.net/book/the-book-2nd.html).
For modern methods, use the source links in [Topic 36](../../quick-read/36-reinforcement-learning-from-human-feedback.md),
[Topic 38](../../quick-read/38-direct-preference-optimization.md), and [Topic 39](../../quick-read/39-grpo-and-reinforcement-learning-for-reasoning.md).

You are ready to continue when you can distinguish **reward from return, observation
from state, policy from learning algorithm, and one successful episode from expected
performance**—and explain each distinction with your own example.

---

[← Chapters](../README.md) | [Quick-read Topic 1](../../quick-read/01-what-is-reinforcement-learning.md) | [Chapter 2 →](../02-the-agent-environment-interaction/README.md)
