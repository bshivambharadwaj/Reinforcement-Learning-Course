# Notation and reading guide

[Chapters](README.md) · [Course home](../README.md) · [Assessment](ASSESSMENT.md)

**Created by Shivam Bharadwaj**

Use this page as a reference while reading. The chapters use plain mathematical notation
in text and code blocks so GitHub can display it without renderer-dependent macros.
Existing SVG diagrams supply visual explanations; a code block containing mathematics
is notation, not necessarily executable Python.

## Objects and their meanings

| Symbol | Meaning | Question to ask |
|---|---|---|
| S_t, s | Random state at time t; a particular state | Is this fully observed and sufficient for prediction? |
| O_t, o | Observation | What state information is hidden? |
| A_t, a | Random action; a particular action | Is it legal, and what is its duration? |
| R_(t+1), r | Reward after action A_t | Does it belong to the stored transition? |
| p(s',r\|s,a) | Joint successor/reward distribution | Are correlated outcomes modeled correctly? |
| P(s'\|s,a) | Successor-state distribution | Do available-action rows normalize? |
| rho(s) | Initial-state distribution | Which tasks does the reported average represent? |
| pi(a\|s), pi(a\|h) | Policy conditioned on state or history | Which information is available when deciding? |
| b(a\|s) | Behavior policy used to collect data | Does it support the target policy's actions? |
| gamma | Discount factor | Is it part of the objective or an explicitly described approximation? |
| G_t | Discounted return starting after decision t | Where does the episode or bootstrap end? |
| V_pi(s) | Expected return under policy pi from s | Which continuation policy is being evaluated? |
| Q_pi(s,a) | Expected return after forcing a, then following pi | Is the first action distinguished from later behavior? |
| A_pi(s,a) | Q_pi(s,a)−V_pi(s) | Is this exact advantage or an estimate? |
| T_pi, T_star | Expectation and optimal Bellman operators | Is a backup exact, sampled, or fitted? |
| theta, w | Policy/value or other learned parameters | Which loss updates which parameters? |
| alpha | Step size, or entropy coefficient when explicitly defined | What local definition is in force? |
| lambda | Trace/GAE mixing parameter | How far does credit propagate through residuals? |
| beta | Reference-KL or preference-objective coefficient | What reward/logit scale accompanies it? |
| pi_old | Rollout-generating policy snapshot | Is it frozen across the update batch? |
| pi_ref | Reference policy defining an anchor | Which checkpoint and stage does it represent? |
| E[X\|s,a] | Conditional expectation | Which randomness is still averaged? |
| ||v||_infinity | Largest absolute vector entry | Is the bound all-state or only empirical? |

Symbols are reused in the literature. Every chapter defines the local meaning when a
symbol has more than one common role. A probability ratio and a reward should not both
be called `r` in implementation code without enough context to distinguish them.

## Time and boundary conventions

At time t the agent observes its available information, chooses A_t, then receives
R_(t+1) and the next observation. The terminal transition's reward is included. Rewards
after true termination contribute zero.

```text
G_t = R_(t+1) + gamma R_(t+2) + gamma² R_(t+3) + ...
```

In some pseudocode, `reward[t]` stores R_(t+1): the array index labels the action's
transition, not the mathematical reward subscript. Check this correspondence before
translating equations into code.

A true termination removes continuation. A collection cutoff may retain a bootstrap
from the final observation. A reset starts a different trajectory. GAE can therefore
need separate bootstrap and trace masks. If a deadline is part of the task, include
remaining time in the state or use time-indexed values.

## Read a theorem by checking its domain

The finite discounted tabular setting supplies several clean guarantees: Bellman
contraction, unique fixed points, and exact policy improvement. These results usually
assume finite states/actions, bounded rewards, an appropriate Markov representation,
gamma<1, and correct exact-model operators.

Finite-horizon gamma=1 problems can be well-defined through backward induction.
Undiscounted continuing problems require other assumptions and objectives. Sampling,
function approximation, nonlinear optimization, and changing data distributions add
steps that need their own analysis; an exact-operator proof does not cover them by name.

When reading a proof, mark:

1. The objects being compared and the norm or expectation used.
2. Every assumption used in an inequality.
3. The distinction between existence, convergence, rate, and finite-sample performance.
4. The quantity the implementation can actually measure.
5. A concrete case outside the assumptions where the conclusion may fail.

## Keep objectives and estimators separate

An objective defines what behavior is preferred. An estimator uses data to approximate
a quantity needed for optimization. A surrogate replaces part of the original objective
with something easier to optimize. A diagnostic measures some aspect of the process.

| Example | Role | What it does not establish alone |
|---|---|---|
| Expected task return | Objective/metric | Hard constraint satisfaction |
| TD target | Bootstrap estimator | Correct values with arbitrary approximation |
| PPO clipped surrogate | Local training objective | A hard KL bound or monotonic return |
| DPO pairwise loss | Preference objective | Good generated responses on unseen tasks |
| Reward-model score | Learned proxy | Independently verified correctness |
| Clip fraction | Optimization diagnostic | Global policy movement limits |
| Training loss | Fit diagnostic | Deployment performance |

For a discounted start-state policy objective, the score-function expression can carry
an outer gamma^t in addition to the reward-to-go's internal discounting. Algorithms
using other occupancy or episodic conventions should state them rather than silently
treating all formulas as identical.

## A practical study routine

Read the opening motivation, then write the chapter's target quantity in your own words.
Reproduce the numerical example on paper. Attempt the final problems before opening
their answers. Follow the notebook link and distinguish an existing implementation from
a proposed extension before planning the experiment.

For experiments, record a hypothesis and the result that would contradict it. Keep a
short error log: one mistaken assumption, one calculation correction, and one unresolved
question are more useful than a copied page of formulas. Return to the relevant earlier
chapter when a modern method reuses a familiar object under a new name.

## License

This guide and chapter prose use [CC BY 4.0](../LICENSE-CC-BY-4.0). Learner-facing code
uses [MIT](../LICENSE-MIT). Linked papers and external course materials retain their
own licenses and notices.
