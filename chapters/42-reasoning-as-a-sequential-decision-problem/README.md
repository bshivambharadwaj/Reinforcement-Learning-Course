# 42. Reasoning as a Sequential Decision Problem

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 41](../41-from-policy-learning-to-inference-time-planning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 43 →](../43-sampling-and-candidate-selection/README.md)

[Quick read](../../quick-read/42-reasoning-as-a-sequential-decision-problem.md) · [Notation](../NOTATION.md)

- [42.1 Choose the unit of reasoning](#section-42-1)
- [42.2 Define a state that supports the intended prediction](#section-42-2)
- [42.3 Specify the arithmetic demonstration](#section-42-3)
- [42.4 Separate transitions from truth checks](#section-42-4)
- [42.5 Represent termination and abstention](#section-42-5)
- [42.6 Include the budget in a computation policy](#section-42-6)
- [42.7 Distinguish trees, graphs, and merged states](#section-42-7)
- [42.8 Specify the action and evidence boundary for tools](#section-42-8)
- [42.9 Laboratory investigation](#section-42-9)
- [42.10 Exercises, answers, and reading](#section-42-10)

---

<a id="section-42-1"></a>

## 42.1 Choose the unit of reasoning

A sequential model of reasoning begins by defining what one decision means. A token, a sentence, a symbolic operation, and a tool call have different durations, branching factors, and verification requirements.

**Prerequisites:** Chapters 2–4 and 41. **Learning targets:** specify reasoning states and actions, distinguish environment state from search state, and define terminal success without relying on the agent's self-report.

The courier's physical move differs from the planner's decision to inspect a route. Inference systems likewise have object-level actions inside a candidate and computation-level actions that allocate search effort.

<a id="section-42-2"></a>

## 42.2 Define a state that supports the intended prediction

For a text-only autoregressive model with fixed parameters, the token prefix supplies its generation context. For an external task, that prefix may omit tool state, earlier observations removed by truncation, or permissions that affect valid actions.

A reasoning state should include the information needed for the chosen transition and reward model. A search controller may additionally need the frontier, explored candidates, cached scores, and remaining budget. The best next search computation cannot generally be chosen from one prefix alone.

<a id="section-42-3"></a>

## 42.3 Specify the arithmetic demonstration

ArithmeticTrace-v1 asks for left-to-right addition of a tuple of integers. For (2,3,4,5), a valid trace contains claimed cumulative sums (5,9,14). Three decisions are needed; an incomplete trace is not a completed answer.

The proposal policy calculates a local sum from its claimed predecessor and the next operand, then adds a sampled offset. This deliberately exposes how earlier mistakes affect later states. Arithmetic is built into the simulator; performance on new operands does not show that a language model learned arithmetic.

<a id="section-42-4"></a>

## 42.4 Separate transitions from truth checks

Appending a proposed sum deterministically creates the next trace state. That transition is available even if the sum is incorrect. The process checker evaluates local consistency; an independent final evaluator checks the requested total and all cumulative ground-truth values.

For (1,1,5), trace (3,7) contains a +1 error followed by a −1 error. Its final answer is correct, but the process is invalid. A final-only objective accepts a different set of trajectories from an objective requiring every arithmetic step to be valid.

<a id="section-42-5"></a>

## 42.5 Represent termination and abstention

Termination occurs after the required number of reasoning actions. Search can stop earlier because its budget is exhausted, but this does not make an incomplete prefix a completed solution. The implementation returns an explicit abstention when no complete candidate is available.

Record completion rate separately from correctness conditional on completion. A method that answers only easy tasks can have high conditional accuracy and poor overall success. In the course reports, abstentions count as unsuccessful tasks in the main correctness metric.

<a id="section-42-6"></a>

## 42.6 Include the budget in a computation policy

Let m describe the current search memory and b the remaining compute allowance. A computation action c can expand a prefix, score a candidate, or stop. Its value depends on the result distribution, cost, and what further computations remain affordable.

Conceptually, V(m,b) can compare the utility of stopping now with E[V(m',b−cost(c))] over legal affordable computations. This is a metareasoning model, not the token policy's ordinary state value. The demo uses fixed search rules rather than solving this computation-control problem optimally.

<a id="section-42-7"></a>

## 42.7 Distinguish trees, graphs, and merged states

Different reasoning histories can reach the same numeric intermediate answer. Merging them saves work only if the retained state preserves everything relevant to later decisions and evaluation. A process-valid history and an invalid history with the same current sum may need different treatment.

If the objective only depends on final answer and future transitions depend solely on the current sum and remaining operands, some merging may be legitimate. If process validity is part of the task, retain that information. State abstraction changes which guarantees a search algorithm can claim.

<a id="section-42-8"></a>

## 42.8 Specify the action and evidence boundary for tools

A generated tool call is a proposed action; the tool result is external evidence. If a wrapper repairs arguments, retries failures, or filters actions, include those mechanisms in the executed system's contract and cost accounting.

Search over hypothetical calls in a simulator differs from executing calls with real side effects. A branch that writes to a live system may change the environment for other branches. Independent search-tree semantics require isolation, reversibility, or an explicit model of shared effects.

<a id="section-42-9"></a>

## 42.9 Laboratory investigation

In [Notebook 13](../../notebooks/13_budgeted_reasoning_search.ipynb), inspect the recorded prefix, proposed child, score, and cumulative cost for each expansion. Reconstruct one selected trace without reading the final evaluator result first.

Then compare final-answer correctness with process validity. Add an explicit compensating-error trace and confirm the two metrics disagree. This small counterexample is more informative than assuming that every correct final number validates the path that produced it.

<a id="section-42-10"></a>

## 42.10 Exercises, answers, and reading

1. For operands (2,4,1), evaluate trace (7,7) for final and process correctness.
2. Why may two identical prefixes require different search decisions at budgets 2 and 20?
3. What extra state is needed if process validity is an evaluation requirement?

<details><summary>Worked answers</summary>

1. The expected total is 7, so the answer is correct. The first correct cumulative sum is 6, so the process fails; the trace's errors cancel.
2. The feasible future computation sequences differ. A costly expansion may be useful only when enough budget remains to complete or verify a candidate.
3. Retain the history or a sufficient summary of whether earlier steps were valid, along with any evidence the continuation policy needs.

</details>

Read [Tree of Thoughts](https://arxiv.org/abs/2305.10601) for a concrete framework using larger reasoning units and search. Its thought-generation and evaluation choices are specific implementations, not a universal state definition.

---

[← Chapter 41](../41-from-policy-learning-to-inference-time-planning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 43 →](../43-sampling-and-candidate-selection/README.md)
