# 41. From Policy Learning to Inference-Time Planning

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 40](../40-multimodal-rl-and-rl-for-ai-agents/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 42 →](../42-reasoning-as-a-sequential-decision-problem/README.md)

[Quick read](../../quick-read/41-from-policy-learning-to-inference-time-planning.md) · [Notation](../NOTATION.md)

- [41.1 Two different ways to improve an answer](#section-41-1)
- [41.2 Define the complete inference system](#section-41-2)
- [41.3 Map the RL objects carefully](#section-41-3)
- [41.4 Derive the gain available from independent attempts](#section-41-4)
- [41.5 Separate probability from quality](#section-41-5)
- [41.6 Distinguish search from online learning](#section-41-6)
- [41.7 Account for the resources consumed](#section-41-7)
- [41.8 Trace a simple budget allocation](#section-41-8)
- [41.9 Laboratory and connection to the next chapters](#section-41-9)
- [41.10 Exercises, answers, and reading](#section-41-10)

---

<p align="center"><img src="../../assets/diagrams/inference-time-planning.svg" width="760" alt="A frozen proposal policy, a scored search loop, and independent final evaluation"></p>

<a id="section-41-1"></a>

## 41.1 Two different ways to improve an answer

During reinforcement learning, feedback changes a policy's parameters. During inference-time planning, a fixed policy can propose several continuations while another procedure decides which to inspect, continue, or return. These are distinct sources of improvement.

**Prerequisites:** Chapters 25, 34, and 39. **Learning targets:** distinguish parameter learning from search, identify the complete inference system, and compare strategies without hiding their compute costs.

Think of a courier consulting an unchanged route manual. Trying several routes in a simulator and choosing one can improve today's decision without rewriting the manual. Updating the manual afterward is a separate learning stage.

<a id="section-41-2"></a>

## 41.2 Define the complete inference system

A useful system specification includes a proposal policy pi, an evaluator e, a search procedure S, a stopping rule, and a budget B. The returned answer is a function of all these objects and the task input, not just the policy weights.

Two systems with identical model checkpoints can behave differently because one samples once while the other scores sixteen candidates. Conversely, replacing the evaluator can change results while holding the model and search algorithm fixed. Record each component when attributing a gain.

<a id="section-41-3"></a>

## 41.3 Map the RL objects carefully

The state can contain the task, generated prefix, tool observations, and remaining budget. An action might be a token, a complete reasoning step, a tool call, or a search decision. The policy proposes actions conditioned on its available information.

A reward is part of a specified objective. A verifier score may estimate correctness or provide a proxy. A value estimate predicts a future quantity under a particular continuation procedure. Search decides which computations to perform next. Giving a heuristic the name value does not establish that it estimates expected return.

<a id="section-41-4"></a>

## 41.4 Derive the gain available from independent attempts

If independent complete attempts succeed with probability p, the probability that at least one of N succeeds is 1−(1−p)^N. At p=0.2 and N=4, this is 0.5904.

This is an oracle opportunity: a correct candidate exists in the sample set. A deployed selector must identify it. If the selector cannot distinguish correct from incorrect answers, actual selected-answer success can be much lower. Correlated attempts also invalidate the simple independence calculation.

<a id="section-41-5"></a>

## 41.5 Separate probability from quality

Policy likelihood describes how the model distributes probability under its decoding procedure. It does not directly measure factual correctness or usefulness. In a three-step task, the most probable local choice can repeatedly introduce the same mistake.

The Part III simulator makes this explicit: offsets −1, 0, and +1 have probabilities 0.1, 0.4, and 0.5. The modal proposal is a +1 arithmetic error. Greedy local selection would confidently accumulate errors. This is a deliberately constructed teaching case, not an empirical model of every LLM.

<a id="section-41-6"></a>

## 41.6 Distinguish search from online learning

Changing a prompt, maintaining memory, generating alternatives, or calling a tool does not by itself establish an RL parameter update. A frozen inference run can adapt its actions to observations while its learned parameters remain unchanged.

If a system updates parameters using task feedback at test time, specify that additional training process, its data access, and its compute. Do not compare it with frozen inference as if both used identical information. Adaptation of behavior and optimization of parameters are related concepts but not interchangeable definitions.

<a id="section-41-7"></a>

## 41.7 Account for the resources consumed

Report proposed tokens or steps, evaluator queries, completed candidates, tool calls, and latency where relevant. A fixed maximum budget is an allowance; methods may use different amounts of it. Single sampling should not be charged fictional extra attempts merely to make a table appear equal.

The demo charges one unit per proposed arithmetic step and a configurable cost per score query. These teaching units make accounting inspectable but do not represent equal FLOPs across real language and reward models. A production comparison must measure its actual resource model.

<a id="section-41-8"></a>

## 41.8 Trace a simple budget allocation

Suppose a complete candidate takes three proposals and a final scoring query costs one unit. With B=12, best-of-N can generate and score three complete candidates. Self-consistency can generate four complete candidates because it does not query that scorer. Single sampling generates one and spends three units.

These methods share a budget ceiling, not an identical number of attempts. If scoring instead costs three units, best-of-N fits only two candidates. An algorithm's ranking can change when the evaluator's cost changes.

<a id="section-41-9"></a>

## 41.9 Laboratory and connection to the next chapters

[Notebook 12](../../notebooks/12_sampling_and_selection.ipynb) compares frozen-policy candidate strategies and distinguishes oracle candidate success from selected success. Inspect the actual cost columns before comparing accuracy.

Use the [interactive demo](../../demos/inference-time/README.md) to switch between a correct local checker and a deliberately misleading scorer. The policy stays fixed. Changes in outcome therefore arise from sampling, selection, search, or evaluator behavior, rather than training a better proposal policy during that run.

<a id="section-41-10"></a>

## 41.10 Exercises, answers, and reading

1. With p=0.3 and four independent attempts, calculate oracle candidate success.
2. At depth 4, score cost 2, and budget 24, how many scored complete candidates fit?
3. Why does a fixed checkpoint not completely specify an inference system?

<details><summary>Worked answers</summary>

1. 1−0.7^4=0.7599. This does not guarantee a selector returns the successful candidate.
2. Each costs 6 units, so four fit. Scoring overhead is part of the budget.
3. Decoding, evaluator, search, tool interface, and stopping rules also determine which answer is returned and which resources are spent.

</details>

Read [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) for experimental context. Keep its model- and task-specific findings distinct from the independent-attempt calculation and the course's simulator.

---

[← Chapter 40](../40-multimodal-rl-and-rl-for-ai-agents/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 42 →](../42-reasoning-as-a-sequential-decision-problem/README.md)
