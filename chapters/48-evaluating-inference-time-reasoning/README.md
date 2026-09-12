# 48. Evaluating Inference-Time Reasoning

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 47](../47-learning-from-search/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Part III exercises →](../PART_III_EXERCISES.md)

[Quick read](../../quick-read/48-evaluating-inference-time-reasoning.md) · [Notation](../NOTATION.md)

- [48.1 Evaluate the whole inference procedure](#section-48-1)
- [48.2 Specify the comparison before running it](#section-48-2)
- [48.3 Define the metrics and denominators](#section-48-3)
- [48.4 Explain why final accuracy can hide invalid reasoning](#section-48-4)
- [48.5 Use paired task comparisons](#section-48-5)
- [48.6 Avoid test-driven evaluator tuning](#section-48-6)
- [48.7 Interpret more-compute curves cautiously](#section-48-7)
- [48.8 Design a failure-oriented ablation](#section-48-8)
- [48.9 Produce a reproducible capstone artifact](#section-48-9)
- [48.10 Exercises, answers, and reading](#section-48-10)

---

<a id="section-48-1"></a>

## 48.1 Evaluate the whole inference procedure

A reasoning benchmark should test the system that will actually select and return an answer. Candidate generation, scoring, search, stopping, and final parsing all affect the result.

**Prerequisites:** Chapters 13, 37, and 41–47. **Learning targets:** build a paired comparison, separate oracle opportunity from deployed success, quantify uncertainty, and write a claim supported by the experiment.

The strongest report does not merely show a rising curve. It explains what changed, what information was available, which costs were counted, and what plausible alternative explanation remains.

<a id="section-48-2"></a>

## 48.2 Specify the comparison before running it

Fix task generation, split rules, proposal policy, decoding distribution, scorer, algorithm settings, seeds, budgets, and primary metrics. Declare whether the experiment matches candidate sets, resource caps, actual compute, or some combination.

The Part III default uses 100 unique arithmetic tasks, three generation seeds, several work caps, and the same frozen proposal distribution. Search strategies consume random draws differently even with matching task/seed identifiers, so shared seeds do not imply identical trajectories across every algorithm.

<a id="section-48-3"></a>

## 48.3 Define the metrics and denominators

Selected-answer correctness is the fraction of all tasks whose returned answer is correct. Process validity requires the entire selected trace to satisfy the arithmetic contract. Completion rate includes any complete candidate selected, whether correct or not.

Candidate oracle success asks whether at least one completed candidate is correct. It cannot be lower than selected correctness when the selector returns a member of that set. Proposal steps and scorer queries describe cost separately; aggregate work applies the declared score-cost multiplier.

<a id="section-48-4"></a>

## 48.4 Explain why final accuracy can hide invalid reasoning

Arithmetic errors can cancel, producing a correct total from an invalid trace. The simulator records both final and process metrics precisely to expose this distinction. A claim about reliable intermediate reasoning requires evidence beyond answer accuracy.

In natural language, a written explanation may not reveal the internal causal process that produced the answer. Process checks evaluate the observable trace under their rubric; they do not automatically establish faithful access to internal computation. Keep that limitation separate from ordinary arithmetic validity.

<a id="section-48-5"></a>

## 48.5 Use paired task comparisons

For two systems evaluated on the same tasks, define a per-task difference in success or cost. Summarizing these differences respects the shared task difficulty. Repeated generation seeds measure sampling variability conditional on the task set.

If a confidence interval resamples tasks, retain all compared methods and seed results together for each task. If generalization across training runs is the claim, additional independent trained policies are required. The demo's three seeds are generation seeds for one fixed policy, not three independent training runs.

<a id="section-48-6"></a>

## 48.6 Avoid test-driven evaluator tuning

Use validation tasks to choose scorer prompts, thresholds, beam widths, and cost coefficients. Reserve final held-out tasks for the reported comparison. Repeatedly inspecting final failures and adjusting the method makes that set part of development.

The shipped report is a controlled instructional experiment with declared settings, not a leaderboard submission or independently held-out benchmark campaign. Learners extending it should create their own validation/final split and record the selection procedure.

<a id="section-48-7"></a>

## 48.7 Interpret more-compute curves cautiously

Increasing budget can improve candidate opportunity while leaving selected success flat or worse. A beam with fixed width and branching can plateau because its tree stops growing. A misleading scorer can make stronger selection systematically favor mistakes.

Plot actual average work as well as budget caps, and inspect incomplete runs. A high budget label on a method that spends only a few units is not evidence that it used additional compute ineffectively; it may have no mechanism to spend that allowance.

<a id="section-48-8"></a>

## 48.8 Design a failure-oriented ablation

Change one evaluator property while holding the proposal policy and task distribution fixed. In a candidate-matched experiment, feed the exact same candidate list to different selectors. This isolates ranking effects from changes in sampling.

Then perform a budget-matched search comparison, where scorer cost affects how many branches can be explored. These experiments answer different questions. Report both if claiming that a better evaluator improves the practical system as well as its ability to rank a fixed set.

<a id="section-48-9"></a>

## 48.9 Produce a reproducible capstone artifact

Run the [demo](../../demos/inference-time/README.md), save the interactive HTML and raw JSON, and accompany them with the experiment contract and interpretation. Use Notebooks [12](../../notebooks/12_sampling_and_selection.ipynb), [13](../../notebooks/13_budgeted_reasoning_search.ipynb), and [14](../../notebooks/14_learning_from_search.ipynb) for candidate selection, search, and the separate learning stage.

The [Part III assessment](../PART_III_EXERCISES.md) requires calculations, a scorer-failure experiment, cost sensitivity, and a written conclusion. A strong submission explains an unsuccessful strategy rather than silently dropping it from the comparison.

<a id="section-48-10"></a>

## 48.10 Exercises, answers, and reading

1. A method completes 80 of 100 tasks and answers 60 correctly. Give overall success and accuracy conditional on completion.
2. Oracle candidate success is 0.8 and selected success is 0.5. What does the gap establish?
3. Why do three generation seeds not establish robustness to model training variation?

<details><summary>Worked answers</summary>

1. Overall success is 60%; conditional accuracy is 75%. Both require the stated denominators.
2. Correct candidates were available on more tasks than the selector solved. The gap motivates selector analysis, but does not by itself identify which scoring failure caused it.
3. The parameters remain identical across those runs. Only sampling randomness varies; different trained checkpoints require a different experimental design.

</details>

Read [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314), [Self-Consistency](https://arxiv.org/abs/2203.11171), and [WebArena](https://arxiv.org/abs/2307.13854) for contrasting task and evaluation designs. Attribute findings to their actual settings rather than combining them into a universal inference-scaling guarantee.

---

[← Chapter 47](../47-learning-from-search/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Part III exercises →](../PART_III_EXERCISES.md)
