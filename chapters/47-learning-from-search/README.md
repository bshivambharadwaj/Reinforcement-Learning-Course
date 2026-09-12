# 47. Learning from Search

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 46](../46-budget-aware-reasoning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 48 →](../48-evaluating-inference-time-reasoning/README.md)

[Quick read](../../quick-read/47-learning-from-search.md) · [Notation](../NOTATION.md)

- [47.1 Move useful search behavior into a later policy](#section-47-1)
- [47.2 Define the teacher-data pipeline](#section-47-2)
- [47.3 Write the imitation objective](#section-47-3)
- [47.4 Derive the categorical demonstration update](#section-47-4)
- [47.5 Compare outcome and process filtering](#section-47-5)
- [47.6 Understand selection bias and coverage](#section-47-6)
- [47.7 Separate training and test distributions](#section-47-7)
- [47.8 Calculate a compute break-even point](#section-47-8)
- [47.9 Laboratory and proposed extensions](#section-47-9)
- [47.10 Exercises, answers, and reading](#section-47-10)

---

<a id="section-47-1"></a>

## 47.1 Move useful search behavior into a later policy

Search can discover trajectories that a proposal policy rarely generates on its own. A separate training stage can use those trajectories to change a student policy, potentially reducing the inference computation needed later.

**Prerequisites:** Chapters 25, 38–39, and 41–46. **Learning targets:** separate teacher search from student learning, specify filtered imitation, and account for both training and inference costs.

The courier first consults a costly route planner, then revises its ordinary route manual from the successful plans. The planner and the manual update are separate stages with separate evidence requirements.

<a id="section-47-2"></a>

## 47.2 Define the teacher-data pipeline

Choose training tasks, run a documented search strategy, select trajectories with an available evaluator, and apply an acceptance rule. Record rejected tasks as well as accepted traces. Freeze this data-generation procedure before measuring final student performance.

If acceptance uses an exact verifier, that is privileged supervision that must be stated. In the simulator, exact arithmetic validation is inexpensive by construction. Real tasks may need costly human or executable checks, and some properties may remain unverifiable.

<a id="section-47-3"></a>

## 47.3 Write the imitation objective

For accepted trajectories, supervised imitation minimizes the negative sum of student action log probabilities conditioned on each recorded prefix. It learns to reproduce selected behavior rather than estimating a policy gradient from an on-policy reward distribution.

Filtering data by reward does not make the resulting maximum-likelihood update PPO or GRPO. RL fine-tuning, preference optimization, and filtered imitation can all use search-generated data, but their objectives and distribution assumptions differ.

<a id="section-47-4"></a>

## 47.4 Derive the categorical demonstration update

The simulator student learns three probabilities for local offsets −1, 0, and +1. Count their occurrences in accepted teacher traces. With unit pseudocounts, the fitted probability of offset j is (count_j+1)/(total_count+3).

If ten accepted three-step traces are entirely process-valid, counts are [0,30,0], giving probabilities [1/33,31/33,1/33]. The student is a new immutable policy object. No parameters changed during teacher search; the probability update occurs explicitly afterward.

<a id="section-47-5"></a>

## 47.5 Compare outcome and process filtering

Outcome filtering can accept traces with compensating errors. Process filtering requires every intermediate claim to be correct under the specified task. These datasets can teach different local behavior even when their final answers all pass.

For task (1,1,5), trace (3,7) contributes offsets [+1,−1]. Accepting it because the final answer is 7 reinforces both mistakes in an independent-offset student. A more expressive history-dependent student might model the correction sequence differently, so the consequence depends on the student representation.

<a id="section-47-6"></a>

## 47.6 Understand selection bias and coverage

Successful teacher traces may concentrate on easy tasks or narrow solution styles. Training only on them can improve average behavior on that subset while leaving difficult tasks underrepresented. Record acceptance rate by task family and the number of useful steps obtained per unit of teacher compute.

If no trajectories are accepted, the demo returns the explicitly smoothed categorical prior rather than pretending that learning succeeded. Report that condition. A larger optimizer budget cannot manufacture missing supervised examples from an empty accepted set.

<a id="section-47-7"></a>

## 47.7 Separate training and test distributions

Use disjoint training and held-out task tuples. For the course lab, training operands lie in [0,10) and test operands in [10,30). This avoids identical tasks, but the proposal mechanism already knows local arithmetic and learns only an error distribution.

Consequently, strong held-out performance demonstrates the simulator's filtering and policy-fitting mechanism, not general mathematical reasoning or language generalization. A real-model extension needs meaningful template/domain splits, model revisions, tokenizer details, and generated-output checks.

<a id="section-47-8"></a>

## 47.8 Calculate a compute break-even point

Suppose teacher-data generation and student training cost C_train units. The teacher inference system costs C_teacher per deployed task and the student costs C_student, with comparable required quality. A simple compute break-even count is C_train/(C_teacher−C_student) when the denominator is positive.

With C_train=12,000, C_teacher=48, and C_student=3, the threshold is about 266.67 tasks, so 267 tasks exceed it. This ignores hardware differences, maintenance, and quality mismatch. Do not claim amortized savings if the student fails the application's accuracy requirement.

<a id="section-47-9"></a>

## 47.9 Laboratory and proposed extensions

[Notebook 14](../../notebooks/14_learning_from_search.ipynb) generates teacher traces, compares outcome/process filtering, fits separate students, and evaluates each with single sampling and a frozen held-out protocol. It records the teacher-search budget instead of treating synthetic data as free.

A later real-model extension could imitate verified traces using the masking discipline from Notebook 09. That implementation is not included in this lab; the categorical example isolates the training/inference boundary with no model download.

<a id="section-47-10"></a>

## 47.10 Exercises, answers, and reading

1. With raw offset counts [2,8,0] and unit pseudocounts, calculate student probabilities.
2. Why is a process-filtered dataset not automatically representative of all tasks?
3. At C_train=900, C_teacher=12, and C_student=3, what is the break-even task count?

<details><summary>Worked answers</summary>

1. [3/13,9/13,1/13]. Smoothing preserves support but changes the exact unsmoothed maximum-likelihood estimate.
2. The teacher may generate valid traces more often for easy tasks, so acceptance changes the task and trajectory distribution.
3. 100 tasks match the training overhead under this simplified cost model; more than 100 yields net savings if quality is acceptable.

</details>

Read [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) for a real-model research example. The course's count-based student is a pedagogical analogue, not a reproduction of STaR.

---

[← Chapter 46](../46-budget-aware-reasoning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 48 →](../48-evaluating-inference-time-reasoning/README.md)
