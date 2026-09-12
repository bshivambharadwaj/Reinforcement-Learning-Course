# Part III exercises and practical assessment

**Created by Shivam Bharadwaj** · [Chapters](README.md) · [Course home](../README.md) · [Demo](../demos/inference-time/README.md)

Complete the calculations before opening the answers. The coding investigations extend
the shipped notebooks; no single winning learning curve is required. State the hypothesis,
resource model, and result that would contradict your explanation before running an ablation.

## 1. Candidate opportunity versus selection

Use [Chapters 41–43](41-from-policy-learning-to-inference-time-planning/README.md) and
[Notebook 12](../notebooks/12_sampling_and_selection.ipynb).

1. With independent success probability 0.2, calculate the chance of at least one success
   among one, four, and eight candidates.
2. At trajectory depth three, scorer cost two, and budget 30, calculate how many candidates
   best-of-N and self-consistency can process under the demo accounting rule.
3. A fixed pool contains five candidates, two correct. Calculate the probability that a
   uniformly selected pair contains a correct candidate.
4. Construct a candidate-matched experiment using the exact same traces for voting and
   scoring. Report selected correctness and total cost separately.
5. Replace default offset probabilities with `(0.1,0.8,0.1)`. Does voting still fail as the
   sample count grows? Compare empirical outcomes with the induced answer distribution.

<details><summary>Worked calculation answers</summary>

1. 0.2, 1−0.8⁴=0.5904, and 1−0.8⁸≈0.83223. These are oracle candidate opportunities.
2. Best-of-N fits floor(30/5)=6; self-consistency fits floor(30/3)=10. Different scoring
   access and cost explain the candidate-count difference.
3. 1−choose(3,2)/choose(5,2)=0.7. This is not the probability that a fallible verifier
   chooses the correct member of that pair.

For the coding tasks, retain all planned generation seeds. Explain a changed ranking
through the changed distribution or selector rather than claiming a universal result.

</details>

## 2. Scorer validity and exploitation

Use [Chapter 44](44-verifiers-process-rewards-and-value-models/README.md) and
[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb).

1. Candidate correctness prevalence is 0.1, verifier true-positive rate 0.8, and false-positive
   rate 0.05. Calculate correctness among accepted candidates under those assumptions.
2. For operands `(1,1,5)`, evaluate trace `(3,7)` for local errors, final correctness, and
   process validity. Explain why the metrics disagree.
3. Run best-of-N with exact-process and misleading scoring on identical candidate pools.
   Identify at least one case where a correct candidate exists but the selected answer fails.
4. Add a noisy scorer using a separate random stream. Report selected quality, false positives,
   and scorer-query count as you vary noise. Do not use final evaluator labels to select answers.

<details><summary>Worked calculation answers</summary>

1. 0.08/(0.08+0.9×0.05)=0.64. The rate assumptions must apply to the candidate distribution.
2. Local errors are +1 and −1. Final answer 7 equals the requested sum, but the first
   correct cumulative value should be 2, so the process is invalid.

The failure experiment should show both the scored candidates and the chosen trace.
A high proxy score alone does not establish that the scorer measures the task correctly.

</details>

## 3. Search and resource accounting

Use [Chapters 45–46](45-search-over-reasoning-trajectories/README.md),
[Notebook 13](../notebooks/13_budgeted_reasoning_search.ipynb), and the demo CLI.

1. Derive full-tree proposal and scoring counts for depth three, beam width two, branching
   two, and scorer cost one. Compare with a complete binary tree explored by best-first.
2. At a budget of 12, explain why the configured beam can fail to reach any complete leaf.
3. Reevaluate at score costs one and three under the same caps. Plot correctness against
   both configured caps and actual average cost.
4. Change beam width while holding the cap fixed. Record completion, not just correctness
   conditional on completion.
5. Design an early-stopping rule from an observable signal. Explain what it assumes about
   the scorer and evaluate the saved work versus any lost quality.

<details><summary>Worked calculation answers</summary>

1. Beam generates 2+4+4=10 steps and queries ten scores, costing 20. The complete binary
   depth-three tree contains 2+4+8=14 generated edges; scoring each gives 28 units.
2. The first layer costs four units and the next costs eight, exhausting 12 before the
   third layer. With no completed candidate, the run abstains.

The configured tree can be exhausted even when budget remains. More allowance is useful
only if the method has an additional computation it can perform.

</details>

## 4. Learning from accepted search traces

Use [Chapter 47](47-learning-from-search/README.md) and
[Notebook 14](../notebooks/14_learning_from_search.ipynb).

1. For offset counts `[2,8,0]`, derive the student probabilities with unit pseudocounts.
2. Compare outcome-only and process-valid filtering. Report accepted tasks, retained action
   counts, teacher-generation cost, and held-out single-sample student performance.
3. Explain why this supervised count update is neither GRPO nor LLM fine-tuning.
4. Choose teacher budgets on validation tasks and reserve a separate final set. Report
   the split rule and show that task tuples do not overlap.
5. If data generation costs 2,400 units and serving costs fall from 24 to 3 units per task,
   calculate the simple search-cost break-even point. State the missing real-world costs.

<details><summary>Worked calculation answers</summary>

1. `[3/13,9/13,1/13]`. Pseudocount smoothing maintains positive support.
3. It maximizes categorical likelihood of selected actions through counts. There are no
   PPO ratios, group-relative policy-gradient updates, neural parameters, or token losses.
5. 2,400/21≈114.29, so 115 tasks exceed this overhead. Actual model training, verification,
   hardware efficiency, maintenance, and acceptable quality must be included for a deployment claim.

</details>

## 5. End-to-end practical submission

Reproduce the [demo report](../demos/inference-time/README.md), then change one meaningful
factor. Submit a short report with a runnable command, raw data, and an interactive HTML
artifact or notebook. Include:

1. The task contract, frozen policy, scorer, selection/search rule, and stopping behavior.
2. A single-sample baseline and at least two candidate/search strategies.
3. A candidate-matched selector comparison and a separately labeled budget-matched comparison.
4. Selected correctness, valid process, candidate oracle opportunity, completion, and actual costs.
5. All declared generation seeds and one inspected failure trajectory.
6. A scorer-quality ablation and a scorer-cost sensitivity experiment.
7. A clear distinction between this simulator and any real-model claims or proposed extensions.

If proposing a real LLM version, specify the checkpoint, decoding distribution, token and
tool boundaries, verifier access, held-out tasks, and measured resource costs. That proposal
is separate from the shipped simulator implementation.

## Rubric

| Criterion | Weight | Required evidence |
|---|---:|---|
| Formal distinctions and calculations | 20% | Correct state, objective, probability, and budget definitions |
| Implementation integrity | 20% | No evaluator leakage, no budget overruns, explicit abstention |
| Experimental comparison | 25% | Controlled tasks, labeled matching criteria, complete seed results |
| Failure analysis | 20% | An explained wrong selection or incomplete search and a targeted ablation |
| Reproduction and communication | 15% | Commands, raw outputs, readable plots/report, bounded claims |

The exercises are original course material. Text and reports use [CC BY 4.0](../LICENSE-CC-BY-4.0);
code uses [MIT](../LICENSE-MIT). External papers retain their own licenses.
