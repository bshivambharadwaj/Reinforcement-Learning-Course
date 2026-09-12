# 44. Verifiers, Process Rewards, and Value Models

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 43](../43-sampling-and-candidate-selection/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 45 →](../45-search-over-reasoning-trajectories/README.md)

[Quick read](../../quick-read/44-verifiers-process-rewards-and-value-models.md) · [Notation](../NOTATION.md)

- [44.1 Different evaluators answer different questions](#section-44-1)
- [44.2 Outcome checks inspect a completed result](#section-44-2)
- [44.3 Process checks inspect intermediate claims](#section-44-3)
- [44.4 Value models require a continuation definition](#section-44-4)
- [44.5 Derive how false positives affect selected quality](#section-44-5)
- [44.6 Evaluate calibration and ranking separately](#section-44-6)
- [44.7 Analyze evaluator exploitation through selection](#section-44-7)
- [44.8 Define evidence independence operationally](#section-44-8)
- [44.9 Laboratory investigation](#section-44-9)
- [44.10 Exercises, answers, and reading](#section-44-10)

---

<a id="section-44-1"></a>

## 44.1 Different evaluators answer different questions

A final verifier, process scorer, reward model, and value model can all guide selection, but their outputs do not mean the same thing.

**Prerequisites:** Chapters 7, 9, 37, and 41–43. **Learning targets:** define each evaluator's target, distinguish local validity from future success, and test robustness under selection pressure.

The courier's receipt confirms delivery, a road-rule checker validates a maneuver, and an estimated arrival probability predicts a future event. A number from one cannot be interpreted as another without a model connecting them.

<a id="section-44-2"></a>

## 44.2 Outcome checks inspect a completed result

An executable checker can compare a final arithmetic answer with a computed total, run tests against a program, or inspect authoritative task state. The check still has a scope: passing a finite test suite does not prove a program correct for every input.

Separate the intended property from the observable test. Document answer parsing, timeout behavior, and whether a partially completed result can pass accidentally. The demo's outcome checker is exact for its finite arithmetic contract and is reserved for post-selection evaluation.

<a id="section-44-3"></a>

## 44.3 Process checks inspect intermediate claims

A process checker evaluates step validity under available evidence. In the simulator it checks whether a claimed sum equals the claimed predecessor plus the next operand. A later locally correct step can continue an already incorrect cumulative state.

The score is a fraction of observed valid steps. It is not the probability of eventual success, and comparing fractions at different depths can favor short easy prefixes. The search implementation documents its depth tie rule instead of disguising this score as a calibrated value.

<a id="section-44-4"></a>

## 44.4 Value models require a continuation definition

V(s) might mean probability that one ordinary continuation succeeds, expected reward under a specific search policy, or utility achievable with remaining budget b. These are different targets.

A prefix with 20% single-rollout success can have much higher oracle opportunity after many independent attempts, but the deployed selector may recover only part of that opportunity. Label training targets with the continuation policy, decoding settings, budget, and evaluation rule that produced them. Changing these can make an old value model miscalibrated.

<a id="section-44-5"></a>

## 44.5 Derive how false positives affect selected quality

Suppose candidate correctness prevalence is p. A binary verifier has true-positive rate t and false-positive rate f. Among accepted candidates, the expected correctness fraction is pt/[pt+(1−p)f], assuming these rates apply to that candidate distribution and the denominator is positive.

At p=0.1, t=0.9, and f=0.1, accepted correctness is 0.09/(0.09+0.09)=0.5. A verifier that sounds accurate can produce a weak accepted set when successes are rare. Adaptive search can also change p, t, and f by selecting unusual candidates.

<a id="section-44-6"></a>

## 44.6 Evaluate calibration and ranking separately

A scorer can rank candidates well but output poorly calibrated probabilities. Another can be calibrated on random samples yet rank the hardest optimizer-selected candidates poorly. Pair accuracy, calibration curves, and selected-answer success measure different properties.

Use held-out candidate sets from the relevant generator and search process. If the scorer is revised after inspecting failures, reserve another test split. Reusing the final benchmark to tune scorer prompts turns evaluation into development feedback.

<a id="section-44-7"></a>

## 44.7 Analyze evaluator exploitation through selection

If a score equals true quality plus error, maximizing it can select positive error. A larger candidate pool expands the opportunity to find both genuinely good outcomes and scoring loopholes.

In the deliberate bad-scoring condition, +1 arithmetic errors receive the highest reward. Stronger optimization of that score can make selected trajectories less correct. This experiment isolates objective mismatch: the search can be functioning exactly as specified while producing a worse task result.

<a id="section-44-8"></a>

## 44.8 Define evidence independence operationally

An evaluator is not independent merely because it has a different function name or model identifier. Check whether it shares training labels, shortcuts, prompts, or implementation errors with the optimizer's score.

The demo computes final truth from sum(operands) and cumulative ground-truth sums after selection. Search does not call this evaluator. Unit tests replace the evaluator with an exception during inference to detect direct leakage. That boundary does not turn the exact local process checker into a realistic learned verifier; its strong domain knowledge remains explicit.

<a id="section-44-9"></a>

## 44.9 Laboratory investigation

[Notebook 13](../../notebooks/13_budgeted_reasoning_search.ipynb) varies scorer quality and scoring cost independently. Compare a useful exact checker with the constructed wrong signal, then inspect traces selected under each.

Extend the experiment by adding a noisy scorer with a documented random mechanism. Use separate random streams for proposal generation and scorer noise so changing the scorer does not unintentionally change the candidate stream in a candidate-matched comparison.

<a id="section-44-10"></a>

## 44.10 Exercises, answers, and reading

1. With p=0.2, t=0.8, and f=0.05, calculate accepted correctness.
2. Why is a prefix's fraction of correct steps not necessarily a continuation value?
3. What evidence would indicate that search is exploiting the scorer?

<details><summary>Worked answers</summary>

1. 0.16/(0.16+0.04)=0.8, under the specified distribution and rate assumptions.
2. It measures observed local validity, not the distribution of future outcomes under a defined policy and budget.
3. Selected scores increase while independent task correctness decreases, with controlled examples showing which scorer feature the chosen failures exploit.

</details>

Read [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) and [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). Their settings motivate distinguishing process supervision, learned proxy quality, and adaptive selection.

---

[← Chapter 43](../43-sampling-and-candidate-selection/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 45 →](../45-search-over-reasoning-trajectories/README.md)
