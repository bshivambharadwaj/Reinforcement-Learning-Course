# 43. Sampling and Candidate Selection

**Advanced · Part III** · **Created by Shivam Bharadwaj**

[← Chapter 42](../42-reasoning-as-a-sequential-decision-problem/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 44 →](../44-verifiers-process-rewards-and-value-models/README.md)

[Quick read](../../quick-read/43-sampling-and-candidate-selection.md) · [Notation](../NOTATION.md)

- [43.1 Generate candidates before deciding how to select](#section-43-1)
- [43.2 Single sampling is a real baseline](#section-43-2)
- [43.3 Best-of-N uses a declared scorer](#section-43-3)
- [43.4 Self-consistency aggregates answers](#section-43-4)
- [43.5 Calculate candidate opportunity](#section-43-5)
- [43.6 Derive a finite-sample pass-at-k estimator](#section-43-6)
- [43.7 Compare selection under an explicit cost model](#section-43-7)
- [43.8 Diagnose a wrong majority](#section-43-8)
- [43.9 Laboratory and reproducible comparison](#section-43-9)
- [43.10 Exercises, answers, and reading](#section-43-10)

---

<a id="section-43-1"></a>

## 43.1 Generate candidates before deciding how to select

Sampling creates a candidate set. Selection decides what to return. These stages use different information and should be evaluated separately.

**Prerequisites:** Chapters 5, 17, and 41–42. **Learning targets:** distinguish likelihood ranking, best-of-N, self-consistency, and oracle pass rates; analyze correlation and resource use.

Asking several couriers for routes can provide alternatives. Choosing the most common route, the route with the highest predicted score, or a route known to succeed are different selection rules.

<a id="section-43-2"></a>

## 43.2 Single sampling is a real baseline

A single sample uses one complete trajectory from the frozen policy. Record its actual cost and its outcome across many task/seed combinations. Greedy decoding is another baseline: it chooses local modes and can differ sharply from sampling.

In the simulator, the modal offset is +1, so a greedy trace is systematically wrong. The shipped comparison uses single sampling, not greedy decoding. Naming this accurately matters when comparing its result with other methods.

<a id="section-43-3"></a>

## 43.3 Best-of-N uses a declared scorer

Generate N complete candidates, score each, and choose the highest score using a fixed tie rule. This method depends on the scorer's ability to rank the sampled responses, not merely on their diversity.

The demo's exact process score is the fraction of locally correct additions. The misleading score instead rewards +1 errors. Neither calls the independent final evaluator during selection. Increasing N can help under a useful scorer and amplify mistakes under a bad one.

<a id="section-43-4"></a>

## 43.4 Self-consistency aggregates answers

Self-consistency groups sampled solutions by their final answer and chooses the most frequent answer under a specified rule. It can succeed when different valid paths converge on one answer while errors disperse across alternatives.

That behavior is not guaranteed. If the proposal distribution concentrates on the same wrong answer, more samples can strengthen the wrong majority. Tie handling and answer canonicalization matter: “2,” “2.0,” and a formatted sentence may represent the same answer in one task but require different parsing in another.

<a id="section-43-5"></a>

## 43.5 Calculate candidate opportunity

With N independent attempts and per-attempt success p, oracle success is 1−(1−p)^N. For p=0.1 and N=8, this is approximately 0.5695. It is an upper opportunity for any selector restricted to that candidate set.

If all attempts are perfectly correlated, repeated sampling supplies no additional opportunity and success remains p. Shared prompts, model biases, and near-identical decoding can create dependence. Report diversity and actual candidate-set correctness rather than relying only on an independence formula.

<a id="section-43-6"></a>

## 43.6 Derive a finite-sample pass-at-k estimator

If n generated candidates contain c correct ones, uniformly drawing k distinct candidates without replacement misses all successes with probability choose(n−c,k)/choose(n,k). Thus 1−that ratio estimates the within-set chance of including at least one correct candidate, for k≤n.

This combinatorial expression is about candidate inclusion, not successful verifier selection. If k exceeds the number of incorrect candidates, the miss probability is zero. Different candidate-generation dependence affects how broadly this finite-set statistic can be interpreted.

<a id="section-43-7"></a>

## 43.7 Compare selection under an explicit cost model

At depth d and final-score cost v, budget B fits floor(B/(d+v)) scored candidates. A voting strategy with no scoring calls fits floor(B/d) candidates under this simplified model. Actual model inference can have variable lengths, batching, caching, and evaluator costs.

Distinguish a candidate-matched experiment, where methods share the same candidates but spend different scoring compute, from a budget-matched experiment, where the number of candidates can differ. Both are useful when labeled; neither automatically answers the other's question.

<a id="section-43-8"></a>

## 43.8 Diagnose a wrong majority

The simulator's three-step correct-answer probability is 0.4³+6×0.1×0.4×0.5=0.184. The first term represents all-correct steps; the second includes compensating −1 and +1 errors.

A net +1 error is more likely: three placements of one +1 and two zeros contribute 0.24, while two +1 and one −1 contribute 0.075, totaling 0.315. A large vote can therefore converge toward a wrong answer even though correct candidates are present. This is a designed counterexample to universal voting improvement.

<a id="section-43-9"></a>

## 43.9 Laboratory and reproducible comparison

[Notebook 12](../../notebooks/12_sampling_and_selection.ipynb) compares single sampling, self-consistency, and best-of-N over several budgets and seeds. It includes a candidate-matched check confirming that scoring changes selection rather than candidate generation in that comparison.

Record selected success, candidate oracle success, process validity, and actual cost. If oracle success rises but selected success falls, examine the selector before changing the proposal policy. The [demo](../../demos/inference-time/README.md) exposes the same metrics interactively.

<a id="section-43-10"></a>

## 43.10 Exercises, answers, and reading

1. With n=5, c=2, and k=2, calculate the finite-set oracle inclusion probability.
2. At B=30, d=4, and v=2, compare scored-candidate and unscored-candidate counts.
3. Why can self-consistency fail as N grows in the simulator?

<details><summary>Worked answers</summary>

1. 1−choose(3,2)/choose(5,2)=1−3/10=0.7.
2. Five scored candidates versus seven unscored candidates, leaving two units unused in the latter case.
3. The most probable aggregate answer is wrong. Voting estimates answer frequency; it does not independently verify correctness.

</details>

Read [Self-Consistency](https://arxiv.org/abs/2203.11171) for the original method and its experimental setting. The course's exact arithmetic probability calculation explains a limitation, not a reproduction of that paper's results.

The finite-set pass@k estimator is described in [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374). It estimates candidate-set success, not the accuracy of a selector that must return one answer without access to the evaluator.

---

[← Chapter 42](../42-reasoning-as-a-sequential-decision-problem/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 44 →](../44-verifiers-process-rewards-and-value-models/README.md)
