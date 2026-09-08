# 37. Reward Models and Preference Learning

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 36](../36-reinforcement-learning-from-human-feedback/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 38 →](../38-direct-preference-optimization/README.md)

[Quick-read Topic 37](../../README.md#topic-37) · [Notation](../NOTATION.md)

- [37.1 Learn a preference score without mistaking it for truth](#section-37-1)
- [37.2 Define the pairwise model](#section-37-2)
- [37.3 Derive the loss gradient](#section-37-3)
- [37.4 Calculate confidence and score invariances](#section-37-4)
- [37.5 Design the annotation process](#section-37-5)
- [37.6 Split data at the source of dependence](#section-37-6)
- [37.7 Analyze best-of-N selection pressure](#section-37-7)
- [37.8 Evaluate under deliberate exploitation tests](#section-37-8)
- [37.9 Laboratory and process feedback](#section-37-9)
- [37.10 Problems, worked answers, and reading](#section-37-10)

---

<a id="section-37-1"></a>

## 37.1 Learn a preference score without mistaking it for truth

A reward model maps a prompt-response pair or trajectory to a score intended to predict preferences. The score is a learned measurement, shaped by data and rubric, rather than an intrinsic measure of correctness or human value.

**Prerequisites:** Chapters 5, 21, and 36; logistic likelihood. **Targets:** derive the pairwise loss, identify what scores are not identifiable, and test whether preference prediction survives optimization pressure.

The courier's satisfaction model may learn that polished delivery notes correlate with good service. If it later rewards polished notes despite failed deliveries, prediction on old data has failed as an optimization target.

<a id="section-37-2"></a>

## 37.2 Define the pairwise model

For prompt x and preferred/rejected responses y_w,y_l, a Bradley–Terry-style model uses P(y_w preferred)=sigmoid(r_phi(x,y_w)−r_phi(x,y_l)). Minimize negative log probability of the observed preference.

The model assumes a scalar score difference captures the preference probability under the specified data process. Context-dependent preferences, ties, inconsistent rankings, and multiple values can violate that simple representation. Define how ties and uncertain labels are handled.

<a id="section-37-3"></a>

## 37.3 Derive the loss gradient

Let d=r_w−r_l and loss=−log sigmoid(d). Then dloss/dd=sigmoid(d)−1. The preferred score receives this derivative; the rejected score receives its negative.

At d=0, predicted preference probability is 0.5, loss is log 2≈0.6931, and derivatives are −0.5 for r_w and +0.5 for r_l. Gradient descent increases the score gap for the labeled pair. This calculation is a useful sign check for an implementation.

<a id="section-37-4"></a>

## 37.4 Calculate confidence and score invariances

With score gap log 3≈1.0986, predicted preference probability is 0.75 and loss is −log 0.75≈0.2877. Adding the same prompt-dependent constant c(x) to both scores changes neither probability nor loss.

Therefore pairwise data do not identify an absolute reward origin for each prompt. Multiplying all scores changes preference confidence under a fixed sigmoid temperature. Reward scale and calibration matter when the score later enters a KL-regularized policy objective.

<a id="section-37-5"></a>

## 37.5 Design the annotation process

Specify the rubric, available context, candidate sampling policy, annotator qualifications relevant to the task, and how disagreement is recorded. Randomize response order to reduce presentation effects and include checks appropriate to the task.

Do not force every pair into a confident binary label when the rubric permits ties or insufficient information. For factual tasks, provide the evidence reviewers need. A preference about writing style is not automatically a verified judgment of factual correctness.

<a id="section-37-6"></a>

## 37.6 Split data at the source of dependence

Pairs sharing prompts or near-identical candidates are dependent. Split by prompt family, task template, or source document when testing generalization along those dimensions. Fit preprocessing and calibration using training/validation data only.

Report pairwise accuracy, probability calibration, and subgroup behavior. A model can rank most easy pairs correctly while failing precisely on the subtle cases a strong policy generates. Validation candidates should include the policy distribution encountered during optimization when feasible.

<a id="section-37-7"></a>

## 37.7 Analyze best-of-N selection pressure

If a scorer equals true quality plus noise, selecting the highest score among N candidates tends to select positive noise as well as quality. Increasing N can raise the selected score even when true quality improves little.

In the extreme case where all candidates have identical true quality and independent symmetric score noise, any score gain from selection is entirely evaluator error. This is the same selection mechanism introduced by maximization bias in Chapter 20, now applied to generated responses.

<a id="section-37-8"></a>

## 37.8 Evaluate under deliberate exploitation tests

Construct controlled pairs that vary one suspected shortcut: length, formatting, confident language, answer correctness, or copied rubric phrases. Hold other properties as constant as possible and inspect score changes.

Then evaluate candidates selected by the model, not only randomly sampled candidates. Use an independent checker or blinded human review under a defined rubric. A second learned judge can share biases, so independence should be argued from data, design, and task checks rather than model naming alone.

<a id="section-37-9"></a>

## 37.9 Laboratory and process feedback

[Notebook 07](../../notebooks/07_preference_and_grpo.ipynb) provides a small preference-learning context. [Notebook 09](../../notebooks/09_small_model_sft_dpo.ipynb) uses pairwise optimization without training a standalone reward model. A full reward-model training pipeline is a proposed extension.

For reasoning, outcome rewards score final answers while process rewards judge intermediate steps. Process labels can improve credit assignment but add annotation and verifier assumptions. A correct final number does not prove every intermediate step was valid, and a stylistically plausible step is not necessarily logically sound.

<a id="section-37-10"></a>

## 37.10 Problems, worked answers, and reading

1. What score gap corresponds to predicted preference probability 0.9?
2. If both response scores increase by 100, how does pairwise loss change?
3. Why is held-out pair accuracy insufficient to certify safe optimization against the model?

<details><summary>Worked answers</summary>

1. log(0.9/0.1)=log 9≈2.1972.
2. It does not change: the difference is identical. Absolute scores are not identified by this pairwise likelihood alone.
3. Optimization can produce a new candidate distribution and deliberately select the model's errors. Pair accuracy on old data does not bound this adaptive selection effect or establish the validity of the underlying rubric.

</details>

Read [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741), [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760), and [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).

---

[← Chapter 36](../36-reinforcement-learning-from-human-feedback/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 38 →](../38-direct-preference-optimization/README.md)
