# 36. Reinforcement Learning from Human Feedback

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 35](../35-multi-agent-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 37 →](../37-reward-models-and-preference-learning/README.md)

[Quick-read Topic 36](../../quick-read/36-reinforcement-learning-from-human-feedback.md) · [Notation](../NOTATION.md)

- [36.1 Turn human judgments into a training signal](#section-36-1)
- [36.2 Specify the sequential decision problem](#section-36-2)
- [36.3 Separate the pipeline stages](#section-36-3)
- [36.4 Write a regularized response objective](#section-36-4)
- [36.5 Distinguish the four models people confuse](#section-36-5)
- [36.6 Calculate reward–KL tradeoffs](#section-36-6)
- [36.7 Analyze feedback quality and coverage](#section-36-7)
- [36.8 Anticipate reward overoptimization](#section-36-8)
- [36.9 Laboratory and the scope of the evidence](#section-36-9)
- [36.10 Problems, worked answers, and reading](#section-36-10)

---

<p align="center">
  <img src="../../assets/diagrams/rlhf-pipeline.svg" width="760" alt="A common RLHF pipeline separates demonstrations, preferences, and policy optimization." />
</p>

*A common RLHF pipeline separates demonstrations, preferences, and policy optimization.*

<a id="section-36-1"></a>

## 36.1 Turn human judgments into a training signal

Reinforcement Learning from Human Feedback (RLHF) uses human judgments to help define an objective for policy improvement. For language models, a common pipeline combines supervised fine-tuning, preference-based reward modeling, and regularized policy optimization.

**Prerequisites:** Chapters 25–32 and the distinction between rewards and values. **Targets:** trace the full data pipeline, identify the roles of each model, and design an evaluation that does not merely repeat the training reward.

The courier receives qualitative reviews such as “this delivery was more useful and considerate.” Turning those reviews into a numerical objective requires choices about what the reviewers saw, whom they represent, and which tradeoffs the labels express.

<a id="section-36-2"></a>

## 36.2 Specify the sequential decision problem

For a language policy, a state can be a prompt plus generated prefix, an action a token, and termination an end-of-response condition or a task-defined limit. A terminal reward can score the completed response; intermediate penalties or tool outcomes can add other feedback.

The environment includes prompt sampling and, for agents, tool responses and external state. A response-only contextual-bandit abstraction is sometimes useful, but token-level optimization and multi-turn tool use expose sequential structure. State the abstraction rather than switching between them implicitly.

<a id="section-36-3"></a>

## 36.3 Separate the pipeline stages

| Stage | Data | Learned object | Main question |
|---|---|---|---|
| Supervised fine-tuning | Demonstration responses | Policy | Can it produce the desired format and basic behavior? |
| Preference learning | Ranked or paired responses | Reward/scoring model | Which outcomes do annotators prefer under the rubric? |
| Policy optimization | Generated responses and feedback | Updated policy, sometimes a critic | Can the policy improve the regularized objective? |
| Independent evaluation | Held-out tasks and judgments | Evidence, not a training target | Did relevant behavior improve? |

This is a common pipeline, not the only way to use human feedback. DPO changes the optimization route and is developed in Chapter 38.

<a id="section-36-4"></a>

## 36.4 Write a regularized response objective

An illustrative objective is E_(x,y~pi)[r_phi(x,y)]−beta E_x[D_KL(pi(.|x)||pi_ref(.|x))]. The reference anchors the updated response distribution, and beta controls the tradeoff in the reward's units.

For an autoregressive response, log pi(y|x) is a sum over generated tokens. A sampled log-ratio sum can estimate the sequence KL under the policy distribution with matching support and boundaries. A single sampled log ratio may be negative even though the expected KL is nonnegative.

### Derive sequence KL as expected prefix-level KL

For autoregressive policies with compatible support and termination conventions, log[pi(y|x)/pi_ref(y|x)] is the sum of token log ratios along the generated response. Taking expectation under pi and conditioning on each visited prefix gives a sum of expected conditional token KLs.

The prefixes themselves are distributed under the updated policy, not the reference. A calculation on fixed reference-generated prefixes estimates a different weighting unless corrected. EOS and maximum-length handling determine which terms exist; prompt tokens supplied externally are not generated actions in this response policy.

This chain-rule interpretation connects a sequence-level anchor with per-token costs while making the required sampling distribution explicit.

<a id="section-36-5"></a>

## 36.5 Distinguish the four models people confuse

The **policy** generates responses. The **reward model** predicts a judgment about a response or trajectory. The **critic** estimates expected future return at a prefix under the current policy. The **reference policy** defines an anchor in the optimization objective.

A PPO **old policy** is an additional rollout snapshot used in importance ratios. It changes each collection cycle, while a reference can remain fixed for an entire stage. Copying the reference each PPO minibatch would change the regularization problem rather than simply refreshing data.

<a id="section-36-6"></a>

## 36.6 Calculate reward–KL tradeoffs

Suppose policy A has expected reward 2.0 and expected KL 0.5, while B has reward 1.8 and KL 0.1. At beta=0.5, their objectives are 1.75 and 1.75. At beta=1, B is preferred: 1.7 versus 1.5.

Reward scale changes this comparison. Multiplying reward-model outputs by ten while holding beta fixed weakens the relative anchor. Numerical reward scores have no universal natural scale, so coefficients must be interpreted with the score calibration and normalization used.

<a id="section-36-7"></a>

## 36.7 Analyze feedback quality and coverage

Preference labels reflect a rubric, prompt distribution, annotator population, and presentation. Annotators may disagree because of ambiguity or legitimate tradeoffs, not only mistakes. A single scalar model compresses that structure.

Split data by prompts or underlying task families before forming train/test response pairs when possible. Near-duplicate prompts, shared generated candidates, and templated responses can leak information across random pair splits. Report subgroup performance and disagreement rather than hiding them in a single accuracy.

### Define what an annotation disagreement means

Suppose two equally represented reviewer groups disagree systematically about whether concise or elaborate answers are preferable. A scalar reward model can learn the majority or average labeling pattern, but that does not remove the underlying tradeoff.

Record rubric dimensions separately when they matter, and evaluate relevant groups rather than presenting a single preference accuracy as universal agreement. If correctness is a requirement, verify it independently where possible; a preferred style can correlate with incorrect content. Dataset documentation should identify who labeled which prompts and what evidence they could inspect.

<a id="section-36-8"></a>

## 36.8 Anticipate reward overoptimization

The policy searches for responses that receive high predicted reward. As optimization becomes stronger, it can exploit features the reward model mistakenly favors, such as verbosity, formatting, or unsupported confidence.

Track reward-model score alongside independent correctness, human judgments, and task-specific outcomes. If score rises while held-out quality stalls or falls, more optimization is not necessarily progress. KL anchoring can limit movement, but it does not make a flawed evaluator correct.

### Separate optimization failure from objective mismatch

If both training reward and independently judged quality decrease, optimization or implementation failure is plausible. If reward increases while independent quality decreases, the policy may be successfully optimizing a flawed proxy or making an unintended tradeoff. If both improve on familiar prompts but fail on a new domain, distribution shift is a distinct explanation.

Build a result table with these measurements at every selected checkpoint under a fixed evaluation protocol. It should let a reader identify which explanation is consistent with the evidence instead of treating every disappointing result as the same kind of RLHF failure.

<a id="section-36-9"></a>

## 36.9 Laboratory and the scope of the evidence

[Notebook 09](../../notebooks/09_small_model_sft_dpo.ipynb) performs small-model SFT followed by DPO on a controlled synthetic task. It does **not** implement a full human-annotated reward-model-plus-PPO RLHF pipeline. [Notebook 07](../../notebooks/07_preference_and_grpo.ipynb) develops preference and group-relative ideas in a small setting.

Use these labs to trace objective terms and evaluate generated outputs. Their narrow task and small test set cannot establish broad helpfulness, safety, or human alignment. A proposed full-pipeline project must specify annotation protocol, reward-model validation, policy optimization budget, and independent evaluation before collecting results.

<a id="section-36-10"></a>

## 36.10 Problems, worked answers, and reading

1. In the numerical example, at what beta do A and B tie?
2. Why can a sampled sequence log ratio be negative without contradicting KL nonnegativity?
3. A policy's reward-model score improves, but independently verified accuracy falls. What conclusions are justified?

<details><summary>Worked answers</summary>

1. Solve 2−0.5beta=1.8−0.1beta, giving beta=0.5.
2. KL is an expectation over the policy distribution. Individual responses can be more likely under the reference, yielding a negative log ratio; the full supported expectation remains nonnegative.
3. The policy improved the measured proxy but not the chosen accuracy metric. Investigate reward exploitation, objective tradeoffs, and distribution shift. Neither broad improvement nor a universal failure of RLHF follows from this observation alone.

</details>

Read [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) and [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). Compare their task definitions, feedback pipelines, and evaluation protocols.

---

[← Chapter 35](../35-multi-agent-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 37 →](../37-reward-models-and-preference-learning/README.md)
