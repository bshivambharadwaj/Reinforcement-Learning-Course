# 39. GRPO and Reinforcement Learning for Reasoning

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 38](../38-direct-preference-optimization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 40 →](../40-multimodal-rl-and-rl-for-ai-agents/README.md)

[Quick-read Topic 39](../../quick-read/39-grpo-and-reinforcement-learning-for-reasoning.md) · [Notation](../NOTATION.md)

- [39.1 Learn from groups of verifiable attempts](#section-39-1)
- [39.2 Define a common group-relative construction](#section-39-2)
- [39.3 Calculate a mixed-success group](#section-39-3)
- [39.4 Examine finite-group baseline dependence](#section-39-4)
- [39.5 Connect the group signal to policy optimization](#section-39-5)
- [39.6 Distinguish outcome and process verification](#section-39-6)
- [39.7 Analyze sparse rewards and group size](#section-39-7)
- [39.8 Separate training gains from search gains](#section-39-8)
- [39.9 Laboratory and implementation scope](#section-39-9)
- [39.10 Problems, worked answers, and reading](#section-39-10)

---

<p align="center">
  <img src="../../assets/diagrams/grpo-training.svg" width="760" alt="Group-relative learning compares multiple attempts at the same task." />
</p>

*Group-relative learning compares multiple attempts at the same task.*

<a id="section-39-1"></a>

## 39.1 Learn from groups of verifiable attempts

Group Relative Policy Optimization (GRPO) uses multiple sampled responses to the same prompt to construct relative reward signals, often avoiding a separately learned value critic. It is associated with reasoning-oriented post-training, but the quality of the verifier and sampled data remains decisive.

**Prerequisites:** Chapters 25, 29, 31, and 36–38. **Targets:** calculate group advantages, distinguish outcome from process feedback, and design a reasoning evaluation that controls sampling budget.

The courier compares several attempted plans for the same delivery. This removes some task-difficulty variation, but a group of equally unsuccessful plans may offer no ranking signal at all.

<a id="section-39-2"></a>

## 39.2 Define a common group-relative construction

For a prompt x, sample G responses under an old policy and obtain rewards r_1,…,r_G. A common advantage is A_i=(r_i−mean(r))/(std(r)+epsilon). The same response-level advantage may weight its generated token terms.

Precise implementations differ in standard-deviation convention, epsilon placement, token/sequence reduction, KL term, and reward design. Specify the actual formula rather than assuming every algorithm called GRPO is identical.

<a id="section-39-3"></a>

## 39.3 Calculate a mixed-success group

For rewards [0,0,1,1], the mean is 0.5 and population standard deviation is 0.5. Ignoring a tiny numerical epsilon, advantages are [−1,−1,1,1]. The two successful samples are reinforced relative to the unsuccessful ones.

For [1,1,1,1] or [0,0,0,0], centered rewards are all zero. With a stabilizing denominator, advantages remain zero. A zero-variance group does not tell the relative estimator whether every response was excellent or every response failed; its absolute reward metric still does.

<a id="section-39-4"></a>

## 39.4 Examine finite-group baseline dependence

For independent samples and unnormalized centered rewards, subtracting the group mean includes each sample's own reward. In a one-step score estimator, E[(r_i−mean(r)) score_i]=(1−1/G)E[r_i score_i], using zero expected scores for cross-sample terms.

A leave-one-out mean removes this particular self-inclusion factor under the same assumptions. Dividing by a sample standard deviation adds further dependence, so the simple factor no longer describes the full normalized estimator. Group normalization is useful, but should not be labeled automatically identical to an unbiased value-baseline estimator.

### Derive the self-inclusion factor explicitly

Let z_i be sample i's score gradient and let mu_G=(r_i+sum_(j≠i)r_j)/G. For independent samples at a fixed prompt, E[r_j z_i]=E[r_j]E[z_i]=0 when j≠i. Therefore E[(r_i−mu_G)z_i]=(1−1/G)E[r_i z_i].

For G=4, this is a factor 0.75. A leave-one-out mean uses only the other G−1 rewards and avoids that particular shrinkage under the same independence assumptions. Standard-deviation normalization adds a random shared denominator, so multiplying by G/(G−1) does not generally make the fully normalized estimator unbiased.

<a id="section-39-5"></a>

## 39.5 Connect the group signal to policy optimization

A common GRPO-style objective applies PPO-like clipped old/new action ratios to group advantages and adds a reference-KL penalty. The old policy generated the group; the reference defines an anchor. Their roles remain separate.

Token-averaged versus response-averaged reductions change length weighting. A response-level reward broadcast to every token provides coarse credit assignment, not a proof that each token caused the outcome. Process rewards or finer estimators alter that structure.

<a id="section-39-6"></a>

## 39.6 Distinguish outcome and process verification

An outcome verifier checks a final answer or completed task. A process verifier evaluates intermediate steps. A trace can reach the right answer through an invalid intermediate claim, accidental cancellation, or a shortcut outside the intended task rules.

For the course's arithmetic task, verify both the intermediate sum and final result. This makes the difference observable. In open-ended reasoning, process verification can itself be difficult and fallible; replacing human labels with a learned judge does not make correctness automatic.

<a id="section-39-7"></a>

## 39.7 Analyze sparse rewards and group size

If each independent response succeeds with probability p, the probability that a group contains at least one success is 1−(1−p)^G. A group containing both success and failure has probability 1−p^G−(1−p)^G for binary rewards.

At p=0.1 and G=8, at-least-one success is about 0.5695. Larger groups improve the chance of observing contrast but consume more samples. If p is essentially zero, an affordable group may still provide little useful signal; warm starts or a curriculum can change that regime.

### Choose group size from the probability of contrast

For binary reward, the probability of a useful mixed group is 1−p^G−(1−p)^G. At p=0.01 and G=8 it is approximately 0.07726; most groups are all failures. At p=0.5 the same group size gives 0.99219.

Group-based learning is therefore sensitive to the policy's starting competence and task difficulty. Increasing G costs more generations per prompt. Compare total sampled responses, not merely the number of prompt groups, when evaluating a warm start, curriculum, or larger group size.

<a id="section-39-8"></a>

## 39.8 Separate training gains from search gains

Report single-attempt accuracy, intermediate-step validity, and success under a fixed multi-sample budget. Best-of-k or verifier-selected performance uses extra inference compute and should be compared with baselines using the same allowance.

Split by meaningful problem structure and inspect duplicates. Holding out arithmetic tuples tests one kind of generalization; larger operands, longer chains, or new operators test others. A small task's success does not establish open-ended reasoning ability or faithful explanations.

### Distinguish verifier-selected success from oracle pass rate

The probability that at least one candidate is correct differs from the probability that a fallible verifier selects a correct candidate. Best-of-k with an oracle measures an upper opportunity under that sample set; a deployed selector must identify the successful candidate from available evidence.

Report candidate correctness, selected-answer correctness, and verifier false positives separately. A verifier that rewards an answer-format exploit can make both training score and selected score rise while actual success falls. For process evaluation, also record whether intermediate steps are valid rather than inferring them from the final answer alone.

<a id="section-39-9"></a>

## 39.9 Laboratory and implementation scope

[Notebook 10](../../notebooks/10_verifiable_reasoning_grpo.ipynb) trains a small structured two-step policy with verifiable arithmetic, a warm start, group-relative updates, and outcome/process comparisons. It is **not** full language-model GRPO training. The restricted action space makes credit assignment and verifier behavior inspectable.

[Notebook 07](../../notebooks/07_preference_and_grpo.ipynb) offers a smaller conceptual entry point. For a language-model extension, additionally specify token masks, EOS handling, old/reference snapshots, generation temperature, group size, and a compute-matched held-out evaluation.

<a id="section-39-10"></a>

## 39.10 Problems, worked answers, and reading

1. For rewards [0,1,1], calculate population-std-normalized advantages, ignoring epsilon.
2. With p=0.5 and G=4, what is the probability of a mixed binary-reward group?
3. Why can higher best-of-eight accuracy occur without any policy improvement?

<details><summary>Worked answers</summary>

1. Mean 2/3, variance 2/9, standard deviation sqrt(2)/3. Advantages are [−sqrt(2),1/sqrt(2),1/sqrt(2)], approximately [−1.4142,0.7071,0.7071].
2. 1−(0.5)^4−(0.5)^4=0.875.
3. Sampling more candidates and selecting a success can improve task completion for an unchanged policy. Training and inference-budget effects need separate comparisons.

</details>

Read [DeepSeekMath](https://arxiv.org/abs/2402.03300) for the original GRPO formulation and [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) for process-supervision context. Keep paper-specific implementations distinct from the course's controlled lab.

---

[← Chapter 38](../38-direct-preference-optimization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 40 →](../40-multimodal-rl-and-rl-for-ai-agents/README.md)
