**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 37](37-reward-models-and-preference-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 39 →](39-grpo-and-reinforcement-learning-for-reasoning.md)

**Go deeper:** [Chapter 38: Direct Preference Optimization](../chapters/38-direct-preference-optimization/README.md)

---

<a id="topic-38"></a>

# 38. Direct Preference Optimization

> **Why this algorithm exists:** A fixed preference dataset can support a direct policy loss under a KL-regularized formulation. DPO avoids a separate reward-model-and-online-RL loop in this setting; it is not a general replacement for sequential environment learning.

**[Try it in Notebook 07 → Preferences, DPO, and GRPO](../notebooks/07_preference_and_grpo.ipynb)**

Direct Preference Optimization (DPO) shows that, under a particular KL-regularized preference-learning formulation, preference optimization can be expressed directly as a classification-style objective without explicitly training a separate reward model and then running an online RL optimizer.

For preferred y<sub>w</sub> and rejected y<sub>l</sub>, DPO favors a larger log-likelihood margin between them, measured relative to a reference policy.

Conceptually:

<p align="center">
  <img src="../assets/diagrams/dpo-pipeline.svg" alt="Learn directly from preferred and rejected pairs" width="760">
</p>

DPO is best understood as **preference optimization**, not as a drop-in replacement for every RL problem. It is particularly useful when pairwise preference data are available and online environment interaction is unnecessary.


### The direct objective

Define the reference-relative preference margin

<p align="center">
  <img src="../assets/equations/equation-71.svg" alt="\Delta_\theta=\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}." width="760">
</p>

DPO minimizes −𝔼[log σ(βΔ<sub>θ</sub>)]. Response log probability is the sum of conditional token log probabilities. See the original [DPO paper](https://arxiv.org/abs/2305.18290).

**Analogy:** learn directly from an editor's paired drafts instead of first constructing a separate automated editor. The objective increases the preferred-versus-rejected margin; it does not guarantee that the preferred response's absolute probability rises on every update.

DPO's formulation is connected to KL-regularized reward optimization, but standard training uses fixed preference pairs and requires no online rollout-and-critic loop.


> **Engineering Note:** mask prompt and padding tokens, include completion termination consistently, and keep the reference fixed. Compare the preference margin with generated-answer quality; improving one need not improve the other. [Train a real model in Notebook 09](../notebooks/09_small_model_sft_dpo.ipynb).

---

**Continue in depth:** [Read Chapter 38](../chapters/38-direct-preference-optimization/README.md)

[← Topic 37](37-reward-models-and-preference-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 39 →](39-grpo-and-reinforcement-learning-for-reasoning.md)
