**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 36](36-reinforcement-learning-from-human-feedback.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 38 →](38-direct-preference-optimization.md)

**Go deeper:** [Chapter 37: Reward Models and Preference Learning](../chapters/37-reward-models-and-preference-learning/README.md)

---

<a id="topic-37"></a>

# 37. Reward Models and Preference Learning

> **Why this method exists:** Comparisons are often easier to obtain than a hand-written reward for every response. A reward model generalizes those comparisons into scores, which remain fallible proxies rather than truth labels.

**[Try it in Notebook 07 → Preferences, DPO, and GRPO](../notebooks/07_preference_and_grpo.ipynb)**

Suppose humans prefer response y<sub>w</sub> over y<sub>l</sub> for prompt x.

A reward model can assign scores

<p align="center">
  <img src="../assets/equations/equation-68.svg" alt="r_\phi(x,y)." width="760">
</p>

A common pairwise preference objective models

<p align="center">
  <img src="../assets/equations/equation-69.svg" alt="P(y_w\succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))." width="760">
</p>

The reward model converts qualitative preference comparisons into a scalar signal that can guide optimization.

But reward models are imperfect proxies. Optimizing them too aggressively can expose misspecification, distribution shift, or reward hacking. Evaluation therefore remains a separate and critical part of alignment.


### Train on comparisons, not absolute truth

The sigmoid is σ(z)=1/(1+e<sup>−z</sup>). A pairwise reward-model loss is

<p align="center">
  <img src="../assets/equations/equation-70.svg" alt="\mathcal L_{RM}(\phi)=-\mathbb E_{(x,y_w,y_l)}
[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]." width="760">
</p>

Minimizing it encourages larger score differences in favor of preferred responses. Adding the same prompt-dependent constant to both scores leaves their preference probability unchanged, so these scores are not uniquely calibrated measures of truth.

**Analogy:** asking an editor “which draft is better?” is often easier than asking for a perfectly calibrated numerical quality score. Different editors may disagree, and models can learn superficial cues such as verbosity. Hold out preference examples and separately check task performance to detect failures beyond the training comparisons.


> **Failure Mode — reward-model overoptimization:** the policy may learn features that the reward model likes more than evaluators do. Keep independent task checks, held-out preferences, and examples from later policies. A rising proxy score with flat or falling true performance is a failure signal. [Reward-model overoptimization study](https://arxiv.org/abs/2210.10760).

**Knowledge check:** does a reward model trained on pairwise choices provide an absolute truth scale? No; pairwise score differences do not identify a unique prompt-dependent offset.

---

**Continue in depth:** [Read Chapter 37](../chapters/37-reward-models-and-preference-learning/README.md)

[← Topic 36](36-reinforcement-learning-from-human-feedback.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 38 →](38-direct-preference-optimization.md)
