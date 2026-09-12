**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 35](35-multi-agent-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 37 →](37-reward-models-and-preference-learning.md)

**Go deeper:** [Chapter 36: Reinforcement Learning from Human Feedback](../chapters/36-reinforcement-learning-from-human-feedback/README.md)

---

<a id="topic-36"></a>

# 36. Reinforcement Learning from Human Feedback

> **Why this method exists:** Next-token imitation does not directly optimize which responses people prefer. RLHF supplies preference-derived feedback; PPO is one possible optimizer within that feedback pipeline.

**[Try it in Notebook 07 → Preferences, DPO, and GRPO](../notebooks/07_preference_and_grpo.ipynb)**

RLHF aligns a model's behavior with human preferences rather than relying only on next-token prediction.

A classic pipeline is:

<p align="center">
  <img src="../assets/diagrams/rlhf-pipeline.svg" alt="From demonstrations to preference feedback" width="760">
</p>

In one common formulation, a reward model learns from comparisons between responses, and the language-model policy is optimized against that learned reward while being regularized against excessive drift from a reference model.

RLHF connects language-model post-training to classical ideas such as policies, rewards, advantages, value estimation, and constrained updates.


### Map language-model training onto RL

| Classical RL object | Language-model example |
|---|---|
| State or context | Prompt plus tokens generated so far |
| Action | Next token; sometimes a complete response in a simplified formulation |
| Policy | Conditional language-model distribution |
| Episode | One response, or a longer tool interaction |
| Reward | Preference-model score, possibly combined with task feedback |

A common response-level objective is

<p align="center">
  <img src="../assets/equations/equation-67.svg" alt="\max_\theta\;\mathbb{E}_{x,y\sim\pi_\theta}[r_\phi(x,y)]
-\beta\mathbb{E}_x[D_{KL}(\pi_\theta(\cdot|x)\Vert\pi_{ref}(\cdot|x))]." width="760">
</p>

The reference is typically a frozen starting policy for the RL stage. It differs from PPO's old policy, which is refreshed as rollout collection proceeds.

**Analogy:** a writer learns from an editor's preferences while retaining useful behavior learned earlier. The result is optimized for measured preferences; a high reward-model score does not establish universal correctness or alignment.


### From tokens to trajectories: what changes in LM PPO?

Autoregressive generation chooses one token conditioned on the prompt and preceding tokens.
The probability of a completion is a product of these conditional decisions; its log probability
is their sum. EOS ends a response. In a tool agent, a tool result changes the subsequent context,
so a response may be one segment of a longer episode.

| Classical term | Post-training counterpart | Implementation consequence |
|---|---|---|
| State | Prompt, generated prefix, and available tool observations | Mask information that would leak future outcomes |
| Action | Next token or a structured tool decision | Store the log probability of the sampled action |
| Trajectory | Completion tokens, or multiple tool turns | Preserve boundaries and valid-token masks |
| Reward | Learned preference score or a verifier result | Separate the task signal from regularization |
| Advantage | Return minus a value baseline, or a group-relative estimate | Detach targets before optimizing the policy |
| Reference policy | Usually the frozen SFT starting policy | Anchors behavior through a reference penalty |
| Old policy | Policy that generated the current rollout | Supplies PPO's denominator; refreshes for new rollouts |
| KL penalty | Cost of drifting from the reference distribution | Its target and coefficient are distinct from PPO clipping |

**A concrete token trace:** a prompt asking for a sum is followed by the actions `The`, ` answer`,
` is`, ` 6`, and EOS (illustrative tokens; tokenization depends on the model). A final verifier
score arrives after the answer. A critic and GAE can carry that signal backward through valid
token positions. A token-level sampled log-ratio penalty can provide additional per-step costs.
Do not compute loss on prompt tokens or padding as if they were sampled completion actions.

An outcome score may arrive only at the end. A process score rates intermediate steps. Assigning
one final score to many tokens does not prove which token caused success. For a continuing agent,
the end of a text segment is not necessarily the end of the whole task.

> **Common Misconception:** PPO's old policy and the SFT reference are interchangeable. They have
different jobs. Updating the reference every minibatch silently changes the regularized objective.

**Knowledge check:** can the reference stay frozen while the old policy changes every rollout?
Yes—that is precisely why the two must be named and stored separately.

Read [InstructGPT](https://arxiv.org/abs/2203.02155) for the SFT, preference-model, and PPO pipeline.
Run [Notebook 09](../notebooks/09_small_model_sft_dpo.ipynb) for an affordable SFT→DPO route that
uses a different optimizer after SFT; it does not implement LM PPO.

---

**Continue in depth:** [Read Chapter 36](../chapters/36-reinforcement-learning-from-human-feedback/README.md)

[← Topic 35](35-multi-agent-reinforcement-learning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 37 →](37-reward-models-and-preference-learning.md)
