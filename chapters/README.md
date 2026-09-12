# Chapters: From Bellman Equations to Reasoning Agents

**Created by Shivam Bharadwaj** · [Course home](../README.md) · [Notebooks](../README.md#practical-track)

The main README is the course home and navigation page. The [quick reads](../quick-read/README.md)
preserve the concise topic lessons. These chapters provide the deeper path:
precise definitions, derivations, fully specified examples, failure analysis, engineering
decisions, and exercises with worked answers. Choose either reading depth and move between
them through the links at the top and bottom of every chapter.

The thread is **classical decisions → value estimation → policy optimization → preference
learning → reasoning and agents**. We reuse mathematical objects across these settings while
making differences in data, feedback, and evaluation explicit. Modern terminology does not
replace the need to understand an expectation, a bootstrap target, or an objective.

## Choose a path

| Path | Chapters | What to produce |
|---|---|---|
| Build the foundations | 1–20 | A task model, exact values, and a sampled-control experiment |
| Understand deep RL | 21–32 | A traced loss, gradient/mask checks, and a multi-seed evaluation |
| Understand modern post-training | 25–32, then 36–39 | A clear account of old/reference policies, preference objectives, and verifier limitations |
| Design an agent system | 33–40, after the foundations | A task contract, independent evaluator, constraints, and a capstone report |

Every chapter has ten numbered sections, such as 3.1–3.10. Chapter prerequisites are stated
in the text. Equations use readable plain notation and the course's existing SVG assets;
raw renderer-dependent LaTeX is avoided. Chapters include small proofs where assumptions
support them, and explicitly distinguish empirical observations from guarantees. Extended
investigations develop sensitivity bounds, estimator variance, policy-update mechanics,
data coverage, and evaluator design within the existing numbered sections.

## All chapters

| Number | Chapter | Level |
|---|---|---|
| 01 | [What Is Reinforcement Learning?](01-what-is-reinforcement-learning/README.md) | Foundations |
| 02 | [The Agent–Environment Interaction](02-the-agent-environment-interaction/README.md) | Foundations |
| 03 | [Markov Decision Processes](03-markov-decision-processes/README.md) | Foundations |
| 04 | [States, Actions, Rewards and Transitions](04-states-actions-rewards-and-transitions/README.md) | Foundations |
| 05 | [Policies](05-policies/README.md) | Foundations |
| 06 | [Returns and Discounting](06-returns-and-discounting/README.md) | Foundations |
| 07 | [State-Value Functions](07-state-value-functions/README.md) | Foundations |
| 08 | [Action-Value Functions](08-action-value-functions/README.md) | Foundations |
| 09 | [Advantage Functions](09-advantage-functions/README.md) | Foundations |
| 10 | [Bellman Expectation Equations](10-bellman-expectation-equations/README.md) | Foundations |
| 11 | [Bellman Optimality Equations](11-bellman-optimality-equations/README.md) | Foundations |
| 12 | [Dynamic Programming](12-dynamic-programming/README.md) | Foundations |
| 13 | [Policy Evaluation](13-policy-evaluation/README.md) | Foundations |
| 14 | [Policy Improvement](14-policy-improvement/README.md) | Foundations |
| 15 | [Policy Iteration](15-policy-iteration/README.md) | Foundations |
| 16 | [Value Iteration](16-value-iteration/README.md) | Foundations |
| 17 | [Monte Carlo Methods](17-monte-carlo-methods/README.md) | Intermediate |
| 18 | [Temporal-Difference Learning](18-temporal-difference-learning/README.md) | Intermediate |
| 19 | [SARSA](19-sarsa/README.md) | Intermediate |
| 20 | [Q-Learning and Exploration](20-q-learning-and-exploration/README.md) | Intermediate |
| 21 | [Function Approximation](21-function-approximation/README.md) | Intermediate |
| 22 | [Deep Q-Networks](22-deep-q-networks/README.md) | Intermediate |
| 23 | [Experience Replay](23-experience-replay/README.md) | Intermediate |
| 24 | [Target Networks](24-target-networks/README.md) | Intermediate |
| 25 | [Policy Gradient Methods](25-policy-gradient-methods/README.md) | Intermediate |
| 26 | [REINFORCE](26-reinforce/README.md) | Intermediate |
| 27 | [Actor–Critic Methods](27-actor-critic-methods/README.md) | Intermediate |
| 28 | [Advantage Actor–Critic](28-advantage-actor-critic/README.md) | Intermediate |
| 29 | [Generalized Advantage Estimation](29-generalized-advantage-estimation/README.md) | Intermediate |
| 30 | [Trust Region Policy Optimization](30-trust-region-policy-optimization/README.md) | Advanced |
| 31 | [Proximal Policy Optimization](31-proximal-policy-optimization/README.md) | Advanced |
| 32 | [Entropy Regularization](32-entropy-regularization/README.md) | Advanced |
| 33 | [Offline Reinforcement Learning](33-offline-reinforcement-learning/README.md) | Advanced |
| 34 | [Model-Based Reinforcement Learning](34-model-based-reinforcement-learning/README.md) | Advanced |
| 35 | [Multi-Agent Reinforcement Learning](35-multi-agent-reinforcement-learning/README.md) | Advanced |
| 36 | [Reinforcement Learning from Human Feedback](36-reinforcement-learning-from-human-feedback/README.md) | Advanced |
| 37 | [Reward Models and Preference Learning](37-reward-models-and-preference-learning/README.md) | Advanced |
| 38 | [Direct Preference Optimization](38-direct-preference-optimization/README.md) | Advanced |
| 39 | [GRPO and Reinforcement Learning for Reasoning](39-grpo-and-reinforcement-learning-for-reasoning/README.md) | Advanced |
| 40 | [Multimodal RL and RL for AI Agents](40-multimodal-rl-and-rl-for-ai-agents/README.md) | Advanced |

## Study and assessment

Do each calculation before opening its worked answer. For proofs, write down the assumptions
before the algebra. For experiments, state the hypothesis, change one factor, fix evaluation
budgets, report all seeds, and explain a failed run. Reproducing a learning curve is only the
start: identify what evidence would contradict your explanation.

The lab links distinguish **implemented exercises** from **proposed extensions**. In
particular, the course does not ship full TRPO, offline RL, model-based RL, multi-agent RL,
or VLM training implementations. The corresponding chapters develop those ideas and explain
how to design the next experiment; a link to a related notebook does not claim otherwise.

Use the [assessment guide](ASSESSMENT.md) to combine chapters into written problem sets and
an original experimental report. The [reproducibility guide](../REPRODUCIBILITY.md) records
the executed notebook environment and its limits.

The [notation and reading guide](NOTATION.md) defines symbols, indexing, and proof
conventions. Read a chapter once for intuition, then work its example without looking,
and finally test the stated claim in the linked lab or proposed extension. Reading time
is only a small part of the workload: the assessment guide includes multi-session projects.

## Academic reference points and distinct scope

[Stanford CS234](https://web.stanford.edu/class/cs234/) combines formal problem definition,
algorithm implementation, and evaluation criteria. [Berkeley CS185/285](https://rail.eecs.berkeley.edu/deeprlcourse/index.html)
provides a reference for connecting deep-RL concepts with substantial experiments. These
are benchmarks for instructional rigor, not affiliations or a claim of equivalent accreditation,
faculty review, or learning outcomes. The problems and explanations here are original course
material, not reproductions of those courses' assignments or solutions.

Our emphasis is the conceptual continuity into small-model post-training, verifiable reasoning,
and tool-agent evaluation, with affordable experiments and explicit failure cases. The
classical chapters support that journey rather than serving as disconnected prerequisites.

Educational text and diagrams use [CC BY 4.0](../LICENSE-CC-BY-4.0); code uses [MIT](../LICENSE-MIT).
