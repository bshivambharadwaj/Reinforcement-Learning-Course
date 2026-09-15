# GoalLab: fourteen stages, one evidence task

[Project home](README.md) · [Course home](../../README.md)

Each notebook in this project track executes a part of GoalLab. The separate course track preserves focused algorithm lessons and their original environments, diagrams, and comparisons. The common task is to inspect generated sources, choose the authoritative measurement, and deliver a briefing with the correct value, period, and citation. Each stage provides a worked implementation and a constraint-changing exercise with a runnable solution.

The stages build capabilities, not a chain of hidden kernel variables. Each notebook reconstructs its prerequisites from the shared modules, so it can run independently or be studied in order. Different algorithms are alternative implementations of a component; the final system does not run every learning algorithm on each request.

## Stage map

| Stage | Problem solved | Contribution to the project | Change-and-explain exercise |
|---|---|---|---|
| [01: MDP and DP](notebooks/01_mdp_dynamic_programming.ipynb) | What constitutes a valid evidence investigation? | Workspace contract, finite decision model, exact policy, actual briefing submission. | Raise tool cost and explain why a fixed horizon changes return but not ordering. |
| [02: MC versus TD](notebooks/02_mc_vs_td.ipynb) | How valuable is an evidence state when transition probabilities are not used? | Sampled value estimators for the same fixed policy. | Reduce episodes and compare estimation error. |
| [03: SARSA and Q-learning](notebooks/03_q_learning.ipynb) | Which source should the controller read and cite? | Learned tabular routing policies deployable through the gateway. | Reward a completion claim and measure actual briefing failure. |
| [04: DQN](notebooks/04_dqn.ipynb) | Can features replace an evidence Q-table? | Neural action-value controller with replay and a target network. | Change seed; compare verification rather than loss alone. |
| [05: Policy gradients](notebooks/05_policy_gradient.ipynb) | Can the controller learn action probabilities directly? | REINFORCE and actor-critic alternatives. | Reduce training and examine sample efficiency. |
| [06: PPO](notebooks/06_ppo.ipynb) | How can repeated rollout updates be controlled? | PPO controller with GAE and frozen rollout likelihoods. | Compare stochastic and greedy deployment. |
| [07: Preferences](notebooks/07_preference_and_grpo.ipynb) | Which citation should a preference objective favor? | Separate reward-ranking and categorical DPO components. | Keep the correct number but substitute an unsupported citation. |
| [08: Comparison](notebooks/08_one_problem_many_algorithms.ipynb) | Which controller justifies its training and execution cost? | Shared comparison across seven methods and three seeds. | Compare against a deterministic evidence rule. |
| [09: Language decisions](notebooks/09_small_model_sft_dpo.ipynb) | Can a pretrained model identify eligible source text? | Actual SmolLM2 SFT/DPO; generated include/exclude decisions feed briefing construction. | Require exactly one included source and measure abstention. |
| [10: Group rewards](notebooks/10_verifiable_reasoning_grpo.ipynb) | Can sampled groups improve citation and extraction decisions? | Autoregressive structured policy, with outcome versus process feedback. | Diagnose zero-variance reward groups. |
| [11: Gateway integration](notebooks/11_tool_gateway_integration.ipynb) | How does a policy produce a checked, reviewable artifact? | Read permissions, hard budgets, independent evaluation, and persisted records. | Exhaust the tool allowance and inspect enforcement. |
| [12: Selection](notebooks/12_sampling_and_selection.ipynb) | Which completed briefing should a frozen policy return? | Single sampling, voting, and evidence-ranked candidates. | Swap the scorer while holding candidates fixed. |
| [13: Search](notebooks/13_budgeted_reasoning_search.ipynb) | Where should the next unit of inference work go? | Cloned session branches, beam/best-first search, complete cost accounting. | Raise scorer cost and inspect incomplete searches. |
| [14: Student and delivery](notebooks/14_learning_from_search.ipynb) | Can accepted searches improve a cheaper future policy? | Separate fitted student, held-out comparison, gateway submission, and final artifact. | Reverse the timestamp correlation and expose a learned shortcut. |

## What is shared, and what changes?

**All stages share** the source-eligibility rule, generated workspace records, `Briefing` schema, and independent final evaluator in [goallab.py](../../rl_course/goallab.py). The reference fixtures intentionally have exactly one approved measurement, a matching requested period, and newer distractors.

**Stages 01–08** use a four-action, two-source decision abstraction. Numerical values are omitted from its state because the extraction tool computes row sums, but deploying a controller reads actual records and constructs the same briefing used later. Stage 02 predicts a policy's value rather than producing a better controller. Stages 07 and 10 begin after both reads to isolate preference and trajectory objectives; those prerequisite read costs are not their training objective.

**Stage 09** replaces symbolic eligibility decisions with actual language-model generations. Its source prompts come from the same workspace factory. **Stage 11** integrates a selected routing controller. **Stages 12–14** vary the number of candidate investigations and explicitly charge proposals and score queries. Their reference proposer is frozen and deliberately imperfect; it is not an LLM or automatically the actor trained in stage 10.

The two-source abstraction and frozen-proposer baselines are explicit implementation choices. Extending to arbitrary document collections requires a richer belief representation, evidence handling, and evaluation. The timestamp-shift exercise is included because success on the original generated schema alone can conceal a shortcut.

## How to study and submit

1. Run the reference stage and inspect its actual artifact or metric.
2. Write a prediction before changing the supplied constraint.
3. Run the worked solution, then attempt a different configuration.
4. Keep evaluation cases fixed for the comparison, and report unsuccessful runs too.
5. Explain which component changed and whether the result transfers beyond the reference schema.

Submit the changed configuration, baseline and modified results, one failed trace, and a brief interpretation. A perfectly scoring simple rule can be the correct engineering choice. Demonstrating a learned policy does not establish that learning was necessary.

## Course lesson → project application

Study the course experiment first, then apply the method in the corresponding project stage. Both tracks run independently.

| Stage | Course lesson | GoalLab application |
|---|---|---|
| 01 | [mdp dynamic programming](../../notebooks/01_mdp_dynamic_programming.ipynb) | [Stage 01](notebooks/01_mdp_dynamic_programming.ipynb) |
| 02 | [mc vs td](../../notebooks/02_mc_vs_td.ipynb) | [Stage 02](notebooks/02_mc_vs_td.ipynb) |
| 03 | [q learning](../../notebooks/03_q_learning.ipynb) | [Stage 03](notebooks/03_q_learning.ipynb) |
| 04 | [dqn](../../notebooks/04_dqn.ipynb) | [Stage 04](notebooks/04_dqn.ipynb) |
| 05 | [policy gradient](../../notebooks/05_policy_gradient.ipynb) | [Stage 05](notebooks/05_policy_gradient.ipynb) |
| 06 | [ppo](../../notebooks/06_ppo.ipynb) | [Stage 06](notebooks/06_ppo.ipynb) |
| 07 | [preference and grpo](../../notebooks/07_preference_and_grpo.ipynb) | [Stage 07](notebooks/07_preference_and_grpo.ipynb) |
| 08 | [one problem many algorithms](../../notebooks/08_one_problem_many_algorithms.ipynb) | [Stage 08](notebooks/08_one_problem_many_algorithms.ipynb) |
| 09 | [small model sft dpo](../../notebooks/09_small_model_sft_dpo.ipynb) | [Stage 09](notebooks/09_small_model_sft_dpo.ipynb) |
| 10 | [verifiable reasoning grpo](../../notebooks/10_verifiable_reasoning_grpo.ipynb) | [Stage 10](notebooks/10_verifiable_reasoning_grpo.ipynb) |
| 11 | [Tool-Agent Lab](../../notebooks/11_tool_agent_lab.ipynb) | [Stage 11](notebooks/11_tool_gateway_integration.ipynb) |
| 12 | [sampling and selection](../../notebooks/12_sampling_and_selection.ipynb) | [Stage 12](notebooks/12_sampling_and_selection.ipynb) |
| 13 | [budgeted reasoning search](../../notebooks/13_budgeted_reasoning_search.ipynb) | [Stage 13](notebooks/13_budgeted_reasoning_search.ipynb) |
| 14 | [learning from search](../../notebooks/14_learning_from_search.ipynb) | [Stage 14](notebooks/14_learning_from_search.ipynb) |
