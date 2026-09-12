# Assessment: from a correct equation to a defensible result

[Chapters](README.md) · [Notation](NOTATION.md) · [Notebooks](../README.md#practical-track)

**Created by Shivam Bharadwaj**

These original assignments combine the chapters into a sustained course of study.
For inference-time reasoning, use the additional [Part III exercises and practical assessment](PART_III_EXERCISES.md).
The chapter exercises check local understanding; these assessments require transferring
ideas, defending assumptions, and producing reproducible evidence. A correct final
number without the model or reasoning is incomplete.

## Suggested sequence

The schedule is a suggested workload, not a measured completion-time promise. Allocate
one or two study sessions to a problem set and several sessions to each experiment;
background, hardware, and the scope of extensions will change the time required.

| Assignment | Chapters | Deliverable | Suggested stage |
|---|---|---|---|
| 1. A deadline changes the policy | 1–16 | Model, exact solution, proof, numerical check | Foundations |
| 2. Data and bootstrap boundaries | 17–24 | Estimator analysis and a controlled ablation | Sampled learning |
| 3. A policy update under a microscope | 25–32 | Gradient derivation and update diagnostics | Policy optimization |
| 4. What the archive cannot tell you | 33–35 | Identification argument and evaluation design | Generalization |
| 5. Preferences, groups, and verification | 36–39 | Objective audit and generated-output report | Post-training |
| 6. End-to-end capstone | Relevant chapters through 40 | Reproducible comparison and failure analysis | Integration |

Prerequisites are Python, probability, basic calculus, linear algebra, and willingness
to inspect a failed experiment. The early assignments use tiny models so mathematical
understanding does not depend on access to a large accelerator.

## Assignment 1: a deadline changes the policy

A courier begins at depot A with h actions remaining. At A, `deliver_now` terminates
with reward 2; `prepare` pays −1 and moves to B. At B, `dispatch` terminates with reward
6 with probability p and reward 0 otherwise; `wait` pays −0.5 and stays at B. No reward
arrives after the horizon. Use gamma=0.9 and p=0.8 unless varying them explicitly.

1. Specify the state representation and transition/reward table, including horizon
   termination. Explain why location alone is insufficient.
2. Derive V_h(A) and V_h(B) for h=0,1,2,3. Identify all optimal actions and ties.
3. Derive the p threshold at which preparing is optimal with two actions remaining.
4. Prove the optimal Bellman operator's discounted contraction for a stationary finite
   MDP. Explain why finite-horizon backward induction does not require gamma<1.
5. Implement an independent enumerator or backward recursion and compare it with the
   planning structure in [Notebook 01](../notebooks/01_mdp_dynamic_programming.ipynb).
   Change p after planning and distinguish solver error from model mismatch.

<details><summary>Solution milestones</summary>

Represent state as (location,h), with h=0 terminal. V_0 is zero. At h=1,
V_1(A)=2 and V_1(B)=4.8. At h=2, V_2(A)=max(2,−1+0.9×4.8)=3.32;
V_2(B)=max(4.8,−0.5+0.9×4.8)=4.8. Values remain [3.32,4.8] at h=3.
Waiting is inferior here because it costs reward and delays an unchanged opportunity.

For h=2, preparing has value −1+5.4p. It ties 2 at p=5/9 and is strictly better above
that threshold. A planner using p=0.8 can choose prepare even if deployment p is below
5/9; accurate solution of the assumed model does not prevent this mismatch.

For contraction, use the nonexpansiveness of max over action backup vectors, then the
fact that each transition row averages continuation differences. This gives a gamma
factor in maximum norm. Backward induction instead has a fixed finite number of stages
and a known boundary, so it does not need an infinite fixed-point iteration to converge.

</details>

## Assignment 2: data and bootstrap boundaries

Use a fixed policy before attempting control comparisons. An observed two-step segment
has rewards [1,2], gamma=0.9, values V(s_0)=0.5 and V(s_1)=1, and an endpoint estimate
V(s_2)=4. Treat it first as a true completed episode, then as a collector cutoff in a
continuing task. The next stored reset observation has value 20.

1. Calculate full/partial returns, one-step TD targets, and GAE with lambda=0.8 under
   both boundary interpretations. Explain which targets are MC and which bootstrap.
2. Calculate the error caused by bootstrapping from the reset observation instead of
   the final observation.
3. Compare MC and TD in [Notebook 02](../notebooks/02_mc_vs_td.ipynb) at equal interaction
   budgets. Report all chosen seeds and prediction error against the known reference.
4. Add one deliberate boundary bug in an isolated experiment. Predict its direction
   before running it, demonstrate its effect, then restore the correct implementation.
5. Explain why a low neural replay loss would not prove the corrupted targets were valid.

<details><summary>Solution milestones</summary>

For true termination, returns are [2.8,2]. One-step targets are [1.9,2], residuals
[1.4,1], and GAE advantages [2.12,1]. For a cutoff with endpoint bootstrap, two-step
and one-step-to-end returns are [6.04,5.6]; TD targets are [1.9,5.6], residuals
[1.4,4.6], and GAE is [4.712,4.6]. These cutoff returns are bootstrapped estimates,
not completed Monte Carlo observations of the continuing task.

Using reset value 20 instead of 4 increases the last target by 0.9×16=14.4,
the first two-step target by 0.9²×16=12.96, and the first GAE advantage by
0.9×0.8×14.4=10.368. The trace must stop before the reset episode in either case.

A function approximator can fit these incorrect numerical targets. Loss measures fit
to the supplied labels; the task contract and an independent transition trace establish
whether those labels correspond to the intended return.

</details>

## Assignment 3: a policy update under a microscope

Consider a one-step policy choosing action L with probability sigmoid(z). L pays 4 and
R pays 0. Begin at probability 0.25. Then examine a PPO batch with old probabilities
recorded at collection time.

1. Derive the exact expected-return gradient and verify it by enumerating the two
   score-function outcomes. Repeat with an action-independent baseline of 1.
2. Derive baseline cancellation and explain why an action-dependent baseline generally
   fails. State what must be detached in the actor implementation.
3. With epsilon=0.2, calculate PPO sample objectives for (A,ratio) equal to
   (2,1.5), (−2,0.5), (−2,1.5), and (2,0.7).
4. Derive the local trust-region step for scalar g=2, F=4, delta=0.02. Compare the
   quadratic KL prediction with an actual Bernoulli-policy KL at a proposed logit step.
5. In [Notebook 06](../notebooks/06_ppo.ipynb), vary update epochs while holding rollout
   and interaction budgets fixed. Report return, measured policy divergence, and
   clipping diagnostics. Explain why the empirical surrogate is not a monotonic-return theorem.

<details><summary>Solution milestones</summary>

The exact gradient is 4p(1−p)=0.75 at p=0.25. With baseline 1, the weighted outcomes
are 2.25 and 0.25, with probabilities 0.25 and 0.75, giving the same mean 0.75.
The baseline term cancels because sum_a gradient pi(a|s)=0. Detach the advantage or
baseline-derived weight when it is used as a score coefficient.

The four clipped sample objectives are 2.4, −1.6, −3, and 1.4. Clipping removes
some incentive for beneficial movement beyond the interval but does not project the
policy into a hard feasible set.

The scalar trust-region step is 0.1. The supplied F=4 is a separate illustrative
quadratic model, not the Bernoulli logit's actual Fisher. For the Bernoulli policy,
F=p(1−p)=0.1875 at the starting point. Use that curvature when comparing an actual
Bernoulli KL with its local approximation; mixing the two models invalidates the check.

</details>

## Assignment 4: what the archive cannot tell you

A fixed dataset contains 100 one-state episodes, all choosing L and receiving reward 1.
Action R was never chosen. In one compatible environment R pays 10; in another it pays −10.

1. Prove that these data do not identify the value of a policy that sometimes chooses R.
2. Write that policy's return in both environments as a function of its R probability q.
3. Explain why a neural estimate of Q(R)=20 and zero fitted loss on L do not resolve the issue.
4. Propose an evaluation protocol under each of two conditions: no new interactions,
   and a limited final online test budget. List the assumptions each protocol requires.
5. For a multi-agent extension, explain why changing an unlogged partner policy can
   create a similar ambiguity in interpreting old transitions.

<details><summary>Solution milestones</summary>

The dataset likelihood is identical in both environments under always-L behavior.
The candidate policy's values are 1+9q and 1−11q. Any q>0 introduces a difference
that the archive cannot resolve. Always-L attains 1 in both.

The unsupported Q prediction is a consequence of extrapolation or initialization,
not observed evidence. Logged-data evaluation must acknowledge the missing support or
introduce explicit model assumptions. A held-out online test can gather new evidence,
but its budget and role in model selection must be recorded. Partner-policy changes
alter the induced transition/reward process and should not be silently treated as a
stationary single-agent environment.

</details>

## Assignment 5: preferences, groups, and verification

Audit an optimization objective before interpreting a generated answer. Use the small
post-training and reasoning labs as controlled experiments, not broad ability benchmarks.

1. Derive the DPO pairwise loss from the reference-KL optimum, identifying the term
   that cancels within a prompt. Calculate the loss when the reference-relative
   log-probability margin is 1 and beta=0.1.
2. Explain a case where a better preference margin does not increase the preferred
   response's absolute likelihood. Specify a generated-output metric to accompany it.
3. For binary rewards with success probability p, derive the probability that a group
   of G independent attempts contains both a success and a failure.
4. For rewards [0,0,1,1], compute centered and standardized group signals. Explain the
   all-failure case and the self-inclusion issue for the group mean baseline.
5. In [Notebook 09](../notebooks/09_small_model_sft_dpo.ipynb), compare base, SFT, and DPO
   checkpoints under the same test prompts and decoding. In
   [Notebook 10](../notebooks/10_verifiable_reasoning_grpo.ipynb), compare outcome and
   process feedback. Report final correctness and intermediate validity separately.

<details><summary>Solution milestones</summary>

The regularized optimum gives r=beta log(pi/pi_ref)+beta log Z(x). The partition
term cancels for two responses to the same prompt, yielding −log sigmoid(beta m).
At m=1 and beta=0.1, loss is approximately 0.6444. A winner log probability can fall
from −2 to −3 while a loser falls from −3 to −5; the margin still improves from 1 to 2.

The mixed-group probability is 1−p^G−(1−p)^G. Centered signals for the specified
group are [−0.5,−0.5,0.5,0.5]; population-standardized signals are [−1,−1,1,1].
An all-failure group's centered signal is zero. Without standardization, including
each sample in its own group mean scales the expected independent-sample score
estimator by (1−1/G); normalization introduces further sample dependence.

Generated accuracy and process checks can reveal differences hidden by pairwise
likelihood metrics. The lab report must retain the narrow task scope and distinguish
the actual small language model in Notebook 09 from the structured policy in Notebook 10.

</details>

## Assignment 6: end-to-end capstone

Choose one primary task and a comparison that can answer a clear question. Two supported
starting points are the common environment in
[Notebook 08](../notebooks/08_one_problem_many_algorithms.ipynb) and the tool task in
[Notebook 11](../notebooks/11_tool_agent_capstone.ipynb). A new environment is acceptable
if its implementation and evaluator are fully specified.

Produce these artifacts:

1. **Task contract:** observations, hidden information, actions, reward, horizon,
   termination/truncation, initial-state distribution, and independent success check.
2. **Methods and baselines:** at least two appropriate learning methods and a simple
   baseline. A DP oracle can be an additional reference, with its model access labeled.
3. **Budget table:** environment transitions, optimizer steps, training data, inference
   attempts, tool calls, and hardware/runtime where measured. Explain which budgets
   are controlled and which are only reported.
4. **Evaluation plan written before final testing:** task splits, seeds, checkpoint
   selection rule, primary metric, failure metrics, and uncertainty summaries.
5. **Results:** all specified seeds, readable curves/tables, and the final frozen-policy
   evaluation. Do not rank prediction estimators as controllers.
6. **Failure-oriented ablation:** intentionally remove a relevant safeguard or assumption
   in an isolated experiment, predict the effect, and demonstrate whether it occurs.
   Examples include a wrong timeout bootstrap or a naive success-claim evaluator.
7. **Reproduction instructions:** dependency versions, exact commands, data provenance,
   model revisions if applicable, and a fresh-kernel notebook or equivalent runnable entry point.
8. **Interpretation:** the strongest supported conclusion, a plausible alternative
   explanation, and the next experiment that would distinguish them.

For a tool-agent project, compare proxy reward with verified completion. For a reasoning
project, compare single-attempt and fixed-budget multi-attempt success. For a multimodal
extension, separate perception accuracy from executed task success. A system diagram
should show where external evidence enters and who determines completion.

## Assessment rubric

| Criterion | Weight | Strong evidence |
|---|---:|---|
| Formal problem and assumptions | 20% | Complete contract; correct boundaries and information structure |
| Mathematical reasoning | 20% | Correct derivations; explicit proof limits; independent numerical checks |
| Implementation and reproducibility | 20% | Traced updates; recorded versions; reproducible fresh execution |
| Experimental design | 20% | Suitable baselines; controlled comparisons; independent held-out evaluation |
| Failure analysis and interpretation | 15% | Failed runs retained; alternative explanations tested; claims match scope |
| Communication and attribution | 5% | Clear figures/tables; original explanation; relevant primary references |

A high task score cannot compensate for evaluation leakage or an incorrect objective.
A low-performing method can still support an excellent report when the experiment
correctly identifies and explains its limitations. This rubric is a self-study and
contribution tool, not an institutional certification.

## Submission review checklist

Confirm that every claimed result has an associated artifact, that each baseline had
the stated information and budget, and that all links and commands work from a clean
checkout. Ask whether a reader could distinguish measured evidence from a proposed
extension without consulting this conversation or the author.

Educational text uses [CC BY 4.0](../LICENSE-CC-BY-4.0); code uses [MIT](../LICENSE-MIT).
