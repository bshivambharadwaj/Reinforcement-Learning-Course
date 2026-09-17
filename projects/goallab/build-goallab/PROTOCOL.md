# GoalLab-v2 benchmark protocol

[Build GoalLab](README.md)

## Cases and splits

`make_case(index, split, level)` deterministically derives an independent seed from the version, split, index, and difficulty. The names are `GoalLab-Train`, `GoalLab-Test`, and `GoalLab-Challenge`. IDs and generator streams are disjoint. All sets share a structured task schema; this is not a held-out domain benchmark.

Each case contains three possible totals and six shuffled sources: correct, stale, incomplete, duplicated, irrelevant, and contradictory evidence. Modalities are document, message, and table. Tables require a query to extract their values. Metadata inspection reveals certification, reporting-period validity, relevance, publisher history, and dependencies. Publisher history is an uncertain descriptive cue, not the authority rule or a calibrated prediction for this particular claim.

| Level | Main complication |
|---|---|
| 1 | Simple lookup; most reports agree but only valid evidence supports a conclusion. |
| 2 | Conflicting values and copied reports. |
| 3 | Publisher-history cues cease reliably identifying authority. |
| 4 | The valid source requires another record's metadata to establish its dependency. |
| 5 | Misleading source text urges an unsupported completion claim. |

The challenge generator reverses a timestamp cue and adds adversarial prose. Challenge cases are held out from training but publicly reproducible, **not secret**. This version uses fixed structured features, so malicious prose is displayed but never executed; it does not test LLM prompt-injection resistance. Source reliability is represented as observed metadata, not a learned latent publisher reputation process.

The JSONL generator includes labels for evaluation. Controllers receive `Investigation.observation()`, never raw JSONL records, source roles, or the answer key. Python introspection is not prevented; service isolation is outside this benchmark.

## Actions, costs, and stopping

The workspace initially exposes the goal, candidate totals, source IDs, types, and timestamps. Search reveals snippets. Other tools open content, inspect metadata, query a table, cross-check the observed sources, or request authoritative evidence. A paid authoritative response costs **6** work units and provides a sufficient proof. Other actions, including STOP and abstention, cost **1** unit. Repeated requests still cost work; the supplied policies mask redundant reads.

STOP selects an observed candidate with reported confidence 0.50, 0.75, or 0.95. The executor attaches the inspected sources that satisfy the visible support rule. It does not substitute a correct answer for the policy's choice. Citation generation is therefore controlled by the executor, not independently learned. A source supports the claim only after its content/value and metadata are collected, its current/relevant/certified flags pass, and its dependency metadata is inspected. The independent evaluator additionally compares against the case's answer key.

Budget exhaustion without an answer is a failed episode. An abstention is counted as non-completion; all current cases have an answer in principle. Abstention avoids the larger wrong/unsupported-answer penalty, which can make it attractive to a poorly trained controller. Report coverage alongside success.

The observation tracks remaining cost, inspected evidence, previous action, contradictions, and reported confidence. Fixed-length features summarize the observed workspace; they do not encode unobserved contents or the hidden correct option. These are compact controllers, not language models.

## Training objective

Each executed action costs `0.03 × work units` in reward. At termination, the training evaluator gives:

```text
+2.0 × verified success
+0.5 × answer correctness
−0.8 × wrong answer
−0.8 × unsupported answer
−0.25 × (reported confidence − verified success)²
```

Abstention or exhaustion instead receives −0.3, plus accumulated action costs. These are transparent teaching weights, not universally optimal values. Unnecessary reads lose reward through their cost; premature answers fail support checks. There is no magical reward-hacking detector: unsupported completion is penalized by the independent contract. The notebook tests an exploit that supplies the correct value with unread evidence.

Both controllers add potential-based shaping, `0.99 × potential(next observation) − potential(current observation)`. The potential uses inspected metadata and collected support, is zero at terminal states, and never reads the answer key. Its discounted contributions telescope; the final benchmark still measures unshaped verified success. Training histories report shaped episode returns.

DQN uses replay, a target network, and Double-DQN targets. PPO uses complete variable-length episodes, GAE, clipped updates, entropy regularization, gradient clipping, and an approximate-KL early stop. Training cases and budgets are sampled from Train only. The optional group-relative variant groups six trajectories from the same case and uses normalized returns with a clipped objective; it is not pretrained-language-model GRPO.

The learned verifier predicts **joint correctness and collected support** from observation features plus a proposed candidate. Its supervised labels use Train only. It receives the same observable evidence at inference, and never queries final evaluation. The PPO critic estimates expected return; the verifier predicts support-aware success. These are different quantities.

## Equal-budget comparisons

| Method | Decision rule |
|---|---|
| Greedy | Inspect metadata in recency order, acquire eligible contents/dependencies, then answer. |
| DQN | Greedy action from the learned Q-network. |
| PPO | Greedy action from the trained categorical policy. |
| PPO + Verifier | Pay for a verifier query, stop above its fixed 0.65 threshold, otherwise continue using PPO's non-STOP actions. |
| RL + Test-Time Search | Expand at most two levels of branches using PPO proposals plus an explicit rule proposal; rank by learned verification and critic value. |

Every comparison uses the same case IDs, tool permissions, and **total work-unit cap**. Verifier queries cost one unit. Search charges all expanded tool calls and one scoring query per branch, including discarded branches; selected branches do not erase these costs. Branch snapshots retain only their own observed evidence. Extra controller forward passes and wall-clock time are not equated to work units. Report total runtime separately; this is not equal-FLOP benchmarking.

Search is a hybrid using both learned and rule proposals. The rule proposal is an explicit engineering choice, so gains cannot be attributed solely to RL. At one remaining unit, verifier-assisted methods use the same visible-evidence rule to stop or abstain without another paid score query. Policies and verifier weights are frozen throughout evaluation. Final verification is called only after a run finishes.

DQN and PPO receive the same number of training episodes, not necessarily the same number of tool actions or optimizer operations. The report includes actual actions. Verifier training consumes additional Train cases and is disclosed separately. The default seed list and budgets are fixed before evaluation. Do not tune on the public challenge scores and present them as untouched final evaluation.

## Metrics

**Verified Success @ Budget** is the fraction of all assigned investigations with a correct, supported answer completed within the cost cap. Accuracy is answer correctness alone. Unsupported-claim rate counts unsupported answers divided by all assigned episodes. Answer coverage makes abstention visible.

Evidence precision is the fraction of cited sources satisfying collected-support requirements. Evidence recall is **sufficient-proof recall**: one inspected independent proof is enough. It is not recall over every document mentioning the answer. Duplicate claims do not create extra gold evidence.

Average cost includes all failed runs, score queries, and discarded branches. Average tool calls excludes score queries. Brier score and three-bin ECE measure the reported probability of **verified success**, conditional on issuing an answer; abstentions are excluded and coverage is reported alongside them. Confidence bins are intentionally coarse. Seed-specific success rates are included rather than hiding variation behind a single mean.

## Reproducibility boundary

The compact 64-unit controller and 32-unit verifier have no pretrained-model or paid-API dependency. CPU reference runs are provided. The architecture is small enough for a 16 GB device by parameter count, but no free Colab T4 memory or runtime measurement is claimed. Colab badges are setup entry points, not proof of hosted execution.

Tests cover split reproducibility, hidden-answer isolation, unread citations, dependency enforcement, hard budgets, exhaustion, frozen inference, and full search cost accounting. All-run records remain available even when a method fails. The interactive offline preview is explicitly labelled playback; the localhost server executes actual tool steps.
