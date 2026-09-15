# Reproduce the GoalLab course stages

These 14 project notebooks implement GoalLab stages alongside the 14 original course notebooks. Use the course lessons for their original diagrams, environments, and algorithm experiments; use this track for the staged project application.

## Install and run

Use Python 3.10 or newer and clone the full repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (RL Course)"
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Select the installed kernel and choose **Restart Kernel and Run All Cells**. Every notebook recreates its inputs from shared GoalLab modules, so previous notebook execution is unnecessary. Training uses CPU with two PyTorch threads.

Stage 09 additionally needs `python -m pip install -r requirements-modern.txt`. It downloads `HuggingFaceTB/SmolLM2-135M` at pinned revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, using safetensors and no remote model code. Only the final decoder layer and final normalization train. Allow several GB of RAM. Set `RL_COURSE_MODEL_CACHE` to choose a cache; `HF_HUB_OFFLINE=1` supports subsequent cached execution. The model has its own Apache-2.0 license.

All Colab notebooks import the repository, cloning it when necessary. Stage 09 additionally installs modern dependencies. Badges target published `main`, so local updates must be pushed before they appear there. Hosted Colab execution has not been independently verified.

## Verified execution

All 14 project notebooks passed fresh-kernel execution before relocation into the project directory on **2026-09-15**, using the available Linux Python environment below. This was not a newly provisioned clean virtual environment. Saved outputs include unsuccessful policies and distribution-shift results. After relocation, stages 01, 11, and 14 passed another fresh-kernel run from their new directory. Both tracks passed notebook structure and local-link checks; original course code cells and saved outputs match the restored revision exactly.

| Component | Execution version |
|---|---|
| Python | 3.12.4 |
| NumPy / Matplotlib | 1.26.4 / 3.8.4 |
| PyTorch | 2.11.0+cu130, tensors and training on CPU |
| Transformers | 5.5.4 |
| nbformat / nbclient | 5.9.2 / 0.8.0 |

New installations should use the checked-in requirements; record resolved versions when comparing outcomes. Training trajectories and wall times can differ across compatible versions or hardware.

## Stage contracts

| Stages | Contract | Interpretation |
|---|---|---|
| 01 | Four actions; exact belief-state DP; cost 0.02 each; success bonus one. | Optimal initial return 0.92 is a mathematical check. |
| 02 | MC and TD share random-policy episodes; seeds 7, 19, 31; exact evaluation. | Prediction RMSE, not controller success. |
| 03–06 | Same two-source abstraction; tabular and neural controllers execute on actual held-out records. | Numerical payloads change; the authority schema does not. |
| 07 | Synthetic source preferences; reward model and separate uniform-reference DPO policy. | Categorical mechanics, not human feedback or LLM training. |
| 08 | Seven controllers; 600 four-action episodes per seed; seeds 7, 19, 31. | Equal interactions do not imply equal optimizer compute. |
| 09 | 32 train / 16 held-out prompts; one seed; 36 SFT and 24 DPO steps. | Real pretrained source-eligibility training, not broad alignment. |
| 10 | Citation then extraction mode; 64 train / 64 held-out workspaces; groups of eight. | Structured autoregressive policy, not pretrained-LM GRPO. |
| 11 | Verified and completion-claim rewards; common gateway and independent checker. | High proxy reward can produce no briefing. |
| 12–13 | 80 cases; seeds 7, 19, 31; caps 6, 12, 24, 48. | Frozen categorical proposals; candidate and prefix scores incur different costs. |
| 14 | 160 training cases for fitting; 100 held-out cases per evaluation seed. | Filtered imitation and a deliberately exposed timestamp shortcut. |

GoalLab-v1 has exactly one approved measurement per requested period. Approved rows use 0..3 in training and 4..7 in test; distractors have different values and newer timestamps. IDs and periods differ across splits. This is a controlled schema split, not a held-out domain or unseen authority rule.

The early MDP has two reads, an internal citation choice, and a submit/claim action. Under exactly two sources with one valid source, reading either reveals the authority placement. This belief-state simplification is not valid for arbitrary document sets. Values are abstracted out because tools compute row sums; deployment still produces the actual artifact.

## Costs and evaluator boundaries

Final verification checks citation eligibility, value, and period. It is not a search action; tests patch it to fail if inference calls it. Training rewards use training records. Python objects are inspectable, so this is a logical interface rather than hardened isolation.

Stages 12–14 charge one unit per proposed decision plus a configured score-query cost. Best-of-N scores completed candidates once; tree methods score every expanded prefix. Duplicates and incomplete branches still cost work. Final evaluation is outside the allowance. These units are not FLOPs, tokens, latency, or money.

The integrated report separates **committed calls** from **search units**. Committed calls describe only the selected execution, excluding discarded branches. The MDP's internal citation choice counts as a decision event, not a real external API call. The JSON also records student-data generation cost; fitting compute is not included in the work-unit break-even example.

Stages 07 and 10 start after both reads to isolate preference and group-update mechanics. Stage 09 reports classification separately from its briefing pipeline, which uses the first generated `include` without correction by the final evaluator.

## Read the results

Initial Stage 08 mean held-out success was 100% for tabular methods and DQN, 62.7% for REINFORCE, 43.3% for actor-critic, and 93.0% for PPO at the common 600-episode budget. These are implementation-specific outcomes, not general algorithm rankings. The fixed evidence rule also succeeds without training and uses fewer decision events.

Stage 09 achieved 100% generated source-decision accuracy after SFT on 16 held-out prompts; DPO retained that accuracy while increasing preference margins. All eight final briefing cases passed. The templates and eligibility rule remain simple, and the sample is small.

Stage 14 fitted a student from 86 accepted training traces. Mean held-out single-attempt success was approximately 97.7% across three generation seeds. Reversing the timestamp/authority relationship reduced success to 0% in a separate 100-case stress test. This demonstrates a brittle shortcut, not reliable general document reasoning.

The [GoalLab report](results.json) uses a separate 40-case configuration and seed 7; its percentages differ from notebook aggregates. The [older arithmetic demo](../../demos/inference-time/README.md) is a separate experiment.

The [visual demo](demo.html) executes an in-browser workspace engine with exported Q-policy and student parameters. It supports Run/Pause/Step/Reset, budget changes, and timestamp shift. Its seeded sampler differs from Python's, so compare contracts and distributions rather than expecting identical individual sampled paths. The recorded Python benchmark is shown separately. No external account or live LLM is connected.

## Repeat checks

```bash
python -m unittest discover -s tests -v
jupyter execute projects/goallab/notebooks/01_mdp_dynamic_programming.ipynb --timeout=600
python -m rl_course.goallab_demo --output /tmp/goallab.html --json /tmp/goallab-results.json
```

Tests cover determinism, artifact validation, read permissions, budgets, snapshot independence, model/deployment agreement, candidate matching, evaluator isolation, and separate student fitting. Legacy arithmetic checks remain for that demo. Performance is reported rather than enforced with brittle success thresholds.

The local authoring runner stays excluded from Git. Use Jupyter Run All or `jupyter execute`. Keep caches and personal experiment outputs outside the tracked tree. See the [stage map](STAGES.md) for each notebook's contribution and worked exercise.
