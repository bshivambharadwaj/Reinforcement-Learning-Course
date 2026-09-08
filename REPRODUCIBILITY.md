# Reproduce the course experiments

The course has 40 theory topics, seven core notebooks, and four integrated labs. Every
notebook includes saved outputs and should run from **Restart Kernel and Run All Cells**.
Saved results are observations from specific configurations, not guaranteed outcomes.

## Install locally

Use Python 3.10 or newer and clone the complete repository. The integrated labs import
the learner-facing `rl_course/` modules. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (RL Course)"
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The explicit PyTorch
command selects CPU wheels on supported platforms; the course does not require CUDA.
Select the new kernel in Jupyter. Dependencies install into the virtual environment,
not inside normal local notebook execution.

For Notebook 09, additionally install:

```bash
python -m pip install -r requirements-modern.txt
```

This lab downloads `HuggingFaceTB/SmolLM2-135M` at immutable revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It uses safetensors and no remote model code.
Only the final decoder layer and normalization train; the model is not saved by default.
Allow several GB of RAM and sufficient cache/disk space. Set `RL_COURSE_MODEL_CACHE` to
choose a cache directory; after downloading, `HF_HUB_OFFLINE=1` supports cached execution.
The model's Apache-2.0 license is separate from the course licenses.

## Run in Colab

Badges open the published `main` version. Notebooks 01–07 use Colab's installed scientific
Python packages. Notebooks 08–11 clone the course's implementation modules into the runtime;
Notebook 09 also installs its pinned modern dependencies. All defaults use CPU. If you have
already imported a conflicting library, restart the runtime after installation.

Colab service availability and preinstalled versions can change. The linked notebooks were
executed locally in fresh kernels; this is not a claim of an independently tested hosted
Colab session. Keep the versions printed by your run when reporting a difference.

## What each result means

All eleven notebooks passed fresh-kernel execution in a clean virtual environment on
2026-09-08. The verification uses Linux x86_64 and Python 3.12.4, with the following
resolved versions. The core dependency ranges also allow other compatible versions;
record yours because floating-point details and training trajectories may differ.

| Dependency | Clean verification version |
|---|---|
| NumPy / Matplotlib | 2.5.3 / 3.11.1 |
| PyTorch | 2.11.0+cpu |
| Transformers / Hugging Face Hub | 5.5.4 / 1.11.0 |
| Tokenizers / Safetensors | 0.22.2 / 0.8.0 |
| nbformat / nbclient | 5.11.1 / 0.10.4 |
| ipykernel / JupyterLab | 7.3.0 / 4.6.3 |

| Notebook | Reproducibility contract | Interpretation |
|---|---|---|
| 01 | Deterministic grid; iterative values checked against a linear solve and shortest paths | Numerical agreement is a correctness check |
| 02 | Paired random-walk episodes, 12 seeds | RMSE estimates one fixed policy; MC/TD settings differ |
| 03 | Cliff walking, five seeds, separate greedy evaluation | A failed greedy seed remains in the report |
| 04–06 | Small CPU networks, three seeds per experiment | Training curves depend on implementation and hyperparameters |
| 07 | Synthetic noisy preferences; exact finite-policy utility | No language model is trained |
| 08 | DeliveryLine-v1; 600 six-step episodes per learner and seed | Model-based DP, fixed-policy prediction, and sampled control are reported separately |
| 09 | SyntheticSentiment-v1; one seed; 48 train / 24 test prompts | Tests unseen nouns under the same template, not general alignment |
| 10 | TwoStepArithmetic-v1; three seeds; disjoint input tuples | Structured neural policy, not pretrained-LM GRPO; fixed-length trajectories |
| 11 | ToolDesk-v1; three seeds; disjoint operand ranges | Learns routing, while tools perform arithmetic; schema stays fixed |

Notebook 08 reports exact expected return, across-seed variation, time to a sampled
checkpoint threshold, peak-to-final regression, and wall-clock time. Equal interaction
budgets are not equal compute budgets. Timing includes periodic evaluation and excludes
imports. A seed standard deviation is not policy-gradient estimator variance.

Notebook 09 deliberately reports both pairwise likelihood preference and free generation.
In the initial run, SFT changed exact generated-label accuracy from 0% to 100%; DPO kept
100% while increasing preference margins. The base model already ranked the correct labels
above incorrect labels. This demonstrates format adaptation and objective behavior, not a
newly learned general sentiment capability. The exact-match metric rejects extra text.

Notebook 10 has limited generalization: outcome-only and process-mixed runs achieve roughly
half of held-out final answers correctly in the initial configuration. Training success
does not prove reasoning generalization. Compare both final and intermediate correctness.

Notebook 11's naive evaluator produces high reward for fake success claims and zero held-out
completion in the initial run. The strict evaluator produces successful submitted results.
Potential shaping preserves total episode return here, but finite training can still produce
different policies and costs. Its symbolic observation is a routing abstraction: accidental
agreement between a wrong arithmetic tool and the correct answer is not fully represented.

## Repeat checks without authoring tools

From the repository root, run one notebook through nbclient in a fresh kernel:

```bash
jupyter execute notebooks/08_one_problem_many_algorithms.ipynb --timeout=600
```

The course's local authoring runner is intentionally excluded from Git. You can also use
Jupyter's Run All operation; the notebooks include assertions for mathematical or system
contracts. Training performance is generally reported rather than enforced as a brittle
pass/fail threshold. Keep private paths and credentials out of saved outputs.

## Upgrade status

| Phase | Delivered material |
|---|---|
| v1.5 | Algorithm family tree, motivation callouts, failure-mode track, nearby references, knowledge checks |
| v2 | One-problem/many-algorithms lab and experiment tables |
| v2.5 | Real small-model SFT/DPO and structured verifiable-reasoning GRPO labs |
| v3 | Tool-agent capstone, expanded Topic 40, evaluation and shaping experiments |

Full language-model PPO/GRPO and VLM fine-tuning are **extensions**, not implemented claims.
The new cover and [social preview](assets/course-social-preview.png) are ready to share;
uploading a GitHub social preview or publishing to LinkedIn/X is a separate account action.
