# Build GoalLab: from evidence to a working agent

**Train an RL agent to decide what to search, what to trust, and when to stop.**

[Project home](../README.md) · [Course stages](../STAGES.md) · [Benchmark protocol](PROTOCOL.md)

You are preparing a progress update. Six sources disagree about how many tasks were completed. One is stale, another repeats a misleading report, and a table may need a separate query. You have a limited investigation budget. Should you inspect another source, check its provenance, buy an expensive authoritative response, or answer now?

GoalLab turns that question into a reproducible, small-controller RL benchmark. The environment generates the evidence and answer key; the controller sees only tool observations. STOP is a policy action, and an independent evaluator checks both the conclusion and its collected support.

## Watch an investigation

The [offline explorer](../demo.html) animates recorded investigations, including unsuccessful ones. Select a controller, a case, and a budget. Follow the evidence graph, contradictions, selected citations, cost, confidence, and final verification. This file needs only a browser.

For **live execution**, install the repository requirements and run from the repository root:

```bash
python -m rl_course.goallab_benchmark_demo \
  --report projects/goallab/build-goallab/results.json \
  --checkpoint projects/goallab/build-goallab/controllers-7.pt
```

Open `http://127.0.0.1:8765`. Each step now runs a controller and executes workspace tools in Python. The server binds to localhost. It is a teaching application, not a hardened production service. It uses the included small trained networks, without a model download, external account, or LLM API.

## Build and investigate

| Notebook | What you build | Run |
|---|---|---|
| [01: Generate evidence cases](notebooks/01_generate_cases.ipynb) | Reproducible cases, split boundaries, tool costs, and an evaluator exploit test. | [Colab](https://colab.research.google.com/github/bshivambharadwaj/Reinforcement-Learning-Course/blob/main/projects/goallab/build-goallab/notebooks/01_generate_cases.ipynb) |
| [02: Baselines and DQN](notebooks/02_baselines_and_dqn.ipynb) | A strong metadata-first rule and a learned controller; inspect stopping failures. | [Colab](https://colab.research.google.com/github/bshivambharadwaj/Reinforcement-Learning-Course/blob/main/projects/goallab/build-goallab/notebooks/02_baselines_and_dqn.ipynb) |
| [03: PPO and learned verification](notebooks/03_ppo_and_verifier.ipynb) | Clipped policy updates, a train-only verifier, and optional group-relative training. | [Colab](https://colab.research.google.com/github/bshivambharadwaj/Reinforcement-Learning-Course/blob/main/projects/goallab/build-goallab/notebooks/03_ppo_and_verifier.ipynb) |
| [04: Evaluate under equal budgets](notebooks/04_evaluation_and_search.ipynb) | Verified Success @ Budget, calibration, seed variation, and fully charged search. | [Colab](https://colab.research.google.com/github/bshivambharadwaj/Reinforcement-Learning-Course/blob/main/projects/goallab/build-goallab/notebooks/04_evaluation_and_search.ipynb) |
| [05: Explore the demo](notebooks/05_investigation_demo.ipynb) | A fresh investigation, its evidence graph, an offline explorer, and local live execution. | [Colab](https://colab.research.google.com/github/bshivambharadwaj/Reinforcement-Learning-Course/blob/main/projects/goallab/build-goallab/notebooks/05_investigation_demo.ipynb) |

Each notebook includes setup, a worked example, results, and a change-and-explain exercise. Colab links target published `main`; local changes must be pushed before they appear there. Hosted T4 execution has not been verified. Default experiments use CPU and small networks; a GPU is optional. The original fourteen [project stages](../STAGES.md) remain the focused prerequisites, including simplified Q-learning and group-relative mechanics.

## Reproduce the benchmark

Generate thousands of deterministic fixtures:

```bash
python -m rl_course.goallab_benchmark --count 3000 --split GoalLab-Train --output /tmp/goallab-train.jsonl
```

Train and compare all five methods on the same cases and cost budgets:

```bash
python -m rl_course.goallab_training --episodes 2400 --cases 50 \
  --seeds 7 19 31 --budgets 4 8 12 --output /tmp/goallab-benchmark
```

The command saves controller checkpoints, training histories, every evaluation outcome, and action traces. Paths under `/tmp` are examples; on Windows choose a local output directory. Run details and measured results are in [RESULTS.md](RESULTS.md).

## What this benchmark does and does not establish

The question is whether better investigation policies and inference-time search improve **verified success under the same work allowance**, using the same small controller architecture. The benchmark retains losing methods and tests a timestamp shortcut on the challenge split.

This is structured evidence investigation with three candidate answers and six generated sources. Source parsing and citation assembly are deterministic tools. It does not measure natural-language reading ability, open-web research, cryptographic provenance, or real-world document accuracy. The learned verifier can fail; final verification is separate. A strong rule may be the best solution to this deliberately explicit schema.

Code: [MIT](../../../LICENSE-MIT). Educational text and visuals: [CC BY 4.0](../../../LICENSE-CC-BY-4.0).
