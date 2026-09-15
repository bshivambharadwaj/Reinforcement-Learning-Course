# GoalLab: Build an Agent That Plans, Acts, and Verifies

**Created by Shivam Bharadwaj** · [Course home](../../README.md) · [Chapters](../../chapters/README.md)

Build an evidence-gathering agent that produces a checked briefing within a limited budget. The 14 project notebooks apply course algorithms to shared workspace records, tools, and evaluation. Each has a worked implementation and a runnable constraint-changing exercise.

**[Start the project stages →](STAGES.md)** · [Reproduce the experiments](REPRODUCIBILITY.md)

## Run the reference solution

Install the root `requirements.txt`, then run from the repository root:

```bash
python -m rl_course.goallab_demo --output /tmp/goallab.html --json /tmp/goallab-results.json
```

Open the HTML in a browser. On Windows, use local output paths such as `goallab.html`. The command trains a small Q-learning controller, fits a categorical student from training-workspace search, and compares five methods on supplied held-out cases. No model download or API key is needed for this demo; Stage 09 separately trains a pinned pretrained language model.

A [live visual demo](demo.html) and [reference benchmark records](results.json) are included. GitHub shows HTML source, so download the HTML and open it locally. Click **Run agent**, pause, or advance one step at a time. The browser executes the local workspace: source cards reveal reads, candidate branches appear during search, the briefing changes as decisions are made, and final checks show whether the submitted result is supported.

Change the work budget, controller, case, or timestamp pattern to run a new investigation. The Q-policy and student probabilities are exported from Python training; the browser does not retrain them or call an LLM. Browser sampling is independently seeded, so it is not a replay of Python's random sequence. Recorded benchmark scores remain available in a collapsed comparison panel.

The simple evidence rule is a serious baseline. Its success can exceed a learned or sampled policy on this fully specified schema. The purpose is to measure what each component contributes, including when learning or additional search is unnecessary.

## The task and its contract

> Report the approved measurement for the requested period, with a supporting source.

Each supplied workspace contains two sources, exactly one of which is an approved measurement for the requested period. The newer source is a draft or forecast. The agent reads evidence, selects the eligible source, and submits the sum of its numerical rows with a citation and period. Newer timestamps alone do not establish authority.

The gateway supports search, source reads, submission, and a completion claim. Saying “done” creates no artifact. The `Briefing` stores value, citation, period, and abstention. An independent evaluator checks the submitted value, source eligibility, and period; it is outside the policy interface.

The source records are generated and included through a reproducible workspace factory. No personal accounts or private data are required. Python objects remain inspectable: the gateway is a logical tool interface, not OS-level isolation. Results measure the supplied evidence task, not general document understanding or a deployed service.

## How the course connects

| Course part | Project implementation | What to inspect |
|---|---|---|
| I: Foundations | Stages 01–03: exact MDP, MC/TD prediction, SARSA and Q-learning. | Returns, value estimates, and learned source choices. |
| II: Modern RL | Stages 04–11: neural controllers, preferences, small-model training, group rewards, tool integration. | Learning curves, held-out briefings, and reward gaming. |
| III: Inference-Time Reasoning | Stages 12–14: frozen-policy selection, budgeted search, and a separately fitted student. | Candidate quality, search costs, verified completion, and timestamp-shift failures. |

Algorithms provide alternative implementations of project components. The final system does not run every training algorithm on each request. Each notebook reconstructs its inputs, so previous notebook execution is unnecessary.

## Reasoning at inference time

Stages 12–13 compare candidate briefings from a frozen categorical proposer. Search explores source, extraction, and citation choices in cloned sessions. A visible-evidence scorer ranks candidates; the independent final evaluator checks the selected result afterward. These experiments use explicit proposals and scoring rules, without an LLM planner or learned value model.

Proposals and scoring queries consume the search budget. Inspect both incomplete runs and successful submissions. The demo distinguishes committed tool calls from search work units; those units are not measurements of model tokens or wall-clock latency.

Stage 14 fits a separate student from accepted training searches. Its weights change during fitting, while the search teacher remains frozen. Reverse the timestamp pattern to see whether the student learned source eligibility or a shortcut. Filtered imitation is a separate learning experiment and is not automatically RL.

## Build, predict, change, measure, explain

Run a reference stage, write a prediction, then change one constraint. The [stage map](STAGES.md) links every notebook and its worked exercise.

| Try this change | Investigate |
|---|---|
| Reduce the tool allowance | Which operation is refused, and was a valid briefing submitted? |
| Reduce training episodes | How do value error and held-out completion change? |
| Reward saying “done” | Can training return rise while verified completion falls? |
| Change the candidate scorer | Does the same candidate set produce a worse selection? |
| Increase scorer cost | How much search finishes within the budget? |
| Reverse the timestamp pattern | Does a learned shortcut survive the changed cases? |

Keep evaluation cases fixed. Report the original and changed configuration, results, one failure trace, and your explanation. Compare against the deterministic evidence rule: a simple workflow can outperform a learned policy on a fully specified task.

## Evaluation and results

The [reference records](results.json) include run settings, artifacts, action traces, and independent checks for the supplied held-out cases. Use the [reproducibility guide](REPRODUCIBILITY.md) for environment versions and execution details.

Keep training reward separate from verified completion. Include unsuccessful cases in comparisons and account for their costs. Read permissions, budgets, incorrect citations, and unsupported values are checked by the implementation tests. Current cases always contain a valid source, although the agent can exhaust its budget before finding it.

The [arithmetic inference mini-lab](../../demos/inference-time/README.md) is a separate course experiment. Its results are not GoalLab measurements.

## Executable project stages

The fourteen notebooks in this project directory are GoalLab stages. The [course notebooks](../../notebooks/) retain their original algorithm experiments and diagrams. Start at the [complete stage map](STAGES.md). The modules are [workspace and MDP](../../rl_course/goallab.py), [controller algorithms](../../rl_course/goallab_learning.py), [preferences and group training](../../rl_course/goallab_preferences.py), [search and fitting](../../rl_course/goallab_search.py), and [integrated demo](../../rl_course/goallab_demo.py).

- [Notebook 08](notebooks/08_one_problem_many_algorithms.ipynb): shared-problem algorithm comparisons.
- [Notebook 09](notebooks/09_small_model_sft_dpo.ipynb) and [Notebook 10](notebooks/10_verifiable_reasoning_grpo.ipynb): preference learning and verifiable training feedback.
- [Notebook 11](notebooks/11_tool_gateway_integration.ipynb): GoalLab gateway integration and evaluator gaming.
- [Notebook 12](notebooks/12_sampling_and_selection.ipynb), [Notebook 13](notebooks/13_budgeted_reasoning_search.ipynb), and [Notebook 14](notebooks/14_learning_from_search.ipynb): selection, search, and a separate learning stage.
- [Inference-time demo](../../demos/inference-time/README.md): inspect search behavior and scorer failures.

Educational text uses [CC BY 4.0](../../LICENSE-CC-BY-4.0); course code uses [MIT](../../LICENSE-MIT). Optional third-party models and data retain their own licenses.
