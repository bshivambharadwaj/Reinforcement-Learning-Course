# GoalLab benchmark results

[Build GoalLab](README.md) · [Protocol](PROTOCOL.md) · [All-run records](results.json)

These are measured CPU development results, not a leaderboard or a claim that RL beats a strong rule. Test and challenge results were inspected during development. A publishable generalization claim should use a new locked evaluation set after fixing the implementation and hyperparameters.

## Reference configuration

- GoalLab-v2; Python 3.12.4; PyTorch 2.11.0+cu130; CPU, two Torch threads.
- Three training seeds: 7, 19, 31. DQN and PPO each train for 2400 episodes per seed.
- Learned verifier: 1200 additional Train cases per seed, 30 epochs.
- Fifty evaluation cases per split, levels 1–5 equally represented, at budgets 4, 8, and 12.
- 4500 assigned runs across methods, budgets, splits, and training seeds; these are repeated cases, not 4500 independent templates.
- Controller: 12,832 parameters; verifier: 3,425 parameters.
- Full training and evaluation command took 101.2 seconds in this environment, excluding final JSON serialization. Other machines can differ. No T4 execution or GPU-memory measurement is claimed.

## Verified Success @ Budget

Each entry is the mean over all 150 runs for that method and budget: 50 cases × 3 training seeds. Deterministic baseline cases repeat across seeds. Cost includes failed runs and discarded search branches.

| Split | Budget | Controller | Verified success | Mean cost | Answer coverage |
|---|---|---|---|---|---|
| Challenge | 4 | DQN | 4.7% | 3.89 | 4.7% |
| Challenge | 4 | Greedy | 0.0% | 4.00 | 0.0% |
| Challenge | 4 | PPO | 0.0% | 2.99 | 0.0% |
| Challenge | 4 | PPO + Verifier | 0.0% | 4.00 | 100.0% |
| Challenge | 4 | RL + Test-Time Search | 0.0% | 4.00 | 100.0% |
| Challenge | 8 | DQN | 96.0% | 7.07 | 96.0% |
| Challenge | 8 | Greedy | 100.0% | 7.96 | 100.0% |
| Challenge | 8 | PPO | 66.7% | 5.00 | 66.7% |
| Challenge | 8 | PPO + Verifier | 66.7% | 8.00 | 100.0% |
| Challenge | 8 | RL + Test-Time Search | 0.0% | 7.33 | 66.7% |
| Challenge | 12 | DQN | 94.7% | 9.71 | 94.7% |
| Challenge | 12 | Greedy | 100.0% | 7.96 | 100.0% |
| Challenge | 12 | PPO | 66.7% | 5.00 | 66.7% |
| Challenge | 12 | PPO + Verifier | 67.3% | 9.99 | 100.0% |
| Challenge | 12 | RL + Test-Time Search | 66.7% | 10.67 | 66.7% |
| Test | 4 | DQN | 30.7% | 3.76 | 31.3% |
| Test | 4 | Greedy | 72.0% | 3.52 | 72.0% |
| Test | 4 | PPO | 0.0% | 2.99 | 0.0% |
| Test | 4 | PPO + Verifier | 0.0% | 4.00 | 100.0% |
| Test | 4 | RL + Test-Time Search | 0.0% | 4.00 | 100.0% |
| Test | 8 | DQN | 100.0% | 7.01 | 100.0% |
| Test | 8 | Greedy | 100.0% | 4.22 | 100.0% |
| Test | 8 | PPO | 66.7% | 5.00 | 66.7% |
| Test | 8 | PPO + Verifier | 66.7% | 8.00 | 100.0% |
| Test | 8 | RL + Test-Time Search | 48.0% | 7.01 | 82.7% |
| Test | 12 | DQN | 92.7% | 9.93 | 95.3% |
| Test | 12 | Greedy | 100.0% | 4.22 | 100.0% |
| Test | 12 | PPO | 66.7% | 5.00 | 66.7% |
| Test | 12 | PPO + Verifier | 68.0% | 10.00 | 100.0% |
| Test | 12 | RL + Test-Time Search | 82.7% | 10.99 | 82.7% |

## What failed, and what the results mean

The [pre-shaping development run](failure-before-shaping.json) recorded PPO collapsing to abstention: its greedy evaluation solved no tasks. Immediate abstention avoided wrong-answer penalties, so a superficially improving reward curve did not imply useful investigation.

The reference run adds potential-based shaping from **observed** metadata and support, with zero potential at terminal states. Training-only probe cases were used to check the change before the comparison was rerun. Some seeds still fail. The report exposes per-seed success, answer coverage, unsupported claims, and calibration rather than presenting only a favorable mean.

The cheap rule remains strong because certification and dependency rules are explicit. Learned policies can choose the expensive authoritative response, while search spends units on alternative branches and scoring. Under tight caps, that overhead can leave too little budget to collect evidence. Search is not automatically better, and these results do not demonstrate a benefit from additional test-time search on this schema.

The search method also uses a rule proposal alongside PPO proposals. It is a disclosed hybrid, not evidence that RL alone discovered every successful investigation. Evidence precision and sufficient-proof recall are influenced by deterministic citation assembly. Brier/ECE are conditional on an answer and refer to verified success; read them with coverage.

## Reproduce and audit

```bash
python -m rl_course.goallab_training --episodes 2400 --cases 50 \
  --seeds 7 19 31 --budgets 4 8 12 --output /tmp/goallab-benchmark
python -m unittest discover -s tests -v
```

All final outcomes and observations are retained in the checked-in report. Full traces are retained for the first five case indices of seed 7 at every budget, split, and method, without filtering successes. The command writes every trace. Checkpoints include all three seeds. The interactive explorer defaults to seed 7 because it is first in the declared seed list, not because it performed best.

## Validation

All five Build GoalLab notebooks executed successfully in fresh Python kernels. The full course and project suite contains 33 structurally valid notebooks. The 38 automated checks passed, including evaluation isolation, hard budgets, dependency reads, shaping telescoping, and safe HTML data embedding.

The live localhost API reproduced all 150 preview runs' recorded outcomes and costs. Desktop and 500-pixel-wide browser checks passed for all 150 playback traces and Run/Pause/Step/Reset controls. The optional group-relative trainer passed a short finite-parameter smoke run; it is not part of the five-method reference comparison. Hosted Colab execution remains unverified.
