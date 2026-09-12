**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 39](39-grpo-and-reinforcement-learning-for-reasoning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Practice →](../README.md#practical-track)

**Go deeper:** [Chapter 40: Multimodal RL and RL for AI Agents](../chapters/40-multimodal-rl-and-rl-for-ai-agents/README.md)

---

<a id="topic-40"></a>

# 40. Multimodal RL and RL for AI Agents

> **Why this method exists:** Useful agents decide what to observe, which tool to call, and when to stop. RL makes those choices trainable from delayed outcomes, but also exposes the environment and evaluator as parts of the learning system.

Sequential optimization also extends beyond text-only responses.

A multimodal policy may condition on

<p align="center">
  <img src="../assets/equations/equation-73.svg" alt="s_t=(\text{text},\text{image},\text{video},\text{audio},\text{tool state},\text{memory},\ldots)" width="760">
</p>

and choose actions such as

<p align="center">
  <img src="../assets/equations/equation-74.svg" alt="a_t\in\{\text{text generation, tool call, click, code execution, navigation, physical action}\}." width="760">
</p>

This creates several research challenges.

### Multimodal credit assignment

If a model succeeds, which visual observation, reasoning step, textual decision, or tool action deserves credit?

### Long-horizon agent rewards

A useful agent may execute dozens of steps before task success can be evaluated. Sparse final rewards make learning difficult.

### Tool-use policies

The model must learn not only **how** to call a tool, but **when**, **which tool**, and whether another observation is needed before acting.

### Process vs outcome rewards

Outcome reward evaluates the final result. Process reward evaluates intermediate decisions. Both can help, but both can also encode incorrect incentives.

### Safety and constraints

A high reward does not imply that every path to it is acceptable. Real agents often need constrained action spaces, permission systems, safety policies, and external verification in addition to learned rewards.

A useful conceptual architecture is:

<p align="center">
  <img src="../assets/diagrams/multimodal-agent.svg" alt="A tool-using agent learns across many steps" width="760">
</p>

The core RL problem remains recognizable: **learn which actions produce desirable long-term outcomes**. What changes is the scale and richness of the state, action, feedback, and environment.


### Example: an agent fixes a failing program

The agent observes the task, source files, and test output. It chooses actions such as reading a file, editing code, or running a test. Tool results become new observations. A final reward might combine task completion and execution cost, while action constraints define which operations are permitted.

**Analogy:** a delivery robot uses cameras to understand roads; a coding agent uses file contents and tool output to understand a software workspace. Both must select useful observations and actions before the final outcome is known.

A practical training design must specify episode boundaries, what information the policy sees, how success is checked, and whether feedback arrives per step or only at completion. Merely calling tools at inference time is not RL; RL requires an update process driven by interaction outcomes.

Evaluate successful completion, failures, and cost on held-out tasks. Process rewards can guide intermediate decisions, but a plausible-looking intermediate step is not proof that it improves final outcomes.


### Design an agent as an RL system

Topic 40 is the final integration lab: observations, decisions, and feedback now cross tool and
modality boundaries. Start with a precise task contract before selecting an optimizer.

| Component | Tool-agent example | Question to settle |
|---|---|---|
| Hidden environment state | Files, database records, task outcome | What can change without being observed? |
| Observation/history | Task text, screenshot, tool response, memory | Is this sufficient, or is a history/belief representation needed? |
| Policy action | Read, query, calculate, edit, submit, stop | Which actions and arguments are permitted? |
| Transition | Tool execution plus environment change | How are errors, latency, and retries represented? |
| Reward and cost | Verified completion, tool cost, violation count | What is optimized, and what is independently measured? |
| Episode boundary | Accepted submission or exhausted budget | Which boundaries are true task termination versus collection cutoffs? |

### Tool selection and long-horizon credit

A useful action may gather information rather than complete the task. Reading the right file can
enable a later correct patch, yet receive no immediate success reward. Represent failed calls and
their observations in the trajectory. Multi-step returns, critics, process feedback, and task
curricula offer different ways to carry delayed credit; none removes the need for an accurate task
boundary and evaluator. A high-level policy may select a tool while another component generates
arguments. State explicitly which component is trained.

<p align="center">
  <img src="../assets/diagrams/agent-evaluation-loop.svg" width="760" alt="Policy selects tools, environment returns observations, and independent evaluation checks success and cost">
</p>

### Multimodal observations and actions

A VLM can encode task text plus a screenshot, image, or video frame before choosing a text token,
tool call, or spatial action. A robotics policy may additionally consume proprioception and emit
continuous controls. Partial observability remains: one frame need not reveal velocity or hidden
UI state. Learning a better policy cannot by itself recover information absent from its observation.

Separate **perception errors** (the relevant object was misread), **decision errors** (the wrong
tool or action was selected), and **execution errors** (the tool failed). Compare with an oracle
symbolic observation to locate the bottleneck. Log preprocessing, image resolution, coordinate
conventions, and action grounding; these are part of the experiment, not incidental UI details.
Multimodal supervised training and multimodal RL are different training setups; a VLM is not
automatically an RL-trained agent. [PaLM-E](https://arxiv.org/abs/2303.03378) provides context on
embodied multimodal representations, not evidence that every such system uses RL.

### Reward shaping, constraints, and safety

For potential-based shaping, add **gamma × Phi(next state) − Phi(current state)** with boundary
conditions appropriate to the task. In a finite episode with gamma=1 and zero terminal potential,
the additions telescope; arbitrary “progress bonuses” need not preserve the original objective.
[Notebook 11](../notebooks/11_tool_agent_capstone.ipynb) checks this identity on an actual tool trace.

A constrained objective can maximize expected return subject to an expected cost budget. That
is weaker than forbidding a dangerous action on every trajectory. Enforce unavailable operations
outside the policy with tool permissions, validation, or action masks; then measure violations
and invalid requests separately. A large negative reward is not a permission system.
[Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) is a reference for learning under
expected constraints; it does not replace those runtime controls.

### Evaluate the agent, not its self-report

| Evaluation axis | Record | Stress test |
|---|---|---|
| Task completion | Independent verified success rate | Hidden tests and perturbed task instances |
| Efficiency | Calls, tokens, elapsed time, retries | Equal resource budgets |
| Reliability | Seed variation and failure traces | Tool errors, missing observations, longer tasks |
| Constraints | Invalid requests and executed violations | Inputs that tempt a forbidden shortcut |
| Generalization | Split definition and held-out success | New operands, new templates, or new tools—reported separately |
| Reward validity | Training score versus external success | Fake completion messages and answer-list attacks |

> **Failure Mode — evaluator gaming:** an agent rewarded for printing “SUCCESS” can learn to do
that without solving the task. In the capstone, the naive evaluator and a strict submission
checker disagree by design. The exploit is measured with a learned policy, not just described.

### Capstone: learn a tool-selection policy

**[Notebook 11 → Tool-Agent Capstone](../notebooks/11_tool_agent_capstone.ipynb)** trains a policy to
read an arithmetic task, call the appropriate tool, and submit a checked answer. It compares
verified reward, potential shaping, and a vulnerable evaluator across seeds, then tests unseen
operands. Tools do arithmetic; the policy learns routing. This is an agent-RL capstone, not an
LLM or VLM training claim. Replacing the symbolic observation with an image is an explicit extension.

Submit a task contract, result table, success and failure traces, an evaluator exploit test, and
a discussion of what generalizes. A second route is to extend the [shared-task comparison](../notebooks/08_one_problem_many_algorithms.ipynb)
with another algorithm under the same interaction budget.

### Open research questions

How can feedback credit information-gathering actions over long horizons? How do we evaluate
agents when task instances and tools change? Can process supervision stay reliable at scale?
How should multimodal policies distinguish uncertainty in perception from uncertainty in action
value? How can evaluation remain independent as the policy learns to exploit observable tests?
These are questions to investigate, not solved capabilities of this course.

**Knowledge check:** would replacing the learned policy with a fixed sequence of tool calls still
be RL? No. Tool use describes an interface; learning from interaction outcomes supplies the RL part.

References: [WebArena](https://arxiv.org/abs/2307.13854),
[potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[process supervision](https://arxiv.org/abs/2305.20050).

---

**Continue in depth:** [Read Chapter 40](../chapters/40-multimodal-rl-and-rl-for-ai-agents/README.md)

[← Topic 39](39-grpo-and-reinforcement-learning-for-reasoning.md) | [Course home](../README.md) | [Quick reads](README.md) | [Practice →](../README.md#practical-track)
