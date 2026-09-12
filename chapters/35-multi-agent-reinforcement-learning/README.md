# 35. Multi-Agent Reinforcement Learning

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 34](../34-model-based-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 36 →](../36-reinforcement-learning-from-human-feedback/README.md)

[Quick-read Topic 35](../../quick-read/35-multi-agent-reinforcement-learning.md) · [Notation](../NOTATION.md)

- [35.1 Decisions interact through other learners](#section-35-1)
- [35.2 Extend the MDP to joint actions](#section-35-2)
- [35.3 Work a coordination game](#section-35-3)
- [35.4 Distinguish equilibrium from social welfare](#section-35-4)
- [35.5 Explain centralized training and decentralized execution](#section-35-5)
- [35.6 Understand credit assignment and baselines](#section-35-6)
- [35.7 Recognize replay and nonstationarity issues](#section-35-7)
- [35.8 Design cross-play evaluation](#section-35-8)
- [35.9 Laboratory extension and LLM teams](#section-35-9)
- [35.10 Problems, worked answers, and reading](#section-35-10)

---

<a id="section-35-1"></a>

## 35.1 Decisions interact through other learners

Multi-agent RL studies environments with several decision-making agents. Their rewards can be shared, opposed, or partly aligned. The central complication is that other agents' policies shape each learner's effective environment.

**Prerequisites:** Chapters 3, 25, and 27; basic probability. **Targets:** define a stochastic game, distinguish optimality from equilibrium, and design evaluation against varied partners and opponents.

Two couriers sharing a narrow bridge must coordinate. A route that works when the other yields may fail when both learn to insist on crossing first.

<a id="section-35-2"></a>

## 35.2 Extend the MDP to joint actions

A stochastic game specifies a state, each agent's action and observation spaces, a transition distribution conditioned on the joint action, and a reward for each agent. Partial observability may give agents different information.

If other policies are fixed, one agent may view their behavior as part of its environment. When they learn simultaneously, that induced transition/reward process changes over training. A stationary single-agent MDP model can then be inappropriate.

<a id="section-35-3"></a>

## 35.3 Work a coordination game

Two agents choose left or right simultaneously. Both receive 1 if their choices match and 0 otherwise. If each independently chooses left with probability 0.5, expected reward is 0.5.

Both-left and both-right each yield reward 1 and are Nash equilibria: neither agent gains by changing alone. The existence of several good conventions creates a coordination problem. Two agents trained separately can each be competent with their own partners yet fail together.

### Derive best responses in the coordination game

If the other agent chooses left with probability p, choosing left yields expected reward p and choosing right yields 1−p. Left is the unique best response for p>0.5; right is best for p<0.5; both tie at p=0.5.

This gives pure equilibria at both-left and both-right, plus a mixed equilibrium where both randomize equally. The mixed equilibrium has lower team reward, 0.5 rather than 1. An equilibrium concept specifies unilateral incentives, not a ranking by team quality.

<a id="section-35-4"></a>

## 35.4 Distinguish equilibrium from social welfare

A Nash equilibrium means no agent can improve its own expected payoff by unilateral deviation, holding the others fixed. It need not maximize total reward, fairness, or safety.

In a cooperative task, shared reward does not automatically solve exploration or credit assignment. In a competitive task, one policy's raw win rate depends on the opponent distribution. “Best policy” requires a specified game and evaluation criterion.

<a id="section-35-5"></a>

## 35.5 Explain centralized training and decentralized execution

A centralized critic may use joint observations and actions during training, while each deployed actor uses only its permitted local observation. This can reduce ambiguity in learning without granting extra information at execution.

Check that the actor input does not leak centralized information unavailable in deployment. A performance gain from privileged test-time state is a different system, not a successful decentralized policy under the original contract.

<a id="section-35-6"></a>

## 35.6 Understand credit assignment and baselines

With a shared team reward, an individual action's contribution can be obscured by teammates. Counterfactual baselines can compare an agent's chosen action with alternatives while holding other actions fixed under a defined critic.

Such baselines depend on model/critic quality and careful conditioning. They do not reveal a unique philosophical attribution of responsibility; they construct a learning signal for a specified objective. Joint action spaces can also grow exponentially with the number of agents.

### Work a counterfactual baseline with a fixed teammate action

Suppose a centralized critic assigns joint-action values Q(L,L)=4 and Q(R,L)=1 when the teammate chooses L. If the first agent's policy chooses its own L with probability 0.25, its counterfactual baseline for that teammate action is 0.25×4+0.75×1=1.75.

Choosing L gives advantage 2.25; choosing R gives −0.75. The policy-weighted mean is zero for this fixed teammate action. The construction compares one agent's alternatives while holding others' actions fixed, relying on the critic's joint-action estimates. It does not reveal counterfactual outcomes directly from one observed trajectory.

<a id="section-35-7"></a>

## 35.7 Recognize replay and nonstationarity issues

Old replay may reflect obsolete teammate or opponent policies. A transition's meaning depends on the policy population and information available at collection time. Logging partner versions can help interpret the data.

Self-play can generate a useful curriculum but may overfit a narrow population or cycle among strategies. A policy that beats its latest predecessor is not necessarily robust against older or external opponents.

<a id="section-35-8"></a>

## 35.8 Design cross-play evaluation

Evaluate each trained agent with multiple partners or opponents, including independently trained seeds, scripted policies, and held-out strategies. Report a payoff matrix rather than only self-play reward.

In the coordination game, one team may learn left and another right. High within-team scores but low cross-team scores reveal convention dependence. For zero-sum games, exploitability can be informative when a sufficiently strong best-response computation is available; approximate responses only provide limited evidence.

### Interpret a cross-play matrix

Train two independent populations in the coordination task. If population A always chooses left and B always chooses right, within-population reward is 1 but cross-population reward is 0. A matrix with train-seed populations as rows and partners as columns makes this immediately visible.

Now introduce a partner that randomizes equally. Either deterministic convention scores 0.5 against it. Report the partner distribution used for an aggregate score; changing mixture weights changes the evaluation objective. In LLM teams, different prompts using the same underlying model may still share a convention or error, so apparent agent count is not evidence of independent expertise.

<a id="section-35-9"></a>

## 35.9 Laboratory extension and LLM teams

The repository does not ship a multi-agent trainer. A bounded extension is the two-action coordination game with independent learners and a cross-play matrix. [Notebook 11](../../notebooks/11_tool_agent_capstone.ipynb) supplies related tool-evaluation ideas, not a multi-agent implementation.

For multiple LLM agents, separate diversity of prompts from diversity of learned policies. Shared model errors can make several agents agree confidently. Measure final verified outcomes and total tool/token budgets against a strong single-agent baseline.

<a id="section-35-10"></a>

## 35.10 Problems, worked answers, and reading

1. In the coordination game, one agent chooses left with probability 0.8 and the other with 0.3. What is expected shared reward?
2. Why can two perfect self-play teams fail in cross-play?
3. Does a centralized critic permit the deployed actor to use all agents' hidden observations?

<details><summary>Worked answers</summary>

1. Matching probability is 0.8×0.3+0.2×0.7=0.38.
2. They may have learned incompatible conventions, such as always-left and always-right. Self-play tests compatibility within the training population only.
3. No. Decentralized execution constrains the actor's inputs. Privileged critic information during training is a separate allowance.

</details>

Read [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) and [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926).

---

[← Chapter 34](../34-model-based-reinforcement-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 36 →](../36-reinforcement-learning-from-human-feedback/README.md)
