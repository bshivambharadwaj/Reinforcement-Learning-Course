# 33. Offline Reinforcement Learning

**Advanced** · **Created by Shivam Bharadwaj**

[← Chapter 32](../32-entropy-regularization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 34 →](../34-model-based-reinforcement-learning/README.md)

[Quick-read Topic 33](../../quick-read/33-offline-reinforcement-learning.md) · [Notation](../NOTATION.md)

- [33.1 Learn when new exploration is unavailable](#section-33-1)
- [33.2 Specify the dataset and deployment setting](#section-33-2)
- [33.3 Prove a simple non-identifiability result](#section-33-3)
- [33.4 Explain extrapolation through a maximum](#section-33-4)
- [33.5 Compare behavior cloning and conservative improvement](#section-33-5)
- [33.6 Describe conservative Q-learning at a high level](#section-33-6)
- [33.7 Evaluate with logged data cautiously](#section-33-7)
- [33.8 Work a weight-concentration example](#section-33-8)
- [33.9 Laboratory extension and preference-data bridge](#section-33-9)
- [33.10 Problems, worked answers, and reading](#section-33-10)

---

<a id="section-33-1"></a>

## 33.1 Learn when new exploration is unavailable

Offline RL learns a policy from a fixed dataset of transitions or trajectories. The learner cannot resolve uncertainty by trying an unsupported action during training. This makes coverage a central limitation rather than an exploration hyperparameter.

**Prerequisites:** Chapters 17, 20–23, and 25. **Targets:** construct an identifiability counterexample, distinguish offline RL from behavior cloning, and design defensible evaluation.

The courier receives an archive of other drivers' journeys but cannot drive new routes before choosing a policy. Unvisited roads remain unknown even if a neural network assigns them confident values.

<a id="section-33-2"></a>

## 33.2 Specify the dataset and deployment setting

Record how the behavior policies collected data, which states and actions appear, reward and termination semantics, and whether the environment has changed. A fixed dataset can contain multiple behavior policies and mixtures of quality.

Offline training and online fine-tuning are different phases. If evaluation rollouts influence hyperparameter selection repeatedly, the procedure is using environment feedback even if gradient updates use only the archive. Report that budget honestly.

<a id="section-33-3"></a>

## 33.3 Prove a simple non-identifiability result

There is one state and two terminal actions. The dataset contains only action left with reward 1. In model M_good, right pays 10; in M_bad, right pays −10. Both models generate exactly the same observed dataset under behavior that always chooses left.

No method using only these observations can determine which model is true without extra assumptions. A policy that confidently chooses right is not justified by the data alone. This is an information limitation, not an optimizer failure.

### Extend the unsupported-action example to a minimax choice

Let q be the probability of choosing the unobserved right action. The two compatible environments give values 1+9q and 1−11q. The worst-case value across them is 1−11q for q≥0, maximized at q=0.

This robust choice retains the known left action. It does not prove right is bad; it protects against a plausible environment the data cannot rule out. If an additional trustworthy assumption bounds right's reward below by 2, the robust decision changes. Offline conclusions depend jointly on observations and stated assumptions.

<a id="section-33-4"></a>

## 33.4 Explain extrapolation through a maximum

Offline Q-learning can assign arbitrary values to unsupported actions through function approximation. Its bootstrap maximum then selects those values and propagates their errors into supported states.

Low training loss does not resolve this problem: the targets themselves can depend on unsupported predictions. Merely increasing the number of epochs on the same dataset can intensify that self-consistency without adding evidence about the missing actions.

<a id="section-33-5"></a>

## 33.5 Compare behavior cloning and conservative improvement

Behavior cloning fits pi(a|s) to logged actions. It avoids explicit reward maximization but inherits dataset behavior, including mistakes, and can suffer compounding errors after leaving the logged state distribution.

Offline RL attempts reward-based improvement. Strategies include keeping actions close to behavior support, pessimistic value estimation, and constrained objectives. They introduce assumptions and tradeoffs; conservative behavior may avoid unsupported optimism while missing genuinely good unobserved actions.

<a id="section-33-6"></a>

## 33.6 Describe conservative Q-learning at a high level

Conservative Q-learning adds pressure against high values on broadly sampled or candidate actions relative to dataset actions, alongside a Bellman regression term. A discrete illustrative expression includes logsumexp_a Q(s,a)−E_(a~data)[Q(s,a)].

This expression alone is not a complete implementation or a universal lower-bound theorem. The full method's sampling, weighting, function class, and theoretical assumptions matter. Use the original paper when implementing a particular variant.

### Differentiate a simple conservative penalty

For one state with dataset action L, the illustrative penalty is log(exp(Q_L)+exp(Q_R))−Q_L. Its derivative with respect to Q_R is softmax_R, positive; its derivative with respect to Q_L is softmax_L−1, negative.

Minimizing this term suppresses R relative to the supported action L. It must be combined with the specified value-learning objective; on its own, it does not identify either true action value. The relative penalty and Bellman regression can pull in different directions, and their coefficient affects how conservatism trades against fitting observed returns.

<a id="section-33-7"></a>

## 33.7 Evaluate with logged data cautiously

Importance sampling needs behavior support and reliable behavior probabilities; long products can have severe variance. Fitted Q evaluation trains an evaluator for a fixed candidate policy, but inherits approximation and coverage risks. Doubly robust methods combine models and weighting under their own assumptions.

An effective sample size diagnostic, (sum_i w_i)²/sum_i w_i², can reveal weight concentration. It is not a guarantee against model misspecification or missing support. Confidence claims should describe both statistical uncertainty and identification assumptions.

### Explain why evaluation and policy selection interact

An evaluator that is approximately unbiased for one fixed candidate can still produce optimistic selected results when many candidates are ranked using the same noisy estimates. The chosen policy can be the one with the most favorable evaluation error.

Separate validation-based model selection from a final untouched test when possible. If only a fixed archive is available, describe how candidate count, coverage, and estimator uncertainty affect selection. A narrow confidence interval conditional on a selected dataset estimate need not account for all adaptive choices made during development.

<a id="section-33-8"></a>

## 33.8 Work a weight-concentration example

For weights [1,1,8], effective sample size is 10²/(1+1+64)=100/66, approximately 1.515. Three trajectories behave like far fewer equally weighted contributions under this diagnostic.

If all three weights are 1, it is 3. If unsupported target trajectories are absent, however, even a comfortable effective sample size on the observed subset cannot certify the missing part of the target policy's return.

<a id="section-33-9"></a>

## 33.9 Laboratory extension and preference-data bridge

The repository does not ship a dedicated offline RL algorithm. A proposed extension to [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) is to freeze datasets from several behavior policies, then compare behavior cloning and a documented offline method with an untouched final online evaluation budget.

Preference optimization also uses fixed datasets, but pairwise response learning is not automatically the same sequential offline-control problem. Both share coverage concerns; their objectives and evidence requirements must still be specified separately.

<a id="section-33-10"></a>

## 33.10 Problems, worked answers, and reading

1. In the two-model example, what return can always-left guarantee across both models?
2. Calculate effective sample size for weights [0,0,5].
3. Why is selecting hyperparameters using many deployment rollouts relevant to an “offline” claim?

<details><summary>Worked answers</summary>

1. Return 1. Choosing right can produce either 10 or −10 under indistinguishable logged evidence.
2. 25/25=1. Only one trajectory contributes to the weighted estimate.
3. Those rollouts supply additional environment information and affect the final selected policy. They belong in the interaction and evaluation budget even if the training dataset stays fixed.

</details>

Read the [Offline RL tutorial](https://arxiv.org/abs/2005.01643) and [Conservative Q-Learning](https://arxiv.org/abs/2006.04779), focusing on coverage assumptions and evaluation limits.

---

[← Chapter 32](../32-entropy-regularization/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 34 →](../34-model-based-reinforcement-learning/README.md)
