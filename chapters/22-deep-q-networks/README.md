# 22. Deep Q-Networks

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 21](../21-function-approximation/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 23 →](../23-experience-replay/README.md)

[Quick-read Topic 22](../../quick-read/22-deep-q-networks.md) · [Notation](../NOTATION.md)

- [22.1 Approximate a value for every action](#section-22-1)
- [22.2 State the objective and data source](#section-22-2)
- [22.3 Trace a two-transition batch](#section-22-3)
- [22.4 Keep tensor shapes visible](#section-22-4)
- [22.5 Understand the Huber loss choice](#section-22-5)
- [22.6 Separate selection from evaluation in Double DQN](#section-22-6)
- [22.7 Recognize the moving pieces](#section-22-7)
- [22.8 Diagnose performance beyond loss](#section-22-8)
- [22.9 Laboratory and limits](#section-22-9)
- [22.10 Problems, worked answers, and reading](#section-22-10)

---

<a id="section-22-1"></a>

## 22.1 Approximate a value for every action

A Deep Q-Network (DQN) uses a neural network Q_theta(s,a) with a Q-learning target. For discrete actions, one forward pass often returns one value per action. The core challenge is learning from targets that depend on other learned predictions.

**Prerequisites:** Chapters 20–21. **Targets:** trace a minibatch update, distinguish the original DQN target from Double DQN, and diagnose unstable value learning.

The network is a shared route estimator. Experience replay is its archive of journeys, and a target network is a slower-changing reference used when preparing training targets.

<a id="section-22-2"></a>

## 22.2 State the objective and data source

Sample transitions (s,a,r,s',terminated) from replay. The original target is y=r+gamma(1−terminated)max_b Q_target(s',b). Minimize a regression loss between y and Q_online(s,a), with no gradient through y.

Behavior commonly uses epsilon-greedy actions from Q_online. The replay distribution, behavior policy, and greedy target policy are therefore distinct objects. The loss is a sampled fitted update, not an exact application of a tabular Bellman operator.

### View a frozen-target phase as fitted regression

During a phase with fixed target parameters and a fixed replay dataset, DQN fits predictions to a set of bootstrap labels. Improving that supervised fit can be meaningful, but the labels still depend on the target network's errors and the replay support.

After target synchronization, the labels change. Therefore a sequence of decreasing within-phase losses need not represent descent of one fixed global loss. Plotting loss across target-copy events can reveal abrupt label shifts that a single smoothed curve obscures. Keep optimizer progress and Bellman consistency conceptually separate.

<a id="section-22-3"></a>

## 22.3 Trace a two-transition batch

Suppose selected predictions are [1,2], rewards [0,3], maximum next target values [4,10], termination flags [0,1], and gamma=0.9. Targets are [3.6,3], errors prediction-minus-target are [−2.6,−1], and mean squared error is 3.88.

The value 10 must not affect the second target. Verify this with an explicit test transition before launching training. If changing a terminated row's next observation changes its target, boundary handling is wrong.

<a id="section-22-4"></a>

## 22.4 Keep tensor shapes visible

For B samples and A actions, network output has shape [B,A]. Gather the selected actions to [B], and compute targets of shape [B]. Assert exact shape equality before loss calculation.

Store action indices as integers and reward/value tensors in compatible floating types. Do not flatten the action dimension before selecting actions. An accidental broadcast can optimize a B×B matrix of unrelated prediction-target pairs while still yielding a decreasing scalar loss.

<a id="section-22-5"></a>

## 22.5 Understand the Huber loss choice

With threshold 1, Huber loss is 0.5e² for |e|≤1 and |e|−0.5 otherwise. It limits the growth of the error derivative for large residuals. For errors [−2.6,−1], the mean Huber loss is (2.1+0.5)/2=1.3.

This can reduce sensitivity to outliers but does not correct an invalid reward, missing action mask, or unstable bootstrap loop. Reward clipping also changes the effective objective and must be distinguished from a robust regression loss.

### Compare the gradients of squared and Huber losses

For prediction error e=q−y, half-squared loss has derivative e. Unit-threshold Huber has derivative e for |e|≤1 and sign(e) outside that interval. At e=10, the derivatives are 10 and 1, so Huber reduces one large residual's direct influence on the selected prediction.

The full parameter gradient multiplies this derivative by gradient Q_theta(s,a). A bounded scalar error derivative does not bound that network Jacobian or the total gradient norm. Huber loss and gradient-norm clipping address different mechanisms and should not be treated as interchangeable stabilization devices.

<a id="section-22-6"></a>

## 22.6 Separate selection from evaluation in Double DQN

Double DQN selects a'=argmax_b Q_online(s',b), then evaluates Q_target(s',a'). Original DQN both selects and evaluates using target values.

If online next values are [5,4] and target values are [2,6], Double DQN uses 2 while original DQN uses 6. The methods deliberately construct different targets. This separation reduces a source of maximization bias, but correlated networks and approximation error remain.

<a id="section-22-7"></a>

## 22.7 Recognize the moving pieces

Replay capacity determines the age and diversity of data. The update-to-data ratio determines how often experiences are reused. Target synchronization controls target movement. Exploration controls future coverage. These mechanisms interact.

Tune and report them as experimental factors. A comparison with equal environment steps but radically different gradient counts measures interaction efficiency under unequal compute, which may be useful if stated explicitly.

<a id="section-22-8"></a>

## 22.8 Diagnose performance beyond loss

Track independent evaluation return, predicted Q scale, TD residuals, replay age, and action frequencies. A low replay loss can reflect overfitting a narrow buffer or mutually consistent incorrect estimates.

Use multiple seeds and retain failed runs. Check whether the learned greedy policy actually reaches the goal. In a known small model, evaluate its policy exactly and compare predicted versus true action values at important states.

### Design a diagnostic probe set outside the replay batch

Choose fixed states representing initial decisions, near-terminal choices, and low-coverage branches. At each checkpoint, record online Q, target Q, greedy action, and actual return of that frozen action policy under an independent evaluator.

If predicted values rise while measured returns stay flat, inspect unsupported actions, terminal masking, and reward scale. If values remain accurate but actions oscillate, examine small action gaps and update noise. The point is to link a diagnostic to a failure mechanism; collecting many metrics without a hypothesis does not explain the learning process.

<a id="section-22-9"></a>

## 22.9 Laboratory and limits

[Notebook 04](../../notebooks/04_dqn.ipynb) implements DQN. [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) compares it with tabular and policy methods on one task. First verify the batch arithmetic, then compare target-update frequencies while holding other factors fixed.

DQN's explicit action maximum fits modest discrete action sets. A language model's enormous sequence action space makes direct sequence-level maximization difficult; token-level decompositions introduce a long-horizon problem rather than eliminating it.

<a id="section-22-10"></a>

## 22.10 Problems, worked answers, and reading

1. Calculate the original and Double DQN targets from the example with reward 1 and gamma=0.9.
2. Why must the target network receive no gradient from this loss?
3. Does a falling replay loss prove improving task performance?

<details><summary>Worked answers</summary>

1. Original: 1+0.9×6=6.4. Double: 1+0.9×2=2.8.
2. It is the fixed regression reference for that update. Updating it through the same loss changes the intended semi-gradient algorithm and removes that separation.
3. No. Replay coverage, target validity, and policy ranking can all be poor despite low fitted error. Evaluate actual behavior independently.

</details>

Read the original [DQN paper](https://doi.org/10.1038/nature14236) and [Double DQN](https://arxiv.org/abs/1509.06461). Distinguish algorithm definitions from implementation conveniences.

---

[← Chapter 21](../21-function-approximation/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 23 →](../23-experience-replay/README.md)
