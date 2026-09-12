# 19. SARSA

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 18](../18-temporal-difference-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 20 →](../20-q-learning-and-exploration/README.md)

[Quick-read Topic 19](../../quick-read/19-sarsa.md) · [Notation](../NOTATION.md)

- [19.1 Evaluate the behavior that will actually continue](#section-19-1)
- [19.2 Define the update and its timing](#section-19-2)
- [19.3 Calculate an exploratory target](#section-19-3)
- [19.4 Expected SARSA removes action-sampling noise](#section-19-4)
- [19.5 Explain the cliff effect without a slogan](#section-19-5)
- [19.6 State the control assumptions](#section-19-6)
- [19.7 Track policies explicitly](#section-19-7)
- [19.8 Construct a controlled comparison](#section-19-8)
- [19.9 Laboratory and agent connection](#section-19-9)
- [19.10 Problems, worked answers, and reading](#section-19-10)

---

<a id="section-19-1"></a>

## 19.1 Evaluate the behavior that will actually continue

SARSA learns action values using the next action chosen by the current behavior policy. Its name records the transition tuple: state, action, reward, next state, next action.

**Prerequisites:** Chapters 8, 18, and epsilon-greedy policies. **Targets:** derive the on-policy target, explain exploration-sensitive behavior, and implement the next-action ordering correctly.

A courier planning near a dangerous road should account for its own occasional exploratory turns. SARSA's continuation value includes those turns rather than assuming perfect greedy behavior immediately after the current action.

<a id="section-19-2"></a>

## 19.2 Define the update and its timing

```text
choose A_t using the current behavior policy
execute A_t and observe reward and S_(t+1)
if nonterminal, choose A_(t+1) using that policy
target = reward + gamma * Q(S_(t+1), A_(t+1))
update Q(S_t,A_t) toward target
continue with the already chosen A_(t+1)
```

At termination the target is just reward. Resampling the next action after the update can make the action in the target differ from the one actually executed; specify the algorithm if deliberately doing this.

<a id="section-19-3"></a>

## 19.3 Calculate an exploratory target

At the next state, Q(s',safe)=4 and Q(s',risky)=−6. Suppose exploration selects risky. With current reward −1 and gamma=0.9, SARSA's sampled target is −6.4.

A greedy continuation target would instead be −1+0.9×4=2.6. The difference is not merely noise around the same conditional target: their expectations correspond to different continuation policies when exploration persists.

<a id="section-19-4"></a>

## 19.4 Expected SARSA removes action-sampling noise

Expected SARSA averages the next-action value under the behavior policy:

```text
y = r + gamma * sum_a pi(a|s') Q(s',a)
```

With epsilon=0.2 and two actions, uniform random exploration gives safe probability 0.9 and risky probability 0.1. The expected continuation is 3, so the target is 1.7. It still samples the environment transition, but integrates over the action choice at s'.

### Quantify the variance removed by Expected SARSA

At a fixed sampled successor, let Q values be [4,−6] and next-action probabilities [0.9,0.1]. The sampled next Q has mean 3 and variance 0.9(4−3)²+0.1(−6−3)²=9.

With gamma=0.9 and a fixed immediate reward, the SARSA target's conditional variance from next-action sampling is 0.9²×9=7.29. Expected SARSA replaces that sampled Q with its mean, removing this conditional action variance. Random successor states and rewards still contribute variance, so the whole target does not generally become deterministic.

<a id="section-19-5"></a>

## 19.5 Explain the cliff effect without a slogan

When exploratory actions can trigger a large penalty, an on-policy value includes that future risk. A longer route can have higher expected return under an epsilon-soft policy even if the shortest route is best under deterministic execution.

This does not make SARSA universally “safer.” Safety depends on the reward, constraints, exploration distribution, and state coverage. A severe outcome absent from training can still be missed, and expected return is not a hard risk constraint.

### Derive an exploration-sensitive route threshold

At a dangerous junction, the greedy action completes for +10, while an exploratory mistake ends for −100. If the probability of that mistake is q, expected terminal reward is 10−110q. A safe route yielding 5 is preferable under this execution policy when q>1/22, approximately 0.04545.

The deterministic best action still has return 10. The route comparison changes because the behavior policy includes mistakes. This example isolates the mechanism behind exploration-aware on-policy values without asserting that one algorithm satisfies a formal safety constraint.

<a id="section-19-6"></a>

## 19.6 State the control assumptions

On-policy tabular prediction for a fixed exploratory policy differs from control with a changing policy. Standard convergence-to-optimality results for SARSA use conditions such as greedy-in-the-limit with infinite exploration and appropriate step sizes.

A schedule that decays epsilon rapidly can stop visiting important actions, despite approaching zero. A fixed epsilon preserves exploration but generally retains exploratory behavior in the policy being evaluated. Report the schedule and distinguish training behavior from greedy test behavior.

<a id="section-19-7"></a>

## 19.7 Track policies explicitly

Store epsilon, tie-breaking rules, and whether the evaluation policy is greedy or exploratory. For epsilon-greedy with m legal actions, a unique greedy action has probability 1−epsilon+epsilon/m; other legal actions each have epsilon/m.

Action masks change m by state. Sampling uniformly over all actions and rejecting illegal choices afterward changes the actual policy unless handled explicitly. That matters for both expected targets and logged probabilities.

<a id="section-19-8"></a>

## 19.8 Construct a controlled comparison

Use the same transition budget, initialization, seeds, and environment for SARSA and Q-learning. Evaluate each learned table under both its exploratory policy and its greedy policy.

If differences disappear under greedy evaluation, explain that the training continuation assumptions still differed. Also report catastrophic events during training, not only final reward. An algorithm that learns a good final policy after many failures may be unsuitable under a constrained data-collection budget.

### Evaluate training and deployment policies separately

Save the learned Q table and run two frozen evaluations: one with the training epsilon and one greedily. Keep the environment distribution identical. Record severe failures as well as mean return in both evaluations.

A policy can show a substantial gap between these scores even when its value estimates correctly describe exploratory continuation. If deployment uses a different temperature, action filter, or retry wrapper, evaluate that complete behavior too. The phrase “the learned policy” is insufficient unless the action-selection rule is specified.

<a id="section-19-9"></a>

## 19.9 Laboratory and agent connection

[Notebook 03](../../notebooks/03_q_learning.ipynb) introduces tabular action-value control; [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) includes the shared comparison with SARSA.

For tool agents, the executed policy includes sampling temperature, retries, and action filters. Estimating continuation under a different execution wrapper can misrepresent operational performance just as ignoring exploration changes a tabular continuation value.

<a id="section-19-10"></a>

## 19.10 Problems, worked answers, and reading

1. For epsilon=0.3 and three legal actions, give the probabilities with a unique greedy action.
2. With Q next=[5,−5], epsilon=0.4, reward 0, and gamma=0.9, calculate Expected SARSA's target.
3. Why is epsilon→0 alone insufficient for the standard convergence story?

<details><summary>Worked answers</summary>

1. Greedy probability 0.8; each other action 0.1.
2. Probabilities are [0.8,0.2], so expected next value is 3 and the target is 2.7.
3. Exploration can vanish before every relevant state-action pair receives enough updates. Infinite visitation and compatible step sizes are separate requirements.

</details>

Read [Sutton and Barto, Chapter 6.4](http://incompleteideas.net/book/the-book-2nd.html), including how persistent exploration changes the learned behavior.

---

[← Chapter 18](../18-temporal-difference-learning/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 20 →](../20-q-learning-and-exploration/README.md)
