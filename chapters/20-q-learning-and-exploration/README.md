# 20. Q-Learning and Exploration

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 19](../19-sarsa/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 21 →](../21-function-approximation/README.md)

[Quick-read Topic 20](../../README.md#topic-20) · [Notation](../NOTATION.md)

- [20.1 Separate the data policy from greedy continuation](#section-20-1)
- [20.2 Define the target](#section-20-2)
- [20.3 Calculate one update](#section-20-3)
- [20.4 Explain why off-policy does not mean assumption-free](#section-20-4)
- [20.5 Design exploration as a data strategy](#section-20-5)
- [20.6 Derive maximization bias](#section-20-6)
- [20.7 Distinguish GLIE from a convenient schedule](#section-20-7)
- [20.8 Audit a Q table before a learning curve](#section-20-8)
- [20.9 Laboratory and modern connection](#section-20-9)
- [20.10 Problems, worked answers, and reading](#section-20-10)

---

<p align="center">
  <img src="../../assets/diagrams/q-learning.svg" width="760" alt="Q-learning evaluates greedy continuation while data collection may explore." />
</p>

*Q-learning evaluates greedy continuation while data collection may explore.*

<a id="section-20-1"></a>

## 20.1 Separate the data policy from greedy continuation

Q-learning updates an action value toward reward plus the best estimated next-action value. The action used to collect data can be exploratory even though the target uses greedy continuation.

**Prerequisites:** Chapters 11, 18–19. **Targets:** derive the update, state convergence assumptions, analyze maximization bias, and design an exploration comparison.

The courier can explore a side road today while estimating what a future greedy route planner would achieve. This separation enables off-policy learning, but does not supply evidence about roads never visited.

<a id="section-20-2"></a>

## 20.2 Define the target

```text
y = r + gamma * (1−terminated) * max_(legal a') Q(s',a')
Q(s,a) += alpha * [y − Q(s,a)]
```

Use the final observation at collection cutoffs. A fully terminal next state needs no maximum; masking an invalid or empty action set after computing a NaN can still contaminate the update. Handle terminal rows directly.

<a id="section-20-3"></a>

## 20.3 Calculate one update

Let Q(s,a)=1, r=2, gamma=0.9, and next-action values [3,5]. The target is 6.5. With alpha=0.2, the updated value is 2.1.

If the transition terminates, the target is 2 and the updated value is 1.2. If the action with value 5 is illegal, use 3 instead: target 4.7 and updated value 1.74. Legal-action masking changes the decision problem's Bellman operator.

<a id="section-20-4"></a>

## 20.4 Explain why off-policy does not mean assumption-free

Finite tabular convergence to Q_star uses sufficient visitation of all relevant state-action pairs, suitable step sizes, bounded rewards, and discounted or appropriate episodic conditions. The behavior policy affects whether useful data arrive.

The theorem does not automatically extend to a deep network trained on a small fixed dataset. Function approximation couples states; unsupported action values can enter the maximum; and distribution mismatch can amplify errors.

<a id="section-20-5"></a>

## 20.5 Design exploration as a data strategy

Epsilon-greedy is easy to interpret but ignores uncertainty magnitude. Optimistic initialization can encourage visits until estimates decrease. Count-based bonuses reward novelty in a tabular setting; neural uncertainty estimates require additional modeling assumptions.

Distinguish an exploration bonus used for training from the task reward used for evaluation. A courier that visits many new streets may score well on novelty while completing few deliveries. Always report the original objective separately.

<a id="section-20-6"></a>

## 20.6 Derive maximization bias

Suppose two true action values are zero and independent estimates are each +1 or −1 with equal probability. The maximum is +1 in three of four cases and −1 in one, so its expectation is 0.5.

Each estimate is individually unbiased, but selecting the larger produces an upward bias. This can feed into bootstrapped targets. Double Q-learning separates selection and evaluation across estimators to reduce this coupling; it is not a universal guarantee of zero error.

<a id="section-20-7"></a>

## 20.7 Distinguish GLIE from a convenient schedule

Greedy in the limit with infinite exploration is a property of the induced visitation process, not merely a label for decaying epsilon. Even epsilon_t=1/t does not by itself guarantee every deep state is visited infinitely often in an arbitrary environment.

Long action sequences may need coordinated exploration. A reward at the end of ten specific choices is unlikely under independent random actions. This motivates structured exploration and model-based planning, while preserving the need to measure actual coverage.

<a id="section-20-8"></a>

## 20.8 Audit a Q table before a learning curve

Log visitation counts, TD-error distribution, value range, and terminal targets. With |r|≤R_max and gamma<1, true values lie within ±R_max/(1−gamma); wildly larger estimates suggest instability or a boundary error, though transient estimates can exceed the bound.

Check a one-state self-loop analytically. Then compare greedy policy performance with value estimates: a high predicted maximum without high measured return is evidence to investigate, not proof of successful learning.

<a id="section-20-9"></a>

## 20.9 Laboratory and modern connection

[Notebook 03](../../notebooks/03_q_learning.ipynb) is the direct implementation. [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) provides the same-environment comparison. Change only the exploration schedule first, and compare both coverage and final return across seeds.

Selecting the highest-scoring generated answer under a noisy reward model has a related selection effect: more candidates can increase the selected score without proportionate true quality. Chapter 37 examines that failure with learned preferences.

<a id="section-20-10"></a>

## 20.10 Problems, worked answers, and reading

1. Repeat the numerical update with gamma=0.5 and alpha=0.1.
2. If the two ±1 estimates are perfectly identical rather than independent, what is the expected maximum?
3. Why should an exploration bonus be excluded from the headline task score?

<details><summary>Worked answers</summary>

1. Target 2+0.5×5=4.5; new value 1+0.1×3.5=1.35.
2. It is the shared estimate, whose expectation is zero. The joint error structure matters, not only marginal unbiasedness.
3. It changes the optimization objective and can reward behavior unrelated to task completion. Report it as a training mechanism alongside independently measured task performance.

</details>

Read [Sutton and Barto, Chapter 6.5](http://incompleteideas.net/book/the-book-2nd.html) and the original [Double Q-learning paper](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

---

[← Chapter 19](../19-sarsa/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 21 →](../21-function-approximation/README.md)
