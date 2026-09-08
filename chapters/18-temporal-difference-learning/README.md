# 18. Temporal-Difference Learning

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 17](../17-monte-carlo-methods/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 19 →](../19-sarsa/README.md)

[Quick-read Topic 18](../../README.md#topic-18) · [Notation](../NOTATION.md)

- [18.1 Learn before an episode ends](#section-18-1)
- [18.2 Define the TD error](#section-18-2)
- [18.3 Work a numerical update](#section-18-3)
- [18.4 Relate the expected update to Bellman evaluation](#section-18-4)
- [18.5 State convergence assumptions carefully](#section-18-5)
- [18.6 Compare n-step targets](#section-18-6)
- [18.7 Distinguish target bias from objective bias](#section-18-7)
- [18.8 Engineer a trustworthy update](#section-18-8)
- [18.9 Laboratory and modern connection](#section-18-9)
- [18.10 Problems, worked answers, and reading](#section-18-10)

---

<p align="center">
  <img src="../../assets/diagrams/td-learning.svg" width="760" alt="A sampled transition supports an update before the episode finishes." />
</p>

*A sampled transition supports an update before the episode finishes.*

<a id="section-18-1"></a>

## 18.1 Learn before an episode ends

Temporal-difference learning updates a value using a reward plus another value estimate. It combines sampling, which avoids a full model, with bootstrapping, which avoids waiting for the complete return.

**Prerequisites:** Chapters 7, 10, and 17. **Targets:** derive TD(0), calculate boundary-sensitive updates, and distinguish target bias from fixed-point correctness.

The courier revises the estimate for the current junction after reaching the next one, using the next junction's estimated remaining cost. The estimate can improve before the final delivery receipt arrives.

<a id="section-18-2"></a>

## 18.2 Define the TD error

```text
y_t = R_(t+1) + gamma * bootstrap_mask * V(S_(t+1))
delta_t = y_t − V(S_t)
V(S_t) += alpha * delta_t
```

The bootstrap mask is zero at true task termination. At an external collection cutoff it is usually one, using the final observation rather than an automatic reset observation. State representation must include any deadline that is part of the task itself.

<a id="section-18-3"></a>

## 18.3 Work a numerical update

Let V(A)=2, V(B)=4, reward=−1, gamma=0.9, and alpha=0.1. For a nonterminal A→B transition, the target is 2.6, delta is 0.6, and the updated V(A)=2.06.

If the same observed reward ends the task, the target is −1 and the updated estimate is 1.7. Carrying V(B) into a terminated transition would invent a future that the objective does not contain.

<a id="section-18-4"></a>

## 18.4 Relate the expected update to Bellman evaluation

Under a fixed policy and the correct conditional transition distribution, E[y_t|S_t=s]=(T_pi V)(s). Thus the expected TD error is the Bellman residual at that state.

At V=V_pi, the expected TD error is zero even though individual errors need not vanish. Stochastic rewards and transitions continue to produce sample variation. A loss that does not approach zero can therefore be compatible with correct expected values.

<a id="section-18-5"></a>

## 18.5 State convergence assumptions carefully

For finite tabular on-policy prediction, standard convergence results require suitable repeated visitation, boundedness conditions, gamma<1 or appropriate episodic assumptions, and per-state step sizes with sum alpha infinite and sum alpha² finite.

A constant step size generally leaves ongoing fluctuations; it can be useful for tracking but is not the same asymptotic claim. Nonlinear approximation, off-policy sampling, and bootstrapping together require additional analysis and can become unstable.

<a id="section-18-6"></a>

## 18.6 Compare n-step targets

An n-step target accumulates n rewards, then bootstraps if the task has not ended:

```text
G_t^(n) = sum_(k=0 to n−1) gamma^k R_(t+k+1)
          + gamma^n V(S_(t+n))
```

Larger n reduces reliance on an early continuation estimate but exposes the target to more random outcomes. At termination, shorten the sum and remove continuation. The optimal choice depends on reward noise, horizon, value accuracy, and data availability; there is no universal ordering of MC and TD.

<a id="section-18-7"></a>

## 18.7 Distinguish target bias from objective bias

If the current V is inaccurate, the conditional expected one-step target differs from V_pi. This is bootstrap target bias during learning. In tabular on-policy prediction, the correct fixed point can nevertheless remain V_pi.

With restricted function approximation, the eventual projected solution may differ from V_pi even with unlimited data. A claim that TD is “biased” should say whether it concerns a transient target, an approximation fixed point, or a finite-sample estimator.

<a id="section-18-8"></a>

## 18.8 Engineer a trustworthy update

Do not accidentally backpropagate through the target when implementing a semi-gradient critic. Log target scale, value scale, and terminal/cutoff counts. For batched code, verify predictions and targets have the same intended shape before subtraction.

A batch of B predictions shaped [B,1] and targets shaped [B] can broadcast into [B,B], producing a plausible scalar loss for the wrong task. One manual transition is a more revealing first check than a long training curve.

<a id="section-18-9"></a>

## 18.9 Laboratory and modern connection

In [Notebook 02](../../notebooks/02_mc_vs_td.ipynb), compare sample efficiency under fixed-policy prediction. Plot error versus environment transitions and versus completed episodes; the choice of horizontal axis changes what early updates mean.

Critics in actor–critic and PPO use related bootstrap targets. A language-model value head at a response prefix estimates future return under a particular generation policy; changing that policy changes the quantity being predicted.

<a id="section-18-10"></a>

## 18.10 Problems, worked answers, and reading

1. With rewards [1,2], gamma=0.5, and continuation value 8 after two nonterminal steps, calculate the two-step target.
2. Why can delta be nonzero when the value function is exactly correct?
3. Does alpha_t=1/t satisfy both step-size sum conditions? Does constant alpha=0.1?

<details><summary>Worked answers</summary>

1. 1+0.5×2+0.25×8=4. If the second transition terminates, the target is 2.
2. V_pi is an expectation. Individual rewards and successor states fluctuate around the expected Bellman backup.
3. The harmonic sum diverges and the squared harmonic sum converges, so 1/t meets the scalar conditions. Constant 0.1 has a divergent squared sum. Visitation and other assumptions still matter.

</details>

Read [Sutton and Barto, Chapters 6–7](http://incompleteideas.net/book/the-book-2nd.html) and connect the sample target to the operator in Chapter 10.

---

[← Chapter 17](../17-monte-carlo-methods/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 19 →](../19-sarsa/README.md)
