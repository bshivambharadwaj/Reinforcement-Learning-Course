# 7. State-Value Functions

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 6](../06-returns-and-discounting/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 8 →](../08-action-value-functions/README.md)

[Quick-read Topic 7](../../quick-read/07-state-value-functions.md) · [Notation](../NOTATION.md)

- [7.1 A value answers a conditional question](#section-7-1)
- [7.2 Define the expectation and its conditioning](#section-7-2)
- [7.3 From the return identity to a value recursion](#section-7-3)
- [7.4 A state with repeated chances to finish](#section-7-4)
- [7.5 What state values hide about individual actions](#section-7-5)
- [7.6 Estimate values from experience and account for coverage](#section-7-6)
- [7.7 Sensitivity and a useful error bound](#section-7-7)
- [7.8 Lab: compare against an independent oracle](#section-7-8)
- [7.9 A critic is a value estimator with a job](#section-7-9)
- [7.10 Problems, worked answers, and sources](#section-7-10)

---

<a id="section-7-1"></a>

## 7.1 A value answers a conditional question

V_pi(s) is the expected return after starting in state s and following pi. It does not mean the highest reward obtainable from s, the value of one selected action, or the return of the most successful observed episode.

**Prerequisites:** Chapters 3, 5, and 6. **Targets:** derive conditional state value, distinguish the start-state objective from the whole value vector, solve a recurrent value equation, and understand how state coverage affects estimation.

For a delivery service, value resembles the expected remaining net outcome given the courier's current situation and routing policy. A good location can have poor value under a policy that repeatedly makes bad decisions. A mediocre immediate reward can lead to a high-value future.

<a id="section-7-2"></a>

## 7.2 Define the expectation and its conditioning

The definition V_pi(s)=E_pi[G_t | S_t=s] averages over future policy actions and environmental outcomes. The reward convention and gamma must be specified. In a finite-horizon task use V_t or include time in s; otherwise the same physical location can have multiple remaining values.

For an initial-state distribution rho, the scalar objective is J(pi)=sum_s rho(s)V_pi(s). This objective can ignore unreachable or zero-probability starting states even though the value function assigns numbers to them.

State value is therefore policy dependent and distributionally defined. Estimating the best observed return at each state does not estimate this expectation. Neither does averaging returns from several different policies without stating which mixture is intended.

<a id="section-7-3"></a>

## 7.3 From the return identity to a value recursion

Start with G_t=R_(t+1)+gamma G_(t+1). Condition on S_t=s and use the law of total expectation over the next action and outcome. Under the Markov and policy assumptions, the conditional expected future return at s' is V_pi(s').

```text
V_pi(s) = sum_a pi(a|s) sum_(s',r) p(s',r|s,a)
          [r + gamma V_pi(s')]
```

This equation is a consequence of the definition, not a separate reward signal. The outer average comes from policy uncertainty; the inner average comes from environment uncertainty. In a deterministic policy or environment the corresponding sum collapses, but the remaining uncertainty still matters.

An absorbing zero-reward terminal state has V=0. The reward for entering it belongs to the preceding transition and is not removed by setting its future value to zero.

### Separate aleatoric spread from value-estimation error

Suppose a terminal transition pays either 0 or 10 with equal probability. The exact state value is 5, but its one-sample return has variance 25. A value prediction of 5 is correct despite an error of magnitude 5 on every individual observed outcome.

For n independent visits under the same fixed policy, the sample-mean standard error is 5/sqrt(n). At n=100 it is 0.5. This calculation concerns uncertainty in the estimated mean under independence, not the spread of the next actual outcome. An evaluator that expects prediction residuals to vanish would confuse these two quantities.

<a id="section-7-4"></a>

## 7.4 A state with repeated chances to finish

Suppose a robot in state B finishes with reward +4 with probability 0.5 and otherwise stays in B with reward −1. Let gamma=0.9 and assume this behavior is fixed by pi. Then V(B)=0.5×4+0.5×(−1+0.9V(B)). Solving gives V(B)=30/11≈2.7273.

This exceeds the expected immediate reward 1.5 because a failed attempt still permits future attempts. It is less than the terminal success reward 4 because failed attempts incur costs and delay success.

A one-step estimate using V(B)=0 initially would produce 1.5. That is a Bellman backup from a particular initial guess, not the solved value. Repeated backups incorporate increasingly distant consequences.

<a id="section-7-5"></a>

## 7.5 What state values hide about individual actions

Suppose Q_pi(s,left)=8 and Q_pi(s,right)=2, while pi chooses each equally. Then V_pi(s)=5. The state's average does not reveal whether the action just taken was good. The action-conditioned value and advantage resolve that question.

Changing pi at s changes this average, and changing pi at later states can change both action values too. Therefore it is generally incorrect to reweight old Q_pi values under a new policy and call the result the fully evaluated V of that new policy.

The reweighting can be useful for a one-step comparison with the old continuation policy. Chapter 14 explains how repeated improvement arguments turn such local comparisons into policy-improvement results under exact-value assumptions.

<a id="section-7-6"></a>

## 7.6 Estimate values from experience and account for coverage

If episodes are generated under a fixed policy and returns from visits to s are averaged appropriately, Monte Carlo prediction estimates V_pi(s). TD uses a bootstrapped target instead. Both require relevant experience or a justified generalization mechanism.

States seen rarely have uncertain estimates. A low average error across frequently visited states can hide a severe mistake at a rare but important state. Report the weighting distribution used in a value-error metric: equal weighting over all states, start-state weighting, and occupancy weighting answer different questions.

For partially observed data, averaging by observation estimates a history-mixture quantity that may change as behavior changes. It need not be a Markov state value suitable for a simple Bellman recursion.

<a id="section-7-7"></a>

## 7.7 Sensitivity and a useful error bound

With a fixed transition matrix and policy, suppose two expected reward vectors differ by at most epsilon in every state. Their discounted values differ by at most epsilon/(1−gamma) in maximum norm. Each future reward can differ by epsilon, and the geometric discounted sum bounds the accumulated difference.

This argument does not cover arbitrary transition-model errors without additional terms. It also shows why values can be sensitive to small persistent reward misspecification when gamma is near one.

> **Failure Mode:** value estimates may be accurately fitted to a biased reward process. A numerical value solver can diagnose internal consistency but cannot establish that the reward represents the intended task.

### Extend sensitivity from reward errors to transition errors

Let v and v_hat evaluate the same policy under models (r,P) and (r_hat,P_hat), with the same discount. Subtract their Bellman equations and write v−v_hat=(r−r_hat)+gamma P(v−v_hat)+gamma(P−P_hat)v_hat.

If reward error is bounded by epsilon_r and the maximum row L1 difference of transition matrices is epsilon_P, then ||v−v_hat||∞≤[epsilon_r+gamma epsilon_P||v_hat||∞]/(1−gamma). With |r_hat|≤R_max, substitute ||v_hat||∞≤R_max/(1−gamma). Transition error can therefore acquire two inverse-discount factors in this crude bound.

The row L1 convention matters: if a source defines total variation as half the L1 distance, its numerical coefficient differs. State the norm before comparing bounds across papers.

<a id="section-7-8"></a>

## 7.8 Lab: compare against an independent oracle

[Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) computes random-policy values using iterative backups and a linear solve. Identify the same policy, dynamics, reward convention, and discount in both calculations. Agreement between them is valuable only if their common task specification is correct.

In [Notebook 02](../../notebooks/02_mc_vs_td.ipynb), compare learned random-walk values with the analytical line. Report errors per state as well as their average. Change the start distribution and inspect which states become harder to estimate without changing the underlying fixed policy's value definition.

An extension is to bootstrap confidence intervals over independent runs rather than pretend every correlated transition is an independent value sample.

### Build a diagnostic value table

For each state in a small model, record exact value, learned value, visitation count, and start-state probability. Sort once by absolute error and once by contribution to the start-state objective. These orderings need not agree.

A rarely visited state can have large estimation error but little current objective weight. It may become important after policy improvement changes visitation. This is why a single average value loss can hide future control problems: it summarizes errors under a chosen distribution rather than certifying every possible decision state.

<a id="section-7-9"></a>

## 7.9 A critic is a value estimator with a job

In actor–critic methods, a value model predicts expected future return so the actor can compare outcomes with a baseline. It is not a classifier that simply labels states “good” or “bad.” Its prediction depends on the current policy and chosen reward objective.

In LM PPO, the critic may predict future regularized return from a generated prefix. Changing the reward model, KL coefficient, or policy changes what that prediction should represent. A critic trained against an old target distribution can become stale even when its architecture is unchanged.

This explains the systems importance of versioning reward and policy components. A value number detached from its policy and reward definition is incomplete information.

<a id="section-7-10"></a>

## 7.10 Problems, worked answers, and sources

**A.** A state self-loops forever with reward +2 and gamma=0.8. Find its value. **B.** Q values are 8 and 2, and the policy probabilities are 0.25 and 0.75. Find V. **C.** With gamma=0.95 and a maximum reward-vector error of 0.02, give the fixed-dynamics value-error bound.

<details><summary>Worked answers</summary>

A: V=2+0.8V, so V=10. B: V=0.25×8+0.75×2=3.5. C: 0.02/(1−0.95)=0.4. The bound is worst case and assumes the policy and transition dynamics are unchanged. If Q values were evaluated under a different continuation policy, B would be only a one-step reweighting, not necessarily the fully evaluated new policy value.

</details>

**Read:** [Sutton and Barto, Chapters 3–4](http://incompleteideas.net/book/the-book-2nd.html). Chapter 8 refines this state average into action-conditioned values.

---

[← Chapter 6](../06-returns-and-discounting/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 8 →](../08-action-value-functions/README.md)
