# 17. Monte Carlo Methods

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 16](../16-value-iteration/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 18 →](../18-temporal-difference-learning/README.md)

[Quick-read Topic 17](../../quick-read/17-monte-carlo-methods.md) · [Notation](../NOTATION.md)

- [17.1 Learn from complete experience](#section-17-1)
- [17.2 Define the prediction estimator](#section-17-2)
- [17.3 Calculate first-visit and every-visit targets](#section-17-3)
- [17.4 Separate prediction from control](#section-17-4)
- [17.5 Understand the bias–variance choice](#section-17-5)
- [17.6 Off-policy importance sampling](#section-17-6)
- [17.7 See how horizon amplifies weights](#section-17-7)
- [17.8 Handle incomplete episodes honestly](#section-17-8)
- [17.9 Laboratory investigation](#section-17-9)
- [17.10 Problems, worked answers, and reading](#section-17-10)

---

<a id="section-17-1"></a>

## 17.1 Learn from complete experience

Monte Carlo (MC) prediction estimates values by averaging observed returns after complete episodes. It needs no transition model and does not substitute an estimated continuation value into its target.

**Prerequisites:** Chapters 5–7 and 13. **Targets:** distinguish first-visit and every-visit estimators, analyze their variability, and separate prediction from control.

A courier reviews complete delivery receipts. Each receipt reveals the actual outcome of one route, but one unusually lucky delivery is weak evidence about its expected return.

<a id="section-17-2"></a>

## 17.2 Define the prediction estimator

For a fixed policy, collect episodes and compute G_t=R_(t+1)+gamma R_(t+2)+… up to termination. For each occurrence selected for state s, update its sample average:

```text
count(s) += 1
V(s) += [G_t − V(s)] / count(s)
```

First-visit MC selects the first occurrence of s within each episode. Every-visit MC selects all occurrences. Returns from the same episode can be correlated; treating them as independent observations understates uncertainty.

### Derive the incremental sample average

After n−1 returns, let the estimate be V_(n−1)=(G_1+…+G_(n−1))/(n−1). Adding G_n gives V_n=[(n−1)V_(n−1)+G_n]/n, which rearranges to V_n=V_(n−1)+(G_n−V_(n−1))/n.

This identity explains the 1/n step size without invoking an optimizer. A constant alpha instead assigns geometrically decaying weights to older returns. It can track nonstationarity, but is no longer the equal-weight sample mean. Store counts per state or state-action pair; a global timestep denominator gives different weighting when visitation is uneven.

<a id="section-17-3"></a>

## 17.3 Calculate first-visit and every-visit targets

An episode visits A, B, A, terminal, with rewards [0,1,2] and gamma=1. The returns are [3,3,2]. For A, first-visit MC uses 3; every-visit MC uses both 3 and 2, giving an initial average of 2.5.

This discrepancy is not an implementation error: the estimators use different samples. Under suitable episodic sampling assumptions both can be consistent, but finite-sample behavior differs. The first-visit return is an unbiased draw from V_pi(A) when the policy and environment remain fixed and the Markov setup applies.

<a id="section-17-4"></a>

## 17.4 Separate prediction from control

Prediction holds pi fixed. MC control estimates Q_pi(s,a), improves the policy, and continues gathering data. Exploration is essential because an untried action cannot obtain a reliable value estimate merely through averaging other actions.

Exploring starts are a useful theoretical device but may not be possible in a physical system. Epsilon-greedy behavior is practical, though a fixed positive epsilon targets an exploratory policy rather than a fully greedy one. Changing policies also means the data are no longer identically distributed under a single fixed pi.

<a id="section-17-5"></a>

## 17.5 Understand the bias–variance choice

MC avoids bootstrap error, but complete returns can be noisy over long horizons. A constant step size tracks a changing policy and forgets old outcomes; a sample average treats all historical returns equally. These are different estimation choices.

For independent return samples with variance sigma², an average of n has variance sigma²/n. Correlated visits, nonstationary policies, and adaptive sampling complicate that calculation. Count independent episodes and seeds separately from the number of updates.

<a id="section-17-6"></a>

## 17.6 Off-policy importance sampling

To estimate a target policy from behavior-policy episodes, a trajectory suffix can receive weight W=product_k pi(A_k|S_k)/b(A_k|S_k). This requires behavior support wherever the target assigns positive probability.

Ordinary importance sampling averages W×G and can have high variance. Self-normalized weighting divides the weighted-return sum by the weight sum; it generally introduces finite-sample bias while often reducing variability. Ratios must refer to the policy that actually generated each action.

### Show why normalized importance sampling is biased at small n

Consider a one-step target distribution [0.5,0.5], behavior [0.9,0.1], and rewards [0,10]. The target value is 5. With one sampled episode, self-normalized importance sampling divides its weighted reward by its own weight, leaving the observed reward unchanged.

Its expectation is therefore the behavior value 1, not 5. Ordinary importance sampling uses weight 5 on the rewarding action and 5/9 on the other; its expected weighted return is 0.1×5×10=5. The ordinary estimator is unbiased here but highly variable. This concrete case makes the finite-sample bias/variance tradeoff visible rather than treating normalization as a free correction.

<a id="section-17-7"></a>

## 17.7 See how horizon amplifies weights

If the target chooses an observed action with probability 0.8 and behavior with 0.4 at each of five decisions, the trajectory ratio is 2⁵=32. Rare matching trajectories can dominate an estimate.

This issue reappears in sequence models: multiplying token probability ratios over a long response can produce extreme sequence weights. Taking logs improves arithmetic stability but does not remove the statistical variance problem. Clipping weights trades bias against variability and must be reported.

<a id="section-17-8"></a>

## 17.8 Handle incomplete episodes honestly

If collection stops before task termination, the partial return is not a full MC target for the original continuing task. You can wait for completion, redefine a finite-horizon objective explicitly, or bootstrap at the cutoff; the last option is no longer pure MC.

For very long or nonterminating tasks, this is a practical limitation rather than a small coding inconvenience. Also distinguish a true terminal transition from an external time limit, as introduced in Chapter 2.

<a id="section-17-9"></a>

## 17.9 Laboratory investigation

[Notebook 02](../../notebooks/02_mc_vs_td.ipynb) compares MC and TD prediction. Keep the evaluation policy fixed, compare error against a known reference, and repeat over independent seeds. Add a longer-horizon variant to test the prediction that full-return variability becomes more problematic.

[Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) distinguishes fixed-policy prediction from control. Do not rank an MC predictor against a controller using final task reward as if their objectives were identical.

### Estimate error across independent datasets

Freeze the policy and generate several independent sets of episodes, each with the same size. Compute an MC estimate from each set and compare the empirical spread with the analytical value where available. This separates estimator variability from variability caused by learning different policies.

For every-visit MC, retain the episode identifier for each return. Resampling individual visits as if independent can understate uncertainty because several returns share a future reward sequence. Resampling complete episodes better preserves that dependence structure when a bootstrap analysis is appropriate.

<a id="section-17-10"></a>

## 17.10 Problems, worked answers, and reading

1. Recompute the A, B, A example with gamma=0.5.
2. For weights [1,3] and returns [2,4], calculate ordinary and self-normalized importance estimates.
3. What fails if behavior never takes an action used by the target?

<details><summary>Worked answers</summary>

1. Returns are [1,2,2]. A receives 1 under first-visit and an average 1.5 under every-visit selection for this episode.
2. Ordinary: (1×2+3×4)/2=7. Self-normalized: 14/4=3.5. Importance estimates need not lie inside the observed return range unless normalized this way.
3. The required ratio has zero behavior denominator, and logged data contain no evidence about that branch. More averaging of the same unsupported dataset cannot identify its value without additional assumptions.

</details>

Read [Sutton and Barto, Chapter 5](http://incompleteideas.net/book/the-book-2nd.html), especially the distinction between ordinary and weighted importance sampling.

---

[← Chapter 16](../16-value-iteration/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 18 →](../18-temporal-difference-learning/README.md)
