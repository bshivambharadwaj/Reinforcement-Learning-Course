# 9. Advantage Functions

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 8](../08-action-value-functions/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 10 →](../10-bellman-expectation-equations/README.md)

[Quick-read Topic 9](../../quick-read/09-advantage-functions.md) · [Notation](../NOTATION.md)

- [9.1 Relative feedback answers a different question](#section-9-1)
- [9.2 Definition and zero-mean property](#section-9-2)
- [9.3 A nonuniform-policy example](#section-9-3)
- [9.4 Derive the TD residual connection](#section-9-4)
- [9.5 Monte Carlo, multi-step, and GAE estimates](#section-9-5)
- [9.6 Why an action-independent baseline cancels](#section-9-6)
- [9.7 Normalization is an engineering choice, not the definition](#section-9-7)
- [9.8 Lab: compare signal definitions without changing the task](#section-9-8)
- [9.9 Relative signals in reasoning and agents](#section-9-9)
- [9.10 Problems, worked answers, and sources](#section-9-10)

---

<a id="section-9-1"></a>

## 9.1 Relative feedback answers a different question

An action can receive a positive return simply because it was taken in an easy state. Advantage asks how the action compares with the policy's usual performance from that state. It is a centered action value, not a new environment reward.

**Prerequisites:** Chapters 7–8 and conditional expectation. **Targets:** derive the zero-mean identity, calculate exact and estimated advantages, explain valid baselines, and separate action ranking from causal attribution.

For a delivery dispatcher, completing an easy nearby job is unsurprising. Finishing an unexpectedly difficult job may deserve a stronger policy update even when its raw return is lower. The baseline supplies the context for that comparison.

<a id="section-9-2"></a>

## 9.2 Definition and zero-mean property

Define A_pi(s,a)=Q_pi(s,a)−V_pi(s). Averaging under the same policy gives:

```text
sum_a pi(a|s) A_pi(s,a)
  = sum_a pi(a|s) Q_pi(s,a) − V_pi(s) sum_a pi(a|s)
  = V_pi(s) − V_pi(s) = 0
```

The result depends on using compatible exact Q_pi and V_pi and on averaging with pi. Estimated advantages need not have exactly zero sample mean, and an average under another policy need not be zero.

Subtracting a state-dependent constant preserves the ranking of actions within that state. It does not preserve comparisons of raw numbers across different states with different baselines, which is precisely why advantage and value answer different questions.

### Distinguish weighted centering from arithmetic centering

For exact A_pi, sum_a pi(a|s)A_pi(s,a)=0 because V_pi is the policy-weighted Q average. The unweighted sum over actions need not be zero. If pi=[0.9,0.1] and Q=[2,12], then V=3 and advantages are [−1,9]. Their arithmetic mean is 4, but their policy-weighted mean is zero.

Subtracting the arithmetic mean of these advantages would produce [−5,5]. This is a different centered signal. As a state-only baseline shift it can retain the expected score gradient under suitable conditions, but the resulting numbers no longer equal the defined Q_pi−V_pi advantages.

<a id="section-9-3"></a>

## 9.3 A nonuniform-policy example

Let Q_pi(s,left)=8, Q_pi(s,right)=2, and pi(left|s)=0.25. Then V_pi(s)=3.5. The advantages are +4.5 and −1.5. Their ordinary unweighted average is +1.5, but their policy-weighted average is 0.25×4.5+0.75×(−1.5)=0.

This example catches the misconception that advantages must sum to zero over actions without weighting. Uniform-policy examples can hide that distinction.

If the policy becomes more likely to choose left, its value and future trajectory distribution may change. Reusing the old exact advantage can support a local improvement argument, but it should still be labeled as the old policy's advantage rather than the new policy's fully evaluated quantity.

<a id="section-9-4"></a>

## 9.4 Derive the TD residual connection

For a transition under pi, define delta_t=R_(t+1)+gamma V_pi(S_(t+1))−V_pi(S_t), with zero terminal continuation. Condition on S_t=s and A_t=a. The expected reward-plus-next-value term is Q_pi(s,a), so E[delta_t|s,a]=A_pi(s,a).

This result uses the exact value function. With an approximation V_hat=V_pi+e, the conditional error in the expected residual is gamma E[e(S_(t+1))|s,a]−e(s), adjusted for terminal masking. A critic's approximation error can therefore bias an advantage estimate differently across actions.

A single residual is also noisy because the next outcome is sampled. Exact conditional expectation and a realized residual are not the same object.

<a id="section-9-5"></a>

## 9.5 Monte Carlo, multi-step, and GAE estimates

A complete-return estimate G_t−V_hat(S_t) uses observed future rewards and a state baseline. A k-step estimate uses k observed rewards and then bootstraps. GAE combines several horizons of TD residuals.

These estimators trade sampling variability, dependence on the critic, and delay before feedback becomes available. A state baseline can reduce policy-gradient variance without changing its expectation under the usual score-function assumptions; using an approximate critic inside a bootstrap can introduce additional bias.

Do not conflate these roles. Subtracting a baseline from a complete return and replacing an unobserved future return with a critic prediction are distinct operations, even when they use the same value network.

<a id="section-9-6"></a>

## 9.6 Why an action-independent baseline cancels

For a fixed state, the expected score is sum_a pi(a|s) grad log pi(a|s)=sum_a grad pi(a|s)=grad 1=0. Multiplying by a baseline b(s) independent of the sampled action leaves this zero expectation unchanged.

Thus replacing a return weight by return minus b(s) does not change the expected score-function term, provided gradients and sampling are handled consistently. In an implementation, the advantage weight is usually detached so its derivative is not accidentally added to the actor objective.

An arbitrary action-dependent baseline does not cancel. Specialized correction methods exist, but merely subtracting a number called `baseline` is not enough. A same-batch statistic can also depend on the sampled action through its own return; Chapter 26 examines this subtlety.

### Derive the variance-minimizing scalar baseline

At a fixed state, let z=gradient log pi(a|s) and use estimator (G−b)z. Its expected gradient is independent of an action-independent b. Minimizing its second moment E[(G−b)²||z||²] gives derivative −2E[(G−b)||z||²]. Setting that derivative to zero yields b_star=E[G||z||²]/E[||z||²], provided the denominator is nonzero.

The ordinary value E[G|s] is recovered when score magnitude is constant or suitably uncorrelated with return. Otherwise the variance-optimal scalar baseline weights outcomes by score magnitude. This result concerns one state's score estimator; it does not guarantee that fitting an approximate value network minimizes the variance of a full correlated trajectory gradient.

<a id="section-9-7"></a>

## 9.7 Normalization is an engineering choice, not the definition

Subtracting a batch mean and dividing by a batch standard deviation can change update scaling and sample dependence. The resulting weights are not automatically exact A_pi. With finite batches, normalization interacts with policy updates, optimizer settings, and trajectory weighting.

For group-relative reasoning, rewards are centered within a prompt's sampled candidates. That group baseline is not a learned state-value function, and the own-sample dependence matters for interpreting the estimator. The normalization rule and clipped objective together define the practical method.

> **Common Misconception:** every positive normalized advantage proves that a reasoning step was correct. It only means the sample scored better relative to the chosen comparison baseline and reward.

<a id="section-9-8"></a>

## 9.8 Lab: compare signal definitions without changing the task

Use [Notebook 05](../../notebooks/05_policy_gradient.ipynb) to compare complete-return weighting with a lagged scalar baseline. Then inspect the critic-based GAE in [Notebook 06](../../notebooks/06_ppo.ipynb). Explain where each estimator uses observed rewards and where it uses predictions.

On a tiny known MDP, compute exact A_pi and compare it with the empirical average TD residual for each state-action pair. Inject a deliberately biased value estimate and predict the direction of residual bias before sampling.

Report both estimator statistics and policy outcomes. A lower variance of scalar weights alone does not prove a lower variance of the full parameter-gradient estimator, because score vectors vary too.

<a id="section-9-9"></a>

## 9.9 Relative signals in reasoning and agents

Suppose four attempts at one arithmetic prompt score [0,1,1,0]. Centering around 0.5 distinguishes the successful attempts from failures. If all attempts score 0, the group supplies no reward-ranking information even if one trace is intuitively closer to correct.

A tool agent may use a critic to judge whether a call was better than expected given its history. This can value information gathering, but only through the defined future-return signal. A good-looking intermediate tool result is not intrinsically a positive advantage.

Advantage helps assign update direction relative to a baseline. It is not, by itself, a causal explanation of which internal reasoning operation produced success.

### Test whether a positive advantage means correctness

Suppose a verifier rewards a wrong response 0.2 and two other wrong responses 0 and 0.1. Relative to their mean, the first wrong response receives positive centered reward. The signal asks the policy to prefer it within the sampled comparison, not to certify it as correct.

Conversely, a correct response can receive negative advantage if the reward also measures cost and the comparison includes cheaper correct responses. Log correctness and relative training signal separately. This makes a group-based learning curve interpretable when score ranking and binary task success do not coincide.

<a id="section-9-10"></a>

## 9.10 Problems, worked answers, and sources

**A.** With Q=[5,1] and pi=[0.6,0.4], compute V and both advantages. **B.** Reward is 1, gamma=0.9, current value 3, and next value 4. Compute delta for a continuing transition and for true termination. **C.** Explain why an arbitrary b(s,a) does not cancel from a policy gradient.

<details><summary>Worked answers</summary>

A: V=3.4 and A=[1.6,−2.4]; 0.6×1.6+0.4×(−2.4)=0. B: continuing delta is 1+3.6−3=1.6; terminal delta is 1−3=−2. C: b cannot be factored outside the sum over actions, so the weighted score sum is not generally grad 1 multiplied by a constant. A correction would need to account for its action dependence.

</details>

**Read:** [Sutton and Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html); [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

---

[← Chapter 8](../08-action-value-functions/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 10 →](../10-bellman-expectation-equations/README.md)
