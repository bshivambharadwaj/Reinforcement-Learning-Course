# 14. Policy Improvement

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 13](../13-policy-evaluation/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 15 →](../15-policy-iteration/README.md)

[Quick-read Topic 14](../../quick-read/14-policy-improvement.md) · [Notation](../NOTATION.md)

- [14.1 Improvement is a statement about future behavior](#section-14-1)
- [14.2 State the sufficient condition](#section-14-2)
- [14.3 Prove the theorem by repeated backups](#section-14-3)
- [14.4 Calculate a stochastic improvement](#section-14-4)
- [14.5 Greedy selection and ties](#section-14-5)
- [14.6 Relate improvement to advantages](#section-14-6)
- [14.7 Quantify approximate greediness](#section-14-7)
- [14.8 Design a falsifiable experiment](#section-14-8)
- [14.9 Laboratory and modern connection](#section-14-9)
- [14.10 Problems, worked answers, and reading](#section-14-10)

---

<a id="section-14-1"></a>

## 14.1 Improvement is a statement about future behavior

Policy improvement uses the values of one policy to construct another. The surprising result is that choosing better one-step actions, while evaluating continuation under the old policy, can improve the entire resulting policy.

**Prerequisites:** Chapters 8, 10–11, and 13. **Targets:** prove the improvement theorem, handle ties, and explain why the theorem does not justify arbitrary neural updates.

Think of a courier reviewing each intersection using the current route manual. If every revised choice is at least as good under that manual, repeatedly following the revised manual is also at least as good under the finite discounted assumptions.

<a id="section-14-2"></a>

## 14.2 State the sufficient condition

For a finite discounted MDP, suppose a new policy pi_new satisfies:

```text
sum_a pi_new(a|s) Q_pi_old(s,a) ≥ V_pi_old(s), for every s
equivalently: T_pi_new V_pi_old ≥ V_pi_old
```

The inequality is componentwise. Greediness is sufficient, but not necessary: a stochastic policy can satisfy it too. An average inequality across a training batch is weaker and does not meet this all-state condition.

<a id="section-14-3"></a>

## 14.3 Prove the theorem by repeated backups

Bellman expectation operators are monotone: v≥w implies T_pi v≥T_pi w, since transition probabilities are nonnegative. Starting with v_old≤T_new v_old, apply monotonicity repeatedly:

```text
v_old ≤ T_new v_old ≤ T_new² v_old ≤ ... → v_new
```

The last step follows from discounted contraction. Thus v_new≥v_old at every state. This proof identifies two distinct ingredients: monotonicity preserves the order, and convergence identifies the limit as the new policy's value.

### Write improvement as a resolvent of local advantages

Let g=T_new V_old−V_old be the vector of one-step improvement margins. Subtract the two policy Bellman equations and rearrange to obtain V_new−V_old=(I−gamma P_new)^(-1)g.

The inverse is a sum of nonnegative matrices, I+gamma P_new+gamma²P_new²+…, so g≥0 implies V_new−V_old≥0. This gives another proof of improvement and shows where gains propagate: a state's improvement includes discounted visits to states with positive local margins under the new policy.

If g is positive only in states unreachable from the chosen start distribution, the start-state objective need not improve strictly. Componentwise policy improvement and strictly higher average evaluation return are related but distinct claims.

<a id="section-14-4"></a>

## 14.4 Calculate a stochastic improvement

At one state, Q_old(left)=8 and Q_old(right)=2. The old policy chooses left with probability 0.25, giving V_old=3.5. A new left probability 0.5 gives a one-step old-continuation value of 5, an improvement margin of 1.5.

That 5 is not necessarily V_new: future choices also change. If the sufficient condition holds at every other state, the theorem guarantees V_new≥V_old, while repeated evaluation determines its actual value. With terminal actions in this particular state, the one-step value is the full new value.

<a id="section-14-5"></a>

## 14.5 Greedy selection and ties

Choose pi_new(s) from argmax_a Q_old(s,a). If the old action is among the maximizers, retaining it avoids meaningless switching. Equality can mean several optimal actions, or simply that the policy has not changed value at a state.

A deterministic tie rule is useful for finite policy-iteration termination. Without stable tie handling, an implementation can alternate among value-equivalent policies indefinitely even though it has already reached optimal values.

<a id="section-14-6"></a>

## 14.6 Relate improvement to advantages

The sufficient condition can be written E_(a~pi_new)[A_old(s,a)]≥0. The old policy has zero expected advantage under its own action distribution; shifting probability toward positive old advantages raises the local surrogate.

Policy-gradient and PPO methods use this insight with samples, approximate advantages, and limited updates. Their data distribution and approximation errors matter. Maximizing an empirical advantage surrogate does not establish the all-state premise of the exact theorem.

<a id="section-14-7"></a>

## 14.7 Quantify approximate greediness

Suppose |Q_hat(s,a)−Q_old(s,a)|≤epsilon for all actions at a state. The action maximizing Q_hat can be at most 2epsilon below the true best Q_old action: insert Q_hat between the two true values and bound both estimation errors.

If the estimated improvement over the old action exceeds 2epsilon, its true one-step improvement is positive. When errors are only known on average, this robust sign argument is unavailable. Confidence intervals must also account for how an action was selected from the same noisy estimates.

### Bound the damage from imperfect improvement

If the new policy only satisfies g(s)≥−eta everywhere, the same inverse expression gives V_new−V_old≥−eta/(1−gamma) componentwise. A small local violation can accumulate along future visits.

At gamma=0.95 and eta=0.02, the bound permits a loss of 0.4. If g≥0 on sampled states but is unknown elsewhere, the all-state bound is unavailable. This is one reason conservative policy changes can help in practice: they reduce opportunities to enter poorly estimated regions, although a heuristic small update does not prove the required inequality.

<a id="section-14-8"></a>

## 14.8 Design a falsifiable experiment

On a small known MDP, evaluate pi_old exactly. Construct a greedy policy, verify the improvement condition at every state, and then solve for V_new. Assert the full vector inequality within numerical tolerance.

Next add bounded perturbations to Q and look for states where the condition fails. This extension separates an exact theorem from its approximate implementation. Record the largest negative true improvement margin, not just the mean start-state return.

### Test strict improvement and tie handling separately

Build one state with two equally valuable terminal actions and another with a clearly better alternative. Verify that the implementation preserves the old action in the tie state while changing the strictly improvable state.

Then evaluate both policies independently. Checking only whether an action changed can falsely label an equivalent policy as better; checking only whether average return changed can hide improvements outside the selected start distribution. Log the full value vector and the local improvement margins on a tiny test MDP before relying on an aggregate result.

<a id="section-14-9"></a>

## 14.9 Laboratory and modern connection

Use [Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) for exact evaluation and improvement. Use [Notebook 06](../../notebooks/06_ppo.ipynb) later to contrast sampled surrogate improvement with measured policy return.

For a language model, increasing probability of responses judged better by an imperfect critic resembles improvement under an approximate Q. The missing guarantee is not repaired by calling the scores “advantages”: coverage, evaluator validity, and distribution change still need evidence.

<a id="section-14-10"></a>

## 14.10 Problems, worked answers, and reading

1. With Q_old=[8,2], what new left probability makes the old-continuation value equal to 4.4?
2. If epsilon=0.2 and the estimated improvement is 0.3, is positive true improvement certified?
3. Where does the proof use nonnegative transition probabilities?

<details><summary>Worked answers</summary>

1. Solve 8p+2(1−p)=4.4, giving p=0.4.
2. No. The true difference can be as low as 0.3−0.4=−0.1. The uniform bound permits a reversal.
3. In monotonicity: an expectation of componentwise nonnegative differences remains nonnegative. An arbitrary signed linear transformation would not preserve that order.

</details>

Read [Sutton and Barto, Chapter 4.2](http://incompleteideas.net/book/the-book-2nd.html), then compare the exact improvement condition with the surrogate objective in [TRPO](https://arxiv.org/abs/1502.05477).

---

[← Chapter 13](../13-policy-evaluation/README.md) | [Chapters](../README.md) | [Course home](../../README.md) | [Chapter 15 →](../15-policy-iteration/README.md)
