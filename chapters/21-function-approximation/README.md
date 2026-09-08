# 21. Function Approximation

**Intermediate** · **Created by Shivam Bharadwaj**

[← Chapter 20](../20-q-learning-and-exploration/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 22 →](../22-deep-q-networks/README.md)

[Quick-read Topic 21](../../README.md#topic-21) · [Notation](../NOTATION.md)

- [21.1 Share experience across states](#section-21-1)
- [21.2 Start with a linear value function](#section-21-2)
- [21.3 Calculate shared-parameter interference](#section-21-3)
- [21.4 Define the weighted approximation objective](#section-21-4)
- [21.5 Identify irreducible state aliasing](#section-21-5)
- [21.6 Understand the deadly triad](#section-21-6)
- [21.7 Keep target differentiation explicit](#section-21-7)
- [21.8 Evaluate generalization along meaningful axes](#section-21-8)
- [21.9 Laboratory and prefix-value connection](#section-21-9)
- [21.10 Problems, worked answers, and reading](#section-21-10)

---

<a id="section-21-1"></a>

## 21.1 Share experience across states

Function approximation replaces a separate table entry for every state or action with a parameterized function. This makes large observations tractable and introduces generalization: changing one prediction can affect many others.

**Prerequisites:** Chapters 7–8, 18, and basic gradients. **Targets:** derive a semi-gradient update, identify representation error, and explain why tabular convergence does not transfer automatically.

A courier recognizes that similar intersections often require similar decisions. That reuse helps only if the chosen features preserve distinctions that matter, such as destination and remaining time.

<a id="section-21-2"></a>

## 21.2 Start with a linear value function

Let phi(s) be a feature vector and V_w(s)=w dot phi(s). For a fixed target y, the half-squared loss is 0.5[y−V_w(s)]². Gradient descent gives w←w+alpha[y−V_w(s)]phi(s).

For TD, y=r+gamma V_w(s'). The common semi-gradient update treats this target as fixed during differentiation. It is not the full gradient of the squared expression with the same parameters on both sides.

<a id="section-21-3"></a>

## 21.3 Calculate shared-parameter interference

Let phi(A)=[1,0], phi(B)=[1,1], and w=[0,0]. A target of 2 at A with alpha=0.1 updates w to [0.2,0]. Both V(A) and V(B) become 0.2, although B was not observed.

This is generalization, not inherently an error. If B's true value is −2, however, the update initially moves it in the wrong direction. Feature design determines which states help or interfere with each other.

<a id="section-21-4"></a>

## 21.4 Define the weighted approximation objective

With known true values, one could minimize sum_s d(s)[V_w(s)−V_pi(s)]² for a chosen weighting distribution d. The best representable fit depends on d as well as the features.

TD generally solves a different projected fixed-point problem under appropriate linear settings. Low error on frequently visited states does not certify accuracy everywhere. A policy change can shift visitation toward precisely the states that were poorly represented.

<a id="section-21-5"></a>

## 21.5 Identify irreducible state aliasing

Suppose two observations map to the same feature vector but one requires left and the other right because their hidden destinations differ. No parameter setting can represent distinct values from identical inputs in a deterministic feedforward model.

More training steps cannot recover information that the representation discards. Add the destination, time, or sufficient history when those variables determine transitions or rewards. A recurrent model can summarize history, but does not guarantee that it learns a sufficient statistic.

<a id="section-21-6"></a>

## 21.6 Understand the deadly triad

Bootstrapping uses current estimates in targets. Off-policy learning trains from a distribution different from the target policy's. Function approximation couples predictions across states. Their combination can lead to divergence, even with linear functions in constructed examples.

This is a warning about missing general guarantees, not a claim that every such algorithm fails. Replay buffers and target networks often improve practical behavior, but do not restore the original tabular theorem for arbitrary neural systems.

<a id="section-21-7"></a>

## 21.7 Keep target differentiation explicit

```text
prediction = value_network(state)
with no gradient:
    target = reward + gamma * mask * target_value(next_state)
loss = mean((prediction − target)²)
```

If deliberately optimizing a residual-gradient objective, explain that different objective and its sampling issues. Differentiating both sides by accident is neither a harmless implementation variant nor standard semi-gradient TD.

<a id="section-21-8"></a>

## 21.8 Evaluate generalization along meaningful axes

Hold out states, layouts, goals, or task templates depending on the intended claim. Randomly splitting nearby transitions from the same trajectory can leak nearly identical observations into both sets.

Plot errors by goal and time remaining, not just a single aggregate loss. Compare against a tabular or linear baseline on small tasks. A larger network with lower training loss can still make worse decisions if its ranking errors concentrate at important choice points.

<a id="section-21-9"></a>

## 21.9 Laboratory and prefix-value connection

[Notebook 04](../../notebooks/04_dqn.ipynb) introduces neural action values. [Notebook 08](../../notebooks/08_one_problem_many_algorithms.ipynb) enables representation comparisons under a common environment contract.

A language-model critic maps many prefixes to shared parameters. Prefix similarity can support transfer, but hidden evaluator rules or omitted tool state can create aliasing. Check what information the critic receives before interpreting its prediction errors as an optimizer problem.

<a id="section-21-10"></a>

## 21.10 Problems, worked answers, and reading

1. From w=[0.2,0], update on B with target −2 and alpha=0.1. What happens to V(A)?
2. Why can zero training error coexist with poor policy return?
3. In a TD loss, what makes the update a semi-gradient?

<details><summary>Worked answers</summary>

1. Prediction at B is 0.2; error is −2.2. New w=[−0.02,−0.22], so V(A)=−0.02. An update at B changed A.
2. Training coverage may omit important states or actions; small errors may reverse close action rankings; or the target itself may be wrong.
3. The derivative passes through the current prediction but treats the bootstrap target as constant, despite its conceptual dependence on learned values.

</details>

Read [Sutton and Barto, Chapters 9 and 11](http://incompleteideas.net/book/the-book-2nd.html), paying attention to which results are linear, on-policy, or tabular.

---

[← Chapter 20](../20-q-learning-and-exploration/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 22 →](../22-deep-q-networks/README.md)
