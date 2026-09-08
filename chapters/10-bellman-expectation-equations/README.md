# 10. Bellman Expectation Equations

**Foundations** · **Created by Shivam Bharadwaj**

[← Chapter 9](../09-advantage-functions/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 11 →](../11-bellman-optimality-equations/README.md)

[Quick-read Topic 10](../../README.md#topic-10) · [Notation](../NOTATION.md)

- [10.1 A recursive definition becomes a solvable equation](#section-10-1)
- [10.2 Define the fixed-policy operator](#section-10-2)
- [10.3 Derive the matrix solution](#section-10-3)
- [10.4 Solve a recurrent two-state example](#section-10-4)
- [10.5 Prove contraction in maximum norm](#section-10-5)
- [10.6 Residuals turn stopping into an error statement](#section-10-6)
- [10.7 Expectations, samples, and projection are different updates](#section-10-7)
- [10.8 Lab: measure both residual and actual error](#section-10-8)
- [10.9 Why the equation matters for critics and world models](#section-10-9)
- [10.10 Problems, worked answers, and sources](#section-10-10)

---

<p align="center">
  <img src="../../assets/diagrams/bellman-decomposition.svg" width="760" alt="A Bellman expectation backup combines immediate reward and continuation value." />
</p>

*A Bellman expectation backup combines immediate reward and continuation value.*

<a id="section-10-1"></a>

## 10.1 A recursive definition becomes a solvable equation

Bellman expectation equations describe the value of a fixed policy. They do not maximize over actions and do not require the policy to be optimal. Their importance is that a long-horizon expectation can be expressed using one-step outcomes and the same value function at the next state.

**Prerequisites:** Chapters 3 and 6–8, matrix-vector multiplication, and maximum norms. **Targets:** derive the operator, prove its contraction under discounting, solve a two-state system, and turn a Bellman residual into a value-error bound.

The delivery analogy is a ledger: current expected net outcome equals the next transaction plus the discounted value of the situation left afterward. The ledger closes because every future situation obeys the same accounting rule.

<a id="section-10-2"></a>

## 10.2 Define the fixed-policy operator

For any candidate vector v, let T_pi v=r_pi+gamma P_pi v. The true value is the fixed point V_pi=T_pi V_pi. The operator acts on a guess; the equation characterizes the correct answer.

```text
(T_pi v)(s) = sum_a pi(a|s) sum_(s',r) p(s',r|s,a)
              [r + gamma v(s')]
```

If rewards depend on both state and next state, average them consistently with the same transition kernel. A Bellman backup using one model's rewards and another model's transitions solves a hybrid problem that may not correspond to any intended environment.

Terminal states can be included as zero-reward absorbing states or removed with explicit zero continuation. State which convention the matrices use.

<a id="section-10-3"></a>

## 10.3 Derive the matrix solution

Rearranging V_pi=r_pi+gamma P_pi V_pi yields (I−gamma P_pi)V_pi=r_pi. For finite discounted tasks, the matrix is invertible because the series I+gamma P_pi+gamma²P_pi²+... converges.

That series gives a second interpretation: P_pi^k r_pi is the vector of expected reward k transitions into the future, and gamma^k discounts it. The linear solve and the trajectory expectation agree because they represent the same accumulated rewards.

Use a numerical linear solver rather than explicitly multiplying by a computed inverse. Check the residual afterward, especially when gamma is close to one and numerical conditioning can worsen. The identity is exact mathematics; finite-precision computation is an approximation to it.

<a id="section-10-4"></a>

## 10.4 Solve a recurrent two-state example

State A always moves to B with reward 0. At B, reward 1 is received and the next state is A or B with equal probability. There is no terminal state. Use gamma=0.9 and a fixed policy already incorporated into these dynamics.

```text
V(A) = 0.9 V(B)
V(B) = 1 + 0.9[0.5 V(A) + 0.5 V(B)]
```

Substitution gives V(B)=1+0.855V(B), hence V(B)=200/29≈6.89655 and V(A)=180/29≈6.20690. Starting from zeros, the first synchronous backup gives [0,1], and the second gives [0.9,1.45]. These early guesses are not inconsistent with the larger fixed point; they have not yet accumulated the long future.

<a id="section-10-5"></a>

## 10.5 Prove contraction in maximum norm

For candidate vectors v and w, their reward terms cancel:

```text
||T_pi v − T_pi w||_infinity
  = gamma ||P_pi(v−w)||_infinity
  ≤ gamma ||v−w||_infinity
```

Each row of P_pi forms a probability-weighted average, whose absolute value cannot exceed the largest absolute input. With gamma<1, repeated applications shrink differences geometrically. This gives a unique fixed point and convergence from any finite initialization.

The proof does not apply unchanged to undiscounted continuing tasks, arbitrary nonlinear projected updates, or a changing policy. Recognizing where the averaging and discount factors enter is more useful than memorizing the word “contraction.”

<a id="section-10-6"></a>

## 10.6 Residuals turn stopping into an error statement

Let the Bellman residual be delta=||T_pi v−v||_infinity. Insert T_pi v between V_pi and v, use contraction, and rearrange:

```text
||V_pi−v|| ≤ gamma ||V_pi−v|| + delta
||V_pi−v|| ≤ delta / (1−gamma)
```

Thus residual 0.001 with gamma=0.9 implies a maximum value error at most 0.01 under the exact-model assumptions. With gamma=0.99 the same residual permits 0.1. The stopping tolerance must be interpreted together with discounting.

If the operator itself uses approximate rewards or transitions, this residual measures consistency with that approximate operator. It does not directly bound error relative to an unknown true environment without a model-error term.

<a id="section-10-7"></a>

## 10.7 Expectations, samples, and projection are different updates

A full Bellman expectation backup averages all modeled outcomes. TD samples an outcome and uses a stochastic approximation. Function approximation can project or fit targets into a restricted class. These methods share a target idea but do not share every convergence guarantee.

With a neural approximator, a small supervised training loss on sampled states does not imply a small maximum Bellman residual across the state space. Unvisited states and shared-parameter interference can be invisible to the batch loss.

> **Failure Mode:** citing the tabular contraction proof as a convergence proof for arbitrary deep TD training skips the sampling, projection, optimization, and distribution-shift steps where the argument can fail.

<a id="section-10-8"></a>

## 10.8 Lab: measure both residual and actual error

[Notebook 01](../../notebooks/01_mdp_dynamic_programming.ipynb) provides a fixed-policy linear-solve oracle. During iterative evaluation, record the residual, the bound delta/(1−gamma), and the actual maximum error to the oracle. Verify the bound for the exact finite model.

Repeat with several gamma values. Compare iteration counts at the same desired value-error tolerance rather than the same raw residual tolerance. Then change initialization to optimistic and pessimistic vectors and check that both approach the same fixed point.

The experiment should include a deliberately perturbed reward model to show the difference between solver convergence and model correctness. Solving a perturbed model accurately is expected behavior, not a solver failure.

<a id="section-10-9"></a>

## 10.9 Why the equation matters for critics and world models

A critic aims to summarize expected future outcomes under a policy. Bellman consistency supplies a local training relationship that avoids waiting for every complete trajectory. That is the bridge from exact planning to learned value models.

For a tool agent, the next outcome can include a tool error or new information. The expectation should include both, according to the environment model or sampled experience. A language-model reward critic may instead predict continuation return from a prefix.

In both cases, the equation's usefulness depends on matching the policy, reward, state representation, and terminal semantics. Bellman notation does not make those modeling choices disappear.

<a id="section-10-10"></a>

## 10.10 Problems, worked answers, and sources

**A.** In the recurrent example, start from zero. The first synchronous iterate is [0,1]; calculate the third iterate. **B.** A fixed-policy residual is 0.02 at gamma=0.8. Bound value error. **C.** Identify the exact step that fails in the contraction proof if rows of a purported P matrix sum to 1.2 rather than one.

<details><summary>Worked answers</summary>

A: the second vector is [0.9,1.45], so the third is [1.305, 1+0.9×(0.45+0.725)]=[1.305,2.0575]. B: at most 0.1. C: the row operation is no longer a convex average; its maximum-norm bound can be 1.2 times the input norm. It is also not a valid transition matrix. Even if gamma×1.2 happens to be below one, that would describe a different algebraic operator rather than the claimed Markov kernel.

</details>

**Read:** [Sutton and Barto, Chapters 3–4](http://incompleteideas.net/book/the-book-2nd.html). Chapter 11 replaces policy averaging with action optimization.

---

[← Chapter 9](../09-advantage-functions/README.md) | [Chapters](../README.md) | [Quick-read course](../../README.md) | [Chapter 11 →](../11-bellman-optimality-equations/README.md)
