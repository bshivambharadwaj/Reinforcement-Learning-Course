**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 8](08-action-value-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 10 →](10-bellman-expectation-equations.md)

**Go deeper:** [Chapter 9: Advantage Functions](../chapters/09-advantage-functions/README.md)

---

<a id="topic-9"></a>

# 9. Advantage Functions

The advantage function measures whether an action is better or worse than the policy's typical action at that state:

<p align="center">
  <img src="../assets/equations/equation-17.svg" alt="A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)." width="760">
</p>

Interpretation:

- A&gt;0: action is better than the state baseline;
- A&lt;0: action is worse;
- A≈0: action is close to expected behavior.

Advantages are central to modern policy-gradient algorithms because subtracting a baseline can reduce variance without changing the expected policy-gradient direction.


### Worked example: better than your usual choice

Using the previous equal-probability policy, V<sup>π</sup>(s)=5.5. Therefore:

<p align="center">
  <img src="../assets/equations/equation-18.svg" alt="A^{\pi}(s,\text{east})=8-5.5=2.5,\qquad
A^{\pi}(s,\text{north})=3-5.5=-2.5." width="760">
</p>

North still has positive expected return, but it is worse than this policy's average choice. That distinction matters when improving behavior.

**Analogy:** scoring 70 on an exam means something different when your expected score was 50 versus 90. Advantage measures performance relative to expectation. For the exact value functions, ∑<sub>a</sub>π(a&#124;s)A<sup>π</sup>(s,a)=0; sampled or approximate advantages need not average to exactly zero.

---

**Continue in depth:** [Read Chapter 9](../chapters/09-advantage-functions/README.md)

[← Topic 8](08-action-value-functions.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 10 →](10-bellman-expectation-equations.md)
