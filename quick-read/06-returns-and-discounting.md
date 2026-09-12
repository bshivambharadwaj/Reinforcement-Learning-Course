**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 5](05-policies.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 7 →](07-state-value-functions.md)

**Go deeper:** [Chapter 6: Returns and Discounting](../chapters/06-returns-and-discounting/README.md)

---

<a id="topic-6"></a>

# 6. Returns and Discounting

Immediate reward alone is insufficient for long-horizon decisions. RL therefore defines the **return**:

<p align="center">
  <img src="../assets/equations/equation-10.svg" alt="G_t=R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\cdots." width="760">
</p>

Equivalently,

<p align="center">
  <img src="../assets/equations/equation-11.svg" alt="G_t=R_{t+1}+\gamma G_{t+1}." width="760">
</p>

The discount factor γ controls how future rewards contribute to current value.

- γ=0: only immediate reward matters.
- γ close to 1: long-term consequences matter strongly.

Discounting can encode time preference, help keep continuing-task returns finite, and affect the effective planning horizon.


### Worked example: immediate reward versus return

Suppose the robot receives rewards −1,−1,+10 and then terminates. With γ=0.9:

<p align="center">
  <img src="../assets/equations/equation-12.svg" alt="G_0=-1+0.9(-1)+0.9^{2}(10)=6.2." width="760">
</p>

Working backward gives G<sub>2</sub>=10, G<sub>1</sub>=−1+0.9(10)=8, and G<sub>0</sub>=−1+0.9(8)=6.2. This is why the recursive definition is useful in code.

**Analogy:** a student may spend effort now to gain a useful skill later. Looking only at today's cost would miss the eventual benefit. Discounting determines how heavily later benefits count; it is not itself a measure of uncertainty in a learned value.

For bounded rewards and γ&lt;1, the geometric weights keep the infinite sum bounded. The rough effective horizon 1/(1−γ) is a useful intuition, not a hard cutoff.


> **Failure Mode — sparse rewards:** if almost every trip fails, most sampled returns look alike. Before changing the optimizer, count successful episodes and verify that random exploration can reach any rewarding outcome. Curriculum or justified shaping can help, but changing the reward also changes what you are teaching.

**Knowledge check:** can two actions with the same immediate reward have different values? Yes: their future state distributions can differ.

---

**Continue in depth:** [Read Chapter 6](../chapters/06-returns-and-discounting/README.md)

[← Topic 5](05-policies.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 7 →](07-state-value-functions.md)
