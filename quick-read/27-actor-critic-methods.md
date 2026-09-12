**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 26](26-reinforce.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 28 →](28-advantage-actor-critic.md)

**Go deeper:** [Chapter 27: Actor–Critic Methods](../chapters/27-actor-critic-methods/README.md)

---

<a id="topic-27"></a>

# 27. Actor–Critic Methods

> **Why this algorithm exists:** REINFORCE asks every episode to supply a fresh full-return estimate. A critic shares predictive information across visits and supplies shorter bootstrap targets or baselines, often reducing variance while introducing approximation error.

**[Try it in Notebook 06 → PPO](../notebooks/06_ppo.ipynb)**

Actor–critic methods combine two learners:

<p align="center">
  <img src="../assets/diagrams/actor-critic.svg" alt="The actor chooses; the critic evaluates" width="760">
</p>

The **actor** changes the policy. The **critic** estimates value information used to judge the actor's actions.

A typical actor update uses

<p align="center">
  <img src="../assets/equations/equation-51.svg" alt="\nabla_\theta\log\pi_\theta(A_t|S_t)\hat A_t." width="760">
</p>

This architecture underlies many modern RL algorithms.


### The performer and the coach

The actor is the courier choosing roads; the critic is a coach estimating how promising each situation is. The coach does not select actions directly in a state-value actor–critic, but provides feedback for changing their probabilities.

For a one-step implementation:

1. Sample an action and observe the transition.
2. Compute a terminal-masked TD error using the critic.
3. Train the critic toward the TD target.
4. Train the actor using a detached TD error as an advantage estimate.

The critic can update from partial trajectories, reducing the need to wait for full Monte Carlo returns. This introduces dependence on critic accuracy: a systematically wrong coach can mislead the actor. Actor and critic may use separate networks or share some representation layers.


**Read the source:** [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 27](../chapters/27-actor-critic-methods/README.md)

[← Topic 26](26-reinforce.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 28 →](28-advantage-actor-critic.md)
