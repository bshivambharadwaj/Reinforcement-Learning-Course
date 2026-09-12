**Quick read** · **Created by Shivam Bharadwaj**

[← Topic 31](31-proximal-policy-optimization.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 33 →](33-offline-reinforcement-learning.md)

**Go deeper:** [Chapter 32: Entropy Regularization](../chapters/32-entropy-regularization/README.md)

---

<a id="topic-32"></a>

# 32. Entropy Regularization

> **Why this method exists:** Early lucky actions can drive a policy toward certainty before it has explored alternatives. Entropy regularization rewards diversity, but its strength must still permit reliable execution.

A policy can collapse too quickly toward deterministic behavior. Entropy measures uncertainty in the action distribution:

<p align="center">
  <img src="../assets/equations/equation-62.svg" alt="\mathcal{H}(\pi(\cdot|s))=-\sum_a\pi(a|s)\log\pi(a|s)." width="760">
</p>

An entropy bonus can be added to encourage exploration:

<p align="center">
  <img src="../assets/equations/equation-63.svg" alt="J&#x27;=J+\beta\mathcal{H}(\pi)." width="760">
</p>

The same general idea appears throughout modern generative-model optimization: policy objectives often need regularization so that optimization does not destroy useful diversity or move too far from a reference behavior.


### Diversity and reference matching are different

For two actions, probabilities (0.5,0.5) have entropy log 2, while (1,0) has entropy zero, taking 0log 0=0. An entropy bonus favors keeping options open. Its coefficient controls how much the objective values that diversity relative to task reward.

**Analogy:** keep trying several routes instead of immediately committing to one after a lucky trip. Too much entropy pressure can also prevent reliable execution of a good route.

A KL penalty to a reference policy serves a different purpose: it discourages departure from a specific distribution, which may itself be highly concentrated. For a practical objective, entropy is averaged over visited states or summed over a trajectory, rather than added as an unspecified single-state number.


**Read the source:** [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html).

---

**Continue in depth:** [Read Chapter 32](../chapters/32-entropy-regularization/README.md)

[← Topic 31](31-proximal-policy-optimization.md) | [Course home](../README.md) | [Quick reads](README.md) | [Topic 33 →](33-offline-reinforcement-learning.md)
