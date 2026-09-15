"""Inspectable CPU algorithms for GoalLab's evidence-controller stages. MIT."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .goallab import EvidenceMDP, make_workspace, verify


def greedy(q):
    return np.eye(2)[np.asarray(q).argmax(1)]


def prediction(seed=7, episodes=600):
    env = EvidenceMDP()
    rng = np.random.default_rng(seed)
    policy = np.full((env.n, 2), .5)
    truth, _ = env.values(policy)
    mc, td, counts = np.zeros(env.n), np.zeros(env.n), np.zeros(env.n)
    history = []
    visited = np.zeros(env.n, dtype=bool)
    for ep in range(episodes):
        state = env.start_index
        trajectory = []
        for _ in range(4):
            action = int(rng.integers(2))
            ns, reward, done = env.step(state, action, rng)
            trajectory.append((state, reward, ns, done))
            visited[state] = True
            state = ns
        total = 0.
        for s, r, ns, done in reversed(trajectory):
            total = r + total
            counts[s] += 1
            mc[s] += (total - mc[s]) / counts[s]
        for s, r, ns, done in trajectory:
            td[s] += .1 * (r + (0 if done else td[ns]) - td[s])
        if (ep + 1) % 24 == 0:
            # All decision states in this enumerated MDP are reachable under the random policy.
            idx = [i for i in range(env.n) if i != env.end_index]
            history.append(((ep + 1) * 4, np.sqrt(np.mean((mc[idx]-truth[idx])**2)),
                            np.sqrt(np.mean((td[idx]-truth[idx])**2))))
    return {'env': env, 'mc': mc, 'td': td, 'truth': truth, 'history': np.array(history)}


def train_tabular(method='Q-learning', seed=7, episodes=600, env=None):
    if method not in {'Q-learning', 'SARSA', 'MC control'}:
        raise ValueError(method)
    env = env or EvidenceMDP()
    rng = np.random.default_rng(seed)
    q, counts = np.zeros((env.n, 2)), np.zeros((env.n, 2))
    history = []
    for ep in range(episodes):
        epsilon = max(.08, .8 * (1 - ep / episodes))
        def choose(s):
            return int(rng.integers(2)) if rng.random() < epsilon else int(q[s].argmax())
        state = env.start_index
        action = choose(state)
        trajectory = []
        for _ in range(4):
            ns, reward, done = env.step(state, action, rng)
            na = choose(ns) if not done else 0
            trajectory.append((state, action, reward))
            if method != 'MC control':
                boot = q[ns, na] if method == 'SARSA' else q[ns].max()
                q[state, action] += .25 * (reward + (0 if done else boot) - q[state, action])
            state, action = ns, na
        if method == 'MC control':
            total = 0.
            for state, action, reward in reversed(trajectory):
                total += reward
                counts[state, action] += 1
                q[state, action] += (total - q[state, action]) / counts[state, action]
        if (ep + 1) % 24 == 0:
            history.append(((ep + 1) * 4, env.values(greedy(q))[0][env.start_index]))
    return {'env': env, 'q': q, 'policy': greedy(q), 'history': np.array(history)}


def network(outputs):
    return nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, outputs))


def train_dqn(seed=7, episodes=600):
    torch.manual_seed(seed)
    env, rng = EvidenceMDP(), np.random.default_rng(seed)
    features = torch.tensor(env.features())
    online, replay = network(2), deque(maxlen=3000)
    target = deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=.003)
    history, losses, steps = [], [], 0
    for ep in range(episodes):
        s = env.start_index
        epsilon = max(.08, .8 * (1 - ep / episodes))
        for _ in range(4):
            with torch.no_grad():
                a = int(rng.integers(2)) if rng.random() < epsilon else int(online(features[s]).argmax())
            ns, r, done = env.step(s, a, rng)
            replay.append((s, a, r, ns, done))
            s, steps = ns, steps + 1
            if len(replay) >= 64:
                batch = np.array([replay[i] for i in rng.choice(len(replay), 64, replace=False)])
                bs, ba, br, bn, bd = [torch.tensor(batch[:, i], dtype=torch.long if i in (0, 1, 3) else torch.float32)
                                      for i in range(5)]
                with torch.no_grad():
                    labels = br + (1 - bd) * target(features[bn]).max(1).values
                loss = nn.functional.smooth_l1_loss(online(features[bs]).gather(1, ba[:, None]).squeeze(1), labels)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5); optimizer.step()
                losses.append(float(loss.detach()))
            if steps % 100 == 0:
                target.load_state_dict(online.state_dict())
        if (ep + 1) % 24 == 0:
            with torch.no_grad():
                policy = greedy(online(features).numpy())
            history.append((steps, env.values(policy)[0][env.start_index]))
    with torch.no_grad():
        policy = greedy(online(features).numpy())
    return {'env': env, 'policy': policy, 'model': online, 'history': np.array(history), 'losses': losses}


def train_policy(method='PPO', seed=7, episodes=600):
    if method not in {'REINFORCE', 'Actor-Critic', 'PPO'} or episodes % 24:
        raise ValueError('Supported policy method and a multiple of 24 episodes required')
    torch.manual_seed(seed)
    env, rng = EvidenceMDP(), np.random.default_rng(seed)
    features = torch.tensor(env.features())
    actor, critic = network(2), network(1)
    params = list(actor.parameters()) + ([] if method == 'REINFORCE' else list(critic.parameters()))
    optimizer = torch.optim.Adam(params, lr=.01 if method != 'PPO' else .004)
    history = []
    for update in range(episodes // 24):
        batch = []
        for _ in range(24):
            s = env.start_index
            for _ in range(4):
                with torch.no_grad():
                    dist = Categorical(logits=actor(features[s]))
                    a = dist.sample()
                    logp, value = dist.log_prob(a), critic(features[s]).squeeze()
                ns, r, done = env.step(s, int(a), rng)
                batch.append((s, int(a), r, float(logp), float(value)))
                s = ns
        states = torch.tensor([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch]).reshape(24, 4)
        old_logp = torch.tensor([b[3] for b in batch])
        values = torch.tensor([b[4] for b in batch]).reshape(24, 4)
        next_values = torch.cat([values[:, 1:], torch.zeros(24, 1)], dim=1)
        delta = rewards + next_values - values
        returns = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1])
        if method == 'REINFORCE':
            advantage, targets = returns.flatten(), returns.flatten()
        elif method == 'Actor-Critic':
            advantage, targets = delta.flatten(), (rewards + next_values).flatten()
        else:
            gae, carry = torch.zeros_like(values), torch.zeros(24)
            for t in reversed(range(4)):
                carry = delta[:, t] + .95 * carry
                gae[:, t] = carry
            targets, advantage = (gae + values).flatten(), gae.flatten()
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        for _ in range(4 if method == 'PPO' else 1):
            dist = Categorical(logits=actor(features[states]))
            logp = dist.log_prob(actions)
            if method == 'PPO':
                ratio = (logp - old_logp).exp()
                actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            else:
                actor_loss = -(logp * advantage).mean()
            loss = actor_loss - .01 * dist.entropy().mean()
            if method != 'REINFORCE':
                loss += .5 * nn.functional.mse_loss(critic(features[states]).squeeze(1), targets)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1); optimizer.step()
        with torch.no_grad():
            policy = actor(features).softmax(-1).numpy()
        history.append(((update + 1) * 24 * 4, env.values(policy)[0][env.start_index]))
    return {'env': env, 'policy': policy, 'model': actor, 'history': np.array(history)}


def heldout(policy, cases=100, seed=1001):
    env = EvidenceMDP()
    sessions = [env.deploy(policy, make_workspace(i, 'test'), seed + i) for i in range(cases)]
    return {'success': float(np.mean([verify(s.workspace, s.artifact)['success'] for s in sessions])),
            'calls': float(np.mean([len(s.events) for s in sessions])), 'sessions': sessions}


def run_control(seeds=(7, 19, 31), episodes=600):
    results = {}
    for method in ['MC control', 'SARSA', 'Q-learning', 'DQN', 'REINFORCE', 'Actor-Critic', 'PPO']:
        print('GoalLab:', method, flush=True)
        if method in {'MC control', 'SARSA', 'Q-learning'}:
            results[method] = [train_tabular(method, s, episodes) for s in seeds]
        elif method == 'DQN':
            results[method] = [train_dqn(s, episodes) for s in seeds]
        else:
            results[method] = [train_policy(method, s, episodes) for s in seeds]
    return results
