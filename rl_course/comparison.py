"""One finite-horizon delivery MDP, with prediction and control experiments.

DeliveryLine-v1: position 0..4, destination 0 or 4, six moves per episode.
Every move costs .01; occupying the requested destination after move six adds 1.
Time is part of the state. There are 60 decision states and one terminal state.
The model is reserved for the DP baseline and exact evaluation, never training
the sampled methods. Gamma is 1 because the task has a finite horizon.
"""
from collections import deque
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HORIZON = 6
TERMINAL = 60
N_STATES = 61
ENV_VERSION = 'DeliveryLine-v1'


def encode(t, position, goal):
    return t * 10 + goal * 5 + position


def decode(state):
    t, remainder = divmod(int(state), 10)
    goal, position = divmod(remainder, 5)
    return t, position, goal


def transition(state, action):
    """Pure transition function; terminal is absorbing with zero future reward."""
    if state == TERMINAL:
        return TERMINAL, 0.0, True
    t, position, goal = decode(state)
    position = int(np.clip(position + (-1 if action == 0 else 1), 0, 4))
    done = t == HORIZON - 1
    reward = -.01 + float(done and position == 4 * goal)
    return (TERMINAL if done else encode(t + 1, position, goal)), reward, done


NEXT = np.array([[transition(s, a)[0] for a in range(2)] for s in range(N_STATES)])
REWARD = np.array([[transition(s, a)[1] for a in range(2)] for s in range(N_STATES)])
FEATURES = torch.tensor([
    [decode(s)[1] / 4, 2 * decode(s)[2] - 1, (HORIZON - decode(s)[0]) / HORIZON]
    if s != TERMINAL else [0, 0, 0] for s in range(N_STATES)
], dtype=torch.float32)
STARTS = [encode(0, 2, g) for g in range(2)]


def evaluate(policy=None):
    """Backward DP: exact V^pi, or V* when policy=None; no sampled rollouts."""
    values = np.zeros(N_STATES)
    for t in reversed(range(HORIZON)):
        states = np.arange(t * 10, (t + 1) * 10)
        q = REWARD[states] + values[NEXT[states]]
        values[states] = q.max(1) if policy is None else (q * policy[states]).sum(1)
    return values


def greedy(q):
    return np.eye(2)[q.argmax(1)]


def random_episode(rng):
    state = STARTS[int(rng.integers(2))]
    episode = []
    for _ in range(HORIZON):
        action = int(rng.integers(2))
        next_state, reward, done = transition(state, action)
        episode.append((state, action, reward, next_state, done))
        state = next_state
    return episode


def prediction(seed=7, episodes=600):
    """MC/TD evaluate the SAME fixed random policy using the SAME episodes.

    RMSE is on states reached with nonzero probability from the start policy,
    with equal weighting across those states. Neither predictor is a controller.
    """
    rng = np.random.default_rng(seed)
    truth = evaluate(np.full((N_STATES, 2), .5))
    reachable = set(STARTS)
    frontier = set(STARTS)
    for _ in range(HORIZON - 1):
        frontier = {int(NEXT[s, a]) for s in frontier for a in range(2)}
        reachable.update(frontier)
    indices = sorted(reachable)
    mc, td = np.zeros(N_STATES), np.zeros(N_STATES)
    counts = np.zeros(N_STATES)
    history = []
    started = perf_counter()
    for ep in range(episodes):
        trajectory = random_episode(rng)
        returns = 0.
        # Time-indexed states cannot repeat within one episode: first/every visit agree.
        for state, _, reward, _, _ in reversed(trajectory):
            returns += reward
            counts[state] += 1
            mc[state] += (returns - mc[state]) / counts[state]
        for state, _, reward, next_state, done in trajectory:
            target = reward + (0 if done else td[next_state])
            td[state] += .1 * (target - td[state])
        if (ep + 1) % 24 == 0:
            history.append(((ep + 1) * HORIZON,
                            np.sqrt(np.mean((mc[indices] - truth[indices]) ** 2)),
                            np.sqrt(np.mean((td[indices] - truth[indices]) ** 2))))
    return {'history': np.array(history), 'seconds': perf_counter() - started,
            'mc': mc, 'td': td, 'truth': truth}


def train_tabular(method, seed=7, episodes=600):
    """SARSA / Q-learning / first-visit MC control, with identical step budgets."""
    assert method in {'SARSA', 'Q-learning', 'MC control'}
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, 2))
    counts = np.zeros_like(q)
    history = []
    started = perf_counter()
    for ep in range(episodes):
        epsilon = max(.08, .8 * (1 - ep / episodes))
        def select(state):
            return int(rng.integers(2)) if rng.random() < epsilon else int(q[state].argmax())
        state = STARTS[int(rng.integers(2))]
        action = select(state)
        trajectory = []
        for _ in range(HORIZON):
            next_state, reward, done = transition(state, action)
            next_action = select(next_state) if not done else 0
            trajectory.append((state, action, reward))
            if method != 'MC control':
                bootstrap = q[next_state, next_action] if method == 'SARSA' else q[next_state].max()
                q[state, action] += .25 * (reward + (0 if done else bootstrap) - q[state, action])
            state, action = next_state, next_action
        if method == 'MC control':
            returns = 0.
            for state, action, reward in reversed(trajectory):
                returns += reward
                counts[state, action] += 1
                q[state, action] += (returns - q[state, action]) / counts[state, action]
        if (ep + 1) % 24 == 0:
            history.append(((ep + 1) * HORIZON, evaluate(greedy(q))[STARTS].mean()))
    return {'history': np.array(history), 'seconds': perf_counter() - started, 'policy': greedy(q)}


def network(outputs):
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, outputs))


def train_dqn(seed=7, episodes=600):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online = network(2)
    target = deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=.003)
    replay = deque(maxlen=3000)
    history, losses = [], []
    steps = 0
    started = perf_counter()
    for ep in range(episodes):
        state = STARTS[int(rng.integers(2))]
        epsilon = max(.08, .8 * (1 - ep / episodes))
        for _ in range(HORIZON):
            with torch.no_grad():
                action = int(rng.integers(2)) if rng.random() < epsilon else int(online(FEATURES[state]).argmax())
            next_state, reward, done = transition(state, action)
            replay.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1
            if len(replay) >= 64:
                batch = np.array([replay[i] for i in rng.choice(len(replay), 64, replace=False)])
                s, a, r, ns, d = [torch.tensor(batch[:, i], dtype=torch.long if i in (0, 1, 3) else torch.float32) for i in range(5)]
                with torch.no_grad():
                    targets = r + (1 - d) * target(FEATURES[ns]).max(1).values
                loss = nn.functional.smooth_l1_loss(online(FEATURES[s]).gather(1, a[:, None]).squeeze(1), targets)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5); optimizer.step()
                losses.append(float(loss.detach()))
            if steps % 120 == 0:
                target.load_state_dict(online.state_dict())
        if (ep + 1) % 24 == 0:
            with torch.no_grad():
                policy = greedy(online(FEATURES).numpy())
            history.append((steps, evaluate(policy)[STARTS].mean()))
    return {'history': np.array(history), 'seconds': perf_counter() - started, 'policy': policy, 'losses': losses}


def train_policy(method, seed=7, episodes=600):
    """REINFORCE, one-step advantage actor-critic, or PPO with GAE.

    Collect 24 complete episodes under a fixed actor per update. REINFORCE has
    no critic. AC uses one-step detached TD advantages. PPO uses lambda=.95,
    four full-batch epochs, clip=.2, and frozen rollout log probabilities.
    """
    assert method in {'REINFORCE', 'Actor-Critic', 'PPO'}
    assert episodes % 24 == 0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor, critic = network(2), network(1)
    params = list(actor.parameters()) + ([] if method == 'REINFORCE' else list(critic.parameters()))
    optimizer = torch.optim.Adam(params, lr=.01 if method != 'PPO' else .004)
    history = []
    started = perf_counter()
    for update in range(episodes // 24):
        batch = []
        for _ in range(24):
            state = STARTS[int(rng.integers(2))]
            for _ in range(HORIZON):
                with torch.no_grad():
                    distribution = Categorical(logits=actor(FEATURES[state]))
                    action = distribution.sample()
                    old_logp = distribution.log_prob(action)
                    value = critic(FEATURES[state]).squeeze()
                next_state, reward, done = transition(state, int(action))
                batch.append((state, int(action), reward, done, float(old_logp), float(value)))
                state = next_state
        s = torch.tensor([b[0] for b in batch])
        a = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch]).reshape(24, HORIZON)
        old_logp = torch.tensor([b[4] for b in batch])
        values = torch.tensor([b[5] for b in batch]).reshape(24, HORIZON)
        next_values = torch.cat([values[:, 1:], torch.zeros(24, 1)], dim=1)
        deltas = rewards + next_values - values
        returns = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1])
        if method == 'REINFORCE':
            advantages = returns.flatten()
            targets = returns.flatten()
        elif method == 'Actor-Critic':
            advantages = deltas.flatten()
            targets = (rewards + next_values).flatten()
        else:
            gae = torch.zeros_like(values)
            carry = torch.zeros(24)
            for t in reversed(range(HORIZON)):
                carry = deltas[:, t] + .95 * carry
                gae[:, t] = carry
            targets = (gae + values).flatten()
            advantages = gae.flatten()
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(4 if method == 'PPO' else 1):
            dist = Categorical(logits=actor(FEATURES[s]))
            logp = dist.log_prob(a)
            if method == 'PPO':
                ratio = (logp - old_logp).exp()
                actor_loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
            else:
                actor_loss = -(logp * advantages).mean()
            loss = actor_loss - .01 * dist.entropy().mean()
            if method != 'REINFORCE':
                loss = loss + .5 * nn.functional.mse_loss(critic(FEATURES[s]).squeeze(1), targets)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1); optimizer.step()
        with torch.no_grad():
            policy = actor(FEATURES).softmax(-1).numpy()
        history.append(((update + 1) * 24 * HORIZON, evaluate(policy)[STARTS].mean()))
    return {'history': np.array(history), 'seconds': perf_counter() - started, 'policy': policy}


def run_control(seeds=(7, 19, 31), episodes=600):
    runners = {'MC control': train_tabular, 'SARSA': train_tabular, 'Q-learning': train_tabular,
               'DQN': train_dqn, 'REINFORCE': train_policy, 'Actor-Critic': train_policy, 'PPO': train_policy}
    runs = {}
    for method, runner in runners.items():
        print(f'Training {method} ...', flush=True)
        runs[method] = [runner(seed=seed, episodes=episodes) if method == 'DQN'
                        else runner(method, seed=seed, episodes=episodes) for seed in seeds]
    return runs
