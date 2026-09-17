"""Small GoalLab-v2 controllers, train-only verifier, and equal-budget evaluation. MIT."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from . import goallab_benchmark as g


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(g.FEATURES, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, g.ACTION_COUNT)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.body(x)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class Verifier(nn.Module):
    """Predict answer correctness + observed support; labels come from Train only."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(g.FEATURES+3, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tensors(obs):
    return torch.tensor(g.features(obs)), torch.tensor(g.mask(obs))


@torch.no_grad()
def policy_action(model, obs, stochastic=False):
    x, valid = tensors(obs)
    logits, _ = model(x)
    logits = logits.masked_fill(~valid, -1e9)
    return int(Categorical(logits=logits).sample()) if stochastic else int(logits.argmax())


def training_case(rng, budget=None):
    return g.Investigation(g.make_case(rng.randrange(1_000_000), level=rng.randrange(1, 6)),
                           budget or rng.choice([4, 6, 8, 12]))


def potential(obs):
    """Dense progress potential from observed evidence, never the answer key."""
    if obs['done']:
        return 0.
    inspected = sum('metadata' in s for s in obs['sources'])
    eligible = any(s.get('metadata', {}).get('certified') and
                   s.get('metadata', {}).get('current') and
                   s.get('metadata', {}).get('relevant') for s in obs['sources'])
    support = any(g.supporting_indices(obs, k) for k in range(3))
    return .08*inspected+.4*eligible+.8*support


def shaped_reward(env, action, before):
    return g.training_reward(env, action)+.99*potential(env.observation())-potential(before)


def train_dqn(seed=7, episodes=1200):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Controller()
    target = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    replay = deque(maxlen=12000)
    steps, history = 0, []
    for episode in range(episodes):
        env = training_case(rng)
        total = 0.
        while not env.done:
            obs = env.observation()
            valid = np.flatnonzero(g.mask(obs))
            epsilon = max(.08, 1-episode/max(1, episodes*.8))
            a = int(rng.choice(valid)) if rng.random() < epsilon else policy_action(model, obs)
            ns = env.step(a)
            reward = shaped_reward(env, a, obs)
            replay.append((g.features(obs), a, reward, g.features(ns), env.done, g.mask(ns)))
            steps += 1
            total += reward
            if len(replay) >= 64:
                batch = rng.sample(list(replay), 64)
                x, actions, rewards, nxt, done, masks = zip(*batch)
                x, nxt = torch.tensor(np.array(x)), torch.tensor(np.array(nxt))
                with torch.no_grad():
                    next_actions = model(nxt)[0].masked_fill(~torch.tensor(np.array(masks)), -1e9).argmax(1)
                    boot = target(nxt)[0].gather(1, next_actions[:, None]).squeeze(1)
                    y = torch.tensor(rewards, dtype=torch.float32)+.99*boot*(1-torch.tensor(done, dtype=torch.float32))
                estimate = model(x)[0].gather(1, torch.tensor(actions)[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(estimate, y)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            if steps % 150 == 0:
                target.load_state_dict(model.state_dict())
        if episode % 50 == 0:
            history.append({'episode': episode, 'actions': steps, 'return': total})
    return model.eval(), {'algorithm': 'Double DQN', 'episodes': episodes, 'actions': steps, 'history': history}


def train_ppo(seed=7, episodes=1200, group_relative=False):
    """Clipped PPO with GAE; optional same-case, group-relative trajectory updates.

    The latter is a small discrete-policy GRPO-style exercise, not LLM GRPO.
    Neither routine uses demonstrations or evaluation split labels.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Controller()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history, steps = [], 0
    for first in range(0, episodes, 24):
        batch, trajectory_returns = [], []
        shared = None
        for j in range(min(24, episodes-first)):
            if not group_relative or j % 6 == 0:
                shared = training_case(rng)
            env = deepcopy(shared)
            trajectory = []
            while not env.done:
                obs = env.observation()
                x, valid = tensors(obs)
                with torch.no_grad():
                    logits, value = model(x)
                    dist = Categorical(logits=logits.masked_fill(~valid, -1e9))
                    action = dist.sample()
                    old_logp = dist.log_prob(action)
                env.step(int(action))
                reward = shaped_reward(env, int(action), obs)
                trajectory.append([x, valid, action, old_logp, float(value), reward])
                steps += 1
            gae, next_value = 0., 0.
            rows = []
            for x, valid, a, old, v, r in reversed(trajectory):
                gae = r+.99*next_value-v+.99*.95*gae
                rows.append([x, valid, a, old, gae, gae+v])
                next_value = v
            total = sum(row[-1] for row in trajectory)
            trajectory_returns.append(total)
            batch.append(rows)
        if group_relative:
            for start in range(0, len(batch), 6):
                returns = np.array(trajectory_returns[start:start+6])
                advantages = (returns-returns.mean())/(returns.std()+1e-6)
                for rows, adv in zip(batch[start:start+6], advantages):
                    for row in rows:
                        row[4] = float(adv)
        rows = [row for trajectory in batch for row in trajectory]
        x, valid, actions, old, advantages, returns = zip(*rows)
        x, valid = torch.stack(x), torch.stack(valid)
        actions, old = torch.stack(actions), torch.stack(old)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-6)
        returns = torch.tensor(returns, dtype=torch.float32)
        for _ in range(4):
            logits, value = model(x)
            dist = Categorical(logits=logits.masked_fill(~valid, -1e9))
            logp = dist.log_prob(actions)
            ratio = (logp-old).exp()
            loss = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages).mean()
            loss += .5*(value-returns).square().mean()-.02*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            if float((old-logp).mean().detach()) > .04:
                break
        history.append({'episode': first+len(batch), 'actions': steps,
                        'return': float(np.mean(trajectory_returns))})
    return model.eval(), {'algorithm': 'group-relative clipped trajectories' if group_relative else 'PPO + GAE',
                          'episodes': episodes, 'actions': steps, 'history': history}


def train_verifier(seed=7, cases=500, epochs=30):
    torch.manual_seed(seed)
    rng = random.Random(seed+43)
    examples, labels = [], []
    for _ in range(cases):
        env = training_case(rng)
        while not env.done:
            obs = env.observation()
            for k in range(3):
                artifact = {'answer': k, 'confidence': .75, 'citations': g.supporting_indices(obs, k)}
                examples.append(np.concatenate([g.features(obs), np.eye(3)[k]]))
                labels.append(g.verify(env.case, artifact, obs)['success'])
            a = g.greedy_action(obs) if rng.random() < .65 else int(rng.choice(np.flatnonzero(g.mask(obs))))
            env.step(a)
    x = torch.tensor(np.array(examples), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    model = Verifier()
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(epochs):
        for indices in torch.randperm(len(y)).split(256):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval(), {'training_cases': cases, 'examples': len(labels), 'epochs': epochs,
                          'positive_fraction': float(y.mean())}


@torch.no_grad()
def scores(verifier, obs):
    x = torch.tensor(np.array([np.concatenate([g.features(obs), np.eye(3)[k]]) for k in range(3)]), dtype=torch.float32)
    return verifier(x).sigmoid().numpy()


def verifier_action(model, verifier, obs):
    values = scores(verifier, obs)
    k = int(values.argmax())
    if values[k] >= .65 or obs['remaining'] == 1:
        confidence_bin = int(np.abs(np.array(g.CONFIDENCES)-values[k]).argmin())
        return g.STOP+3*k+confidence_bin
    a = policy_action(model, obs)
    if a >= g.STOP:
        x, valid = tensors(obs)
        valid[g.STOP:] = False
        if valid.any():
            with torch.no_grad():
                return int(model(x)[0].masked_fill(~valid, -1e9).argmax())
        return g.ABSTAIN
    return a


def investigate(case, method='Greedy', budget=8, model=None, verifier=None):
    env = g.Investigation(case, budget)
    while not env.done:
        obs = env.observation()
        if method == 'Greedy':
            action = g.greedy_action(obs)
        elif method == 'PPO + Verifier':
            if env.charge_search():
                action = verifier_action(model, verifier, env.observation())
            else:
                action = g.greedy_action(obs)
        elif method == 'RL + Test-Time Search' and obs['remaining'] >= 5:
            env = search_step(env, model, verifier)
            continue
        elif method == 'RL + Test-Time Search':
            if env.charge_search():
                action = verifier_action(model, verifier, env.observation())
            else:
                action = g.greedy_action(obs)
        else:
            action = policy_action(model, obs)
        env.step(action)
    return env


def search_step(env, model, verifier):
    """Expand two tool trajectories with a shared ledger; discard no costs.

    Branches call real simulated tools, never the answer checker. Every tool
    response and verifier query is charged, including discarded branches. A
    one-unit reserve allows a final STOP. Search depth is at most two.
    """
    base = deepcopy(env)
    spent, calls = env.spent, env.calls
    ledger = list(env.events)
    frontier, candidates = [base], []
    for depth in range(2):
        next_frontier = []
        for node in frontier:
            obs = node.observation()
            with torch.no_grad():
                x, valid = tensors(obs)
                logits = model(x)[0].masked_fill(~valid, -1e9)
                choices = logits.argsort(descending=True).tolist()
            # Include the evidence-rule action as an explicit proposal baseline.
            choices = list(dict.fromkeys([g.greedy_action(obs)] + choices))[:2]
            for action in choices:
                expense = g.cost(action)+1  # tool + learned-verifier query
                if spent+expense+1 > env.budget:
                    continue
                branch = deepcopy(node)
                branch.spent = spent
                branch.calls = calls
                branch.done = False
                branch.step(action)
                spent += expense
                calls += 1
                branch.spent = spent
                event = dict(branch.events[-1], branch=f'{depth+1}.{len(candidates)+1}', cost=expense)
                event['observation'] = branch.observation()
                ledger.append(event)
                obs2 = branch.observation()
                probability = scores(verifier, obs2)
                if branch.artifact and branch.artifact['answer'] is not None:
                    score = float(probability[branch.artifact['answer']])
                else:
                    with torch.no_grad():
                        value = float(model(torch.tensor(g.features(obs2)))[1])
                    score = float(probability.max())+.05*np.tanh(value)
                candidates.append((score, branch))
                if not branch.done:
                    next_frontier.append((score, branch))
        frontier = [row[1] for row in sorted(next_frontier, key=lambda row: row[0], reverse=True)[:1]]
    if not candidates:
        if env.charge_search():
            env.step(verifier_action(model, verifier, env.observation()))
        else:
            env.step(g.greedy_action(env.observation()))
        return env
    _, best = max(candidates, key=lambda row: row[0])
    best.spent, best.calls, best.events = spent, calls, ledger
    best.done = best.artifact is not None
    best.events.append({'action': 'select search branch', 'cost': 0,
                        'observation': best.observation(), 'artifact': best.artifact})
    return best


def record(env, method, seed):
    obs = env.observation()
    evaluation = g.verify(env.case, env.artifact, obs)
    return {'case': env.case.id, 'split': env.case.split, 'level': env.case.level,
            'seed': seed, 'method': method, 'budget': env.budget, 'cost': env.spent,
            'tool_calls': env.calls, 'artifact': env.artifact, 'evaluation': evaluation,
            'events': env.events, 'initial': g.Investigation(env.case, env.budget).observation(),
            'final_observation': obs}


def aggregate(records):
    rows = []
    for key in sorted({(r['split'], r['budget'], r['method']) for r in records}):
        subset = [r for r in records if (r['split'], r['budget'], r['method']) == key]
        answered = [r for r in subset if r['artifact'] and r['artifact']['answer'] is not None]
        row = {'split': key[0], 'budget': key[1], 'method': key[2], 'runs': len(subset)}
        for metric in ['success', 'accuracy', 'evidence_precision', 'evidence_recall', 'unsupported']:
            row[metric] = float(np.mean([r['evaluation'][metric] for r in subset]))
        row['average_cost'] = float(np.mean([r['cost'] for r in subset]))
        row['average_tool_calls'] = float(np.mean([r['tool_calls'] for r in subset]))
        row['answer_coverage'] = len(answered)/len(subset)
        row['brier'] = float(np.mean([r['evaluation']['brier'] for r in answered])) if answered else None
        ece = 0.
        for low, high in [(0,.6),(.6,.85),(.85,1.01)]:
            bin_rows = [r for r in answered if low <= r['artifact']['confidence'] < high]
            if bin_rows:
                ece += len(bin_rows)/len(answered)*abs(np.mean([r['artifact']['confidence'] for r in bin_rows])-np.mean([r['evaluation']['success'] for r in bin_rows]))
        row['ece'] = float(ece) if answered else None
        row['seed_success'] = {str(seed): float(np.mean([r['evaluation']['success'] for r in subset if r['seed'] == seed])) for seed in sorted({r['seed'] for r in subset})}
        rows.append(row)
    return rows


def run_benchmark(output, episodes=1200, cases=40, seeds=(7,19,31), budgets=(4,8,12)):
    if episodes < 1 or cases < 1 or not seeds or any(b < 1 for b in budgets):
        raise ValueError('positive episodes, cases, budgets and at least one seed required')
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records, training = [], []
    started = time.time()
    for seed in seeds:
        print(f'Training seed {seed}: DQN / PPO / verifier', flush=True)
        dqn, dqn_info = train_dqn(seed, episodes)
        ppo, ppo_info = train_ppo(seed, episodes)
        verifier, verifier_info = train_verifier(seed, max(100, episodes//2))
        torch.save({'version': g.VERSION, 'dqn': dqn.state_dict(), 'ppo': ppo.state_dict(),
                    'verifier': verifier.state_dict()}, output/f'controllers-{seed}.pt')
        training.append({'seed': seed, 'dqn': dqn_info, 'ppo': ppo_info, 'verifier': verifier_info,
                         'controller_parameters': sum(p.numel() for p in ppo.parameters()),
                         'verifier_parameters': sum(p.numel() for p in verifier.parameters())})
        for split in g.SPLITS[1:]:
            for budget in budgets:
                for i in range(cases):
                    case = g.make_case(i, split, i%5+1)
                    for method in ['Greedy', 'DQN', 'PPO', 'PPO + Verifier', 'RL + Test-Time Search']:
                        model = dqn if method == 'DQN' else ppo
                        env = investigate(case, method, budget, model, verifier)
                        records.append(record(env, method, seed))
        print(f'Completed seed {seed}', flush=True)
    report = {'version': g.VERSION, 'config': {'episodes': episodes, 'cases_per_split': cases,
              'seeds': list(seeds), 'budgets': list(budgets)}, 'training': training,
              'elapsed_seconds': time.time()-started, 'torch_version': torch.__version__,
              'summary': aggregate(records), 'records': records}
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=1200)
    p.add_argument('--cases', type=int, default=40)
    p.add_argument('--seeds', type=int, nargs='+', default=[7,19,31])
    p.add_argument('--budgets', type=int, nargs='+', default=[4,8,12])
    a = p.parse_args()
    report = run_benchmark(a.output, a.episodes, a.cases, a.seeds, a.budgets)
    print(json.dumps(report['summary'], indent=2))
