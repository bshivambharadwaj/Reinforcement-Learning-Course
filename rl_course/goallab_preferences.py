"""GoalLab synthetic preferences and autoregressive citation training. MIT."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .goallab import Briefing, make_workspace, usable, verify


def evidence_features(workspaces):
    # These stages explicitly start AFTER both source reads. No unread facts enter.
    return torch.tensor([[float(usable(s, w.period)) for s in w.sources]
                         for w in workspaces], dtype=torch.float32)


def preference_lab(seed=7, steps=100):
    """Two contexts, two citations. Synthetic rules, not human annotations.

    Reward model uses pairwise logistic loss. DPO uses a fixed uniform reference.
    The model sees evidence eligibility; this isolates preference mechanics.
    """
    torch.manual_seed(seed)
    workspaces = [make_workspace(i) for i in range(40)]
    x = evidence_features(workspaces)
    chosen = x.argmax(1)
    rejected = 1 - chosen
    reward_model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=.05)
    for _ in range(steps):
        scores = reward_model(x)
        loss = -nn.functional.logsigmoid(scores.gather(1, chosen[:, None]) - scores.gather(1, rejected[:, None])).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Separate direct-preference policy; reward model is NOT used by DPO.
    policy = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(policy.weight)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.05)
    history = []
    for i in range(steps):
        logp = policy(x).log_softmax(-1)
        margin = logp.gather(1, chosen[:, None]) - logp.gather(1, rejected[:, None])
        loss = -nn.functional.logsigmoid(.5 * margin).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append(float(loss.detach()))
    test = [make_workspace(i, 'test') for i in range(100)]
    with torch.no_grad():
        tx = evidence_features(test)
        reward_accuracy = float((reward_model(tx).argmax(1) == tx.argmax(1)).float().mean())
        actions = policy(tx).argmax(1).tolist()
        verified = np.mean([verify(w, Briefing(w.sources[a].value, w.sources[a].source_id, w.period))['success']
                            for w, a in zip(test, actions)])
    return {'reward_accuracy': reward_accuracy, 'success': float(verified), 'loss': history,
            'model': policy, 'reward_model': reward_model}


class CitationPolicy(nn.Module):
    """Autoregressive citation then extraction mode: exact row sum or +1 error."""
    def __init__(self):
        super().__init__()
        self.cite = nn.Linear(2, 2)
        self.extract = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))

    def distributions(self, x, citation=None):
        first = Categorical(logits=self.cite(x))
        citation = first.sample() if citation is None else citation
        second = Categorical(logits=self.extract(torch.cat([x, nn.functional.one_hot(citation, 2).float()], 1)))
        return first, second, citation

    def sample(self, x):
        first, second, citation = self.distributions(x)
        offset = second.sample()
        return citation, offset, torch.stack([first.log_prob(citation), second.log_prob(offset)], 1)


def training_rewards(x, citation, offset, process=False):
    authority = x.gather(1, citation[:, None]).squeeze(1)
    exact = (offset == 0).float()
    # Outcome requires both. Process feedback adds an intermediate authority signal.
    return .5 * authority + .5 * authority * exact if process else authority * exact


def evaluate_policy(model, workspaces, seed=1001, samples=8):
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        x = evidence_features(workspaces).repeat_interleave(samples, 0)
        citations, offsets, _ = model.sample(x)
    results = []
    for w, c, offset in zip([w for w in workspaces for _ in range(samples)], citations.tolist(), offsets.tolist()):
        source = w.sources[c]
        results.append(verify(w, Briefing(source.value + offset, source.source_id, w.period))['success'])
    return float(np.mean(results))


def group_train(seed=7, updates=60, process=False):
    torch.manual_seed(seed)
    train = [make_workspace(i) for i in range(64)]
    test = [make_workspace(i, 'test') for i in range(64)]
    x = evidence_features(train)
    model = CitationPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    # A short SFT warm start; no assumption that all sampled groups contain a success.
    for _ in range(8):
        first, second, _ = model.distributions(x, x.argmax(1))
        loss = -(first.log_prob(x.argmax(1)) + second.log_prob(torch.zeros(len(x), dtype=torch.long))).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    reference = deepcopy(model).requires_grad_(False)
    before = evaluate_policy(model, test)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history = []
    for u in range(updates):
        batch = x[torch.randint(len(x), (16,))].repeat_interleave(8, 0)
        with torch.no_grad():
            citation, offset, old_logp = model.sample(batch)
            rewards = training_rewards(batch, citation, offset, process).reshape(16, 8)
            advantage = ((rewards - rewards.mean(1, keepdim=True)) /
                         (rewards.std(1, keepdim=True, unbiased=False) + 1e-8)).flatten()
            rf, rs, _ = reference.distributions(batch, citation)
        for _ in range(3):
            first, second, _ = model.distributions(batch, citation)
            logp = torch.stack([first.log_prob(citation), second.log_prob(offset)], 1)
            ratio = (logp - old_logp).exp()
            objective = torch.minimum(ratio * advantage[:, None], ratio.clamp(.8, 1.2) * advantage[:, None]).mean()
            kl = (torch.distributions.kl_divergence(first, rf) + torch.distributions.kl_divergence(second, rs)).mean() / 2
            loss = -objective + .02 * kl
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if (u + 1) % 10 == 0:
            history.append((u + 1, evaluate_policy(model, test),
                            float((rewards.std(1, unbiased=False) == 0).float().mean()), float(kl.detach())))
    return {'model': model, 'reference': reference, 'before': before, 'history': np.array(history)}


def language_data(split):
    """Generated source-eligibility prompts from actual GoalLab Source records."""
    examples = []
    for seed in range(16 if split == 'train' else 8):
        workspace = make_workspace(seed, split)
        for source in workspace.sources:
            chosen = 'include' if usable(source, workspace.period) else 'exclude'
            examples.append({'prompt': f'Requested period: {workspace.period}.\n'
                             f'Source {source.source_id}: status={source.status}, kind={source.kind}, '
                             f'period={source.period}, rows={list(source.rows)}.\n'
                             'Include only approved measurements for the requested period.\nDecision:',
                             'chosen': ' ' + chosen, 'rejected': ' ' + ('exclude' if chosen == 'include' else 'include')})
    return examples
