"""Two-decision arithmetic policy with outcome verification and group updates.

This is a learned structured-output policy, not a pretrained language model.
It samples an intermediate sum and a final sum. The final head sees the
sampled intermediate decision, so the trajectory really is autoregressive.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Reasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Linear(3, 48), nn.Tanh(), nn.Linear(48, 7))
        self.final = nn.Sequential(nn.Linear(4, 48), nn.Tanh(), nn.Linear(48, 10))

    def distributions(self, problems, intermediate=None):
        first = Categorical(logits=self.first(problems / 3))
        if intermediate is None:
            intermediate = first.sample()
        features = torch.cat([problems / 3, intermediate[:, None].float() / 6], dim=1)
        final = Categorical(logits=self.final(features))
        return first, final, intermediate

    def sample(self, problems):
        first, final, intermediate = self.distributions(problems)
        answer = final.sample()
        return intermediate, answer, torch.stack([first.log_prob(intermediate), final.log_prob(answer)], dim=1)


def verify(problems, intermediate, answer, process=False):
    outcome = (answer == problems.sum(1)).float()
    if process:
        return .5 * outcome + .5 * (intermediate == problems[:, :2].sum(1)).float()
    return outcome


def evaluate(model, problems, seed=1001, samples=32):
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        expanded = problems.repeat_interleave(samples, dim=0)
        middle, answer, _ = model.sample(expanded)
        correct = verify(expanded, middle, answer)
        process = (middle == expanded[:, :2].sum(1)).float()
        return float(correct.mean()), float(process.mean())


def run(seed=7, updates=100, process=False):
    torch.manual_seed(seed)
    problems = torch.tensor([[a, b, c] for a in range(4) for b in range(4) for c in range(4)], dtype=torch.float32)
    # Disjoint input tuples, deterministic split. Both cover the same numeric range.
    test_mask = torch.tensor([(a + 2 * b + c) % 5 == 0 for a, b, c in problems.tolist()])
    train, test = problems[~test_mask], problems[test_mask]
    model = Reasoner()
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    # SFT warm start demonstrates that a verifier cannot rank groups with no successes.
    for _ in range(120):
        middle = train[:, :2].sum(1).long()
        first, final, _ = model.distributions(train, middle)
        loss = -(first.log_prob(middle) + final.log_prob(train.sum(1).long())).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    reference = deepcopy(model).requires_grad_(False)
    before = evaluate(model, test)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    for update in range(updates):
        chosen = train[torch.randint(len(train), (16,))]
        batch = chosen.repeat_interleave(8, dim=0)
        with torch.no_grad():
            middle, answer, old_logp = model.sample(batch)
            rewards = verify(batch, middle, answer, process=process).reshape(16, 8)
            advantages = ((rewards - rewards.mean(1, keepdim=True)) /
                          (rewards.std(1, keepdim=True, unbiased=False) + 1e-8)).flatten()
            ref_first, ref_final, _ = reference.distributions(batch, middle)
        for _ in range(3):
            first, final, _ = model.distributions(batch, middle)
            logp = torch.stack([first.log_prob(middle), final.log_prob(answer)], dim=1)
            ratio = (logp - old_logp).exp()
            weighted = torch.minimum(ratio * advantages[:, None], ratio.clamp(.8, 1.2) * advantages[:, None])
            # Exact action KL at the two sampled-prefix states. This is not
            # an exact expectation over every possible trajectory prefix.
            kl = (torch.distributions.kl_divergence(first, ref_first) +
                  torch.distributions.kl_divergence(final, ref_final)).mean() / 2
            loss = -weighted.mean() + .02 * kl
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if (update + 1) % 10 == 0:
            final_accuracy, step_accuracy = evaluate(model, test)
            history.append((update + 1, final_accuracy, step_accuracy,
                            float((rewards.std(1, unbiased=False) == 0).float().mean()), float(kl.detach())))
    return {'model': model, 'reference': reference, 'before': before, 'history': np.array(history),
            'train': train, 'test': test}
