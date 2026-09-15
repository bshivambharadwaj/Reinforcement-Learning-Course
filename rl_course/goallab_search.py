"""Frozen-policy evidence search and later student fitting for GoalLab. MIT.

Candidates are (source read, extraction offset, citation). Every expansion
represents one work unit; score queries cost separately. Reading is executed
through Session. Exact eligibility scores use visible evidence, never verify().
"""
from collections import Counter
from dataclasses import dataclass
import random
import math
import numpy as np
from .goallab import Briefing, Session, make_workspace, usable, verify

METHODS = ('single', 'self_consistency', 'best_of_n', 'beam', 'best_first')


@dataclass(frozen=True)
class FrozenProposer:
    # Prefer recency by default, even though the newest source is a draft.
    newest_probability: float = .75
    exact_probability: float = .65
    matching_citation_probability: float = .8

    def __post_init__(self):
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in
               (self.newest_probability, self.exact_probability, self.matching_citation_probability)):
            raise ValueError('probabilities must lie in [0, 1]')

    def propose(self, session, prefix, rng):
        workspace = session.workspace
        if len(workspace.sources) != 2:
            raise ValueError('GoalLab search v1 uses two sources')
        if len(prefix) == 0:
            newest = max(range(2), key=lambda i: workspace.sources[i].updated)
            return newest if rng.random() < self.newest_probability else 1 - newest
        if len(prefix) == 1:
            return 0 if rng.random() < self.exact_probability else 1
        return prefix[0] if rng.random() < self.matching_citation_probability else 1 - prefix[0]


def artifact(session, prefix):
    if len(prefix) != 3:
        return None
    source, offset, citation = prefix
    observed = session.reads[session.workspace.sources[source].source_id]
    return Briefing(observed.value + offset, session.workspace.sources[citation].source_id, session.workspace.period)


def score(session, prefix, signal='evidence'):
    if signal not in {'evidence', 'misleading'}:
        raise ValueError(signal)
    if not prefix:
        return 0.
    source = session.reads[session.workspace.sources[prefix[0]].source_id]
    checks = [float(usable(source, session.workspace.period)) if signal == 'evidence'
              else float(source.updated == max(s['updated'] for s in session.workspace.search()))]
    if len(prefix) >= 2:
        checks.append(float(prefix[1] == 0))
    if len(prefix) == 3:
        checks.append(float(prefix[2] == prefix[0]))
    return sum(checks) / len(checks)


def search(workspace, method='best_of_n', budget=24, seed=7, signal='evidence',
           policy=None, score_cost=1, width=2, branching=2):
    if method not in METHODS or signal not in {'evidence', 'misleading'}:
        raise ValueError('unknown method or scorer')
    if any(not isinstance(v, int) for v in (budget, score_cost, width, branching)) or budget < 0 or min(score_cost, width, branching) < 1:
        raise ValueError('invalid budget/search configuration')
    policy, rng = policy or FrozenProposer(), random.Random(seed)
    spent = proposals = scores = 0
    events, completed = [], []

    def expand(node, scored=True):
        nonlocal spent, proposals, scores
        prefix, session, _ = node
        if spent + 1 + (score_cost if scored else 0) > budget:
            return None
        action = policy.propose(session, prefix, rng)
        child_session = session.snapshot()
        child = prefix + (action,)
        if not prefix:
            child_session.execute('read', action)
        spent += 1; proposals += 1
        value = score(child_session, child, signal) if scored else 0.
        if scored:
            spent += score_cost; scores += 1
        events.append({'prefix': child, 'score': value if scored else None, 'spent': spent})
        return child, child_session, value

    root = ((), Session(workspace, budget=4), 0.)
    if method in {'single', 'self_consistency', 'best_of_n'}:
        per_candidate = 3 + (score_cost if method == 'best_of_n' else 0)
        while spent + per_candidate <= budget:
            node = root
            for _ in range(3):
                node = expand(node, scored=False)
            if method == 'best_of_n':
                spent += score_cost; scores += 1
                node = (node[0], node[1], score(node[1], node[0], signal))
            completed.append(node)
            if method == 'single':
                break
    else:
        frontier = [root]
        while frontier and spent + 1 + score_cost <= budget:
            if method == 'beam':
                parents, frontier = frontier, []
            else:
                # Score, then deeper prefix; stable order resolves remaining ties.
                frontier.sort(key=lambda n: (n[2], len(n[0])), reverse=True)
                parents = [frontier.pop(0)]
            children = []
            for parent in parents:
                for _ in range(branching):
                    child = expand(parent)
                    if child is None:
                        break
                    if len(child[0]) == 3:
                        completed.append(child)
                    else:
                        children.append(child)
            if method == 'beam':
                frontier = sorted(children, key=lambda n: n[2], reverse=True)[:width]
            else:
                frontier.extend(children)
    selected = None
    if completed:
        if method == 'self_consistency':
            # Vote on final numerical answer; first observed candidate wins a tie.
            counts = Counter(artifact(s, p).value for p, s, _ in completed)
            answer = counts.most_common(1)[0][0]
            selected = next(n for n in completed if artifact(n[1], n[0]).value == answer)
        else:
            selected = max(completed, key=lambda n: n[2])
    result = artifact(selected[1], selected[0]) if selected else None
    return {'artifact': result, 'prefix': selected[0] if selected else None,
            'candidates': [artifact(s, p) for p, s, _ in completed],
            'events': events, 'spent': spent, 'proposals': proposals, 'score_calls': scores,
            'session': selected[1] if selected else None}


def benchmark(cases=80, budgets=(6, 12, 24, 48), seeds=(7, 19, 31), signal='evidence',
              methods=METHODS, policy=None, score_cost=1):
    rows = []
    for budget in budgets:
        for method in methods:
            for seed in seeds:
                runs = []
                for i in range(cases):
                    workspace = make_workspace(i, 'test')
                    run = search(workspace, method, budget, seed * 100003 + i, signal, policy, score_cost)
                    # All final truth checks are post-selection and outside the search budget.
                    runs.append((verify(workspace, run['artifact'])['success'],
                                 any(verify(workspace, a)['success'] for a in run['candidates']),
                                 run['spent'], run['artifact'] is not None))
                metrics = np.mean(runs, axis=0)
                rows.append({'method': method, 'budget': budget, 'seed': seed, 'signal': signal,
                             **dict(zip(['success', 'oracle', 'spent', 'completion'], metrics.tolist()))})
    return rows


def distill(cases=160, budget=24, seed=7, acceptance='verified'):
    """Fit a new categorical student from accepted TRAINING-workspace traces.

    With no accepted examples, keep the teacher rather than inventing evidence.
    This is counts-based filtered imitation, not PPO/GRPO or LLM fine-tuning.
    """
    if acceptance not in {'verified', 'proxy'}:
        raise ValueError(acceptance)
    teacher = FrozenProposer()
    traces, cost = [], 0
    for i in range(cases):
        workspace = make_workspace(i, 'train')
        run = search(workspace, budget=budget, seed=seed * 100003 + i, policy=teacher)
        cost += run['spent']
        accept = verify(workspace, run['artifact'])['success'] if acceptance == 'verified' else run['artifact'] is not None
        if accept:
            newest = max(range(2), key=lambda j: workspace.sources[j].updated)
            source, offset, citation = run['prefix']
            traces.append((source == newest, offset == 0, citation == source))
    probabilities = (np.sum(traces, axis=0) + 1) / (len(traces) + 2) if traces else None
    student = FrozenProposer(*probabilities.tolist()) if traces else teacher
    return {'teacher': teacher, 'student': student, 'accepted': len(traces), 'teacher_units': cost}
