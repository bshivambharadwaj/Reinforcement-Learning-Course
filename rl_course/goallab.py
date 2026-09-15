"""GoalLab-v1: a generated evidence workspace and a bounded decision abstraction.

Code: MIT. A source is usable only when it is an approved measurement for the
requested period. Every case has exactly one such source; its row sum is truth.
This controlled contract is intentionally narrower than arbitrary documents.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
import json
from pathlib import Path
import random

VERSION = 'GoalLab-v1'


@dataclass(frozen=True)
class Source:
    source_id: str
    status: str
    kind: str
    period: str
    updated: int
    rows: tuple[int, ...]

    @property
    def value(self):
        return sum(self.rows)


@dataclass(frozen=True)
class Workspace:
    case_id: str
    period: str
    sources: tuple[Source, ...]

    def search(self):
        # Metadata does not disclose approval, kind, or numerical content.
        return [{'index': i, 'source_id': s.source_id, 'updated': s.updated}
                for i, s in enumerate(self.sources)]


@dataclass(frozen=True)
class Briefing:
    value: int | None
    citation: str | None
    period: str
    abstained: bool = False


def usable(source, period):
    return source.status == 'approved' and source.kind == 'measurement' and source.period == period


def make_workspace(seed=7, split='train', sources=2):
    """Deterministic, disjoint IDs/periods/ranges; one approved measurement.

    Train rows are 0..3 and held-out rows are 4..7. This tests new values under
    the same authority schema, not generalization to arbitrary business rules.
    """
    if split not in {'train', 'test'} or sources < 2:
        raise ValueError('Use train/test and at least two sources')
    rng = random.Random(seed)
    period = '2026-01' if split == 'train' else '2026-02'
    low = 0 if split == 'train' else 4
    rows = tuple(rng.randrange(low, low + 4) for _ in range(2))
    approved = rng.randrange(sources)
    docs = []
    for i in range(sources):
        valid = i == approved
        # Wrong candidates cannot accidentally equal the ground-truth total.
        doc_rows = rows if valid else (rows[0] + 10 + i, rows[1])
        docs.append(Source(f'{split}-{seed}-source-{i}', 'approved' if valid else 'draft',
                           'measurement' if valid or i % 2 else 'forecast', period,
                           1 if valid else 2 + i, doc_rows))
    return Workspace(f'{split}-{seed}', period, tuple(docs))


def verify(workspace, briefing):
    """Independent final check. Never a search action or a learned scorer."""
    if briefing is None or briefing.abstained:
        return {'success': False, 'supported': False, 'value_correct': False}
    truth = next(s for s in workspace.sources if usable(s, workspace.period))
    source = next((s for s in workspace.sources if s.source_id == briefing.citation), None)
    supported = source is not None and usable(source, workspace.period) and source.value == briefing.value
    correct = briefing.value == truth.value and briefing.period == workspace.period
    return {'success': bool(supported and correct), 'supported': bool(supported),
            'value_correct': bool(correct)}


class Session:
    """Isolated tool state with a hard call budget and replayable event records.

    Python objects are inspectable in these lessons. This is a logical boundary,
    not an OS sandbox. External services and private accounts are not connected.
    """
    def __init__(self, workspace, budget=6):
        if not isinstance(budget, int) or budget < 0:
            raise ValueError('budget must be a nonnegative integer')
        self.workspace = workspace
        self.budget = budget
        self.events = []
        self.reads = {}
        self.artifact = None
        self.closed = False

    def execute(self, action, argument=None):
        if self.closed or len(self.events) >= self.budget:
            raise RuntimeError('Session closed or budget exhausted')
        result = {'ok': False}
        if action == 'search':
            result = {'ok': True, 'sources': self.workspace.search()}
        elif action == 'read' and isinstance(argument, int) and 0 <= argument < len(self.workspace.sources):
            source = self.workspace.sources[argument]
            self.reads[source.source_id] = source
            result = {'ok': True, 'source': asdict(source)}
        elif action == 'submit' and isinstance(argument, Briefing):
            # Enforce citation-read permission without grading correctness.
            if argument.abstained or argument.citation in self.reads:
                self.artifact = argument
                self.closed = True
                result = {'ok': True, 'submitted': True}
        elif action == 'claim_done':
            result = {'ok': True, 'message': 'Claim recorded; no artifact submitted'}
        self.events.append({'action': action,
                            'argument': asdict(argument) if isinstance(argument, Briefing) else argument,
                            'result': result})
        return result

    def snapshot(self):
        return deepcopy(self)


def baseline(workspace, budget=6, newest=False):
    session = Session(workspace, budget)
    candidates = sorted(workspace.search(), key=lambda s: -s['updated'])
    for item in candidates:
        if len(session.events) + 2 > budget:
            break
        session.execute('read', item['index'])
        source = workspace.sources[item['index']]
        if newest or usable(source, workspace.period):
            session.execute('submit', Briefing(source.value, source.source_id, workspace.period))
            break
    return session


def save_record(session, path):
    """Write learner-generated artifacts outside the tracked source tree."""
    record = {'version': VERSION, 'case': session.workspace.case_id, 'budget': session.budget,
              'events': session.events, 'artifact': asdict(session.artifact) if session.artifact else None,
              'evaluation': verify(session.workspace, session.artifact)}
    Path(path).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


class EvidenceMDP:
    """Exact four-action decision model used by stages 01–08 and 11.

    Two reads -> choose a citation -> submit (0) or claim-done (1). Actions are
    binary at each phase, not movement commands. State contains phase, read mask,
    authority inferred from a read, and selected citation. Before the first read,
    either source is authoritative with probability 1/2. Under the two-source
    contract a read resolves that uncertainty, making this a sufficient belief
    state. Numerical payloads are omitted only because tools compute row sums.
    """
    horizon = 4
    start = (0, 0, -1, -1)
    terminal = (4, 0, -1, -1)

    def __init__(self, cost=.02, naive=False):
        if cost < 0:
            raise ValueError('cost must be nonnegative')
        self.cost, self.naive = cost, naive
        states = {self.start, self.terminal}
        frontier = {self.start}
        for _ in range(self.horizon):
            nxt = {ns for s in frontier for a in (0, 1) for _, ns, _, _ in self.outcomes(s, a)}
            states.update(nxt)
            frontier = nxt - {self.terminal}
        self.states = sorted(states)
        self.index = {s: i for i, s in enumerate(self.states)}
        self.n = len(self.states)
        self.start_index, self.end_index = self.index[self.start], self.index[self.terminal]

    def outcomes(self, state, action):
        if action not in (0, 1):
            raise ValueError('binary action required')
        t, mask, authority, citation = state
        if t == 4:
            return [(1., self.terminal, 0., True)]
        if t < 2:
            roles = (0, 1) if authority == -1 else (authority,)
            return [(1 / len(roles), (t + 1, mask | (1 << action), role, -1), -self.cost, False)
                    for role in roles]
        if t == 2:
            return [(1., (3, mask, authority, action), -self.cost, False)]
        success = action == 0 and citation == authority and bool(mask & (1 << citation))
        bonus = 1.5 if self.naive and action == 1 else float(success)
        return [(1., self.terminal, bonus - self.cost, True)]

    def step(self, index, action, rng):
        outcomes = self.outcomes(self.states[index], action)
        p, state, reward, done = outcomes[int(rng.choice(len(outcomes), p=[x[0] for x in outcomes]))]
        return self.index[state], reward, done

    def features(self):
        import numpy as np
        return np.array([[t / 4, bool(mask & 1), bool(mask & 2), authority == 0,
                          authority == 1, citation == 0, citation == 1]
                         for t, mask, authority, citation in self.states], dtype='float32')

    def values(self, policy=None):
        import numpy as np
        values = np.zeros(self.n)
        q = np.zeros((self.n, 2))
        for i in reversed(range(self.n)):
            if i == self.end_index:
                continue
            for a in (0, 1):
                q[i, a] = sum(p * (r + (0 if done else values[self.index[s]]))
                              for p, s, r, done in self.outcomes(self.states[i], a))
            values[i] = q[i].max() if policy is None else q[i] @ policy[i]
        return values, q

    def deploy(self, policy, workspace, seed=1001):
        """Execute a phase policy on actual GoalLab documents and produce a briefing."""
        import numpy as np
        if len(workspace.sources) != 2:
            raise ValueError('This abstraction supports exactly two sources')
        rng = np.random.default_rng(seed)
        session = Session(workspace, budget=4)
        state = self.start
        citation = None
        for t in range(4):
            action = int(rng.choice(2, p=policy[self.index[state]]))
            _, mask, authority, _ = state
            if t < 2:
                session.execute('read', action)
                source = workspace.sources[action]
                authority = action if usable(source, workspace.period) else 1 - action
                state = (t + 1, mask | (1 << action), authority, -1)
            elif t == 2:
                citation = action
                # Citation selection is an internal decision with one work unit.
                session.events.append({'action': 'choose_citation', 'argument': action, 'result': {'ok': True}})
                state = (3, mask, authority, action)
            elif action == 0:
                source = workspace.sources[citation]
                # Unread content is not available: the gateway rejects its citation.
                visible = session.reads.get(source.source_id)
                session.execute('submit', Briefing(visible.value if visible else None,
                                                    source.source_id, workspace.period))
            else:
                session.execute('claim_done')
        return session
