"""Budgeted inference experiments with a frozen, transparent proposal policy.

MIT licensed. This is a controlled arithmetic simulator, not an LLM benchmark.
The policy proposes locally computed sums with categorical errors. Search can
query a declared process scorer; only the separate evaluator judges task truth.
One proposed step costs one unit, and a score query costs ``score_cost`` units.
These are pedagogical accounting units, not FLOPs, tokens, or measured latency.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Task:
    """Add operands left to right, retaining every claimed intermediate sum."""

    operands: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, 'operands', tuple(self.operands))
        if len(self.operands) < 2 or not all(type(x) is int for x in self.operands):
            raise ValueError('A task needs at least two integer operands.')

    @property
    def depth(self) -> int:
        return len(self.operands) - 1


@dataclass(frozen=True)
class Trace:
    values: tuple[int, ...] = ()
    log_probability: float = 0.0


@dataclass(frozen=True)
class FrozenPolicy:
    """Immutable error probabilities for offsets (-1, 0, +1).

    Defaults intentionally make +1 the most probable local proposal. This
    constructs a systematic-error counterexample to majority/likelihood ranking.
    The exact local sum is built into this toy proposal interface, so its task
    generalization is not evidence that a model learned arithmetic.
    """

    probabilities: tuple[float, float, float] = (0.10, 0.40, 0.50)

    def __post_init__(self):
        object.__setattr__(self, 'probabilities', tuple(self.probabilities))
        if (len(self.probabilities) != 3 or
                any(not math.isfinite(p) or p <= 0 for p in self.probabilities) or
                not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-10)):
            raise ValueError('Provide three positive finite probabilities summing to one.')

    def propose(self, task: Task, prefix: Trace, rng: random.Random) -> Trace:
        """Sample exactly one action; callers charge the proposal before use."""
        t = len(prefix.values)
        if t >= task.depth:
            raise ValueError('Cannot extend a completed trace.')
        previous = prefix.values[-1] if t else task.operands[0]
        index = rng.choices(range(3), weights=self.probabilities, k=1)[0]
        value = previous + task.operands[t + 1] + index - 1
        return Trace(prefix.values + (value,), prefix.log_probability + math.log(self.probabilities[index]))


def local_errors(task: Task, trace: Trace) -> tuple[int, ...]:
    """Errors relative to the claimed predecessor, not an oracle future value."""
    if len(trace.values) > task.depth:
        raise ValueError('Trace exceeds the task horizon.')
    previous = (task.operands[0],) + trace.values[:-1]
    return tuple(value - (prev + operand) for value, prev, operand in
                 zip(trace.values, previous, task.operands[1:]))


def process_score(task: Task, trace: Trace, scorer: str = 'process') -> float:
    """Return an explicitly chosen inference signal, never a hidden outcome.

    ``process`` is an exact local-arithmetic checker (an intentionally strong
    oracle for this small domain, not a learned process reward model).
    ``misleading`` rewards +1 errors and constructs evaluator exploitation.
    Scores are fractions over observed steps, not continuation probabilities.
    """
    if scorer not in ('process', 'misleading'):
        raise ValueError('Unknown scorer.')
    errors = local_errors(task, trace)
    if not errors:
        return 0.0
    desired = 0 if scorer == 'process' else 1
    return sum(error == desired for error in errors) / len(errors)


def evaluate(task: Task, trace: Trace | None) -> dict[str, bool]:
    """Independent task evaluation after selection; never called by search.

    Check final answer with sum(operands), and process validity with independently
    recomputed cumulative ground truth. Opposite errors may cancel at the end.
    """
    if trace is None or len(trace.values) != task.depth:
        return {'completed': False, 'answer_correct': False, 'process_valid': False}
    total = task.operands[0]
    expected = []
    for operand in task.operands[1:]:
        total += operand
        expected.append(total)
    return {'completed': True, 'answer_correct': trace.values[-1] == sum(task.operands),
            'process_valid': trace.values == tuple(expected)}


@dataclass
class Ledger:
    limit: int
    score_cost: int = 1
    proposal_steps: int = 0
    score_calls: int = 0

    def __post_init__(self):
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError('Budget must be a nonnegative integer.')
        if type(self.score_cost) is not int or self.score_cost < 1:
            raise ValueError('Score cost must be a positive integer.')

    @property
    def spent(self) -> int:
        return self.proposal_steps + self.score_cost * self.score_calls

    def can_afford(self, units: int) -> bool:
        return self.spent + units <= self.limit


@dataclass
class Result:
    method: str
    task: Task
    selected: Trace | None
    candidates: list[Trace]
    ledger: Ledger
    events: list[dict] = field(default_factory=list)

    def metrics(self) -> dict:
        truth = evaluate(self.task, self.selected)
        return {**truth, 'candidate_oracle_success': any(evaluate(self.task, t)['answer_correct'] for t in self.candidates),
                'candidates': len(self.candidates), 'proposal_steps': self.ledger.proposal_steps,
                'score_calls': self.ledger.score_calls, 'spent': self.ledger.spent}


METHODS = ('single', 'self_consistency', 'best_of_n', 'beam', 'best_first')


def infer(task: Task, policy: FrozenPolicy = FrozenPolicy(), *, method: str = 'best_of_n',
          budget: int = 48, seed: int = 0, scorer: str = 'process', score_cost: int = 1,
          width: int = 2, branching: int = 2) -> Result:
    """Run one frozen-policy strategy within a hard, explicit work budget.

    All strategies sample from the same policy. Beam retains ``width`` prefixes
    each depth; best-first expands the highest-scoring prefix, breaking ties by
    greater depth then insertion order. Both sample ``branching`` children per
    expansion, with replacement, and charge duplicates. Completed candidates are
    selected by the declared score, never by the independent outcome evaluator.
    Self-consistency ties use the first-observed answer; within that answer, take
    the first trace. Unfinished runs abstain rather than returning a fake answer.
    """
    if method not in METHODS or scorer not in ('process', 'misleading'):
        raise ValueError('Unknown method or scorer.')
    if type(width) is not int or type(branching) is not int or width < 1 or branching < 1:
        raise ValueError('Width and branching must be positive integers.')
    ledger = Ledger(budget, score_cost)
    rng = random.Random(seed)
    candidates: list[Trace] = []
    scored: list[tuple[float, Trace]] = []
    events: list[dict] = []

    def extend(prefix: Trace, use_score: bool) -> tuple[Trace, float]:
        cost = 1 + (score_cost if use_score else 0)
        if not ledger.can_afford(cost):
            raise RuntimeError('Internal budget accounting error.')
        ledger.proposal_steps += 1
        child = policy.propose(task, prefix, rng)
        score = 0.0
        if use_score:
            ledger.score_calls += 1
            score = process_score(task, child, scorer)
        events.append({'prefix': list(prefix.values), 'values': list(child.values),
                       'score': score if use_score else None, 'spent': ledger.spent})
        return child, score

    if method in ('single', 'self_consistency', 'best_of_n'):
        per_trace = task.depth + (score_cost if method == 'best_of_n' else 0)
        while ledger.can_afford(per_trace):
            trace = Trace()
            for _ in range(task.depth):
                trace, _ = extend(trace, False)
            candidates.append(trace)
            if method == 'best_of_n':
                ledger.score_calls += 1
                score = process_score(task, trace, scorer)
                scored.append((score, trace))
                events.append({'values': list(trace.values), 'score': score, 'spent': ledger.spent})
            if method == 'single':
                break
        if method == 'self_consistency' and candidates:
            counts = Counter(t.values[-1] for t in candidates)
            answer = counts.most_common(1)[0][0]
            selected = next(t for t in candidates if t.values[-1] == answer)
        elif method == 'best_of_n':
            selected = max(scored, key=lambda pair: pair[0])[1] if scored else None
        else:
            selected = candidates[0] if candidates else None
    elif method == 'beam':
        frontier = [Trace()]
        for _ in range(task.depth):
            children = []
            for parent in frontier:
                for _ in range(branching):
                    if not ledger.can_afford(1 + score_cost):
                        break
                    child, score = extend(parent, True)
                    children.append((score, child))
                    if len(child.values) == task.depth:
                        candidates.append(child)
                        scored.append((score, child))
            frontier = [child for _, child in sorted(children, key=lambda pair: pair[0], reverse=True)[:width]]
            if not frontier:
                break
        selected = max(scored, key=lambda pair: pair[0])[1] if scored else None
    else:
        queue = [(0.0, 0, 0, Trace())]
        serial = 0
        while queue and ledger.can_afford(1 + score_cost):
            _, _, _, parent = heapq.heappop(queue)
            for _ in range(branching):
                if not ledger.can_afford(1 + score_cost):
                    break
                child, score = extend(parent, True)
                if len(child.values) == task.depth:
                    candidates.append(child)
                    scored.append((score, child))
                else:
                    serial += 1
                    heapq.heappush(queue, (-score, -len(child.values), serial, child))
        selected = max(scored, key=lambda pair: pair[0])[1] if scored else None
    return Result(method, task, selected, candidates, ledger, events)


def make_tasks(count: int = 100, *, seed: int = 2026, low: int = 10, high: int = 30,
               depth: int = 3) -> list[Task]:
    """Generate unique operand tuples; ranges can separate train/test tasks."""
    if (any(type(x) is not int for x in (count, depth, low, high)) or
            count < 1 or depth < 1 or high <= low or count > (high - low) ** (depth + 1)):
        raise ValueError('Invalid task count, range, or depth.')
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    tasks = []
    while len(tasks) < count:
        operands = tuple(rng.randrange(low, high) for _ in range(depth + 1))
        if operands not in seen:
            tasks.append(Task(operands))
            seen.add(operands)
    return tasks


def benchmark(tasks: Iterable[Task], *, budgets=(6, 12, 24, 48), seeds=(7, 19, 41),
              methods=METHODS, scorer='process', policy=FrozenPolicy(), score_cost=1) -> list[dict]:
    """Return one aggregate row per strategy, budget, and generation seed.

    Identical task/seed pairs share an RNG seed. Algorithms consume random draws
    differently; this is a paired task design, not identical trajectories.
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError('Provide at least one task.')
    rows = []
    for budget in budgets:
        for seed in seeds:
            for method in methods:
                metrics = [infer(task, policy, method=method, budget=budget,
                                 seed=seed * 1_000_003 + i, scorer=scorer,
                                 score_cost=score_cost).metrics() for i, task in enumerate(tasks)]
                rows.append({'method': method, 'budget': budget, 'seed': seed, 'scorer': scorer,
                             **{key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}})
    return rows


def distill(tasks: Iterable[Task], teacher: FrozenPolicy = FrozenPolicy(), *, budget=48,
            seed=7, scorer='process', acceptance='process') -> tuple[FrozenPolicy, dict]:
    """Fit a new categorical proposal from selected teacher traces.

    This is supervised maximum-likelihood counting with unit pseudocounts, not
    PPO/GRPO and not an LLM. ``process`` acceptance checks every step; ``outcome``
    accepts correct final answers even with compensating errors. A separate
    returned policy makes the training/inference boundary explicit.
    """
    if acceptance not in ('process', 'outcome'):
        raise ValueError('Acceptance must be process or outcome.')
    counts = [1, 1, 1]
    accepted = total = spent = 0
    for i, task in enumerate(tasks):
        total += 1
        result = infer(task, teacher, method='best_of_n', budget=budget,
                       seed=seed * 1_000_003 + i, scorer=scorer)
        spent += result.ledger.spent
        truth = evaluate(task, result.selected)
        keep = truth['process_valid'] if acceptance == 'process' else truth['answer_correct']
        if keep:
            accepted += 1
            for error in local_errors(task, result.selected):
                counts[error + 1] += 1
    student = FrozenPolicy(tuple(c / sum(counts) for c in counts))
    return student, {'tasks': total, 'accepted': accepted, 'teacher_search_units': spent,
                     'acceptance': acceptance, 'counts_with_pseudocounts': counts}
