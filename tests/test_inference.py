"""Meaningful invariants for Part III. Run: python -m unittest discover -s tests."""
import unittest
from unittest.mock import patch

from rl_course.inference import (FrozenPolicy, METHODS, Task, Trace, distill,
                                 evaluate, infer, make_tasks, process_score)
from rl_course.inference_demo import build_report, report_html


class InferenceTests(unittest.TestCase):
    def test_input_containers_cannot_mutate_frozen_objects(self):
        numbers = [1, 2, 3]
        probabilities = [.1, .4, .5]
        task = Task(numbers)
        policy = FrozenPolicy(probabilities)
        numbers[0] = 99
        probabilities[0] = .8
        self.assertEqual(task.operands, (1, 2, 3))
        self.assertEqual(policy.probabilities, (.1, .4, .5))

    def test_cancellation_does_not_certify_process(self):
        task = Task((1, 1, 5))
        trace = Trace((3, 7))  # +1 then -1: right final answer, invalid steps.
        self.assertEqual(evaluate(task, trace), {'completed': True, 'answer_correct': True, 'process_valid': False})
        self.assertEqual(process_score(task, trace), 0)

    def test_independent_evaluator_is_never_used_by_search(self):
        with patch('rl_course.inference.evaluate', side_effect=AssertionError('Oracle leak')):
            for method in METHODS:
                infer(Task((2, 4, 6, 8)), method=method, budget=30)

    def test_budget_never_exceeded_and_frozen_policy_unchanged(self):
        policy = FrozenPolicy()
        before = policy.probabilities
        for depth in (1, 3, 5):
            for budget in (0, 1, 5, 12, 49):
                for cost in (1, 3):
                    for method in METHODS:
                        result = infer(Task(tuple(range(depth + 1))), policy, method=method,
                                       budget=budget, score_cost=cost)
                        self.assertLessEqual(result.ledger.spent, budget)
                        self.assertEqual(result.ledger.spent, result.ledger.proposal_steps + cost * result.ledger.score_calls)
                        if result.selected:
                            self.assertEqual(len(result.selected.values), depth)
                        metrics = result.metrics()
                        self.assertLessEqual(metrics['answer_correct'], metrics['candidate_oracle_success'])
        self.assertEqual(policy.probabilities, before)

    def test_zero_budget_abstains(self):
        for method in METHODS:
            result = infer(Task((1, 2)), method=method, budget=0)
            self.assertIsNone(result.selected)
            self.assertFalse(result.metrics()['completed'])

    def test_best_of_n_charges_every_completed_score(self):
        result = infer(Task((1, 2, 3, 4)), method='best_of_n', budget=22, score_cost=2)
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual(result.ledger.proposal_steps, 12)
        self.assertEqual(result.ledger.score_calls, 4)
        self.assertEqual(result.ledger.spent, 20)

    def test_common_full_candidate_stream(self):
        task = Task((1, 2, 3, 4))
        sampling = infer(task, method='self_consistency', budget=12, seed=7)
        ranking = infer(task, method='best_of_n', budget=16, seed=7)
        self.assertEqual(sampling.candidates, ranking.candidates)

    def test_wrong_scorer_selects_known_wrong_candidate(self):
        task = Task((1, 1))
        def alternating(_self, _task, prefix, rng):
            # Deterministic candidate stream for a causal selector test.
            value = next(values)
            return Trace((value,))
        for scorer, expected in [('process', 2), ('misleading', 3)]:
            values = iter([2, 3])
            with patch.object(FrozenPolicy, 'propose', alternating):
                result = infer(task, method='best_of_n', budget=4, scorer=scorer)
            self.assertEqual(result.selected.values, (expected,))
            self.assertTrue(result.metrics()['candidate_oracle_success'])

    def test_distillation_creates_a_separate_policy(self):
        teacher = FrozenPolicy()
        student, metadata = distill(make_tasks(30, low=0, high=5), teacher, budget=96)
        self.assertEqual(teacher.probabilities, (0.1, 0.4, 0.5))
        self.assertGreater(metadata['accepted'], 0)
        self.assertGreater(student.probabilities[1], teacher.probabilities[1])
        self.assertAlmostEqual(sum(student.probabilities), 1)

    def test_input_validation(self):
        for probs in ((0, .5, .5), (.1, .1, .1), (float('nan'), .5, .5)):
            with self.assertRaises(ValueError):
                FrozenPolicy(probs)
        with self.assertRaises(ValueError):
            infer(Task((1, 2)), budget=-1)
        with self.assertRaises(ValueError):
            infer(Task((1, 2)), score_cost=0)
        with self.assertRaises(ValueError):
            make_tasks(100, low=0, high=1)

    def test_tasks_are_unique_and_seeded(self):
        a = make_tasks(50)
        self.assertEqual(a, make_tasks(50))
        self.assertEqual(len(set(a)), 50)
        self.assertTrue(set(a).isdisjoint(make_tasks(50, low=0, high=10)))

    def test_reproducible_report_and_embedded_data(self):
        report = build_report(count=5, budgets=(12,), seeds=(7,))
        self.assertEqual(report, build_report(count=5, budgets=(12,), seeds=(7,)))
        self.assertEqual(len(report['rows']), 10)
        html = report_html(report)
        self.assertIn('application/json', html)
        self.assertNotIn('<script src=', html)


if __name__ == '__main__':
    unittest.main()
