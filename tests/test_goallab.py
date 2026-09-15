"""Behavioral contracts for the shared GoalLab stages. MIT."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from rl_course import goallab as goal
from rl_course import goallab_search as search


class WorkspaceTests(unittest.TestCase):
    def test_reproducible_split_and_authority(self):
        for i in range(30):
            a, b = goal.make_workspace(i), goal.make_workspace(i, 'test')
            self.assertEqual(a, goal.make_workspace(i))
            self.assertNotEqual(a.case_id, b.case_id)
            self.assertEqual(sum(goal.usable(s, a.period) for s in a.sources), 1)
            self.assertTrue(goal.verify(a, goal.baseline(a).artifact)['success'])
            self.assertFalse(goal.verify(a, goal.baseline(a, newest=True).artifact)['success'])

    def test_citation_period_value_and_abstention(self):
        w = goal.make_workspace()
        a = goal.baseline(w).artifact
        for bad in [replace(a, value=a.value+1), replace(a, citation='absent'),
                    replace(a, period='wrong'), replace(a, abstained=True), None]:
            self.assertFalse(goal.verify(w, bad)['success'])

    def test_gateway_rejects_unread_and_respects_budget(self):
        w = goal.make_workspace()
        s = goal.Session(w, 1)
        a = goal.baseline(w).artifact
        self.assertFalse(s.execute('submit', a)['ok'])
        with self.assertRaises(RuntimeError):
            s.execute('read', 0)
        self.assertIsNone(s.artifact)

    def test_snapshots_do_not_share_mutation(self):
        a = goal.Session(goal.make_workspace())
        b = a.snapshot()
        b.execute('read', 0)
        self.assertEqual(a.reads, {})
        self.assertEqual(a.events, [])

    def test_bellman_and_deployment_agree(self):
        env = goal.EvidenceMDP()
        values, q = env.values()
        self.assertAlmostEqual(values[env.start_index], .92)
        policy = np.eye(2)[q.argmax(1)]
        for i in range(50):
            w = goal.make_workspace(i, 'test')
            session = env.deploy(policy, w)
            self.assertTrue(goal.verify(w, session.artifact)['success'])
        for state in env.states:
            for action in (0, 1):
                outcomes = env.outcomes(state, action)
                self.assertAlmostEqual(sum(o[0] for o in outcomes), 1.)

    def test_naive_reward_does_not_complete_a_briefing(self):
        env = goal.EvidenceMDP(naive=True)
        values, q = env.values()
        self.assertAlmostEqual(values[env.start_index], 1.42)
        session = env.deploy(np.eye(2)[q.argmax(1)], goal.make_workspace())
        self.assertIsNone(session.artifact)


class SearchTests(unittest.TestCase):
    def test_hard_budgets_and_accounting(self):
        w = goal.make_workspace()
        for method in search.METHODS:
            for budget in (0, 1, 3, 6, 12, 24, 48):
                for cost in (1, 3):
                    r = search.search(w, method, budget, score_cost=cost)
                    self.assertLessEqual(r['spent'], budget)
                    self.assertEqual(r['spent'], r['proposals'] + cost * r['score_calls'])

    def test_candidate_matching_and_scorer_ablation(self):
        w = goal.make_workspace()
        vote = search.search(w, 'self_consistency', 12, 7)
        rank = search.search(w, 'best_of_n', 16, 7)
        misleading = search.search(w, 'best_of_n', 16, 7, signal='misleading')
        self.assertEqual(vote['candidates'], rank['candidates'])
        self.assertEqual(rank['candidates'], misleading['candidates'])

    def test_final_evaluator_unavailable_to_search(self):
        with patch.object(search, 'verify', side_effect=AssertionError('leak')):
            for method in search.METHODS:
                search.search(goal.make_workspace(), method, 24)

    def test_zero_budget_abstains(self):
        for method in search.METHODS:
            self.assertIsNone(search.search(goal.make_workspace(), method, 0)['artifact'])

    def test_proposal_freezing_and_separate_student(self):
        r = search.distill(cases=20)
        self.assertEqual(r['teacher'], search.FrozenProposer())
        self.assertIsNot(r['teacher'], r['student'])
        with self.assertRaises(AttributeError):
            r['teacher'].exact_probability = 0

    def test_benchmark_reproduces(self):
        a = search.benchmark(cases=4, budgets=(6,), seeds=(7,))
        self.assertEqual(a, search.benchmark(cases=4, budgets=(6,), seeds=(7,)))


if __name__ == '__main__':
    unittest.main()
