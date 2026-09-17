"""Evidence boundaries, fair search accounting, splits, and stopping. MIT."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import json
import re
import numpy as np
import torch
from rl_course import goallab_benchmark as g
from rl_course import goallab_training as t
from rl_course.goallab_benchmark_demo import render


class BenchmarkTests(unittest.TestCase):
    def test_generator_disjoint_reproducible_and_all_roles(self):
        ids = set()
        for split in g.SPLITS:
            for i in range(20):
                c = g.make_case(i, split, i%5+1)
                self.assertEqual(c, g.make_case(i, split, i%5+1))
                self.assertNotIn(c.id, ids); ids.add(c.id)
                self.assertEqual(len({d.role for d in c.documents}), 6)
                self.assertEqual(next(d.candidate for d in c.documents if d.role == 'correct'), c.truth)

    def test_truth_and_roles_not_in_policy_observation(self):
        c = g.make_case()
        a = g.Investigation(c).observation()
        b = g.Investigation(replace(c, truth=(c.truth+1)%3)).observation()
        self.assertEqual(a, b)
        self.assertNotIn('truth', a)
        self.assertTrue(all('role' not in s for s in a['sources']))
        a['sources'][0]['opened'] = True
        self.assertNotIn('opened', g.Investigation(c).observation()['sources'][0])

    def test_unread_citation_cannot_pass(self):
        c = g.make_case()
        obs = g.Investigation(c).observation()
        artifact = {'answer': c.truth, 'confidence': .95,
                    'citations': [next(i for i,d in enumerate(c.documents) if d.role=='correct')]}
        result = g.verify(c, artifact, obs)
        self.assertTrue(result['accuracy'])
        self.assertFalse(result['success'])

    def test_dependency_must_be_inspected(self):
        c = g.make_case(level=4)
        i = next(i for i,d in enumerate(c.documents) if d.role=='correct')
        env = g.Investigation(c)
        env.step(g.META+i)
        env.step((g.QUERY if c.documents[i].kind=='table' else g.OPEN)+i)
        self.assertEqual(g.supporting_indices(env.observation(),c.truth), [])
        env.step(g.META+c.documents[i].dependency)
        self.assertEqual(g.supporting_indices(env.observation(),c.truth), [i])

    def test_paid_authority_and_stop_costs(self):
        env = g.Investigation(g.make_case(),7)
        env.step(g.AUTHORITY)
        env.step(g.STOP+3*env.visible['authority']+2)
        self.assertEqual(env.spent,7)
        self.assertTrue(g.verify(env.case,env.artifact,env.observation())['success'])
        with self.assertRaises(ValueError): env.step(g.SEARCH)
        with self.assertRaises(ValueError): g.Investigation(g.make_case(),5).step(g.AUTHORITY)

    def test_exhaustion_is_not_success_and_no_free_retries(self):
        env = g.Investigation(g.make_case(),2)
        env.step(g.OPEN); env.step(g.OPEN)
        self.assertEqual(env.spent,2)
        self.assertTrue(env.done)
        self.assertFalse(g.verify(env.case,env.artifact,env.observation())['success'])

    def test_frozen_inference_never_calls_final_evaluator(self):
        torch.manual_seed(7)
        model, verifier = t.Controller().eval(), t.Verifier().eval()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(g,'verify',side_effect=AssertionError('oracle leakage')):
            for method in ['Greedy','DQN','PPO','PPO + Verifier','RL + Test-Time Search']:
                env = t.investigate(g.make_case(3,'GoalLab-Test',5),method,12,model,verifier)
                self.assertLessEqual(env.spent,12)
                self.assertEqual(env.spent,sum(e['cost'] for e in env.events))
                self.assertTrue(env.done)
        for k,v in before.items(): self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_observed_support_does_not_depend_on_timestamp(self):
        c = g.make_case(3,'GoalLab-Challenge',5)
        env = g.Investigation(c,16)
        while not env.done: env.step(g.greedy_action(env.observation()))
        self.assertTrue(g.verify(c,env.artifact,env.observation())['success'])

    def test_metrics_count_failed_runs_and_abstentions(self):
        good = g.Investigation(g.make_case(),7)
        good.step(g.AUTHORITY);good.step(g.STOP+3*good.visible['authority']+2)
        failed = g.Investigation(g.make_case(),7);failed.step(g.ABSTAIN)
        row = t.aggregate([t.record(good,'rule',7),t.record(failed,'rule',7)])[0]
        self.assertEqual(row['success'],.5)
        self.assertEqual(row['answer_coverage'],.5)
        self.assertEqual(row['average_cost'],4.)
        self.assertAlmostEqual(row['brier'],.05**2)

    def test_potential_shaping_telescopes_on_completed_episode(self):
        env = g.Investigation(g.make_case(), 7)
        before = env.observation()
        delta = 0.
        for step, action in enumerate([g.AUTHORITY, g.STOP+3*env.case.truth+2]):
            env.step(action)
            delta += .99**step*(t.shaped_reward(env,action,before)-g.training_reward(env,action))
            before = env.observation()
        self.assertAlmostEqual(delta,0.)
        self.assertEqual(t.potential(before),0.)

    def test_demo_json_cannot_escape_script(self):
        report={'summary':[], 'records':[], 'config':{'seeds':[7],'note':'</script><script>evil()</script>'}}
        page=render(report)
        payload=re.search(r'<script id="data" type="application/json">(.*?)</script>',page,re.S).group(1)
        self.assertEqual(json.loads(payload)['config']['note'],report['config']['note'])
        self.assertNotIn('</script>',payload)


if __name__ == '__main__': unittest.main()
