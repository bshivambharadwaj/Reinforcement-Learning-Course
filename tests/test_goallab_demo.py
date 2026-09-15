"""Export and artifact contracts for the live GoalLab browser demo. MIT."""
import json
import re
import unittest
from rl_course import goallab as goal
from rl_course.goallab_demo import build_report, render


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report(cases=4)

    def test_exported_controller_has_every_decision_state(self):
        env = goal.EvidenceMDP()
        policy = self.report['browser']['q_policy']
        for state in env.states:
            distribution = policy[','.join(map(str, state))]
            self.assertEqual(len(distribution), 2)
            self.assertAlmostEqual(sum(distribution), 1.)
            self.assertTrue(all(p >= 0 for p in distribution))

    def test_report_artifacts_match_independent_checks(self):
        self.assertEqual(len(self.report['records']), 20)
        for record in self.report['records']:
            workspace = goal.make_workspace(int(record['case'].split('-')[1]), 'test')
            artifact = goal.Briefing(**record['artifact']) if record['artifact'] else None
            self.assertEqual(goal.verify(workspace, artifact), record['evaluation'])

    def test_embedded_data_cannot_close_its_script_element(self):
        report = dict(self.report, note='</script><script>alert(1)</script>')
        html = render(report)
        payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertEqual(json.loads(payload)['note'], report['note'])
        self.assertNotIn('</script>', payload)
        self.assertNotIn('__GOALLAB_DATA__', html)


if __name__ == '__main__':
    unittest.main()
