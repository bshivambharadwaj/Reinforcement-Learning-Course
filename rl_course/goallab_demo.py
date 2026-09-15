"""Run GoalLab's supplied workspace, controllers, search, and final artifacts. MIT."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from . import goallab as goal
from . import goallab_learning as learning
from . import goallab_search as search


def build_report(cases=40, seed=7):
    if cases < 1:
        raise ValueError('cases must be positive')
    controller = learning.train_tabular('Q-learning', seed, 1200)
    fitted = search.distill(160, 24, seed)
    methods = ['Newest source', 'Evidence rule', 'Q-learning', 'Frozen search', 'Student single']
    records = []
    for i in range(cases):
        w = goal.make_workspace(i, 'test')
        for method in methods:
            work = 0
            if method in methods[:2]:
                session = goal.baseline(w, newest=method == methods[0])
            elif method == 'Q-learning':
                session = controller['env'].deploy(controller['policy'], w, seed * 100003 + i)
            else:
                result = search.search(w, 'best_of_n' if method == 'Frozen search' else 'single',
                                       24 if method == 'Frozen search' else 3, seed * 100003 + i,
                                       policy=None if method == 'Frozen search' else fitted['student'])
                session = result['session'] or goal.Session(w, 4)
                work = result['spent']
                if result['artifact'] is not None:
                    session.execute('submit', result['artifact'])
            records.append({'method': method, 'case': w.case_id,
                            'sources': [asdict(s) for s in w.sources],
                            'artifact': asdict(session.artifact) if session.artifact else None,
                            'evaluation': goal.verify(w, session.artifact),
                            'committed_calls': len(session.events), 'search_units': work,
                            'events': session.events})
    summary = []
    for method in methods:
        subset = [r for r in records if r['method'] == method]
        summary.append({'method': method, 'success': float(np.mean([r['evaluation']['success'] for r in subset])),
                        'committed_calls': float(np.mean([r['committed_calls'] for r in subset])),
                        'search_units': float(np.mean([r['search_units'] for r in subset]))})
    return {'version': goal.VERSION, 'cases': cases, 'seed': seed,
            'training': {'q_learning_episodes': 1200, 'student_training_cases': 160,
                         'student_accepted': fitted['accepted'], 'teacher_data_units': fitted['teacher_units']},
            'summary': summary, 'records': records,
            'browser': {'q_policy': {','.join(map(str, s)): controller['policy'][i].tolist()
                                     for i, s in enumerate(controller['env'].states)},
                        'teacher': asdict(search.FrozenProposer()), 'student': asdict(fitted['student'])}}


def render(report):
    """Embed trained policies and fixtures in a standalone live browser engine."""
    template = Path(__file__).with_name('goallab_demo.html').read_text(encoding='utf-8')
    data = json.dumps(report).replace('<', '\\u003c')
    return template.replace('__GOALLAB_DATA__', data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True, help='HTML report destination')
    parser.add_argument('--json', type=Path, help='Optional complete run-record destination')
    args = parser.parse_args()
    report = build_report(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding='utf-8')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print('Open:', args.output.resolve())


if __name__ == '__main__':
    main()
