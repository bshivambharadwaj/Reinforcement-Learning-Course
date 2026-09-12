"""Run the Part III demo: python -m rl_course.inference_demo --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from .inference import FrozenPolicy, METHODS, benchmark, infer, make_tasks


def build_report(*, count=100, budgets=(6, 12, 24, 48), seeds=(7, 19, 41), depth=3, score_cost=1):
    tasks = make_tasks(count, depth=depth)
    policy = FrozenPolicy()
    rows = []
    traces = []
    for scorer in ('process', 'misleading'):
        rows.extend(benchmark(tasks, budgets=budgets, seeds=seeds, scorer=scorer,
                              policy=policy, score_cost=score_cost))
        for budget in budgets:
            for method in METHODS:
                result = infer(tasks[0], policy, method=method, budget=budget,
                               seed=seeds[0] * 1_000_003, scorer=scorer, score_cost=score_cost)
                traces.append({'scorer': scorer, 'budget': budget, 'method': method,
                               'operands': tasks[0].operands,
                               'selected': asdict(result.selected) if result.selected else None,
                               'metrics': result.metrics(), 'events': result.events})
    return {'configuration': {'tasks': count, 'task_seed': 2026, 'generation_seeds': list(seeds),
                               'budgets': list(budgets), 'depth': depth, 'score_cost': score_cost,
                               'policy': list(policy.probabilities), 'operand_range': [10, 30],
                               'task': 'ArithmeticTrace-v1', 'methods': list(METHODS)},
            'rows': rows, 'traces': traces}


def report_html(report):
    """Standalone interactive report; no CDN, network request, or model download."""
    data = json.dumps(report).replace('<', '\\u003c')
    return '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference-Time Reasoning Lab</title>
<style>
:root{--ink:#292332;--gold:#af824a;--sage:#768165;--plum:#87708c;--paper:#f5f1e8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:32px 20px}h1{font-size:clamp(28px,5vw,46px);line-height:1.15}
h2{font-size:24px}header{border-top:6px solid var(--gold)}.eyebrow{color:#6c563e;font-weight:700;letter-spacing:.1em}
.panel{background:#fff;border:1px solid #d9d1c7;border-radius:12px;padding:24px;margin:24px 0}
.controls{display:flex;gap:24px;flex-wrap:wrap}label{display:flex;flex-direction:column;font-weight:600}
select{font:inherit;padding:8px;border:1px solid var(--plum);background:#fff;border-radius:6px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:10px;text-align:left;border-bottom:1px solid #e5dfd7}th{color:#655269}
.bar{height:12px;background:#ede6ef;border-radius:8px;width:120px}.fill{height:100%;background:var(--sage);border-radius:8px}
pre{background:var(--ink);color:#f5f1e8;padding:18px;border-radius:8px;overflow:auto;font:14px/1.6 monospace}
small{color:#655a68}.callout{border-left:4px solid var(--gold);padding-left:16px}
summary{cursor:pointer;font-weight:600}a{color:#66516e}
</style><main><header><p class="eyebrow">RL COURSE / PART III</p>
<h1>One frozen policy.<br>Several ways to spend inference compute.</h1>
<p>Created by Shivam Bharadwaj</p>
<p>Explore recorded runs of a controlled arithmetic simulator. Switching controls filters real
precomputed results; it does not run an LLM in your browser.</p></header>
<section class="panel"><h2>Experiment contract</h2><p id="contract"></p>
<p>A proposed arithmetic step costs 1 unit. Each scorer query has the configured cost.
Units are a teaching abstraction, not FLOPs, tokens, or latency. Final correctness is checked
after selection and is never fed back into search.</p>
<p class="callout">The exact process scorer checks local arithmetic. The misleading scorer rewards +1 errors.
Neither is a learned value model. Policy probabilities remain fixed throughout all inference runs.</p></section>
<section class="panel"><div class="controls">
<label>Scoring signal<select id="scorer"><option value="process">Exact process checker</option><option value="misleading">Misleading +1 reward</option></select></label>
<label>Maximum work units<select id="budget"></select></label></div>
<h2>Success and actual cost</h2><p>Means over the recorded generation seeds on the same tasks.
Single sampling may leave most of its allowance unused. Beam search can exhaust its configured tree before spending the budget.</p>
<div class="table-wrap"><table><thead><tr><th>Strategy</th><th>Answer correct</th><th>Valid process</th><th>Oracle candidate success</th><th>Units used</th><th>Completed</th></tr></thead><tbody id="results"></tbody></table></div>
<p><small>Oracle candidate success means a correct answer occurred somewhere in the completed candidate set.
It is an opportunity upper bound, not the deployed selector's success rate. Correct answers may contain compensating intermediate errors.</small></p></section>
<section class="panel"><h2>Inspect the first task</h2><label>Strategy<select id="method"></select></label>
<p id="task"></p><pre id="trace"></pre><details><summary>Show the recorded expansion log</summary><pre id="events"></pre></details></section>
<section class="panel"><h2>Investigate</h2><ol>
<li>Does a larger candidate set improve the selected answer, or only the oracle opportunity?</li>
<li>Switch to the misleading scorer. Which strategies concentrate on its mistakes?</li>
<li>Compare answer correctness with valid-process rate. Where can arithmetic errors cancel?</li>
<li>Rerun the CLI with a higher scorer cost. Does the ranking survive that cost model?</li></ol>
<p>Reproduce or change the experiment with <code>python -m rl_course.inference_demo --help</code>.</p>
<p>Educational report: CC BY 4.0. Implementation: MIT. No claim about open-domain LLM reasoning follows from this simulator.</p></section></main>
<script id="data" type="application/json">''' + data + '''</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), config=data.configuration;
const names={single:'Single sampling',self_consistency:'Self-consistency',best_of_n:'Best-of-N',beam:'Beam search',best_first:'Best-first search'};
for(const b of config.budgets){const o=document.createElement('option');o.value=b;o.textContent=b;$('budget').append(o)}
$('budget').value=config.budgets[config.budgets.length-1];
for(const m of config.methods){const o=document.createElement('option');o.value=m;o.textContent=names[m];$('method').append(o)}
$('contract').textContent=`${config.tasks} unique tasks; ${config.depth} decisions per trace; generation seeds ${config.generation_seeds.join(', ')}. Frozen error probabilities (-1, 0, +1): ${config.policy.join(', ')}. Scorer cost: ${config.score_cost}.`;
function render(){const scorer=$('scorer').value,budget=Number($('budget').value);$('results').replaceChildren();
for(const m of config.methods){const rows=data.rows.filter(r=>r.scorer===scorer&&r.budget===budget&&r.method===m);
const avg=k=>rows.reduce((a,r)=>a+r[k],0)/rows.length;const tr=document.createElement('tr');
const vals=[names[m],...( ['answer_correct','process_valid','candidate_oracle_success'].map(k=>(100*avg(k)).toFixed(1)+'%')),avg('spent').toFixed(1),(100*avg('completed')).toFixed(1)+'%'];
for(const val of vals){const td=document.createElement('td');td.textContent=val;tr.append(td)}$('results').append(tr)}
const trace=data.traces.find(t=>t.scorer===scorer&&t.budget===budget&&t.method===$('method').value);
$('task').textContent='Task: '+trace.operands.join(' + ')+'; expected answer: '+trace.operands.reduce((a,b)=>a+b,0);
$('trace').textContent=trace.selected?JSON.stringify({claimed_sums:trace.selected.values,...trace.metrics},null,2):'No complete candidate within the strategy and budget. This run abstained.';
$('events').textContent=JSON.stringify(trace.events,null,2)}
for(const id of ['scorer','budget','method'])$(id).addEventListener('change',render);render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=int, default=100)
    parser.add_argument('--budgets', type=int, nargs='+', default=[6, 12, 24, 48])
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--score-cost', type=int, default=1)
    parser.add_argument('--output', type=Path, help='Write a standalone interactive HTML report.')
    parser.add_argument('--json', type=Path, help='Write all seed-level results and example traces.')
    args = parser.parse_args()
    try:
        report = build_report(count=args.tasks, budgets=args.budgets, seeds=args.seeds,
                              depth=args.depth, score_cost=args.score_cost)
    except ValueError as error:
        parser.error(str(error))
    print('ArithmeticTrace-v1: frozen categorical simulator, not an LLM benchmark')
    print('scorer       budget strategy           correct  process  units')
    for scorer in ('process', 'misleading'):
        for budget in args.budgets:
            for method in METHODS:
                rows = [r for r in report['rows'] if r['scorer'] == scorer and r['budget'] == budget and r['method'] == method]
                print(f'{scorer:12} {budget:6} {method:18} {mean(r["answer_correct"] for r in rows):7.1%} '
                      f'{mean(r["process_valid"] for r in rows):7.1%} {mean(r["spent"] for r in rows):6.1f}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_html(report), encoding='utf-8')
        print(f'HTML report: {args.output.resolve()}')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Raw results: {args.json.resolve()}')


if __name__ == '__main__':
    main()
