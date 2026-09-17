"""Serve real GoalLab-v2 investigations or export an offline trace explorer. MIT."""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import uuid
import torch
from . import goallab_benchmark as g
from . import goallab_training as t

METHODS = ['Greedy','DQN','PPO','PPO + Verifier','RL + Test-Time Search']


def render(report):
    # Keep a small, uncurated prefix of cases per method/budget/split for preview.
    selected = [r for r in report['records'] if r['seed']==report['config']['seeds'][0]
                and int(r['case'].split('/')[-1]) < 5]
    data = dict(version=g.VERSION, summary=report['summary'], records=selected,
                config=report['config'])
    template = Path(__file__).with_name('goallab_benchmark_demo.html').read_text()
    return template.replace('__BENCHMARK_DATA__',json.dumps(data).replace('<','\\u003c'))


def export(report_path, output):
    report = json.loads(Path(report_path).read_text())
    Path(output).write_text(render(report),encoding='utf-8')


def serve(report_path, checkpoint, port=8765):
    torch.set_num_threads(2)
    report = json.loads(Path(report_path).read_text())
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['version'] != g.VERSION:
        raise ValueError('Checkpoint version mismatch')
    dqn, ppo, verifier = t.Controller(),t.Controller(),t.Verifier()
    for name,model in [('dqn',dqn),('ppo',ppo),('verifier',verifier)]:
        model.load_state_dict(saved[name]);model.eval()
    sessions = OrderedDict()
    page = render(report).encode()

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, status=200, mime='application/json'):
            payload=data if isinstance(data,bytes) else json.dumps(data).encode()
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(payload)));self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path=='/': self.send(page,mime='text/html; charset=utf-8')
            elif self.path=='/api/health': self.send({'live':True,'version':g.VERSION})
            else: self.send({'error':'Not found'},404)

        def do_POST(self):
            # Local teaching server; no filesystem paths or arbitrary code in requests.
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 8192: raise ValueError('Invalid body size')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/start':
                    method=data['method']
                    if method not in METHODS: raise ValueError('Unknown controller')
                    budget=int(data['budget']);index=int(data['index']);level=int(data['level'])
                    if not 1<=budget<=32 or not 0<=index<=1_000_000: raise ValueError('Invalid budget or case')
                    env=g.Investigation(g.make_case(index,data['split'],level),budget)
                    token=uuid.uuid4().hex
                    sessions[token]=(env,method)
                    while len(sessions)>32: sessions.popitem(last=False)
                    self.send({'token':token,'observation':env.observation(),'case':env.case.id})
                elif self.path=='/api/step':
                    env,method=sessions[data['token']]
                    old=len(env.events)
                    model=dqn if method=='DQN' else ppo
                    if not env.done:
                        if method=='Greedy':env.step(g.greedy_action(env.observation()))
                        elif method=='RL + Test-Time Search' and env.observation()['remaining']>=5:
                            env=t.search_step(env,model,verifier)
                        elif method in {'PPO + Verifier','RL + Test-Time Search'}:
                            if env.charge_search():env.step(t.verifier_action(model,verifier,env.observation()))
                            else:env.step(g.greedy_action(env.observation()))
                        else:env.step(t.policy_action(model,env.observation()))
                    sessions[data['token']]=(env,method)
                    self.send({'events':env.events[old:],'done':env.done,'observation':env.observation(),
                               'artifact':env.artifact,
                               'evaluation':g.verify(env.case,env.artifact,env.observation()) if env.done else None})
                else:self.send({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                self.send({'error':str(exc)},400)

    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'GoalLab live: http://127.0.0.1:{server.server_port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if a.output:export(a.report,a.output)
    elif a.checkpoint:serve(a.report,a.checkpoint,a.port)
    else:p.error('Specify --output for an offline explorer or --checkpoint for live execution')
