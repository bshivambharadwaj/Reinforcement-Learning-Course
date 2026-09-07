"""Regenerate the course diagrams with Python and matplotlib.

Run from the repository root: python3 assets/diagrams/generate_diagrams.py
SVGs are used by README.md; PNG previews are written to /tmp/rl-diagram-previews.
"""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parent
PREVIEWS = Path('/tmp/rl-diagram-previews')
PREVIEWS.mkdir(exist_ok=True)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = {
    'observe': ('#DBEAFE', '#2563EB', '#172554'),
    'world': ('#CCFBF1', '#0D9488', '#134E4A'),
    'learn': ('#EDE9FE', '#7C3AED', '#3B0764'),
    'choice': ('#FEF3C7', '#D97706', '#78350F'),
    'success': ('#DCFCE7', '#16A34A', '#14532D'),
    'failure': ('#FFE4E6', '#E11D48', '#881337'),
}
# A node is (id, label, color role). Each list is one horizontal row.
# An edge is (source, destination, label); backward edges route outside the nodes.
DIAGRAMS = [
('delivery-robot', 'One parcel, two possible routes', [
 [('S','Depot / start','observe')],
 [('J','Junction','choice')],
 [('K','Safe residential street','world'),('U','Uncertain shortcut','choice')],
 [('G','Delivered / terminal','success'),('F','Breakdown / terminal','failure')]],
 [('S','J','Move: -1'),('J','K','Move: -1'),('J','U','Try shortcut'),('K','G','Delivery: +10'),('U','G','Success: +10'),('U','F','Failure: -10')]),
('course-roadmap', 'Your path through reinforcement learning', [
 [('A','01 / Foundations\nAgent loop, MDPs, returns','observe')],
 [('B','02 / Evaluate decisions\nValues, advantage, Bellman','observe')],
 [('C','03 / Plan and learn\nDP, Monte Carlo, TD','world')],
 [('D','04 / Learn action values\nSARSA, Q-learning, DQN','learn')],
 [('E','05 / Improve the policy\nGradients, actor-critic, GAE','learn')],
 [('F','06 / Control updates\nTRPO, PPO, entropy','choice')],
 [('G','07 / Extend the setting\nOffline, model-based, MARL','world')],
 [('H','08 / Modern applications\nRLHF, DPO, GRPO, agents','success')]],
 [('A','B',''),('B','C',''),('C','D',''),('D','E',''),('E','F',''),('F','G',''),('G','H','')]),
('agent-environment', 'Act, observe, and learn', [
 [('A','Agent / policy','learn')],
 [('E','Environment','world')]],
 [('A','E','Action'),('E','A','Next state + reward')]),
('interaction-loop', 'From one step to the next episode', [
 [('S','Observe the current state','observe')],
 [('A','Policy chooses an action','choice')],
 [('E','Execute; receive next state and reward','world')],
 [('U','Update estimates during training','learn')],
 [('T','Did the episode end?','choice')],
 [('N','Reset the environment','success')]],
 [('S','A',''),('A','E',''),('E','U',''),('U','T',''),('T','S','No: next step'),('T','N','Yes'),('N','S','New episode')]),
('bellman-decomposition', 'A long future, one step at a time', [
 [('R','Immediate reward','world'),('F','Discounted next-state value','observe')],
 [('V','Add both to get the one-step value target','learn')]],
 [('R','V','Reward'),('F','V','Future value')]),
('bellman-optimality', 'Average outcomes, then choose an action', [
 [('S','Current state','observe')],
 [('A','Try east','choice'),('B','Try north','choice')],
 [('AO','Average possible east outcomes','world'),('BO','Average possible north outcomes','world')],
 [('M','Choose the larger expected target','success')]],
 [('S','A',''),('S','B',''),('A','AO',''),('B','BO',''),('AO','M',''),('BO','M','')]),
('dynamic-programming', 'A Bellman backup with a known model', [
 [('S','Select a current state','observe')],
 [('A','Enumerate available actions','choice')],
 [('P','Use transition probabilities and rewards','world')],
 [('F','Combine reward with next-state values','learn')],
 [('U','Update the current value','success')]],
 [('S','A',''),('A','P',''),('P','F',''),('F','U','')]),
('policy-iteration', 'Evaluate a policy, then improve it', [
 [('I','Initialize a policy','observe')],
 [('E','Evaluate: estimate its state values','learn')],
 [('P','Improve: choose greedy actions','learn')],
 [('S','Is the policy unchanged?','choice')],
 [('O','Return an optimal policy','success')]],
 [('I','E',''),('E','P',''),('P','S',''),('S','E','No: repeat'),('S','O','Yes')]),
('td-learning', 'TD combines sampling and bootstrapping', [
 [('MC','Monte Carlo\nLearn from experience','observe'),('DP','Dynamic programming\nUse value estimates','learn')],
 [('TD','Temporal difference\nSample a step and bootstrap','success')]],
 [('MC','TD','Sampling'),('DP','TD','Bootstrapping')]),
('q-learning', 'Explore with behavior; learn a greedy target', [
 [('S','Observe the current state','observe')],
 [('E','Explore with probability epsilon?','choice')],
 [('R','Choose a random action','choice'),('G','Choose a greedy action','choice')],
 [('X','Execute and observe the transition','world')],
 [('T','Target: reward + discounted max next Q\nZero bootstrap at true termination','learn')],
 [('U','Move current Q toward the target','learn')],
 [('D','Did the task terminate?','choice')],
 [('N','Start a new episode','success')]],
 [('S','E',''),('E','R','Yes'),('E','G','No'),('R','X',''),('G','X',''),('X','T',''),('T','U',''),('U','D',''),('D','S','No'),('D','N','Yes'),('N','S','Reset')]),
('experience-replay', 'Reuse experience in shuffled minibatches', [
 [('P','Behavior policy selects an action','choice')],
 [('E','Environment produces a transition','world')],
 [('B','Store it in the replay buffer','observe')],
 [('M','Sample a random minibatch','observe')],
 [('L','Compute the TD loss','learn')],
 [('Q','Update the online Q-network','success')]],
 [('P','E',''),('E','B',''),('B','M',''),('M','L',''),('L','Q',''),('Q','P','Updated behavior')]),
('actor-critic', 'The actor chooses; the critic evaluates', [
 [('S','Current state','observe')],
 [('A','Actor\nAction probabilities','choice'),('C','Critic\nValue estimate','learn')],
 [('E','Environment\nAction leads to feedback','world')],
 [('T','TD error / advantage estimate','learn')]],
 [('S','A',''),('S','C',''),('A','E','Action'),('E','T','Reward + next state'),('C','T','Baseline'),('T','A','Improve actor'),('T','C','Train critic')]),
('ppo-training', 'PPO alternates collection and improvement', [
 [('O','Freeze the old policy snapshot','observe')],
 [('R','Collect a fresh rollout','world')],
 [('A','Compute returns and GAE advantages','learn')],
 [('M','Run minibatch update epochs','learn')],
 [('L','Clipped actor objective\nCritic loss + entropy bonus','choice')],
 [('U','Update current parameters','success')]],
 [('O','R',''),('R','A',''),('A','M',''),('M','L',''),('L','U',''),('U','O','Refresh snapshot')]),
('world-model', 'Learn in the world; plan with a model', [
 [('R','Collect real experience','observe')],
 [('M','Learn a world model','learn')],
 [('I','Imagine future trajectories','world')],
 [('P','Plan or improve the policy','choice')],
 [('A','Take an action in the real world','success')]],
 [('R','M',''),('M','I',''),('I','P',''),('P','A',''),('A','R','Observe results')]),
('model-based-loop', 'Close the loop with real feedback', [
 [('E','Real environment','world')],
 [('D','Observed transitions','observe')],
 [('M','Fit dynamics and reward models','learn')],
 [('P','Simulate candidate futures','learn')],
 [('A','Choose the next action','choice')]],
 [('E','D',''),('D','M',''),('M','P',''),('P','A',''),('A','E','Act, then replan')]),
('rlhf-pipeline', 'From demonstrations to preference feedback', [
 [('P','Pretrained language model','observe')],
 [('S','Supervised fine-tuned policy','learn')],
 [('H','Human preference comparisons','observe'),('REF','Frozen reference policy','observe')],
 [('R','Train a reward model','learn')],
 [('O','PPO optimization\nStart from the SFT policy','choice')],
 [('F','Preference-optimized policy','success')]],
 [('P','S',''),('S','H','Sample responses'),('S','REF','Copy'),('H','R',''),('R','O','Reward scores'),('REF','O','Regularization'),('O','F','')]),
('dpo-pipeline', 'Learn directly from preferred and rejected pairs', [
 [('P','Preference pairs','observe'),('R','Frozen reference policy','observe')],
 [('M','Reference-relative likelihood margin','learn')],
 [('L','Optimize the DPO loss','choice')],
 [('U','Updated language policy','success')]],
 [('P','M',''),('R','M',''),('M','L',''),('L','U','')]),
('grpo-training', 'Compare several attempts at the same prompt', [
 [('P','One prompt','observe')],
 [('A','Output 1','choice'),('B','Output 2','choice'),('C','Output G','choice')],
 [('RA','Score 1','world'),('RB','Score 2','world'),('RC','Score G','world')],
 [('N','Compute group-relative advantages','learn')],
 [('U','Clipped policy update\nWith reference regularization','success')]],
 [('P','A',''),('P','B',''),('P','C',''),('A','RA',''),('B','RB',''),('C','RC',''),('RA','N',''),('RB','N',''),('RC','N',''),('N','U','')]),
('multimodal-agent', 'A tool-using agent learns across many steps', [
 [('O','Text, images, audio, tool results, memory','observe')],
 [('P','Agent policy chooses the next action','learn')],
 [('T','Tool call / click / code / physical action','choice')],
 [('E','Tools or the real environment','world')],
 [('F','New observations and task feedback','success')]],
 [('O','P',''),('P','T',''),('T','E',''),('E','F',''),('F','O','Continue the task')]),
]

def draw(name, title, rows, edges):
    max_cols = max(map(len, rows))
    # A single-column chart stays compact; branching charts get space to breathe.
    bw = 4.3 if max_cols == 1 else 3.45
    dx, dy, bh = bw + .9, 1.75, .96
    points = {}
    for ri, row in enumerate(rows):
        for ci, (nid, label, role) in enumerate(row):
            points[nid] = ((ci-(len(row)-1)/2)*dx, (len(rows)-1-ri)*dy, label, role)
    xmin=min(p[0] for p in points.values())-bw/2-.35
    xmax=max(p[0] for p in points.values())+bw/2+.35
    ymax=(len(rows)-1)*dy+1.4
    backward=[(a,b,l) for a,b,l in edges if points[b][1]>points[a][1]]
    skips=any(points[a][1]-points[b][1]>dy*1.2 for a,b,l in edges)
    if skips: xmax+=.9
    # Both margins serve as return lanes for charts with several feedback edges.
    if backward:
        xmax+=1.2
        if len(backward)>1: xmin-=1.2
    fig, ax=plt.subplots(figsize=(xmax-xmin,ymax+.7))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
    ax.set_xlim(xmin,xmax);ax.set_ylim(-.75,ymax);ax.axis('off')
    ax.text((xmin+xmax)/2,ymax-.25,title,ha='center',va='top',fontsize=15,weight='bold',color='#0F172A')
    back_index=0
    for src,dst,label in edges:
        x1,y1,_,_=points[src];x2,y2,_,_=points[dst]
        if y2>y1:
            side=(-1 if x2<0 else 1) if x2 else (1 if back_index%2==0 else -1)
            lane=(xmax-.35) if side==1 else (xmin+.35)
            back_index+=1
            start=(x1+side*bw/2,y1);end=(x2+side*bw/2,y2)
            verts=[start,(lane,y1),(lane,y2),end]
            patch=FancyArrowPatch(path=PlotPath(verts,[PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO]),arrowstyle='-|>',mutation_scale=15,lw=1.5,color='#64748B',zorder=1)
            ax.add_patch(patch)
            if label:ax.text(lane-.12*side,(y1+y2)/2,label,rotation=90,ha='center',va='center',fontsize=9,color='#475569',bbox=dict(facecolor='#F8FAFC',edgecolor='none',pad=3))
        else:
            start=(x1,y1-bh/2);end=(x2,y2+bh/2)
            # Route long skip edges outside intermediate nodes.
            if y1-y2>dy*1.2:
                lane=xmax-.35
                start=(x1+bw/2,y1);end=(x2+bw/2,y2)
                if name=='actor-critic' and src=='C':
                    lane=x1
                    start=(x1,y1-bh/2)
                verts=[start,(lane,y1),(lane,y2),end]
                if name=='actor-critic' and src=='C':
                    verts=[start,(lane,y2),end]
                patch=FancyArrowPatch(path=PlotPath(verts,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(verts)-1)),arrowstyle='-|>',mutation_scale=15,lw=1.5,color='#64748B',zorder=1)
                labelpos=(lane,(y1+y2)/2)
            else:
                patch=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,lw=1.5,color='#64748B',zorder=1,connectionstyle='arc3,rad=0')
                labelpos=((x1+x2)/2,(y1+y2)/2)
            ax.add_patch(patch)
            if label:ax.text(*labelpos,label,rotation=90 if y1-y2>dy*1.2 else 0,ha='center',va='center',fontsize=9,color='#475569',bbox=dict(facecolor='#F8FAFC',edgecolor='none',pad=2.5),zorder=3)
    for nid,(x,y,label,role) in points.items():
        fill,stroke,ink=COLORS[role]
        box=FancyBboxPatch((x-bw/2,y-bh/2),bw,bh,boxstyle='round,pad=0.025,rounding_size=0.12',linewidth=1.6,edgecolor=stroke,facecolor=fill,zorder=4)
        ax.add_patch(box)
        # Explicit labels supplement color, so meaning never depends on color alone.
        wrapped='\n'.join(textwrap.fill(part,width=36 if max_cols==1 else 27) for part in label.split('\n'))
        ax.text(x,y,wrapped,ha='center',va='center',fontsize=11,weight='medium',color=ink,zorder=5,linespacing=1.4)
    fig.tight_layout(pad=.4)
    fig.savefig(ROOT/(name+'.svg'),facecolor=fig.get_facecolor(),metadata={'Title':title,'Description':'; '.join(p[2].replace('\n',': ') for p in points.values())})
    fig.savefig(PREVIEWS/(name+'.png'),dpi=100,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':
    for diagram in DIAGRAMS: draw(*diagram)
    print(f'Generated {len(DIAGRAMS)} SVG diagrams in {ROOT}')
