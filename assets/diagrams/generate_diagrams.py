"""Regenerate the course diagrams with Python and matplotlib.

Run from the repository root: python3 assets/diagrams/generate_diagrams.py
SVGs are used by README.md; high-resolution PNG previews go to /tmp/rl-diagram-previews.
The Ink / Brass / Sage theme and 760px layouts are specific to this course.
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

# The course's visual identity: ink-plum, antique gold, sage, and warm paper.
PAPER='#F5F1E8'
INK='#292332'
MUTED='#766C7C'
GOLD='#AF824A'
SAGE='#768165'
COLORS={
 'observe':('#FBF9F4','#AA9B88',INK),
 'world':('#EBEEE3',SAGE,INK),
 'learn':('#EEE7F0','#87708C',INK),
 'choice':('#F2E5CE',GOLD,INK),
 'success':('#E1E8D6','#687653',INK),
 'failure':('#F0DFD8','#A56D58',INK),
}
ROLES={'observe':'CONTEXT','world':'ENVIRONMENT','learn':'LEARNING','choice':'DECISION','success':'RESULT','failure':'FAILURE'}
plt.rcParams['svg.fonttype']='path'
plt.rcParams['font.family']='DejaVu Sans'

# Deliberate compact layouts for graphs whose topology is not a simple chain.
CUSTOM={
 'agent-environment':{'A':(215,175),'E':(545,175)},
 'q-learning':{'S':(145,160),'E':(380,160),'R':(615,160),'N':(145,300),'G':(380,300),'X':(615,300),'D':(145,440),'U':(380,440),'T':(615,440)},
 'actor-critic':{'S':(145,165),'A':(380,165),'E':(615,165),'C':(145,335),'T':(380,335)},
 'rlhf-pipeline':{'P':(145,160),'S':(380,160),'REF':(615,160),'H':(145,310),'R':(380,310),'O':(615,310),'F':(615,460)},
}
SHORT={
 'q-learning':{'S':'Observe state','E':'Explore?','R':'Random action','G':'Greedy action','X':'Execute and observe','T':'Reward + discounted max Q','U':'Update current Q','D':'Task terminated?','N':'Reset episode'},
 'rlhf-pipeline':{'P':'Pretrained model','S':'Supervised fine-tuning','H':'Human comparisons','REF':'Frozen reference','R':'Reward model','O':'PPO from the SFT policy','F':'Optimized policy'},
 'actor-critic':{'S':'Current state','A':'Actor / choose action','E':'Environment / feedback','C':'Critic / value estimate','T':'TD error / advantage'},
}

def layout(name, rows):
    flat=[n for row in rows for n in row]
    if name in CUSTOM:
        return CUSTOM[name],206 if name in ('q-learning','actor-critic','rlhf-pipeline') else 270
    if all(len(r)==1 for r in rows):
        pos={}
        for i,(nid,*_) in enumerate(flat):
            row,col=divmod(i,2)
            if row%2:col=1-col
            pos[nid]=(215+330*col,155+132*row)
        return pos,270
    cols=max(map(len,rows));width=206 if cols==3 else 270
    pos={}
    spacing=102 if len(rows)>=5 else 112
    for r,row in enumerate(rows):
        for c,(nid,*_) in enumerate(row):
            pos[nid]=(380+(c-(len(row)-1)/2)*(235 if cols==3 else 330),150+spacing*r)
    return pos,width

# Route arrows around card interiors. Existing routes have a small cost to keep
# opposite-direction feedback paths separate. Candidate ports preserve topology.
def route(src,dst,positions,bw,bh,occupied,limits,reverse=False,forced_ports=None):
    import heapq,math
    step=5
    W,H=limits
    rects=[(x-bw/2-6,y-bh/2-6,x+bw/2+6,y+bh/2+6) for x,y in positions.values()]
    def ports(nid):
        x,y=positions[nid]
        return {'r':((x+bw/2,y),(x+bw/2+15,y)), 'l':((x-bw/2,y),(x-bw/2-15,y)),
                't':((x,y-bh/2),(x,y-bh/2-15)), 'b':((x,y+bh/2),(x,y+bh/2+15))}
    sx,sy=positions[src];tx,ty=positions[dst]
    if abs(ty-sy)<1:pref=('r','l') if tx>sx else ('l','r')
    elif abs(tx-sx)<1:pref=('b','t') if ty>sy else ('t','b')
    else:pref=('b','t') if ty>sy else ('t','b')
    if reverse:pref=('b','b')
    candidates=[pref,('r','l'),('l','r'),('b','t'),('t','b'),('b','b'),('l','l'),('r','r'),('t','t')]
    if reverse:candidates=[('b','b'),('t','t')]
    if forced_ports:candidates=[forced_ports]
    candidates=list(dict.fromkeys(candidates))
    def cell(p):return (round(p[0]/step),round(p[1]/step))
    def free(c):
        x,y=c[0]*step,c[1]*step
        return 20<=x<=W-20 and 105<=y<=H-12 and not any(a<x<b and c_<y<d for a,c_,b,d in rects)
    best=None
    for rank,(sp,tp) in enumerate(candidates):
        a,astart=ports(src)[sp];b,bend=ports(dst)[tp]
        start,end=cell(astart),cell(bend)
        heap=[(0,0,start,None)];cost={(start,None):0};parent={};goal=None
        while heap:
            _,g,cur,direction=heapq.heappop(heap)
            key=(cur,direction)
            if g!=cost.get(key):continue
            if cur==end:goal=key;break
            for dire,(dx,dy) in enumerate(((1,0),(0,1),(-1,0),(0,-1))):
                nxt=(cur[0]+dx,cur[1]+dy)
                if not free(nxt):continue
                ng=g+1+(3 if direction is not None and dire!=direction else 0)+occupied.get(nxt,0)*1.3
                nk=(nxt,dire)
                if ng<cost.get(nk,float('inf')):
                    cost[nk]=ng;parent[nk]=key
                    heuristic=abs(nxt[0]-end[0])+abs(nxt[1]-end[1])
                    heapq.heappush(heap,(ng+heuristic,ng,nxt,dire))
        if goal is None:continue
        score=cost[goal]+rank*3
        if best is None or score<best[0]:
            cells=[];k=goal
            while k in parent:cells.append(k[0]);k=parent[k]
            cells.append(start);cells.reverse()
            pts=[a,astart]+[(x*step,y*step) for x,y in cells]+[bend,b]
            best=(score,pts,cells)
        if rank==0 and best and best[0]<40:break
    if best is None:raise RuntimeError(f'No route for {src}->{dst}')
    for c in best[2]:occupied[c]=occupied.get(c,0)+1
    # Remove redundant collinear segments without changing obstacle avoidance.
    pts=[]
    for p in best[1]:
        if pts and p==pts[-1]:continue
        if len(pts)>1 and abs((pts[-1][0]-pts[-2][0])*(p[1]-pts[-1][1])-(pts[-1][1]-pts[-2][1])*(p[0]-pts[-1][0]))<.001:pts.pop()
        pts.append(p)
    return pts


def draw(name,title,rows,edges):
    from matplotlib.patches import Circle
    positions,bw=layout(name,rows);bh=86
    flat=[n for row in rows for n in row]
    bottom=max(y for x,y in positions.values())+bh/2
    graph_end=bottom+48
    notes=[label for a,b,label in edges if label]
    footer_height=(max(1,(len(notes)+1)//2)*28+28) if notes else 30
    height=round(graph_end+footer_height+25)
    fig=plt.figure(figsize=(7.6,height/100),dpi=100)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,760);ax.set_ylim(height,0);ax.axis('off')
    fig.patch.set_facecolor(PAPER)
    def text(x,y,t,size=16,**kwargs):return ax.text(x,y,t,fontsize=size*.72,color=kwargs.pop('color',INK),**kwargs)
    # Editorial header, restrained course signature, generous internal margins.
    ax.add_patch(FancyBboxPatch((0,0),760,83,boxstyle='square,pad=0',facecolor=INK,edgecolor='none'))
    text(30,23,'RL  /  FIELD NOTES',10,color='#D7BC95',weight='bold',va='center')
    title_size=22 if len(title)<51 else 20
    text(30,55,title,title_size,color=PAPER,weight='bold',va='center')
    ax.plot([30,68],[84,84],color=GOLD,lw=3)
    occupied={};paths=[];seen=set();note_index=0
    for src,dst,label in edges:
        forced=('b','r') if name=='actor-critic' and src=='E' and dst=='T' else None
        pts=route(src,dst,positions,bw,bh,occupied,(760,graph_end),reverse=(dst,src) in seen,forced_ports=forced)
        seen.add((src,dst));paths.append((src,dst,pts))
        path=PlotPath(pts,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(pts)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=12,lw=1.35,color='#8A7F89',zorder=1,joinstyle='round',capstyle='round'))
        if label:
            note_index+=1
            segments=[(abs(b[0]-a[0])+abs(b[1]-a[1]),a,b) for a,b in zip(pts,pts[1:])]
            _,a,b=max(segments,key=lambda x:x[0])
            xx,yy=(a[0]+b[0])/2,(a[1]+b[1])/2
            ax.add_patch(Circle((xx,yy),8,facecolor=PAPER,edgecolor=GOLD,lw=.8,zorder=3))
            text(xx,yy,str(note_index),10,ha='center',va='center',color=GOLD,weight='bold',zorder=4)
    text_boxes=[]
    for i,(nid,label,role) in enumerate(flat):
        x,y=positions[nid];fill,stroke,ink=COLORS[role]
        label=SHORT.get(name,{}).get(nid,label)
        if name=='course-roadmap':label=label.split(' / ',1)[1]
        ax.add_patch(FancyBboxPatch((x-bw/2,y-bh/2+3),bw,bh,boxstyle='round,pad=0,rounding_size=9',linewidth=0,facecolor='#E4DDD3',zorder=3))
        ax.add_patch(FancyBboxPatch((x-bw/2,y-bh/2),bw,bh,boxstyle='round,pad=0,rounding_size=9',linewidth=.8,edgecolor='#D5CBBF',facecolor=fill,zorder=4))
        ax.plot([x-bw/2+1,x-bw/2+1],[y-bh/2+13,y+bh/2-13],color=stroke,lw=3,zorder=5)
        text(x-bw/2+15,y-bh/2+18,ROLES[role],9,color=stroke,weight='bold',va='center',zorder=6)
        text(x+bw/2-14,y-bh/2+18,f'{i+1:02d}',9,color=MUTED,ha='right',va='center',zorder=6)
        from matplotlib.font_manager import FontProperties
        size=15.5 if bw<220 else 16
        metrics=fig.canvas.get_renderer()
        prop=FontProperties(family='DejaVu Sans',size=size*.72,weight='medium')
        lines=[]
        for part in label.split('\n'):
            current=''
            for word in part.split():
                candidate=(current+' '+word).strip()
                if current and metrics.get_text_width_height_descent(candidate,prop,False)[0]>bw-32:
                    lines.append(current);current=word
                else:current=candidate
            lines.append(current)
        wrapped='\n'.join(lines)
        artist=text(x-bw/2+15,y+9,wrapped,size,ha='left',va='center',weight='medium',linespacing=1.3,zorder=6)
        text_boxes.append((nid,artist,(x-bw/2+10,y-bh/2+28,x+bw/2-10,y+bh/2-5)))
    if notes:
        ax.plot([30,730],[graph_end,graph_end],color='#D5CBBF',lw=.8)
        for i,note in enumerate(notes):
            col,row=i%2,i//2;x=30+col*360;y=graph_end+22+row*28
            text(x,y,f'{i+1:02d}',10,color=GOLD,weight='bold',va='center')
            text(x+25,y,textwrap.fill(note,44),11.5,color=MUTED,va='center',linespacing=1.25)
    text(30,height-14,'REINFORCEMENT LEARNING  /  FROM FUNDAMENTALS TO MODERN RL',8,color=MUTED,va='center')
    # Verify card labels against actual rendered font metrics before exporting.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for nid,artist,(left,top,right,bot) in text_boxes:
        extent=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        if extent.x0<left-1 or extent.x1>right+1 or min(extent.y0,extent.y1)<top-3 or max(extent.y0,extent.y1)>bot+2:
            raise RuntimeError(f'Text does not fit {name}/{nid}: {artist.get_text()}')
    fig.savefig(ROOT/(name+'.svg'),facecolor=PAPER,metadata={'Title':title,'Description':'; '.join(n[1].replace('\n',': ') for n in flat)})
    fig.savefig(PREVIEWS/(name+'.png'),dpi=160,facecolor=PAPER)
    plt.close(fig)
    # Give SVGs explicit pixel dimensions, a viewBox, and stable accessible labels.
    import xml.etree.ElementTree as ET
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    out=ROOT/(name+'.svg');tree=ET.parse(out);svg=tree.getroot()
    svg.set('width','760');svg.set('height',str(height));svg.set('role','img');svg.set('aria-label',title)
    tree.write(out,encoding='utf-8',xml_declaration=True)
    return height

if __name__=='__main__':
    heights=[]
    for diagram in DIAGRAMS:heights.append(draw(*diagram))
    print(f'Generated {len(DIAGRAMS)} diagrams: width 760px, heights {min(heights)}–{max(heights)}px. All card labels fit.')
