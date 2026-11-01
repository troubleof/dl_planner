#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle

def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/'audit.json').read_text())['groups'];policies=['full','reuse','repair'];colors=['#2667b4','#dc7c20','#198c6b']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for ax,metric,title in zip(axes,['full_astar_calls','search_ms','planning_rpc_ms'],['Full A* calls (16 runs/backend)','Routing stage P95 (ms)','Planning request P95 incl. map + IPC (ms)']):
        for j,b in enumerate(['ego','rog']):
            vals=[next(x[metric] if metric=='full_astar_calls' else x[metric]['p95'] for x in data if x['backend']==b and x['policy']==p) for p in policies]
            at=np.arange(3)+(j-.5)*.34;ax.bar(at,vals,width=.32,label=b.upper(),color=['#477fb4','#eca562'][j])
            for xx,yy in zip(at,vals):ax.text(xx,yy+max(vals)*.025,f'{yy:.0f}' if metric=='full_astar_calls' else f'{yy:.1f}',ha='center',fontsize=8)
        ax.set(xticks=np.arange(3),xticklabels=policies,title=title);ax.margins(y=.18);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('96 matched-scenario closed-loop runs | routing policy is the changed factor',fontsize=12)
    fig.savefig(out/'routing_cost.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    for row,backend in enumerate(['ego','rog']):
        for col,(scene,seed) in enumerate([('offset_wall',0),('forest',42),('forest',102)]):
            ax=axes[row,col];folder=root/f'{scene}_{seed}_{backend}_full_delay0_r0';world=json.loads((folder/'scene.json').read_text())
            for ob in world['obstacles']:
                if ob['kind']=='box':
                    lo=np.array(ob['lo']);hi=np.array(ob['hi']);ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],color='#aaa'))
                else:ax.add_patch(Circle(ob['xy'],ob['radius'],color='#aaa'))
            for policy,color in zip(policies,colors):
                points=np.loadtxt(root/f'{scene}_{seed}_{backend}_{policy}_delay0_r0/trajectory.csv',delimiter=',',skiprows=1)
                ax.plot(points[:,1],points[:,2],label=policy,color=color,lw=1.8)
            ax.set(xlim=(-8,8),ylim=(-4.8,4.8),xlabel='x (m)',ylabel='y (m)',title=f'{backend.upper()} | {scene} {seed if seed else ""}')
            ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Executed trajectories | route reuse can trade path quality for lower computation',fontsize=12)
    fig.savefig(out/'routing_trajectories.png',dpi=170);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
