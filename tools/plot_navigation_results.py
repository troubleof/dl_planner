#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[('empty',0),('wall_gap',0),('offset_wall',0),('forest',41),('forest',42),('forest',43)]
    colors={'ego':'#2266bb','rog':'#dc681c'}
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    for ax,(scene,seed) in zip(axes.ravel(),cases):
        folder=root/'navigation-benchmark-v1'/f'{scene}_{seed}_ego_delay0';world=json.loads((folder/'scene.json').read_text())
        for ob in world['obstacles']:
            if ob['kind']=='box':
                lo=np.array(ob['lo']);hi=np.array(ob['hi']);ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],color='#aaaaaa'))
            else:ax.add_patch(Circle(ob['xy'],ob['radius'],color='#aaaaaa'))
        for b in ['ego','rog']:
            data=np.loadtxt(root/'navigation-benchmark-v1'/f'{scene}_{seed}_{b}_delay0/trajectory.csv',delimiter=',',skiprows=1)
            ax.plot(data[:,1],data[:,2],label=b.upper(),color=colors[b],lw=2,alpha=.85)
        ax.scatter([-7,7],[0,0],c=['#333333','#169342'],marker='*',s=65,zorder=4)
        ax.set(xlim=(-8,8),ylim=(-4.8,4.8),xlabel='x (m)',ylabel='y (m)',title=f'{scene.replace("_"," ").title()}'+(f' | held-out seed {seed}' if seed else ''))
        ax.set_aspect('equal');ax.grid(alpha=.18);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Single-UAV navigation | matched body, sensor, EGO optimizer and execution | no extra delay',fontsize=13)
    fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    replay=json.loads((root/'navigation-replay-v1/summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,key,title in zip(axes,['map_update_ms','rpc_ms'],['Mapping adapter: median / P95','Binary cloud + IPC + mapping: median / P95']):
        x=np.arange(2);p50=[replay[b][key]['p50'] for b in ['ego','rog']];p95=[replay[b][key]['p95'] for b in ['ego','rog']]
        ax.bar(x-.17,p50,width=.34,label='Median',color='#477fb4');ax.bar(x+.17,p95,width=.34,label='P95',color='#eca562')
        for xx,yy in zip(x-.17,p50):ax.text(xx,yy+.4,f'{yy:.2f}',ha='center',fontsize=9)
        for xx,yy in zip(x+.17,p95):ax.text(xx,yy+.4,f'{yy:.2f}',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['EGO map','ROG map'],ylabel='Milliseconds',title=title,ylim=(0,max(p95)*1.25));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('39 identical point-cloud frames | fixed volume, 0.1 m voxels, common cube inflation',fontsize=12)
    fig.savefig(out/'map_replay_timing.png',dpi=170);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
