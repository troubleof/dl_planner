#!/usr/bin/env python3
"""Geometry-only wall-end probes after normal following acceptance.

Same trees/walls and start prefix; no hidden stop/reverse leader schedule.
Candidate choice uses clearance/connectivity only, never closed-loop outcomes.
"""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from generate_forest import save,route_for_scene,validate_scene,plot_scene,sample_segment
from forest_geometry import World
from vehicle_geometry import VehicleEnvelope


def probe(scene):
    world=World(scene['trees']+scene['added_obstacles']);cfg=scene['config'];body=VehicleEnvelope.from_config(cfg['vehicle'])
    attempts=[];z=cfg['flight']['cruise_height_m']
    for wall in scene['added_obstacles']:
        if wall['shape']!='wall':continue
        vertices=np.array(wall['vertices']);center=vertices.mean(axis=0);u=vertices[1]-vertices[0];u/=np.linalg.norm(u)
        for direction in (1,-1):
            axis=u*direction;normal=np.array([-axis[1],axis[0]]);end=center+5*axis
            for side in (1,-1):
                normal=normal*side;radius=1.0
                # Three metre side approach, then turn around the wall end.
                angles=np.linspace(math.pi/2,-math.pi/2,13)
                xy=[end-3*axis+radius*normal]+[end+radius*(math.cos(t)*axis+math.sin(t)*normal) for t in angles]+[end-3*axis-radius*normal]
                points=np.column_stack((xy,np.full(len(xy),z)))
                key=dict(wall=wall['id'],direction=direction,side=side)
                if world.distance(points).min()<body.planning_radius+.2:
                    attempts.append(dict(**key,reason='probe clearance'));continue
                candidate=copy.deepcopy(scene);rc=candidate['config']['route'];rc['via_m']=points.tolist();rc['route_extra_clearance_m']=.35
                rc['grid_step_m']=.1;rc['head_gap_m']=2.6;rc['goal_region_radius_m']=3.2
                candidate['config']['experiment']['duration_s']=120.
                try:route=route_for_scene(candidate)
                except ValueError as e:
                    attempts.append(dict(**key,reason=str(e)));continue
                candidate['route']=route
                candidate['route_protocol']=dict(kind='wall_end_v2',selection='first geometrically feasible deterministic candidate; no closed-loop outcome selection',
                    selected=key,rejected_candidates=attempts,side_offset_m=radius,side_approach_m=3.,normal_acceptance_parent_seed=scene['seed'])
                return candidate
    raise ValueError('no feasible wall-end route: '+json.dumps(attempts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in a.seeds:
        out=a.out/f'seed_{seed}';out.mkdir()
        try:
            scene=probe(json.loads((a.scenes/f'seed_{seed}/scene.json').read_text()));audit=validate_scene(scene)
            save(out/'scene.json',scene);save(out/'geometry_audit.json',audit);plot_scene(scene,out)
            row=dict(seed=seed,passed=audit['passed'],route_length_m=scene['route']['length_m'],min_body_gap_m=scene['route']['min_body_gap_m'],selected=scene['route_protocol']['selected'])
        except Exception as e:row=dict(seed=seed,passed=False,error=repr(e));save(out/'failure.json',row)
        rows.append(row);print(json.dumps(row),flush=True)
    save(a.out/'summary.json',rows)
    if not all(r['passed'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
