import argparse,json
from pathlib import Path
import numpy as np
from d435i_adapter import from_depth,OPTICAL_TO_WORLD,D435iSensor,camera_observation,install_simulator
from navigation_common import make_scene,stationary,save,Worker

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    args=dict(intrinsics=dict(fx=2.,fy=2.,cx=1.,cy=1.,distortion='none'),depth_scale=.001,camera_to_world=OPTICAL_TO_WORLD,origin=[0,0,2.5],stamp=1.,pose_stamp=1.,clock_domain='navigation',block=2)
    depth=np.array([[0,2000,0,0],[1000,3000,0,0],[0,0,0,0],[0,0,0,0]],np.uint16);cloud=from_depth(depth,**args)
    assert len(cloud['points'])==1;assert np.allclose(cloud['points'][0],[1,.5,2.5]) # min pixel (u=0,v=1), NOT block centre
    image,state=camera_observation(cloud,dict(t=1.,start_state=stationary([0,0,2.5]),goal=[5,0,2.5]),'ego');assert image.shape==(6,2,2) and np.count_nonzero(image[2])==1
    for field,value in [('clock_domain','hardware'),('pose_stamp',1.01)]:
        bad=dict(args);bad[field]=value
        try:from_depth(depth,**bad);raise AssertionError('invalid metadata accepted')
        except ValueError:pass
    empty=D435iSensor().capture(make_scene('empty'),[-7,0,2.5],.6);assert len(empty['points'])==0 and not np.any(empty['depth_valid'])
    assert np.all(np.asarray(D435iSensor().d)[:,0]>0),'camera must not see behind'
    results=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=D435iSensor().capture(make_scene('empty'),[-7,0,2.5],i*.2)))
            r=w.request(dict(op='query',t=1.,points=[[-6,0,2.5],[-8,0,2.5]]));assert all(cell[0]==2 for cell in r['cells']);results.append(dict(backend=backend,empty_is_unknown=True))
    save(a.out/'summary.json',dict(passed=True,backends=results,checks=['Z-depth projection','nearest actual pixel','invalid remains unknown','clock mismatch reject','capture pose mismatch reject','no rear rays']))
    print('D435i projection/unknown/time contracts passed',flush=True)

if __name__=='__main__':main()
