import argparse,json,time
from pathlib import Path
import numpy as np
from d435i_adapter import from_depth,OPTICAL_TO_WORLD
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rng=np.random.default_rng(20261031);depth=rng.integers(300,6000,(480,848),dtype=np.uint16);depth[rng.random(depth.shape)<.1]=0
    args=dict(intrinsics=dict(fx=448.,fy=433.,cx=423.5,cy=239.5,distortion='none'),depth_scale=.001,camera_to_world=OPTICAL_TO_WORLD,origin=[0,0,2.5],stamp=1.,pose_stamp=1.,clock_domain='navigation',block=8)
    values=[]
    for i in range(550):
        t=time.perf_counter();cloud=from_depth(depth,**args);duration=(time.perf_counter()-t)*1000
        if i>=50:values.append(duration)
    save(a.out,dict(native_shape=[480,848],output_shape=cloud['depth_shape'],maximum_points=6360,repeats=500,
        ms={**{f'p{p}':float(np.percentile(values,p)) for p in [50,95,99]},'max':max(values)},
        scope='CPU synthetic native Z16 buffer including downsample, projection, lists. Excludes camera/USB, driver, state estimation and time synchronization; intrinsics here are benchmark fixtures.'))

if __name__=='__main__':main()
