#!/usr/bin/env python3
import argparse,unittest
from navigation_common import *

class ExecutionContract(unittest.TestCase):
    def test_envelope_and_grid(self):
        self.assertAlmostEqual(RADIUS,.3272634126602354);self.assertEqual(math.ceil(RADIUS/.1),4)
    def test_late_plan_not_retimed(self):
        p=dict(start=1.3);s=stationary([0,0,2.5])
        self.assertEqual(CommitGate.admit(p,1.28,2,2,s,s),(False,'deadline'));self.assertEqual(p['start'],1.3)
    def test_map_and_handoff_versions(self):
        p=dict(start=1.3);s=stationary([0,0,2.5]);changed=stationary([.01,0,2.5])
        self.assertEqual(CommitGate.admit(p,1.1,2,3,s,s),(False,'map_changed'))
        self.assertEqual(CommitGate.admit(p,1.1,2,2,s,changed),(False,'handoff_state_changed'))
    def test_sensor_occlusion(self):
        s=Sensor(360);world=make_scene('offset_wall');c=s.capture(world,[-2,0,2.5],0)
        points=np.asarray(c['points']);horizontal=points[np.abs(points[:,2]-2.5)<1e-6]
        self.assertAlmostEqual(horizontal[0,0],-.25,places=5)
    def test_terminal_roundoff_stays_finite(self):
        p=dict(start=1.,dt=.1,duration=.3000000000000001,controls=[[0,0,2.5]]*3+[[1,0,2.5]]*3)
        state=Curve(p).state(2.)
        self.assertTrue(np.all(np.isfinite(state['p'])));self.assertEqual(state['v'],[0.,0.,0.])

def map_contract(backend,out):
    tests=[]
    with Worker(backend,out) as w:
        origin=[-2.05,.05,2.55];hit=[.05,.05,2.55];corner=[.45,.45,2.55]
        def query(points):return w.request(dict(op='query',t=20,points=points))['cells']
        def update(t,points,frame='world',now=None):return w.request(dict(op='update',t=t if now is None else now,cloud=dict(frame=frame,stamp=t,origin=origin,points=points)))
        assert query([hit])[0][0]==2;tests.append('initial_unknown')
        for i in range(4):update(i*.1,[hit])
        occupied=query([hit,corner]);assert occupied[0][0]==1 and occupied[1][1]==1,occupied;tests.append('hit_and_cube_corner')
        for i in range(4,18):update(i*.1,[[6.05,.05,2.55]])
        cleared=query([hit,corner]);assert cleared[0]==[0,0] and cleared[1][1]==0,cleared;tests.append('ray_free_and_inflation_deletion')
        for case,kwargs in [('wrong_frame',dict(t=2,points=[],frame='body')),('stale_stamp',dict(t=1.6,points=[])),('future_stamp',dict(t=3,points=[],now=2))]:
            try:update(**kwargs)
            except ValueError:tests.append(case)
            else:raise AssertionError(case+' admitted')
        for i in range(18,285):update(i*.1,[[6.05,.05,2.55]])
        assert query([hit])[0]==[0,0];tests.append('more_than_256_frames')
        scene=make_scene('empty');sensor=Sensor()
        for i in range(4):
            stamp=30.+i*.2;w.request(dict(op='update',t=stamp,cloud=sensor.capture(scene,scene['start'],stamp)))
        face=query([[-6.8,0.,2.5],[-6.8,0.,2.5-3.4e-9]])
        assert face[0]==face[1]==[0,0],face;tests.append('shared_voxel_face_tolerance')
    save(Path(out)/'checks.json',dict(backend=backend,passed=tests));return tests

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExecutionContract)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    for backend in ['ego','rog']:print(backend,map_contract(backend,args.out/backend),flush=True)
