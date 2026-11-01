"""Offline contract tests and sampler timing; does not contact a flight vehicle."""
import argparse,json,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from navigation_common import save
from px4_execution_contract import FrameAlignment,messages,CommittedSchedule

def slow_planner():
    # An independent planner process can be busy without blocking executor ticks.
    end=time.perf_counter()+.65
    while time.perf_counter()<end:np.linalg.norm(np.arange(1000,dtype=float))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    pairs=json.loads((a.case/'plans.json').read_text());pair=next(p for p in pairs if p['backup']['start']<p['nominal']['start']+p['nominal']['duration']);plans=[pair['nominal'],pair['backup']];start=plans[0]['start'];end=plans[-1]['start']+plans[-1]['duration'];schedule=CommittedSchedule(plans,start,end+.2,1)
    frame=FrameAlignment([10,20,30],0);hb,sp=messages(dict(p=[11,22,33],v=[1,2,3],a=[4,5,6]),0,.2,123,frame);assert sp['position']==[2.,1.,-3.] and sp['velocity']==[2.,1.,-3.] and sp['acceleration']==[5.,4.,-6.];assert sp['yaw']==np.pi/2 and sp['yawspeed']==-.2 and hb['position'] and not hb['velocity']
    rotated=FrameAlignment([0,0,0],np.pi/2);assert np.allclose(rotated.ned([1,0,0]),[1,0,0],atol=1e-12)
    timings=[]
    for t in np.linspace(start,end,1000):
        before=time.perf_counter();state,kind=schedule.sample(float(t));messages(state,0.,0.,int(t*1e6),rotated);timings.append((time.perf_counter()-before)*1000)
    process=mp.get_context('spawn').Process(target=slow_planner);process.start();wall=time.perf_counter();ticks=[];kinds=[]
    # 50 Hz real wall-clock dry run. Start the clock at nominal activation;
    # the precommitted stop activates even while no new plan can arrive.
    for i in range(60):
        deadline=wall+i*.02;remaining=deadline-time.perf_counter()
        if remaining>0:time.sleep(remaining)
        now=time.perf_counter();state,kind=schedule.sample(min(start+i*.02,end));messages(state,0.,0.,int((start+i*.02)*1e6),rotated);ticks.append(now);kinds.append(kind)
    process.join();assert process.exitcode==0 and 'backup' in kinds
    schedule.revoke()
    try:schedule.sample(start);raise AssertionError('revocation ignored')
    except RuntimeError:pass
    schedule=CommittedSchedule(plans,start,end+.2,1)
    try:schedule.sample(end+.21);raise AssertionError('expired validity ignored')
    except RuntimeError:pass
    report=dict(checks=['position/velocity/acceleration frame conversion','yaw and yaw-rate sign','nonzero frame origin/heading','PVA-continuous precommitted backup','independent 650 ms busy planner process','revoked and expired schedules refused'],sampler_ms=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),p99=float(np.percentile(timings,99)),maximum=max(timings)),wall_tick_interval_ms=dict(p95=float(np.percentile(np.diff(ticks)*1000,95)),maximum=float(max(np.diff(ticks)*1000))),ticks=len(ticks),backup_ticks=kinds.count('backup'),limitations=['Transport-free desktop test, not PX4 SITL or onboard timing','No ROS/DDS publisher, state estimator or flight controller included','A supervisor must handle lost validity; no guaranteed emergency escape is invented'])
    save(a.out/'result.json',report);print(json.dumps(report))

if __name__=='__main__':main()
