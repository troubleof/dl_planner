#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];target=Path(sys.argv[1]);target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():raise FileExistsError(target)
files=list((ROOT/'src/pointcloud_nav').rglob('*'))+[ROOT/'config/navigation.json',ROOT/'build_navigation.sh',ROOT/'run_navigation.sh']
files+=list((ROOT/'tools').glob('*navigation*.py'))+list((ROOT/'tools').glob('*route*.py'))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and '__pycache__' not in str(p)}
manifest=dict(branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),source_sha256=hashes,
    development_seed=11,evaluation_seeds=[41,42,43,101,102],scenes=['empty','wall_gap','offset_wall','forest'],
    backends=['ego','rog'],policies=['full','reuse','repair'],delays=[0.,.08],expected_closed_loop_runs=96,
    note='EGO/ROG map generation and complete spline validation retained. full uses the previous sampled A* reference. reuse/repair use conservative voxel traversal for cached edges and local search.')
target.write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen',len(hashes),'files')
