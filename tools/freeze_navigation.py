#!/usr/bin/env python3
"""Record the actual source/config before the held-out benchmark."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
target=Path(sys.argv[1]);target.parent.mkdir(parents=True,exist_ok=True)
files=list((ROOT/'src/pointcloud_nav').rglob('*'))+list((ROOT/'config').glob('navigation*'))
files+=list((ROOT/'tools').glob('*navigation*.py'))+[ROOT/'build_navigation.sh',ROOT/'run_navigation.sh']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and '__pycache__' not in str(p)}
data=dict(parent_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),
    development_seeds=[11],held_out_seeds=[41,42,43],scenes=['empty','wall_gap','offset_wall','forest'],
    delays_s=[0.,.08],maximum_flight_time_s=35.,source_sha256=hashes)
target.write_text(json.dumps(data,indent=2)+'\n');print('Frozen',len(hashes),'source and config files to',target)
