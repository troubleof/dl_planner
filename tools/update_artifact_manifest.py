"""Index retained binary assets; does not copy or archive the workspace."""
import json
from navigation_common import ROOT
from extract_case import digest

if __name__=='__main__':
    files={}
    for f in ROOT.rglob('*'):
        rel=f.relative_to(ROOT)
        if {'runs','build','install','log','.git','.venv','__pycache__'}.intersection(rel.parts):continue
        if f.is_file() and (f.suffix in ['.pt','.npz','.torchscript'] or f.name.endswith('.tar.gz')):
            if f.stat().st_size>=50*1024*1024:raise ValueError(f'Large retained asset needs a storage decision: {rel}')
            files[str(rel)]=dict(bytes=f.stat().st_size,sha256=digest(f))
    destination=ROOT/'provenance/artifacts.json'
    destination.write_text(json.dumps(dict(format='retained-artifacts-v1',files=files),indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files.values()))))
