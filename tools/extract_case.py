"""Verify and extract one retained case into a new local directory."""
import argparse, hashlib, json, tarfile
from pathlib import Path, PurePosixPath
from navigation_common import ROOT


def digest(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def extract(spec, destination):
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(f'Refuse overwriting {destination}')
    archive = ROOT/spec['archive']
    assert digest(archive) == spec['sha256']
    destination.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tf:
        seen = set()
        for member in tf:
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unexpected archive member: {member.name}')
            if member.name in seen: raise ValueError('Duplicate archive member')
            seen.add(member.name)
            data = tf.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == spec['files'][member.name]
            target = destination/Path(*rel.parts)
            assert target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(data)
    assert seen == set(spec['files'])
    replacements = [(spec['original_path'], str(destination)),
                    (spec['original_path'].split('/runs/')[0], str(ROOT))]
    def relocate(value):
        if isinstance(value,str):
            for old,new in replacements:
                if value==old or value.startswith(old+'/'):return new+value[len(old):]
            return value
        if isinstance(value,list): return [relocate(v) for v in value]
        if isinstance(value,dict): return {k:relocate(v) for k,v in value.items()}
        return value
    changed=0
    for file in destination.rglob('*'):
        if file.suffix=='.json':
            old=json.loads(file.read_text());new=relocate(old)
            if new!=old:file.write_text(json.dumps(new,indent=2)+'\n');changed+=1
        elif file.suffix=='.jsonl':
            rows=[json.loads(line) for line in file.read_text().splitlines()];new=relocate(rows)
            if new!=rows:file.write_text(''.join(json.dumps(row)+'\n' for row in new));changed+=1
    return dict(case=spec['id'],files=len(seen),relocated_json_files=changed,destination=str(destination),archive_unchanged=True)


def main():
    specs=json.loads((ROOT/'examples/cases/manifest.json').read_text())['cases']
    ap=argparse.ArgumentParser();ap.add_argument('case',choices=[s['id'] for s in specs]);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    print(json.dumps(extract(next(s for s in specs if s['id']==args.case),args.out),indent=2))


if __name__=='__main__': main()
