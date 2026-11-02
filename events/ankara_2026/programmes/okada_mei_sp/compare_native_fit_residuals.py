"""Compare identical source rows, retaining both old and newly introduced tails."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stats(values):
    finite=values[np.isfinite(values)]
    return dict(rows=len(values),invalid=int(len(values)-len(finite)),
        rms=float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
        median=float(np.median(finite)) if len(finite) else None,
        p95=float(np.quantile(finite,.95)) if len(finite) else None,
        maximum=float(np.max(finite)) if len(finite) else None,
        above_20px=int(np.sum(finite>20)),above_30px=int(np.sum(finite>30)))


def compare(args):
    m=read(args.run_root/'joint_manifest.json');mapping=read(args.run_root/'mapping.json')
    with np.load(args.run_root/'point_observations.npz') as z:
        pts=z['pts'];paths=z['path_indices'];uv=z['uv']
    values=[]
    for folder in (args.before,args.after):
        q=read(folder/'current_quality.json')
        with np.load(folder/'current_residuals.npz') as z:
            assert np.array_equal(z['rows'],np.arange(len(pts))), 'Full identical source rows required'
            errors=z['errors']
        cost=float(.5*np.dot(errors,errors));assert abs(cost-q['cost'])<1e-5
        with np.load(folder/'accepted_pixels_latest.npz') as z:
            assert str(z['point_input_sha256'])==m['point_input_sha256']
        values.append(errors)
    before,after=values
    scene=read(m['scene_path']);labels={p['id']:p['label'] for p in scene['paths']}
    pair=lambda mask:dict(before=stats(before[mask]),after=stats(after[mask]))
    output=dict(input_sha256=m['point_input_sha256'],
        before_state=dict(path=str(args.before/'accepted_pixels_latest.npz'),sha256=digest(args.before/'accepted_pixels_latest.npz')),
        after_state=dict(path=str(args.after/'accepted_pixels_latest.npz'),sha256=digest(args.after/'accepted_pixels_latest.npz')),
        full=pair(np.ones(len(pts),bool)),shots={},entities=[],focus=[],old_tail=pair(before>20),
        new_tail=pair((after>20)&(before<=20)),old_tail_improved_below_20=int(np.sum((before>20)&(after<=20))),
        new_above_20=int(np.sum((before<=20)&(after>20))),
        interpretation='Same measured rows and raw pixels; training residuals are not independent visual acceptance')
    rowshot=np.empty(len(pts),dtype='U3')
    for shot in m['shots']:
        a,b=shot['row_begin'],shot['row_stop'];rowshot[a:b]=shot['id'];mask=np.arange(a,b)
        output['shots'][shot['id']]=pair(mask)
        for index in np.unique(paths[mask]):
            selected=mask[paths[mask]==index];identity=mapping['path_ids'][index]
            output['entities'].append(dict(shot=shot['id'],path_id=identity,label=labels.get(identity,identity),**pair(selected)))
    windows=[('s02_fast','s02',47,49),('s07_fast','s07',112.5,114.3),
        ('s09_spin1','s09',156.5,160.3),('s09_bulge','s09',167.4,168),
        ('s09_spin2','s09',172.4,179.3),('s09_tail','s09',182.2,188.68)]
    for name,shot,a,b in windows:
        mask=(rowshot==shot)&(pts>=round(a*12800))&(pts<=round(b*12800))
        output['focus'].append(dict(id=name,shot=shot,source_seconds=[a,b],**pair(mask)))
    tail=np.flatnonzero((before>20)|(after>20))
    output['tail_rows']=[dict(input_row=int(i),shot=str(rowshot[i]),pts=int(pts[i]),
        source_seconds=float(pts[i]/12800),uv=uv[i].tolist(),path_id=mapping['path_ids'][paths[i]],
        label=labels.get(mapping['path_ids'][paths[i]],mapping['path_ids'][paths[i]]),
        before_px=float(before[i]),after_px=float(after[i])) for i in tail]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k] for k in ('full','shots','focus','old_tail','new_tail','old_tail_improved_below_20','new_above_20')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run-root','before','after','output'):p.add_argument('--'+k,type=Path,required=True)
    compare(p.parse_args())
