"""Apply a source-checked ad revision, explicitly bound to the frozen baseline."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def apply(baseline,increment,output,state=None):
    baseline=Path(baseline).resolve();increment=Path(increment).resolve();output=Path(output).resolve()
    if output==baseline or output.is_relative_to(baseline) or output==increment:raise ValueError('Use a separate output directory')
    manifest=read(increment/'increment_manifest.json');digest=sha(baseline/'point_observations.npz')
    if digest!=manifest['baseline_point_input_sha256']:raise ValueError('Wrong baseline or already applied increment; 002 replaces 001 and is applied to the frozen incomplete baseline')
    for e in manifest['files']:
        if sha(increment/e['path'])!=e['sha256']:raise ValueError('Checked increment changed: '+e['path'])
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be new or empty')
    z=dict(np.load(baseline/'point_observations.npz'));a=dict(np.load(increment/'ads_only/point_observations.npz'));m=read(baseline/'mapping.json');am=read(increment/'ads_only/mapping.json')
    for key in ['scene_revision','camera_sha256','control_sha256']:
        if m[key]!=am[key]:raise ValueError('Current native authority differs: '+key)
    if set(z)!=set(a):raise ValueError('Native observation schema mismatch')
    oldpaths=np.asarray(m['path_ids'])[z['path_indices']];ads=np.array([p.startswith('print_instance/Ad') for p in oldpaths]);first=int(manifest['source_scope']['first_pts']);last=int(manifest['source_scope']['last_pts'])
    keep=np.flatnonzero((~ads)&(z['pts']>=first)&(z['pts']<=last))
    if np.any(a['pts']<first) or np.any(a['pts']>last):raise ValueError('Ad observations outside the authorized source interval')
    paths=list(m['path_ids']);lookup={p:i for i,p in enumerate(paths)};apaths=np.asarray(am['path_ids'])[a['path_indices']]
    for p in apaths:
        if p not in lookup:lookup[str(p)]=len(paths);paths.append(str(p))
    mapped=dict(a,path_indices=np.array([lookup[p] for p in apaths],np.int32));merged={k:np.concatenate([z[k][keep],mapped[k]]) for k in z}
    if len(set(merged['observation_ids']))!=len(merged['pts']) or len(set(merged['source_rows']))!=len(merged['pts']):raise ValueError('Duplicate observation identity')
    for k in z:
        if not np.array_equal(merged[k][:len(keep)],z[k][keep]):raise ValueError('Retained non-ad observation changed: '+k)
    init=dict(np.load(baseline/'initialized_points.npz'));ai=dict(np.load(increment/'ads_only/initialized_points.npz'));newinit={k:np.concatenate([v[keep],ai[k]]) for k,v in init.items()}
    snapshot=None;state_sha=None
    if state is not None:
        state=Path(state).resolve();state_sha=sha(state);snapshot=dict(np.load(state))
        if str(snapshot['point_input_sha256'])!=digest or not np.array_equal(snapshot['source_rows'],z['source_rows']):raise ValueError('Numeric state belongs to another input')
        if sha(state)!=state_sha:raise ValueError('Numeric state changed during read')
    output.mkdir(parents=True,exist_ok=True);np.savez(output/'point_observations.npz',**merged);np.savez(output/'initialized_points.npz',**newinit)
    result_sha=sha(output/'point_observations.npz')
    m.update(path_ids=paths,selected_rows=len(merged['pts']),source_rows=int(merged['source_rows'].max()+1),frames=len(np.unique(merged['pts'])),current_point_input=str(output/'point_observations.npz'),fit_started=False,source_interval=manifest['source_scope'],advertisement_source_revision=dict(increment_id=manifest['increment_id'],manifest_sha256=sha(increment/'increment_manifest.json')),visual_mapping_review='Per-point actual source pattern, material-side and current native entity checks; user visual acceptance pending')
    m['features']=[x for x in m.get('features',[]) if not str(x.get('label',x.get('path_id',''))).startswith(('Ad','print_instance/Ad'))]+read(increment/'ads_only/native_entities.json')['entities']
    write(output/'mapping.json',m)
    if snapshot is not None:
        original=dict(snapshot);stations=np.concatenate([snapshot['stations'][keep],ai['stations']]);snapshot.update(stations=stations,source_rows=merged['source_rows'],point_input_sha256=np.array(result_sha))
        for k in original:
            if k not in ['stations','source_rows','point_input_sha256'] and not np.array_equal(snapshot[k],original[k]):raise ValueError('Native global parameter changed')
        np.savez(output/'rebound_numeric_state.npz',**snapshot)
        write(output/'rebound_numeric_state.json',dict(source_state=str(state),source_state_sha256=state_sha,global_native_parameters_bitwise_preserved=True,retained_nonad_stations_unchanged=len(keep),ad_stations='Initialized only on the independently identified native entity',optimizer_steps=0))
    receipt=dict(schema='rinkspace.current_point_input_ad_replacement.v1',input=str(output/'point_observations.npz'),sha256=result_sha,baseline_point_input_sha256=digest,increment_id=manifest['increment_id'],rows=len(merged['pts']),new_ad_rows=len(a['pts']),unchanged_nonad_rows=len(keep),outside_new_source_interval_rows=int(np.count_nonzero((z['pts']<first)|(z['pts']>last))),operations=manifest['operations'],source_interval=manifest['source_scope'],all_retained_nonad_fields_bitwise_preserved=True,native_parameters_changed=False,optimizer_steps=0)
    write(output/'point_input.json',receipt)
    if sha(baseline/'point_observations.npz')!=digest:raise ValueError('Baseline changed during application')
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--increment',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--state',type=Path);a=p.parse_args();print(json.dumps(apply(a.baseline,a.increment,a.output,a.state)))
