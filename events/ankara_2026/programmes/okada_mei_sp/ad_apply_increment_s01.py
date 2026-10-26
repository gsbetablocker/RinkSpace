"""Apply an additive S01 observation delta to a separate prepared input directory."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def apply(baseline,increment,output,state=None):
    baseline=Path(baseline).resolve();increment=Path(increment).resolve();output=Path(output).resolve()
    if output==baseline or output.is_relative_to(baseline) or output==increment:raise ValueError('Output must be separate from input and increment')
    manifest=read(increment/'increment_manifest.json')
    if manifest['operations']!={'add':manifest['added_rows'],'rebind':0,'remove':0}:raise ValueError('Unsupported delta operations')
    for entry in manifest['files']:
        if sha(increment/entry['path'])!=entry['sha256']:raise ValueError('Increment file changed: '+entry['path'])
    input_hash=sha(baseline/'point_observations.npz')
    if input_hash!=manifest['baseline_point_input_sha256']:raise ValueError('Wrong baseline or increment already applied')
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be new or empty')
    z=dict(np.load(baseline/'point_observations.npz',allow_pickle=False));a=dict(np.load(increment/'additions.npz',allow_pickle=False))
    m=read(baseline/'mapping.json');am=read(increment/'mapping.json')
    if m['scene_revision']!=am['scene_revision']:raise ValueError('Native scene mismatch')
    if set(z)!=set(a):raise ValueError('Observation schema mismatch')
    if set(z['observation_ids'].astype(str))&set(a['observation_ids'].astype(str)):raise ValueError('Repeated source observation identity')
    paths=list(m['path_ids']);lookup={p:i for i,p in enumerate(paths)};apaths=np.asarray(am['path_ids'])[a['path_indices']]
    for path in apaths:
        if path not in lookup:lookup[path]=len(paths);paths.append(str(path))
    mapped=dict(a,path_indices=np.array([lookup[p] for p in apaths],np.int32))
    merged={k:np.concatenate([z[k],mapped[k]]) for k in z}
    for k in z:
        if not np.array_equal(merged[k][:len(z['pts'])],z[k]):raise ValueError('Baseline field changed: '+k)
    init=dict(np.load(baseline/'initialized_points.npz',allow_pickle=False));ainit=dict(np.load(increment/'initialized_points.npz',allow_pickle=False))
    mergedinit={k:np.concatenate([v,ainit[k]]) for k,v in init.items()}
    state_data=None;state_hash=None
    if state is not None:
        state=Path(state).resolve();state_hash=sha(state);state_data=dict(np.load(state,allow_pickle=False))
        if str(state_data.get('point_input_sha256',''))!=input_hash or not np.array_equal(state_data.get('source_rows'),z['source_rows']):raise ValueError('Chosen numeric state does not belong to this baseline')
        if sha(state)!=state_hash:raise ValueError('Numeric state changed while reading')
    output.mkdir(parents=True,exist_ok=True);np.savez(output/'point_observations.npz',**merged);np.savez(output/'initialized_points.npz',**mergedinit)
    output_hash=sha(output/'point_observations.npz');m.update(path_ids=paths,selected_rows=len(merged['pts']),frames=len(np.unique(merged['pts'])),source_rows=int(max(m.get('source_rows',int(z['source_rows'].max()+1)),merged['source_rows'].max()+1)),current_point_input=str(output/'point_observations.npz'),applied_increment=dict(id=manifest['increment_id'],manifest_sha256=sha(increment/'increment_manifest.json')),fit_started=False)
    m['advertisement_increment']=dict(manifest=manifest,baseline_observation_arrays_preserved=True,new_source_measurement_ledgers=manifest.get('source_measurements',[]));m['visual_mapping_review']='Increment source audit completed; user video review pending'
    write(output/'mapping.json',m)
    total_ad=sum(p.startswith('print_instance/Ad') for p in np.asarray(paths)[merged['path_indices']])
    receipt=dict(schema='rinkspace.current_point_input_ad_increment.v1',input=str(output/'point_observations.npz'),sha256=output_hash,baseline=str(baseline),baseline_point_input_sha256=input_hash,increment_id=manifest['increment_id'],rows=len(merged['pts']),added_ad_rows=len(a['pts']),new_ad_rows=total_ad,unchanged_nonad_rows=len(merged['pts'])-total_ad,all_baseline_observation_arrays_preserved=True,native_parameters_changed=False,solver_run=False)
    if state_data is not None:
        old_state=dict(state_data);state_data.update(stations=np.concatenate([state_data['stations'],ainit['stations']]),source_rows=merged['source_rows'],point_input_sha256=np.array(output_hash))
        for k in old_state:
            if k not in ('stations','source_rows','point_input_sha256') and not np.array_equal(state_data[k],old_state[k]):raise ValueError('Global state changed')
        np.savez(output/'rebound_numeric_state.npz',**state_data)
        write(output/'rebound_numeric_state.json',dict(source_state=str(state),source_state_sha256=state_hash,output_sha256=sha(output/'rebound_numeric_state.npz'),global_parameters_bitwise_preserved=True,existing_stations_preserved=True,new_stations='increment initialization; solver refines finite curves before first step',optimizer_resumed=False))
    if sha(baseline/'point_observations.npz')!=input_hash:raise ValueError('Baseline changed while applying delta')
    write(output/'point_input.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--increment',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--state',type=Path);a=p.parse_args()
    print(json.dumps(apply(a.baseline,a.increment,a.output,a.state)),flush=True)
