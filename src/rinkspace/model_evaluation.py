"""Small, local freeze contract for a new-source model inference run."""
from datetime import datetime, timezone
from pathlib import Path
import platform
from .io import dump_json,load_json,sha256
from .workspace import load_index,source_path

CORE_FILES = ('model_localization.py','curve_registration.py','event_model.py',
              'continuous_registration.py','local_registration.py','masked_photometric.py',
              'geometry.py','rink_reference.py','model_evaluation.py','foreground_rows.py')

def freeze_evaluation(root,model_path,windows_path,output,*,runtime_config=None):
    root=Path(root).resolve();model_path=Path(model_path).resolve()
    windows=load_json(windows_path)['windows']
    chosen=[w for w in windows if w['role']=='new_source_inference_after_freeze']
    if not chosen:raise ValueError('No preselected new-source evaluation windows')
    model=load_json(model_path)
    sources={}
    for w in chosen:
        source=w['source_id'];index=load_index(root,source)
        digest=sha256(source_path(root,source))
        if digest!=index['video_id']:raise ValueError('Source has changed since indexing')
        if digest in {v['source_sha256'] for v in model['reference_views']}:
            raise ValueError('Evaluation source already occurs in model references')
        sources[source]=digest
    files={}
    for name in CORE_FILES:
        path=root/'src/rinkspace'/name
        if path.exists():files[path.relative_to(root).as_posix()]=sha256(path)
    assets={}
    for view in model['reference_views']:
        image=(model_path.parent/view['image_path']).resolve(strict=True)
        if not image.is_relative_to(root):raise ValueError('Model reference must be inside project')
        assets[image.relative_to(root).as_posix()]=sha256(image)
    lock={'schema':'rinkspace.frozen-model-evaluation.v1',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'model_path':model_path.relative_to(root).as_posix(),'model_sha256':sha256(model_path),
        'model_id':model['model_id'],'model_revision':model['revision'],
        'windows':chosen,'sources':sources,'core_file_sha256':files,'reference_asset_sha256':assets,
        'runtime_config':runtime_config or {'refinement_motion_model':'affine','reobserve_every':10},
        'window_selection_file':str(windows_path),'window_selection_sha256':sha256(windows_path),
        'protocol':'Source-only windows selected before inference; no evaluation pixels may fit the model or tune the frozen locator. Failure frames stay in the denominator.',
        'evaluation_outputs_seen_at_freeze':False}
    dump_json(output,lock);return lock

def check_evaluation(root,lock_path,model_path,source,start_s,end_s,cuts,*,runtime_config=None):
    root=Path(root).resolve();lock=load_json(lock_path)
    if sha256(model_path)!=lock['model_sha256']:raise ValueError('Frozen model bytes changed')
    for name,digest in lock['core_file_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen localization code changed: '+name)
    if not lock.get('reference_asset_sha256'):raise ValueError('Frozen reference images are missing')
    for name,digest in lock['reference_asset_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen model reference image changed: '+name)
    config=runtime_config or {'refinement_motion_model':'affine','reobserve_every':10}
    if config!=lock['runtime_config']:raise ValueError('Frozen runtime configuration changed')
    match=[w for w in lock['windows'] if w['source_id']==source and
           w['start_s']==start_s and w['end_s']==end_s and list(w.get('cuts',[]))==list(cuts)]
    if len(match)!=1:raise ValueError('Window or cuts differ from frozen source-only selection')
    if sha256(source_path(root,source))!=lock['sources'].get(source):
        raise ValueError('Frozen evaluation source bytes changed')
    return {'evaluation_lock_sha256':sha256(lock_path),'evaluation_frozen_at_utc':lock['frozen_at_utc'],
            'role':'new_source_inference_after_freeze','model_and_code_checked_before_inference':True,
            'independent_geometry_validation':False}
