"""Direct RS replay of already qualified original corner tracks outside this batch.

No closest-point reprofile, added observation, frame interpolation or rejection
by model residual is performed. Prior calibration may have used these sources;
only exclusion from the current batch is claimed.
"""
import json
from pathlib import Path
import numpy as np
from .source_quality_compare_v1 import _camera,_metric,_windows,_ref,_sha
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene

def evaluate(model_path,source_table,output):
    model_path=Path(model_path).resolve();source_table=Path(source_table).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Immutable validation output exists')
    model=json.loads(model_path.read_text(encoding='utf8'));rows=[json.loads(s)for s in source_table.read_text(encoding='utf8').splitlines()if s.strip()]
    candidates=len(rows);rows=[r for r in rows if not r['selected_g6']]
    refs={};keys=set()
    for r in rows:
        key=(r['shot_id'],r['native_pts'],r['target_id'])
        if key in keys:raise ValueError('Duplicate same entity/native exposure')
        keys.add(key)
        if not r['source_geometry_eligible']or r['surface']!='board'or not r['known_world_corner_correspondence']or r['source_from_model']or r['world_or_H_used']:
            raise ValueError('Not a qualified static board corner source')
        d=r['source_journal'];f=Path(d['file'])
        if str(f)not in refs:
            if _sha(f)!=d['sha256']:raise ValueError('Changed original source qualification')
            refs[str(f)]=d
    scene=NativeScene(as_event_view(model,'s01'));lookup={}
    for mi,m in enumerate(model['materials']):
        for pi,p in enumerate(m['paths']):lookup[(m['label'],p['id'])]=(mi,pi)
    X=[];uv=[];pred=np.full((len(rows),2),np.nan);valid=np.zeros(len(rows),bool)
    for r in rows:
        mi,pi=lookup[(r['target_id'],r['existing_point_path'])];p=scene.path(mi,pi)
        if not p.get('physical_point')or len(p.get('points',[]))!=1:raise ValueError('Corner became a sliding entity')
        X.append(scene.world(mi,pi,0.));uv.append(r['qualified_source_uv'])
    X=np.asarray(X);uv=np.asarray(uv);shots=np.asarray([r['shot_id']for r in rows]);pts=np.asarray([r['native_pts']for r in rows]);entities=np.asarray([r['target_id']for r in rows]);exceptions={}
    for sid in np.unique(shots):
        sel=shots==sid
        try:
            camera=_camera(model,sid);pr=camera.project_rs(X[sel],(pts[sel]-model['shots'][sid]['time_origin_pts'])/12800.)
            pred[sel]=pr['uv'];valid[sel]=pr['valid']
        except Exception as e:exceptions[str(sid)]=type(e).__name__+': '+str(e)
    error=np.linalg.norm(pred-uv,axis=1)
    byshot={}
    for sid in np.unique(shots):
        sel=shots==sid;framepts=np.arange(int(pts[sel].min()),int(pts[sel].max())+256,256)
        ids=np.searchsorted(framepts,pts[sel]);good=valid[sel]&np.isfinite(error[sel]);sse=np.zeros(len(framepts));den=np.zeros(len(framepts),int);bad=den.copy()
        np.add.at(den,ids,1);np.add.at(bad,ids,~good);np.add.at(sse,ids,np.where(good,error[sel]**2,0.))
        byshot[str(sid)]=dict(**_metric(error[sel],valid[sel]),exposures=len(set(pts[sel].tolist())),temporal=_windows(framepts,sse,den,bad,den>0))
    report=dict(schema='rinkfit.current_batch_excluded_fixed_corner_replay.v1',model=_ref(model_path),source_table=_ref(source_table),
        source_qualification_refs=list(refs.values()),source_candidate_rows=candidates,current_selected_rows_excluded=candidates-len(rows),
        validation_rows=len(rows),all=_metric(error,valid),by_shot=byshot,
        by_entity={str(e):_metric(error[entities==e],valid[entities==e])for e in np.unique(entities)},exceptions=exceptions,
        actual_fixed_entity_2D_UV=True,RS_only_no_station_alpha_reprofile=True,all_failure_denominators_retained=True,
        excludes_current_batch_selected_frames=True,never_seen_by_any_prior_calibration_claimed=False,
        source_UV_moved_by_model=False,quality_accepted=False,producer=_ref(__file__))
    out.mkdir(parents=True);np.savez_compressed(out/'arrays.npz',predicted_uv=pred,qualified_source_uv=uv,world_xyz=X,native_pts=pts,shot_id=shots,error_px=error,valid=valid,target_id=entities)
    report['arrays']=_ref(out/'arrays.npz');(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report
