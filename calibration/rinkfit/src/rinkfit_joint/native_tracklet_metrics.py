"""Same-source independent RS and observed-motion metrics of fixed entities.

A diagnostic/development comparison, not a claim of untouched held-out truth.
The source ledger stays unchanged; invalid projections remain in denominators.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .joint_model import as_event_view,graph_from_joint
from .camera_graph import CameraGraph
from rinkfit_start.native_scene import NativeScene


def _metric(values,valid):
    a=np.asarray(values,float);valid=np.asarray(valid,bool)&np.isfinite(a).all(axis=1)
    d=np.linalg.norm(a[valid],axis=1)
    return dict(total=len(a),valid=int(valid.sum()),invalid=int((~valid).sum()),
        rms_px=float(np.sqrt(np.mean(d*d))) if len(d) else None,
        median_px=float(np.median(d)) if len(d) else None,
        p95_px=float(np.quantile(d,.95)) if len(d) else None,
        max_px=float(d.max()) if len(d) else None)


def compare_native_tracklets(source_rows,models,output):
    out=Path(output)
    if out.exists():raise ValueError('New immutable actual-source comparison required')
    rows=list(read_jsonl(source_rows));n=len(rows)
    if not n or len({r['source_row_id'] for r in rows})!=n:raise ValueError('Unique nonempty actual source ledger required')
    shots=np.array([r['shot_id']for r in rows]);pts=np.array([r['native_pts']for r in rows],np.int64);observed=np.array([r['current_uv']for r in rows],float)
    tracks=defaultdict(list)
    for i,r in enumerate(rows):
        if not r['point_correspondence']or not r.get('entity_station_fixed'):raise ValueError('Only the same fixed physical point can form a 2D track')
        tracks[(r['shot_id'],r['physical_track_id'],r['target_id'])].append(i)
    pairs=[];triples=[]
    for key,ii in tracks.items():
        ii.sort(key=lambda i:pts[i])
        pairs.extend((a,b) for a,b in zip(ii[:-1],ii[1:]) if pts[b]-pts[a]==256)
        triples.extend((a,b,c) for a,b,c in zip(ii[:-2],ii[1:-1],ii[2:]) if pts[b]-pts[a]==pts[c]-pts[b]==256)
    pairs=np.asarray(pairs,int).reshape(-1,2);triples=np.asarray(triples,int).reshape(-1,3)
    report=dict(schema='rinkfit.native_fixed_entity_same_source_motion_comparison.v1',source_rows=dict(file=str(Path(source_rows).resolve()),sha256=file_sha256(source_rows)),source_row_count=n,source_exposures=len(set(pts.tolist())),source_clock='native 1/12800, PTS step 256, 50fps',same_source_ID_UV_operator_and_fixed_entity_for_all_models=True,projection_semantics='independently solved native RS at the fixed physical entity; no station profiling and no explicit checkpoint auxiliary inference',comparison_role='development source evidence; not an untouched independent validation certificate',native_motion_pair_denominator=len(pairs),native_motion_triple_denominator=len(triples),models={})
    per_source=[]
    for name,filename in models.items():
        model=read_json(filename);camera=CameraGraph(graph_from_joint(model));world=np.empty((n,3))
        for sid in sorted(set(shots)):
            scene=NativeScene(as_event_view(model,str(sid)))
            for mi in sorted({int(rows[i]['joint_material_index'])for i in np.flatnonzero(shots==sid)}):
                ii=np.array([i for i in np.flatnonzero(shots==sid) if int(rows[i]['joint_material_index'])==mi])
                if any(model['materials'][mi]['label']!=rows[i]['target_id']for i in ii):raise ValueError('Physical material identity changed across model comparison')
                paths=[j for j,p in enumerate(model['materials'][mi]['paths'])if p.get('physical_point')]
                if len(paths)!=1:raise ValueError('Fixed entity path is not unique')
                world[ii]=scene.world(mi,paths[0],np.zeros(len(ii)))
        projection=camera.project_rows(shots,pts,world);predicted=np.asarray(projection['uv']);valid=np.asarray(projection['valid'],bool);error=predicted-observed
        dp=error[pairs[:,1]]-error[pairs[:,0]];vp=valid[pairs].all(axis=1)
        dt=error[triples[:,2]]-2*error[triples[:,1]]+error[triples[:,0]];vt=valid[triples].all(axis=1)
        record=dict(model=dict(file=str(Path(filename).resolve()),sha256=file_sha256(filename)),position=_metric(error,valid),motion_difference=_metric(dp,vp),second_motion_difference=_metric(dt,vt),by_shot={})
        for sid in ['s01','s02','s03','s04','s05','s06','s07','s08','s09']:
            take=shots==sid;pm=shots[pairs[:,1]]==sid;tm=shots[triples[:,1]]==sid
            record['by_shot'][sid]=dict(position=_metric(error[take],valid[take]),motion_difference=_metric(dp[pm],vp[pm]),second_motion_difference=_metric(dt[tm],vt[tm]))
        def worst(a,good,indices):
            if not len(a) or not np.any(good):return None
            dd=np.linalg.norm(a,axis=1);j=int(np.argmax(np.where(good,dd,-np.inf)));ix=np.atleast_1d(indices[j]).astype(int)
            return dict(error_px=float(dd[j]),source_row_ids=[rows[i]['source_row_id']for i in ix],native_pts=[int(pts[i])for i in ix],shot_id=str(shots[ix[0]]),target_id=rows[ix[0]]['target_id'],original_uv=[observed[i].tolist()for i in ix],predicted_uv=[predicted[i].tolist()for i in ix])
        record['worst_position']=worst(error,valid,np.arange(n));record['worst_motion_difference']=worst(dp,vp,pairs);record['worst_second_motion_difference']=worst(dt,vt,triples)
        for i,r in enumerate(rows):per_source.append(dict(model_name=name,source_row_id=r['source_row_id'],shot_id=r['shot_id'],target_id=r['target_id'],physical_track_id=r['physical_track_id'],native_pts=int(pts[i]),source_uv=observed[i].tolist(),predicted_uv=predicted[i].tolist() if np.isfinite(predicted[i]).all()else None,independent_RS_valid=bool(valid[i]),native_error_uv=error[i].tolist() if np.isfinite(error[i]).all()else None))
        report['models'][name]=record
    out.mkdir(parents=True);write_jsonl(out/'per_source.jsonl.gz',per_source);write_json(out/'comparison.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--model',action='append',required=True,help='name=absolute model path');p.add_argument('--out',required=True);a=p.parse_args()
    models={}
    for item in a.model:
        name,path=item.split('=',1)
        if name in models:raise ValueError('Model label repeated')
        models[name]=path
    print(json.dumps(compare_native_tracklets(a.source,models,a.out),ensure_ascii=False),flush=True)
