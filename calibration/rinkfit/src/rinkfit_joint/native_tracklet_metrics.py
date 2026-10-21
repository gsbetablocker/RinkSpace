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
from .joint_replay import render_visibility
from rinkfit_start.native_scene import NativeScene


def _metric(values,valid):
    a=np.asarray(values,float);valid=np.asarray(valid,bool)&np.isfinite(a).all(axis=1)
    d=np.linalg.norm(a[valid],axis=1)
    return dict(total=len(a),valid=int(valid.sum()),invalid=int((~valid).sum()),
        rms_px=float(np.sqrt(np.mean(d*d))) if len(d) else None,
        median_px=float(np.median(d)) if len(d) else None,
        p95_px=float(np.quantile(d,.95)) if len(d) else None,
        max_px=float(d.max()) if len(d) else None)


def compare_native_tracklets(source_rows,models,output,*,scope_file=None,source_bundle=None):
    out=Path(output)
    if out.exists():raise ValueError('New immutable actual-source comparison required')
    rows=list(read_jsonl(source_rows));n=len(rows)
    if not n or len({r['source_row_id'] for r in rows})!=n:raise ValueError('Unique nonempty actual source ledger required')
    shots=np.array([r['shot_id']for r in rows]);pts=np.array([r['native_pts']for r in rows],np.int64);observed=np.array([r['current_uv']for r in rows],float)
    tracks=defaultdict(list)
    for i,r in enumerate(rows):
        shared=r.get('world_position_is_shared_track_latent') is True
        fixed=r.get('entity_station_fixed') or (shared and (
            (r.get('surface_class')=='ice' and r.get('surface_instance_id')=='ice') or
            (r.get('surface_class')=='board' and r.get('source_geometry_eligible') is True
             and r.get('entity_station')==0 and r.get('known_world_corner_correspondence') is False)))
        if not r['point_correspondence']or not fixed:raise ValueError('Only the same fixed physical point can form a 2D track')
        tracks[(r['shot_id'],r['physical_track_id'],r['target_id'])].append(i)
    pairs=[];triples=[]
    for key,ii in tracks.items():
        ii.sort(key=lambda i:pts[i])
        pairs.extend((a,b) for a,b in zip(ii[:-1],ii[1:]) if pts[b]-pts[a]==256)
        triples.extend((a,b,c) for a,b,c in zip(ii[:-2],ii[1:-1],ii[2:]) if pts[b]-pts[a]==pts[c]-pts[b]==256)
    pairs=np.asarray(pairs,int).reshape(-1,2);triples=np.asarray(triples,int).reshape(-1,3)
    report=dict(schema='rinkfit.native_fixed_entity_same_source_motion_comparison.v1',source_rows=dict(file=str(Path(source_rows).resolve()),sha256=file_sha256(source_rows)),source_row_count=n,source_exposures=len(set(pts.tolist())),source_clock='native 1/12800, PTS step 256, 50fps',same_source_ID_UV_operator_and_fixed_entity_for_all_models=True,projection_semantics='independently solved native RS at the fixed physical entity; no station profiling and no explicit checkpoint auxiliary inference',comparison_role='development source evidence; not an untouched independent validation certificate',native_motion_pair_denominator=len(pairs),native_motion_triple_denominator=len(triples),models={})
    report['motion_differences_are_diagnostic_only_not_additional_fit_factors']=True
    report['implementation_sha256']=file_sha256(__file__)
    report['validity_includes_whole_RS_certificate_physical_visibility_and_surface_domain']=True
    if source_bundle is not None:
        bundle=read_json(source_bundle)
        if bundle['source_rows_ref']['sha256']!=file_sha256(source_rows):raise ValueError('Motion diagnostic bundle binds another source ledger')
        report['qualified_source_bundle']=dict(file=str(Path(source_bundle).resolve()),sha256=file_sha256(source_bundle))
        report['original_source_generation']=bundle['source_generation']
        report['static_surface_qualification']=bundle['static_surface_qualification']
        generation_file=Path(bundle['source_generation'].get('file',bundle['source_generation'].get('path')))
        if file_sha256(generation_file)!=bundle['source_generation']['sha256']:raise ValueError('Motion diagnostic generation changed')
        report['source_actor_occlusion_ledgers']=read_json(generation_file).get('actor_ledgers',[])
    scope=read_json(scope_file)if scope_file is not None else None
    if scope is not None:
        if any(r.get('source_video_sha256')!=scope['source']['sha256']for r in rows):raise ValueError('Motion diagnostic source/scope identity differs')
        report['complete_programme_scope']=dict(file=str(Path(scope_file).resolve()),sha256=file_sha256(scope_file))
    source_gaps=[]
    for key,ii in tracks.items():
        for a,b in zip(ii[:-1],ii[1:]):
            if pts[b]-pts[a]!=256:source_gaps.append(dict(shot_id=key[0],physical_track_id=key[1],last_pts=int(pts[a]),next_pts=int(pts[b]),delta_pts=int(pts[b]-pts[a]),interpolated=False))
    report['source_track_gaps']=source_gaps
    per_source=[];per_pair=[]
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
        projection=camera.project_rows(shots,pts,world);predicted=np.asarray(projection['uv']);rs_valid=np.asarray(projection['valid'],bool)
        visible=np.zeros(n,bool);surface_valid=np.zeros(n,bool)
        for sid in sorted(set(shots)):
            ii=np.flatnonzero(shots==sid);scene=NativeScene(as_event_view(model,str(sid)))
            board=np.array([scene.path(int(rows[i]['joint_material_index']),next(j for j,p in enumerate(model['materials'][int(rows[i]['joint_material_index'])]['paths'])if p.get('physical_point')))['kind'] in (2,3,6,7)for i in ii])
            inside=scene.containment(world[ii,:2])<=1e-10
            surface_valid[ii]=np.where(board,(world[ii,2]>=-1e-10)&(world[ii,2]<=scene.height+1e-10),inside&(np.abs(world[ii,2])<=1e-10))
            for k in ii:
                if rows[k].get('world_position_is_shared_track_latent') is True and rows[k].get('surface_class')=='board':
                    mi=int(rows[k]['joint_material_index']);paths=model['materials'][mi]['paths']
                    if len(paths)!=1 or paths[0].get('kind')!=2 or not paths[0].get('physical_point') or paths[0].get('points')!=[[0.,0.]]:
                        raise ValueError('Shared unknown board point changed physical material role')
                    path=paths[0];q=np.asarray(model['blocks'][path['geometry_block']]['values'],float)
                    uv=np.array([q[2],q[5]]);ad=np.asarray(model['blocks'][path['advertisement_block']]['values'],float)
                    surface_valid[k] &= bool(np.isfinite(uv).all() and np.all(uv>=-1e-10) and np.all(uv<=1+1e-10)
                        and 0<ad[1]<=scene.perimeter+1e-10 and -1e-10<=ad[2]<=ad[3]<=scene.height+1e-10)

            changes=np.r_[0,np.flatnonzero(board[1:]!=board[:-1])+1,len(ii)]
            records=[dict(begin=int(lo),end=int(hi),label='',board_material_surface=bool(board[lo]))for lo,hi in zip(changes[:-1],changes[1:])]
            visible[ii]=render_visibility(camera.native_camera(str(sid)),scene,world[ii],records,camera.local_seconds(str(sid),pts[ii]),predicted[ii])
        valid=rs_valid&visible&surface_valid;error=predicted-observed
        dp=error[pairs[:,1]]-error[pairs[:,0]];vp=valid[pairs].all(axis=1)
        dt=error[triples[:,2]]-2*error[triples[:,1]]+error[triples[:,0]];vt=valid[triples].all(axis=1)
        record=dict(independent_RS_invalid_rows=int((~rs_valid).sum()),physical_visibility_invalid_rows=int((~visible).sum()),source_surface_domain_invalid_rows=int((~surface_valid).sum()),physical_visibility_and_finite_surface_checked=True,model=dict(file=str(Path(filename).resolve()),sha256=file_sha256(filename)),position=_metric(error,valid),motion_difference=_metric(dp,vp),second_motion_difference=_metric(dt,vt),by_shot={},by_track={},contiguous_source_segments=[])
        for sid in ['s01','s02','s03','s04','s05','s06','s07','s08','s09']:
            take=shots==sid;pm=shots[pairs[:,1]]==sid;tm=shots[triples[:,1]]==sid
            record['by_shot'][sid]=dict(position=_metric(error[take],valid[take]),motion_difference=_metric(dp[pm],vp[pm]),second_motion_difference=_metric(dt[tm],vt[tm]))
            if scope is not None:
                s=next(x for x in scope['shots']if x['id']==sid)
                possible=(s['last_pts']-s['first_pts'])//256
                measured={int(pts[a])for a,b in pairs[pm]}
                projected={int(pts[a])for (a,b),good in zip(pairs[pm],vp[pm])if good}
                record['by_shot'][sid]['unique_native_adjacent_pair_coverage']=dict(
                    possible_pairs=possible,source_qualified_pairs=len(measured),valid_physical_projection_pairs=len(projected),
                    unknown_source_motion_pairs=possible-len(measured),qualified_but_invalid_physical_pairs=len(measured-projected),
                    possible_pair_clock='native source step 256; unique exposure pairs, not track multiplicity',
                    no_source_at_this_shot=not bool(take.any()))
        for key,ii in sorted(tracks.items()):
            if np.max(np.ptp(world[ii],axis=0))>1e-12:raise ValueError('One source track was assigned varying physical world points')
            tid=key[1];pm=np.array([rows[a]['physical_track_id']==tid and rows[a]['shot_id']==key[0]for a,b in pairs]);tm=np.array([rows[a]['physical_track_id']==tid and rows[a]['shot_id']==key[0]for a,b,c in triples])
            record['by_track'][tid]=dict(shot_id=key[0],target_id=key[2],world_XYZ_L=world[ii[0]].tolist(),
                physical_position_kind=('shared_unknown_'+rows[ii[0]]['surface_class']+'_material_point'
                    if rows[ii[0]].get('world_position_is_shared_track_latent') is True else 'defined_entity_corner'),
                material_owner_id=rows[ii[0]].get('surface_instance_id'),
                known_world_corner_correspondence=rows[ii[0]].get('known_world_corner_correspondence',not rows[ii[0]].get('world_position_is_shared_track_latent',False)),
                first_pts=int(pts[ii[0]]),last_pts=int(pts[ii[-1]]),position=_metric(error[ii],valid[ii]),
                motion_difference=_metric(dp[pm],vp[pm]),second_motion_difference=_metric(dt[tm],vt[tm]))
            segments=[]
            for i in ii:
                if not segments or pts[i]-pts[segments[-1][-1]]!=256:segments.append([])
                segments[-1].append(i)
            for segment in segments:
                ix=np.asarray(segment,int);sp=np.c_[ix[:-1],ix[1:]]
                delta=error[sp[:,1]]-error[sp[:,0]]
                record['contiguous_source_segments'].append(dict(shot_id=key[0],physical_track_id=tid,
                    first_pts=int(pts[ix[0]]),last_pts=int(pts[ix[-1]]),source_row_count=len(ix),
                    position=_metric(error[ix],valid[ix]),motion_difference=_metric(delta,valid[sp].all(axis=1)),
                    source_gap_or_cut_bridged=False))
        candidates=[s for s in record['contiguous_source_segments']if s['motion_difference']['rms_px']is not None]
        record['worst_contiguous_source_segment_by_motion_rms']=max(candidates,key=lambda s:s['motion_difference']['rms_px'])if candidates else None
        def worst(a,good,indices):
            if not len(a) or not np.any(good):return None
            dd=np.linalg.norm(a,axis=1);j=int(np.argmax(np.where(good,dd,-np.inf)));ix=np.atleast_1d(indices[j]).astype(int)
            return dict(error_px=float(dd[j]),source_row_ids=[rows[i]['source_row_id']for i in ix],native_pts=[int(pts[i])for i in ix],shot_id=str(shots[ix[0]]),target_id=rows[ix[0]]['target_id'],physical_track_id=rows[ix[0]]['physical_track_id'],original_uv=[observed[i].tolist()for i in ix],predicted_uv=[predicted[i].tolist()for i in ix])
        record['worst_position']=worst(error,valid,np.arange(n));record['worst_motion_difference']=worst(dp,vp,pairs);record['worst_second_motion_difference']=worst(dt,vt,triples)
        for i,r in enumerate(rows):per_source.append(dict(model_name=name,source_row_id=r['source_row_id'],shot_id=r['shot_id'],target_id=r['target_id'],physical_track_id=r['physical_track_id'],native_pts=int(pts[i]),source_uv=observed[i].tolist(),predicted_uv=predicted[i].tolist() if np.isfinite(predicted[i]).all()else None,independent_RS_valid=bool(rs_valid[i]),physical_visibility_valid=bool(visible[i]),source_surface_domain_valid=bool(surface_valid[i]),physical_projection_valid=bool(valid[i]),native_error_uv=error[i].tolist() if np.isfinite(error[i]).all()else None,original_source_record=r))
        for j,(a,b)in enumerate(pairs):
            per_pair.append(dict(model_name=name,shot_id=rows[a]['shot_id'],physical_track_id=rows[a]['physical_track_id'],
                source_row_ids=[rows[a]['source_row_id'],rows[b]['source_row_id']],native_pts=[int(pts[a]),int(pts[b])],
                observed_delta_uv=(observed[b]-observed[a]).tolist(),predicted_delta_uv=(predicted[b]-predicted[a]).tolist()if np.isfinite(predicted[[a,b]]).all()else None,
                motion_difference_residual_uv=dp[j].tolist()if np.isfinite(dp[j]).all()else None,
                independent_RS_valid=bool(rs_valid[[a,b]].all()),physical_projection_valid=bool(vp[j]),used_as_extra_independent_fit_factor=False,
                original_FB_NCC_and_source_receipts_preserved_in='per_source.jsonl.gz'))
        report['models'][name]=record
    out.mkdir(parents=True);write_jsonl(out/'per_source.jsonl.gz',per_source);write_jsonl(out/'per_pair.jsonl.gz',per_pair);write_json(out/'comparison.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--model',action='append',required=True,help='name=absolute model path');p.add_argument('--out',required=True);p.add_argument('--scope');p.add_argument('--source-bundle');a=p.parse_args()
    models={}
    for item in a.model:
        name,path=item.split('=',1)
        if name in models:raise ValueError('Model label repeated')
        models[name]=path
    print(json.dumps(compare_native_tracklets(a.source,models,a.out,scope_file=a.scope,source_bundle=a.source_bundle),ensure_ascii=False),flush=True)
