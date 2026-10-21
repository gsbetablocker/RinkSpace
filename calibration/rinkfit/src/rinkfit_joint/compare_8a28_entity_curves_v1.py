"""Read-only current native vs unpatched 8a28 original-Brown entity curves.

Each source UV is independently compared to each model's own finite projected
physical curve. No same-station/angle or old-to-current XYZ point correspondence
is asserted. Only frozen 41-entity crosswalk and matching material roles enter.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,gzip,json,time
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .source_generation_input_v2 import QualifiedSourceGenerationInput
from .baseline_surface_replay import BaselineSurfaceReplay
from .template_native_seed_v2 import _old_curve
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene
from rinkfit_joint.source_quality_compare_v1 import _camera


def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha256(p))


def bound(r):
    p=Path(r.get('path',r.get('file')))
    if file_sha256(p)!=r['sha256']:raise ValueError('changed fixed comparison evidence: '+str(p))
    return p


def nearest(points,curve,valid,*,closed=False):
    points=np.asarray(points,float);curve=np.asarray(curve,float);valid=np.asarray(valid,bool)
    legal=valid[:-1]&valid[1:]&np.isfinite(curve[:-1]).all(1)&np.isfinite(curve[1:]).all(1)
    ids=np.flatnonzero(legal);a=curve[ids];d=curve[ids+1]-a;den=np.sum(d*d,axis=1);keep=den>1e-20;ids,a,d,den=ids[keep],a[keep],d[keep],den[keep]
    result=dict(distance=np.full(len(points),np.inf),normal_distance=np.full(len(points),np.nan),uv=np.full_like(points,np.nan),curve_parameter=np.full(len(points),np.nan),finite_endpoint=np.zeros(len(points),bool))
    if not len(ids):return result
    # Small per-entity arrays, not a screen cache or programme video replay.
    delta=points[:,None,:]-a[None,:,:];u=np.clip(np.einsum('ijk,jk->ij',delta,d)/den,0.,1.);foot=a[None,:,:]+u[:,:,None]*d
    sq=np.sum((points[:,None,:]-foot)**2,axis=2);chosen=np.argmin(sq,axis=1);rr=np.arange(len(points));step=d[chosen];normal=np.c_[-step[:,1],step[:,0]]/np.sqrt(den[chosen])[:,None]
    parameter=(ids[chosen]+u[rr,chosen])/(len(curve)-1);error=points-foot[rr,chosen]
    result.update(distance=np.sqrt(sq[rr,chosen]),normal_distance=np.sum(error*normal,axis=1),uv=foot[rr,chosen],curve_parameter=parameter,finite_endpoint=(~closed)&((parameter<=1e-12)|(parameter>=1-1e-12)))
    return result


def converged_distance(points,projector,*,circle,numerical_tolerance_px=.001):
    count=257 if circle else 65;previous=None
    while True:
        parameter=np.linspace(0.,1.,count);curve,valid=projector(parameter);now=nearest(points,curve,valid,closed=circle)
        if previous is not None:
            same=np.isfinite(now['distance'])&np.isfinite(previous['distance'])
            failure_equal=np.array_equal(np.isfinite(now['distance']),np.isfinite(previous['distance']))
            difference=float(np.max(abs(now['distance'][same]-previous['distance'][same]),initial=0.))
            if failure_equal and difference<=numerical_tolerance_px:
                return now,dict(projection_curve_samples=count,successive_distance_change_max_px=difference,numerical_tolerance_px=numerical_tolerance_px,finite_projected_curve_segments=int(np.sum(np.asarray(valid)[:-1]&np.asarray(valid)[1:])))
        previous=now;count=2*count-1


def metric(v):
    a=np.asarray(v,float);good=np.isfinite(a);values=a[good]
    return dict(denominator=len(a),valid=int(good.sum()),failed=int((~good).sum()),full_rms_px=float(np.sqrt(np.mean(a*a))) if good.all() and len(a) else None,available_rms_px=float(np.sqrt(np.mean(values*values))) if len(values) else None,median_px=float(np.median(values)) if len(values) else None,p95_px=float(np.quantile(values,.95)) if len(values) else None,maximum_px=float(np.max(values)) if len(values) else None)


def run(binding,model_file,initial_state,crosswalk,audit_file,output):
    started=time.perf_counter();out=Path(output)
    if out.exists():raise ValueError('comparison output must be a new generation')
    Q=QualifiedSourceGenerationInput(binding);model=read_json(model_file);cross=read_json(crosswalk);audit=read_json(audit_file)
    role=read_json(bound(cross['role_contract']));old=BaselineSurfaceReplay(initial_state,use_patch=False)
    if bound(cross['old_scene']).resolve()!=old.scene_path.resolve():raise ValueError('41-entity crosswalk does not name this old scene')
    if len(cross['paired_paths'])!=41 or len(audit['curves'])!=41:raise ValueError('incomplete 41-entity crosswalk/audit')
    evidence={x['curve_id']:x for x in audit['curves']};paired={evidence[x['curve_id']]['current_material_label']:x for x in cross['paired_paths']}
    labels={m['label']:i for i,m in enumerate(model['materials'])};rows={r['source_row_id']:r for r in Q.rows()};source=defaultdict(dict);source_rejections=[];input_counts=Counter()
    expected_midline={'B1','B2','C','C0','A1','A2','A3','A4'}
    for support in Q.supports():
        members=[rows[i] for i in support['representative_row_ids'] if i in rows]
        if not members or members[0].get('category')!='ice_curve':continue
        target=members[0].get('target_id');input_counts['ice_support_instances']+=1;input_counts['ice_actual_raw_audit_instances']+=len(support['actual_source_audits']);input_counts['ice_failed_raw_audit_instances']+=sum(x.get('current_uv') is None for x in support['actual_source_audits'])
        if target not in paired or target not in labels:
            source_rejections.append(dict(statistical_input_id=support['statistical_input_id'],target_id=target,reason='no frozen same-entity finite primitive crosswalk/current material'))
            continue
        item=paired[target];fid=item['feature_id'];operators={r.get('measurement_operator_resolved') for r in members};expected='paint_center' if fid in expected_midline or item['current_path_kind']==0 else 'material_interface'
        if operators!={expected}:
            source_rejections.append(dict(statistical_input_id=support['statistical_input_id'],target_id=target,operators=sorted(str(x) for x in operators),expected_operator=expected,reason='current material operator differs from frozen physical curve role'))
            continue
        audits={a['raw_index']:a for a in support['actual_source_audits']}
        for index in support['current_support_indices']:
            a=audits[index]
            if a.get('current_uv') is None:raise ValueError('admitted source raw audit missing current material support')
            ev=a.get('original_profile_evidence',{});origin=ev.get('measurement_origin','')
            if expected=='paint_center':
                sides=np.asarray(ev.get('actual_source_sides_uv',[]),float)
                if sides.shape!=(2,2) or np.linalg.norm(sides[0]-sides[1])<=0 or 'outside_paint_to_ice' in origin:
                    source_rejections.append(dict(statistical_input_id=support['statistical_input_id'],raw_index=index,target_id=target,reason='missing distinct original two-material flanks for midline'))
                    continue
            key=(support['source_packet_sha256'],tuple(a['raw_uv']));bucket=(support['shot_id'],support['native_pts'],target)
            source[bucket][key]=dict(uv=a['raw_uv'],raw_index=index,statistical_input_id=support['statistical_input_id'],source_packet_sha256=support['source_packet_sha256'],physical_measurement_operator=expected,measurement_origin=origin)
    scenes={};cameras={}
    for sid in sorted({k[0] for k in source}):
        view=as_event_view(model,sid);scenes[sid]=NativeScene(view);cameras[sid]=_camera(model,sid)
    out.mkdir(parents=True);records=[];frames=[];iterations=[];L=float(audit['normalization']['L']);origin=np.asarray(audit['normalization']['origin'])
    with gzip.open(out/'raw_entity_distances.jsonl.gz','wt',encoding='utf8') as stream:
      for (sid,pts,target),measurements in sorted(source.items()):
        item=paired[target];mi=labels[target];pi=item['current_path_index'];path=scenes[sid].path(mi,pi)
        if path['kind']!=item['current_path_kind'] or path['id']!=item['current_path_id']:raise ValueError('current material finite primitive topology changed: '+target)
        oldcurve=_old_curve(old.scene,item,L,origin);circle=item['current_path_kind']==1;points=np.asarray([x['uv'] for x in measurements.values()],float)
        def old_project(s):
            if circle:xy=oldcurve['center']+oldcurve['radius']*np.c_[np.cos(2*np.pi*s),np.sin(2*np.pi*s)]
            else:xy=(1-s[:,None])*oldcurve['endpoints'][0]+s[:,None]*oldcurve['endpoints'][1]
            xyz=np.c_[xy,np.zeros(len(s))]*L+origin
            uv,depth,valid=old.project(int(pts),xyz,scene_sha256=old.scene_sha256,optics='original_brown',shot_id=sid)
            return uv,np.asarray(valid,bool)
        def current_project(s):
            xyz=scenes[sid].world(mi,pi,s);time=(pts-model['shots'][sid]['time_origin_pts'])/12800.
            projected=cameras[sid].project_rs(xyz,np.full(len(s),time))
            return projected['uv'],projected['valid']
        before,bproof=converged_distance(points,old_project,circle=circle);after,aproof=converged_distance(points,current_project,circle=circle)
        iterations.append(dict(shot_id=sid,native_pts=pts,target_id=target,old=bproof,current=aproof))
        for source_row,b,a in zip(measurements.values(),range(len(points)),range(len(points))):
            r=dict(shot_id=sid,native_pts=pts,target_id=target,**source_row,old_distance_px=float(before['distance'][b]) if np.isfinite(before['distance'][b]) else None,current_distance_px=float(after['distance'][a]) if np.isfinite(after['distance'][a]) else None,old_signed_normal_distance_px=float(before['normal_distance'][b]) if np.isfinite(before['normal_distance'][b]) else None,current_signed_normal_distance_px=float(after['normal_distance'][a]) if np.isfinite(after['normal_distance'][a]) else None,old_finite_endpoint=bool(before['finite_endpoint'][b]),current_finite_endpoint=bool(after['finite_endpoint'][a]),old_nearest_uv=before['uv'][b].tolist() if np.isfinite(before['uv'][b]).all() else None,current_nearest_uv=after['uv'][a].tolist() if np.isfinite(after['uv'][a]).all() else None)
            records.append(r);stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    def summarize(items):
        oldv=[np.inf if r['old_distance_px'] is None else r['old_distance_px'] for r in items];newv=[np.inf if r['current_distance_px'] is None else r['current_distance_px'] for r in items];oldm=metric(oldv);newm=metric(newv)
        return dict(old_original_full_Brown=oldm,current_physical_Brown_RS=newm,full_rms_delta_px=(newm['full_rms_px']-oldm['full_rms_px']) if newm['full_rms_px'] is not None and oldm['full_rms_px'] is not None else None,old_finite_endpoint_feet=sum(r['old_finite_endpoint'] for r in items),current_finite_endpoint_feet=sum(r['current_finite_endpoint'] for r in items))
    shots={sid:summarize([r for r in records if r['shot_id']==sid]) for sid in sorted({r['shot_id'] for r in records})};entities={target:summarize([r for r in records if r['target_id']==target]) for target in sorted({r['target_id'] for r in records})}
    selected=sorted({(r['shot_id'],r['native_pts']) for r in rows.values()})
    provenance=[old.provenance(int(pts)) for sid,pts in selected]
    report=dict(model=ref(model_file),input_binding=ref(binding),crosswalk=ref(crosswalk),audit=ref(audit_file),role_contract=ref(bound(cross['role_contract'])),old_baseline_references=old.references,old_use_patch=False,old_optics='original_brown',selected_source_frames=len(selected),frames_with_comparable_entities=len({(r['shot_id'],r['native_pts']) for r in records}),comparable_unique_raw_readouts=len(records),comparable_targets=len(entities),all=summarize(records),by_shot=shots,by_entity=entities,input_denominators=dict(input_counts),excluded_source_scope=source_rejections,entity_projection_numerical_convergence=iterations,exact_native_PTS_old_camera_provenance=provenance,source_UV_observations_from_model=False,same_station_angle_or_XYZ_point_correspondence_used=False,current_or_old_model_changed=False,no_screen_cache_or_video_generated=True,scope_limits=['Only actual admitted raw source readings on the frozen same-entity 41-primitive crosswalk and matching measurement roles.','Endpoint feet use finite-curve Euclidean distance; interior feet are nearest normal distance.','C0/A1-A4 exterior interfaces do not compare against a centreline; missing original two flanks remain excluded.','Current controls can have been fitted to these source data; this is a development comparison, not independent validation.','Old bridge source modes/provenance remain explicit; original Brown means numerical historical optics, not physical accuracy.','S01 full source is retained elsewhere and is not represented by these selected S02-S04 frames.','This finite-entity comparison does not replace full-shot same-model overlay review or establish no grid regression.'],arrays=ref(out/'raw_entity_distances.jsonl.gz'),producer=ref(__file__),seconds=time.perf_counter()-started)
    write_json(out/'comparison.json',report);print(json.dumps(dict(output=str(out),frames=report['frames_with_comparable_entities'],raw=report['comparable_unique_raw_readouts'],all=report['all'],by_shot=shots,seconds=report['seconds']),ensure_ascii=False),flush=True);return report


def main():
    a=argparse.ArgumentParser()
    for k in ['binding','model','initial-state','crosswalk','audit','out']:a.add_argument('--'+k,required=True)
    q=a.parse_args();run(q.binding,q.model,q.initial_state,q.crosswalk,q.audit,q.out)
if __name__=='__main__':main()
