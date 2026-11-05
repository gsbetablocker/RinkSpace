#!/usr/bin/env python3
"""One-shot read-only saved-blade motion report; no solve, weights, source decode, or endpoint relabeling."""
import contextlib
import csv
import datetime
import hashlib
import io
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

PROJECT = Path(r'D:\OneDrive\codex_projs\RinkSpace')
BASE = Path(r'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029')
AUDIT = BASE/'blade_observability_1105'
OUT = BASE/'blade_orientation_1105'
LEGACY = Path(r'D:\CodexData\RinkSpace_Data\handoff\1105\rinkspace_blade_audit\audit_blade_motion.py')
WINDOWS = [(77.6,78.2),(81.2,82.0),(84.8,85.1),(87.6,90.0),(92.4,93.6)]
TB = (1,12800)
PTS = np.arange(972800,1216000+1,256,dtype=np.int64)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def record(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def stats(a):
    return dict(count=int(len(a)),minimum=float(a.min()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    identity=read(AUDIT/'identity_audit.json')
    audit=read(AUDIT/'blade_observability_audit.json')
    manifest=read(AUDIT/'dual_source_manifest.json')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    assert head==identity['head']=='0078c8db0e50e36993b67cb29aed61b2fca3e7fc'
    raw_sources={}
    composite_sources={}
    source_metadata=[]
    def add_source(pt,path,expected,metadata):
        if not path:return
        q=dict(path=str(path),expected_sha256=expected,metadata=str(metadata))
        bucket=raw_sources.setdefault(int(pt),[])
        if q not in bucket:bucket.append(q)
    for f in manifest['frames']:
        add_source(f['pts'],f.get('source_frame'),f.get('source_frame_sha256'),AUDIT/'dual_source_manifest.json')
        for r in f.get('frame_references',[]):
            add_source(f['pts'],r['path'],r.get('actual_sha256') or r.get('sha256'),r.get('document','phase1_manifest'))
    for f in audit['reviewed_source_exposures']:
        add_source(f['pts'],f.get('source_frame'),f.get('source_frame_sha256'),AUDIT/'blade_observability_audit.json')
    extra=[('spin_grasp_1104/source_review/early_native_frames.json','rows','frame','sha256'),
           ('spin_grasp_1104/source_review/native_frames.json','rows','frame','sha256'),
           ('spin_continuous_1105/source_review/extracted_native_frames.json',None,'path','sha256'),
           ('spin_continuous_1105/source_review/dense_frame_manifest.json','frames','source_frame','source_frame_sha256'),
           ('spin_continuous_1105/source_review/flight_body_extract_metadata.json',None,'source_frame','source_frame_sha256'),
           ('spin_local_280_1105/source_review/manifest.json','records','source_frame','source_frame_sha256'),
           ('support_spin_1104/source_review/selection.json','rows','source_frame','source_frame_sha256')]
    for rel,key,pk,hk in extra:
        path=BASE/rel;doc=read(path);source_metadata.append(record(path))
        for f in doc[key] if key else doc:add_source(f['pts'],f[pk],f.get(hk),path)
    for f in audit['source_contacts'].get('supplemental_sources',[]):
        if f['provenance']=='verified native source PNG':add_source(f['pts'],f['image_path'],None,AUDIT/'blade_observability_audit.json')
        else:
            c=record(f['image_path']);c.update(panel='original_source only',original_frame_path='unknown',provenance=f['provenance'])
            composite_sources.setdefault(f['pts'],[]).append(c)
    sources={};hash_cache={};hash_mismatches=[]
    for pt,refs in sorted(raw_sources.items()):
        merged={}
        for r in refs:
            path=Path(r['path'])
            if not path.is_file():continue
            if str(path) not in hash_cache:hash_cache[str(path)]=sha(path)
            actual=hash_cache[str(path)]
            if r['expected_sha256'] and actual!=r['expected_sha256']:
                hash_mismatches.append(dict(pts=pt,**r,actual_sha256=actual));continue
            m=merged.setdefault(str(path),dict(path=str(path),sha256=actual,exact_pts=pt,provenance='existing original-source PNG; exact PTS bound by listed metadata',metadata=[],expected_hash_verified=False))
            if r['metadata'] not in m['metadata']:m['metadata'].append(r['metadata'])
            m['expected_hash_verified']|=bool(r['expected_sha256'])
        sources[pt]=list(merged.values())
    assert not hash_mismatches,hash_mismatches
    review={int(f['pts']):f for f in audit['reviewed_source_exposures']}
    def source_info(pt,foot):
        frame=review.get(int(pt));visual=None
        if frame:
            fs=next((f for f in frame['feet'] if f['foot']==foot),None)
            if fs:
                v=fs.get('visual_review',{})
                visual={k:v.get(k,'unknown') for k in ['anatomical_identity','identity_confidence','blade_front_visible','blade_rear_visible','blade_axis_observable','source_image_axis_visible','orientation_unknown_reason']}
                visual['fixed_template_endpoint_identity']='not established by this numerical report'
        return dict(pts=int(pt),source_seconds=float(int(pt)/12800),original_source_status='exact_pts_original_cached' if sources.get(int(pt)) else 'unknown',
                    original_source_frames=sources.get(int(pt),[]),unknown_reason=None if sources.get(int(pt)) else 'No exact-PTS original image in the audited manifests; no neighboring frame substituted',
                    known_composite_source_panes=composite_sources.get(int(pt),[]),stage1_visual_review=visual,
                    source_3d_axis_ground_truth='unknown',physical_endpoint_identity='not established by saved model points')
    all_rows=[];candidates={};global_events=[]
    legacy_main=runpy.run_path(str(LEGACY))['main']
    for parent in identity['parents']:
        name=parent['candidate'];input_path=Path(parent['path']);actual_hash=sha(input_path)
        assert actual_hash==parent['sha256']==parent['expected_sha256']
        with np.load(input_path,allow_pickle=False) as z:
            pts=np.asarray(z['pts'],np.int64);tb=tuple(map(int,z['time_base']))
            np.testing.assert_array_equal(pts,PTS);np.testing.assert_array_equal(z['source_pts'],PTS);assert tb==TB
            blade=np.asarray(z['blade_material_world_mid'],float)
            R=np.asarray(z['skate_R_world_mid'],float);origin=np.asarray(z['skate_origin_world_mid'],float)
            binding=json.loads(str(z['footwear_binding_json']));template_version=str(z['template_version'])
        assert blade.shape==(951,2,3,3) and np.isfinite(blade).all()
        delta=blade[:,:,2]-blade[:,:,0];length=np.linalg.norm(delta,axis=-1)
        assert np.all(length>1e-9)
        axis=delta/length[...,None]
        dot=np.clip(np.einsum('nfi,nfi->nf',axis[:-1],axis[1:]),-1.,1.)
        cross=np.linalg.norm(np.cross(axis[:-1],axis[1:]),axis=-1)
        # Independent stable calculation; atan2(cross,dot) agrees with legacy acos(dot).
        directed=np.arctan2(cross,dot)
        undirected=np.arctan2(cross,np.abs(dot))
        dt=np.diff(pts).astype(float)*tb[0]/tb[1]
        seconds=pts.astype(float)*tb[0]/tb[1]
        rate=directed/dt[:,None];axis_rate=undirected/dt[:,None]
        np.testing.assert_allclose(directed,np.arccos(dot),atol=1e-11,rtol=0)
        np.testing.assert_allclose(undirected,np.minimum(directed,np.pi-directed),atol=1e-14,rtol=0)
        assert np.all(dt==.02)
        local_axis=np.einsum('nfji,nfj->nfi',R,axis)
        local_points=np.einsum('nfji,nfkj->nfki',R,blade-origin[:,:,None])
        reconstructed=np.einsum('nfij,fj->nfi',R,local_axis[0])
        deviation=np.linalg.norm(reconstructed-axis,axis=-1)
        local_point_deviation=np.linalg.norm(local_points-local_points[0],axis=-1)
        ortho=np.linalg.norm(np.swapaxes(R,-1,-2)@R-np.eye(3),axis=(-2,-1))
        det=np.linalg.det(R)
        assert deviation.max()<1e-10 and local_point_deviation.max()<1e-10
        assert np.all(local_points[:,:,2,0]>local_points[:,:,0,0])
        # Execute the existing packet helper unchanged; temporary JSON is discarded after comparison.
        with tempfile.TemporaryDirectory(prefix='blade_legacy_check_',dir=OUT) as temporary:
            old_argv=sys.argv[:]
            try:
                legacy_output=Path(temporary)/'audit.json'
                sys.argv=[str(LEGACY),'--input',str(input_path),'--output',str(legacy_output)]
                with contextlib.redirect_stdout(io.StringIO()):legacy_main()
                legacy=read(legacy_output)
            finally:sys.argv=old_argv
        legacy_diff=[]
        for f in (0,1):
            d=abs(float(rate[:,f].max())-legacy['summary'][f]['maximum_rad_s']);legacy_diff.append(d)
            assert d<1e-8
        def event(i,foot):
            return dict(candidate=name,interval_index=int(i),foot=int(foot),model_foot_identity='existing saved foot0/foot1; no anatomical reassignment',
                        start_pts=int(pts[i]),end_pts=int(pts[i+1]),start_seconds=float(seconds[i]),end_seconds=float(seconds[i+1]),actual_dt_seconds=float(dt[i]),
                        directed_angle_rad=float(directed[i,foot]),directed_angle_deg=float(np.degrees(directed[i,foot])),directed_rate_rad_s=float(rate[i,foot]),directed_rate_deg_s=float(np.degrees(rate[i,foot])),
                        undirected_axis_angle_rad=float(undirected[i,foot]),undirected_axis_angle_deg=float(np.degrees(undirected[i,foot])),undirected_axis_rate_rad_s=float(axis_rate[i,foot]),undirected_axis_rate_deg_s=float(np.degrees(axis_rate[i,foot])),
                        dot=float(dot[i,foot]),obtuse_directed_change=bool(dot[i,foot]<0),axis_world_start=axis[i,foot].tolist(),axis_world_end=axis[i+1,foot].tolist(),
                        start_source=source_info(pts[i],foot),end_source=source_info(pts[i+1],foot),
                        interpretation='model-derived adjacent 3D chord-axis motion; physical direction error and cause remain unverified')
        def summarize(indices):
            return [dict(foot=f,intervals=len(indices),directed_angle_deg=stats(np.degrees(directed[indices,f])),directed_rate_rad_s=stats(rate[indices,f]),
                         undirected_axis_angle_deg=stats(np.degrees(undirected[indices,f])),undirected_axis_rate_rad_s=stats(axis_rate[indices,f]),
                         maximum_directed_interval=event(indices[np.argmax(rate[indices,f])],f),maximum_undirected_interval=event(indices[np.argmax(axis_rate[indices,f])],f)) for f in (0,1)]
        def top_intervals(values):
            peak=np.max(values,axis=1);indices=np.argsort(-peak,kind='stable')[:20]
            return [dict(rank=rank+1,**event(int(i),int(np.argmax(values[i]))),other_foot_directed_rate_rad_s=float(rate[i,1-int(np.argmax(values[i]))]),other_foot_undirected_axis_rate_rad_s=float(axis_rate[i,1-int(np.argmax(values[i]))])) for rank,i in enumerate(indices)]
        window_reports=[]
        for lo,hi in WINDOWS:
            a=int(round(lo*12800));b=int(round(hi*12800))
            indices=np.flatnonzero((pts[:-1]>=a)&(pts[1:]<=b))
            context=np.flatnonzero((pts[:-1]>=a-5120)&(pts[1:]<=b+5120))
            coverage=next(r for r in audit['window_coverage'] if r['candidate']==name and r['window_seconds']==[lo,hi])
            window_reports.append(dict(window_seconds=[lo,hi],selection='both interval endpoints inside the closed window',interval_count=int(len(indices)),
                summary=summarize(indices),context_seconds=[lo-.4,hi+.4],context_interval_count=int(len(context)),context_summary=summarize(context),
                stage1_coverage=coverage,judgment_zh='仅为保存模型刀轴的连续性描述；需要同PTS源图投影比较后才能归因或评价方向准确性。'))
        candidates[name]=dict(input=record(input_path),expected_sha256=parent['sha256'],sha_matches=True,frames=951,intervals_per_foot=950,summary=summarize(np.arange(950)),
            directed_obtuse_interval_count=int(np.sum(dot<0)),
            top20_directed_intervals=top_intervals(rate),top20_undirected_axis_intervals=top_intervals(axis_rate),
            top20_by_foot={str(f):[dict(rank=k+1,**event(int(i),f)) for k,i in enumerate(np.argsort(-rate[:,f],kind='stable')[:20])] for f in (0,1)},
            windows=window_reports,saved_rotation_consistency=dict(method='R_world_from_B.T applied to front-minus-rear and all 3 fixed material points; constant local coordinates checked without endpoint swaps',
                normalized_local_front_minus_rear_axis_first_exposure=local_axis[0].tolist(),max_world_axis_reconstruction_vector_error=float(deviation.max()),
                max_local_material_point_deviation_exported_units=float(local_point_deviation.max()),max_rotation_orthogonality_frobenius_error=float(ortho.max()),determinant_range=[float(det.min()),float(det.max())],
                first_exposure_local_rear_middle_front_exported_units=local_points[0].tolist(),front_x_greater_than_rear_x_all_samples=True,
                template_version=template_version,template_axes=binding['template_axes'],physical_template_identity_validation='not provided; saved model self-consistency only'),
            existing_helper_comparison=dict(script=record(LEGACY),unchanged_helper_executed=True,helper_definition=legacy['definition'],maxima_absolute_difference_rad_s=legacy_diff,independent_formula='atan2(norm(cross), dot); PTS integer difference / time_base',acceptance_threshold_used=False))
        for f in (0,1):
            for i in range(950):
                s0=sources.get(int(pts[i]),[]);s1=sources.get(int(pts[i+1]),[])
                row=dict(candidate=name,foot=f,interval_index=i,start_pts=int(pts[i]),end_pts=int(pts[i+1]),start_seconds=float(seconds[i]),end_seconds=float(seconds[i+1]),actual_dt_seconds=float(dt[i]),
                    directed_angle_rad=float(directed[i,f]),directed_angle_deg=float(np.degrees(directed[i,f])),directed_rate_rad_s=float(rate[i,f]),directed_rate_deg_s=float(np.degrees(rate[i,f])),
                    undirected_axis_angle_rad=float(undirected[i,f]),undirected_axis_angle_deg=float(np.degrees(undirected[i,f])),undirected_axis_rate_rad_s=float(axis_rate[i,f]),undirected_axis_rate_deg_s=float(np.degrees(axis_rate[i,f])),
                    dot=float(dot[i,f]),obtuse_directed_change=bool(dot[i,f]<0),
                    start_original_source_paths=' | '.join(q['path'] for q in s0) or 'unknown',end_original_source_paths=' | '.join(q['path'] for q in s1) or 'unknown',
                    start_original_source_sha256=' | '.join(q['sha256'] for q in s0) or 'unknown',end_original_source_sha256=' | '.join(q['sha256'] for q in s1) or 'unknown',
                    start_composite_source_pane_paths=' | '.join(q['path'] for q in composite_sources.get(int(pts[i]),[])) or 'unknown',end_composite_source_pane_paths=' | '.join(q['path'] for q in composite_sources.get(int(pts[i+1]),[])) or 'unknown',
                    source_ground_truth_3d_direction='unknown',metric_origin='saved_model_derived',template_endpoint_swap=False,anatomical_identity_reassigned=False,
                    task_windows=' | '.join(f'{lo:.1f}-{hi:.1f}' for lo,hi in WINDOWS if pts[i]>=round(lo*12800) and pts[i+1]<=round(hi*12800)))
                for k,v in zip(['start_axis_x','start_axis_y','start_axis_z','end_axis_x','end_axis_y','end_axis_z'],np.r_[axis[i,f],axis[i+1,f]]):row[k]=float(v)
                all_rows.append(row)
        global_events.extend(event(int(i),f) for f in (0,1) for i in np.argsort(-rate[:,f],kind='stable')[:20])
    csvpath=OUT/'blade_orientation_intervals.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    global_events.sort(key=lambda r:-r['directed_rate_rad_s'])
    source_index={str(int(pt)):dict(original_source_status='exact_pts_original_cached' if sources.get(int(pt)) else 'unknown',original_source_frames=sources.get(int(pt),[]),known_composite_source_panes=composite_sources.get(int(pt),[])) for pt in PTS}
    report=dict(schema='rinkspace.blade-orientation-anomaly-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=head,task_stage=2,
        status='completed_readonly_model_motion_report; stopped before projection attribution or fitting',scope=identity['scope'],
        definitions=dict(axis='u=(saved front[2]-saved rear[0])/norm(front-rear); fixed rear/middle/front order retained',directed_angle='atan2(norm(cross(u_i,u_i+1)), dot(u_i,u_i+1)); equivalent acos(dot), 0..180 degrees',
            undirected_axis_angle='atan2(norm(cross),abs(dot)); equivalent acos(abs(dot)), 0..90 degrees; 180-degree line symmetry only, no endpoint swap',
            rate='adjacent axis angle / actual dt; dt=integer PTS difference*time_base[0]/time_base[1]',units='3D world axis angles; angle rate in rad/s and degree/s',
            exposure='native PTS951, not duplicated display frames',note='This measures movement of a line/chord direction, not full skate SO(3) angular velocity about that axis'),
        evidence_boundary=dict(model_derived=['saved blade rear/middle/front world points','saved skate_R_world_mid','all angles, rates and rankings'],source=['exact-PTS original image path/SHA and source-clock metadata','unchanged phase1 visual review, where available'],
            unknown=['physical 3D blade direction ground truth','source-vs-model orientation error','fixed template endpoint material identity in source','cause attribution between ankle pose, static registration, template or identity'],
            physical_endpoint_identity='Directed axis preserves saved template labels; this does not certify the same physical blade endpoints in source pixels.',
            acceptance='No angle-rate threshold, speed threshold, accuracy pass or candidate selection is applied. A smoother candidate is not certified more accurate.'),
        stage1_inputs=[record(AUDIT/'identity_audit.json'),record(AUDIT/'blade_observability_audit.json'),record(AUDIT/'dual_source_manifest.json')],additional_source_metadata=source_metadata,
        source_index=source_index,source_cache_summary=dict(exact_original_pts=sum(bool(sources.get(int(pt))) for pt in PTS),unknown_original_pts=sum(not bool(sources.get(int(pt))) for pt in PTS),
            composite_only_pts=[int(pt) for pt in PTS if not sources.get(int(pt)) and composite_sources.get(int(pt))],source_image_hash_mismatches=hash_mismatches,no_neighbor_substitution=True,no_new_source_decode=True),
        candidates=candidates,top20_directed_across_all_candidates_and_feet=[dict(rank=k+1,**e) for k,e in enumerate(global_events[:20])],
        outputs=dict(interval_csv=record(csvpath),rows=len(all_rows),reproduction_script=record(Path(__file__))),
        conclusions_zh=['本报告只检查保存模型刀轴的相邻曝光变化，不是对真实源图刀轴方向的测量或验收。',
            'rear/middle/front及foot0/foot1保持父结果身份；acos(abs(dot))只处理无向线180度对称，绝不通过交换刀端降低变化。',
            '所有950个相邻区间的实际dt均由PTS差计算为20ms，四候选父NPZ的SHA及951 PTS已独立复核。',
            '五个指定窗口及每侧0.4秒上下文均报告峰值；没有新拟合、源观测、阈值、模型权重读取或GUI。',
            '精确PTS原图缓存缺失写unknown；未用临近源图或插值替代，未从模型变化反推源图观测。',
            'saved skate_R与固定刀端局部坐标一致只证明导出自洽；方向错误来源仍需主会话的同PTS投影反事实比较。'],
        execution_boundaries=dict(no_solve=True,no_weights_loaded=True,no_source_observation_changes=True,no_endpoint_or_foot_swaps=True,no_core_worktree_changes=True,no_video=True,no_GUI=True,no_commit_or_push=True))
    outpath=OUT/'blade_orientation_anomaly_report.json'
    outpath.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(outpath),csv=str(csvpath),rows=len(all_rows),source_cache_summary=report['source_cache_summary'],
        candidates={k:dict(global_max=[s['directed_rate_rad_s']['maximum'] for s in v['summary']],window_maxima=[[s['directed_rate_rad_s']['maximum'] for s in w['summary']] for w in v['windows']],obtuse=v['directed_obtuse_interval_count']) for k,v in candidates.items()}),ensure_ascii=False))

if __name__=='__main__':main()
