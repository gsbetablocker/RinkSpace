"""Actual adjacent-native-exposure errors for genuine fixed material points.

Uses complete-CP independent true-RS exports. Curves with sliding stations are
not static track points. Pending numerical bindings keep their denominators.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json
import numpy as np
from .io_utils import file_sha256
from .r_inner_prefix_v1 import array
from .first4_source_compare_v1 import metric

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p): return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))

def compare(packet_path,before_RS,after_RS,complete_source_comparison,output):
    pp=Path(packet_path).resolve();p=read(pp);proof=read(complete_source_comparison);out=Path(output).resolve()
    if out.exists(): raise FileExistsError('Immutable source-motion comparison exists')
    digest=file_sha256(pp);n=p['row_count'];seen=set()
    if proof['packet']['sha256']!=digest: raise ValueError('Different complete-CP source proof')
    A={k:array(p,k,pp.parent,seen)for k in ('kind','admission','fit_role','shot_index','material_index','native_pts','uv','path')}
    good_source=(A['admission']==1)&(A['kind']==1);active=good_source&(A['fit_role']==1)
    state={};refs=[ref(pp),ref(complete_source_comparison)];models={}
    for key,folder in [('before',before_RS),('after',after_RS)]:
        folder=Path(folder).resolve();meta=read(folder/'value_report.json');mp=folder/'model.json'
        if meta['model_sha256']!=file_sha256(mp)or meta['model_sha256']!=proof['states'][key]['model']['sha256']or not meta['exact_checkpoint_model_blocks']:raise ValueError('Not the actual verified complete-CP model')
        if meta['mathematical_source_id']!=digest:raise ValueError('RS export belongs to another source')
        files={}
        for name,dtype,shape in [('raw_station_RS_error_native_uv.f64','<f8',(n,2)),('raw_station_RS_valid.u8','u1',(n,))]:
            f=folder/name
            if file_sha256(f)!=meta['files'][name]['sha256'] or f.stat().st_size!=np.prod(shape)*np.dtype(dtype).itemsize:raise ValueError('Changed RS source export')
            files[name]=np.memmap(f,mode='r',dtype=dtype,shape=shape);refs.append(ref(f))
        cmroot=Path(proof['states'][key]['model']['file']).parent/'CP_explicit_metrics';cm=read(cmroot/'manifest.json');vf=cmroot/'explicit_valid.u8'
        if cm['packet_sha256']!=digest or cm['state_token']!=proof['states'][key]['complete_state_token']or vf.stat().st_size!=n or file_sha256(vf)!=cm['files']['explicit_valid.u8']['sha256']:raise ValueError('Complete CP physical domain flags differ')
        domain=np.memmap(vf,mode='r',dtype='u1',shape=(n,))==1;refs.extend([ref(vf),ref(cmroot/'manifest.json')])
        state[key]=dict(error=files['raw_station_RS_error_native_uv.f64'],valid=(files['raw_station_RS_valid.u8']==1)&domain&active)
        models[key]=read(mp);refs.extend([ref(mp),ref(folder/'value_report.json')])
    return _summarize_motion(pp,p,models,state,A,refs,out,ref(complete_source_comparison),
        "True RS, finite material domain, and complete native CP explicit physical domain flags")

def _summarize_motion(pp,p,models,state,A,refs,out,source_proof,validity_scope):
    good_source=(A["admission"]==1)&(A["kind"]==1)
    active=good_source&(A["fit_role"]==1)
    tracks=defaultdict(list);bad_topology=[]
    for i in np.flatnonzero(good_source):
        mi=int(A['material_index'][i]);pi=int(A['path'][i]);material=models['after']['materials'][mi];path=material['paths'][pi]
        if not path.get('physical_point')or len(path.get('points',[]))!=1:
            bad_topology.append(int(i));continue
        tracks[(int(A['shot_index'][i]),mi,pi)].append(int(i))
    pair_rows=[];ambiguous=[]
    for key,rows in tracks.items():
        bypts=defaultdict(list)
        for i in rows:bypts[int(A['native_pts'][i])].append(i)
        unique={}
        for pts,rr in bypts.items():
            if len(rr)==1:unique[pts]=rr[0]
            elif all(np.array_equal(A['uv'][rr[0]],A['uv'][i])and active[rr[0]]==active[i]for i in rr[1:]):unique[pts]=rr[0]
            else:ambiguous.append(dict(shot=p['shot_ids'][key[0]],material_index=key[1],native_pts=pts,source_rows=rr))
        for pt,i in sorted(unique.items()):
            if pt+256 in unique:pair_rows.append((i,unique[pt+256]))
    pair=np.asarray(pair_rows,dtype=np.int64).reshape(-1,2);pv={};pe={}
    for key,s in state.items():
        pv[key]=np.all(s['valid'][pair],axis=1)if len(pair)else np.zeros(0,bool)
        pe[key]=np.linalg.norm(s['error'][pair[:,1]]-s['error'][pair[:,0]],axis=1)if len(pair)else np.zeros(0)
    def block(rows,pairs):
        return {key:dict(fixed_point_UV=metric(np.linalg.norm(s['error'][rows],axis=1),s['valid'][rows]),adjacent_native_motion_UV=metric(pe[key][pairs],pv[key][pairs]))for key,s in state.items()}
    byshot={}
    for j,sid in enumerate(p['shot_ids']):
        rows=np.flatnonzero(good_source&(A['shot_index']==j));sel=(A['shot_index'][pair[:,0]]==j)if len(pair)else np.zeros(0,bool)
        scope=models['after']['shots'][sid]['scope'];first,last=scope['first_pts'],scope['last_pts'];pts=np.arange(first,last,256,dtype=np.int64)
        pairpts=A['native_pts'][pair[:,0]][sel];sp=set(map(int,pairpts));row=block(rows,sel)
        temporal={}
        for key in state:
            validpts=set(map(int,pairpts[pv[key][sel]]));temporal[key]=dict(valid_adjacent_pairs=len(validpts),complete_native_pair_denominator=len(pts),pair_coverage_fraction=len(validpts)/len(pts)if len(pts)else None,unobserved_or_unresolved_pairs=[int(t)for t in pts if int(t)not in validpts])
        row.update(source_qualified_fixed_point_rows=len(rows),numeric_pending_rows=int((good_source&(A['shot_index']==j)&~active).sum()),genuine_source_adjacent_pairs=len(sp),complete_native_pair_denominator=len(pts),temporal_coverage=temporal);byshot[sid]=row
    out.mkdir(parents=True);np.savez_compressed(out/'arrays.npz',pair_rows=pair,before_motion_error_px=pe['before'],after_motion_error_px=pe['after'],before_valid=pv['before'],after_valid=pv['after'])
    report=dict(schema='rinkfit.actual_fixed_material_source_motion.v1',packet=ref(pp),complete_CP_source_proof=source_proof,validity_scope=validity_scope,by_shot=byshot,all=block(np.flatnonzero(good_source),np.ones(len(pair),bool)),static_point_tracks=len(tracks),genuine_source_pair_rows=len(pair),ambiguous_same_exposure_points=ambiguous,nonpoint_topology_rows=bad_topology,qualified_point_rows=int(good_source.sum()),numeric_pending_rows=int((good_source&~active).sum()),rows_rejected_by_source_qualification=int(((A['kind']==1)&~good_source).sum()),source_UV_ID_nativePTS_changed=False,curve_sliding_station_used_as_static_point=False,pairs_cross_cut=False,interpolated_motion_observations=False,measurement_and_pair_failures_kept=True,source_correlation_independence_claimed=False,physical_unknown_point_coordinates_are_not_world_truth=True,all_frames_coverage_claimed=False,quality_accepted=False,semantics='Motion residual is (predicted UV[t+256]-predicted UV[t])-(actual source UV[t+256]-actual source UV[t]) for the same literal physical material point in one shot. No interpolation across gaps; model coordinates may be estimated unknowns. Source pair members and all unresolved denominators are retained.',arrays=ref(out/'arrays.npz'),dependencies=refs,producer=ref(__file__))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');return report


def compare_capture_audits(packet_path,before_audit,after_audit,output):
    """Compare genuine point motion using independently verified true-RS capture CPs.

    This accepts the new audit contract directly. It does not synthesize legacy
    explicit-constraint exports or claim that RS validity proves visibility.
    """
    pp=Path(packet_path).resolve();p=read(pp);out=Path(output).resolve()
    if out.exists(): raise FileExistsError('Immutable source-motion comparison exists')
    digest=file_sha256(pp);n=p['row_count'];seen=set()
    A={k:array(p,k,pp.parent,seen) for k in ('kind','admission','fit_role','shot_index','material_index','native_pts','uv','path')}
    active=(A['admission']==1)&(A['fit_role']==1)&(A['kind']==1)
    state={};models={};refs=[ref(pp)];proof={}
    for key,ap in [('before',before_audit),('after',after_audit)]:
        ap=Path(ap).resolve();r=read(ap);proof[key]=ref(ap);refs.append(ref(ap))
        if r.get('schema') not in ('rinkfit.first4_true_RS_capture_independent_audit.v1','rinkfit.first4_true_RS_capture_independent_audit.v3','rinkfit.first4_true_RS_capture_independent_audit.v4','rinkfit.first4_true_RS_capture_independent_audit.v5'):
            raise ValueError('Unsupported independent capture audit schema')
        if r['schema'].endswith(('.v3','.v4','.v5')):
            d=r['independent_certified_capture_domain']
            if r['complete_CP']['objective_id']!='rinkfit.sourceUV_trueRS_finite_capture.'+('v4' if r['schema'].endswith('.v5') else r['schema'].rsplit('.',1)[1]) or not d['scene_values_unchanged'] or d['source_optical_calibration_claimed'] or r['unchanged_numerical_consumer']['sha256']!='0b373951e7a1f3e75a0d979a51d5f1d2201faa194e2981f0c5a92531c7d6efca':
                raise ValueError('Unsupported strict v3 numerical audit contract')
            if r['schema'].endswith('.v5'):
                ap5=r['independent_finite_cell_state'].get('source_append_policy')
                if ap5 is not None and (not ap5['all_18_canonical_source_prefix_arrays_bitwise_verified'] or not ap5['all_4_selected_complete_auxiliary_prefixes_bitwise_verified'] or not ap5['selected_complete_FNV_recomputed'] or not ap5['all_old_cell_domain_and_graph_records_verified'] or not ap5['source_identity_Query_W_and_finite_domains_preserved'] or ap5['old_source_UV_moved'] or ap5['formal_Query_or_W_consumed']):
                    raise ValueError('Unverified actual capture source append')
            for f in (d['certificate'],d['actual_Ceres_domain_receipt'],r['producer'],r['unchanged_numerical_consumer']):
                if file_sha256(f['file'])!=f['sha256']:raise ValueError('Changed strict v3 audit dependency')
            if r['schema'].endswith(('.v4','.v5')):
                d4=r['independent_finite_cell_state'];v3ref=r['unchanged_domain_consumer']
                if v3ref['sha256']!='ddce5bcd33f5eda6ac5657494608a6695c250e0db34b5afdbee50f91d27313b7' or file_sha256(v3ref['file'])!=v3ref['sha256'] or not d4['all_saved_stations_bitwise_verified'] or not d4['all_original_finite_domains_and_cells_checked'] or not d4['all_actual_Ceres_constant_and_bounds_records_checked'] or d4['source_domain_expanded']:
                    raise ValueError('Unverified strict v4 cell state')
                for dep in d4['dependencies']:
                    if file_sha256(dep['file'])!=dep['sha256']:raise ValueError('Changed v4 active cell dependency')
        if r['packet']['sha256']!=digest or file_sha256(r['packet']['file'])!=digest or r['raw_rows']!=n:
            raise ValueError('Capture audit belongs to another source packet')
        c=r['complete_CP']
        if not c['all_parameter_values_verified'] or not c['FNV_token_independently_recomputed'] or not c['state_token']:
            raise ValueError('Incomplete or unverified capture checkpoint')
        if r['native_bitmaps_used_for_independent_validity'] or not r['original_UV_and_failure_denominators_preserved']:
            raise ValueError('Expected independent all-source true RS audit')
        mr=r['model'];mp=Path(mr['file'])
        if file_sha256(mp)!=mr['sha256']: raise ValueError('Changed complete-CP model')
        models[key]=read(mp);refs.append(ref(mp));data={}
        for name,dtype,shape in [('raw_station_RS_error_native_uv.f64','<f8',(n,2)),('raw_station_RS_predicted_uv.f64','<f8',(n,2)),('raw_station_RS_valid.u8','u1',(n,)),('finite_station_valid.u8','u1',(n,))]:
            fr=r['files'][name];fp=Path(fr['file'])
            if tuple(fr['shape'])!=shape or np.dtype(fr['dtype'])!=np.dtype(dtype) or fp.stat().st_size!=np.prod(shape)*np.dtype(dtype).itemsize or file_sha256(fp)!=fr['sha256']:
                raise ValueError('Changed independent capture source export: '+name)
            data[name]=np.memmap(fp,mode='r',dtype=dtype,shape=shape);refs.append(ref(fp))
        valid=(data['raw_station_RS_valid.u8']==1)&(data['finite_station_valid.u8']==1)&active
        error=data['raw_station_RS_error_native_uv.f64'];pred=data['raw_station_RS_predicted_uv.f64']
        if not np.isfinite(error[valid]).all() or not np.allclose(error[valid],pred[valid]-A['uv'][valid],rtol=0.,atol=1e-10):
            raise ValueError('Capture errors no longer refer to the unchanged raw source UV')
        state[key]=dict(error=error,valid=valid)
    return _summarize_motion(pp,p,models,state,A,refs,out,proof,
        'Independent finite material/path domain and certified true RS only; complete native physical visibility is not independently reimplemented. Numerical pending source rows remain failed in denominators.')
