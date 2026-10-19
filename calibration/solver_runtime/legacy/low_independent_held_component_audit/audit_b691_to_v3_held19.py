"""Read-only original19 HELD component replay, never fitting/solver/graph."""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['RINKSPACE_PROJECTION_BACKEND']='numpy';os.environ['RINKSPACE_MOTION_BACKEND']='numpy'
import sys,json,hashlib,pickle,itertools,math,time
from dataclasses import replace
import numpy as np
BASE=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from isolated_solver_bootstrap import install
install()
from z190_physical_camera import SharedCalibration,ViewState
from z190_source_camera_diagnostic import absolute_value_jac15,scanline_branch_value_jac15
from projection import project_central_valid
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import CorrectedSharedRinkNumericGeometryLanesAdPairs
from shared_ice_hockey_geometry_candidate.shared_ice_hockey_numeric_geometry import SharedIceHockeyNumericGeometry
from z190_joint_source_solver import _frozen_profile_token
from shared_nominal_geometry_reference_candidate.report_actual_soft_reference_values import CheckpointProfiles

UNIT=.31353729994266055;PTS=230011136
FOLDERS=[BASE/'current_reviewed_motion_ice_source_freezes/b69132266659',BASE/'current_common_hockey_soft_reference_source_freezes/9a8c1ac9d049']
PACKET=BASE/'shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalarstat(values):
    v=np.asarray(values,float)
    return dict(RMS=float(np.sqrt(np.mean(v*v))),point_norm_RMS=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),max_abs=float(np.max(abs(v))),max_point_norm=float(np.max(np.linalg.norm(v,axis=1))))
def read_version(folder):
    cp=folder/('solver_snapshot_UNACCEPTED.npz'if 'b691'in folder.name else'complete_frozen_solver_checkpoint_UNACCEPTED.npz')
    with np.load(cp,allow_pickle=False)as z:
        q=z['q'].copy();aux={k:z[k].copy()for k in z.files if k.startswith('aux__')}
        event=dict(q_dtype=q.dtype.str,q_shape=list(q.shape),q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),frozen_profile_sha256=_frozen_profile_token(CheckpointProfiles(z)))
    review=json.loads((folder/'latest_actual_source_review.json').read_text(encoding='utf-8'))
    assert all(review['source_event_state_contract'].get(k)==v for k,v in event.items())
    lens_contract=json.loads(aux['aux__all_installation_optics_contract_json'].item())
    iid=lens_contract['installation_order'][-1]
    basecal=SharedCalibration.from_manifest(lens_contract['installations'][iid]['base_calibration'])
    count=aux['aux__block_lengths']//4;stamps=aux['aux__canonical_pts'][sum(count[:3]):sum(count[:4])]
    idx=np.flatnonzero(stamps==PTS).item();start=int(aux['aux__block_start'][3])+4*idx
    xyzslots=aux['aux__installation_xyz_slots'][-1]
    if 'aux__current_shared_mount_parameter_indices'in aux:
        mount=q[aux['aux__current_shared_mount_parameter_indices'][-3:]].copy()
        lens_indices=aux['aux__current_all_installation_optics_parameter_indices'][-3:]
    else:
        mount=np.zeros(3);lens_indices=aux['aux__all_installation_optics_parameter_indices'][-3:]
    optics=q[lens_indices]*[1.,1.,.001]
    physical=dict(P=100.*(q[xyzslots]/(100.*UNIT)),L=np.asarray(basecal.lever_xyz)+mount/UNIT,
        optics=optics,Rf=q[start:start+4].copy())
    return dict(folder=folder,checkpoint=cp,q=q,aux=aux,basecal=basecal,event=event,review=review,
        physical=physical,view_slots=np.arange(start,start+4),low_canonical_index=idx,
        mount_indices=aux.get('aux__current_shared_mount_parameter_indices',np.array([],int))[-3:],lens_indices=lens_indices)

def main():
    begun=time.perf_counter();old,new=[read_version(x)for x in FOLDERS]
    assert old['aux']['aux__shared_scene_json'].item()==new['aux']['aux__shared_scene_json'].item()
    scene_base=json.loads(old['aux']['aux__shared_scene_json'].item())
    geometry=SharedIceHockeyNumericGeometry(CorrectedSharedRinkNumericGeometryLanesAdPairs(scene_base,UNIT))
    assert old['basecal'].manifest()==new['basecal'].manifest()
    go=geometry.initial.copy();go[:11]=old['q'][old['aux']['aux__geometry_parameter_indices']]
    gn=new['q'][new['aux']['aux__current_shared_world_geometry_parameter_indices']].copy()
    full_scene_checks=[]
    for version,g in [(old,go),(new,gn)]:
        rebuilt=geometry.source_scene(g)
        frozen=json.loads((version['folder']/'latest_candidate_scene_UNACCEPTED.json').read_text(encoding='utf-8'))
        actual_features={f['id']:f for f in rebuilt['ice_features']};frozen_features={f['id']:f for f in frozen['ice_features']}
        exact={}
        for fid,keys in [('F1',('center_xyz','radius')),('lane1',('polygon_xyz',))]:
            for key in keys:exact[fid+'.'+key]=bool(np.array_equal(np.asarray(actual_features[fid][key]),np.asarray(frozen_features[fid][key])))
        assert all(exact.values());full_scene_checks.append(exact)
    packet=pickle.load(PACKET.open('rb'))
    entries=[x for x in packet['programme_absolute_rows']if x['block']=='low'and x['source_row'].get('source_role')=='independent_user_frame_report_only']
    rows=[x['source_row']for x in entries];assert len(rows)==19
    assert all(x['collection_role'].lower()=='held'and x['group_actual_pts']==PTS and r['split']=='held'and r['fit_admitted']is False for x,r in zip(entries,rows))
    assert sha(PACKET)=='ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78'
    def cam(P,L,optics,Rf):
        base=old['basecal'];dx,dy,dk=optics
        cal=replace(base,pivot_xyz=tuple(P),lever_xyz=tuple(L),
            principal_x_fraction=(base.principal_x_fraction[0]+dx/1920.,),
            principal_y_fraction=(base.principal_y_fraction[0]+dy/1080.,),k1=(base.k1[0]+dk,))
        return cal.camera25(ViewState(tuple(Rf[:3]),float(Rf[3]-np.log(1920.))))
    def camera(masks):
        p=[old['physical'],new['physical']]
        return cam(p[masks[0]]['P'],p[masks[1]]['L'],p[masks[2]]['optics'],p[masks[3]]['Rf'])
    def evaluate(g,c,trace=False):
        moved=[geometry.rebind(r,g)for r in rows]
        errors=absolute_value_jac15(c,moved,False).reshape(-1,2)
        raw=[];details=[]
        for row in moved:
            value=scanline_branch_value_jac15(c,dict(row,normal_interval=None),False).reshape(-1,2)[0]
            raw.append(value)
            if trace:
                w=row['world'];uv=row['uv'][0];proj,depth,valid=project_central_valid(c,w)
                start=proj[:-1];d=np.diff(proj,axis=0)
                eligible=valid[:-1]&valid[1:]&(depth[:-1]>0)&(depth[1:]>0)&(abs(d[:,0])>1e-12)
                crossing=np.flatnonzero(eligible&(np.minimum(start[:,0],proj[1:,0])<=uv[0])&(np.maximum(start[:,0],proj[1:,0])>=uv[0]))
                if len(crossing):
                    t=(uv[0]-start[crossing,0])/d[crossing,0];yy=start[crossing,1]+t*d[crossing,1]
                    picked=int(np.argmin(yy));i=int(crossing[picked]);tt=float(t[picked]);worldpoint=w[i]+tt*(w[i+1]-w[i])
                    predicted=[float(uv[0]),float(yy[picked])];fallback=False
                else:
                    available=np.flatnonzero(valid&(depth>0)&np.isfinite(proj).all(axis=1))
                    i=int(available[np.argmin(abs(proj[available,0]-uv[0]))]);tt=None;worldpoint=w[i]
                    predicted=proj[i].tolist();fallback=True
                details.append(dict(physical_id=row['physical_curve_id'],uv=row['uv'].tolist(),
                    fixed_scanline_axis='x',branch='min_y',native_y_support=(row['uv'][0,1]+row['normal_interval'][0]).tolist(),
                    projected_complete_X_range=[float(proj[:,0].min()),float(proj[:,0].max())],
                    crossing_count=len(crossing),selected_segment_or_vertex=i,finite_segment_t=tt,endpoint_fallback=fallback,
                    selected_world_point_conditional_m=(worldpoint*UNIT).tolist(),predicted_nativeUV=predicted,
                    raw_error_xy=value.tolist(),outside_support_xy=errors[len(details)].tolist()))
        return dict(raw=np.asarray(raw),outside=errors,trace=details)
    co=camera([0,0,0,0]);cn=camera([1,1,1,1])
    baseline=evaluate(go,co,True);terminal=evaluate(gn,cn,True)
    replay=[]
    for version,result in [(old,baseline),(new,terminal)]:
        original=[r for r in version['review']['rows']if r['block']=='low'and r.get('source_role')=='independent_user_frame_report_only']
        assert len(original)==19
        expected=np.asarray([x['outside_support_xy_px'][0]for x in original])
        diff=float(np.max(abs(result['outside']-expected)));assert diff<1e-9;replay.append(diff)
    old_model=np.load(old['folder']/'real_camera_renderer_audit/complete_physical_camera_model_UNACCEPTED.npz',allow_pickle=False)
    old_camera=old_model['camera25'][np.flatnonzero(old_model['pts']==PTS).item()]
    new_meta=json.loads((new['folder']/'actual_terminal_four_source_images/actual_render_input_metadata.json').read_text(encoding='utf-8'))
    new_camera=np.asarray(next(x for x in new_meta['programme_cameras']if x['actual_exposure_pts']==PTS)['source_camera25'])
    camera_exact=[bool(np.array_equal(co,old_camera)),bool(np.array_equal(cn,new_camera))]
    assert all(camera_exact)
    # 5-family exact Shapley decomposition evaluates all component orders;
    # interaction is distributed transparently, no candidate state adopted.
    family_names=['geometry','shared_P','shared_L','PP_and_k1','view_Rf']
    cache={}
    for bits in itertools.product(range(2),repeat=5):cache[bits]=evaluate([go,gn][bits[0]],camera(bits[1:]))
    contributions={name:{k:np.zeros((19,2))for k in ['raw','outside']}for name in family_names}
    for i,name in enumerate(family_names):
        for bits in itertools.product(range(2),repeat=5):
            if bits[i]:continue
            n=sum(bits);weight=math.factorial(n)*math.factorial(4-n)/math.factorial(5)
            plus=list(bits);plus[i]=1;plus=tuple(plus)
            for k in ['raw','outside']:contributions[name][k]+=weight*(cache[plus][k]-cache[bits][k])
    for k in ['raw','outside']:assert np.max(abs(sum(v[k]for v in contributions.values())-(terminal[k]-baseline[k])))<1e-10
    def compact(result):return dict(raw_error_xy_px=result['raw'].tolist(),outside_support_xy_px=result['outside'].tolist(),
        all19=scalarstat(result['outside']),F1_10=scalarstat(result['outside'][:10]),Lane1_9=scalarstat(result['outside'][10:]))
    family_report={name:dict(shapley_raw_shift_px=contributions[name]['raw'].tolist(),shapley_postclip_shift_px=contributions[name]['outside'].tolist(),
        from_old_one_family=compact(cache[tuple(int(j==i)for j in range(5))]),
        from_current_restore_one_old=compact(cache[tuple(int(j!=i)for j in range(5))]))for i,name in enumerate(family_names)}
    # Individual geometry changes at both actual camera contexts. The exact
    # family interaction is NOT hidden by treating these one-at-a-time shifts
    # as an additive attribution.
    geometry_report={}
    for col in range(7):
        g=go.copy();g[col]=gn[col]
        geometry_report[geometry.parameter_names[col]]=dict(local_index=col,old_m=float(go[col]),new_m=float(gn[col]),
            at_old_camera=compact(evaluate(g,co)),at_current_camera=compact(evaluate(g,cn)))
    camera_coordinates={}
    for key in ['P','L','optics','Rf']:
        for col in range(len(old['physical'][key])):
            p={k:v.copy()for k,v in old['physical'].items()};p[key][col]=new['physical'][key][col]
            camera_coordinates[key+str(col)]=dict(old=float(old['physical'][key][col]),current=float(new['physical'][key][col]),
                at_old_world=compact(evaluate(go,cam(**p))),at_current_world=compact(evaluate(gn,cam(**p))))
    geometry_internal={}
    # The Lane Y coordinates have three coupled changes. Compare their
    # exact all-order allocations at both actual camera contexts.
    for cname,cc in [('old_camera',co),('current_camera',cn)]:
        cases={}
        for bits in itertools.product(range(2),repeat=3):
            g=go.copy()
            for j,col in enumerate([2,4,5]):g[col]=[go,gn][bits[j]][col]
            cases[bits]=evaluate(g,cc)
        parts={}
        for j,col in enumerate([2,4,5]):
            shift=np.zeros((19,2))
            for bits in itertools.product(range(2),repeat=3):
                if bits[j]:continue
                weight=math.factorial(sum(bits))*math.factorial(2-sum(bits))/math.factorial(3)
                pp=list(bits);pp[j]=1
                shift+=weight*(cases[tuple(pp)]['raw']-cases[bits]['raw'])
            parts[geometry.parameter_names[col]]=shift.tolist()
        geometry_internal[cname]=dict(raw_shift_exact_Shapley=parts,
            all3_raw_shift_px=(cases[(1,1,1)]['raw']-cases[(0,0,0)]['raw']).tolist(),
            other_geometry_coordinates_kept_old=True)
    fitcoverage=[]
    fit_entries=[x for x in packet['programme_absolute_rows']if x['block']=='low'and x['collection_role'].lower()=='fit'and
        x['source_row'].get('canonical_physical_geometry_id',x['source_row'].get('physical_curve_id'))in ['F1_outer','lane1_edge0']]
    assert len(fit_entries)==4
    for x in fit_entries:
        row=x['source_row'];p=int(x['group_actual_pts']);versions=[]
        for version,g in [(old,go),(new,gn)]:
            aa=version['aux'];counts=aa['aux__block_lengths']//4;stamps=aa['aux__canonical_pts'][sum(counts[:3]):sum(counts[:4])]
            slot=int(aa['aux__block_start'][3])+4*np.flatnonzero(stamps==p).item()
            physical={k:v.copy()for k,v in version['physical'].items()};physical['Rf']=version['q'][slot:slot+4]
            rr=absolute_value_jac15(cam(**physical),[geometry.rebind(row,g)],False).reshape(-1,2)*row['sigma'][:,None]
            original=next(y for y in version['review']['rows']if y['block']=='low'and y['actual_pts']==p and y['id']==row['id']and y.get('source_role')=='development_fit')
            error=float(np.max(abs(rr-np.asarray(original['outside_support_xy_px']))));assert error<1e-9
            versions.append(dict(outside_support_xy_px=rr.tolist(),point_norm_RMS=scalarstat(rr)['point_norm_RMS'],
                max_point_norm=scalarstat(rr)['max_point_norm'],review_replay_max_abs_px=error,Rf_global_slots=list(range(slot,slot+4))))
        fitcoverage.append(dict(id=row['id'],pts=p,source_role=row['source_role'],original_UV=row['uv'].tolist(),
            original_sigma_px=row['sigma'].tolist(),normal=row['normal'].tolist(),normal_interval=row['normal_interval'].tolist(),
            operator=row.get('curve_observation_kind')or'default finite nearest then original normal support',old_current=versions,
            FIT_is_separate_from_original19=True))
    external_start=int(old['aux']['aux__external_start']);external_pts=old['aux']['aux__external_canonical_pts']
    fixed_pts=int(external_pts[(34126-external_start)//4])
    report=dict(schema='rinkspace.b691-to-v3-original19-low-independent-held-source-component-causal-replay.v1',
        read_only=True,held_fitted=False,parameters_optimized=False,full_graph_loaded=False,new_solver_run=False,
        component_transport_is_diagnostic_only_no_model_adopted=True,source_packet=str(PACKET),source_packet_sha256=sha(PACKET),
        complete_event_tokens=[v['event']for v in [old,new]],checkpoint_sha256=[sha(v['checkpoint'])for v in [old,new]],
        complete_source_19_replay_max_abs_px=replay,actual_source_camera25_reconstruction_bitexact=camera_exact,
        source_role='independent_user_frame_report_only',actual_PTS=PTS,source_count=19,
        low_Rf_indices=old['view_slots'].tolist(),low_canonical_index=old['low_canonical_index'],
        low_Rf_are_not_external_fixed_HELD_columns=True,external_fixed_Rf_indices=[34126,34127,34128,34129],
        external_fixed_Rf_PTS=fixed_pts,original_external_fixed_values_unchanged=bool(np.array_equal(old['q'][34126:34130],new['q'][34126:34130])),
        programme_Rf_source_role='1851 actual low view variables coupled to original motion; report-only absolute UV does not freeze its Rf',
        old_and_current_low_physical=[{k:(v*UNIT if k in ['P','L']else v).tolist()for k,v in version['physical'].items()}for version in [old,new]],
        physical_units=dict(P='conditional metres',L='conditional camera-local metres',optics='PPx px/PPy px/k1',Rf='Rodrigues3 radians/logf native pixels'),
        current_moved_mount_indices=new['mount_indices'].tolist(),current_optics_indices=new['lens_indices'].tolist(),
        old_low_camera25=co.tolist(),current_low_camera25=cn.tolist(),old_world43=go.tolist(),current_world43=gn.tolist(),
        actual_optical_centre_old_current_m=[(cc[:3]*UNIT).tolist()for cc in [co,cn]],
        actual_optical_centre_change_m=((cn[:3]-co[:3])*UNIT).tolist(),
        actual_focal_old_current_px=np.exp([co[6],cn[6]]).tolist(),
        actual_PP_old_current_native_px=[co[8:10].tolist(),cn[8:10].tolist()],actual_k1_old_current=[float(co[11]),float(cn[11])],
        baseline=compact(baseline),current=compact(terminal),baseline_trace=baseline['trace'],current_trace=terminal['trace'],
        family_exact_Shapley_component_attribution=family_report,
        all32_family_diagnostic_cases={''.join(map(str,b)):compact(v)for b,v in cache.items()},
        individual_geometry_coordinate_changes=geometry_report,individual_camera_coordinate_changes=camera_coordinates,
        Lane_pitch_midpoint_halfwidth_exact_internal_diagnostic_attribution=geometry_internal,
        original_same_family18_FIT_source_coverage_and_actual_errors=fitcoverage,
        geometry_source_full_scene_API='SharedIceHockeyNumericGeometry.scene/source_scene + original material rebind; same frozen original scene, actual current q43',
        actual_full_scene_target_primitives_reconstructed_bitexact=full_scene_checks,
        actual_boundary_dims_not_transported_to_60x30=True,nominal_rink_dimensions_not_fitted_or_changed=True,
        original_fixed_external_Rf_solver_definition=dict(runtime=str(BASE/'run_common_hockey_soft_reference_full_source_candidate.py'),
            runtime_sha256=sha(BASE/'run_common_hockey_soft_reference_full_source_candidate.py'),fixed_rule='external_start+4*external.index[p]+c only for external.held_pts; c0..3',
            source_file_start_line=67,not_programme_report_only_PTS=True),
        row1_upper_world_Y_change_m=dict(row_pitch=-1.5*(gn[2]-go[2]),midpoint=gn[4]-go[4],
            half_width=gn[5]-go[5],total=-1.5*(gn[2]-go[2])+(gn[4]-go[4])+(gn[5]-go[5])),
        row1_right_X_change_m=float(gn[6]-go[6]),diagnostic_accuracy_claim=False,
        original_source_pixels_intervals_sigma_roles_and_rows_unchanged=True,seconds=time.perf_counter()-begun)
    out=BASE/'low_independent_held_component_audit/b691_to_v3_actual_held19_component_audit.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(out),actual_replay=replay,camera_bitexact=camera_exact,
        baseline_RMS=report['baseline']['all19']['point_norm_RMS'],current_RMS=report['current']['all19']['point_norm_RMS'],
        worst_lane_index13_raw_baseline=baseline['raw'][13].tolist(),worst_lane_current=terminal['outside'][13].tolist(),
        worst_lane_family_shapley={k:v['outside'][13].tolist()for k,v in contributions.items()},seconds=report['seconds']),indent=2))

if __name__=='__main__':main()
