"""Reproduce original C interfaces before superseding derived midpoint factors.

The original source rows, UV and IDs remain valid. Only their redundant
midpoint factor role may be replaced by the two actual saved material edges.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from scipy.ndimage import spline_filter
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .tail_curve_source_append import ref,SOURCE_SHA
from .board_source_qualification_guard import _load_ref

BUNDLE_SCHEMA='rinkfit.actual_C_midpoint_to_interfaces_supersession_source.v1'
OPERATOR='frozen_original_two_C_material_interfaces_own_RS_v1'


def replay_original_flanks(rgb,receipt):
    from . import physical_source_operator as narrow
    from . import native_paint_wide as wide
    module=wide if receipt['operator_version']==wide.CONFIG['version']else narrow
    if receipt['operator_version']!=module.CONFIG['version']or receipt['configuration']!=module.CONFIG:raise ValueError('Original paint acquisition/replay version differs')
    seed=np.asarray(receipt['original_navigation_seed_uv'],float);n=np.asarray(receipt['normal'],float);t=np.asarray(receipt['tangent'],float);offsets=np.asarray(receipt['normal_offsets'],float);toff=np.asarray(receipt['tangent_offsets'],float);channel=int(receipt['channel'])
    channels=module._channels(np.asarray(rgb));coeff=np.asarray([spline_filter(q,order=3)for q in channels]);coords=seed+toff[:,None,None]*t+offsets[None,:,None]*n
    values=module._sample(coeff,coords.reshape(-1,2)).reshape(len(toff),len(offsets),3);positions=[];boundaries=0
    for window in receipt['frozen_flank_windows'][2]:
        lo,hi=window['normal_window_offsets'];indices=np.flatnonzero((offsets>=lo-1e-8)&(offsets<=hi+1e-8));gradient=int(window['polarity'])*np.gradient(values[2,:,channel],.25);k=int(indices[np.argmax(gradient[indices])]);boundaries+=int(k in(indices[0],indices[-1]));shift=0.
        if 0<k<len(offsets)-1:
            v=gradient[k-1:k+2];den=v[0]-2*v[1]+v[2]
            if abs(den)>1e-10:shift=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
        positions.append(float(offsets[k]+shift))
    center=seed+.5*sum(positions)*n;width=positions[1]-positions[0];points=np.asarray([seed+(.5*sum(positions)+side*.5*width)*n for side in(-1,1)])
    profile_error=float(np.max(abs(values[:,:,channel]-np.asarray(receipt['actual_five_profiles']))));operator=module.replay_paint_midline(rgb,receipt)
    return dict(flanks=points,current_uv=center,positions=positions,profile_max_abs_error=profile_error,window_boundary_count=boundaries,
        existing_frozen_midpoint_replay=operator,reader_module_ref=ref(module.__file__),material_reselected=False)


def prepare_supersession_source(audit_rows_path,parent_packet_path,qualification_path,output):
    from .native_source import iter_scope_native_rgb
    from .source_texture_tracklets import _source_mask
    from rinkspace.vision_programme_ice_review import _candidate_context_excluded
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Immutable midpoint operator supersession generation exists')
    audit=list(read_jsonl(audit_rows_path));parent=read_json(parent_packet_path);q=read_json(qualification_path);seen=set();gp=_load_ref(q['source_generation'],Path(qualification_path).parent,seen);gen=read_json(gp)
    if len(audit)!=13 or q.get('authority')!='assistant_exact_current_native_pixel_review':raise ValueError('No complete exact original C source derivations')
    A={}
    for key in('uv','native_pts','admission','fit_role','material_index','kind'):
        d=parent['arrays'][key];A[key]=np.memmap(_load_ref(d,Path(parent_packet_path).parent,seen),dtype=d['dtype'],shape=tuple(d['shape']),mode='r')
    wanted={int(x['native_pts'])for x in audit};actors={}
    for r in gen['actor_ledgers']:
        for row in read_jsonl(_load_ref(r,gp.parent,seen)):
            if row['native_pts']in wanted:actors[int(row['native_pts'])]=row
    roles=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/roles/frame_roles.json');scope=_load_ref(gen['scope'],gp.parent,seen);frames={p:(rgb,decode)for p,rgb,decode in iter_scope_native_rgb(scope,sorted(wanted),roles_path=roles)}
    out.mkdir(parents=True);rows=[];proof=[];failures=[]
    for item in audit:
        i=item['raw_index'];original=item['original_source_row'];rid=item['source_row_id'];pts=int(item['native_pts']);uv=item['correct_original_packet_UV'];rgb,decode=frames[pts]
        digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest();binding=original['native_source_binding'];receipt=original['source_operator_result']['source_material_receipt'];mask,_=_source_mask(rgb,actors[pts],gen['source_video'],pts,'s09')
        if (parent['source_row_ids'][i]!=rid or A['uv'][i].tolist()!=uv or int(A['native_pts'][i])!=pts or A['admission'][i]!=1 or A['fit_role'][i]!=1 or A['kind'][i]!=0 or A['material_index'][i]!=145
            or original.get('source_geometry_eligible')is not True or original.get('physical_surface')!='ice'or binding['source_video_sha256']!=SOURCE_SHA or binding['decoded_native_RGB_sha256']!=digest
            or binding['time_base']!='1/12800'or binding['native_pts']!=pts):raise ValueError('Supersession redirects original correct source/role/native clock')
        result=replay_original_flanks(rgb,receipt);actual=np.asarray(item['source_flanks_uv']);error=float(np.max(abs(result['flanks']-actual)));center_error=float(np.max(abs(result['current_uv']-uv)));clear=not any(_candidate_context_excluded(mask,point,item['source_normal'])for point in[uv,*actual])
        good=bool(error<=1e-10 and center_error<=1e-10 and result['profile_max_abs_error']<=1e-10 and result['window_boundary_count']==0 and clear)
        proof.append(dict(raw_index=i,source_row_id=rid,native_pts=pts,current_RGB_sha256=digest,existing_journal=item['journal'],existing_source_derivation=item,
            native_decode=decode,actual_flank_replay_max_abs_error_px=error,actual_midpoint_replay_max_abs_error_px=center_error,actual_five_profile_replay_error=result['profile_max_abs_error'],
            frozen_replay_module=result['reader_module_ref'],actor_exclusion_clear=clear,qualified_for_operator_supersession=good,
            old_UV_correct_and_retained=True,world_station_or_width_known=False,reason='same measured image midpoint is a deterministic function of these two actual interfaces; replacing factor avoids double count, not rejection by residual'))
        if not good:failures.append(i);continue
        seed=item['original_acquisition_seed_uv'];n=np.asarray(item['source_normal']);bounds=item['original_normal_offsets_bounds'];query=dict(alpha=bounds,native_uv=[(np.asarray(seed)+a*n).tolist()for a in bounds],normal=n.tolist(),source_acquisition_window_only=True,source_query_index_is_not_world_station=True,original_frozen_profile_ref=item['journal'],original_midpoint_source_row_id=rid)
        for side in(0,1):
            rows.append(dict(source_row_id='native_C_operator_supersession_v1/'+rid+'/flank'+str(side),source_family_id=original['source_family_id'],shot_id='s09',native_pts=pts,time_base=[1,12800],
                source_uv=actual[side].tolist(),current_uv=actual[side].tolist(),current_rgb_sha256=digest,source_video_sha256=SOURCE_SHA,
                joint_material_id='ice.C:segment0',joint_material_index=145,target_id='ice.C:segment0',normal=n.tolist(),point_correspondence=False,
                source_geometry_eligible=True,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,fit_role='development',frame_role='development_candidate',
                surface_class='ice',surface_instance_id='ice',pixels_from_model=False,entity_station_is_unknown=True,source_width_is_not_position_covariance=True,
                measurement_operator_resolved=OPERATOR,source_side_index=side,source_flanks_uv=actual.tolist(),source_acquisition_query=query,
                source_navigation_seed_uv=seed,source_measured_alpha=item['source_alpha_flanks'][side],derived_midpoint_source_row_id=rid,
                half_width_scope_id='source_operator_supersession/'+rid,old_correct_midpoint_raw_index=i,old_correct_midpoint_UV=uv,
                original_source_row_and_replay_proof_index=len(proof)-1,source_position_correction=False,old_midpoint_factor_role='navigation_derived_nonindependent',
                frozen_original_material_receipt=receipt))
    def lines(f,values):
        with f.open('w',encoding='utf8')as stream:
            for x in values:stream.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
    lines(out/'source_rows.jsonl',rows);lines(out/'replay_proofs.jsonl',proof)
    manifest=dict(schema=BUNDLE_SCHEMA,parent_packet=ref(parent_packet_path),old_derivations=ref(audit_rows_path),same_current_C_material_qualification=ref(qualification_path),
        source_rows_ref=ref(out/'source_rows.jsonl'),actual_replay=ref(out/'replay_proofs.jsonl'),reader_module_refs=[ref(__file__)],source_video=gen['source_video'],
        original_midpoints=13,qualified_midpoint_supersessions=len(rows)//2,new_actual_flank_rows=len(rows),pending_old_rows=failures,
        old_raw_UV_and_ID_preserved=True,reason_is_operator_definition_not_residual=True,statistical_precision_invented=False,
        packet_admission_or_groups_changed=False,new_source_rows_not_yet_installed=True)
    write_json(out/'manifest.json',manifest)
    if failures:raise ValueError('Some exact old C interfaces failed current source replay: '+str(failures))
    return ref(out/'manifest.json')
