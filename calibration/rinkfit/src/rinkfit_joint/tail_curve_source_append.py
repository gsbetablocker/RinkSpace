"""Actual qualified C paint source gate and finite acquisition Queries.

Pixel midpoints are an explicitly approximate coarse operator. The strict pair
branch uses both original interfaces, their separate RS rows, and a per-section
positive half-width nuisance. No source width is a covariance or world truth.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from .io_utils import file_sha256,object_sha256,read_jsonl,read_json,write_json
from .board_source_qualification_guard import _load_ref
from .static_board_prefix_qualification import _source_refs,_model_table

SOURCE_SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
BUNDLE_SCHEMA='rinkfit.native_finite_ice_curve_qualified_bundle.v1'
GATE_SCHEMA='rinkfit.actual_two_boundary_C_paint_qualification.v1'
MIDPOINT='actual_two_material_boundary_image_midpoint_coarse_v1'
PAIRED='actual_two_material_boundary_separate_RS_local_width_v1'


def ref(f):
    f=Path(f).resolve();return dict(file=str(f),sha256=file_sha256(f))


def source_sections(qualification_path):
    qp=Path(qualification_path).resolve();q=read_json(qp);seen=set();_source_refs(q,qp.parent,seen)
    if (q.get('schema')!='rinkfit.tail_C_current_source_material_qualification.v1'
        or q.get('authority')!='assistant_exact_current_native_pixel_review'
        or q.get('source_rows_qualified')!=3164 or q.get('candidate_decisions')!=6737
        or q.get('actual_all_6737_candidate_UV_replay_max_abs_diff_px')!=0
        or any(q.get(k)is not False for k in ('source_UV_modified','source_truth_from_model','source_station_known','statistical_covariance_issued','reviewed_images_pending'))):
        raise ValueError('C source material qualification is not complete and source based')
    gp=_load_ref(q['source_generation'],qp.parent,seen);g=read_json(gp)
    if g['source_video']['sha256']!=SOURCE_SHA or g['source_video']['time_base']!='1/12800' or g.get('source_UV_from_model_or_screen')is not False:raise ValueError('C native RGB identity differs')
    _source_refs({k:g[k]for k in ('scope','screen_state','actor_ledgers','reader_modules','candidates','frame_ledger')},gp.parent,seen)
    src=list(read_jsonl(_load_ref(q['source_rows'],qp.parent,seen)))
    decisions=list(read_jsonl(_load_ref(q['decisions'],qp.parent,seen)))
    approved={(x['native_pts'],x['candidate_order_0based']):x for x in decisions if x['status']=='qualified_C_paint_width_center'}
    if len(src)!=3164 or len(decisions)!=6737 or len(approved)!=3164:raise ValueError('C full qualified and rejected denominator differs')
    wanted={int(x['native_pts'])for x in src};profiles={}
    for frame in read_jsonl(_load_ref(g['candidates'],gp.parent,seen)):
        if int(frame['source_binding']['source_pts']) in wanted:profiles[int(frame['source_binding']['source_pts'])]=frame
    if len(profiles)!=80:raise ValueError('C actual profile exposure coverage differs')
    out=[]
    for row in src:
        pts=int(row['native_pts']);idx=row['original_candidate_reference'];order=int(idx['candidate_order_0based']);qi=int(idx['query_index']);frame=profiles[pts]
        curve=frame['curve_outputs'][0];actual=curve['source_candidates'][order];reading={int(x['source_query_index']):x for x in curve['native_source_readings']['readings']}[qi]
        uv=np.asarray(row['current_uv'],float);flanks=np.asarray(row['source_flanks_uv'],float);normal=np.asarray(row['normal'],float)
        seed=np.asarray(reading['source_search_point_uv'],float);bound=np.asarray(reading['source_search_normal_interval_px'],float)
        if (row['source_uv']!=row['current_uv'] or row['joint_material_id']!='ice.C:segment0' or row['shot_id']!='s09'
            or row.get('source_geometry_eligible')is not True or row.get('pixels_from_model')is not False
            or row.get('entity_station_is_unknown')is not True or row.get('station_unknown_and_not_shared_across_source_query_indices')is not True
            or actual['current_uv']!=row['source_uv'] or actual['source_flanks_uv']!=row['source_flanks_uv']
            or row['decision']!=approved[(pts,order)] or np.max(abs(normal-np.asarray(reading['source_search_normal'])))>1e-12
            or flanks.shape!=(2,2) or not np.isfinite(flanks).all() or abs(np.linalg.norm(normal)-1)>1e-10
            or np.max(abs(flanks.mean(0)-uv))>1e-10 or frame['source_binding']['decoded_native_rgb_sha256']!=row['current_rgb_sha256']):raise ValueError('C actual source mean/flanks/profile/decision differs '+str((pts,order,qi, row['decision']==approved[(pts,order)],actual['current_uv']==row['source_uv'],actual['source_flanks_uv']==row['source_flanks_uv'],np.array_equal(normal,reading['source_search_normal']),float(np.max(abs(flanks.mean(0)-uv))),frame['source_binding']['decoded_native_rgb_sha256']==row['current_rgb_sha256'])))
        alpha=(flanks-seed)@normal
        if np.max(abs(seed+alpha[:,None]*normal-flanks))>1e-9 or np.any(alpha<bound[0])or np.any(alpha>bound[1]):raise ValueError('Actual C flanks outside original acquired finite Query')
        query=dict(alpha=bound.tolist(),native_uv=[(seed+a*normal).tolist()for a in bound],normal=normal.tolist(),
            source_acquisition_window_only=True,source_query_index_is_not_world_station=True,
            source_profile_ref=g['candidates'],source_profile_native_pts=pts,source_profile_query_index=qi)
        out.append(dict(source_row=row,actual_candidate=actual,query=query,
            source_navigation_seed_uv=seed.tolist(),source_alpha_midpoint=float(np.dot(uv-seed,normal)),
            source_alpha_flanks=alpha.tolist(),source_full_profile_ref=g['candidates'],
            source_profile_coordinate_sha256=object_sha256(dict(offsets=reading['actual_source_normal_offsets_px'],values=reading['actual_source_color_profile'])),
            decision=approved[(pts,order)],source_width_px=float(np.linalg.norm(flanks[1]-flanks[0]))))
    return out,q,g


def build_curve_source_bundle(qualification_path,parent_packet_path,parent_model_path,roles_path,output,*,paired=False):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('C source generation is immutable')
    sections,q,g=source_sections(qualification_path);model=read_json(parent_model_path);parent=read_json(parent_packet_path)
    mi=[i for i,m in enumerate(model['materials'])if m['label']=='ice.C:segment0']
    if mi!=[145]or model['materials'][145]['paths'][0]['kind']!=0 or _model_table(model)!=parent['material_binding']['materials']:raise ValueError('C physical finite line identity changed')
    roles=read_json(roles_path)['frame_roles'];out.mkdir(parents=True);rows=[];receipts=[]
    for section in sections:
        original=section['source_row'];rid=original['source_row_id'];pts=int(original['native_pts'])
        if roles[str(pts)]!={'shot_id':'s09','role':'development_candidate'}:raise ValueError('C fitting exposure is not development')
        sides=(0,1)if paired else (None,)
        for side in sides:
            row=copy.deepcopy(original);row['joint_material_index']=145;row['measurement_operator_resolved']=PAIRED if paired else MIDPOINT
            row['source_row_id']=rid+('/flank'+str(side)if paired else '/coarse_midpoint')
            row['derived_midpoint_source_row_id']=rid;row['point_correspondence']=False
            row['source_uv']=row['current_uv']=original['source_flanks_uv'][side]if paired else original['source_uv']
            row['source_side_index']=side;row['source_family_id']=original['source_family_id']
            row['zero_width_image_midpoint_is_coarse_approximation']=not paired
            row['physical_accuracy_claimed']=False;row['source_position_covariance_issued']=False
            row['source_acquisition_query']=section['query'];row['source_navigation_seed_uv']=section['source_navigation_seed_uv']
            row['source_measured_alpha']=section['source_alpha_flanks'][side]if paired else section['source_alpha_midpoint']
            row['source_width_px']=section['source_width_px'];row['half_width_scope_id']=rid if paired else None
            rows.append(row);receipts.append(dict(source_row_id=row['source_row_id'],original_source_row=original,
                qualification_ref=ref(qualification_path),query=section['query'],source_profile_coordinate_sha256=section['source_profile_coordinate_sha256'],
                original_candidate_order_0based=original['original_candidate_reference']['candidate_order_0based'],
                chosen_uv=row['current_uv'],normal=row['normal'],source_side_index=side,
                estimator=row['measurement_operator_resolved'],status='qualified_actual_C_material_interfaces',
                local_width_is_numerical_nuisance_not_metric_truth=paired,source_mean_unchanged=True))
    def lines(f,items):
        with f.open('w',encoding='utf8')as s:
            for row in items:s.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    lines(out/'source_rows.jsonl',rows);lines(out/'qualification_receipts.jsonl',receipts)
    gate=dict(schema=GATE_SCHEMA,qualification=ref(qualification_path),source_generation=q['source_generation'],
        source_rows_ref=ref(out/'source_rows.jsonl'),qualification_receipts=ref(out/'qualification_receipts.jsonl'),
        source_sections=3164,source_exposures=80,actual_rows=len(rows),original_candidate_denominator=6737,unknown_or_rejected=3573,
        unresolved_queries_retained=1574,nonfinite_failures_retained=98,operator=PAIRED if paired else MIDPOINT,
        source_UV_modified=False,statistical_precision_invented=False,source_width_is_not_E_sigma_or_tolerance=True,
        across_exposure_station_or_width_correspondence_assumed=False,source_family_exposure_correlation_preserved=True)
    write_json(out/'qualification_gate.json',gate)
    manifest=dict(schema=BUNDLE_SCHEMA,model=ref(parent_model_path),parent_packet=ref(parent_packet_path),parent_model=ref(parent_model_path),
        source_rows_ref=ref(out/'source_rows.jsonl'),qualification_receipts=ref(out/'qualification_receipts.jsonl'),
        source_generation=q['source_generation'],static_surface_qualification=ref(qualification_path),qualification_gate=ref(out/'qualification_gate.json'),
        roles=ref(roles_path),reader_module_refs=[ref(__file__)],source_ledger_preserved=True,operator=gate['operator'],source_sections=3164,actual_source_rows=len(rows),
        strict_paired=paired,source_profile_payload_copied=False,physical_accuracy_claimed=False)
    write_json(out/'manifest.json',manifest)
    write_json(out/'material_map.json',dict(semantic_to_material={'ice.C:segment0':dict(material_index=145,geometry_ready=True)}))
    adapter=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',source_rows_ref=manifest['source_rows_ref'],geometry_rows_ref=manifest['source_rows_ref'],
        source_gate_manifest=ref(out/'manifest.json'),source_dataset_tag='native_tail_C_paired_v1'if paired else'native_tail_C_coarse_v1',
        actual_source_rows=len(rows),all_requested_frames_accounted=True,retain_unmeasured_candidates_in_source_journal_only=True)
    write_json(out/'chunk.json',adapter);return ref(out/'manifest.json')


def install_actual_queries(packet,source_rows,out,parent):
    n=int(parent['row_count']);rows=list(read_jsonl(source_rows));queries=list(parent['queries'])
    for key in ('alpha','query_uv','alpha_bounds'):
        d=packet['arrays'][key];a=np.memmap(d['file'],dtype=d['dtype'],shape=tuple(d['shape']),mode='r+')
        for j,row in enumerate(rows):
            value=row['source_measured_alpha']if key=='alpha'else row['source_navigation_seed_uv']if key=='query_uv'else row['source_acquisition_query']['alpha']
            a[n+j]=value
        a.flush();del a;packet['arrays'][key]={**d,'sha256':file_sha256(d['file'])}
    for j,row in enumerate(rows):queries.append(dict(row_index=n+j,source_factor_row_id=row['source_row_id'],**row['source_acquisition_query']))
    packet['queries']=queries
    packet['new_C_operator_approximation']=dict(operator=rows[0]['measurement_operator_resolved'],image_midpoint_is_physical_projection_truth=False,
        correlated_repeated_profiles_not_independent_precision=True,old_source_UV_and_flags_unchanged=True)
    return packet
