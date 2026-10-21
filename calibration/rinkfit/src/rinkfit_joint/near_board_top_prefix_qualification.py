"""Exact source-qualified near top silhouette append after the existing prefix.

This is an explicit next-generation interface. It neither edits a running packet
nor replaces its frozen reader. The installer issues a real geometry-only suffix;
its independent guard recursively checks the accepted parent, immutable arrays,
actual native optical profiles, current role/mask provenance and exact Queries.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from . import near_board_top_source as source
from . import curve_source_prefix_qualification as parent_guard
from . import static_board_prefix_qualification as helper
from .static_ice_prefix_qualification import _array
from .board_source_qualification_guard import _load_ref
from .io_utils import read_json,read_jsonl,write_json,file_sha256

SCHEMA='rinkfit.native_near_board_top_prefix_qualification.v1'
BUNDLE_SCHEMA='rinkfit.native_near_board_top_qualified_bundle.v1'
CONTRACT='actual_iceward_cap_top_optical_curve_after_existing_qualified_prefix_v1'
EXECUTED_MODULE_REF=source.ref(__file__)
SOURCE_READER_REF=source.ref(source.__file__)
PARENT_GUARD_REF=source.ref(parent_guard.__file__)


def _json(reference,base,seen):
    p=_load_ref(reference,base,seen)
    return p,read_json(p)


def source_qualification(qualification_path,*,redecode=False):
    qp=Path(qualification_path).resolve();q=read_json(qp);seen=set()
    helper._source_refs(q,qp.parent,seen)
    if q.get('physical_interface_qualification_status')!='qualified_current_top_interface':
        raise ValueError('Pending near-board physical inner rim: optical q50 cannot be installed as board.top truth')
    if (q.get('schema')!=source.QUALIFICATION_SCHEMA or q.get('source_geometry_qualified')is not True
        or q.get('source_operator')!=source.OPERATOR
        or any(q.get(k)is not False for k in ('source_UV_changed','physical_accuracy_claimed','physical_boundary_mean_exact','source_covariance_issued','source_station_known','known_world_point'))):
        raise ValueError('Near top source qualification claims unsupported authority')
    gp,g=_json(q['source_generation'],qp.parent,seen);helper._source_refs(g,gp.parent,seen)
    if (g.get('schema')!=source.SCHEMA or g.get('source_UV_from_model_or_screen')is not False
        or g['source_video']['sha256']!=source.SOURCE_SHA or g['source_video']['time_base']!='1/12800'
        or SOURCE_READER_REF not in g['reader_modules']):raise ValueError('Native near top reader/source changed')
    rp,replay=_json(q['numeric_source_replay'],qp.parent,seen)
    if (replay.get('passed')is not True or replay.get('source_UV_max_abs_difference_px')!=0.
        or replay.get('raw_profile_max_abs_difference')!=0. or replay.get('all_current_RGB_hashes_and_exclusion_masks_valid')is not True):
        raise ValueError('Near top exact current source replay incomplete')
    reviewpath,review=_json(q['source_semantic_review'],qp.parent,seen)
    if (review.get('authority')!='assistant_current_original_RGB_review'
        or review.get('accepted_semantic')!='iceward_top_occluding_silhouette_of_continuous_board'
        or review.get('source_video_sha256')!=source.SOURCE_SHA
        or review.get('optical_half_rise_is_exact_geometric_surface_mean')is not False
        or review.get('cap_thickness_or_outer_edge_used')is not False
        or review.get('physical_inner_interface_resolved')is not True
        or not review.get('exact_visible_interface_choice')
        or not review.get('bevel_shadow_exclusion_source_evidence')):raise ValueError('Near top semantic surface review differs')
    helper._source_refs(review,reviewpath.parent,seen)
    ledger=list(read_jsonl(_load_ref(g['frame_ledger'],gp.parent,seen)));bypts={r['native_pts']:r for r in ledger}
    scope=read_json(_load_ref(g['scope'],gp.parent,seen));shot=next(s for s in scope['shots']if s['id']=='s06')
    if [r['native_pts']for r in ledger]!=list(range(shot['first_pts'],shot['last_pts']+256,256)):
        raise ValueError('Near top complete native shot ledger differs')
    for frame in ledger:
        if frame['navigation_rows']:_load_ref(frame['mask'],gp.parent,seen)
    allrows=list(read_jsonl(_load_ref(q['source_rows'],qp.parent,seen)))
    rows=list(read_jsonl(_load_ref(q['fit_source_rows'],qp.parent,seen)))
    roles=read_json(_load_ref(q['roles'],qp.parent,seen))['frame_roles']
    if rows!=[r for r in allrows if r['frame_role']=='development_candidate'] or len(rows)!=q['development_rows'] or len(allrows)!=g['qualified_optical_rows']:
        raise ValueError('Near top actual source/DEV denominator differs')
    ids=set()
    for r in allrows:
        f=bypts[r['native_pts']];p=r['source_profile'];x,y=r['navigation_seed_uv']
        token=hashlib.sha256(json.dumps(p,separators=(',',':'),sort_keys=True).encode()).hexdigest()
        cross=source._transition(np.asarray(p['median_native_gray']),10,42,1,p['grey_plateau'],p['iceward_plateau']-p['grey_plateau'])
        quantiles=[float(x-26+v)for v in cross]if cross is not None else None
        if (r['source_row_id']in ids or r['source_uv']!=r['current_uv'] or r['shot_id']!='s06'
            or r['source_video_sha256']!=source.SOURCE_SHA or r['time_base']!=[1,12800] or r['actual_pts']!=r['native_pts']
            or r['current_rgb_sha256']!=f['decoded_native_BGR_sha256'] or r['source_profile_sha256']!=token
            or quantiles!=p['q10_q50_q90_x'] or r['source_uv']!=[quantiles[1],float(y)]
            or r['normal']!=[1.,0.] or not p['current_exclusion_clear']
            or r['surface_class']!='board' or r['material_surface']!='iceward_upper_rim_of_grey_board_cap'
            or r['surface_instance_id']!='near_continuous_board_iceward_top_silhouette'
            or r['joint_material_id']!='board.top' or r['measurement_operator_resolved']!=source.OPERATOR
            or r['point_correspondence']is not False or r['entity_station_is_unknown']is not True
            or r['source_physical_boundary_mean_exact']is not False or r['source_geometry_eligible']is not True
            or any(r[k]is not False for k in ('pixels_from_model','ice_H_used','physical_accuracy_claimed','source_position_covariance_issued','statistical_objective_eligible','numeric_statistics_ready','solver_eligible'))):
            raise ValueError('Near top raw UV/profile/surface/clock contract differs')
        ids.add(r['source_row_id'])
        if r in rows and (roles.get(str(r['native_pts']))!={'shot_id':'s06','role':'development_candidate'}or r['fit_role']!='development'):
            raise ValueError('Protected source exposure cannot be appended')
    if redecode:
        actual=source.replay_source_generation(gp)
        if actual!=replay:raise ValueError('Near top sealed source replay differs from current original RGB')
    return rows,q,g


def prepare_bundle(qualification_path,parent_packet_path,parent_model_path,output,*,redecode=True):
    """Seal a nonempty actual source bundle for an explicit accepted parent."""
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Near top source bundle must be a new immutable directory')
    rows,q,g=source_qualification(qualification_path,redecode=redecode)
    parent=read_json(parent_packet_path);model=read_json(parent_model_path)
    if helper._model_table(model)!=parent['material_binding']['materials']:raise ValueError('Near top changes inherited physical material topology')
    indices=[i for i,m in enumerate(model['materials'])if m['label']=='board.top']
    if len(indices)!=1:raise ValueError('One existing continuous board.top identity required')
    mi=indices[0];mat=model['materials'][mi]
    if len(mat['paths'])!=8 or {p['boundary_segment']for p in mat['paths']}!=set(range(8)) or any(p.get('kind')!=3 or p.get('fixed_height_fraction')!=1 or p.get('physical_point')for p in mat['paths']):
        raise ValueError('Near source top must remain the existing continuous upper curve, never contact or a known point')
    if not rows:raise ValueError('Empty source adapter is not an installation')
    rows=[dict(r,joint_material_index=mi)for r in rows];out.mkdir(parents=True)
    receipts=[dict(source_row_id=r['source_row_id'],native_pts=r['native_pts'],chosen_uv=r['current_uv'],
        qualification_ref=source.ref(qualification_path),original_source_profile_sha256=r['source_profile_sha256'],
        source_optical_support_interval_x=r['source_optical_support_interval_x'],estimator=source.OPERATOR,
        material_index=mi,material_id='board.top',point_correspondence=False,world_station_known=False,
        exact_geometric_half_rise_claimed=False,statistical_covariance_issued=False)for r in rows]
    def lines(name,items):
        with (out/name).open('w',encoding='utf8')as f:
            for row in items:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    lines('source_rows.jsonl',rows);lines('qualification_receipts.jsonl',receipts)
    write_json(out/'material_map.json',dict(semantic_to_material={'board.top':dict(material_index=mi,geometry_ready=True)}))
    gate=dict(schema=source.QUALIFICATION_SCHEMA,qualification=source.ref(qualification_path),source_rows_ref=source.ref(out/'source_rows.jsonl'),
        qualification_receipts=source.ref(out/'qualification_receipts.jsonl'),actual_rows=len(rows),source_geometry_qualified=True,
        parent_packet=source.ref(parent_packet_path),source_UV_modified=False,new_statistical_precision_invented=False,
        world_station_known=False,physical_accuracy_claimed=False,source_operator=source.OPERATOR)
    write_json(out/'gate.json',gate)
    bundle=dict(schema=BUNDLE_SCHEMA,parent_packet=source.ref(parent_packet_path),parent_model=source.ref(parent_model_path),model=source.ref(parent_model_path),
        source_generation=q['source_generation'],source_qualification=source.ref(qualification_path),source_rows_ref=source.ref(out/'source_rows.jsonl'),
        qualification_receipts=source.ref(out/'qualification_receipts.jsonl'),qualification_gate=source.ref(out/'gate.json'),
        roles=q['roles'],material_map=source.ref(out/'material_map.json'),reader_module_refs=[SOURCE_READER_REF,EXECUTED_MODULE_REF],
        source_ledger_preserved=True,actual_source_rows=len(rows),source_rows_nonempty=True,shared_unknown_world_points_allocated=0,
        source_covariance_issued=False,source_UV_from_model=False,complete_accepted_parent_unchanged=True)
    write_json(out/'manifest.json',bundle)
    write_json(out/'chunk.json',dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',source_rows_ref=bundle['source_rows_ref'],
        geometry_rows_ref=bundle['source_rows_ref'],source_gate_manifest=source.ref(out/'manifest.json'),source_dataset_tag='native_near_board_top_v1',
        actual_source_rows=len(rows),all_requested_frames_accounted=True,retain_unmeasured_candidates_in_source_journal_only=True))
    return source.ref(out/'manifest.json')


def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache;path=Path(packet_path).resolve()
    cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)
    if cert.get('schema')!=SCHEMA:return parent_guard.assert_board_source_qualification_current(packet,path,verified_dependency_cache=seen)
    return assert_near_top_prefix(packet,path,certificate=(cp,cert),verified_dependency_cache=seen)


def assert_near_top_prefix(packet,packet_path,*,parent_document=None,certificate=None,verified_dependency_cache=None):
    path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache
    cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    if (cert.get('schema')!=SCHEMA or cert.get('parent_recursion_contract')!=CONTRACT
        or cert.get('issuer_module_ref')!=EXECUTED_MODULE_REF or cert.get('source_reader_module_ref')!=SOURCE_READER_REF
        or cert.get('recursive_parent_guard_module_ref')!=PARENT_GUARD_REF
        or cert.get('reader_module_refs')!=[helper.EXECUTED_MODULE_REF]):raise ValueError('Near top exact recursive issuer code differs')
    for r in (EXECUTED_MODULE_REF,SOURCE_READER_REF,PARENT_GUARD_REF):_load_ref(r,cp.parent,seen)
    pp=_load_ref(cert['parent_packet'],cp.parent,seen);parent=read_json(pp)if parent_document is None else parent_document
    parent_guard.assert_board_source_qualification_current(parent,pp,verified_dependency_cache=seen)
    bp,bundle=_json(cert['actual_append_source_bundle'],cp.parent,seen);helper._source_refs(bundle,bp.parent,seen)
    if bundle.get('schema')!=BUNDLE_SCHEMA or bundle['parent_packet']!=cert['parent_packet']:raise ValueError('Near top bundle parent differs')
    original,q,g=source_qualification(_load_ref(bundle['source_qualification'],bp.parent,seen))
    rows=list(read_jsonl(_load_ref(bundle['source_rows_ref'],bp.parent,seen)));receipts=list(read_jsonl(_load_ref(bundle['qualification_receipts'],bp.parent,seen)))
    model=read_json(_load_ref(bundle['model'],bp.parent,seen));mi=next(i for i,m in enumerate(model['materials'])if m['label']=='board.top')
    if helper._model_table(model)!=parent['material_binding']['materials']or packet['material_binding']!=parent['material_binding']:
        raise ValueError('Near top changes inherited model topology/material binding')
    if rows!=[dict(r,joint_material_index=mi)for r in original]or len(rows)!=len(receipts)or packet['row_count']-parent['row_count']!=len(rows):raise ValueError('Near top qualified suffix lost exact source denominator')
    oldq=parent['queries'];expected=[dict(row_index=parent['row_count']+j,source_factor_row_id=r['source_row_id'],**r['source_acquisition_query'])for j,r in enumerate(rows)]
    if packet['queries']!=oldq+expected:raise ValueError('Near top redirects finite actual source Query')
    arrays=helper._prefix(dict(packet,queries=oldq),parent,path,pp,cert,seen);n=int(parent['row_count']);proof=[]
    for j,(r,receipt)in enumerate(zip(rows,receipts)):
        i=n+j
        if (packet['source_row_ids'][i]!=r['source_row_id']or arrays['native_pts'][i]!=r['native_pts']or arrays['shot_index'][i]!=5
            or arrays['uv'][i].tolist()!=r['current_uv']or arrays['profile_normal'][i].tolist()!=r['normal']
            or arrays['kind'][i]!=0 or arrays['material_index'][i]!=mi or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1
            or arrays['E'][i]!=2 or arrays['source_sigma'][i]!=1 or arrays['stroke_pair_index'][i]!=-1 or arrays['stroke_source_side'][i]!=-1
            or arrays['query_uv'][i].tolist()!=r['source_navigation_seed_uv']or arrays['alpha_bounds'][i].tolist()!=r['source_acquisition_query']['alpha']
            or packet['allowed_paths'][str(i)]!=list(range(8))
            or receipt['source_row_id']!=r['source_row_id']or receipt['chosen_uv']!=r['current_uv']
            or receipt['original_source_profile_sha256']!=r['source_profile_sha256']or receipt['qualification_ref']!=bundle['source_qualification']):
            raise ValueError('Near top native packet does not consume exact top source UV/clock/material/operator')
        proof.append(dict(row_index=i,source_row=r,qualification_receipt=receipt))
    if 'qualified_rows_native'in cert and list(read_jsonl(_load_ref(cert['qualified_rows_native'],cp.parent,seen)))!=proof:
        raise ValueError('Near top native full-source proof differs')
    return proof


def finalize_near_top_prefix(parent_packet,packet,bundle_path,certificate_path,*,parent_document=None):
    out=Path(certificate_path).resolve();pp=Path(parent_packet).resolve()
    if out.exists():raise FileExistsError('Near top prefix certificate is immutable')
    parent=read_json(pp)if parent_document is None else parent_document
    cert=dict(schema=SCHEMA,parent_recursion_contract=CONTRACT,issuer_module_ref=EXECUTED_MODULE_REF,
        source_reader_module_ref=SOURCE_READER_REF,recursive_parent_guard_module_ref=PARENT_GUARD_REF,
        reader_module_refs=[helper.EXECUTED_MODULE_REF],parent_packet=source.ref(pp),actual_append_source_bundle=source.ref(bundle_path),
        issued_arrays=copy.deepcopy(packet['arrays']),original_raw_rows=parent['row_count'],raw_rows=packet['row_count'],
        appended_raw_rows=packet['row_count']-parent['row_count'],source_UV_preserved=True,statistical_precision_invented=False)
    proof=assert_near_top_prefix(packet,out.parent/'packet.json',parent_document=parent,certificate=(out,cert))
    f=out.parent/'qualified_source_rows_native.jsonl'
    with f.open('w',encoding='utf8')as stream:
        for row in proof:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    cert['qualified_rows_native']=source.ref(f);write_json(out,cert);packet['board_material_source_qualification']=source.ref(out)
    packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet


def append_bundle(parent_packet,bundle_path,output):
    """Create a real next packet; no caller input is mutated or overwritten."""
    from .reference_source_append import append_constraint_source_chunks
    b=read_json(bundle_path);base=Path(bundle_path).resolve().parent
    if source.ref(parent_packet)!=b['parent_packet']:raise ValueError('Explicit accepted parent differs; prepare a new bundle')
    def finalize(packet,out,parent):
        rows=list(read_jsonl(source.verified(b['source_rows_ref'])));n=int(parent['row_count'])
        for key in ('alpha','query_uv','alpha_bounds'):
            d=packet['arrays'][key];a=np.memmap(d['file'],dtype=d['dtype'],shape=tuple(d['shape']),mode='r+')
            for j,r in enumerate(rows):a[n+j]=r['source_measured_alpha']if key=='alpha'else r['source_navigation_seed_uv']if key=='query_uv'else r['source_acquisition_query']['alpha']
            a.flush();del a;packet['arrays'][key]={**d,'sha256':file_sha256(d['file'])}
        packet['queries']=parent['queries']+[dict(row_index=n+j,source_factor_row_id=r['source_row_id'],**r['source_acquisition_query'])for j,r in enumerate(rows)]
        return finalize_near_top_prefix(parent_packet,packet,bundle_path,out/'qualified_certificate.json',parent_document=parent)
    return append_constraint_source_chunks(parent_packet,[source.ref(base/'chunk.json')],source.verified(b['model']),source.verified(b['material_map']),source.verified(b['roles']),output,finalize_packet_with_parent=finalize)
