"""Explicit finite C profile prefix issuer; no board or known-point authority.

V7 is validated in full. The frozen prefix checker is delegated only after the
new original Queries/pair bindings are independently checked here. Pixel means,
roles, W and all inherited source identities remain immutable.
"""
from pathlib import Path
import json,copy
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .board_source_qualification_guard import _load_ref
from .static_board_prefix_qualification import _json,_source_refs,_model_table
from . import static_board_prefix_qualification as prefix_helper
from . import static_board_successor_prefix_qualification as parent_guard
from . import tail_curve_source_append as source

SCHEMA='rinkfit.native_source_fit_qualification.v8_finite_ice_curve_prefix'
CONTRACT='actual_finite_ice_curve_after_static_board_v1'
EXECUTED_MODULE_REF=source.ref(__file__)
PARENT_GUARD_REF=source.ref(parent_guard.__file__)
SOURCE_READER_REF=source.ref(source.__file__)


def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache;path=Path(packet_path).resolve()
    ref=packet.get('board_material_source_qualification')
    if ref:
        cp,cert=_json(ref,path.parent,seen)
        if cert.get('schema')==SCHEMA:return assert_curve_source_prefix_qualification(packet,path,certificate=(cp,cert),verified_dependency_cache=seen)
    return parent_guard.assert_board_source_qualification_current(packet,path,verified_dependency_cache=seen)


def _proof(packet,parent,path,parentpath,bundlepath,cert,seen):
    b=read_json(bundlepath);base=bundlepath.parent;_source_refs(b,base,seen)
    if b.get('schema')!=source.BUNDLE_SCHEMA or b.get('reader_module_refs')!=[SOURCE_READER_REF]or b.get('source_ledger_preserved')is not True:raise ValueError('C actual source bundle/version differs')
    qpath=_load_ref(b['static_surface_qualification'],base,seen);sections,q,g=source.source_sections(qpath)
    rows=list(read_jsonl(_load_ref(b['source_rows_ref'],base,seen)));receipts=list(read_jsonl(_load_ref(b['qualification_receipts'],base,seen)));gate=read_json(_load_ref(b['qualification_gate'],base,seen))
    paired=b['strict_paired'];factor=2 if paired else 1;N=int(parent['row_count']);M=int(packet['row_count'])
    if (len(rows)!=3164*factor or len(receipts)!=len(rows)or M-N!=len(rows)or gate.get('schema')!=source.GATE_SCHEMA
        or gate['qualification']!=b['static_surface_qualification']or gate['source_rows_ref']!=b['source_rows_ref']
        or gate['qualification_receipts']!=b['qualification_receipts']or gate['source_generation']!=q['source_generation']
        or gate['source_sections']!=3164 or gate['source_exposures']!=80 or gate['original_candidate_denominator']!=6737
        or gate['unknown_or_rejected']!=3573 or gate.get('source_UV_modified')is not False
        or gate.get('statistical_precision_invented')is not False):raise ValueError('C complete actual-source gate denominator differs')
    parentmodel=read_json(_load_ref(b['parent_model'],base,seen));model=read_json(_load_ref(b['model'],base,seen))
    if parentmodel['materials']!=model['materials']or packet['material_binding']!=parent['material_binding']or _model_table(model)!=parent['material_binding']['materials']:raise ValueError('C changed existing finite physical material identity')
    if parentmodel!=model and not paired:raise ValueError('Coarse C cannot allocate or change physical values')
    if paired:_width_authority(model,parentmodel,sections,b,base,seen)
    if cert['inherited_physical_block_count']!=len(parentmodel['blocks'])or cert['inherited_material_count']!=len(parentmodel['materials']):raise ValueError('C recursive actual model prefix view differs')
    oldq=parent['queries'];queries=packet['queries']
    if len(queries)!=len(oldq)+len(rows)or queries[:len(oldq)]!=oldq:raise ValueError('C changed inherited Query/connectivity/order')
    proofs=[];expectedpairs=[]
    for j,(row,receipt)in enumerate(zip(rows,receipts)):
        s=sections[j//factor];original=s['source_row'];side=j%2 if paired else None;i=N+j;uv=original['source_flanks_uv'][side]if paired else original['source_uv'];rid=original['source_row_id']+('/flank'+str(side)if paired else'/coarse_midpoint')
        expectedquery=dict(row_index=i,source_factor_row_id=rid,**s['query']);operator=source.PAIRED if paired else source.MIDPOINT
        if (row.get('source_row_id')!=rid or row.get('derived_midpoint_source_row_id')!=original['source_row_id']
            or row.get('source_uv')!=uv or row.get('current_uv')!=uv or row.get('source_flanks_uv')!=original['source_flanks_uv']
            or row.get('normal')!=original['normal']or row.get('native_pts')!=original['native_pts']or row.get('shot_id')!='s09'
            or row.get('source_video_sha256')!=source.SOURCE_SHA or row.get('current_rgb_sha256')!=original['current_rgb_sha256']
            or row.get('joint_material_index')!=145 or row.get('joint_material_id')!='ice.C:segment0'or row.get('point_correspondence')is not False
            or row.get('entity_station_is_unknown')is not True or row.get('source_geometry_eligible')is not True
            or any(row.get(k)is not False for k in ('statistical_objective_eligible','numeric_statistics_ready','solver_eligible','pixels_from_model','physical_accuracy_claimed','source_position_covariance_issued'))
            or row.get('measurement_operator_resolved')!=operator or row.get('source_acquisition_query')!=s['query']
            or row.get('source_navigation_seed_uv')!=s['source_navigation_seed_uv']or row.get('source_side_index')!=side):raise ValueError('C redirects actual qualified UV/material/role/operator')
        if (receipt.get('original_source_row')!=original or receipt.get('chosen_uv')!=uv or receipt.get('source_row_id')!=rid
            or receipt.get('query')!=s['query']or receipt.get('qualification_ref')!=b['static_surface_qualification']
            or receipt.get('source_profile_coordinate_sha256')!=s['source_profile_coordinate_sha256']or receipt.get('estimator')!=operator):raise ValueError('C actual source receipt differs')
        if queries[len(oldq)+j]!=expectedquery:raise ValueError('C Query is not the original finite acquired source profile')
        proofs.append(dict(row_index=i,source_row=row,qualification_receipt=receipt,query=expectedquery,decision=s['decision']))
    return proofs


def _width_authority(model,parentmodel,sections,b,base,seen):
    wpath=_load_ref(b['local_width_qualification'],base,seen);w=read_json(wpath)
    n=len(parentmodel['blocks']);scope=sections
    if w.get('schema')!='rinkfit.source_local_paint_half_width_allocation.v1'or w.get('source_bundle_operator')!=source.PAIRED or w.get('parent_model')!=b['parent_model']or w.get('model')!=b['model']:raise ValueError('C half-width allocation has no separate actual source authority')
    if len(model['blocks'])!=n+len(scope)or model['blocks'][:n]!=parentmodel['blocks']or model['materials']!=parentmodel['materials']:raise ValueError('C width suffix changed inherited physical state')
    for k,v in parentmodel.items():
        if k not in ('blocks','measurement_widths','stroke_pair_model_bindings')and model.get(k)!=v:raise ValueError('C width suffix changed camera/scene/role')
    oldw=parentmodel.get('measurement_widths',[]);oldb=parentmodel.get('stroke_pair_model_bindings',[])
    if model['measurement_widths'][:len(oldw)]!=oldw or model['stroke_pair_model_bindings'][:len(oldb)]!=oldb or len(w['pairs'])!=len(scope):raise ValueError('C changed inherited paired operators')
    for j,(s,record)in enumerate(zip(scope,w['pairs'])):
        rid=s['source_row']['source_row_id'];block=model['blocks'][n+j];mw=model['measurement_widths'][len(oldw)+j];binding=model['stroke_pair_model_bindings'][len(oldb)+j]
        if (record['source_midpoint_id']!=rid or record['half_width_block']!=n+j or mw['width_scope_id']!=rid or mw['half_width_block']!=n+j
            or binding['pair_id']!=rid+'/strict_pair'or binding['width_scope_id']!=rid or binding['half_width_block']!=n+j
            or binding['material_index']!=145 or binding['source_world_signs']not in [[-1,1],[1,-1]]
            or binding['allowed_world_side_assignments']!=[[-1,1],[1,-1]]or block['name']!=record['block_name']
            or len(block['values'])!=1 or not 0<block['values'][0]<=1 or block['lower']!=[1e-12]or block['upper']!=[1.0]
            or block['constant']is not False or block['fixed_indices']!=[]or record['source_UV_modified']is not False
            or record['world_width_is_measured_truth']is not False or record['world_signs_are_material_truth']is not False):raise ValueError('C needs one local positive nuisance per actual same-frame profile section')


def assert_curve_source_prefix_qualification(packet,packet_path,*,parent_document=None,certificate=None,verified_dependency_cache=None):
    path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache
    cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    if (cert.get('schema')!=SCHEMA or cert.get('parent_recursion_contract')!=CONTRACT or cert.get('issuer_module_ref')!=EXECUTED_MODULE_REF
        or cert.get('recursive_parent_guard_module_ref')!=PARENT_GUARD_REF or cert.get('source_reader_module_ref')!=SOURCE_READER_REF
        or cert.get('reader_module_refs')!=[prefix_helper.EXECUTED_MODULE_REF]):raise ValueError('C issuer/recursive/delegated actual code contract differs')
    for ref in (EXECUTED_MODULE_REF,PARENT_GUARD_REF,SOURCE_READER_REF):_load_ref(ref,cp.parent,seen)
    pp=_load_ref(cert['parent_packet'],cp.parent,seen);parent=parent_document if parent_document is not None else read_json(pp)
    parent_guard.assert_board_source_qualification_current(parent,pp,verified_dependency_cache=seen)
    bpath=_load_ref(cert['actual_append_source_bundle'],cp.parent,seen)
    proof=_proof(packet,parent,path,pp,bpath,cert,seen)
    oldpairs=parent.get('stroke_pairs',[]);pairs=packet.get('stroke_pairs',[]);paired=read_json(bpath)['strict_paired']
    if pairs[:len(oldpairs)]!=oldpairs or len(pairs)!=len(oldpairs)+(3164 if paired else 0):raise ValueError('C paired source inherited prefix/count differs')
    view=dict(packet,queries=parent['queries'])
    if 'stroke_pairs'in parent:view['stroke_pairs']=parent['stroke_pairs']
    else:view.pop('stroke_pairs',None)
    arrays=prefix_helper._prefix(view,parent,path,pp,cert,seen)
    N=int(parent['row_count'])
    for j,p in enumerate(proof):
        i=N+j;r=p['source_row'];alpha=r['source_measured_alpha']
        if (packet['source_row_ids'][i]!=r['source_row_id']or int(arrays['native_pts'][i])!=r['native_pts']or arrays['shot_index'][i]!=8
            or not np.array_equal(arrays['uv'][i],r['current_uv'])or not np.array_equal(arrays['profile_normal'][i],r['normal'])
            or arrays['kind'][i]!=0 or arrays['material_index'][i]!=145 or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1
            or arrays['E'][i]!=2 or arrays['source_sigma'][i]!=1 or not np.array_equal(arrays['alpha_bounds'][i],r['source_acquisition_query']['alpha'])
            or not np.array_equal(arrays['query_uv'][i],r['source_navigation_seed_uv'])):raise ValueError('C native packet does not consume exact actual source curve/clock/normal')
        if paired:
            pair_index=len(oldpairs)+j//2
            if arrays['stroke_pair_index'][i]!=pair_index or arrays['stroke_source_side'][i]!=j%2:raise ValueError('C source flank pair order differs')
            pair=pairs[pair_index]
            if pair!=dict(id=r['derived_midpoint_source_row_id']+'/strict_pair',source_row_id=r['derived_midpoint_source_row_id'],rows=[N+2*(j//2),N+2*(j//2)+1],material_index=145,target_id='ice.C:segment0',shot_id='s09',native_pts=r['native_pts'],width_scope_id=r['derived_midpoint_source_row_id'],allowed_world_side_assignments=[[-1,1],[1,-1]],actual_flank_uv=r['source_flanks_uv'],source_normal=r['normal'],source_family_id=r['source_family_id'],source_refs=[read_json(bpath)['source_rows_ref']]):raise ValueError('C paired geometry redirects original two actual interfaces')
        elif arrays['stroke_pair_index'][i]!=-1 or arrays['stroke_source_side'][i]!=-1:raise ValueError('Coarse C midpoint cannot masquerade as flank pair')
    if 'qualified_rows_native'in cert and list(read_jsonl(_load_ref(cert['qualified_rows_native'],cp.parent,seen)))!=proof:raise ValueError('C native full-source proof differs')
    return proof


def finalize_curve_source_prefix_certificate(parent_packet_path,appended_packet_document,bundle_path,output_certificate_path,*,parent_document=None):
    pp=Path(parent_packet_path).resolve();out=Path(output_certificate_path).resolve();bpath=Path(bundle_path).resolve();packet=appended_packet_document
    if out.exists():raise FileExistsError('C source certificate is immutable')
    parent=parent_document if parent_document is not None else read_json(pp);b=read_json(bpath);model=read_json(_load_ref(b['parent_model'],bpath.parent,set()))
    cert=dict(schema=SCHEMA,parent_recursion_contract=CONTRACT,issuer_module_ref=EXECUTED_MODULE_REF,recursive_parent_guard_module_ref=PARENT_GUARD_REF,
        source_reader_module_ref=SOURCE_READER_REF,parent_packet=source.ref(pp),actual_append_source_bundle=source.ref(bpath),
        reader_module_refs=[prefix_helper.EXECUTED_MODULE_REF],issued_arrays=copy.deepcopy(packet['arrays']),
        original_raw_rows=parent['row_count'],raw_rows=packet['row_count'],appended_raw_rows=packet['row_count']-parent['row_count'],
        inherited_material_count=len(model['materials']),inherited_physical_block_count=len(model['blocks']),
        source_UV_preserved=True,statistical_precision_invented=False,source_width_is_not_noise=True)
    proof=assert_curve_source_prefix_qualification(packet,out.parent/'packet.json',parent_document=parent,certificate=(out,cert))
    f=out.parent/'qualified_source_rows_native.jsonl'
    with f.open('w',encoding='utf8')as s:
        for row in proof:s.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    cert['qualified_rows_native']=source.ref(f);write_json(out,cert);packet['board_material_source_qualification']=source.ref(out)
    packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
