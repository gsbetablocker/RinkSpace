"""Rebind a verified fitted prefix and unchanged raw suffix to current source facts.

This is a metadata-only boundary. It never imports a rejected whole tail state,
reprofiles sources, estimates poses, or reissues any source qualification.
"""
from pathlib import Path
from copy import deepcopy
import json
from .io_utils import read_json
from .dense_joint_proofs import _ref,_checked,_write


def rebind_complete_prefix_raw_tail(*,template_overlay,current_packet,current_qualification,output):
    template_path=Path(template_overlay).resolve();packet_path=Path(current_packet).resolve()
    qualification_path=Path(current_qualification).resolve();out=Path(output).resolve()
    template=read_json(template_path);qual=read_json(qualification_path)
    current=json.loads(packet_path.read_text(encoding='utf-8-sig'))
    current_ref=_ref(packet_path);qualref=_ref(qualification_path)
    if template.get('schema')!='rinkfit.segmented_native_numeric_auxiliary_overlay.v1'or qual['source_packet']!=current_ref:raise ValueError('An exact segmented template and current source qualification are required')
    flags={'numerical_initialization_only':True,'true_RS_verified':False,'source_UV_ID_PTS_qualification_Query_W_unchanged':True,'old_AL_duals_imported':False,'full_current_checkpoint_claimed':False}
    if any(template.get(k)is not v for k,v in flags.items()):raise ValueError('Numeric template boundary differs')
    pieces=template.get('pieces',[])
    if len(pieces)!=2 or pieces[0].get('kind')!='complete_native_checkpoint_prefix'or pieces[1].get('kind')!='raw_packet_numeric_seed':raise ValueError('Only a true fitted prefix and raw packet suffix can be rebound')
    oldpacket_path=_checked(template['source_packet'],template_path.parent)
    oldqual=read_json(_checked(template['source_geometry_qualification'],template_path.parent))
    old=json.loads(oldpacket_path.read_text(encoding='utf-8-sig'))
    if set(current.get('arrays',{}))!=set(old.get('arrays',{}))or len(current['arrays'])!=18:raise ValueError('Qualified append packet raw ABI differs')
    for name,spec in current['arrays'].items():
        if name not in ('admission','fit_role')and spec!=old['arrays'][name]:raise ValueError('Qualification rebind changed a raw source or raw seed column: '+name)
    for name in ('actual_column_identity_sha256','compact_registry','raw_rows','new_rows','parent_rows'):
        if oldqual[name]!=qual[name]:raise ValueError('Qualified raw geometry registry or denominator differs: '+name)
    n=int(qual['raw_rows']);n0=int(qual['parent_rows'])
    if int(template['raw_rows'])!=n or pieces[0]['source_packet']!=qual['parent_packet']or (int(pieces[0]['start']),int(pieces[0]['count']),int(pieces[0]['source_offset']))!=(0,n0,0)or (int(pieces[1]['start']),int(pieces[1]['count']),int(pieces[1]['source_offset']))!=(n0,n-n0,n0):raise ValueError('Fitted prefix/raw suffix partition differs')
    if pieces[1]['arrays']!={key:old['arrays'][key]for key in ('station','row','alpha','path')}:raise ValueError('Template did not use the actual raw numeric suffix')
    verification=read_json(_checked(pieces[0]['complete_state_verification'],template_path.parent))
    if verification.get('passed')is not True or verification['state_token']!=pieces[0]['complete_state_token']or int(verification['raw_rows'])!=n0 or verification['source_identity']!=pieces[0]['source_packet']['sha256']:raise ValueError('Complete fitted prefix verification differs')
    closure=read_json(_checked(template['projection_closure'],template_path.parent))
    if closure.get('clock_rigs_materials_scene_and_first4_metadata_exact')is not True or closure.get('all_other_blocks_exact')is not True or closure['current_model']!=template['candidate_model']or closure['prefix_model']!=pieces[0]['physical_model']:raise ValueError('Fitted physical prefix projection closure differs')
    updated=deepcopy(template);updated.update(source_packet=current_ref,source_geometry_qualification=qualref,issuer=_ref(__file__),numeric_overlay_template=_ref(template_path),source_qualification_only_rebind=True,prefix_numeric_columns_reused_exact=True,rejected_captured_tail_state_imported=False,source_search_repeated=False)
    updated['pieces'][1].update(source_packet=current_ref,arrays={key:current['arrays'][key]for key in ('station','row','alpha','path')})
    if updated['pieces'][0]!=pieces[0]:raise ValueError('Rebinding changed a fitted prefix numerical value or reference')
    out.mkdir(exist_ok=False,parents=True)
    overlayref=_write(out/'numeric_overlay.json',updated)
    child=dict(schema=current['schema'],dense_parent_packet=current_ref,dense_numeric_auxiliary_overlay=overlayref)
    packetref=_write(out/'packet.json',child)
    result=dict(schema='rinkfit.actual_fitted_prefix_raw_tail_qualification_rebind.v1',passed=True,packet=packetref,overlay=overlayref,current_source_packet=current_ref,current_source_qualification=qualref,template_overlay=_ref(template_path),prefix_complete_state_token=pieces[0]['complete_state_token'],raw_rows=n,prefix_rows=n0,raw_numeric_seed_tail_rows=n-n0,source_UV_ID_PTS_qualification_Query_W_unchanged=True,original_eighteen_raw_array_specs_unchanged=True,prefix_fitted_numeric_refs_exact=True,rejected_captured_tail_imported=False,new_array_files=0,source_pixels_decoded=0,source_search_repeated=False,complete_checkpoint_state_search_repeated=False,old_AL_or_physical_state_imported=False,source_qualification_and_weights_rebind_required=True,true_RS_verified=False,quality_accepted=False)
    _write(out/'receipt.json',result);return result
