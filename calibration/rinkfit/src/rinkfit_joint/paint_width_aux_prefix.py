"""Complete accepted CP plus separately qualified local paint-width allocation.

Existing physical values must equal the complete CP. Only the source-qualified
positive scalar nuisance suffix is initialized; no old dual is resumed.
"""
from pathlib import Path
import copy,json
import numpy as np
from .accepted_aux_prefix import _ref,_verified,_array
from .curve_source_prefix_qualification import SCHEMA,assert_board_source_qualification_current
from .successor_aux_prefix import _definition

SUFFIX_SCHEMA='rinkfit.qualified_physical_suffix_for_accepted_aux.v3_local_paint_width'


def _authority(packet,path):
    cp=_verified(packet['board_material_source_qualification'],path.parent);cert=json.loads(cp.read_text(encoding='utf8'))
    if cert.get('schema')!=SCHEMA:raise ValueError('Paint-width allocation requires the actual finite-curve prefix authority')
    assert_board_source_qualification_current(packet,path)
    bp=_verified(cert['actual_append_source_bundle'],cp.parent);bundle=json.loads(bp.read_text(encoding='utf8'))
    if bundle.get('strict_paired')is not True:raise ValueError('A midpoint approximation has no physical width authority')
    wp=_verified(bundle['local_width_qualification'],bp.parent);widths=json.loads(wp.read_text(encoding='utf8'))
    if widths.get('schema')!='rinkfit.source_local_paint_half_width_allocation.v1':raise ValueError('Width suffix has no separate source allocation contract')
    return cp,cert,bp,bundle,wp,widths


def _roles(value):
    value=copy.deepcopy(value);value.pop('source_supported_motion_flags',None)
    for block in value['blocks']:
        for key in ('values','constant','fixed_indices'):block.pop(key,None)
    return value


def _prefix_model(current,parent):
    prefix=copy.deepcopy(current);prefix['blocks']=prefix['blocks'][:len(parent['blocks'])]
    for key in ('measurement_widths','stroke_pair_model_bindings'):prefix[key]=prefix.get(key,[])[:len(parent.get(key,[]))]
    return prefix


def merge_paint_width_suffix(accepted_model_path,source_packet_path,output_model_path):
    accepted_path=Path(accepted_model_path).resolve();path=Path(source_packet_path).resolve();out=Path(output_model_path).resolve()
    if out.exists():raise FileExistsError('Immutable accepted paint-width model exists')
    accepted=json.loads(accepted_path.read_text(encoding='utf8'));packet=json.loads(path.read_text(encoding='utf8'));cp,cert,bp,b,wp,widths=_authority(packet,path)
    issued=json.loads(_verified(b['model'],bp.parent).read_text(encoding='utf8'));parent=json.loads(_verified(b['parent_model'],bp.parent).read_text(encoding='utf8'))
    if len(accepted['blocks'])!=len(parent['blocks'])or accepted['materials']!=parent['materials']or _roles(accepted)!=_roles(parent):raise ValueError('Accepted complete CP has different source/material/projection allocation or roles')
    if issued['blocks'][:len(parent['blocks'])]!=parent['blocks']or issued['materials']!=parent['materials']:raise ValueError('Issued paint-width candidate changed inherited physical state')
    merged=copy.deepcopy(accepted);merged['blocks']+=copy.deepcopy(issued['blocks'][len(parent['blocks']):])
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        if issued[key][:len(parent.get(key,[]))]!=parent.get(key,[]):raise ValueError('Width allocation changed existing pair identities')
        merged[key]=copy.deepcopy(accepted.get(key,[]))+copy.deepcopy(issued[key][len(parent.get(key,[])):])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(model=_ref(out),accepted_model=_ref(accepted_path),source_packet=_ref(path),local_width_qualification=_ref(wp),
        old_physical_blocks=len(accepted['blocks']),new_local_scalar_widths=len(widths['pairs']),materials_unchanged=True,
        old_complete_CP_values_exact=True,old_source_UV_W_Query_or_aux_modified=False,new_Solve_started=False)


def qualify_paint_width_physical_suffix(packet,packet_path,accepted_model,current_model_path,old_n):
    path=Path(packet_path).resolve();current_path=Path(current_model_path).resolve();current=json.loads(current_path.read_text(encoding='utf8'));cp,cert,bp,b,wp,widths=_authority(packet,path)
    issued=json.loads(_verified(b['model'],bp.parent).read_text(encoding='utf8'));parent=json.loads(_verified(b['parent_model'],bp.parent).read_text(encoding='utf8'))
    n=int(packet['row_count']);first=int(cert['original_raw_rows']);end=int(cert['raw_rows']);old_b=len(accepted_model['blocks']);old_m=len(accepted_model['materials']);records=widths['pairs']
    if (old_n!=first or end!=n or old_b!=len(parent['blocks'])or accepted_model['materials']!=parent['materials']
        or len(current['blocks'])!=old_b+len(records)or len(issued['blocks'])!=len(current['blocks'])
        or current['materials']!=accepted_model['materials']or end-first!=2*len(records)):raise ValueError('Paint width/current CP exact source allocation differs')
    if _definition(_prefix_model(current,parent))!=_definition(accepted_model):raise ValueError('Current old physical prefix differs from exact complete CP values/topology/roles')
    used=set();pairs=packet['stroke_pairs'][len(parent.get('stroke_pair_model_bindings',[])):]
    if len(pairs)!=len(records):raise ValueError('Width nuisance lacks exactly two actual flank rows')
    for j,(record,pair)in enumerate(zip(records,pairs)):
        bi=record['half_width_block'];block=current['blocks'][bi];expected=issued['blocks'][bi]
        if (bi!=old_b+j or bi in used or block['name']!=record['block_name']or len(block['values'])!=1 or not 0<block['values'][0]<=1
            or block.get('constant')is not False or block.get('fixed_indices')!=[]or block.get('lower')!=[1e-12]or block.get('upper')!=[1.0]
            or {k:v for k,v in block.items()if k!='values'}!={k:v for k,v in expected.items()if k!='values'}
            or pair['id']!=record['source_midpoint_id']+'/strict_pair'or pair['rows']!=[first+2*j,first+2*j+1]
            or pair['material_index']!=145 or pair['width_scope_id']!=record['source_midpoint_id']):raise ValueError('New local paint width changed qualified source pair/DOF/identity')
        used.add(bi)
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        if current.get(key)!=issued.get(key):raise ValueError('New width scope or finite side branch changed without a new numerical contract')
    arrays={key:_array(packet,key,path.parent)for key in ('kind','station','material_index','stroke_pair_index','stroke_source_side')}
    if np.any(arrays['kind'][first:end]!=0)or np.any(arrays['material_index'][first:end]!=145):raise ValueError('Width suffix is not actually consumed by finite C curves')
    return dict(schema=SUFFIX_SCHEMA,current_model=_ref(current_path),source_qualification_certificate=packet['board_material_source_qualification'],
        geometry_qualification_certificate=packet['board_material_source_qualification'],actual_source_bundle=cert['actual_append_source_bundle'],
        source_issued_model=b['model'],source_issued_parent_model=b['parent_model'],local_width_qualification=b['local_width_qualification'],
        old_physical_blocks=old_b,new_physical_blocks=len(current['blocks']),old_materials=old_m,new_materials=old_m,
        qualified_suffix_blocks=len(records),qualified_suffix_materials=0,qualified_new_source_rows=end-first,
        geometry_source_first_row=first,geometry_source_end_row=end,old_measurement_width_count=len(parent.get('measurement_widths',[])),
        old_pair_binding_count=len(parent.get('stroke_pair_model_bindings',[])),old_CP_model_is_not_replaced=True,old_physical_prefix_values_exact=True,
        old_projection_roles_and_topology_exact=True,suffix_is_not_part_of_old_checkpoint=True,source_qualification_required_before_consume=True,
        local_scalar_width_is_not_metric_truth=True,new_flank_station_RS_Query_independent=True)


def rebind_paint_width_engineering_nuisance(packet_path,candidate,current_model_path,output_path):
    from .accepted_aux_prefix import rebind_engineering_nuisance
    return rebind_engineering_nuisance(packet_path,candidate,output_path,current_model_path=current_model_path,physical_suffix_qualifier=qualify_paint_width_physical_suffix)
