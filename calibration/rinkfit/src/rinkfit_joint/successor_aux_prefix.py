"""Numerical complete-CP carry across an explicitly qualified V7 point suffix."""
from pathlib import Path
import copy,json
import numpy as np
from .io_utils import file_sha256
from .accepted_aux_prefix import _ref,_verified,_array
from .static_board_successor_prefix_qualification import SCHEMA,_shared_board_point,assert_board_source_qualification_current

SUFFIX_SCHEMA='rinkfit.qualified_physical_suffix_for_accepted_aux.v2_successor'


def _authority(packet,path):
    # This generation is V7. A future source-only descendant must retain this
    # explicit geometry authority, rather than attributing points to a curve.
    certpath=_verified(packet['board_material_source_qualification'],path.parent)
    cert=json.loads(certpath.read_text(encoding='utf8'))
    if cert.get('schema')!=SCHEMA:raise ValueError('Successor physical suffix currently requires the actual V7 geometry generation')
    assert_board_source_qualification_current(packet,path)
    bp=_verified(cert['actual_append_source_bundle'],certpath.parent);bundle=json.loads(bp.read_text(encoding='utf8'))
    return certpath,cert,bp,bundle


def _definition(value):
    value=copy.deepcopy(value);value.pop('source_supported_motion_flags',None)
    for block in value['blocks']:block.pop('constant',None);block.pop('fixed_indices',None)
    return value


def merge_successor_static_board_suffix(accepted_model_path,source_packet_path,output_model_path):
    accepted_path=Path(accepted_model_path).resolve();packet_path=Path(source_packet_path).resolve();out=Path(output_model_path).resolve()
    if out.exists():raise FileExistsError('Immutable successor physical model exists')
    accepted=json.loads(accepted_path.read_text(encoding='utf8'));packet=json.loads(packet_path.read_text(encoding='utf8'));cp,cert,bp,bundle=_authority(packet,packet_path)
    issued=json.loads(_verified(bundle['model'],bp.parent).read_text(encoding='utf8'));parent=json.loads(_verified(bundle['parent_model'],bp.parent).read_text(encoding='utf8'))
    if len(accepted['blocks'])!=len(parent['blocks'])or len(accepted['materials'])!=len(parent['materials']):raise ValueError('Accepted CP allocation differs from successor physical parent')
    def roles(value):
        value=copy.deepcopy(value);value.pop('source_supported_motion_flags',None)
        for block in value['blocks']:
            for key in ('values','constant','fixed_indices'):block.pop(key,None)
        return value
    if roles(accepted)!=roles(parent):raise ValueError('Accepted CP changed inherited projection/material roles')
    merged=copy.deepcopy(accepted);merged['blocks'].extend(copy.deepcopy(issued['blocks'][len(parent['blocks']):]));merged['materials'].extend(copy.deepcopy(issued['materials'][len(parent['materials']):]))
    for key,value in issued.items():
        if key not in parent:merged[key]=copy.deepcopy(value)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(model=_ref(out),accepted_model=_ref(accepted_path),source_packet=_ref(packet_path),geometry_certificate=packet['board_material_source_qualification'],old_physical_blocks=len(parent['blocks']),old_materials=len(parent['materials']),new_physical_blocks=len(merged['blocks'])-len(parent['blocks']),new_materials=len(merged['materials'])-len(parent['materials']),old_CP_or_source_UV_changed=False,new_optimization_performed=False)


def qualify_successor_physical_suffix(packet,packet_path,accepted_model,current_model_path,old_n):
    path=Path(packet_path).resolve();current_path=Path(current_model_path).resolve();current=json.loads(current_path.read_text(encoding='utf8'));cp,cert,bp,bundle=_authority(packet,path)
    issued=json.loads(_verified(bundle['model'],bp.parent).read_text(encoding='utf8'));parent=json.loads(_verified(bundle['parent_model'],bp.parent).read_text(encoding='utf8'));bindings=json.loads(_verified(bundle['track_material_bindings'],bp.parent).read_text(encoding='utf8'))
    old_b=len(accepted_model['blocks']);old_m=len(accepted_model['materials']);first=int(cert['original_raw_rows']);end=int(cert['raw_rows']);n=int(packet['row_count'])
    if old_n>first or end!=n or old_b!=len(parent['blocks'])or old_m!=len(parent['materials'])or len(current['blocks'])!=len(issued['blocks'])or len(current['materials'])!=len(issued['materials'])or len(bindings)!=len(current['blocks'])-old_b or len(bindings)!=len(current['materials'])-old_m:raise ValueError('Successor source/model allocation does not equal qualified physical suffix')
    prefix=copy.deepcopy(current);prefix['blocks']=prefix['blocks'][:old_b];prefix['materials']=prefix['materials'][:old_m];extras=[]
    for key in issued:
        if key not in parent:
            if current.get(key)!=issued[key]:raise ValueError('Successor model provenance changed')
            prefix.pop(key,None);extras.append(key)
    if _definition(prefix)!=_definition(accepted_model):raise ValueError('Successor current1959 physical prefix differs from exact complete CP')
    used_m=set();used_b=set()
    for binding in bindings:
        mi=binding['material_index'];bi=binding['geometry_block'];tid=binding['track_id']
        if mi in used_m or bi in used_b or bi<old_b or current['materials'][mi]!=issued['materials'][mi]:raise ValueError('Successor track/material allocation changed')
        used_m.add(mi);used_b.add(bi);actual_b,pathdesc=_shared_board_point(current,mi,tid,binding['board_instance_id'],old_m)
        if actual_b!=bi or pathdesc['advertisement_block']!=binding['advertisement_block']:raise ValueError('Successor source owner/block binding changed')
        a={k:v for k,v in current['blocks'][bi].items()if k not in ('values','constant')};b={k:v for k,v in issued['blocks'][bi].items()if k not in ('values','constant')}
        if a!=b:raise ValueError('Successor latent ABI/roles changed')
    if used_b!=set(range(old_b,len(current['blocks'])))or used_m!=set(range(old_m,len(current['materials'])))or sum(x['shared_native_rows']for x in bindings)!=end-first:raise ValueError('Successor allocation or source denominator contains unqualified suffix')
    material=_array(packet,'material_index',path.parent)
    if np.any(material[old_n:first]>=old_m):raise ValueError('Nongeometry source interval introduces an unqualified material')
    return dict(schema=SUFFIX_SCHEMA,current_model=_ref(current_path),source_qualification_certificate=packet['board_material_source_qualification'],geometry_qualification_certificate=packet['board_material_source_qualification'],actual_source_bundle=cert['actual_append_source_bundle'],source_issued_model=bundle['model'],source_issued_parent_model=bundle['parent_model'],track_material_bindings=bundle['track_material_bindings'],old_physical_blocks=old_b,new_physical_blocks=len(current['blocks']),old_materials=old_m,new_materials=len(current['materials']),qualified_suffix_blocks=len(bindings),qualified_suffix_materials=len(bindings),qualified_new_source_rows=end-first,geometry_source_first_row=first,geometry_source_end_row=end,additional_model_provenance_keys=extras,old_CP_model_is_not_replaced=True,old_physical_prefix_values_exact=True,old_projection_roles_and_topology_exact=True,suffix_is_not_part_of_old_checkpoint=True,source_qualification_required_before_consume=True)


def rebind_successor_engineering_nuisance(packet_path,candidate,current_model_path,output_path):
    from .accepted_aux_prefix import rebind_engineering_nuisance
    return rebind_engineering_nuisance(packet_path,candidate,output_path,current_model_path=current_model_path,physical_suffix_qualifier=qualify_successor_physical_suffix)
