"""Compact same-denominator dense packet writes against an exact immutable parent.

This continuation is separate from the frozen first-append serializer. It does
not change source geometry or authorize changed qualifications; native source
and budget guards still validate the resulting packet.
"""
from pathlib import Path
from .dense_joint_packet import IDS,_registry_columns,_read_parent,_json_equal
from .dense_joint_proofs import _write


def write_dense_packet(path,value):
    """Write a full consumer view without duplicating original IDs or Query/W."""
    path=Path(path).resolve()
    if not isinstance(value,dict)or 'dense_source_binding'not in value or 'dense_parent_packet'not in value:
        from .packet_storage import write_packet
        write_packet(path,value);return
    registry=value['dense_source_binding'];n0=int(registry['row_start']);nnew=int(registry['count']);n=int(value['row_count'])
    if n!=n0+nnew:raise ValueError('Dense packet full denominator differs')
    parent=_read_parent(value['dense_parent_packet'],path.parent,(str(path).casefold(),))
    if int(parent['row_count'])==n0:
        from .packet_storage import write_packet
        write_packet(path,value);return
    if int(parent['row_count'])!=n or parent['dense_source_binding']!=registry:raise ValueError('Same-N writer requires the exact existing raw source registry')
    suffix=_registry_columns(registry,path.parent)
    for key,tail in zip(IDS,suffix):
        current=value[key]
        if len(current)!=n or current[n0:]!=tail or current!=parent[key]:raise ValueError('Same-N writer refuses changed, truncated or duplicated source IDs')
    missing=set(parent)-set(value)
    if missing:raise ValueError('Hydrated same-N packet is missing inherited fields: '+','.join(sorted(missing)))
    raw={'schema':value['schema'],'dense_parent_packet':value['dense_parent_packet']};appends={}
    for key,val in value.items():
        if key in {'schema','dense_parent_packet','dense_parent_append'}|set(IDS):continue
        if key in ('groups','queries'):
            if val!=parent[key]:raise ValueError('Same-N writer refuses changing original raw Query/W rows: '+key)
        elif key in ('source_manifests','recapture_rows')and isinstance(parent.get(key),list):
            before=parent[key]
            if not isinstance(val,list)or val[:len(before)]!=before:raise ValueError('Same-N packet changed inherited metadata prefix: '+key)
            if len(val)>len(before):appends[key]=val[len(before):]
        elif key in ('allowed_paths','engineering_pixel_dependency_groups'):
            if val!=parent[key]:raise ValueError('Same-N packet changed original source branch or dependence authority: '+key)
        elif key not in parent or val!=parent[key]:raw[key]=val
    if appends:raw['dense_parent_append']=appends
    _write(path,raw)
