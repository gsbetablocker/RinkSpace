"""Correct per-vertex operator metadata without changing measured pixels.

A source segment may contain both an endpoint entity and ordinary edge rows.
Their common source statistics are retained while the measurement binding is
resolved separately for every original vertex.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .materials import measurement_definition,semantic_key
from .source_worker import assemble_s09

def revise_bindings(parent,output):
 parent=Path(parent);out=Path(output)
 if out.exists():raise ValueError('A new binding revision generation is required')
 rows=list(read_jsonl(parent/'relocated_rows.jsonl.gz'));changes=[];by=defaultdict(list)
 for row in rows:
  definition=measurement_definition(row);operator=definition['operator'];key=semantic_key({**row,'measurement_operator':operator})
  if key!=row.get('joint_material_id') or definition!=row.get('measurement_definition'):
   changes.append({'source_row_id':row['source_row_id'],'before':{'joint_material_id':row.get('joint_material_id'),'operator':row.get('measurement_operator_resolved')},'after':{'joint_material_id':key,'operator':operator},'reason':'per_vertex_typed_binding_overrides_segment_endpoint_class','source_uv_unchanged':True,'current_native_uv_unchanged':True})
  row['joint_material_id']=key;row['measurement_operator_resolved']=operator;row['measurement_definition']=definition
  by[key or 'unbound:'+row['shot_id']].append(row)
 registry=[]
 for key,members in sorted(by.items()):
  r=members[0];registry.append({'id':key,'source_target':r.get('target_id'),'surface':r.get('physical_surface'),'measurement_operator':r['measurement_operator_resolved'],'boundary_role':r.get('boundary_role'),'physical_entity':(r.get('target_id') or '').split('.print.')[0],'source_row_count':len(members),'current_pixel_supported_identity_candidate_rows':sum(x['source_support_status']=='source_supported_identity_candidate' for x in members),'shots':sorted({x['shot_id'] for x in members}),'native_pts':sorted({x['native_pts'] for x in members}),'descriptor_candidates':[],'identity_accepted':False,'identity_status':'current_native_support_and_typed_descriptor_audited_pending_multiview_shape_crosswalk','geometry_frozen':False})
 out.mkdir(parents=True);write_jsonl(out/'relocated_rows.jsonl.gz',rows);write_jsonl(out/'operator_revision.jsonl.gz',changes)
 oldreg=read_json(parent/'joint_material_registry.json');write_json(out/'joint_material_registry.json',{**oldreg,'materials':registry})
 support={'path':str((parent/'retained_support.jsonl.gz').resolve()),'sha256':file_sha256(parent/'retained_support.jsonl.gz'),'unchanged':True};write_json(out/'retained_support.ref.json',support)
 summary={**read_json(parent/'summary.json'),'binding_revision':{'schema':'rinkfit.joint9.per_vertex_operator_revision.v1','parent_dir':str(parent.resolve()),'parent_rows_sha256':file_sha256(parent/'relocated_rows.jsonl.gz'),'changed_binding_rows':len(changes),'source_or_current_uv_changed':False,'retained_support':support},'materials':len(registry)};write_json(out/'summary.json',summary)
 assemble_s09(out/'relocated_rows.jsonl.gz',out/'s09_packet');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(revise_bindings(a.parent,a.out)))
