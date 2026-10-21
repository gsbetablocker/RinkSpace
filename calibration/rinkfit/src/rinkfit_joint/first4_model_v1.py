"""Versioned first-four-shot activity view of the complete native12 state.
Keeps the original physical container and freezes unreferenced old shot controls.
Asset edits are semantic, never numeric-index transplantation.
"""
from pathlib import Path
from copy import deepcopy
from .io_utils import read_json,write_json,file_sha256,object_sha256

def _ref(p):
 p=Path(p);return {'path':str(p.resolve()),'sha256':file_sha256(p),'bytes':p.stat().st_size}
def _motion_blocks(shot):
 out={int(shot['reference_angles_block'])}
 if 'control_coordinates' in shot:
  for c in shot['control_coordinates']:out.add(int(c['block']))
 else:out.update(map(int,shot['control_blocks']))
 return out

def build_first4_model(base_path,asset_path,asset_audit_path,chart_repair_path,out,*,s01_material_indices=()):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 base=read_json(base_path);asset=read_json(asset_path);audit=read_json(asset_audit_path);repair=read_json(chart_repair_path)
 model=deepcopy(base);bm={b['name']:i for i,b in enumerate(model['blocks'])}
 if len(bm)!=len(model['blocks']):raise ValueError('Nonunique complete-container block names')
 am={b['name']:i for i,b in enumerate(asset['blocks'])};mm={m['label']:i for i,m in enumerate(model['materials'])}
 if len(mm)!=len(model['materials']):raise ValueError('Nonunique material labels')
 protected=set(map(int,s01_material_indices));closures=[];appended=[];blockmap={}
 def mapblock(i):
  i=int(i);name=asset['blocks'][i]['name']
  if name not in bm:
   bm[name]=len(model['blocks']);model['blocks'].append(deepcopy(asset['blocks'][i]));appended.append({'source_block':i,'block':bm[name],'name':name})
  blockmap[i]=bm[name];return bm[name]
 def pathcopy(p):
  q=deepcopy(p)
  for k,v in list(q.items()):
   if k.endswith('_block') and isinstance(v,int) and v>=0:q[k]=mapblock(v)
  return q
 for rec in audit['closure_missing_paths']:
  label=rec['material'];mi=mm[label];src=asset['materials'][next(i for i,m in enumerate(asset['materials']) if m['label']==label)]
  oldpaths=model['materials'][mi]['paths'];pi=next(i for i,p in enumerate(oldpaths) if p['id']==rec['path_id'])
  if mi in protected:raise ValueError('Closure rechart touches preserved S01 material '+label)
  old=deepcopy(oldpaths[pi]);new=pathcopy(next(p for p in src['paths'] if p['id']==rec['path_id']))
  if old['points']!=new['points'][:-1] or new['points'][-1]!=new['points'][0]:raise ValueError('Closure is not an explicit repeated endpoint')
  oldpaths[pi]=new;closures.append({'material':label,'material_index':mi,'path_index':pi,'path_id':rec['path_id'],'station_multiplier':rec['new_station_from_old_station_multiplier'],'old_path_sha256':object_sha256(old),'new_path_sha256':object_sha256(new)})
 changes=[]
 for rec in repair['changed_blocks']:
  name=rec['name'];dst=bm[name];src=am[name];old=deepcopy(model['blocks'][dst]);new=deepcopy(asset['blocks'][src])
  model['blocks'][dst]=new;changes.append({'name':name,'block':dst,'old':old,'new':new})
 for mat in asset['materials']:
  if mat['label'] in mm:continue
  if not mat['label'].startswith('Ad24.print.native1518_ring_ink_'):continue
  ni=len(model['materials']);model['materials'].append({**deepcopy(mat),'paths':[pathcopy(p) for p in mat['paths']]});mm[mat['label']]=ni
 scope=('s01','s02','s03','s04');active_blocks=set().union(*(_motion_blocks(base['shots'][s]) for s in scope))
 old_only=set().union(*(_motion_blocks(v) for s,v in base['shots'].items() if s not in scope))-active_blocks
 freeze=[]
 for i in sorted(old_only):
  b=model['blocks'][i];freeze.append({'block':i,'name':b['name'],'old_constant':b.get('constant',False),'new_constant':True});b['constant']=True
 model['shots']={s:deepcopy(base['shots'][s]) for s in scope}
 manifolds=base.get('linear_control_manifolds',[])
 model['linear_control_manifolds']=[deepcopy(x) for x in manifolds if int(x['block']) in active_blocks]
 if any(v['reference_shot'] not in scope for v in model['rigs'].values()):raise ValueError('Rig reference outside first4 scope')
 for s in scope:
  if model['shots'][s]!=base['shots'][s]:raise ValueError('Shot layout changed')
 if model['rigs']!=base['rigs'] or model['source_clock']!=base['source_clock']:raise ValueError('Projection fixed metadata changed')
 receipt={'schema':'rinkfit.first4_complete_native_activity_view.v1','base':_ref(base_path),'asset':_ref(asset_path),'asset_audit':_ref(asset_audit_path),'chart_repair':_ref(chart_repair_path),'shots':list(scope),'rigs':list(model['rigs']),'one_scene':True,'source_clock_exact':True,'rig_metadata_exact':True,'all_first4_shot_motion_layout_exact':True,'parent_blocks':len(base['blocks']),'blocks':len(model['blocks']),'parent_materials':len(base['materials']),'materials':len(model['materials']),'closure_recharts':closures,'changed_chart_blocks':changes,'appended_blocks':appended,'asset_block_map':blockmap,'unused_old_shot_motion_freezes':freeze,'retained_control_manifold_blocks':[x['block'] for x in model['linear_control_manifolds']],'S01_protected_materials':sorted(protected),'S01_preserved_aux_import_still_requires_full_forward_proof':True,'model_quality_accepted':False}
 model['first4_source_generation_activity_view']={'schema':receipt['schema'],'base_model':receipt['base'],'source_scope':list(scope),'unused_controls_retained_frozen':True,'source_UV_not_generated_from_model':True}
 write_json(out/'model_first4.json',model);receipt['model']=_ref(out/'model_first4.json');write_json(out/'model_scope_receipt.json',receipt);return receipt
