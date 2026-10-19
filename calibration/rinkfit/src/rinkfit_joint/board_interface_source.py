"""Native blue-cap/body and yellow-strip/body interfaces on one shared board.

The physical height is a shared fitted fraction of the actual board height.
No source height, per-point depth, new camera, or old parameter reset is used.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,copy,json
import numpy as np
from PIL import Image
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import _sample
from .role_guard import FrameRoleGuard

ROLES={'blue_cap_lower_body_interface':'blue_cap_body_interface','yellow_strip_upper_body_interface':'yellow_strip_body_interface'}

def append_board_interfaces(model,material_map,identity_registry,source_summary,initial_fractions=None):
 result=copy.deepcopy(model);mapping=copy.deepcopy(material_map);registry=copy.deepcopy(identity_registry);fractions=initial_fractions or {'blue_cap_body_interface':.9,'yellow_strip_body_interface':.1};blocks={b['name']:i for i,b in enumerate(result['blocks'])};labels={m['label']:i for i,m in enumerate(result['materials'])};existing={m['id'] for m in registry['materials']};added=[]
 for role in ROLES.values():
  target='board.'+role;semantic='|'.join([target,'board','material_interface',role]);name='board/material_height_fraction/'+role
  if name not in blocks:
   h=float(fractions[role])
   if not 0<h<1:raise ValueError('Initial shared material height must be in the board interior')
   blocks[name]=len(result['blocks']);result['blocks'].append({'name':name,'values':[h],'lower':[0.],'upper':[1.],'scales':[.1],'constant':False,'physical_role':'shared_board_material_height_fraction','initializer_is_physical_measurement':False})
  if target not in labels:
   labels[target]=len(result['materials']);result['materials'].append({'label':target,'paths':[{'id':'joint/'+target,'variant_id':'joint/'+target,'kind':3,'geometry_block':blocks[name],'geometry_offset':0,'boundary_segment':-1,'fixed_height_fraction':float(result['blocks'][blocks[name]]['values'][0]),'closed':True,'physical_point':False,'provenance':json.dumps({'physical_role':role,'surface':'same continuous board course as all existing board/ad materials','shared_height_is_fitted_not_measured':True,'source_summary':source_summary},ensure_ascii=False)}]});added.append({'target':target,'material_index':labels[target],'shared_height_block':blocks[name]})
  mapping['semantic_to_material'][semantic]={'target':target,'material_index':labels[target],'geometry_ready':True};proof=source_summary.get('roles',{}).get(role,{})
  if semantic not in existing:registry['materials'].append({'id':semantic,'source_target':target,'surface':'board','category':'board_edge','measurement':{'operator':'material_interface','expected_operator':'material_interface','authority':'actual_native_colored_material_interface_on_same_continuous_board','conflict':False},'boundary_role':role,'point_correspondence':False,'identity_accepted':bool(proof.get('source_verified_rows',0)>0),'identity_status':'native_two_material_color_transition_and_preserved_continuous_interface_source_identity','shots':proof.get('shots',[]),'rigs':proof.get('rigs',[]),'native_pts':proof.get('native_pts',[]),'source_row_count':proof.get('source_verified_rows',0),'source_evidence':proof,'height_is_shared_unknown_fitted_parameter':True,'unknown_per_point_depth':False})
 result['shared_board_material_interfaces']={'added_materials':added,'existing_blocks_and_material_indices_preserved':True,'source_summary':source_summary};mapping['board_material_interface_extension']=result['shared_board_material_interfaces'];return result,mapping,registry

def audit_board_interfaces(audit_dir,native_dir,role_file,identity_dir,model_path,map_path,outdir):
 audit=Path(audit_dir);native=Path(native_dir);identity=Path(identity_dir);out=Path(outdir)
 if out.exists():raise ValueError('New immutable board interface source generation required')
 rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));candidates=[];by=defaultdict(list)
 for r in rows:
  role=next((new for old,new in ROLES.items() if old in r['source_feature_id']),None)
  if role:candidates.append((r,role));by[(r['shot_id'],r['native_pts'])].append((r,role))
 frames={f['native_pts']:f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(role_file);data_root=next(x for x in native.parents if x.name=='RinkSpace_Data');mask_records={}
 for sid in {r['shot_id'] for r,_ in candidates}:
  nr=int(sid[1:]);base=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=base/'r2/frame_manifest.json' if nr==2 else base/'frame_manifest.json'
  for f in read_json(mp)['frames']:mask_records[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);decisions=[];overlays=[];counts=Counter();role_rows=defaultdict(list)
 for (sid,pts),members in sorted(by.items(),key=lambda x:x[0][1]):
  guard.assert_training_pixels([pts],shot_id=sid);f=frames[pts];png=native/f['native_png']
  if file_sha256(png)!=f['native_png_sha256']:raise ValueError('Native source PNG changed')
  rgb=np.asarray(Image.open(png).convert('RGB'));channels=np.moveaxis(rgb.astype(np.float32),2,0);mr=mask_records[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));ms=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if ms and file_sha256(mp)!=ms:raise ValueError('Current source actor mask changed')
  mask=np.asarray(Image.open(mp).convert('L')).copy();mask[:170,:620]=255;mask[900:,1600:]=255
  for row,role in members:
   m=row['source_measurement_audit'];normal=m.get('normal');q=row.get('current_uv');proof=None;reason=None
   if q is None or normal is None:reason='current_board_material_interface_not_source_supported'
   else:
    q=np.asarray(q);n=np.asarray(normal);n=n/np.linalg.norm(n);positions=q+np.asarray([-6,-4,-2,2,4,6])[:,None]*n;idx=np.rint(positions).astype(int)
    if np.any(idx[:,0]<0) or np.any(idx[:,0]>=1920) or np.any(idx[:,1]<0) or np.any(idx[:,1]>=1080):reason='board_color_interface_sample_outside_native_domain'
    elif np.any(mask[idx[:,1],idx[:,0]]):reason='current_board_color_interface_actor_or_graphics_excluded'
    else:
     values=_sample(channels,positions);side=np.asarray([np.median(values[:3],axis=0),np.median(values[3:],axis=0)])
     score=side[:,2]-.5*(side[:,0]+side[:,1]) if role.startswith('blue') else np.minimum(side[:,0],side[:,1])-side[:,2];contrast=float(abs(score[0]-score[1]));good=bool(max(score)>8. and contrast>6.)
     if good:proof={'actual_native_two_side_RGB':side.tolist(),'actual_sample_uv':positions.tolist(),'material_color_score':score.tolist(),'material_vs_body_contrast':contrast,'role':role,'native_png_sha256':f['native_png_sha256'],'native_mask_sha256':file_sha256(mp),'source_feature_identity':row['source_feature_id'],'world_camera_or_height_used':False,'operator_rebinding':'unspecified and material_interface both execute the identical existing single-material-interface _remeasure branch','source_and_current_uv_unchanged':True}
     else:reason='two_actual_native_material_colors_do_not_certify_this_role'
   valid=proof is not None;decision={'source_row_id':row['source_row_id'],'source_uv':row['source_uv'],'current_uv':row['current_uv'],'shot_id':sid,'native_pts':pts,'role':role,'source_color_identity_verified':valid,'reason':reason,'source_identity_proof':proof,'source_family_id':row['source_family_id'],'solver_eligible':False};decisions.append(decision);counts['native_material_color_identity_verified' if valid else reason]+=1
   if valid:
    target='board.'+role;semantic='|'.join([target,'board','material_interface',role]);overlays.append({'source_row_id':row['source_row_id'],'source_uv':row['source_uv'],'current_uv':row['current_uv'],'target_id':target,'joint_material_id':semantic,'physical_surface':'board','category':'board_edge','measurement_operator_resolved':'material_interface','measurement_definition':{'raw_operator':row.get('measurement_operator'),'operator':'material_interface','expected_operator':'material_interface','authority':'actual_native_colored_material_interface_on_same_continuous_board','conflict':False},'boundary_role':role,'source_measurement_audit':{**m,'material_identity_color_proof':proof},'physical_identity_retained':True,'new_binding_source':'actual_current_two_side_material_colors_and_preserved_continuous_interface_source','source_and_current_uv_unchanged':True,'numeric_statistics_ready':False,'solver_eligible':False});role_rows[role].append(row)
  print(json.dumps({'native_pts':pts,'source_rows_complete':len(decisions),'counts':dict(counts)}),flush=True)
 summary={'schema':'rinkfit.joint9.continuous_board_material_interface_sources.v1','input_source_rows':len(candidates),'counts':dict(counts),'roles':{role:{'source_verified_rows':len(rr),'shots':sorted({r['shot_id'] for r in rr}),'rigs':sorted({r['rig_id'] for r in rr}),'native_pts':sorted({r['native_pts'] for r in rr})} for role,rr in role_rows.items()},'all_source_rows_accounted':len(decisions)==len(candidates),'any_per_point_unknown_depth_added':False,'material_height_measured_or_locked_as_0_or_1':False,'source_audit_rows_sha256':file_sha256(audit/'relocated_rows.jsonl.gz')};write_jsonl(out/'rows_overlay.jsonl.gz',overlays);write_jsonl(out/'source_decisions.jsonl.gz',decisions);write_json(out/'source_summary.json',summary)
 model,mapping,registry=append_board_interfaces(read_json(model_path),read_json(map_path),read_json(identity/'joint_material_registry.json'),summary);write_json(out/'model.json',model);write_json(out/'material_map.json',mapping);write_json(out/'joint_material_registry.json',registry);write_json(out/'manifest.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--native',required=True);p.add_argument('--roles',required=True);p.add_argument('--identity',required=True);p.add_argument('--model',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit_board_interfaces(a.audit,a.native,a.roles,a.identity,a.model,a.material_map,a.out)))
