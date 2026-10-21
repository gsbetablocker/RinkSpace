"""Independently recompute source attachment identities and their pinned evidence."""
from pathlib import Path
import hashlib,json,math,copy
import numpy as np
from .io_utils import file_sha256
from .r_inner_prefix_v1 import checked,array

_CANONICAL_DIGEST_CACHE={}

def require(ok,why):
 if not ok:raise ValueError('source mathematics identity: '+why)
def compact(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest_text(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def load(ref,base,seen):return json.loads(checked(ref,base,seen).read_text(encoding='utf8'))
def refs(v,base,seen):
 if isinstance(v,dict):
  if 'sha256'in v and ('file'in v or 'path'in v):checked(v,base,seen)
  for x in v.values():refs(x,base,seen)
 elif isinstance(v,list):
  for x in v:refs(x,base,seen)

def visible_identity(config,model,base,seen):
 require(set(config)=={'schema','contract','source_replay','source_qualification'}and config['schema']=='rinkfit.visible_ice_source_constraint_consumer.v1','visible config/schema')
 refs(config,base,seen);c=load(config['contract'],base,seen);r=load(config['source_replay'],base,seen);q=load(config['source_qualification'],base,seen)
 require(c['schema']=='rinkfit.visible_ice_sensor_ray_free_space_contract.v1'and r['schema']=='rinkfit.visible_ice_ray_source_replay.v1'and q['schema']=='rinkfit.visible_ice_ray_source_qualification.v1','visible source schemas')
 require(q['ray_feasibility_consumer_eligible']and q['source_geometry_point_fit_rows']==0 and q['statistical_rank_added']==0,'visible support promoted to point statistics')
 require(all(r[k]for k in ('passed','source_UV_unchanged','all_support_patches_exclusion_clear','all_source_uvs_beyond_entire_conservative_cap_profile_window'))and not r['cap_inner_physical_edge_mean_qualified'],'visible replay incomplete')
 for x in q['producer_references']+q['reader_module_refs']:checked(x,base,seen)
 independent=load(q['independent_source_replay'],base,seen)
 require(all(independent[k]for k in ('passed','source_UV_and_native_PTS_unchanged','original_three_frozen_files_unchanged'))and not independent['top757_point_qualification_revived']and independent['source_support_count']==q['source_support_count'],'independent visible replay')
 for k in ('programme_scope','frame_roles','actual_source_replay_script'):checked(independent[k],base,seen)
 for k in ('source_rows','cap_uncertainty'):checked(c[k],base,seen)
 generation=load(c['source_generation'],base,seen);video=generation['source_video']['sha256'];require(model['source_clock']['source_video_sha256']==video,'visible model/video clock')
 require(q['contract_sha256']==config['contract']['sha256']and q['source_rows_sha256']==c['source_rows']['sha256']and q['source_replay_sha256']==config['source_replay']['sha256'],'visible qualification pins')
 rows=[json.loads(v)for v in checked(c['source_rows'],base,seen).read_text(encoding='utf8').splitlines()if v];ids=[];pts=set()
 for x in rows:
  require(x['schema']=='rinkfit.visible_static_ice_sensor_ray_source_support.v1','visible row schema');rid=x['source_ray_id'];ids.append(rid);uv=x['source_uv'];t=x['native_pts'];pts.add(t)
  require(x['actual_pts']==t and t%256==0 and x['time_base']==[1,12800]and x['source_video_sha256']==video and r['original_video_redecoded_at_native_pts']==t,'visible original clock')
  require(len(uv)==2 and all(math.isfinite(v)for v in uv)and 0<=uv[0]<=1919 and 0<=uv[1]<=1079,'visible source UV')
  require(x['surface_class']=='ice'and all(x[k]for k in ('physical_ground_support','visible_ice_body_and_not_actor_or_reflection','physical_visible_ice_ray_support_qualified','current_exclusion_clear_9x9','world_point_is_free_ray_ice_intersection_not_material_station','exact_source_row_time_required')),'visible material support')
  require(x['mask_nonzero_means']=='excluded'and x['minimum_x_clearance_from_entire_cap_profile_window_px']>0,'visible exclusion/cap support')
  require(x['world_XYZ']is None and not any(x[k]for k in ('point_correspondence','known_world_corner','world_depth_allocated','source_geometry_point_fit_qualified','source_covariance_issued','statistical_objective_eligible','source_UV_from_model','old_B1_station_used_to_define_ray','source_rim_or_board_height_measured')),'forbidden visible point/depth/world substitution')
  for k in ('native_image','current_exclusion_mask','cap_uncertainty'):checked(x[k],base,seen)
  require(x['current_decoded_native_BGR_sha256']==r['current_RGB_hash'],'visible RGB replay')
  patch=np.asarray(x['native_BGR_support_patch']);require(patch.shape==(9,9,3)and np.all((patch>=0)&(patch<=255))and np.all(patch==patch.astype(np.uint8)),'visible BGR patch')
  require(hashlib.sha256(patch.astype(np.uint8).tobytes()).hexdigest()==x['native_BGR_support_patch_sha256'],'visible patch digest')
 require(len(rows)==4==c['source_support_count']==q['source_support_count']==r['source_rows']and len(pts)==c['source_exposures']==1 and len(set(ids))==4 and ids==q['qualified_source_ray_ids'],'visible ordered inventory')
 return digest_text(compact(config)),dict(visible_ice_source_count='4',visible_ice_constraint_count='16')

def branch_identity(binding,packet,model,base,seen):
 require(set(binding)=={'registry','source_gate','source_freeze','consumer_module_ref'}and packet.get('source_query_branch_registry')==binding,'branch binding/top-level policy')
 refs(binding,base,seen);reg=load(binding['registry'],base,seen);gate=load(binding['source_gate'],base,seen);freeze=load(binding['source_freeze'],base,seen)
 require(reg['schema']=='rinkfit.C_source_branch_registry.v2'and gate['schema']=='rinkfit.C_source_branch_overlay_gate.v2'and freeze['schema']=='rinkfit.C_source_navigation_complete_projection_freeze.v3','branch registry/gate/freeze schemas')
 for d in (reg,gate,freeze):refs(d,base,seen)
 require(gate['registry']['sha256']==binding['registry']['sha256']==freeze['registry']['sha256']and gate['accepted_for_source_identity_domain']and gate['full_source_rows_retained']==6328 and not gate['physical_capture_complete'],'branch source gate')
 neg=gate['negative_cases'];require(all(neg[k]==6328 for k in ('opposite_flank_measured_alpha_rejected','original_ROI_lower_rejected','original_ROI_upper_rejected'))and neg['wrong_source_identity_rejected'],'branch negative closure')
 policy=load(reg['source_branch_policy'],base,seen);refs(policy,base,seen)
 require(policy['raw_rows']==6328 and policy['source_q50_replay_max_abs_diff']<1e-9 and policy['frozen_source_rows']['sha256']==reg['frozen_source']['sha256']and not policy['model_or_screen_geometry_used_in_policy']and not policy['source_UV_ID_or_packet_modified'],'branch source policy')
 generation=load(policy['source_generation'],base,seen);refs(generation,base,seen);qualification=load(policy['parent_qualification'],base,seen);refs(qualification,base,seen)
 require(generation['source_video']['sha256']==model['source_clock']['source_video_sha256']and generation['source_video']['time_base']=='1/12800'and generation['candidates']['sha256']==policy['frozen_source_profiles']['sha256']and qualification['source_rows_qualified']==3164 and qualification['source_generation']['sha256']==policy['source_generation']['sha256']and not qualification['source_UV_modified'],'branch native source provenance')
 for k in freeze['required_exact_metadata_keys']:
  require((k in model)==freeze['metadata_presence'][k]and(k not in model or model[k]==freeze['metadata_values'][k]),'branch projection metadata '+k)
 old=load(reg['base_model'],base,seen);nb=len(old['blocks']);require(len(model['blocks'])in (nb,nb+13),'branch exact physical suffix')
 for a,b in zip(model['blocks'][:nb],old['blocks']):
  for k in ('name','lower','upper','scales'):require(a[k]==b[k],'branch physical block '+k)
  require(len(a['values'])==len(b['values']),'branch block shape')
 if len(model['blocks'])>nb:
  for a,b in zip(model['blocks'][nb:],freeze['allowed_suffix_blocks']):
   require(all(a[k]==b[k]for k in ('name','lower','upper','scales'))and len(a['values'])==1 and math.isfinite(a['values'][0])and a['values'][0]>0,'branch V9 suffix block')
 suffix=len(model['blocks'])>nb
 require(model['measurement_widths']==freeze['base_measurement_widths']+(freeze['allowed_suffix_measurement_widths']if suffix else[])and model['stroke_pair_model_bindings']==freeze['base_stroke_pair_model_bindings']+(freeze['allowed_suffix_stroke_pair_model_bindings']if suffix else[]),'branch pair/width metadata')
 require(model['materials']==old['materials'],'branch finite material carrier projection')
 n=packet['row_count'];require(n>=916340 and len(reg['alpha_domains'])==6328,'branch row count');qs={q['row_index']:q for q in packet['queries']if 910012<=q['row_index']<916340};require(len(qs)==6328,'branch full original Query ledger')
 arrays={k:array(packet,k,base,seen)for k in ('uv','native_pts','kind','material_index','E')};pairs={v['pair_id']:v for v in model['stroke_pair_model_bindings']}
 overlay=reg['numerical_seed_overlay']['arrays'];aa={k:np.memmap(checked(v,base,seen),dtype=v['dtype'],mode='r',offset=v.get('offset_bytes',0),shape=tuple(v['shape']))for k,v in overlay.items()}
 for j,d in enumerate(reg['alpha_domains']):
  i=910012+j;lo,hi=d['allowed_alpha'];a=d['source_measured_alpha'];q=qs[i]
  require(d['row_index']==i and d['status']=='source_identity_branch_qualified'and lo<hi and lo<=a<=hi and aa['alpha'][j]==a,'branch own-flank domain')
  require(q==d['original_full_Query']and q['source_acquisition_window_only']and q['source_profile_ref']['sha256']==policy['frozen_source_profiles']['sha256'],'branch original raw Query')
  require(packet['source_row_ids'][i]==d['source_row_id']and packet['source_dataset_ids'][i]==d['source_dataset_id']and packet['source_family_ids'][i]==d['source_family_id']and int(arrays['native_pts'][i])==d['native_pts']and arrays['uv'][i].tolist()==d['source_uv']and int(arrays['kind'][i])==d['kind']and int(arrays['material_index'][i])==d['material_index']and float(arrays['E'][i])==d['original_E'],'branch actual immutable source row')
  require(packet['stroke_pairs'][d['pair_index']]==d['original_packet_pair']and pairs[d['pair_id']]==d['original_model_pair_binding']and model['materials'][d['material_index']]==d['material_projection_definition']and model['blocks'][d['width_block']]['name']==d['width_block_name']and model['blocks'][d['width_block']]['values'][0]>0,'branch physical pair binding')
  require(math.isfinite(float(aa['row'][j]))and 0<=aa['row'][j]<=1079,'branch source RS seed')
  if j%2:require(reg['alpha_domains'][j-1]['allowed_alpha'][1]<=lo,'opposite flank overlap')
 return binding['registry']['sha256'],dict(source_query_branch_count='6328')

def validate_source_math_bindings(bindings,packet,model_path,base,*,raw_packet_sha256,verified_dependency_cache=None):
 seen=set()if verified_dependency_cache is None else verified_dependency_cache;model=json.loads(Path(model_path).read_text(encoding='utf8'))if not isinstance(model_path,dict)else model_path
 require(isinstance(bindings,list)and 1<=len(bindings)<=2,'missing ordered mathematics attachments')
 if 'source_math_source_authority'in packet:
  from .source_enriched_current_v1 import source_view
  from .accepted_aux_prefix import _digest
  a=packet['source_math_source_authority'];cp=checked(a['canonical_source_packet'],base,seen);enriched=a['numerical_state_kind']=='source_enriched_complete_primal';key=(str(cp),a['canonical_source_packet']['sha256'],enriched)
  if key not in _CANONICAL_DIGEST_CACHE:
   canonical=json.loads(cp.read_text(encoding='utf8'));_CANONICAL_DIGEST_CACHE[key]=_digest(source_view(canonical,enriched=enriched))
  require(_digest(source_view(packet,enriched=enriched))==_CANONICAL_DIGEST_CACHE[key],'canonical all non-auxiliary source fields/float bits changed')
 identity=raw_packet_sha256;last=-1;discrete={};reports=[]
 for b in bindings:
  kind=b.get('kind');position={'visible_ice_source_constraints':0,'source_query_branches':1}.get(kind,-1);require(position>last and position>=0,'unknown, duplicate or unordered attachment');last=position
  if position==0:
   require(set(b)=={'kind','consumer_config'},'visible attachment fields');part,extra=visible_identity(b['consumer_config'],model,base,seen);extra['visible_ice_source_identity']=part
  else:
   require(set(b)=={'kind','binding'},'branch attachment fields');part,extra=branch_identity(b['binding'],packet,model,base,seen);extra['source_query_branch_identity']=part
  identity=digest_text(identity+'|'+kind+'|'+part);discrete.update(extra);reports.append(dict(kind=kind,attachment_identity=part,resulting_source_id=identity))
 require(('source_query_branch_identity'in discrete)==('source_query_branch_registry'in packet),'branch outside mathematics attachments')
 return dict(mathematical_source_id=identity,source_packet_sha256=raw_packet_sha256,discrete=discrete,attachments=reports,verified_dependency_count=len(seen))
