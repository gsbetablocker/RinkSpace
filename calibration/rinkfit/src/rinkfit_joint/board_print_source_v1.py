"""Install actual printed-board readouts at a new immutable source boundary.

No image point comes from a homography. Protected observations remain raw only.
The complete numerical CP is deliberately not inherited by a longer source packet.
"""
from pathlib import Path
import json,gzip,struct,collections,hashlib
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .board_print_append_v1 import checked,require,ref

def jwrite(path,value):
 with gzip.open(path,'wt',encoding='utf8')as f:json.dump(value,f,separators=(',',':'),allow_nan=False)
def lines(path,rows):
 with gzip.open(path,'wt',encoding='utf8')as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def verify_generation_ref(rr,seen):return read_json(checked(rr,seen))
def build_bundle(old_generation,new_generation,output):
 out=Path(output);require(not out.exists(),'Immutable printed source bundle already exists');seen=set();old=read_json(old_generation);new=read_json(new_generation);require(old['complete_frame_denominator']==94 if 'complete_frame_denominator'in old else len(old['closed_frames'])==94,'Wrong old frame denominator');require(new['complete']and new['frames']==94,'Incomplete actual printed source generation')
 om={verify_generation_ref(r,seen)['native_pts']:(verify_generation_ref(r,seen),r)for r in old['closed_frames']};nm={verify_generation_ref(r,seen)['native_pts']:(verify_generation_ref(r,seen),r)for r in new['closed_frames']};require(set(om)==set(nm),'Actual printed source frame sets differ');out.mkdir(parents=True);(out/'frames').mkdir();allraw=[];allgeo=[];queries=[];original=[];frames=[];counts=collections.Counter();status=collections.Counter();cache={};profilecache={}
 oldrecipe=verify_generation_ref(old['recipe'],seen);newrecipe=verify_generation_ref(new['recipe'],seen)
 for recipe in(oldrecipe,newrecipe):
  for r in recipe['refs']:checked(r,seen)
 for pts in sorted(om):
  m,mref=om[pts];n,nref=nm[pts];parent=verify_generation_ref(m['parent'],seen);require(m['native_receipt']==n['native_receipt']==parent['native_receipt']and m['current_mask']==n['current_mask']==parent['current_mask'],'Actual source clock/RGB/mask mismatch');protected=bool(m['protected_evaluation_only']);require(protected==n['protected_evaluation_only'],'Protected source role changed');checked(m['current_mask'],seen)
  if 'source_journal'in parent:
   jp=checked(parent['source_journal'],seen)
   if str(jp)not in cache:cache[str(jp)]=list(read_jsonl(jp))
   raw=[r for r in cache[str(jp)]if int(r['native_pts'])==pts]
  else:raw=list(read_jsonl(checked(parent['files']['source_rows.jsonl.gz'],seen)))
  members={r['source_row_id']:r for r in read_jsonl(checked(m['files']['membership.jsonl.gz'],seen))};physical={r['source_row_id']:r for r in read_jsonl(checked(m['files']['physical_role.jsonl.gz'],seen))};require(len(raw)==m['raw_rows']==len(members)==len(physical),'Old source full denominator changed');fresh=list(read_jsonl(checked(n['source_rows'],seen)));require(len(fresh)==n['new_current_native_readouts'],'New source full denominator changed');checked(n['physical_qualification'],seen);checked(n['material_query_audit'],seen);checked(n['current_source_context'],seen)
  f=out/'frames'/str(pts);f.mkdir();membership=[];frame_refs=[mref,nref,m['recipe'],n['recipe'],m['current_mask'],n['physical_qualification'],n['material_query_audit'],n['current_source_context']];inline=[];inline_queries=[];frame_count=collections.Counter()
  entries=[(r,members[r['source_row_id']],physical[r['source_row_id']],members[r['source_row_id']]['frozen_original_operator'],False)for r in raw]+[(r,r,r['physical_role_review'],r['original_operator_review'],True)for r in fresh]
  for r,b,phy,op,isnew in entries:
   rid=r['source_row_id'];h=object_sha256(r);mu=r['current_uv'];uvhash=struct.pack('<2d',*mu).hex();profile=r['source_measurement_audit']['profile'];cont=r['source_measurement_audit']['continuous_cubic_material_profile'];off=np.asarray(profile['normal_offsets'],float);normal=np.asarray(r['normal'],float);seed=np.asarray(r['navigation_seed_uv'],float);alpha=float(cont['normal_offset_px']);require(off.shape==(49,)and np.all(np.diff(off)>0),'Original49 input missing');require(float(b['actual_original_RGB_replay_max_abs_px'])==0.,'Actual native RGB optical mean replay failed')
   if not isnew:require(b['source_record_sha256']==h and b['original_UV_binary64_le']==uvhash,'Old source fullrecord/UV changed')
   else:require(r['original_UV_binary64_le']==uvhash and r['current_native_receipt']==n['native_receipt']and r['current_mask']==n['current_mask'],'New source UV/RGB/mask changed')
   require(np.max(abs(seed+alpha*normal-np.asarray(mu)))<=1e-9,'Source optical mean not original49 operator output');ir=profile.get('intensity_reference');values=None
   if ir:
    pp=checked(ir,seen)
    if str(pp)not in profilecache:profilecache[str(pp)]=np.load(pp)
    values=np.asarray(profilecache[str(pp)][ir['array']][int(ir['row_index'])],dtype='<f8')
   else:values=np.asarray(profile['intensity'],dtype='<f8');ir=dict(file=str(f/'inline_profiles.npz'),sha256='pending',array='actual_signed_material_profile',row_index=len(inline));inline.append(values);inline_queries.append(len(queries))
   require(values.shape==(49,)and np.isfinite(values).all(),'Missing actual source intensity values');ophash=op.get('actual_profile_49_values_sha256');require(ophash is None or ophash==object_sha256(values.tolist()),'Stored original49 value hash mismatch')
   ok=b['combined_status']=='pass';require(not ok or(phy['status']=='pass'and op['status']=='operator_pass'),'Source qualification bypassed actual material/operator');conflict=isnew and r['source_measurement_audit']['source_RGB_material_evidence']['actual_local_query_exclusion_conflict'];require(not(ok and conflict),'Masked/exclusion-conflicting source admitted');eligible=ok and not protected;combined='evaluation_only_material_pass'if ok and protected else b['combined_status'];mi=348 if r['material_role']=='board.top'else 349;require(r['material_role']in('board.top','board.contact'),'Unknown board material');family='printed_body/'+r['shot_id']+'/'+str(pts)+'/'+r['material_role'];semantic=r['material_role']+'|board|'+('board_upper_material_edge|top'if mi==348 else'ice_board_interface|contact')
   item=dict(source_row_id=rid,source_family_id=family,shot_id=r['shot_id'],native_pts=pts,source_uv=mu,current_uv=mu,normal=r['normal'],joint_material_id=semantic,joint_material_index=mi,point_correspondence=False,source_geometry_eligible=eligible,solver_eligible_for_statistical_objective=False,statistical_objective_eligible=False,numeric_statistics_ready=False,solver_eligible=False,original_source_record_sha256=h,raw_UV_binary64_le=uvhash,native_video_decode_receipt=m['native_receipt'],current_mask=m['current_mask'],source_admitted=eligible,source_material_role_requalification=phy,measurement_contract_certificate=dict(actual_RGB_material_identity_verified=eligible,operator_selected_correct_material_transition=eligible,operator_qualification_status=op['status'].replace('operator_',''),source_evidence=frame_refs,protected_original_role=m['original_frame_role'],source_noise_W_not_supplied=True),source_journal=None,current_role_operator_membership_ref=None)
   # Literal source journal is never rewritten to install new metadata.
   original.append(r);allraw.append(item)
   if eligible:allgeo.append(item)
   q=dict(source_row_id=rid,exact_original_source_record_sha256=h,source_uv=mu,seed_uv=seed.tolist(),normal=r['normal'],normal_offsets=off.tolist(),channel=profile['channel'],polarity=profile['polarity'],operator_version=cont['operator_version'],intensity_reference=ir);queries.append(q)
   membership.append(dict(source_row_id=rid,current_UV_binary64_le=uvhash,source_record_sha256=h,current_measurement_sha256=phy['current_measurement_sha256'],new_status=combined,physical_role_review=phy,original_operator_review=op,protected_evaluation_only=protected,original_frame_role=m['original_frame_role'],source_geometry_eligible=eligible,actual_source_mean_replay_error_px=0.));counts['raw_rows']+=1;counts['source_geometry_qualified_rows']+=int(eligible);status[combined]+=1;frame_count[combined]+=1
  if inline:
   np.savez_compressed(f/'inline_profiles.npz',actual_signed_material_profile=np.asarray(inline));iref=ref(f/'inline_profiles.npz')
   for qi in inline_queries:queries[qi]['intensity_reference'].update(iref)
  jwrite(f/'membership.json.gz',membership);write_json(f/'input_audit.json',dict(old_frame=mref,new_actual_source_frame=nref,counts=dict(frame_count),protected_evaluation_only=protected,old_means_preserved=True,new_means_from_actual_RGB_profiles_not_H=True))
  fm=dict(schema='rinkfit.current_printed_board_source_export_frame.v1',native_pts=pts,raw_source_UV_Query_W_or_solver_packet_unchanged=True,max_actual_RGB_cubic_UV_replay_error_px=0.,native_receipt=m['native_receipt'],source_frame_proof=nref,source_journal=None,current_mask=m['current_mask'],recipe=None,files={'membership.json.gz':ref(f/'membership.json.gz'),'input_audit.json':ref(f/'input_audit.json')},original_frame_role=m['original_frame_role'],protected_evaluation_only=protected)
  frames.append((f,fm))
 journal=out/'original_source_journal.jsonl.gz';lines(journal,original);jref=ref(journal);recipe=dict(schema='rinkfit.actual_printed_board_source_export_recipe.v1',old_generation=ref(old_generation),new_actual_generation=ref(new_generation),actual_input_dependencies=[dict(file=k[0],sha256=k[1])for k in sorted(seen)],consumer=ref(__file__),protected_not_fitted=True,source_noise_W_not_supplied=True);write_json(out/'recipe.json',recipe);rref=ref(out/'recipe.json');frame_refs=[]
 for f,fm in frames:fm['source_journal']=jref;fm['recipe']=rref;write_json(f/'manifest.json',fm);frame_refs.append(ref(f/'manifest.json'))
 fby={int(read_json(x['file'])['native_pts']):x for x in frame_refs}
 for r in allraw:r['source_journal']=jref;r['current_role_operator_membership_ref']=fby[r['native_pts']]
 lines(out/'raw_ledger.jsonl.gz',allraw);lines(out/'geometry_rows.jsonl.gz',allgeo);lines(out/'frozen_queries.jsonl.gz',queries);write_json(out/'chunk.json',dict(original_source_journal=jref,raw_rows=len(allraw),source_geometry_rows=len(allgeo)))
 bundle=dict(schema='rinkfit.board_geometry_source_bundle.v1',source_geometry_only=True,requirements=dict(source_noise_W_not_supplied=True),code_refs=[ref(__file__),ref(Path(__file__).with_name('board_print_append_v1.py')),ref(Path(__file__).with_name('board_print_current_v1.py'))],current_exact_qualification=dict(recipe=rref),counts=dict(counts),status_counts=dict(status),source_parts=[dict(chunk=ref(out/'chunk.json'),raw_ledger_ref=ref(out/'raw_ledger.jsonl.gz'),geometry_rows_ref=ref(out/'geometry_rows.jsonl.gz'),frozen_queries_ref=ref(out/'frozen_queries.jsonl.gz'),current_exact_qualifications=frame_refs)],all94_native_frames_retained=True,protected_original_roles_unchanged=True,old_source_generation_full_UV_and49_preserved=True,new_means_actual_current_RGB_only=True,source_noise_W_or_precision_issued=False,current_live_packet_modified=False)
 write_json(out/'bundle.json',bundle);return bundle
