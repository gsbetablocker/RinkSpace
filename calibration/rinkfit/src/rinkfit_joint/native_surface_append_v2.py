"""Append source-verified native ice and board readings to an existing joint packet.

Old arrays, source IDs, operators, groups and observations remain exact prefixes.
Current RGB evidence and complete physical/nuisance state are separate inputs.
"""
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import json,numpy as np
from .joint_metrics import PacketArrays
from .io_utils import read_json,write_json,file_sha256
from .dense_joint_packet import _ref,_array,_fast_seed_frame,REGISTRY
from .joint_model import as_event_view
from .screen_native_motion_v2 import _camera
from rinkfit_start.native_scene import NativeScene

def append(*,parent_packet,parent_qualification,parent_weights,model,evidence,output):
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);pp=Path(parent_packet);rd=PacketArrays(pp);p=rd.packet;m=read_json(model);ev=read_json(evidence)
 if ev['schema']!='rinkfit.native_surface_source_readout_evidence.v2' or ev['source_model_residual_used'] or not ev['visual_surface_identity_review_completed']:raise ValueError('Missing actual native advertisement evidence')
 for rr in [ev['rows'],ev['frames'],ev['reader'],ev['source_profile_reader'],*ev['review_images']]:
  if file_sha256(rr['file'])!=rr['sha256']:raise ValueError('Changed native ad evidence')
 rows=read_json(ev['rows']['file']);frames=read_json(ev['frames']['file']);n0=int(p['row_count']);count=len(rows);N=n0+count
 if any(r['source_dataset_id']!=ev['source_dataset_id']for r in rows):raise ValueError('Native source dataset identity differs')
 if not count or len({r['source_row_id']for r in rows})!=count or set(r['source_row_id']for r in rows)&set(p['source_row_ids']):raise ValueError('Missing or duplicate source identities')
 for r in rows:
  fr=frames[r['frame_index']]
  if m['materials'][r['material_index']]['label']!=r['material_label'] or not ((r['physical_surface']=='board' and r['material_label'].startswith('Ad')) or (r['physical_surface']=='ice' and r['material_label'].startswith('ice.'))) or r['source_model_residual_used'] or r['original_UV_changed']:raise ValueError('Unverified ad source identity')
  if r['native_pts']!=fr['native_pts'] or r['shot_id']!=fr['shot_id'] or fr['frame_role']!='native_material_repair_training':raise ValueError('Native source exposure/role differs')
 scene=NativeScene(as_event_view(m,'s01'));cameras={sid:_camera(m,sid)for sid in m['shots']};paths=np.zeros(count,'<u4');stations=np.zeros(count,'<f8')
 frameids=defaultdict(list)
 for i,r in enumerate(rows):frameids[r['frame_index']].append(i)
 for fi,ii in frameids.items():
  block=[dict(rows[i],path_index=0)for i in ii];fr=frames[fi];t=(fr['native_pts']-m['shots'][fr['shot_id']]['time_origin_pts'])/12800
  pa,st,_,_=_fast_seed_frame(scene,cameras[fr['shot_id']],t,block);paths[ii]=pa;stations[ii]=st
 suffix=dict(uv=[r['source_uv']for r in rows],native_pts=[r['native_pts']for r in rows],shot_index=[p['shot_ids'].index(r['shot_id'])for r in rows],kind=np.zeros(count),material_index=[r['material_index']for r in rows],path=paths,admission=[r['active']for r in rows],fit_role=[r['active']for r in rows],E=[r['engineering_E']for r in rows],station=stations,row=[r['source_uv'][1]for r in rows],alpha=np.zeros(count),source_sigma=np.ones(count),profile_normal=[r.get('current_native_profile',{}).get('normal') or [0.,0.]for r in rows],query_uv=[r['source_uv']for r in rows],alpha_bounds=np.zeros((count,2)),stroke_pair_index=np.full(count,-1),stroke_source_side=np.full(count,-1))
 if set(suffix)!=set(p['arrays']):raise ValueError('Unsupported source numerical ABI')
 arrays={}
 def save(name,a):
  a=np.asarray(a);a.tofile(out/name);return dict(_ref(out/name),dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
 for name in p['arrays']:
  old=rd.load(name);tail=np.asarray(suffix[name],dtype=old.dtype);a=np.concatenate([old,tail]);assert a[:n0].tobytes()==old.tobytes();arrays[name]=save(name+'.bin',a)
 names=list(dict.fromkeys(r['group_id']for r in rows));name_index={name:i for i,name in enumerate(names)};ids=[r['source_row_id'].encode('utf8')for r in rows];blob=b''.join(ids);offsets=np.r_[0,np.cumsum([len(s)for s in ids])].astype('<u8')
 registry=dict(schema=REGISTRY,row_start=n0,count=count,source_id_utf8=save('source_id_utf8.u8',np.frombuffer(blob,'u1')),source_id_offsets=save('source_id_offsets.u64',offsets),source_family_ids=names,source_family_index=save('source_family_index.u32',np.array([name_index[r['group_id']]for r in rows],'<u4')),source_store_ordinal=save('source_store_ordinal.u64',np.arange(count,dtype='<u8')),source_dataset_id=ev['source_dataset_id'],allowed_paths_are_explicit=True,original_source_selection_IDs_retained=True,source_evidence=_ref(evidence))
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r['group_id']].append(n0+i)
 ng=[dict(id='native_ad_unknown_W:'+name,source_group_id=name,rows=ii,rank=0,W=[],statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=[_ref(evidence)],source_noise_covariance_claimed=False)for name,ii in groups.items()]
 active=sum(r['active']for r in rows);val=dict(schema=p['schema'],dense_parent_packet=_ref(pp),row_count=N,arrays=arrays,native_ad_source_binding=registry,native_ad_source_bindings=p.get('native_ad_source_bindings',[p['native_ad_source_binding']] if 'native_ad_source_binding' in p else [])+[registry],active_rows=int(p['active_rows'])+active,admitted_rows=int(p.get('admitted_rows',p['active_rows']))+active,native_ad_source_append=dict(schema='rinkfit.native_ad_source_append.v1',parent_packet=_ref(pp),row_start=n0,count=count,evidence=_ref(evidence)),dense_parent_append={'groups':ng,'source_manifests':[_ref(evidence),_ref(__file__),_ref(model)],'engineering_pixel_dependency_groups.groups':[dict(id=name,rows=ii)for name,ii in groups.items()],})
 val['allowed_paths']=dict(p.get('allowed_paths',{}),**{str(n0+i):r['allowed_paths']for i,r in enumerate(rows)})
 val['material_binding']=dict(p['material_binding'],model_material_prefix_count=len(m['materials']),materials=[dict(index=i,label=ma['label'],paths=[dict(index=k,id=pa['id'],physical_point=pa.get('physical_point',False))for k,pa in enumerate(ma['paths'])])for i,ma in enumerate(m['materials'])],model_topology_ref=_ref(model))
 write_json(out/'packet.json',val);packetref=_ref(out/'packet.json')
 q=dict(schema='rinkfit.native_surface_source_append_qualification.v2',source_packet=packetref,parent_packet=_ref(pp),parent_qualification=_ref(parent_qualification),parent_rows=n0,raw_rows=N,new_rows=count,active_new_rows=active,source_evidence=_ref(evidence),issuer=_ref(__file__),membership_array_sha256={k:v['sha256']for k,v in arrays.items()},compact_registry=registry,original_prefix_values_unchanged=True,native_profile_normals_preserved=True,source_UV_from_model=False,quality_accepted=False)
 write_json(out/'source_qualification.json',q);qr=_ref(out/'source_qualification.json');w=read_json(parent_weights);wf={};values={};budgets=[]
 for name,dt in [('weights.f64','<f8'),('active.u8','u1'),('information_class.u8','u1')]:
  old=_array(w['files'][name],Path(parent_weights).parent);tail=np.array([float(r['active'])for r in rows],dtype=dt)
  if name=='information_class.u8':tail*=5
  if name=='weights.f64':
   for name2,ii in groups.items():
    ix=np.array(ii)-n0;nn=sum(rows[i]['active']for i in ix)
    if nn:tail[ix]/=nn;budgets.append(dict(id=name2,rows=nn,per_row_weight=1./nn,engineering_budget=1.))
  values[name]=np.concatenate([old,tail]);wf[name]=save(name,values[name])
 retained=float(values['weights.f64'][:n0].sum());added=float(values['weights.f64'][n0:].sum());weights=dict(schema='rinkfit.native_curve_append_engineering_weights.v1',source_packet=packetref,parent_packet=_ref(pp),parent_weights=_ref(parent_weights),issuer=_ref(__file__),dense_source_qualification=qr,parent_rows=n0,raw_rows=N,active_rows=int(np.count_nonzero(values['active.u8'])),new_active_rows=active,component_budgets=budgets,files=wf,retained_parent_budget=retained,new_dense_budget=added,actual_total_engineering_budget=retained+added,source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False)
 write_json(out/'engineering_weights.json',weights);receipt=dict(packet=packetref,qualification=qr,weights=_ref(out/'engineering_weights.json'),parent_rows=n0,new_raw_rows=count,new_active_rows=active,all_original_raw_arrays_bitwise_preserved=True);write_json(out/'receipt.json',receipt);return receipt
