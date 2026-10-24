"""Restore reviewed native ice readings without changing UV, time or weight bits.

The current source streams remain immutable. A new source generation carries the
explicit current material evidence and retains every unresolved reading.
"""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import importlib.util,json,gzip
import numpy as np
from .io_utils import read_json,file_sha256
from .dense_joint_proofs import _ref,_checked,_write
from .dense_joint_packet import _array
from .joint_metrics import PacketArrays
ROOT=Path(__file__).resolve().parents[4]

def recover(*,packet,qualification,weights,original_weights,evidence,base_frames_by_stream,output,stream_prefix,preserved_negative_packet_rows=(),physical_identity_review=None):
 if physical_identity_review is None:raise ValueError('Separate positive physical source-identity review is required for ice recovery')
 from . import native_ice_identity_gate_v1 as identity_gate
 p=Path(packet).resolve();reader=PacketArrays(p);doc=reader.packet;qpath=Path(qualification);wpath=Path(weights);q=read_json(qpath);w=read_json(wpath);ow=read_json(original_weights);e=read_json(evidence);eref=_ref(evidence)
 if q['source_packet']!=_ref(p)or w['source_packet']!=_ref(p):raise ValueError('Current qualification/weight packet differs')
 if e.get('schema')!='rinkfit.native_ice_source_restoration_evidence.v1'or e.get('source_UV_changed')is not False or e.get('source_model_residual_used')is not False:raise ValueError('Native material recovery evidence required')
 for r in [e['reader'],e['per_row_evidence'],e['source_anchors'],e['source_span_matches'],*e['source_files']]:_checked(r,Path(evidence).parent)
 wanted=np.fromfile(_checked(e['restored_packet_rows'],Path(evidence).parent),dtype='<u8').astype(np.int64)
 if len(wanted)!=e['restored_rows']or len(np.unique(wanted))!=len(wanted):raise ValueError('Recovery membership must be unique')
 a={k:reader.load(k)for k in ('admission','fit_role','native_pts','shot_index','material_index','path','uv')};active=(a['admission']==1)&(a['fit_role']==1);n=len(active);n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],p.parent)
 if np.any(wanted<n0)or np.any(wanted>=n)or np.any(active[wanted]):raise ValueError('Recovery must identify inactive original dense readings')
 original={k:_array(ow['files'][k],Path(original_weights).parent)for k in ('weights.f64','active.u8','information_class.u8')}
 # Earlier individual negative reviews retain precedence over bulk-absence repair.
 wanted=wanted[(original['active.u8'][wanted]!=0)&~np.isin(wanted,preserved_negative_packet_rows)]
 expected=set(map(int,wanted));seen=set();proofs={}
 with gzip.open(_checked(e['per_row_evidence'],Path(evidence).parent),'rt',encoding='utf8')as f:
  for line in f:
   r=json.loads(line);i=int(r['row'])
   if i not in expected:continue
   if i in seen or r['source_UV']!=a['uv'][i].tolist()or r['native_pts']!=int(a['native_pts'][i])or r['material_index']!=int(a['material_index'][i])or r['physical_surface']!='ice'or r['adjacent_native_exposure_support']<1 or r['source_span_distance_px']>8:raise ValueError('Recovery evidence differs from original row/material/exposure')
   seen.add(i);proofs[i]=r
 if seen!=expected:raise ValueError('Missing native per-row recovery evidence')
 source_ids=identity_gate.selected_packet_source_ids(doc,p.parent,wanted)
 physical_review_ref=identity_gate.require_physical_recovery_review(physical_identity_review,_ref(p),wanted,doc,a,proofs,source_ids=source_ids)
 bindingref=doc['dense_qualification_binding'];binding=read_json(_checked(bindingref,p.parent));parts=deepcopy(binding['qualified_source_parts']);previous_unknown=_array(binding['withdrawn_source_store_ordinals'],p.parent);old_unknown=set(map(int,previous_unknown));restored_ord=ordinal[wanted-n0].astype('<u8')
 if not set(map(int,restored_ord))<=old_unknown:raise ValueError('Restoration includes a row not currently withheld')
 out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True);commits=[];allchanges=defaultdict(lambda:defaultdict(dict))
 for i,o in zip(wanted,restored_ord):
  part=next(v for v in parts if v['ordinal_start']<=int(o)<v['ordinal_start']+v['row_count']);sid=doc['shot_ids'][int(a['shot_index'][i])];pts=int(a['native_pts'][i]);ident=source_ids[int(i)]
  allchanges[part['name']][sid,pts][ident]=dict(physical_instance_qualification='eligible',source_fit_eligible=True,physical_instance_support_state='native_material_and_temporal_source_confirmed',qualification_reason_code='original_native_ice_material_recovered',qualification_review_id='ice_recovery/'+sid+'/'+str(pts),qualification_evidence=physical_review_ref,source_recovery_original_row_index=int(i))
 spec=importlib.util.spec_from_file_location('rinkspace_public_source_storage',ROOT/'scripts/source_storage.py');public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
 for part in parts:
  changes=allchanges.get(part['name'])
  if not changes:continue
  sub=out/part['name'];sub.mkdir();source=part['reference'];req=dict(schema='rinkspace.source_storage_commit_request.v1',db=source['store'],base_rows=source,base_frames=base_frames_by_stream[source['stream']],new_rows_stream=stream_prefix+'/'+part['name']+'/selected_rows',new_frames_stream=stream_prefix+'/'+part['name']+'/selected_frames',changes=[dict(shot_id=k[0],native_pts=k[1],update=v)for k,v in sorted(changes.items())],source_evidence=[eref,physical_review_ref,_ref(identity_gate.__file__),_ref(__file__)])
  reqref=_write(sub/'request.json',req);published=public.commit(reqref['file'],sub/'source_manifest.json');result=read_json(published['manifest']);transition=result['actual_validation']['actual_transition']
  if transition['removed_selected_rows']or transition['added_selected_rows']or result['source_UV_unchanged']is not True:raise ValueError('Recovery altered original readings')
  commits.append(dict(name=part['name'],parent_reference=source,selected_rows=result['selected_rows'],selected_frames=result['selected_frames'],source_generation=_ref(published['manifest']),restored_rows=sum(len(v)for v in changes.values())))
  part['reference']=result['selected_rows'];print('source restored',part['name'],commits[-1]['restored_rows'],flush=True)
 generation=_write(out/'source_generation.json',dict(schema='rinkfit.multi_part_source_recovery_generation.v1',complete=True,source_commits=commits,evidence=eref,independent_physical_identity_review=physical_review_ref,physical_identity_consumer=_ref(identity_gate.__file__),issuer=_ref(__file__),source_UV_ID_PTS_unchanged=True))
 remaining=np.array(sorted(old_unknown-set(map(int,restored_ord))),dtype='<u8');remaining.tofile(out/'withdrawn_source_store_ordinals.u64');restored_ord.sort();restored_ord.tofile(out/'restored_source_store_ordinals.u64');wanted.astype('<u8').tofile(out/'restored_packet_rows.u64')
 def arrayref(name,dtype,count):return dict(_ref(out/name),dtype=dtype,shape=[count])
 unknownref=arrayref('withdrawn_source_store_ordinals.u64','<u8',len(remaining));restoreref=arrayref('restored_source_store_ordinals.u64','<u8',len(wanted));rowref=arrayref('restored_packet_rows.u64','<u8',len(wanted));newactive=int(active.sum())+len(wanted);newdense=int(binding['total_active_rows'])+len(wanted)
 nextbinding=deepcopy(binding);nextbinding.update(parent_packet=_ref(p),source_observation_generation=generation,qualified_source_parts=parts,withdrawn_source_store_ordinals=unknownref,total_active_rows=newdense,total_unknown_rows=len(remaining),parent_qualification_binding=bindingref,restored_source_store_ordinals=restoreref,restored_packet_rows=rowref,restoration_evidence=eref,independent_physical_identity_review=physical_review_ref,issuer=_ref(__file__))
 b_ref=_write(out/'source_binding.json',nextbinding)
 delta=dict(schema='rinkfit.dense_source_requalification_delta.v1',parent_source_qualification=_ref(qpath),source_packet=_ref(p),actual_review_consumer=_ref(__file__),source_observation_generation=generation,source_review_evidence=eref,independent_physical_identity_review=physical_review_ref,physical_identity_consumer=_ref(identity_gate.__file__),withdrawn_source_store_ordinals=unknownref,restored_source_store_ordinals=restoreref,restored_packet_rows=rowref,from_qualification='unknown',to_qualification='eligible',source_semantics_verified=True,qualification_granted=True,selected_by_model_residual=False,cumulative_withdraw_only=False,source_qualification_binding=b_ref,restored_rows=len(wanted),original_source_UV_ID_PTS_material_path_preserved=True)
 d_ref=_write(out/'qualification_delta.json',delta);arrays=deepcopy(doc['arrays'])
 for key in ('admission','fit_role'):
  value=a[key].copy();value[wanted]=1;value.tofile(out/(key+'.u8'));arrays[key]=dict(arrayref(key+'.u8',value.dtype.str,len(value)),layout='C',byte_order='little')
 child=dict(schema=doc['schema'],dense_parent_packet=_ref(p),arrays=arrays,active_rows=newactive,dense_qualification_binding=b_ref,source_qualification_delta=d_ref,source_requalification=d_ref,dense_parent_append=dict(source_manifests=[b_ref,d_ref,eref,generation,_ref(__file__)]));packetref=_write(out/'packet.json',child)
 q=deepcopy(q);q.update(source_packet=packetref,issuer=_ref(__file__),qualification_delta=d_ref,active_new_rows=newdense,unknown_new_rows=len(remaining),source_qualification_binding=b_ref,source_geometry_qualification_inherited_exact=True,source_accuracy_accepted=False,membership_array_sha256={k:v['sha256']for k,v in arrays.items()},actual_input_references=q['actual_input_references']+[eref,physical_review_ref,_ref(identity_gate.__file__),_ref(__file__),d_ref,b_ref,generation,rowref,restoreref]);qref=_write(out/'source_qualification.json',q)
 current={k:_array(w['files'][k],wpath.parent).copy()for k in original};wf={}
 for name,v in current.items():
  v[wanted]=original[name][wanted];v.tofile(out/name);wf[name]=arrayref(name,v.dtype.str,len(v))
 if np.any(current['weights.f64'][wanted]<=0):raise ValueError('Original restored source weight is not positive')
 budgets=[];summary=defaultdict(lambda:dict(rows=0,exposures=set(),groups=0,budget=0.));covered=np.zeros(n-n0,np.uint8)
 for g in doc['engineering_pixel_dependency_groups']['groups']:
  ii=np.array(g['rows'],np.int64)
  if not len(ii)or ii.max()<n0:continue
  if ii.min()<n0:raise ValueError('Dense source group crosses immutable prefix')
  ii=ii[current['active.u8'][ii]!=0]
  if not len(ii):continue
  if np.any(covered[ii-n0]):raise ValueError('Duplicated engineering source budget')
  covered[ii-n0]=1;v=current['weights.f64'][ii];per=float(v[0])
  if per<=0 or np.any(v!=per):raise ValueError('Original source-component absolute weight differs')
  amount=float(v.sum());budgets.append(dict(id=g['id'],rows=len(ii),per_row_weight=per,engineering_budget=amount));sid=doc['shot_ids'][int(a['shot_index'][ii[0]])];b=summary[sid];b['rows']+=len(ii);b['exposures'].add(int(a['native_pts'][ii[0]]));b['groups']+=1;b['budget']+=amount
 if not np.array_equal(covered.astype(bool),current['active.u8'][n0:]!=0):raise ValueError('Unaccounted restored source budget')
 by=[]
 for sid,b in sorted(summary.items()):
  old=next(x for x in ow['per_shot_new_budgets']if x['shot_id']==sid);by.append(dict(old,new_rows=b['rows'],covered_native_exposures=len(b['exposures']),named_material_exposure_components=b['groups'],engineering_budget=b['budget'],absolute_prior_per_row_weights_preserved=True))
 retained=float(current['weights.f64'][:n0].sum());new=float(current['weights.f64'][n0:].sum());w.update(source_packet=packetref,issuer=_ref(__file__),dense_source_qualification=qref,active_rows=newactive,new_active_rows=newdense,files=wf,component_budgets=budgets,retained_parent_budget=retained,new_dense_budget=new,actual_total_engineering_budget=retained+new,per_shot_new_budgets=by,source_qualification_delta=d_ref,parent_dense_weights=_ref(wpath),restored_original_dense_weights=_ref(original_weights),restored_source_weight_bits_exact=True,unresolved_source_rows_have_zero_budget=True)
 wref=_write(out/'engineering_weights.json',w);receipt=dict(schema='rinkfit.original_native_ice_recovery_packet.v1',packet=packetref,source_qualification=qref,engineering_weights=wref,source_generation=generation,source_binding=b_ref,qualification_delta=d_ref,independent_physical_identity_review=physical_review_ref,restored_packet_rows=rowref,restored_rows=len(wanted),raw_rows=n,active_rows=newactive,original_source_UV_ID_nativePTS_material_path_ordinals_preserved=True,unaffected_weights_and_restored_original_weights_bitwise_equal=True,physical_solver_started=False,quality_accepted=False)
 _write(out/'receipt.json',receipt);return receipt
