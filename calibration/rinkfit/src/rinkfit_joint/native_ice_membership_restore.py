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

def recover(*,packet,qualification,weights,original_weights,evidence,base_frames_by_stream,output,stream_prefix,preserved_negative_packet_rows=()):
 p=Path(packet).resolve();reader=PacketArrays(p);doc=reader.packet;qpath=Path(qualification);wpath=Path(weights);q=read_json(qpath);w=read_json(wpath);ow=read_json(original_weights);e=read_json(evidence);eref=_ref(evidence)
 if q['source_packet']!=_ref(p)or w['source_packet']!=_ref(p):raise ValueError('Current qualification/weight packet differs')
 if e.get('schema')!='rinkfit.native_ice_source_restoration_evidence.v1'or e.get('source_UV_changed')is not False or e.get('source_model_residual_used')is not False:raise ValueError('Native material recovery evidence required')
 for r in [e['reader'],e['per_row_evidence'],e['source_anchors'],e['source_span_matches'],*e['source_files']]:_checked(r,Path(evidence).parent)
 wanted=np.fromfile(_checked(e['restored_packet_rows'],Path(evidence).parent),dtype='<u8').astype(np.int64)
 if len(wanted)!=e['restored_rows']or len(np.unique(wanted))!=len(wanted):raise ValueError('Recovery membership must be unique')
 a={k:reader.load(k)for k in ('admission','fit_role','native_pts','shot_index','material_index','uv')};active=(a['admission']==1)&(a['fit_role']==1);n=len(active);n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],p.parent)
 if np.any(wanted<n0)or np.any(wanted>=n0+len(ordinal))or np.any(active[wanted]):raise ValueError('Recovery must identify inactive original dense readings')
 original={k:_array(ow['files'][k],Path(original_weights).parent)for k in ('weights.f64','active.u8','information_class.u8')}
 # Earlier individual negative reviews retain precedence over bulk-absence repair.
 negatives=set(preserved_negative_packet_rows);ancestor=q;ancestor_base=qpath.parent
 while True:
  if ancestor.get('revision'):
   revision=read_json(_checked(ancestor['revision'],ancestor_base));negatives.update(revision.get('withheld_rows',()))
  if not ancestor.get('parent_qualification'):break
  ancestor_path=_checked(ancestor['parent_qualification'],ancestor_base);ancestor=read_json(ancestor_path);ancestor_base=ancestor_path.parent
 preserved_negative_packet_rows=sorted(negatives)
 wanted=wanted[(original['active.u8'][wanted]!=0)&~np.isin(wanted,preserved_negative_packet_rows)]
 expected=set(map(int,wanted));seen=set();proofs={}
 with gzip.open(_checked(e['per_row_evidence'],Path(evidence).parent),'rt',encoding='utf8')as f:
  for line in f:
   r=json.loads(line);i=int(r['row'])
   if i not in expected:continue
   if i in seen or r['source_UV']!=a['uv'][i].tolist()or r['native_pts']!=int(a['native_pts'][i])or r['material_index']!=int(a['material_index'][i])or r['physical_surface']!='ice'or r['adjacent_native_exposure_support']<1 or r['source_span_distance_px']>8:raise ValueError('Recovery evidence differs from original row/material/exposure')
   seen.add(i);proofs[i]=r
 if seen!=expected:raise ValueError('Missing native per-row recovery evidence')
 bindingref=doc['dense_qualification_binding'];binding=read_json(_checked(bindingref,p.parent));parts=deepcopy(binding['qualified_source_parts']);previous_unknown=_array(binding['withdrawn_source_store_ordinals'],p.parent);old_unknown=set(map(int,previous_unknown));restored_ord=ordinal[wanted-n0].astype('<u8')
 if not set(map(int,restored_ord))<=old_unknown:raise ValueError('Restoration includes a row not currently withheld')
 out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True);commits=[];allchanges=defaultdict(lambda:defaultdict(dict))
 for i,o in zip(wanted,restored_ord):
  part=next(v for v in parts if v['ordinal_start']<=int(o)<v['ordinal_start']+v['row_count']);sid=doc['shot_ids'][int(a['shot_index'][i])];pts=int(a['native_pts'][i]);ident=doc['source_row_ids'][i]
  allchanges[part['name']][sid,pts][ident]=dict(physical_instance_qualification='eligible',source_fit_eligible=True,physical_instance_support_state='native_material_and_temporal_source_confirmed',qualification_reason_code='original_native_ice_material_recovered',qualification_review_id='ice_recovery/'+sid+'/'+str(pts),qualification_evidence=eref,source_recovery_original_row_index=int(i))
 spec=importlib.util.spec_from_file_location('rinkspace_public_source_storage',ROOT/'scripts/source_storage.py');public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
 for part in parts:
  changes=allchanges.get(part['name'])
  if not changes:continue
  sub=out/part['name'];sub.mkdir();source=part['reference'];req=dict(schema='rinkspace.source_storage_commit_request.v1',db=source['store'],base_rows=source,base_frames=base_frames_by_stream[source['stream']],new_rows_stream=stream_prefix+'/'+part['name']+'/selected_rows',new_frames_stream=stream_prefix+'/'+part['name']+'/selected_frames',changes=[dict(shot_id=k[0],native_pts=k[1],update=v)for k,v in sorted(changes.items())],source_evidence=[eref,_ref(__file__)])
  reqref=_write(sub/'request.json',req);published=public.commit(reqref['file'],sub/'source_manifest.json');result=read_json(published['manifest']);transition=result['actual_validation']['actual_transition']
  if transition['removed_selected_rows']or transition['added_selected_rows']or result['source_UV_unchanged']is not True:raise ValueError('Recovery altered original readings')
  commits.append(dict(name=part['name'],parent_reference=source,selected_rows=result['selected_rows'],selected_frames=result['selected_frames'],source_generation=_ref(published['manifest']),restored_rows=sum(len(v)for v in changes.values())))
  part['reference']=result['selected_rows'];print('source restored',part['name'],commits[-1]['restored_rows'],flush=True)
 generation=_write(out/'source_generation.json',dict(schema='rinkfit.multi_part_source_recovery_generation.v1',complete=True,source_commits=commits,evidence=eref,issuer=_ref(__file__),source_UV_ID_PTS_unchanged=True))
 remaining=np.array(sorted(old_unknown-set(map(int,restored_ord))),dtype='<u8');remaining.tofile(out/'withdrawn_source_store_ordinals.u64');restored_ord.sort();restored_ord.tofile(out/'restored_source_store_ordinals.u64');wanted.astype('<u8').tofile(out/'restored_packet_rows.u64')
 def arrayref(name,dtype,count):return dict(_ref(out/name),dtype=dtype,shape=[count])
 unknownref=arrayref('withdrawn_source_store_ordinals.u64','<u8',len(remaining));restoreref=arrayref('restored_source_store_ordinals.u64','<u8',len(wanted));rowref=arrayref('restored_packet_rows.u64','<u8',len(wanted));newactive=int(active.sum())+len(wanted);newdense=int(binding['total_active_rows'])+len(wanted)
 nextbinding=deepcopy(binding);nextbinding.update(parent_packet=_ref(p),source_observation_generation=generation,qualified_source_parts=parts,withdrawn_source_store_ordinals=unknownref,total_active_rows=newdense,total_unknown_rows=len(remaining),parent_qualification_binding=bindingref,restored_source_store_ordinals=restoreref,restored_packet_rows=rowref,restoration_evidence=eref,issuer=_ref(__file__))
 b_ref=_write(out/'source_binding.json',nextbinding)
 delta=dict(schema='rinkfit.dense_source_requalification_delta.v1',parent_source_qualification=_ref(qpath),source_packet=_ref(p),actual_review_consumer=_ref(__file__),source_observation_generation=generation,source_review_evidence=eref,withdrawn_source_store_ordinals=unknownref,restored_source_store_ordinals=restoreref,restored_packet_rows=rowref,from_qualification='unknown',to_qualification='eligible',source_semantics_verified=True,qualification_granted=True,selected_by_model_residual=False,cumulative_withdraw_only=False,source_qualification_binding=b_ref,restored_rows=len(wanted),original_source_UV_ID_PTS_material_path_preserved=True)
 d_ref=_write(out/'qualification_delta.json',delta);arrays=deepcopy(doc['arrays'])
 for key in ('admission','fit_role'):
  value=a[key].copy();value[wanted]=1;value.tofile(out/(key+'.u8'));arrays[key]=dict(arrayref(key+'.u8',value.dtype.str,len(value)),layout='C',byte_order='little')
 child=dict(schema=doc['schema'],dense_parent_packet=_ref(p),arrays=arrays,active_rows=newactive,dense_qualification_binding=b_ref,source_qualification_delta=d_ref,source_requalification=d_ref,dense_parent_append=dict(source_manifests=[b_ref,d_ref,eref,generation,_ref(__file__)]));packetref=_write(out/'packet.json',child)

 # Membership can be restored after native advertisement batches have been appended.
 # Keep the newest suffix denominator and all its original source registries.
 registry_kind='native_ad_source_binding' if 'native_ad_source_binding' in doc else 'dense_source_binding'
 registry=doc[registry_kind];tail=int(registry['row_start']);tail_count=int(registry['count'])
 now=active.copy();now[wanted]=True
 qnext=dict(schema='rinkfit.native_material_recovery_qualification.v1',source_packet=packetref,parent_packet=_ref(p),parent_qualification=_ref(qpath),recovery=d_ref,issuer=_ref(__file__),registry_kind=registry_kind,compact_registry=registry,parent_rows=tail,raw_rows=n,new_rows=tail_count,active_new_rows=int(now[tail:].sum()),unknown_new_rows=int((~now[tail:]).sum()),membership_array_sha256={k:v['sha256']for k,v in arrays.items()},source_UV_ID_PTS_operators_unchanged=True,source_rows_deleted=0,quality_accepted=False)
 qref=_write(out/'source_qualification.json',qnext)
 current={k:_array(w['files'][k],wpath.parent).copy()for k in original};wf={}
 for name,value in current.items():
  value[wanted]=original[name][wanted];value.tofile(out/name);wf[name]=arrayref(name,value.dtype.str,len(value))
 if np.any(current['weights.f64'][wanted]<=0)or not np.array_equal(current['active.u8'].astype(bool),now):raise ValueError('Restored weight/membership mismatch')
 wn=dict(schema='rinkfit.native_material_recovery_engineering_weights.v1',source_packet=packetref,parent_packet=_ref(p),parent_weights=_ref(wpath),original_weights=_ref(original_weights),issuer=_ref(__file__),dense_source_qualification=qref,recovery=d_ref,restored_packet_rows=rowref,raw_rows=n,parent_rows=tail,active_rows=newactive,new_active_rows=int(now[tail:].sum()),files=wf,retained_parent_budget=float(current['weights.f64'][:tail].sum()),new_dense_budget=float(current['weights.f64'][tail:].sum()),actual_total_engineering_budget=float(current['weights.f64'].sum()),source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False,unaffected_weights_and_restored_original_weights_bitwise_equal=True)
 wref=_write(out/'engineering_weights.json',wn);receipt=dict(schema='rinkfit.original_native_ice_recovery_packet.v1',packet=packetref,source_qualification=qref,engineering_weights=wref,source_generation=generation,source_binding=b_ref,qualification_delta=d_ref,restored_packet_rows=rowref,restored_rows=len(wanted),raw_rows=n,active_rows=newactive,original_source_UV_ID_nativePTS_material_path_ordinals_preserved=True,unaffected_weights_and_restored_original_weights_bitwise_equal=True,physical_solver_started=False,quality_accepted=False)
 _write(out/'receipt.json',receipt);return receipt
