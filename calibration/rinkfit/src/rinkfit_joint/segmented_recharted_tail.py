"""Reuse fitted first4 state plus actual six-shot captured numeric tail, preserving raw facts."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .dense_packet_delta_io import write_dense_packet

def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p))
def _check(r):
 p=Path(r['file']);assert file_sha256(p)==r['sha256'],str(p);return p
def prepare_recharted_tail_packet(*,prefix_overlay,current_packet,current_qualification,current_model,tail_state,tail_source_packet,tail_complete_verification,output):
 out=Path(output).resolve();assert not out.exists()
 j=read_json(prefix_overlay);assert j['schema']=='rinkfit.segmented_native_numeric_auxiliary_overlay.v1'
 current=read_json(current_packet);old=read_json(tail_source_packet);n=current['row_count'];n0=current['dense_source_binding']['row_start'];assert n==old['row_count'] and j['raw_rows']==n
 model=read_json(current_model);state=Path(tail_state);bm=read_json(state/'model.json');assert bm==model,'All six physical cameras/scene must be identical'
 for k in current['arrays']:
  if k not in ('admission','fit_role'):assert current['arrays'][k]==old['arrays'][k],k
 for k in ('row_count','shot_ids','dense_source_binding','queries','groups','allowed_paths','engineering_pixel_dependency_groups'):
  assert current.get(k)==old.get(k),k
 def col(d,k):
  s=d['arrays'][k];_check(s);return np.memmap(s['file'],dtype=s['dtype'],mode='r',shape=tuple(s['shape']))
 ca=(col(current,'admission')!=0)&(col(current,'fit_role')!=0);oa=(col(old,'admission')!=0)&(col(old,'fit_role')!=0);assert not np.any(ca&~oa)
 b=read_json(state/'source_uv_capture_state_binding.json');manifest=read_json(state/'explicit_nuisance/manifest.json');verified=read_json(tail_complete_verification)
 assert b['packet_sha256']==file_sha256(tail_source_packet)==manifest['packet_sha256']==verified['source_packet_sha256']
 assert b['model_sha256']==file_sha256(state/'model.json') and b['state_token']==verified['complete_state_token'] and verified['passed'] and verified['all_four_raw_numeric_columns_bitexact_to_complete_state'] and verified['complete_physical_model_exact_to_a0']
 cpref=_ref(state/'engineering_checkpoint.json');assert cpref['sha256']==verified['complete_checkpoint_sha256']
 arrays={};aux={}
 for k in ('station','row','alpha','path'):
  dtype='<u4'if k=='path'else'<f8';name=k+('.u32'if k=='path'else'.f64');p=state/'explicit_nuisance'/name;r=manifest['files'][name];assert file_sha256(p)==r['sha256'] and p.stat().st_size==r['size_bytes']==n*np.dtype(dtype).itemsize
  arrays[k]=dict(**_ref(p),dtype=dtype,shape=[n]);aux[k]=np.memmap(p,dtype=dtype,mode='r',shape=(n,));assert np.isfinite(aux[k]).all()
 status_file=Path(b['actual_derived_row_status']['RS_status']);status=np.memmap(status_file,dtype='u1',mode='r',shape=(n,));failed=ca&(status!=1);failed[:n0]=False
 out.mkdir(parents=True)
 subset=dict(schema='rinkfit.actual_captured_tail_source_subset.v1',old_source_packet=_ref(tail_source_packet),current_source_packet=_ref(current_packet),current_model=_ref(current_model),tail_model=_ref(state/'model.json'),current_qualification=_ref(current_qualification),all_six_physical_cameras_and_scene_JSON_exact=True,original_source_identity_and_raw_16_arrays_exact=True,current_active_is_strict_subset=True,old_active=int(oa.sum()),current_active=int(ca.sum()),newly_withdrawn=int((oa&~ca).sum()),newly_admitted=0,tail_active=int(ca[n0:].sum()),tail_failed_RS_retained=int(failed.sum()),RS_status=dict(**_ref(status_file),dtype='|u1',shape=[n]),complete_state_verification=_ref(tail_complete_verification),complete_checkpoint=cpref,numeric_array_refs=arrays,source_pixels_decoded=0,source_search_repeated=False,old_AL_or_physical_state_imported=False,quality_accepted=False)
 write_json(out/'tail_source_subset.json',subset)
 j['source_packet']=_ref(current_packet);j['source_geometry_qualification']=_ref(current_qualification);j['candidate_model']=_ref(current_model)
 j['pieces'][1]=dict(kind='complete_native_checkpoint_tail',start=n0,count=n-n0,source_offset=n0,source_packet=_ref(tail_source_packet),current_source_packet=_ref(current_packet),physical_model=_ref(state/'model.json'),complete_checkpoint=cpref,state_binding=_ref(state/'source_uv_capture_state_binding.json'),explicit_nuisance_manifest=_ref(state/'explicit_nuisance/manifest.json'),complete_state_token=b['state_token'],complete_state_verification=_ref(tail_complete_verification),source_subset_verification=_ref(out/'tail_source_subset.json'),arrays=arrays)
 j['issuer']=_ref(__file__);j['tail_root_capture_is_not_complete']=True;j['tail_failed_RS_retained']=int(failed.sum());write_json(out/'numeric_overlay.json',j)
 doc=deepcopy(current);doc['dense_parent_packet']=_ref(current_packet);doc['dense_numeric_auxiliary_overlay']=_ref(out/'numeric_overlay.json');write_dense_packet(out/'packet.json',doc)
 receipt=dict(schema='rinkfit.actual_fitted_prefix_captured_tail_packet.v1',packet=_ref(out/'packet.json'),overlay=_ref(out/'numeric_overlay.json'),tail_source_subset=_ref(out/'tail_source_subset.json'),prefix_complete_state_token=j['pieces'][0]['complete_state_token'],tail_complete_state_token=b['state_token'],raw_rows=n,active_rows=int(ca.sum()),failed_tail_RS_retained=int(failed.sum()),new_numeric_array_files=0,physical_model_changed=False,source_qualification_changed=False,source_qualification_and_weights_rebind_required=True,quality_accepted=False);write_json(out/'receipt.json',receipt);return receipt
