"""Re-read every selected exposure with the shared material-aware operator.

Writes a complete diagnosis/remeasurement generation without changing source
UV, manual clicks, old statistics, or flags. New means require new statistics.
"""
from pathlib import Path
from itertools import groupby
from collections import Counter,defaultdict
import argparse,json,hashlib,gzip
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .physical_source_operator import prepare_current_source,remeasure_physical_source
from .role_guard import FrameRoleGuard

def run_selected(project,run,source,outdir):
 project=Path(project);run=Path(run);source=Path(source);out=Path(outdir)
 if out.exists():raise ValueError('New immutable semantic source generation required')
 native=run/'native_selected_v2';manifest=read_json(native/'manifest.json');frames={(f['shot_id'],int(f['native_pts'])):f for f in manifest['frames']};guard=FrameRoleGuard(run/'roles/frame_roles.json');data=next(x for x in run.parents if x.name=='RinkSpace_Data');masks={}
 for sid in sorted({v[0] for v in frames}):
  n=int(sid[1:]);folder=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';path=folder/'r2/frame_manifest.json' if n==2 else folder/'frame_manifest.json'
  for f in read_json(path)['frames']:masks[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);(out/'frame_certificates').mkdir();counts=Counter();bykind=defaultdict(Counter);seen=set();frame_reports=[]
 code_refs=[dict(path=str(Path(__file__).parent/n),sha256=file_sha256(Path(__file__).parent/n)) for n in ('semantic_source_worker.py','physical_source_operator.py','board_outer_source.py','measurement_contract.py')];write_json(out/'code_references.json',code_refs)
 with gzip.open(out/'relocated_rows.jsonl.gz','wt',encoding='utf8') as writer,gzip.open(out/'operator_decisions.jsonl.gz','wt',encoding='utf8') as audit_writer:
  for key,members in groupby(read_jsonl(source),key=lambda r:(r['shot_id'],int(r['native_pts']))):
   if key in seen:raise ValueError('Source input is not grouped by complete exposure')
   seen.add(key);sid,pts=key;guard.assert_training_pixels([pts],shot_id=sid);fr=frames[key];png=native/fr['native_png']
   if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Official selected PNG changed')
   bgr=cv2.imread(str(png));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mr=masks[key];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));sha=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
   if sha and file_sha256(mp)!=sha:raise ValueError('Current source mask changed')
   mask=cv2.imread(str(mp),0)>0;context=prepare_current_source(rgb,mask,project);fc=Counter();decisions=[]
   for original in members:
    row=dict(original);expected=context['contract'].expected(row.get('target_id'),row.get('category'));old=row.get('source_measurement_audit') or {};seed=row.get('current_uv') or row['source_uv'];normal=old.get('normal');manual=bool(row.get('manual_reference_preserved'))
    rr=remeasure_physical_source(context,row,seed,normal,old.get('flank_uv'))
    row['previous_source_audit_current_uv']=row.get('current_uv');row['previous_source_audit_operator']=row.get('measurement_operator_resolved');row['previous_source_measurement_audit']=old;row['measurement_contract_sha256']=context['contract'].digest
    if manual:
     row['current_uv']=original.get('current_uv');row['source_measurement_audit']=old;row['semantic_automatic_source_diagnostic']=rr;status='manual_click_preserved_no_automatic_profile_noise_substitution'
    else:
     row['current_uv']=rr.get('current_uv');row['source_measurement_audit']=rr;row['measurement_operator_resolved']=expected.get('operator') if expected else 'unspecified';status=rr['status']
    if rr.get('source_material_receipt'):
     rr['source_material_receipt']['source_family_id']=row.get('source_family_id');rr['source_material_receipt']['source_row_id']=row['source_row_id'];rr['source_material_receipt']['native_pts']=pts;rr['source_material_receipt']['native_png_sha256']=fr['native_png_sha256']
    row['solver_eligible']=False;row['numeric_statistics_ready']=False;row['semantic_local_material_operator_passed']=bool(rr.get('current_uv') is not None);row['semantic_finite_identity_reaudit_pending']=True;row['source_definition_reaudit_status']='current_material_remeasured_finite_identity_pending' if row['semantic_local_material_operator_passed'] else 'current_material_operator_pending';row['source_uv_and_original_flags_unchanged']=True
    d=dict(source_row_id=row['source_row_id'],shot_id=sid,native_pts=pts,target_id=row.get('target_id'),source_family_id=row.get('source_family_id'),status=status,manual_click_preserved=manual,expected=expected,current_uv=row['current_uv'],local_material_operator_passed=row['semantic_local_material_operator_passed'],finite_material_identity_proved=False,measurement_contract_sha256=context['contract'].digest)
    writer.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');audit_writer.write(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');fc[status]+=1;counts[status]+=1;bykind[expected['operator'] if expected else 'unclosed'][status]+=1
   certfile=out/'frame_certificates'/f'{pts}.json';write_json(certfile,dict(shot_id=sid,native_pts=pts,native_png=str(png),native_png_sha256=file_sha256(png),decoded_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),current_mask=str(mp),current_mask_sha256=file_sha256(mp),measurement_contract_sha256=context['contract'].digest,authority_references=context['contract'].references,actual_outer_source_certificates=context['outer_certificates'],actual_outer_rows=context['outer_rows'],counts=dict(fc),solver_admission=False))
   frame_reports.append(dict(shot_id=sid,native_pts=pts,counts=dict(fc),source_certificate_file=str(certfile),sha256=file_sha256(certfile)));print(json.dumps(dict(native_pts=pts,frames=len(seen),counts=dict(counts))),flush=True);context['contract'].verify_unchanged()
 if seen!=set(frames):raise ValueError('Not every selected exposure was actually re-read')
 result=dict(schema='rinkfit.full_selected_actual_material_semantic_reread.v1',source=dict(path=str(source),sha256=file_sha256(source)),selected_frames=len(seen),rows=sum(counts.values()),counts=dict(counts),by_operator={k:dict(v) for k,v in bykind.items()},frame_reports=frame_reports,actual_RGB_reread_all_selected_frames=True,raw_source_uv_or_manual_clicks_or_flags_changed=False,original_W_reused_for_new_mean=False,solver_admitted_rows=0,finite_source_identity_and_new_statistics_pending=True,protected_pixels_consumed=False)
 for ref in code_refs:
  if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Source operator implementation changed within the remeasurement generation')
 result['code_references']=code_refs;write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run_selected(a.project,a.run,a.source,a.out)))
