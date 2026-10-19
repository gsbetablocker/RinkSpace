"""Exact-exposure old readouts navigate current complete board material tests."""
from pathlib import Path
import argparse,json
from collections import Counter
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .role_guard import FrameRoleGuard

def compile_contexts(run,navigation,outdir,selected_only=False):
 run=Path(run);nav=Path(navigation);out=Path(outdir)
 if out.exists():raise ValueError('New immutable context ledger required')
 guard=FrameRoleGuard(run/'roles/frame_roles.json');scope=read_json(run/'programme_scope.json');selected={int(f['native_pts']):f for f in read_json(run/'native_selected_v2/manifest.json')['frames']};native={int(f['native_pts']):f for n in ('dense_s03_v1','dense_remaining_v1')for f in read_jsonl(run/n/'frame_ledger.jsonl.gz')};frames=[];ledger=[];counts=Counter();navref=dict(path=str(nav/'frame_anchors.jsonl.gz'),sha256=file_sha256(nav/'frame_anchors.jsonl.gz'))
 for fr in read_jsonl(nav/'frame_anchors.jsonl.gz'):
  pts=int(fr['native_pts']);sid=fr['shot_id']
  if selected_only and pts not in selected:continue
  guard.assert_training_pixels([pts],shot_id=sid);anchors=fr['navigation_anchors'];top=np.asarray([a['navigation_uv']for a in anchors if a['material_role']=='board.top'],float).reshape(-1,2);bottom=np.asarray([a['navigation_uv']for a in anchors if a['material_role']=='board.contact'],float).reshape(-1,2);reason=None
  if not len(top) and not len(bottom):reason='same_exposure_board_navigation_absent'
  if not reason:
   allnav=np.r_[top,bottom];x0=max(0,int(np.floor(allnav[:,0].min()-96)));x1=min(1920,int(np.ceil(allnav[:,0].max()+97)))
   if len(top):t0=max(0,int(np.floor(top[:,1].min()-48)));t1=min(1079,int(np.ceil(top[:,1].max()+8)))
   else:t0=max(0,int(np.floor(bottom[:,1].min()-600)));t1=max(0,int(np.ceil(bottom[:,1].max()-24)))
   if len(bottom):b0=max(0,int(np.floor(bottom[:,1].min()-12)));b1=min(1079,int(np.ceil(bottom[:,1].max()+56)))
   else:b0=min(1079,int(np.floor(top[:,1].min()+32)));b1=min(1079,int(np.ceil(top[:,1].max()+600)))
   if x1-x0<120 or b1-t0<40 or b1-t0>1000 or t0>=b1:reason='same_native_navigation_object_search_domain_ambiguous'
  row=dict(native_pts=pts,shot_id=sid,status=reason or'current_complete_material_context_read_required',native_navigation_only=True,old_UV_identity_W_or_geometry_not_inherited=True)
  if not reason:
   nr=native[pts];sf=selected.get(pts);row.update(time_base='1/12800',source_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=nr['decoded_native_BGR_sha256'],native_png=str(run/'native_selected_v2'/sf['native_png'])if sf else None,native_png_sha256=sf['native_png_sha256']if sf else None,board_instance_roi_xyxy=[x0,max(0,t0-8),x1,min(1080,b1+8)],top_blue_upper_search_y_range=[t0,t1],contact_yellow_lower_search_y_range=[b0,b1],top_blue_upper_visible=bool(len(top)),contact_yellow_lower_visible=bool(len(bottom)),ROI_is_search_domain_only=True,source_navigation_anchors=anchors,source_navigation_index=navref,source_top_readout_navigation_uv=top.tolist(),source_contact_readout_navigation_uv=bottom.tolist(),physical_semantic_authority='current_native_actual_board_material_sequence_with_same_exposure_readout_navigation_only',source_object_identity_not_proved_by_navigation=True,local_navigation_support_mode='both'if len(top)and len(bottom)else('top_only'if len(top)else'contact_only'),missing_navigation_role_not_emitted_as_fit_source=True,fit_UV_created=False);frames.append(row)
  ledger.append(row);counts.update([row['status']])
 out.mkdir(parents=True);write_json(out/'board_instance_regions.json',dict(schema='rinkfit.joint9.same_native_readout_local_board_navigation.v1',frames=frames,navigation_source=navref,source_navigation_is_not_physical_identity_or_UV=True,actual_source_FIT_UV_created=False));write_jsonl(out/'frame_ledger.jsonl.gz',ledger);result=dict(frames=len(ledger),navigation_context_frames=len(frames),selected_only=selected_only,counts=dict(counts),no_actual_source_UV_created=True);write_json(out/'manifest.json',result);print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--navigation',required=True);p.add_argument('--out',required=True);p.add_argument('--selected-only',action='store_true');a=p.parse_args();compile_contexts(a.run,a.navigation,a.out,a.selected_only)
