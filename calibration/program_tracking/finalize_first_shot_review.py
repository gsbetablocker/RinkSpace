"""Assemble the remaining first-shot native observations without changing approval."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from .validate_opening_frame_batch import read,sha,reference,require,assemble,source_vertices
from .extract_selected_frame_review import make_binding
from .framewise_model import evaluate_rows
from .render_opening_frame_batch import run as render,page_html,timestamp
from .render_single_frame_source_review import LAYERS,source_segments,display_segments

PARENT=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
OUT=PARENT/'first_shot_remaining_50fps'
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps')
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014'
OLD_MODEL=MODEL.parent/'framewise_8a28_20261013'
FIRST,LAST=201,1248
SOURCE_DATA=DATA
SOURCE_OUT=OUT
REPAIR_MODE=False
EDITOR_URL='http://127.0.0.1:48172/'

def frame_manifest_path():
 return DATA/'frame_manifest.json' if REPAIR_MODE and (DATA/'frame_manifest.json').exists() else SOURCE_DATA/'frame_manifest.json'

def navigation_mapping_path():
 return OUT/'navigation_mapping.json' if REPAIR_MODE and (OUT/'navigation_mapping.json').exists() else SOURCE_OUT/'navigation_mapping.json'

def configure_repair(model=None):
 """Keep the failed release intact while assembling a reviewed replacement."""
 global OUT,DATA,MODEL,REPAIR_MODE,EDITOR_URL
 OUT=SOURCE_OUT/'repair_20261014';DATA=SOURCE_DATA/'repair_20261014'
 MODEL=Path(model).resolve() if model else ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
 REPAIR_MODE=True;EDITOR_URL='http://127.0.0.1:48173/'
 require((DATA/'baseline_receipt.json').is_file(),'Repair baseline archive is missing')

def baseline_integrity():
 if not REPAIR_MODE:return None
 receipt=read(DATA/'baseline_receipt.json');checked=0
 for item in receipt['files']:
  path=Path(item['path'])
  # Extraction code is deliberately repaired; delivered data remain untouched.
  if path.is_relative_to(SOURCE_OUT) or path.parent==ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014':
   require(sha(path)==item['sha256'],f'Original delivered data changed: {path}')
   if path.parent in (SOURCE_OUT/'frames',SOURCE_OUT/'bindings'):checked+=1
 require(checked==2096,'Original baseline does not cover every packet and binding')
 return dict(status='passed',original_packet_and_binding_files=checked,receipt=reference(DATA/'baseline_receipt.json'))

def repair_quality_review(validation):
 """This receipt records actual source inspection, not a numerical proxy for it."""
 baseline_integrity();path=DATA/'repair_quality_review.json';review=read(path)
 require(review['model_entry_validation']['sha256']==sha(DATA/'model_entry_validation.json'),'Repair review belongs to a different candidate')
 require(review['programme_frame_range']==[FIRST,LAST],'Repair source review does not cover the entire candidate shot')
 require(review['unresolved_defects']==[],'Unresolved repair defects remain')
 for key in ('wide_line_material_midlines','shared_projective_ring_layers','ad11_material_circle','motion_compensated_continuity','continuous_native_visual_review'):
  require(review['checks'][key]['evidence'],'Missing source-review evidence: '+key)
 for row in validation['frames']:
  for key in ('packet','model_binding'):
   require(sha(row[key]['path'])==row[key]['sha256'],'Candidate changed after repair validation')
 return reference(path)

def prepare_repair_sources():
 require(REPAIR_MODE,'Source revision preparation requires --repair')
 from .render_result import SceneRenderer
 from .extract_first_shot_ice import current_ice_visible_domain
 manifest=read(SOURCE_DATA/'frame_manifest.json');navigation=read(SOURCE_OUT/'navigation_mapping.json')
 receipt_path=DATA/'masks/Ad11_mask_repair_receipt.json';corrections=read(receipt_path)['rows']
 by_index={r['index']:r for r in corrections};cameras=np.load(SOURCE_DATA/'camera_seeds.npz')['camera25']
 renderer=SceneRenderer(read(MODEL/'scene.json'),draw_grid=True,draw_advertisements=True)
 navigation['historical_model_seed']=deepcopy(navigation['model_seed'])
 navigation['model_seed']['scene']=reference(MODEL/'scene.json')
 nav_rows={r['programme_frame_index_1based']:r for r in navigation['frames']};checks=[]
 for row in manifest['frames']:
  i=row['index'];nav=nav_rows[i]
  if i in by_index:
   fix=by_index[i];source=row['source'];require(sha(fix['path'])==fix['sha256'],'New source mask changed')
   require(source['mask_sha256']==fix['parent_mask_sha256']==sha(fix['parent_mask_path']),'Original source mask changed')
   old=cv2.imread(fix['parent_mask_path'],0);mask=cv2.imread(fix['path'],0)
   changed=old!=mask;outside_ice,_=current_ice_visible_domain(i)
   require(not np.any(changed&~outside_ice),'Ad11 mask correction unexpectedly changed the ice region')
   source.update(mask_path=fix['path'],mask_sha256=fix['sha256'],mask_revision_provenance=reference(receipt_path))
   row['mask_path']=fix['path']
   checks.append(dict(index=i,changed_mask_pixels=int(changed.sum()),changed_ice_region_pixels=0,
     actual_source_pixels_unchanged=True,new_mask=reference(fix['path'])))
  source=row['source'];path=DATA/f'navigation/{i:06d}.png';metadata=path.with_suffix('.json')
  expected=dict(scene_sha256=sha(MODEL/'scene.json'),mask_sha256=source['mask_sha256'],camera_sha256=nav['camera25_float64_sha256'])
  if not (path.exists() and metadata.exists() and read(metadata)==expected):
   raw=cv2.imread(source['image_path']);mask=cv2.imread(source['mask_path'],0)
   rendered,_=renderer.draw(raw,cameras[i-FIRST],person_mask=mask>0,title=None)
   require(np.array_equal(rendered[mask>0],raw[mask>0]),'Revised navigation overwrote excluded foreground')
   path.parent.mkdir(parents=True,exist_ok=True)
   require(cv2.imwrite(str(path),rendered,[cv2.IMWRITE_PNG_COMPRESSION,1]),'Navigation write failed')
   write(metadata,expected)
  nav['navigation_overlay'].update(path=str(path),sha256=sha(path),current_mask_sha256=source['mask_sha256'],
    current_model_scene=reference(MODEL/'scene.json'),all_mask_pixels_preserved_from_source=True)
  row['source']['navigation_overlay_path']=nav['navigation_overlay']['path']
 manifest['repair_source_revision']=dict(original_frame_manifest=reference(SOURCE_DATA/'frame_manifest.json'),
   official_images_reused_without_changes=True,mask_corrections=reference(receipt_path),checks=checks)
 accepted_navigation=[]
 for name in ('opening_100_50fps','continuation_101_200_50fps'):
  directory=PARENT/name;media=SOURCE_DATA.parent/name;accepted_cameras=np.load(media/'camera_seeds.npz')
  accepted_indices=accepted_cameras['programme_indices'] if 'programme_indices' in accepted_cameras.files else 1+(accepted_cameras['official_pts']-256256)//256
  for i,camera in zip(accepted_indices,accepted_cameras['camera25']):
   i=int(i);packet=read(directory/f'frames/{i:06d}.json');source=packet['source']
   path=DATA/f'navigation/{i:06d}.png';metadata=path.with_suffix('.json')
   expected=dict(scene_sha256=sha(MODEL/'scene.json'),mask_sha256=source['mask_sha256'],camera_sha256=hashlib.sha256(camera.tobytes()).hexdigest())
   if not (path.exists() and metadata.exists() and read(metadata)==expected):
    raw=cv2.imread(source['image_path']);mask=cv2.imread(source['mask_path'],0)
    rendered,_=renderer.draw(raw,camera,person_mask=mask>0,title=None)
    require(cv2.imwrite(str(path),rendered,[cv2.IMWRITE_PNG_COMPRESSION,1]),'Accepted-frame reference navigation write failed')
    write(metadata,expected)
   accepted_navigation.append(dict(index=i,navigation=reference(path),source_packet=reference(directory/f'frames/{i:06d}.json')))
 manifest['accepted_frame_current_model_navigation']=accepted_navigation
 navigation['repair_mask_revision']=manifest['repair_source_revision']
 write(DATA/'frame_manifest.json',manifest);write(OUT/'navigation_mapping.json',navigation)
 print(json.dumps(dict(revised_masks=len(checks),current_model_navigation_frames=len(manifest['frames'])+len(accepted_navigation),native_source_images_copied=0)),flush=True)

def model_revision_integrity(scene,registry):
 if not REPAIR_MODE:
  require(sha(MODEL/'scene.json')==sha(OLD_MODEL/'scene.json'),'Original scene geometry changed')
  original=read(OLD_MODEL/'feature_registry.json')['targets']
  require(all(registry['targets'].get(k)==v for k,v in original.items()),'Existing model target changed')
  return None
 receipt_path=MODEL/'duplicate_centerline_retirement_receipt.json';receipt=read(receipt_path)
 parent=MODEL.parent;before=read(parent/'scene.json');old_registry=read(parent/'feature_registry.json')
 require(sha(parent/'scene.json')==receipt['scene_before_sha256'],'Model baseline scene changed')
 require(sha(MODEL/'scene.json')==receipt['scene_after_sha256'],'Model revision differs from reviewed scene')
 require(sha(parent/'feature_registry.json')==receipt['registry_before_sha256'],'Model baseline registry changed')
 require(sha(MODEL/'feature_registry.json')==receipt['registry_after_sha256'],'Model revision differs from reviewed registry')
 expected=deepcopy(before);expected['ice_features']=[f for f in before['ice_features'] if f['id']!='curling_centerline_row3']
 actual=deepcopy(scene)
 for key in ('superseded_duplicate_ice_features','superseded_ice_feature_aliases','physical_topology_corrections'):actual.pop(key,None)
 require(actual==expected,'Unexpected model geometry change outside duplicate retirement')
 targets=dict(old_registry['targets']);targets.pop('ice.curling_centerline_row3:segment0')
 require(registry['targets']==targets,'Unexpected target change outside duplicate retirement')
 return reference(receipt_path)

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def accepted_integrity():
 receipts=[]
 for name in ('opening_100_50fps','continuation_101_200_50fps'):
  directory=PARENT/name;receipt=read(directory/'user_acceptance.json')
  require(receipt['accepted'] and len(receipt['frames'])==100,'Missing accepted100 receipt')
  for r in receipt['frames']:
   for folder,key in [('frames','packet_sha256'),('bindings','binding_sha256')]:
    require(sha(directory/f'{folder}/{r["index"]:06d}.json')==r[key],'Accepted observations changed')
  receipts.append(dict(**reference(directory/'user_acceptance.json'),frame_count=100,point_count=receipt['point_count']))
 require(sum(x['point_count'] for x in receipts)==32883,'Accepted total changed')
 return dict(status='passed',frame_count=200,point_count=32883,receipts=receipts)

def compact_evidence(feature,key):
 """Keep final coordinates in OneDrive; bulky native profiles stay in Data."""
 f=deepcopy(feature)
 for fit_key in ('source_line_fit','source_circle_fit'):
  fit=f.get(fit_key,{})
  if 'direct_readouts' in fit:
   fit.pop('direct_readouts');fit['direct_readouts_reference']=dict(packet_reference=key,source_feature_id=f['id'],field=fit_key+'.direct_readouts')
 for si,s in enumerate(f['segments']):
  if 'source_evidence' in s:
   require(len(s['source_evidence'])==len(s['uv']),'Source evidence length mismatch')
   s['source_evidence']=[dict(packet_reference=key,source_feature_id=f['id'],source_segment_index=si,source_vertex_index=vi) for vi in range(len(s['uv']))]
 return f

def merge(indices=None):
 accepted_integrity();baseline_integrity();manifest=read(frame_manifest_path());registry=read(MODEL/'feature_registry.json')
 require(len(manifest['frames'])==LAST-FIRST+1,'Incomplete source preparation')
 for row in manifest['frames']:
  i=row['index']
  if indices and i not in indices:continue
  dest=OUT/f'frames/{i:06d}.json'
  if dest.exists() and read(dest).get('manual_editor'):continue
  paths={k:DATA/f'{k}_candidates/{i:06d}.json' for k in ('board','ice')}
  packets={k:read(p) for k,p in paths.items()};board,ice=packets['board'],packets['ice']
  for p in packets.values():
   require(p['source']['native_pts']==row['source']['native_pts']==256256+(i-1)*256,'Candidate clock differs')
   require(p['source']['mask_sha256']==row['source']['mask_sha256']==sha(row['source']['mask_path']),'Candidate mask stale')
  packet=deepcopy(board);features=[]
  for kind,p in packets.items():
   key=kind+'_source_readout_packet';packet[key]=reference(paths[kind])
   features.extend(compact_evidence(f,key) for f in p['features'] if f.get('point_count',sum(len(s['uv']) for s in f['segments'])))
  source=deepcopy(row['source']);source.update(source_points_extracted=True,optimization_performed=False)
  if ice['source'].get('broadcast_timer_exclusion'):source['broadcast_timer_exclusion']=ice['source']['broadcast_timer_exclusion']
  if ice['source'].get('ice_visible_domain'):source['ice_visible_domain']=ice['source']['ice_visible_domain']
  packet.update(source=source,features=features,point_count=sum(len(s['uv']) for f in features for s in f['segments']),
   programme_frame_index_1based=i,batch_sequence_index_1based=i,frame_count=1,status='awaiting_user_review_of_first_shot',
   user_accepted=False,user_hard_point_ceiling=500,optimization_performed=False,observation_coordinates_from_projection=False,
   inferred_segments=ice.get('inferred_segments',[]),source_correlated_groups=board.get('source_correlated_groups',[])+ice.get('source_correlated_groups',[]),
   ice_feature_coverage=ice.get('per_feature_coverage',[]),ice_parallel_family_audit=ice.get('parallel_family_geometry_audit'),
   measurement_definitions=dict(wide_straight_ice_lines='Both material sides measured; midpoint denotes zero-width model line.',
     finite_width_circular_ice_lines='C0 and A1-A4: midpoint between the two material edges denotes a zero-width circular model line; legacy C0:outer is only an identifier.',
     target_rings='F1-F4 and R1-R4 only: declared physical material contour; never substitute paint width center for a target boundary.',
     advertisement='Contours on continuous unwrapped board material surface; may span straight and curved portions.'))
  if REPAIR_MODE:
   packet['repair_of_packet']=reference(SOURCE_OUT/f'frames/{i:06d}.json')
   packet['repair_scope']='Current native source remeasurement of identified geometry and material failures; original delivery preserved.'
  require(packet['point_count']<=500,f'Frame {i} exceeds500: {packet["point_count"]}')
  write(dest,packet);write(OUT/f'bindings/{i:06d}.json',make_binding(packet,dest,registry))
  if i%100==0:print(f'Merged {i}: {packet["point_count"]} points',flush=True)

def validate(indices=None):
 scene,registry,template=[read(MODEL/n) for n in ('scene.json','feature_registry.json','ad2_material_template.json')]
 model_revision=model_revision_integrity(scene,registry)
 manifest=read(frame_manifest_path());nav=read(navigation_mapping_path());cameras=np.load(SOURCE_DATA/'camera_seeds.npz')
 representation=read(SOURCE_DATA/'official_representation_verification.json')
 require(representation.get('all_1048_decoded_pixels_identical') is True and representation['frame_count']==1048,'Official representation comparison incomplete')
 require(np.array_equal(cameras['programme_indices'],np.arange(FIRST,LAST+1)),'Camera indices wrong')
 source_audits={}
 if not indices and not REPAIR_MODE:
  board_audit=read(DATA/'board_candidate_audit.json');ice_audit=read(DATA/'ice_diagnostics/final_ice_audit_summary.json')
  require(board_audit['frames_processed']==1048 and not board_audit['errors'],'Board source audit incomplete or failed')
  require(ice_audit['native_frame_count']==1048 and not ice_audit['issues'],'Ice source audit incomplete or failed')
  require(ice_audit['visual_review']['all_frames_contact_pages_rendered'] and ice_audit['visual_review']['manually_inspected_page_ranges'],'Ice visual review incomplete')
  board_visual=read(DATA/'board_final_review_receipt.json')
  require(board_visual['frames']==1048 and board_visual['whole_sequential_boundary_visual_qc_complete'],'Board visual review incomplete')
  source_audits=dict(board=reference(DATA/'board_candidate_audit.json'),ice=reference(DATA/'ice_diagnostics/final_ice_audit_summary.json'),
   board_visual=reference(DATA/'board_final_review_receipt.json'),ice_material_identity=reference(DATA/'ice_diagnostics/extra_circle_material_identity_audit.json'))
 result=dict(schema='rinkspace.first-shot-model-entry-validation.v1',status='failed',frames=[],errors=[],optimization_performed=False,
  model_accuracy_accepted=False,diagnostic_weighting='explicit unit pixel scale, not final measurement covariance',
  original_scene_and_existing_targets_unchanged=not REPAIR_MODE,model_revision=model_revision,accepted_200_integrity=accepted_integrity(),
  current_source_to_historical_correspondence_pixels=reference(SOURCE_DATA/'official_representation_verification.json'),source_audits=source_audits,
  original_delivery_integrity=baseline_integrity(),validation_scope='Clock, model binding, source integrity and exclusions; does not establish visual accuracy.')
 totals=Counter()
 for row,camera,navigation in zip(manifest['frames'],cameras['camera25'],nav['frames']):
  i=row['index']
  if indices and i not in indices:continue
  try:
   path=OUT/f'frames/{i:06d}.json';bp=OUT/f'bindings/{i:06d}.json';p,b=read(path),read(bp)
   s=p['source'];pts=256256+(i-1)*256
   require(s['native_pts']==s['actual_pts']==b['source_pts']==navigation['official_pts']==pts,'Native PTS mismatch')
   require(s['time_base']==b['source_time_base']=='1/12800','Time base mismatch')
   require(s['source_video_sha256']==manifest['source']['sha256'],'Source hash mismatch')
   require(b['source_packet']['sha256']==sha(path),'Binding packet hash mismatch')
   require(navigation['camera25_float64_sha256']==hashlib.sha256(camera.tobytes()).hexdigest(),'Navigation camera changed')
   require(navigation['navigation_overlay']['current_mask_sha256']==s['mask_sha256'],'Navigation mask stale')
   for key in ('board_source_readout_packet','ice_source_readout_packet'):
    require(sha(p[key]['path'])==p[key]['sha256'],'Candidate changed after merge')
   raw=cv2.imread(s['image_path']);mask=cv2.imread(s['mask_path'],0)>0
   require(raw.shape==(1080,1920,3) and hashlib.sha256(raw.tobytes()).hexdigest()==s['decoded_BGR_sha256'],'Source pixels changed')
   require(sha(s['mask_path'])==s['mask_sha256'],'Mask changed')
   features,vertices=source_vertices(p);uv=np.asarray(list(vertices.values()));xy=np.rint(uv).astype(int)
   require(not mask[xy[:,1],xy[:,0]].any(),'Point in foreground')
   box=s.get('broadcast_timer_exclusion',{}).get('actual_native_rectangle_xyxy_inclusive')
   if box:
    halo=s['broadcast_timer_exclusion'].get('edge_halo_px',0);x0,y0,x1,y1=box
    require(not ((xy[:,0]>=x0-halo)&(xy[:,0]<=x1+halo)&(xy[:,1]>=y0-halo)&(xy[:,1]<=y1+halo)).any(),'Point in timer')
   for boundary in ('board_top','board_contact'):
    boundary_count=sum(len(s['uv']) for f in features.values() if f.get('review_layer')==boundary or f['id']==boundary or f['id'].startswith(boundary+'.') for s in f['segments'])
    require(boundary_count>=2,'Visible required board boundary missing: '+boundary)
   rows,targets,corners=assemble(p,b,registry,scene,template)
   residual,jacobian=evaluate_rows(camera,rows,jacobian=True);n=p['point_count']
   require(residual.shape==(2*n,) and jacobian.shape==(2*n,15),'Wrong model residual entry dimensions')
   require(np.isfinite(residual).all() and np.isfinite(jacobian).all(),'Nonfinite model residual entry')
   totals.update(targets)
   result['frames'].append(dict(index=i,status='passed',point_count=n,packet=reference(path),model_binding=reference(bp),
    native_pts=pts,source_time=s['source_time'],target_point_counts=dict(targets),corner_bindings=corners,
    source_coordinate_addresses_covered_exactly_once=True,foreground_point_count=0))
  except Exception as exc:result['errors'].append(dict(index=i,error=f'{type(exc).__name__}: {exc}'))
  if i%100==0:print(f'Validated {i}: {len(result["errors"])} errors',flush=True)
 result['aggregate_target_counts']=dict(totals);result['status']='passed' if len(result['frames'])==(len(indices) if indices else LAST-FIRST+1) and not result['errors'] else 'failed'
 write(DATA/('model_entry_validation_subset.json' if indices else 'model_entry_validation.json'),result)
 require(result['status']=='passed',str(result['errors'][:20]));return result

def publish(validation):
 counts=[r['point_count'] for r in validation['frames']];require(len(counts)==1048,'Incomplete validation')
 quality=repair_quality_review(validation) if REPAIR_MODE else None
 frames=[]
 for r in validation['frames']:
  i=r['index'];frames.append(dict(**r,user_accepted=False,image_path=str(SOURCE_DATA/f'frames/{i:06d}.png'),overlay_path=str(DATA/f'overlays/{i:06d}.png')))
 manifest=dict(schema='rinkspace.first-shot-remaining-batch.v1',event_id='ankara_2026',programme_id='okada_mei_sp',
  status='first_shot_completed_awaiting_user_review',programme_frame_range=[FIRST,LAST],frame_count=1048,
  first_shot_frame_range=[1,1248],source_fps='50/1',source_time_base='1/12800',first_pts=307456,last_pts=575488,end_pts_exclusive=575744,
  source_interval='00:24.020–00:44.960',whole_first_shot_source_interval='00:20.020–00:44.960',user_accepted=False,
  density=dict(minimum=min(counts),maximum=max(counts),mean=float(np.mean(counts)),total=sum(counts),user_hard_ceiling=500),
  accepted_200_integrity=validation['accepted_200_integrity'],source_clock_manifest=reference(frame_manifest_path()),
  navigation_mapping=reference(navigation_mapping_path()),manual_annotation_inventory=reference(SOURCE_OUT/'historical_manual_annotation_inventory.json'),
  model_entry_validation=reference(DATA/'model_entry_validation.json'),model_id=read(MODEL/'model.json')['model_id'],
  source_audits=validation['source_audits'],
  model_path=str(MODEL),optimization_performed=False,model_accuracy_accepted=False,strict_v6_used=False,
  review_html=str(OUT/'review.html'),editable_review_url=EDITOR_URL,repair_quality_review=quality,frames=frames)
 if REPAIR_MODE:
  manifest['accepted_review_batches']=[dict(dataset=str(PARENT/name),media=str(SOURCE_DATA.parent/name))
      for name in ('opening_100_50fps','continuation_101_200_50fps')]
 write(OUT/'batch_manifest.json',manifest)
 render(SimpleNamespace(dataset=OUT,media=DATA,source_media=SOURCE_DATA,html=None,expected_frames=1048,skip_overlays=False))
 whole=[];batches=[]
 for name,lo,hi in [('opening_100_50fps',1,100),('continuation_101_200_50fps',101,200),('first_shot_remaining_50fps',201,1248)]:
  directory=PARENT/name;media=DATA.parent/name
  source_media=SOURCE_DATA.parent/name
  if hi<=200:media=source_media
  else:directory=OUT;media=DATA;source_media=SOURCE_DATA
  batches.append(dict(**reference(directory/'batch_manifest.json'),programme_frame_range=[lo,hi],user_accepted=hi<=200))
  for i in range(lo,hi+1):
   p=read(directory/f'frames/{i:06d}.json');segments=list(source_segments(p));source=Path(p['source']['image_path']);navigation=Path(p['source'].get('navigation_overlay_path',source_media/f'navigation/{i:06d}.png'))
   if REPAIR_MODE and (DATA/f'navigation/{i:06d}.png').exists():navigation=DATA/f'navigation/{i:06d}.png'
   whole.append(dict(index=i,source=source.as_uri(),navigation=navigation.as_uri() if navigation.exists() else None,
    overlay=(media/f'overlays/{i:06d}.png').as_uri(),source_time=timestamp(p['source']),pts=p['source']['native_pts'],time_base='1/12800',
    point_count=p['point_count'],counts={k:sum(len(uv) for l,_,uv,_ in segments if l==k) for k in LAYERS},
    status='已由用户验收' if i<=200 else '等待用户验收',notes=[],inferred_segments=p.get('inferred_segments',[]),
    segments=display_segments(p)))
 write(OUT/'whole_first_shot_manifest.json',dict(schema='rinkspace.whole-first-shot-review.v1',frame_count=1248,
  programme_frame_range=[1,1248],accepted_frame_range=[1,200],candidate_frame_range=[201,1248],point_count=32883+sum(counts),
  source_interval='00:20.020–00:44.960',source_fps='50/1',batches=batches,geometry_optimized=False,
  review_kind='publication_snapshot',subsequent_manual_changes_url=EDITOR_URL,
  current_source_to_historical_correspondence_pixels=reference(SOURCE_DATA/'official_representation_first_shot_1_1248_verification.json')))
 whole_html=page_html(whole,OUT/'whole_first_shot_manifest.json').replace('<header><h1>',f'<header><p>这是本次交付时的取点快照；后续手动修改请查看 <a href="{EDITOR_URL}">可编辑页面中的当前结果</a>。</p><h1>',1)
 (OUT/'whole_first_shot_review.html').write_text(whole_html,encoding='utf8')
 model=read(MODEL/'model.json');model.update(candidate_batch_datasets=[dict(**reference(OUT/'batch_manifest.json'),
  frame_count=1048,point_count=sum(counts),programme_frame_range=[201,1248],status='awaiting_user_review')],
  candidate_source_frame_count=1048,candidate_source_point_count=sum(counts),candidate_user_acceptance_pending=True)
 if not REPAIR_MODE or MODEL!=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014':write(MODEL/'model.json',model)
 selection_path=PARENT.parents[1]/'primary_source_selection.json';selection=read(selection_path)
 selection['continuation'].update(source_frame_count_processed=1248,last_contiguous_processed_programme_frame=1248,
  last_contiguous_processed_pts=575488,last_pts=575488,batch_manifest=str(OUT/'batch_manifest.json'),
  user_accepted_frame_count=200,user_accepted_point_count=32883,frame_201_processed=True,user_accepted=False,
  current_review_frame_range=[201,1248],next_requested_preview_frame=None,source_framewise_model_path=str(MODEL/'model.json'),
  next_step='await_user_review_of_first_shot_remaining_frames_201_to_1248')
 selection['representation_verification']['first_shot_decoded_pixel_verification']=reference(SOURCE_DATA/'official_representation_first_shot_1_1248_verification.json')
 write(selection_path,selection)
 entry=ROOT/'START_HERE.html';text=entry.read_text(encoding='utf8');begin='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_BEGIN -->';end='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_END -->'
 relative=OUT.relative_to(ROOT).as_posix()
 block=f'''{begin}<section class="surface"><h3>冈田芽衣：第一个镜头取点完成</h3>
<p>官方50fps：第1–1248帧，源时间00:20.020–00:44.960。已验收1–200帧保持原样；新增201–1248帧共1048帧，等待审核，每帧{min(counts)}–{max(counts)}点，硬上限500。</p>
<p><a href="{EDITOR_URL}">打开可加点、删点的逐帧页面</a> · <a href="START_POINT_EDITOR.cmd">启动编辑器</a> · <a href="{relative}/whole_first_shot_review.html">完整第一镜头交付快照（1–1248帧）</a> · <a href="{relative}/batch_manifest.json">新增批次记录</a> · <a href="{SOURCE_OUT.relative_to(ROOT).as_posix()}/historical_manual_annotation_inventory.json">人工标注来源</a></p>
<p class="small">B1/B2/C等宽直线取宽度中线；靶圈取指定材料轮廓。广告沿围挡展开面建模。尚未重新求解相机，新增取点尚未由用户验收。</p></section>{end}'''
 require(text.count(begin)==1 and text.count(end)==1,'Review entry markers not unique')
 entry.write_text(re.sub(re.escape(begin)+r'[\s\S]*?'+re.escape(end),lambda _:block,text,count=1),encoding='utf8')
 accepted_integrity();return manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--indices');ap.add_argument('--merge-only',action='store_true');ap.add_argument('--validate-only',action='store_true');ap.add_argument('--publish-only',action='store_true')
 ap.add_argument('--repair',action='store_true');ap.add_argument('--model',type=Path);ap.add_argument('--prepare-sources',action='store_true');args=ap.parse_args()
 if args.repair:configure_repair(args.model)
 if args.prepare_sources:prepare_repair_sources();return
 indices=set(map(int,args.indices.split(','))) if args.indices else None
 if args.publish_only:print(json.dumps(publish(read(DATA/'model_entry_validation.json'))['density']));return
 if not args.validate_only:merge(indices)
 if args.merge_only:return
 result=validate(indices)
 if not indices and not args.validate_only:print(json.dumps(publish(result)['density']),flush=True)
if __name__=='__main__':main()
