"""Retire only the user-identified duplicate row-3 line in the repair model.

The canonical finite connection, all scene numbers and historical model files
remain byte-for-byte as previously recorded. No new endpoint is measured.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from rinkspace.packed_assets import materialize_project_asset, read_project_json
BASE=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014'
MODEL=BASE/'repair_20261014'
SOURCE=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014')
OLD='curling_centerline_row3';CURRENT='house_connection_F3_R3';OLD_TARGET='ice.'+OLD+':segment0';TARGET='ice.'+CURRENT+':segment0'
WIDE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/photo_review_20260923/Ankara2026_photo_review_Codex_bundle/input/evidence/wide_original_010100083.png')
DRAWING=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_lane_centrelines_original_20261005.png')
def read(path):return read_project_json(path, project_root=ROOT)
def sha(path):
 with materialize_project_asset(path, project_root=ROOT).open("rb") as stream:
  return hashlib.file_digest(stream, "sha256").hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 scene_path=MODEL/'scene.json';registry_path=MODEL/'feature_registry.json';metadata_path=MODEL/'model.json'
 scene=read(scene_path);registry=read(registry_path);metadata=read(metadata_path)
 receipt_path=MODEL/'duplicate_centerline_retirement_receipt.json'
 if not any(f['id']==OLD for f in scene['ice_features']):
  if not receipt_path.exists():raise ValueError('Duplicate absent without authenticated repair receipt')
  receipt=read(receipt_path)
  if receipt['scene_after_sha256']!=sha(scene_path) or receipt['registry_after_sha256']!=sha(registry_path):raise ValueError('Already-retired model has changed')
  print('Already retired; no files changed');return
 before_scene=deepcopy(scene);before_registry=deepcopy(registry)
 before={p.name:sha(p) for p in (scene_path,registry_path,metadata_path)}
 historical={p.name:sha(p) for p in (BASE/'scene.json',BASE/'feature_registry.json',BASE/'model.json')}
 duplicate=next(f for f in scene['ice_features'] if f['id']==OLD)
 canonical=deepcopy(next(f for f in scene['ice_features'] if f['id']==CURRENT))
 if OLD_TARGET not in registry['targets'] or TARGET not in registry['targets']:raise ValueError('Expected target topology changed')
 references=[];point_total=0
 for index in range(1,1249):
  folder=SOURCE/('opening_100_50fps' if index<=100 else 'continuation_101_200_50fps' if index<=200 else 'first_shot_remaining_50fps')/'frames'
  path=folder/('%06d.json'%index);packet=read(path)
  for f in packet['features']:
   n=sum(len(s.get('uv',[])) for s in f['segments']);point_total+=n
   if n and (f.get('target_id')==OLD_TARGET or f.get('world_curve')==OLD+':segment0'):references.append(dict(index=index,path=str(path)))
 if references:raise ValueError('Existing source points reference duplicate '+str(references))
 scene['ice_features']=[f for f in scene['ice_features'] if f['id']!=OLD]
 scene.setdefault('superseded_duplicate_ice_features',[]).append(deepcopy(duplicate))
 scene.setdefault('superseded_ice_feature_aliases',{})[OLD]=CURRENT
 correction=dict(schema='rinkspace.user-authorized-duplicate-physical-ice-feature-retirement.v1',
  date_local='2026-10-14',retired_feature_id=OLD,canonical_feature_id=CURRENT,
  reason='Both named the same user-confirmed F3-R3 material centreline; the duplicate retained a longer unrelated Lane3 core extent and was independently rendered as a ghost continuation',
  user_authorized_physical_topology_correction=True,canonical_geometry_parameters_changed=False,
  canonical_right_endpoint_remains_estimated_not_measured=True,new_terminal_source_observation_created=False,
  preserved_canonical_right_extent_current=canonical.get('right_extent_current'),
  wide_original=dict(path=str(WIDE),sha256=sha(WIDE),source_time='01:01:00.083'),
  user_drawing=dict(path=str(DRAWING),sha256=sha(DRAWING)),
  historical_source_alias_record_preserved=True,model_fitting_performed=False)
 scene.setdefault('physical_topology_corrections',[]).append(correction)
 retired_target=registry['targets'].pop(OLD_TARGET)
 registry.setdefault('retired_duplicate_targets',{})[OLD_TARGET]=dict(original_entry=retired_target,superseded_by=TARGET,
  active_for_ui_or_fitting=False,reason='same physical finite F3-R3 centreline; historical duplicate must not create a second selectable target')
 registry.setdefault('superseded_target_aliases',{})[OLD_TARGET]=TARGET
 active={f['id']:i for i,f in enumerate(scene['ice_features'])};pointer_changes=[]
 inherited=[]
 for record in registry['inherited_ice_features']:
  if record['id']==OLD:continue
  updated=deepcopy(record);updated['scene_pointer']='/ice_features/'+str(active[record['id']])
  if updated!=record:pointer_changes.append(dict(id=record['id'],before=record['scene_pointer'],after=updated['scene_pointer']))
  inherited.append(updated)
 registry['inherited_ice_features']=inherited
 # All other active feature objects, targets and canonical numeric geometry
 # must compare exactly, independently of JSON key/array index bookkeeping.
 if scene['ice_features']!=[f for f in before_scene['ice_features'] if f['id']!=OLD]:raise ValueError('Unrelated active scene change')
 if canonical!=next(f for f in scene['ice_features'] if f['id']==CURRENT):raise ValueError('Canonical finite geometry changed')
 if registry['targets']!={k:v for k,v in before_registry['targets'].items() if k!=OLD_TARGET}:raise ValueError('Unrelated target change')
 write(scene_path,scene);write(registry_path,registry)
 metadata['scene']['sha256']=sha(scene_path);metadata['feature_registry']['sha256']=sha(registry_path)
 metadata['repair_physical_topology_correction']=dict(**correction,receipt_path=str(receipt_path),
  prior_scene_sha256=before['scene.json'],prior_registry_sha256=before['feature_registry.json'])
 write(metadata_path,metadata)
 receipt=dict(schema='rinkspace.repair-model-exact-duplicate-retirement-receipt.v1',model_path=str(MODEL),
  scene_before_sha256=before['scene.json'],scene_after_sha256=sha(scene_path),registry_before_sha256=before['feature_registry.json'],registry_after_sha256=sha(registry_path),
  metadata_before_sha256=before['model.json'],metadata_after_sha256=sha(metadata_path),
  exact_scene_diff=dict(active_ice_features_removed=[OLD],active_ice_features_added=[],other_active_ice_features_changed=[],
   appended_keys=['superseded_duplicate_ice_features','superseded_ice_feature_aliases','physical_topology_corrections']),
  exact_registry_diff=dict(active_targets_removed=[OLD_TARGET],active_targets_added=[],other_active_targets_changed=[],
   inherited_feature_removed=OLD,derived_scene_pointer_index_updates=pointer_changes,
   appended_keys=['retired_duplicate_targets','superseded_target_aliases']),
  original_duplicate_feature=duplicate,canonical_feature_unchanged=canonical,correction=correction,
  original_all_1248_frame_source_packets_scanned=True,original_source_point_count=point_total,duplicate_point_references=references,
  canonical_endpoint_not_tied_to_lane_end=True,actual_terminal_endpoint_new_measurement=False,
  historical_model_sha256=historical,historical_model_files_unchanged=all(sha(BASE/name)==value for name,value in historical.items()))
 write(receipt_path,receipt);write(DATA/'duplicate_centerline_retirement_receipt.json',receipt)
 print(json.dumps(dict(scene_sha256=sha(scene_path),registry_sha256=sha(registry_path),source_points=point_total,duplicate_references=len(references),receipt=str(receipt_path)),ensure_ascii=False))
if __name__=='__main__':main()
