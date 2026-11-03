from pathlib import Path
import sys,json,numpy as np
root=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(root),str(root/'src')]
from rinkspace.viewer3d.data import ResultCatalog,load_npz
from rinkspace.viewer3d.prepare import prepare_variant,FullSurface
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
r=Path(r'D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/action_integration_1103');programme=root/'events/ankara_2026/programmes/okada_mei_sp';rows=[];s01_caches=[]
seed=load_npz(programme/'skate_s01_1103.npz');policy=json.loads((programme/'ice_soft_1103.json').read_text(encoding='utf-8'))['policy']
base=ResultCatalog(root,Path('D:/CodexData/RinkSpace_Data'),programme/'viewer3d_fullbody_1102.json');assert base.config['default_variant']=='skate_s01_1103';surface=FullSurface(base)
for name in base.config['comparison_variants']:
 path=prepare_variant(base,name,surface);s01_caches.append(dict(variant=name,exposures=len(base.pts),cache=str(path)))
for clip in ['jump_s02','step_s01','spin_s05']:
 c=ResultCatalog(root,Path('D:/CodexData/RinkSpace_Data'),programme/('viewer3d_action_'+clip+'_1103.json'));assert c.config['default_variant']=='A';surface=FullSurface(c)
 for arm in ['A','B']:
  params=load_npz(programme/('action_'+clip+'_'+arm+'_1103.npz'));fit=load_npz(r/clip/arm/'fit/candidate.npz');packet=load_npz(r/clip/arm/'fitted/result.npz')
  validate_checkpoint_identity(params,fit);np.testing.assert_array_equal(params['x'],fit['x']);np.testing.assert_array_equal(params['x'][:17],seed['x'][:17]);np.testing.assert_array_equal(params['active'],np.arange(69))
  assert json.loads(str(params['ice_soft_policy_json']))==policy;assert not bool(params['ice_is_delivery_gate']);assert digest(programme/str(params['initial_state_file']))==str(packet['initialization_sha256'])
  path=prepare_variant(c,arm,surface);meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'));state=load_npz(path/'state.npz');np.testing.assert_array_equal(state['pts'],packet['pts']);np.testing.assert_allclose(state['Q'],packet['Q_world_mid'],rtol=0,atol=1e-12)
  assert meta['no_foot_lock_or_ground_snapping'] and not meta['accuracy_accepted']
  rows.append(dict(clip=clip,arm=arm,source_exposures=len(c.pts),state_pts_and_cache_equal=True,full_body_69_columns_free=True,shared_s01_shoe_parameters_equal=True,final_parameter_identity_equal_to_fit=True,initial_state_file=str(programme/str(params['initial_state_file'])),cache=str(path),maximum_coco_reconstruction_error_m=meta['saved_coco_mid_maximum_reconstruction_error_m'],maximum_native_reconstruction_error_m=meta['saved_native22_mid_maximum_reconstruction_error_m']))
(r/'final_delivery_checks.json').write_text(json.dumps(dict(s01_default_preserved=True,whole_s01_caches=s01_caches,comparisons=rows,one_body_state=True,no_new_fit=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('FINAL_CHECKS_OK',len(rows),flush=True)
