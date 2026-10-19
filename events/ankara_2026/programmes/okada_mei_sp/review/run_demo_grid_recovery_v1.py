import os,sys,json,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
R=Path('D:/OneDrive/codex_projs/RinkSpace');W=Path('D:/CodexData/RinkSpace_Data/runs/okada_rec_1019');sys.path[:0]=[str(R/'src'),str(R/'calibration/rinkfit/src')]
from rinkfit_joint.io_utils import read_json,write_json,file_sha256
from rinkfit_joint.joint_solver import execute_explicit
for ready in (W/'demo_physical_init65_v1_exit.json',W/'build_v16_exit.json'):
 while not ready.exists():time.sleep(.5)
 if read_json(ready)['code']!=0:raise RuntimeError('Required production preparation failed: '+str(ready))
state=read_json(W/'demo_physical_init65_v1/ready_state.json');model=Path(state['model']['file']);packet=Path(state['packet']['file']);checkpoint=Path(state['initial_checkpoint']['file'])
for ref in (state['model'],state['packet'],state['initial_checkpoint']):
 if file_sha256(ref['file'])!=ref['sha256']:raise ValueError('Changed full initialization state')
spec=W/'screen_grid_probe_v1/navigation_spec_g9_unbound.json';nav=read_json(spec)
if nav['source_packet_sha256']!=state['packet']['sha256']:raise ValueError('Navigation/current qualified source identity mismatch')
cfg=read_json(W/'camera_g9_config.json')
for key in ('source_camera_mode_navigation','motion_navigation_restart','coupled_motion_navigation_restart','finite_line_navigation_restart','resume_engineering_outer_state','baseline_auxiliary_manifest'):
 cfg.pop(key,None)
m=read_json(model);fixed=[b['name'] for b in m['blocks'] if not(b['name'].startswith('rig/')or b['name'].startswith('shot/'))]
cfg.update(parameter_stage='optics',temporary_runtime_fixed_physical_prefixes=fixed,
 resume_engineering_checkpoint=str(checkpoint),reinitialize_native_auxiliaries=False,
 refine_ordinary_station_seeds=False,refine_query_navigation_seeds=False,initialization_only=False,
 direct_screen_grid_navigation_spec=str(spec),write_accepted_physical_preview=True,navigation_stage_only=True)
config=W/'demo_grid_recovery_v1_config.json';write_json(config,cfg)
write_json(W/'demo_grid_recovery_v1_start.json',dict(schema='rinkfit.actual_grid_guided_production_start.v1',model=state['model'],source_packet=state['packet'],complete_checkpoint=state['initial_checkpoint'],direct_screen_initial_state=nav['initial_screen_grid_state'],numerical_navigation_spec=dict(file=str(spec),sha256=file_sha256(spec)),S01_complete_baseline_restored=True,S01_Demo_values_copied=False,original_native_pixels_RS_physical_terms_present_from_first_iteration=True,GPU_raw_J_and_CUDA_CGNR_retained=True,old_failed_candidate_resumed=False,physical_candidate_is_separate_from_immutable_initial_grid=True))
print('Starting actual full G9 native-pixel + direct-grid numerical navigation on CUDA CGNR',flush=True)
raise SystemExit(execute_explicit(str(W/'build_v16/joint_fit.exe'),str(model),str(packet),str(W/'demo_grid_recovery_v1'),config=str(config),engineering_recovery=True))
