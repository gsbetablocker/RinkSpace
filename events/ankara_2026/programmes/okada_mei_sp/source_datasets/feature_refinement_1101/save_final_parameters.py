from pathlib import Path
import hashlib,json,numpy as np
source=Path(r'D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/solver_speed_validation/continue_1102/fit_completion/accepted_pixels_latest.npz')
compact=Path(r'D:/OneDrive/codex_projs/RinkSpace/events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.npz')
lib=Path(r'D:/CodexData/RinkSpace_Data/build/solver_continue_1102/rinkspace_core.dll')
for p in (compact,):
 assert len(str(p).encode('utf-16-le'))//2<260 and all(len(x.encode('utf-16-le'))//2<=255 for x in p.parts)
 assert not p.exists(),p
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(source,allow_pickle=False) as z:
 arrays={k:z[k] for k in z.files if k not in ('stations','source_rows')}
 assert str(arrays['native_library_provenance_sha256'])==sha(lib),'Snapshot/library mismatch'
 assert str(arrays['point_input_sha256'])=='ecb7012af725c94026ba85c9249e731e560c20d2f2189efd25cb28113512f5fb'
 assert np.array_equal(arrays['scene_values'][:6],arrays['board'])
 assert np.array_equal(arrays['scene_values'][6:152],arrays['layout'])
 assert np.isfinite(z['stations']).all() and len(z['stations'])==2213335
 for k in ('controls','rig','board','layout','scene_values'):assert np.isfinite(arrays[k]).all(),k
 assert np.max(np.abs(arrays['layout'][39:]))<=.01
 assert np.array_equal(arrays['rig'][:,37],np.repeat(arrays['rig'][0,37],3))
 np.savez_compressed(compact,**arrays)
 with np.load(compact,allow_pickle=False) as verify:
  for k,v in arrays.items():assert np.array_equal(verify[k],v),k
 print(json.dumps({'compact':str(compact),'compact_sha256':sha(compact),'full_state':str(source),'full_state_sha256':sha(source),'native_library':str(lib),'native_library_sha256':sha(lib),'readout_ms':(arrays['rig'][:,37]*1000).tolist(),'final_parameters_equal_full_state':True,'maximum_ice_deviation':float(np.max(np.abs(arrays['layout'][39:])))}))
