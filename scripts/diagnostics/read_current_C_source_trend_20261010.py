"""Actual fixed C/C0 pixels against complete saved candidate material/camera."""
from pathlib import Path
import os, sys, json, pickle, time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry
HERE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUNS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
SEED=RUNS/'okada_full9_current_review71_clarabel_20261009_152520_411/accepted_states/27fa214561703310c832f3884656cf16c9461870a44ed3acf808bd866a4eeb47'
RUN=RUNS/'okada_review85_corrected_compact_20261010_2002'


def main():
    started=time.perf_counter()
    config=json.loads((ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_corrected_compact_20261010.json').read_text(encoding='utf8'))
    provider, world, proof=current_complete_source_geometry(SEED/'complete_state_UNACCEPTED.npz',config['source_options']['--world-init'])
    from z190_source_camera_diagnostic import absolute_value_jac15
    packet=json.loads((HERE/'nearboard49_original_source_mass_coarsening_evidence_20261010.json').read_text(encoding='utf8'))['original_packet']
    entries=pickle.loads(Path(packet['path']).read_bytes())['programme_absolute_rows']
    selected=[(i,e) for i,e in enumerate(entries) if e['collection_role']=='FIT' and e['group_actual_pts']==231678464 and e['source_row']['id'] in ('C','C0_outer')]
    for i,e in selected:provider.register_row(e['source_row'],actual_pts=e['group_actual_pts'])
    cycles=json.loads((RUN/'actual_accepted_cycle_performance.json').read_text(encoding='utf8'))['cycles']
    directories=[SEED]+[RUN/'accepted_states'/x['q_sha256'] for x in cycles]
    results=[]
    for directory in directories:
        identity=json.loads((directory/'state_identity.json').read_text(encoding='utf8'))
        with np.load(directory/'complete_state_UNACCEPTED.npz',allow_pickle=False) as z:
            slots=z['aux__current_shared_world_geometry_parameter_indices'].copy();q=z['q'].copy();w=q[slots]
        with np.load(identity['camera_path'],allow_pickle=False) as z:
            indices=np.flatnonzero(z['pts']==231678464)
            assert len(indices)==1
            camera=z['camera25'][indices[0]].copy()
        rows=[]
        for i,e in selected:
            original=e['source_row'];row=provider.rebind(original,w)
            residual=absolute_value_jac15(camera,[row],False)
            if isinstance(residual,tuple):residual=residual[0]
            raw=residual.reshape(-1,2)*np.asarray(row['sigma'])[:,None]
            rows.append(dict(entry_index=i,id=row['id'],operator=row.get('curve_observation_kind'),
                             source_native_UV=np.asarray(row['uv']).tolist(),sigma_px=np.asarray(row['sigma']).tolist(),
                             original_operator_outside_support_vector_px=raw.tolist(),
                             outside_support_mean_px=float(np.linalg.norm(raw,axis=1).mean()),
                             outside_support_maximum_px=float(np.linalg.norm(raw,axis=1).max())))
        results.append(dict(q_sha256=identity['q_sha256'],rows=rows))
    report=dict(schema='rinkspace.actual-C231678464-source-operator-trend.v1',original_packet=packet,
                exact_saved_complete_material_and_camera=True,original_UV_sigma_support_unchanged=True,
                is_raw_RGB_accuracy_verification=False,Ad4_same_exposure_source_rows=0,states=results,seconds=time.perf_counter()-started)
    path=HERE/'actual_C231678464_original_source_operator_trend_20261010.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'report':str(path),'seconds':report['seconds'],'states':[{'q':r['q_sha256'][:8],'rows':[{k:x[k] for k in ('entry_index','id','outside_support_mean_px','outside_support_maximum_px')} for x in r['rows']]} for r in results]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
