"""Replay source loss algebra on real saved C0 UV and production native J.

This is a fragment/algebra verification, not a programme fit or an accuracy
evaluation. The saved original world is intentionally retained; no geometry
or observation is invented. A camera PP perturbation exercises Huber tails.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.sparse import csr_matrix, vstack
from calibration.solver_runtime import install, status
install()
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal, replay_source_cost
from calibration.solver_runtime.source_loss import SourceLossWeights, source_irls_weights
from rinkspace.source_sampling import compile_curve_sampling

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
PACKET=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl')


def accumulate(r,J,scales,payload=None):
    block=SourceRowBlock(r,J,np.zeros(len(r),bool),scales,
        dict(family='real-original-C0-fragment',original_start=0,original_stop=len(r)),payload)
    return accumulate_source_normal(lambda:iter([block]),J.shape[1],expected_row_count=len(r),
        frozen_profile_token='no-profile-owned-by-curve-fragment'),block


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    models=json.loads((BASE/'camera_probe_contract_readonly.json').read_text(encoding='utf8'))['models']
    camera_path=Path(models['27fa']['camera_path'])
    with np.load(camera_path,allow_pickle=False) as z:
        at=np.flatnonzero(z['pts']==231678464)
        if len(at)!=1: raise ValueError('Exact saved original exposure camera required')
        camera=z['camera25'][at[0]].copy()
    bound=[row for row in pickle.loads(PACKET.read_bytes())['programme_absolute_rows']
        if row['collection_role']=='FIT' and row['group_actual_pts']==231678464 and row['source_row']['id']=='C0_outer']
    rows=[row['source_row']for row in bound]
    if [len(row['uv'])for row in rows]!=[7,1]: raise ValueError('Original split C0 inventory changed')
    records=[]
    for i,row in enumerate(rows):
        records.append(dict(id='C0_part'+str(i),split='fit',source_video_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
            actual_pts=231678464,physical_curve_id='C0_outer',operator_id=row['curve_observation_kind'],uv=row['uv'],
            stored_sigma_px=row['sigma'],working_sigma_px=row['sigma'],prior_sampling_multiplier=1.,
            sampling_kind=['measured']*len(row['uv']),point_integration_mass=np.ones(len(row['uv']))))
    plan=compile_curve_sampling(records)
    weights=np.concatenate([np.repeat(plan['rows'][record['id']]['point_integration_weights'],2)for record in records])
    scales=np.concatenate([np.repeat(row['sigma'],2)for row in rows])
    checks=[];arrays={}
    for label,pp_delta in (('saved_camera',0.),('source_fixed_PP_y_diagnostic_80px',80.)):
        observer=camera.copy();observer[9]+=pp_delta
        r,dense=absolute_value_jac15(observer,rows,True);J=csr_matrix(dense)
        payload=SourceLossWeights(weights)
        value,block=accumulate(r,J,scales,payload)
        IRLS=source_irls_weights(r,np.zeros(len(r),bool),payload)
        expectedH=dense.T@(IRLS[:,None]*dense);expectedg=dense.T@(IRLS*r)
        copies=64
        duplicate,_=accumulate(np.tile(r,copies),vstack([J]*copies,format='csr'),np.tile(scales,copies),SourceLossWeights(np.tile(weights/copies,copies)))
        cost_replay=replay_source_cost(lambda:iter([SourceRowBlock(r,None,block.quadratic,scales,block.row_ledger,payload)]),
            value.contract,frozen_profile_token='no-profile-owned-by-curve-fragment')
        checks.append(dict(label=label,native_original_pts=231678464,source_UV_counts=[7,1],sigma_px=4.,
            same_original_source_operator_and_UV_used=True,source_observations_unchanged=True,
            max_original_sigma_normalized_residual=float(abs(r).max()),Huber_tail_scalar_count=int((abs(r)>2).sum()),
            native_production_J_shape=list(dense.shape),weighted_source_cost=value.cost,trial_replay_cost=cost_replay,
            complete_duplicate_copies=copies,duplicate_cost_abs_difference=abs(value.cost-duplicate.cost),
            duplicate_gradient_max_abs_difference=float(abs(value.gradient-duplicate.gradient).max()),
            duplicate_H_max_abs_difference=float(abs(value.normal.toarray()-duplicate.normal.toarray()).max()),
            reference_gradient_max_abs_difference=float(abs(value.gradient-expectedg).max()),
            reference_H_max_abs_difference=float(abs(value.normal.toarray()-expectedH).max())))
        arrays[label+'_r']=r;arrays[label+'_J']=dense;arrays[label+'_integration']=weights
    # A representative 65,536-row production-J sparsity fragment. Grouping
    # payload construction and digest happen once, outside timed passes.
    tile=4096;r=np.tile(r,tile);J=vstack([J]*tile,format='csr');scales=np.tile(scales,tile)
    payload=SourceLossWeights(np.tile(weights,tile))
    timing=[]
    for mode,w in (('historical_None',None),('optional_integration',payload)):
        accumulate(r,J,scales,w)
        times=[]
        for _ in range(7):
            begin=time.perf_counter();accumulate(r,J,scales,w);times.append(time.perf_counter()-begin)
        timing.append(dict(mode=mode,rows=len(r),J_nnz=J.nnz,median_seconds=float(np.median(times)),samples_seconds=times,
            source_producer_time_excluded=True,group_policy_precomputed_once=True))
    report=dict(schema='rinkspace.real-source-curve-integration-algebra-replay.v1',production_solver_started=False,
        source_packet=dict(path=str(PACKET),sha256=sha(PACKET)),camera_file=dict(path=str(camera_path),sha256=sha(camera_path)),
        original_source_bindings=[dict(block=row['block'],group_actual_pts=row['group_actual_pts'],group_order=row['group_order'],row_order=row['row_order'])for row in bound],
        checks=checks,timing=timing,optional_to_None_median_ratio=timing[1]['median_seconds']/timing[0]['median_seconds'],
        production_operator='z190_source_camera_diagnostic.absolute_value_jac15: original scanline finite branch + horizontal extent, original optical support and sigma',
        explicit_limit='This verifies algebra on saved original finite world/C0 source rows and saved camera, not current complete geometry or accuracy. Uniform source-functional integration mass is a proof fixture, not a promoted programme policy. No complete solver, current full source profile cycle, persistent weighted checkpoint, or source guard migration was run.',
        runtime=status(),accuracy_accepted=False)
    (BASE/'curve_grouping_real_native_source_replay_20261010.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    np.savez(BASE/'curve_grouping_real_native_source_replay_20261010.npz',**arrays)
    print(json.dumps(dict(report_path=str(BASE/'curve_grouping_real_native_source_replay_20261010.json'),checks=checks,timing=timing,
        optional_to_None_median_ratio=report['optional_to_None_median_ratio']),ensure_ascii=False))


if __name__=='__main__':main()
