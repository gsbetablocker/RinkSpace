"""All original low unknown-depth FIT values, using frozen numeric ledgers.

No graph, J, optimizer or material assignment. Typed suffix points are
explicitly accounted for and excluded from the unknown-depth domain.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
import cv2,numpy as np
cv2.setNumThreads(1)
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
ADMISSION=DATA/'history/work_sessions/20261002_01a0fa4f-6592-7ab2-bc33-42ff983aff59/z190_low_fixed_mount_foreground_rf_v1/original_motion_foreground_admission.npz'
FINITE=DATA/'calibration/runs/solver_acceleration_20261003/prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory/finite_material_source_binding.json'
RETIREMENT=DATA/'calibration/runs/spin_source_long_baseline_20261005/actual50_jitter_source_diagnostic_20261005/spin_micro_broadcast_overlay_original_motion_role_retirement_SOURCE_ONLY_v1.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def stats(v):
    x=np.asarray(v);return dict(count=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))if len(x)else dict(count=0)
def shot(p):
    return 3 if 229671680<=p<229702144 else 5 if 229945856<=p<230369024 else 8 if 230898688<=p<231013376 else -1
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    from calibration.solver_runtime import install
    install()
    from z190_motion_native import native_motion
    z=arrays(a.state);identity=read(a.state.parent/'state_identity.json');assert sha(identity['camera_path'])==identity['camera_sha256']
    cameras=arrays(identity['camera_path']);source=arrays(ADMISSION);retirement=read(RETIREMENT)
    assert sha(ADMISSION)==retirement['original_foreground_admission_npz']['sha256']
    bi=list(z['aux__block_names']).index('low');assert bi==3
    lengths=z['aux__block_lengths']//4;lo=int(lengths[:bi].sum());stamps=z['aux__canonical_pts'][lo:lo+int(lengths[bi])];assert len(stamps)==1851
    active=source['active_original_global_index'];assert len(active)==434991
    records=read(FINITE);assert len(records)==274
    known={int(r['global_bundle_index'])for r in records}
    ice=json.loads(str(z['aux__qualified_original_motion_ice_contract_json'].item()));known.update(map(int,ice['global_bundle_indices']))
    assert len(known)==277 and all(np.array_equal(source['source_uv_a'][active[int(r['global_bundle_index'])]],r['xy0'])and
        np.array_equal(source['source_uv_b'][active[int(r['global_bundle_index'])]],r['xy1'])for r in records)
    unknown=np.array([i for i in range(len(active))if i not in known]);assert len(unknown)==434714
    retired={int(r['current_bundle_motion_row'])for r in retirement['records']};assert len(retired)==89 and not retired&known
    loss_active=np.array([int(i)not in retired for i in unknown])
    src=active[unknown];ia,ib=source['source_endpoint_index_a'][src],source['source_endpoint_index_b'][src]
    stamp_lookup={int(p):i for i,p in enumerate(cameras['exposure_representative_pts'])}
    actual=np.array([cameras['camera25'][stamp_lookup[int(p)]]for p in stamps]);rot=np.array([cv2.Rodrigues(c[3:6])[0]for c in actual])
    weights=z['profile__weights_3'];assert weights.shape==(435185,2)
    contract=json.loads(str(z['aux__unknown_depth_physical_domain_contract_json'].item()))
    unit=float(json.loads(str(z['aux__current_depth_and_material_domain_contract_json'].item()))['scene_unit_metres'])
    output_r=[];output_t=[]
    for begin in range(0,len(src),65536):
        end=min(begin+65536,len(src));ii=ia[begin:end];jj=ib[begin:end];ss=src[begin:end];ca,cb=actual[ii],actual[jj]
        r,checks,trace=native_motion(a=source['source_uv_a'][ss],b=source['source_uv_b'][ss],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
            ra=rot[ii],rb=rot[jj],ca=ca[:,:3],cb=cb[:,:3],pp=actual[0,8:10],k1=actual[0,11],
            sigma=source['original_sigma_xy'][ss],profile_weight=weights[unknown[begin:end]],planes=np.zeros((len(ss),4)),jacobian=False,
            da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
            minimum_source_depth_scene_units=1./unit,schur=True,return_trace=True)
        output_r.append(r.reshape(-1,2));output_t.append(trace)
        print(json.dumps(dict(stage='original_low_unknown_values',completed=end,total=len(src),elapsed_seconds=time.perf_counter()-start)),flush=True)
    residual,trace=np.vstack(output_r),np.vstack(output_t)
    axial=np.divide(1.-trace[:,1],trace[:,1],out=np.full(len(trace),np.inf),where=trace[:,1]!=0)*unit
    floor=np.isclose(axial,1.,atol=1e-9,rtol=0.);interior=trace[:,9]!=0;cuts=np.array([shot(int(p))for p in stamps[ia]])
    assert set(cuts)=={3,5,8} and not np.any(trace[:,12])
    rr=np.linalg.norm(residual,axis=1);per=[]
    for cut in (3,5,8):
        mask=(cuts==cut)&loss_active;f=mask&floor
        per.append(dict(shot=cut,actual_unknown_FIT_points=int(mask.sum()),engineering_one_metre_floor_active_points=int(f.sum()),
            floor_fraction=float(f.sum()/mask.sum()),interior_depth_points=int((mask&interior).sum()),
            infinite_depth_endpoint_points=int((mask&np.isinf(axial)).sum()),floor_residual_native_px=stats(rr[f]),all_unknown_residual_native_px=stats(rr[mask])))
    path=a.output/'original_low_all_unknown_depth_values_v1.npz'
    np.savez_compressed(path,current_original_prefix_bundle_index=unknown,original_full_low_bundle_row=src,
        source_pts=stamps[ia],target_pts=stamps[ib],axial_depth_working_m=axial,engineering_one_metre_floor_mask=floor,
        interior_depth_mask=interior,actual_FIT_loss_active_mask=loss_active,raw_native_residual_px=residual,shot=cuts)
    report=dict(schema='rinkspace.original-low-all-unknown-FIT-depth-domain-value-audit.v1',state=str(a.state),state_sha256=sha(a.state),q_sha256=identity['q_sha256'],
        source_foreground_admission=dict(path=str(ADMISSION),sha256=sha(ADMISSION)),typed_original_prefix_274=dict(path=str(FINITE),sha256=sha(FINITE)),
        three_further_original_prefix_ice_indices=ice['global_bundle_indices'],original_prefix_points=434991,actual_low_total_points=435185,
        typed_original_prefix_points=277,typed_appended_points=194,total_actual_finite_points=471,unknown_original_prefix_points=434714,
        retired_unknown_loss_rows_unchanged=89,actual_unknown_FIT_points=int(loss_active.sum()),
        engineering_one_metre_floor_active_FIT_points=int((floor&loss_active).sum()),floor_fraction_actual_unknown_FIT=float((floor&loss_active).sum()/loss_active.sum()),
        floor_fraction_archived_unknown_profile_buffers=float(floor.mean()),per_shot=per,unknown_depth_domain_contract=contract,
        domain_semantics='source optical-camera axial Z; 1m engineering hypothesis, stronger than mathematical positive-depth domain, unmeasured arena distance',
        floor_count_does_not_establish_real_foreground_distance_or_material_role=True,real_RGB_role_admission_not_inferred_from_profile_or_residual=True,
        numerical_values=dict(path=str(path),sha256=sha(path)),whole_source_graph_J_solver_or_video_not_started=True,
        source_UV_PTS_sigma_weights_planes_and_loss_retirement_unchanged=True,elapsed_seconds=time.perf_counter()-start)
    out=a.output/'original_low_all_unknown_depth_floor_check_v1.json';out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),actual_unknown_FIT=report['actual_unknown_FIT_points'],floor_active=report['engineering_one_metre_floor_active_FIT_points'],
        floor_fraction=report['floor_fraction_actual_unknown_FIT'],per_shot=per)),flush=True)
if __name__=='__main__':main()
