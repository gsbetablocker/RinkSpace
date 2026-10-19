"""Read-only actual source/J attribution for a current shared goal crease.

These local directions are diagnostics, not alternate fitted states or new
geometry. Every reported nonlinear check uses the existing production source.
"""
from pathlib import Path
import argparse,json,os,sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.optimize import lsq_linear
from verify_current_native_primitives_20261007 import state_camera_adapter
from rinkspace.vision_camera_initialization import current_complete_source_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--ice-packet',type=Path,required=True)
    p.add_argument('--auxiliary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    arrays=dict(np.load(a.state,allow_pickle=False))
    identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    state,archive_error=state_camera_adapter(arrays,cameras,provider)
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    factor=NativeIceCurveFactor(state,a.ice_packet,len(arrays['q']))
    q=arrays['q'];block=factor.block(q,True)
    offsets=np.r_[0,np.cumsum([len(r['uv'])*2 for r in factor.rows])]
    goal=[(i,r)for i,r in enumerate(factor.rows)if r['curve']['kind']=='current_goal_crease']
    selected=np.concatenate([np.arange(offsets[i],offsets[i+1])for i,r in goal])
    raw=block.r[selected]*factor.scales[selected]
    J=block.J[selected].multiply(factor.scales[selected,None]).tocsr()
    wi=np.array(proof['actual_world_parameter_indices'],int)
    before=factor.review(q)['rows']
    names=provider.current.parameter_names
    cols_world=wi[[0,1,2,7,9]]
    cols_cameras=np.unique(np.concatenate([b.start+4*b.bundle.index[r['actual_pts']]+np.arange(4)
        for _,r in goal for b in [state.bypts[r['actual_pts']]]]))
    scales={int(c):.3 for c in cols_world}
    scales[int(wi[2])]=.005;scales[int(wi[9])]=.15
    for c in cols_cameras:scales[int(c)]=.005
    families={'hockey_center_yaw_goal_distance_radius':cols_world,
        'goal_distance_radius':wi[[7,9]],'crease_radius':wi[[9]],
        'source_camera_RF':cols_cameras,
        'shared_hockey_and_source_camera_RF':np.r_[cols_world,cols_cameras]}
    report=dict(schema='rinkspace.current-shared-crease-direction-attribution.v1',state_identity=identity,
        original_archive_reproduce_max=archive_error,world_proof=proof,
        source_unchanged=True,state_and_camera_not_written=True,whole_video_accuracy_accepted=False,
        baseline_source_rows=before,experiments=[])
    for name,columns in families.items():
        scale=np.array([scales[int(c)]for c in columns]);A=J[:,columns].toarray()*scale[None,:]
        singular=np.linalg.svd(A,compute_uv=False)
        # Minimum scaled norm linear diagnostic; one declared trust cap stops
        # a poorly observed family from inventing a large geometry excursion.
        bounded=lsq_linear(A,-raw,bounds=(-np.ones(len(columns)),np.ones(len(columns))),tol=1e-12,lsmr_tol=1e-12,max_iter=150)
        step=bounded.x
        cap=1.;delta=step*scale
        trial=q.copy();trial[columns]+=delta
        result=dict(name=name,columns=columns.tolist(),delta=delta.tolist(),
            physical_column_labels=[names[int(np.flatnonzero(wi==c)[0])]if c in wi else 'existing_source_RF_'+str(c) for c in columns],
            declared_diagnostic_step_scales=scale.tolist(),trust_cap_multiplier=1/cap,
            singular_values=singular.tolist(),baseline_goal_rms_native_px=float(np.sqrt(np.mean(raw**2))),
            predicted_goal_rms_native_px=float(np.sqrt(np.mean((raw+A@(step/cap))**2))))
        try:
            provider.validate_parameters(trial[wi])
            after=factor.review(trial)['rows']
            result['nonlinear_source_rows']=after
            result['largest_other_ice_mean_regressions']=sorted([
                dict(id=x['id'],pts=x['pts'],split=x['split'],feature_id=x['feature_id'],
                    before_mean_px=y['mean_native_px'],after_mean_px=x['mean_native_px'],delta_mean_px=x['mean_native_px']-y['mean_native_px'])
                for x,y in zip(after,before)if x['feature_id']!='goal_crease_left'],
                key=lambda x:x['delta_mean_px'],reverse=True)[:10]
        except ValueError as error:result['actual_domain_failure']=str(error)
        report['experiments'].append(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(a.output),experiments=[{k:v for k,v in e.items()if k not in ('nonlinear_source_rows','largest_other_ice_mean_regressions')}for e in report['experiments']]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
