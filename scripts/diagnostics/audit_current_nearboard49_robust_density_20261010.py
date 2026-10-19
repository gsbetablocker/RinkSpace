"""Evaluate only saved nearboard source R and two existing loss policies."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,os,pickle,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
RUNS=DATA/'calibration/runs/solver_acceleration_20261003'
SYS=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
HELP=RUNS/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics'),str(HELP)]
from audit_user_left_goal_line_and_ad_edges_candidate import load_state
from structured_programme_source_candidate.original_review_source import load_original_absolute_evaluators
from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
from calibration.solver_runtime.source_loss import SourceLossWeights,weighted_source_cost,source_irls_weights
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);a=ap.parse_args();began=time.perf_counter()
    config=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_painted_center_ad4_corrected_compact_q27fa_cold_20261010.json')
    proofpath=SYS/'nearboard49_original_source_mass_coarsening_evidence_20261010.json';proof=read(proofpath)
    original=pickle.loads(Path(proof['original_packet']['path']).read_bytes())['programme_absolute_rows']
    state,q,identity,error=load_state(a.state,Path(config['source_options']['--world-init']),read(a.state.parent.parent/'input_manifest.json'));assert error==0.
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={int(p):v.copy()for p,v in zip(z['pts'],z['camera25'])}
    evaluators,_=load_original_absolute_evaluators();budgets=defaultdict(float)
    for r in proof['records']:budgets[(r['original_locator']['block'],r['original_locator']['group_actual_pts'],r['source_identity'])]+=sum(r['retained_point_integration_mass'])
    checks=[];current_total=physical_total=0.;changed=0
    for r in proof['records']:
        e=original[r['original_entry_index']];old=e['source_row'];loc=r['original_locator'];pts=loc['group_actual_pts']
        state.geometry.register_row(old,actual_pts=pts)
        row=slice_observed_points(old,r['kept_original_point_indices']);bound=state.geometry.rebind(row,q[state.geometry_parameter_indices])
        residual=np.asarray(evaluators[loc['block']](cameras[pts],[bound],False));sigma=np.repeat(row['sigma'],2)
        multiplier=np.repeat(np.asarray(r['retained_original_prior_sampling_multiplier']),2)
        mass=np.repeat(np.asarray(r['retained_point_integration_mass'])/budgets[(loc['block'],pts,r['source_identity'])],2)
        quadratic=np.zeros(len(residual),bool)
        current=SourceLossWeights(multiplier**2*mass);physical=SourceLossWeights(mass,multiplier)
        current_cost=weighted_source_cost(residual,quadratic,current);physical_cost=weighted_source_cost(residual,quadratic,physical)
        current_total+=current_cost;physical_total+=physical_cost
        old_robust=abs(residual)>2.;new_robust=abs(residual*multiplier)>2.;changed+=int(np.count_nonzero(new_robust!=old_robust))
        current_coefficient=source_irls_weights(residual,quadratic,current);physical_coefficient=source_irls_weights(residual,quadratic,physical)
        checks.append(dict(original_packet_row_index=r['original_entry_index'],actual_pts=pts,old_points=len(old['uv']),active_points=len(row['uv']),
            stored_sigma_px_range=[float(np.min(row['sigma'])),float(np.max(row['sigma']))],physical_working_sigma_px=4.,
            current_component_Huber_pixel_threshold_range=[float(np.min(2*sigma)),float(np.max(2*sigma))],proposed_component_Huber_pixel_threshold=8.,
            component_error_px= (residual*sigma).tolist(),current_normalized_residual=residual.tolist(),
            current_robust_scalar_count=int(old_robust.sum()),physical_robust_scalar_count=int(new_robust.sum()),
            robust_classification_changed_scalars=int(np.count_nonzero(new_robust!=old_robust)),current_cost=current_cost,physical_sigma_cost=physical_cost,
            maximum_IRLS_coefficient_difference=float(np.max(abs(current_coefficient-physical_coefficient)))))
    result=dict(schema='rinkspace.actual-nearboard49-physical-sigma-vs-historical-density-Huber-review.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],complete_state=ref(a.state/'complete_state_UNACCEPTED.npz'),camera_archive=ref(identity['camera_path']),
        source_mass_evidence=ref(proofpath),actual_public_original_finite_source_residual_only=True,J_or_full_graph_or_solver_created=False,
        current_formula='N*w*Huber2(e/(4*sqrt(N)))',proposed_formula='w*Huber2(e/4)',
        proposal_reuses_existing_SourceLossWeights_working_normalization_multiplier=True,source_UV_sigma_WORLD_or_active_rows_changed=False,
        working_scale4_preserves_quadratic_information_budget=True,current_aggregate_cost=current_total,physical_sigma_aggregate_cost=physical_total,
        robust_classification_changed_scalars=changed,rows=checks,seconds=time.perf_counter()-began,accuracy_accepted=False)
    out=SYS/f'actual_nearboard49_robust_density_q{identity["q_sha256"][:8]}_20261010.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k in ['complete_q_sha256','current_aggregate_cost','physical_sigma_aggregate_cost','robust_classification_changed_scalars','seconds']}|{'report':str(out)}))
if __name__=='__main__':main()
