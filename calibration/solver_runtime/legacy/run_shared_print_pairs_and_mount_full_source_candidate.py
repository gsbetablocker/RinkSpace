"""Full-source candidate sharing print geometry and constant installation levers.

Source measurements, loss roles, surface qualifications, depth domain and
statistical roll covariance retain their authenticated parent definitions.
Nine lever and twelve physical print coordinates are explicitly added. Original photo search ranges
were working initializations, not measurements, and are not hard constraints.
"""
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import argparse, os, sys, json, hashlib
BASE=_runtime_data_root()
sys.dont_write_bytecode=True
for n in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[n]="1"
os.environ.update(RINKSPACE_MOTION_BACKEND="native",RINKSPACE_MOTION_THREADS="3",
    RINKSPACE_SCATTER_THREADS="3",RINKSPACE_SCATTER_BACKEND="native",
    RINKSPACE_PROJECTION_BACKEND="native",RINKSPACE_PROJECTION_VALUE_BACKEND="opencv",
    RINKSPACE_GEOMETRY_SPARSE_FD="1",RINKSPACE_GEOMETRY_FD_STEP_M="1e-7",
    RINKSPACE_PROFILE_MOTION_ONLY="1",RINKSPACE_PATTERN_WORKERS="4",RINKSPACE_SOLVER_ACCELERATION="auto")
from isolated_solver_bootstrap import install
install()
import numpy as np
from run_secondary_optics_candidate import load_graph, capture_driver
from installation_mount_candidate.mount_graph import install_mount_and_optics
from shared_advertisement_pairs_candidate import z190_joint_intercamera_fixed_xyz_paired_mount_source as driver
from correlated_roll_prior_candidate.explicit_bias_source import ExplicitBiasSource
from z190_joint_source_solver import make_profile_update, _frozen_profile_token, solve_joint_source
from z190_streaming_source_normal import accumulate_source_normal, replay_source_cost

DEFAULT_SEED=BASE/"current_reviewed_motion_ice_source_freezes/b69132266659/solver_snapshot_UNACCEPTED.npz"
EXPECTED_SEED_SHA="b691322666594ea90a41b94c9211b4f03b1c93077c80165ee54d8f9e880892c8"
OLD_TO_NEW=np.r_[np.arange(34195),np.arange(34204,34213),np.arange(34225,34228)]

def write(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),default=lambda x:
        x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding="utf8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qtoken(q):return hashlib.sha256(np.ascontiguousarray(q).view(np.uint8)).hexdigest()

def prepare(seed,out):
    """Authenticate the parent, restore its domains, then add only lever chart."""
    seed=Path(seed)
    if seed.resolve()!=DEFAULT_SEED.resolve() or sha(seed)!=EXPECTED_SEED_SHA:
        raise ValueError("This candidate requires the authenticated complete reviewed-material parent")
    original=BASE/"current_source_freezes/02c2309cbcd1/solver_snapshot_UNACCEPTED.npz"
    state,args,_,provenance=load_graph(original,out)
    install_mount_and_optics(state)
    with np.load(seed,allow_pickle=False) as z:template={k:z[k].copy() for k in z.files}
    if template["q"].shape!=(34207,) or state.q.shape!=(34213,):
        raise ValueError("Explicit parent41 to physical47 chart required")
    parent_review=json.loads((seed.parent/"latest_actual_source_review.json").read_text(encoding="utf8"))
    parent_event=parent_review["source_event_state_contract"]
    parent_stat=json.loads(str(template["aux__correlated_roll_prior_contract_json"].item()))
    parent_domain=json.loads(str(template["aux__current_depth_and_material_domain_contract_json"].item()))
    parent_material=json.loads(str(template["aux__qualified_original_motion_ice_contract_json"].item()))
    core=np.r_[template["q"][:34195],np.zeros(9),template["q"][34195:34204]]
    state.q=core
    make_profile_update(state.blocks).restore_checkpoint({k[9:]:v for k,v in template.items() if k.startswith("profile__")})
    if (parent_event["q_sha256"]!=qtoken(template["q"]) or
            parent_event["frozen_profile_sha256"]!=_frozen_profile_token(make_profile_update(state.blocks))):
        raise ValueError("Parent complete q/profile event authentication failed")
    from z190_continuous_source_bundle import SCENE_UNIT_M
    if getattr(state,"source_actors",None) is not None:raise ValueError("Domains precede actors")
    for block,domain in zip(state.blocks,parent_domain["blocks"]):
        if block.name!=domain["block"]:raise ValueError("Parent six-block order changed")
        kernel=block.bundle.motion_kernel
        old=float(kernel.minimum_source_depth_scene_units)*SCENE_UNIT_M
        expected=.01 if block.name=="low" else domain["candidate_minimum_source_Z_m"]
        if abs(old-expected)>1e-12:raise ValueError("Original native kernel depth domain changed")
        kernel.minimum_source_depth_scene_units=domain["candidate_minimum_source_Z_m"]/SCENE_UNIT_M
    from source_material_197_review.finite_ice_candidate import install_three_reviewed_motion_ice
    material=install_three_reviewed_motion_ice(state)
    for key in ("source_packet","source_motion_graph","original_selected_rows","source_material_review",
                "original_motion_raw_indices","global_bundle_indices","new_planes"):
        actual,expected=material[key],parent_material[key]
        equal=actual==expected
        if not equal and key in ('source_packet','source_motion_graph','source_material_review'):
            # Migration changes a file address, never the original evidence.
            # Authenticate the current bytes against the frozen source hash.
            equal=(isinstance(actual,dict) and isinstance(expected,dict) and
                {k:v for k,v in actual.items()if k!='path'}=={k:v for k,v in expected.items()if k!='path'} and
                sha(Path(actual['path']))==expected['sha256'])
        if not equal:raise ValueError("Parent actual qualified material changed: "+key)
    if material["candidate_frozen_profile_token"]!=parent_event["frozen_profile_sha256"]:
        raise ValueError("Zero lever chart failed to restore the same qualified frozen profiles")
    if [int(b.bundle.finite_mask.sum()) for b in state.blocks]!=[13,0,6,471,0,0]:
        raise ValueError("Original plus three reviewed finite surface inventory changed")
    from shared_advertisement_pairs_candidate.paired_ad_graph import install_paired_ad_geometry
    install_paired_ad_geometry(state)
    if state.q.shape!=(34225,) or len(state.geometry_parameter_indices)!=23:
        raise ValueError('Explicit physical34225 and common world23 required')
    args.initial_source_cost_fd_step=1e-7
    args.initial_source_fd_columns=None;args.initial_source_fd_evidence=None
    args.source_cost_fd_width_overrides={34210:1e-5}
    return state,args,template,dict(provenance=provenance,parent_event=parent_event,
        parent_stat=parent_stat,parent_depth=parent_domain,parent_material=parent_material,current_material_binding=material)

def make_source(state,args,template,binding):
    cap=capture_driver(driver,args,state)
    source=ExplicitBiasSource(cap,state);source.initial_bias(state.q)
    parent_stat=dict(binding["parent_stat"])
    expected=dict(parent_stat,original_core_parameter_count=34225,parameter_count=34228,
        statistical_bias_parameter_indices=[34225,34226,34227])
    if source.contract!=expected:raise ValueError("Roll covariance changed beyond explicit parameter reindexing")
    if not np.array_equal(source.quadratic,template["quadratic"]) or not np.array_equal(source.scales,template["scales"]):
        raise ValueError("Original complete source roles or scales changed")
    q=np.r_[state.q,template["q"][-3:]]
    if not np.array_equal(q[OLD_TO_NEW],template["q"]):raise ValueError("Zero lever parent coordinate remap failed")
    if _frozen_profile_token(source.updater)!=binding["parent_event"]["frozen_profile_sha256"]:
        raise ValueError("Profile changed during source capture")
    return cap,source,q

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=Path,default=DEFAULT_SEED)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(BASE.resolve()):parser.error("New own-task output required")
    state,solveargs,template,binding=prepare(args.seed,out)
    out.mkdir()
    cap,source,q=make_source(state,solveargs,template,binding)
    fixed=[] if state.external is None else [state.external_start+4*state.external.index[p]+c
        for p in state.external.held_pts for c in range(4)]
    shared=np.r_[cap["shared_schur_indices"],source.slots]
    if len(shared)!=62 or len(set(shared))!=62:raise ValueError("Complete62 shared chart required")
    manifest=dict(cap["manifest"],schema="rinkspace.complete-source-shared-print-pairs-and-mount-candidate.v1",
        statistical_roll_prior=source.contract,initial_physical_source_seed=str(args.seed),
        initial_physical_source_seed_sha256=sha(args.seed),parent_coordinate_remap=OLD_TO_NEW.tolist(),
        actual_shared_Schur_indices=shared.tolist(),inherited_correlated_roll_covariance_unchanged=True,
        inherited_depth_and_material_domain=binding["parent_depth"],
        inherited_source_material_qualification=binding["parent_material"],
        current_source_material_chart_binding=binding["current_material_binding"],
        explicit_shared_mount_candidate=state.mount_contract,
        explicit_shared_print_pairs_candidate=state.paired_ad_contract,extra_shared_physical_parameters=21,
        added_shared_mount_parameters=9,added_shared_print_geometry_parameters=12,
        old_print_geometry_was_estimated_not_measured=True,
        old_print_finite_polyline_domain_and_samples_preserved=True,
        original_objective_unchanged=False,source_data_and_loss_and_prior_unchanged=True,
        physical_camera_feasible_set_changed_by_shared_mount_estimation=True,
        common_print_geometry_candidate_domain_changed_explicitly=True,
        old_photo_lever_working_search_ranges_are_not_measurements=True,
        old_photo_lever_working_search_ranges_used_as_hard_constraints=False,
        physical_camera_parameter_count=source.core_count,statistical_nuisance_parameters=3,
        production_changed=False,accuracy_accepted=False)
    write(out/"input_manifest.json",manifest)
    write(out/"statistical_roll_prior_contract.json",source.contract)
    write(out/"shared_installation_mount_contract.json",state.mount_contract)
    write(out/"shared_print_pairs_contract.json",state.paired_ad_contract)
    write(out/"inherited_depth_and_material_domain_contract.json",binding["parent_depth"])
    write(out/"current_material_chart_binding.json",binding["current_material_binding"])
    def review(value,event=None):
        core,bias=source.unpack(value)
        cap["source_review"](core,event)
        for name in ("latest_actual_source_review.json","latest_candidate_scene_UNACCEPTED.json"):
            path=out/name;payload=json.loads(path.read_text(encoding="utf8"))
            original=dict(payload["source_event_state_contract"])
            payload["physical_core_event_state_contract"]=original
            payload["source_event_state_contract"]=dict(original,q_shape=list(value.shape),
                q_sha256=qtoken(value),statistical_parameter_tail_count=3)
            payload["statistical_roll_prior_candidate"]=dict(contract=source.contract,
                bias_radians=bias.tolist(),bias_degrees=np.rad2deg(bias).tolist(),old_primary_source_diagnostics_preserved=True)
            payload["inherited_depth_and_material_domain_candidate"]=binding["parent_depth"]
            payload["inherited_source_qualified_original_motion_ice_candidate"]=binding["parent_material"]
            payload["current_source_material_chart_binding"]=binding["current_material_binding"]
            write(path,payload)
    def evaluate(value,jacobian=True):
        from scipy.sparse import vstack
        blocks=list(source.blocks(value,jacobian));r=np.concatenate([np.asarray(b.r) for b in blocks])
        return (r,vstack([b.J for b in blocks],format="csr")) if jacobian else r
    try:
        source.guard(q);review(q,dict(stage="shared_print_pairs_and_mount_initial_unaccepted"))
        source.updater(q);token=_frozen_profile_token(source.updater)
        normal=accumulate_source_normal(lambda:source.blocks(q,True),len(q),
            expected_row_count=len(source.scales),frozen_profile_token=token,fixed_parameter_indices=fixed)
        checks=[]
        for col in shared:
            width=float(cap["args"].initial_source_cost_fd_steps.get(int(col),1e-7))
            plus,minus=q.copy(),q.copy();plus[col]+=width;minus[col]-=width
            source.guard(plus);source.guard(minus)
            cp=replay_source_cost(lambda:source.blocks(plus,False),normal.contract,frozen_profile_token=token)
            cm=replay_source_cost(lambda:source.blocks(minus,False),normal.contract,frozen_profile_token=token)
            fd=(cp-cm)/(2*width);analytic=float(normal.gradient[col])
            relative=abs(fd-analytic)/max(1.,abs(fd),abs(analytic))
            checks.append(dict(column=int(col),step=width,analytic=analytic,finite_difference=float(fd),relative_error=float(relative)))
            print(json.dumps(dict(stage="fresh_shared_print_pairs_and_mount_complete62_cost_direction",**checks[-1])),flush=True)
        write(out/"actual_global_source_cost_direction_checks.json",checks)
        if max(c["relative_error"] for c in checks)>.005:raise RuntimeError("Complete62 fresh source-cost direction gate failed")
        if _frozen_profile_token(source.updater)!=token:raise RuntimeError("Frozen profile changed during complete fresh gate")
        del normal
        arrays={k[5:]:v for k,v in template.items() if k.startswith("aux__")}
        # Keep inherited old chart arrays as provenance; current arrays are explicitly named.
        arrays["parent41_correlated_roll_prior_contract_json"]=arrays["correlated_roll_prior_contract_json"]
        arrays["correlated_roll_prior_contract_json"]=np.array(json.dumps(source.contract,ensure_ascii=False))
        arrays["current_shared_mount_parameter_indices"]=state.mount_parameter_indices
        arrays["current_shared_mount_contract_json"]=np.array(json.dumps(state.mount_contract,ensure_ascii=False))
        arrays["current_all_installation_optics_parameter_indices"]=state.optics_parameter_indices
        arrays["current_all_installation_optics_contract_json"]=np.array(json.dumps(state.optics_contract,ensure_ascii=False))
        arrays["current_source_material_chart_binding_json"]=np.array(json.dumps(binding["current_material_binding"],ensure_ascii=False))
        arrays["parent41_to_current62_coordinate_indices"]=OLD_TO_NEW
        arrays["current_shared_world_geometry_parameter_indices"]=state.geometry_parameter_indices
        arrays["current_shared_print_pairs_parameter_indices"]=state.paired_ad_parameter_indices
        arrays["current_shared_print_pairs_contract_json"]=np.array(json.dumps(state.paired_ad_contract,ensure_ascii=False))
        identity=dict(input_manifest=dict(path=str(out/"input_manifest.json"),sha256=sha(out/"input_manifest.json")),
            coherent_complete41_parent_state=dict(path=str(args.seed),sha256=sha(args.seed),
                source_event_state_contract=binding["parent_event"]),
            original_physical_core_and_source_binding=binding["provenance"],
            shared_mount_estimation_explicitly_added=True,shared_print_geometry_estimation_explicitly_added=True,
            inherited_source_surface_qualification_and_depth_and_covariance_unchanged=True)
        result=solve_joint_source(evaluate,q,source.quadratic,source.scales,out/"solver",source.guard,
            update_profiles=source.updater,checkpoint_callback=review,checkpoint_identity=identity,
            checkpoint_arrays=arrays,diagnostic_values=lambda value:dict(cap["diagnostics"](value[:source.core_count]),
                statistical_roll_bias_degrees=np.rad2deg(value[source.slots]).tolist()),
            global_schur_count=len(shared),global_schur_indices=shared,source_block_factory=source.blocks,
            fixed_parameter_indices=fixed,schur_backtracking=True,schur_view_lm_ratio=1.,
            schur_backtracking_aware_damping=True,cache_native_source_replay=True,source_stage_timing=True,initial_damping=.001)
        review(result.q,dict(stage="final_unaccepted",status=result.status,stop_reason=result.stop_reason))
        write(out/"actual_complete_solver_summary.json",dict(status=result.status,seconds=result.seconds,
            source_rows=len(source.scales),physical_camera_parameter_count=source.core_count,
            statistical_nuisance_count=3,accepted_steps=sum(bool(e["accepted"]) for e in result.history),
            inherited_covariance_unchanged=True,shared_mount_estimation_explicitly_added=True,
            shared_print_geometry_estimation_explicitly_added=True,accuracy_accepted=False))
    finally:
        actors=getattr(state,"source_actors",None)
        if actors is not None:actors.close()
        from z190_parallel_pattern_evaluation import close_pattern_pools
        close_pattern_pools()
if __name__=="__main__":main()
