"""Endpoint add/drop source costs for one saved secondary shared-camera step.

Values only, one original 39,679-point motion block, no source J, fitting,
image decoding, source cache, or report file. The original RUN3 profile and
installed observation budgets remain fixed for all eight evaluated states.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time

import joint_absolute_source_context_20261011 as ctx
import joint_absolute_global_tail_context_20261011 as global_tail
import low_shot_camera_dependent_tail_values_20261011 as tails
import compare_complete_original_source_states_20261011 as motion_replay
from audit_production_motion_coverage import certify
from calibration.solver_runtime.source_loss import weighted_source_cost, loss_weight_arrays, SourceLossWeights
from z190_motion_native import native_motion

a, np = ctx.a, ctx.np
RUN3 = a.R / "okada_review86_q8_bounded_motion_nuisance_fix_20261011"
CHILD = "8dcd3b9104dd1281bfcb6c09e2f7edb62af4e2712351ec2e7acfba1c4ce94817"
FROZEN_PROFILE = "f3008d040fa4500af71b6d8ad0a4f13616613bf8f9963d7d178c7fd6241c0984"


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False, default=a.plain), flush=True)


def ref_checked(record):
    path = Path(record["path"])
    if tails.sha(path) != record["sha256"]:
        raise ValueError("Authenticated source changed: " + str(path))
    return path


def reconstruct_secondary(c, arrays, manifest):
    coverage = a.read(a.H / "current_motion_role_whole_programme_exact_947_20261011.json")
    previous = a.read(ref_checked(coverage["actual_runtime_manifest"]))
    for key in ("original_motion_far_board_role_correction", "original_motion_rows", "original_motion_edge_counts"):
        if previous[key] != manifest[key]:
            raise ValueError("Original motion role/count changed: " + key)
    expected = previous["persistent_source_replay_identity"]["actual_original_source_input_arrays"]
    authority = a.read(a.H / "actual_pan_pairs_original_FIT_source_endpoints_699_readonly_20261011.json")
    source_ref = next(r for r in authority["source_input_authorities"] if "source_ray_matches.npz" in r["path"])
    with np.load(ref_checked(source_ref), allow_pickle=False) as z:
        pairs = z["pair_pts"].copy(); counts = np.diff(z["offset"])
        first, second = z["source_uv"].astype(float), z["target_uv"].astype(float)
    names = arrays["aux__block_names"].tolist(); ordinal = names.index("secondary")
    lengths = arrays["aux__block_lengths"] // 4; begin = int(sum(lengths[:ordinal]))
    stamps = arrays["aux__canonical_pts"][begin:begin+lengths[ordinal]].copy()
    certify(stamps, expected["programme:secondary:stamps"], "secondary:stamps")
    index = {int(p): i for i, p in enumerate(stamps)}
    board = np.zeros(len(first), bool)
    board[manifest["original_motion_far_board_role_correction"]["original_global_bundle_indices_by_block"]["secondary"]] = True
    finite = board.copy()
    sources = [source_ref]
    for path in (
        Path("D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp/z190_finite_motion_source_inventory_v1/secondary_fullray_2_ice_source_admission.json"),
        a.R.parents[2] / "history/work_sessions/20261002_private-thread-00008/source_solver_audit/secondary_core_ice_current_review_v1/source_qualified_original_ice_motion_rows.json",
    ):
        packet = a.read(path); sources.append(a.ref(path))
        ids = [r["original_global_index"] for r in packet["rows"]] if "rows" in packet else [r["original_global_bundle_index"] for r in packet["records"] if r["status"] == "qualified_ice_z0"]
        finite[ids] = True
    source = dict(a=first, b=second,
        ia=np.repeat([index[int(p)] for p in pairs[:, 0]], counts),
        ib=np.repeat([index[int(p)] for p in pairs[:, 1]], counts),
        motion_sigma=np.full(2*len(first), 2.), finite_mask=finite,
        active_to_original=np.arange(len(first)))
    for field in ("a", "b", "ia", "ib", "motion_sigma", "finite_mask"):
        certify(source[field], expected["programme:secondary:"+field], "secondary:"+field)
    certify(np.flatnonzero(board), expected["programme:secondary:course_board_indices"], "secondary:board")
    certify(np.flatnonzero(finite & ~board), expected["programme:secondary:static_noncourse_finite_point_indices"], "secondary:ice")
    profile = arrays["profile__weights_"+str(ordinal)]
    if profile.shape != first.shape:
        raise ValueError("Original frozen motion profile shape changed")
    contract = json.loads(str(arrays["aux__current_depth_and_material_domain_contract_json"].item()))
    floor = next(r["candidate_minimum_source_Z_m"] for r in contract["blocks"] if r["block"] == "secondary") / contract["scene_unit_metres"]
    planes = np.zeros((len(first), 4)); planes[finite] = [0, 0, 1, 0]
    return dict(source=source, stamps=stamps, board=board, base_planes=planes, profile=profile, floor=floor,
                old=SimpleNamespace(native_motion=native_motion)), sources


def source_families(c):
    result = []
    for family, factor in (("additional_real_native_ice_curves", c.ice), ("additional_real_native_board_edges", c.board),
                           ("additional_real_native_board_material_points", c.material), ("additional_real_native_board_circles", c.circle)):
        view = factor._qualified_active_fit_view() if hasattr(factor, "_qualified_active_fit_view") else factor
        if view is None: view = factor
        rows = view.rows
        result.append(dict(family=family, evaluate=lambda q, f=factor, rows=rows: ctx.native_value(c, f, q, rows),
                           quadratic=np.ones(sum(2*len(r["uv"]) for r in rows), bool), loss=c.integration._payloads.get(family)))
    for ordinal, factor in enumerate(c.centres):
        result.append(dict(family="direct_manual_sparse_correlated_centre_secondary#"+str(ordinal),
            evaluate=lambda q, f=factor: f._residual(q, f.rows, cameras={p:ctx.camera(c,q,p) for p in f.blocks}) / f._manual_centre_mean_scale,
            quadratic=np.ones(sum(2*len(r["uv"]) for r in factor.rows), bool), loss=None))
    for item in global_tail.prepare_tails(c):
        result.append(dict(family=item["family"], evaluate=item["evaluate"], quadratic=item["quadratic"], loss=item["loss"]))
    auth = tails.SourceBytes()
    with tails.authenticated_track_constructor(auth):
        track = tails.track_module.NativeBoardTrackFactor(c.state, c.options["--additional-native-board-track-source"], len(c.q))
    stored, quad, _ = tails._saved(c, "additional_native_static_board_tracks")
    result.append(dict(family="additional_native_static_board_tracks", evaluate=lambda q:track.block(q,False).r, quadratic=quad, loss=None))
    persistent, evaluate, _, _ = tails._persistent(c, auth)
    _, quad, _ = tails._saved(c, "additional_real_native_persistent_ice_long_baseline")
    result.append(dict(family="additional_real_native_persistent_ice_long_baseline", evaluate=evaluate, quadratic=quad, loss=None))
    owner = SimpleNamespace(c=c, families=[])
    tails._add_cross(owner, auth)
    for item in owner.families:
        result.append(dict(family=item["family"], evaluate=item["evaluate"], quadratic=item["quadratic"], loss=None))
    auth.finish()
    # Source geometry / independent wide25 are held at each endpoint in an
    # add/drop experiment; neither reads programme installation optics/lever.
    result.append(dict(family="strong_high_wide_source", evaluate=None, unchanged_by_columns=True))
    result.append(dict(family="soft_nominal_geometry_engineering_references", evaluate=None, unchanged_by_columns=True))
    return result


def original_absolute(c, value):
    total = 0.; scalars = 0; rows = 0
    ledger=c.ledgers["programme:secondary"][0]["original_row_ledger"]
    begin=int(ledger["original_start"])+2*39679
    for index, item in enumerate(c.originals):
        if item["block"] != "secondary" or item["collection_role"] != "FIT": continue
        row, keep = c.current_original(index)
        if not len(keep): continue
        size=2*len(keep)
        quadratic=c.arrays["quadratic"][begin+scalars:begin+scalars+size]
        scalars += size; rows += 1
        retired = (int(item["group_actual_pts"]) == 230445824 and item["group_order"] == 131 and item["row_order"] == 2)
        if retired: continue
        r = ctx.original_value(c, index, value)
        total += weighted_source_cost(r, quadratic, ctx.original_loss(c,index))
    return total, dict(operator_rows=rows, scalar_rows_including_retired_two=scalars,
        original_quadratic_scalars=int(c.arrays["quadratic"][begin:begin+scalars].sum()))


def evaluate(c, motion, value, families, before_pp):
    course, world, _ = c.provider.print_course_binding(value[c.world_indices])
    border = np.asarray(course.sample_perimeter(world, with_jacobian=False)[0])
    raw, checks = motion_replay.motion_values(c,motion,value,border)
    sigma = motion["source"]["motion_sigma"].reshape(-1,2)
    normalized = raw / sigma
    costs = {"secondary_motion": weighted_source_cost(normalized.ravel(), np.zeros(normalized.size,bool), None)}
    for name, mask in (("secondary_motion_board",motion["board"]), ("secondary_motion_ice",motion["source"]["finite_mask"] & ~motion["board"]), ("secondary_motion_unknown",~motion["source"]["finite_mask"])):
        z = normalized[mask].ravel(); costs[name] = weighted_source_cost(z,np.zeros(len(z),bool),None)
    costs["secondary_original_absolute"], proof = original_absolute(c,value)
    roll = c._global_tail_roll; inst = c.state.ids[1]; bias = int(c.state.structured_programme_coordinate_chart.bias_indices[1])
    positions = c.state.bypts[int(motion["stamps"][0])].start + 4*np.arange(len(motion["stamps"]))
    import cv2
    angles=[]
    for p in positions:
        R=cv2.Rodrigues(value[p:p+3])[0];x,y=R[0,2],-R[1,2]
        if x*x+y*y < 1e-8: raise ValueError("Original roll domain")
        angles.append(np.arctan2(x,y))
    rr=(np.asarray(angles)-value[bias])/roll["component_sigma_radians"]
    costs["secondary_roll_engineering_prior"] = weighted_source_cost(rr,np.ones(len(rr),bool),None)
    costs["programme:secondary"] = costs["secondary_motion"]+costs["secondary_original_absolute"]+costs["secondary_roll_engineering_prior"]
    for item in families:
        if item["evaluate"] is None: continue
        r=np.asarray(item["evaluate"](value),float)
        costs[item["family"]]=weighted_source_cost(r,item["quadratic"],item["loss"])
    calibration = next(b for b in c.state.blocks if b.name=="secondary").bundle.calibration
    reference=calibration.field_settings["reference_optical_height_prior"]
    ref=int(reference["reference_exposure_pts"])
    h=(ctx.camera(c,value,ref)[2]*c.provider.scene_unit_m-float(reference["target_above_ice_working_m"]))/float(reference["sigma_m"])
    costs["optical_height:"+inst]=.5*h*h
    columns=np.asarray(c.manifest["principal_point_engineering_prior"]["principal_parameter_indices"],int)
    pp=np.asarray(before_pp,float)+(value[columns]-c.q[columns]).reshape(3,2)
    costs["explicit_optics_engineering_working_prior"] = float(.5*np.sum(((pp-[960,540])/32.)**2))
    costs["secondary_PP2_engineering_prior"] = float(.5*np.sum(((pp[1]-[960,540])/32.)**2))
    return costs, proof


def main():
    started=time.perf_counter();c=ctx.load_context();emit("context_loaded",seconds=c.load_seconds)
    path=RUN3/"accepted_states"/CHILD/"complete_state_UNACCEPTED.npz"
    identity=a.read(path.parent/"state_identity.json"); manifest=a.read(RUN3/"input_manifest.json")
    if tails.sha(path)!=identity["state_npz_sha256"] or identity["frozen_profile_token"]!=FROZEN_PROFILE:
        raise ValueError("RUN3 frozen state/profile changed")
    with np.load(path,allow_pickle=False) as z:
        keys=("q","aux__block_names","aux__block_lengths","aux__canonical_pts","aux__current_depth_and_material_domain_contract_json","aux__current_evaluation_to_canonical_parameter_indices")
        arrays={k:z[k].copy() for k in keys};ordinal=z["aux__block_names"].tolist().index("secondary")
        arrays["profile__weights_"+str(ordinal)]=z["profile__weights_"+str(ordinal)].copy()
    if not np.array_equal(arrays["aux__current_evaluation_to_canonical_parameter_indices"], c.state.canonical_parameter_layout.evaluation_to_canonical):
        raise ValueError("Complete q chart changed")
    q1=c.state.evaluation_q(arrays["q"])
    motion, refs=reconstruct_secondary(c,arrays,manifest)
    families=source_families(c)
    block=next(b for b in c.state.blocks if b.name=="secondary")
    groups={"PPx_PPy":np.asarray(block.optics_chart.indices[:2],int),"k1":np.asarray(block.optics_chart.indices[2:],int),"leverXYZ":np.asarray(block.optics_chart.mount_indices,int)}
    if len(np.unique(np.concatenate(list(groups.values())))) != 6: raise ValueError("Shared groups overlap")
    pp=a.read(RUN3/"actual_principal_point_working_prior_initial.json")["actual_native_principal_points_px"]
    base={};states={}
    for endpoint,q in (("q8",c.q),("q8dcd",q1)):
        base[endpoint], proof=evaluate(c,motion,q,families,pp)
        emit("endpoint",endpoint=endpoint,costs=base[endpoint],original_absolute_inventory=proof)
    original=a.read(RUN3/"latest_initial_J_family_timings.json")["families"]
    trial=a.read(RUN3/"actual_trial_family_cost_pass_0001.json")
    if trial["frozen_profile_token"]!=FROZEN_PROFILE: raise ValueError("Cost pass profile differs")
    for label,records in (("q8",original),("q8dcd",trial["families"])):
        expected={r.get("family",r.get("original_row_ledger",{}).get("family")):r.get("actual_normalized_cost",r.get("normalized_cost")) for r in records}
        centres=[r for r in records if r.get("family",r.get("original_row_ledger",{}).get("family"))=="direct_manual_sparse_correlated_centre_secondary"]
        for ordinal,record in enumerate(centres):
            expected["direct_manual_sparse_correlated_centre_secondary#"+str(ordinal)]=record.get("actual_normalized_cost",record.get("normalized_cost"))
        for family,cost in base[label].items():
            if family not in expected: continue
            if abs(cost-expected[family])>2e-8: raise ValueError("Original family cost mismatch: "+str((label,family,cost,expected[family])))
    emit("original_family_baselines_authenticated",profile=FROZEN_PROFILE,motion_points=len(motion["source"]["a"]),
         exact_source_refs=refs,groups={k:v.tolist() for k,v in groups.items()},same_sigma_profile_budget=True)
    for name,columns in groups.items():
        for mode,endpoint,other in (("add","q8",q1),("drop","q8dcd",c.q)):
            q=(c.q if endpoint=="q8" else q1).copy();q[columns]=other[columns]
            costs,_=evaluate(c,motion,q,families,pp)
            delta={family:cost-base[endpoint][family] for family,cost in costs.items()}
            states[mode+":"+name]=delta
            emit("conditional_source_cost_delta",mode=mode,endpoint=endpoint,parameter_group=name,columns=columns.tolist(),
                 evaluation_parameter_change=(q[columns]-(c.q if endpoint=="q8" else q1)[columns]).tolist(),delta=delta,
                 whole_family_not_one_frame=True,conditional_not_additive=True)
    emit("complete",seconds=time.perf_counter()-started,controls=6,source_J=False,fitting=False,image_decode=False,image_writes=0,report_files_written=0,
         actual_family_costs_not_source_gradient_or_unique_cause=True,original_guard_feasibility_not_claimed=True)


if __name__=="__main__": main()
