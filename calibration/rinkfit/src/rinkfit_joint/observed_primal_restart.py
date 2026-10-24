"""Prepare a lossless observed-row primal restart; never starts a solver.

The original model/packet replay their qualification and numerical transforms.
Native code then restores the complete candidate checkpoint and all four raw
auxiliary columns. Ceres/LM history is deliberately fresh. This command never
promotes observed-row state to true-RS/formal acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex

OBJECTIVE = "rinkfit.sourceUV_observed_row_finite_initialization.v1"
EXECUTION_KEYS = {
    "observed_primal_restart_v1", "observed_primal_warm_start_v1", "num_threads", "max_inner_iterations",
    "audit_accepted_cost_breakdown_v1", "function_tolerance", "parameter_tolerance", "gradient_tolerance", "initial_trust_region_radius", "periodic_camera_angle_rebase",
    "linear_solver_backend", "write_accepted_complete_checkpoint",
    "defer_complete_continuation_auxiliary_v1", "defer_auxiliary_minimization_to_joint_phase_v1",
    "write_accepted_physical_preview", "accepted_source_anchor_review",
    "source_uv_capture_construct_problem_only", "archived_dependency_aliases", "audit_group_gradients_v2", "audit_actual_directional_v1", "audit_actual_family_directional_v1", "audit_actual_gradient_locations_v1", "observed_polyline_kkt_working_set_v1", "audit_actual_solver_step_v1",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": sha256(path)}


def checked(ref):
    path = Path(ref["file"])
    if not path.is_absolute() or sha256(path) != ref["sha256"]:
        raise ValueError("Missing or changed absolute pinned file: " + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path).resolve()
    if len(str(path).encode("utf-16-le")) // 2 >= 260:
        raise ValueError("Prepared path exceeds Windows 259 UTF-16-unit limit")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return reference(path)


def semantic_config(cfg):
    return {k: v for k, v in cfg.items() if k not in EXECUTION_KEYS}


def validate_code_archive_extension(original, cfg):
    old_ref, new_ref = original.get("archived_dependency_aliases"), cfg.get("archived_dependency_aliases")
    if old_ref == new_ref:
        return False
    if old_ref is None or new_ref is None:
        raise ValueError("Code archive metadata must preserve the producer registry")
    old, new = read(checked(old_ref)), read(checked(new_ref))
    old_meta = {k: v for k, v in old.items() if k != "aliases"}
    new_meta = {k: v for k, v in new.items() if k != "aliases"}
    if (old.get("schema") != "rinkfit.exact_dependency_aliases.v1" or old_meta != new_meta
            or new["aliases"][:len(old["aliases"])] != old["aliases"]):
        raise ValueError("Code archive metadata must be an append-only exact extension")
    keys = set()
    for item in new["aliases"]:
        legacy, actual = Path(item["legacy_path"]), Path(item["actual_file"])
        key = (str(legacy.resolve()).casefold(), item["sha256"])
        if (not legacy.is_absolute() or not actual.is_absolute() or legacy.suffix != ".py"
                or actual.suffix != ".py" or legacy.resolve() == actual.resolve() or key in keys):
            raise ValueError("Only separate, unique, exact Python code archives are allowed")
        keys.add(key)
        checked({"file": str(actual), "sha256": item["sha256"]})
    return True


def validate_problem_change(original, cfg, warm, allowed_change=None):
    validate_code_archive_extension(original, cfg)
    old_semantics, new_semantics = semantic_config(original), semantic_config(cfg)
    upgrade = cfg.get("runtime_shared_parameter_stage_v2", {}).get("gpu_kernel_transition_v1")
    if upgrade:
        if not warm: raise ValueError("GPU implementation transition requires explicit stage")
        proof = read(checked(upgrade))
        if (proof.get("schema") != "rinkfit.same_objective_GPU_implementation_transition.v1" or proof.get("passed") is not True
            or proof.get("previous_ptx_sha256") != original["gpu_raw_uv"]["ptx_sha256"]
            or proof.get("current_ptx_sha256") != cfg["gpu_raw_uv"]["ptx_sha256"]
            or proof.get("shadow_receipt") != cfg["gpu_raw_uv"]["shadow_receipt"]):
            raise ValueError("GPU implementation transition lacks actual old/new identity")
        shadow = read(checked(proof["shadow_receipt"]))
        if shadow.get("passed") is not True or shadow.get("ptx_sha256") != proof["current_ptx_sha256"]:
            raise ValueError("GPU implementation shadow failed")
        a,b=deepcopy(original["gpu_raw_uv"]),deepcopy(cfg["gpu_raw_uv"])
        for item in (a,b):
            for key in ("ptx","ptx_sha256","shadow_receipt"): item.pop(key,None)
        if a!=b: raise ValueError("GPU transition changed cache or objective semantics")
        old_semantics.pop("gpu_raw_uv",None);new_semantics.pop("gpu_raw_uv",None)
    if warm:
        if allowed_change == "runtime_atlas_coordinate_gauge_only":
            if "runtime_atlas_gauge_v1" not in cfg or "runtime_atlas_gauge_v1" in original:
                raise ValueError("Atlas transition requires its first explicit gauge")
            allowed = ("runtime_atlas_gauge_v1", "audit_expected_free_physical_blocks", "joint6_atlas_gauge_transition")
        elif allowed_change == "explicit_source_withdrawal_v1":
            # Same physical graph and source values; only an audited subset can
            # leave the objective. All original raw rows and auxiliaries remain.
            reviewed_geometry = cfg.get("runtime_shared_parameter_stage_v2",{}).get("native_identity_revision")
            if reviewed_geometry:
                revision=read(checked(reviewed_geometry));qualification=read(checked(cfg["dense_source_qualification"]))
                if revision.get("schema")!="rinkfit.native_material_identity_revision.v1" or qualification.get("revision")!=reviewed_geometry or revision.get("source_rows_deleted")!=0: raise ValueError("Combined geometry repair lacks preserved native identity review")
            for name in (() if reviewed_geometry else ("engineering_support_budget_v2", "runtime_shared_parameter_stage_v2")):
                left, right = deepcopy(original.get(name)), deepcopy(cfg.get(name))
                if left is not None: left.pop("source_packet_sha256", None)
                if right is not None: right.pop("source_packet_sha256", None)
                if name == "runtime_shared_parameter_stage_v2":
                    for stage in (left, right):
                        if stage is None: continue
                        for key in ("observed_circle_station_navigation", "observed_finite_curve_station_navigation"): stage.pop(key, None)
                        for row in stage.get("blocks", []):
                            if "source_support" in row:
                                support = read(checked(row["source_support"]))
                                support.pop("source_packet_sha256", None)
                                row["source_support"] = support
                if left != right: raise ValueError("Withdrawal changed physical stage or budget: " + name)
            allowed = ("dense_source_qualification", "engineering_dense_source_weights_v1", "capture_certified_numeric_domain",
                       "engineering_support_budget_v2", "runtime_shared_parameter_stage_v2", "audit_expected_source_counts",
                       "runtime_source_curve_coverage_v1")
            if reviewed_geometry: allowed += ("audit_expected_free_physical_blocks",)
        elif allowed_change in ("explicit_engineering_stage_v2", "explicit_source_requalification_v1", "explicit_native_ad_append_v1"):
            if "engineering_support_budget_v2" not in cfg:
                raise ValueError("Explicit new engineering stage requires its budget identity")
            if original.get("static_surface_motion_v1") not in (None, cfg.get("static_surface_motion_v1")):
                raise ValueError("A warm stage cannot replace the original motion source set")
            allowed = ("runtime_atlas_gauge_v1", "audit_expected_free_physical_blocks", "joint6_atlas_gauge_transition", "engineering_support_budget_v2", "static_surface_motion_v1")
            if allowed_change in ("explicit_source_requalification_v1", "explicit_native_ad_append_v1"):
                allowed += ("dense_source_qualification", "engineering_dense_source_weights_v1", "capture_certified_numeric_domain", "audit_expected_source_counts", "runtime_source_curve_coverage_v1")
            if "runtime_shared_parameter_stage_v2" in cfg:
                allowed += ("runtime_shared_parameter_stage_v2", "temporary_runtime_fixed_physical_blocks", "temporary_runtime_fixed_physical_prefixes", "temporary_runtime_fixed_nuisance_shots")
        else:
            raise ValueError("Unsupported explicit stage transition")
        for item in (old_semantics, new_semantics):
            for name in allowed:
                item.pop(name, None)
    if old_semantics != new_semantics:
        raise ValueError("Objective/stage/source/domain configuration differs from producer")


def packet_row_count(ref):
    # Latest packets are immutable overlays; their denominator is inherited.
    # Full expansion and source gates still run in dense_joint_run/native.
    seen = set()
    while True:
        path = checked(ref)
        if ref["sha256"] in seen:
            raise ValueError("Cyclic packet inheritance")
        seen.add(ref["sha256"])
        packet = read(path)
        if packet.get("schema") != "rinkfit.joint_packet.v1":
            raise ValueError("Unexpected inherited packet schema")
        if "row_count" in packet:
            return int(packet["row_count"])
        ref = packet["dense_parent_packet"]


def checkpoint_header(path):
    # Cheap preparation check only; the native reader validates every parameter,
    # discrete chart, complete state token, and objective identity before Solve.
    with Path(path).open(encoding="utf-8") as stream:
        magic = stream.readline().strip()
        fingerprint = shlex.split(stream.readline().strip())
        outer = stream.readline().split()
        lam = stream.readline().split()
        nu = stream.readline().split()
        present = stream.readline().strip()
        token = shlex.split(stream.readline().strip())
        identity = shlex.split(stream.readline().strip())
    if (magic != "RINKFIT_AL_CHECKPOINT_V1" or len(fingerprint) != 1
            or len(outer) != 7 or float(outer[4]) != 0 or lam != ["0"]
            or nu != ["0"] or present != "1" or len(token) != 1
            or len(identity) != 6 or identity[5] != OBJECTIVE):
        raise ValueError("Not a complete observed-row engineering checkpoint")
    return {"fingerprint": fingerprint[0], "state_token": token[0],
            "source_packet_sha256": identity[0], "objective_id": identity[5]}


def validate_source_withdrawal(previous, current):
    from .joint_metrics import PacketArrays
    import numpy as np
    old, new = PacketArrays(checked(previous)), PacketArrays(checked(current))
    a, b = old.packet, new.packet
    if a["row_count"] != b["row_count"] or a["source_row_ids"] != b["source_row_ids"]:
        raise ValueError("Withdrawal changed raw source identity/count")
    for key in a["arrays"]:
        if key not in ("admission", "fit_role") and a["arrays"][key] != b["arrays"][key]:
            raise ValueError("Withdrawal changed original column " + key)
    for key in ("queries", "groups", "stroke_pairs", "dense_source_binding", "engineering_pixel_dependency_groups"):
        if a.get(key) != b.get(key): raise ValueError("Withdrawal changed source semantics " + key)
    oa = (old.load("admission") == 1) & (old.load("fit_role") == 1)
    na = (new.load("admission") == 1) & (new.load("fit_role") == 1)
    if np.any(na & ~oa) or not np.any(oa & ~na): raise ValueError("Expected strict withdraw-only source subset")
    n0 = int(a["dense_source_binding"]["row_start"])
    if not np.array_equal(oa[:n0], na[:n0]):
        # Earlier original-source rows can also have a wrong material identity.
        # Require the exact reviewed row set; retain their raw UV/PTS/operators.
        revision = read(checked(b["native_material_identity_revision"]))
        review = read(checked(revision["review"]))
        withdrawn = np.flatnonzero(oa & ~na).tolist()
        if (revision.get("schema") != "rinkfit.native_material_identity_revision.v1" or
                revision.get("parent_packet") != previous or revision.get("source_rows_deleted") != 0 or
                revision.get("withheld_rows") != withdrawn or
                review.get("schema") != "rinkfit.native_material_identity_review.v1" or
                review.get("qualification_uses_model_residual") is not False or
                sorted(item["raw_row"] for item in review["rows"]) != withdrawn):
            raise ValueError("Original-prefix withdrawal lacks exact native RGB identity review")
        checked(review["native_frame_extractor"])
        for item in review["rows"]:
            i = item["raw_row"]
            if (item["source_row_id"] != a["source_row_ids"][i] or item["uv"] != old.load("uv")[i].tolist() or
                    item["native_pts"] != int(old.load("native_pts")[i]) or
                    item["qualification"] != "unknown" or not item["native_images"] or not item["finding"]):
                raise ValueError("Reviewed original source identity differs")
            for image in item["native_images"]:
                checked(image)

    return int(np.sum(oa & ~na))



def validate_source_requalification(previous,current):
    from .joint_metrics import PacketArrays
    from .io_utils import read_json
    import numpy as np
    old,new=PacketArrays(checked(previous)),PacketArrays(checked(current));a,b=old.packet,new.packet
    if a['row_count']!=b['row_count']or a['source_row_ids']!=b['source_row_ids']:raise ValueError('Recovery changed raw source identities')
    for k in a['arrays']:
        if k not in ('admission','fit_role')and a['arrays'][k]!=b['arrays'][k]:raise ValueError('Recovery changed original '+k)
    for k in ('queries','groups','stroke_pairs','dense_source_binding','engineering_pixel_dependency_groups'):
        if a.get(k)!=b.get(k):raise ValueError('Recovery changed original source operator '+k)
    oa=(old.load('admission')==1)&(old.load('fit_role')==1);na=(new.load('admission')==1)&(new.load('fit_role')==1);restored=np.flatnonzero(na&~oa)
    delta=read_json(checked(b['source_requalification']))
    if delta['schema']!='rinkfit.dense_source_requalification_delta.v1'or delta['source_packet']!=previous or delta['selected_by_model_residual']is not False:raise ValueError('Recovery evidence binds another parent')
    rr=delta['restored_packet_rows'];verified=np.fromfile(checked(rr),dtype='<u8')
    if np.any(oa&~na)or not len(restored)or not np.array_equal(restored,verified):raise ValueError('Recovered source membership differs from actual native evidence')
    n0=int(a['dense_source_binding']['row_start'])
    if not np.array_equal(oa[:n0],na[:n0]):raise ValueError('Recovery changed original prefix')
    return len(restored)


def validate_source_append(previous,current):
    import numpy as np
    from .joint_metrics import PacketArrays
    old,new=PacketArrays(checked(previous)),PacketArrays(checked(current));a,b=old.packet,new.packet;n=int(a['row_count'])
    delta=b['native_ad_source_append']
    prefix = old
    if delta['parent_packet'] != previous:
        validate_source_withdrawal(previous, delta['parent_packet'])
        prefix = PacketArrays(checked(delta['parent_packet']))
        if 'native_material_identity_revision' not in prefix.packet: raise ValueError('Combined append lacks a reviewed prefix identity revision')
    if delta['schema']!='rinkfit.native_ad_source_append.v1' or delta['row_start']!=n or int(b['row_count'])!=n+delta['count']:raise ValueError('Native source append has a different parent/count')
    for k in ('source_row_ids','source_dataset_ids','source_family_ids'):
        if b[k][:n]!=a[k]:raise ValueError('Native ad append changed source IDs')
    for k in a['arrays']:
        if prefix.load(k).tobytes()!=new.load(k)[:n].tobytes():raise ValueError('Native source append changed its verified prefix '+k)
    for k in ('queries','stroke_pairs','independent_rows','dense_source_binding','source_station_domains'):
        if a.get(k)!=b.get(k):raise ValueError('Native ad append changed original operator '+k)
    if b['groups'][:len(a['groups'])]!=a['groups']:raise ValueError('Native ad append changed W')
    ev=read(checked(delta['evidence']))
    if ev['schema'] not in ('rinkfit.native_ad_source_readout_evidence.v1','rinkfit.native_surface_source_readout_evidence.v2') or ev['source_model_residual_used'] or not ev.get('visual_surface_identity_review_completed',ev.get('visual_board_identity_review_completed',False)):raise ValueError('Native ad source evidence missing')
    return n


def validate_material_append(previous, current):
    """Append source-confirmed contours without replacing any saved primal."""
    old, new = read(checked(previous)), read(checked(current))
    a, b = old["materials"], new["materials"]
    if ({k: v for k, v in old.items() if k != "materials"} !=
            {k: v for k, v in new.items() if k != "materials"} or
            len(b) <= len(a) or b[:len(a)] != a):
        raise ValueError("Material append must preserve every existing model field and material")
    labels = {m["label"] for m in a}
    for material in b[len(a):]:
        label = material["label"]
        if not label.startswith("Ad") or ".print." not in label or label in labels or not material["paths"]:
            raise ValueError("Only unique printed native contour identities may be appended")
        labels.add(label)
        for path in material["paths"]:
            g, ad = path.get("geometry_block", -1), path.get("advertisement_block", -1)
            if (path["kind"] != 2 or path.get("physical_point", False) or
                    g < 0 or ad < 0 or g >= len(old["blocks"]) or ad >= len(old["blocks"]) or
                    len(old["blocks"][g]["values"]) != 6 or len(old["blocks"][ad]["values"]) != 4 or
                    len(path["points"]) < 2 or not path.get("provenance")):
                raise ValueError("New contours must use existing free print/install coordinates")
            if not all(len(xy) == 2 and all(__import__("math").isfinite(v) for v in xy) for xy in path["points"]):
                raise ValueError("Nonfinite appended native print contour")
    return {"schema": "rinkfit.append_only_native_material_topology.v1",
            "parent_model": previous, "model": current,
            "existing_material_count": len(a), "appended_material_count": len(b) - len(a),
            "existing_physical_values_bounds_and_metadata_unchanged": True}


def validate_restart_manifest(cfg, input_model, source_packet):
    if (cfg.get("source_uv_capture_objective") != OBJECTIVE
            or cfg.get("dense_source_fresh_capture_v1") is not True
            or cfg.get("reinitialize_native_auxiliaries", True)
            or cfg.get("source_uv_capture_reseed_new_sources", False)):
        raise ValueError("Observed restart requires same qualified mode without reseeding")
    for forbidden in ("resume_checkpoint", "resume_engineering_checkpoint",
            "resume_engineering_outer_state", "resume_source_uv_capture_checkpoint",
            "source_capture_init", "source_uv_capture_source_append_v4",
            "source_uv_capture_source_append_v5"):
        if forbidden in cfg:
            raise ValueError("Conflicting restart entry: " + forbidden)
    warm = "observed_primal_warm_start_v1" in cfg
    if warm and "observed_primal_restart_v1" in cfg:
        raise ValueError("Ambiguous observed state import mode")
    key = "observed_primal_warm_start_v1" if warm else "observed_primal_restart_v1"
    manifest = read(checked(cfg[key]))
    if (manifest.get("schema") != ("rinkfit.observed_primal_warm_start.v1" if warm else "rinkfit.observed_primal_restart.v1")
            or manifest.get("objective_id") != OBJECTIVE
            or manifest.get("optimizer_history_restored") is not False
            or manifest.get("quality_accepted") is not False):
        raise ValueError("Unsupported restart manifest")
    original = read(checked(manifest["original_configuration"]))
    repair = warm and manifest.get("allowed_semantic_change") == "native_source_and_print_repair_v2"
    if repair:
        validate_code_archive_extension(original, cfg)
        if cfg.get("gpu_raw_uv") != original.get("gpu_raw_uv"):
            raise ValueError("Source repair changed GPU mathematics")
        proof = manifest["native_print_repair"]
        if (proof["model"] != manifest["input_model"] or proof["parent_model"] != manifest["model"]
                or proof["source_pixels_changed"] is not False or proof["original_physical_values_preserved"] is not True):
            raise ValueError("Missing explicit source/shape repair provenance")
        for source in proof["source_evidence"]: checked(source)
        if packet_row_count(source_packet) <= packet_row_count(manifest["producer_source_packet"]):
            raise ValueError("Complete raw source append missing")
    else:
        validate_problem_change(original, cfg, warm, manifest.get("allowed_semantic_change"))
    for label, requested in (("input_model", input_model), ("source_packet", source_packet)):
        if sha256(checked(requested)) != sha256(checked(manifest[label])):
            raise ValueError("Changed original producer input: " + label)
    source_delta = repair or manifest.get("allowed_semantic_change") in ("explicit_source_withdrawal_v1", "explicit_source_requalification_v1", "explicit_native_ad_append_v1")
    producer_packet = manifest["producer_source_packet"] if source_delta else source_packet
    if source_delta and not repair:
        validator = validate_source_append if manifest["allowed_semantic_change"] == "explicit_native_ad_append_v1" else validate_source_requalification if manifest["allowed_semantic_change"] == "explicit_source_requalification_v1" else validate_source_withdrawal
        validator(producer_packet, source_packet)
    if "material_append_topology" in manifest:
        if not source_delta or manifest["allowed_semantic_change"] != "explicit_native_ad_append_v1":
            raise ValueError("Material append requires the explicit native source append")
        if validate_material_append(manifest["model"], input_model) != manifest["material_append_topology"]:
            raise ValueError("Material append topology does not match its saved model")
    binding = read(checked(manifest["state_binding"]))
    nuisance = read(checked(manifest["nuisance_manifest"]))
    token = manifest["expected_state_token"]
    if (binding.get("schema") != "rinkfit.provisional_measured_row_initialization_state_binding.v1"
            or binding.get("objective_id") != OBJECTIVE or binding.get("state_token") != token
            or binding.get("source_packet_sha256") != producer_packet["sha256"]
            or binding.get("true_RS_verified") is not False
            or binding.get("formal_Query_closed") is not False
            or binding.get("initialization_only") is not True
            or binding.get("phase_II_allowed") is not False
            or binding.get("fresh_trueRS_v3_transition_required") is not True):
        raise ValueError("Candidate binding is not this unaccepted observed-row state")
    raw_rows = packet_row_count(producer_packet)
    if (nuisance.get("schema") != "rinkfit.joint_explicit_PHR_nuisance_state.v1"
            or nuisance["packet_sha256"] != producer_packet["sha256"]
            or nuisance["raw_rows"] != raw_rows):
        raise ValueError("Auxiliary source/denominator mismatch")
    for name, width in (("station.f64", 8), ("row.f64", 8), ("alpha.f64", 8), ("path.u32", 4)):
        ref = manifest["files"][name]
        path = checked(ref)
        expected = nuisance["files"][name]
        if (ref["sha256"] != expected["sha256"] or path.stat().st_size != expected["size_bytes"]
                or path.stat().st_size != raw_rows * width):
            raise ValueError("Complete raw auxiliary mismatch: " + name)
    checked(manifest["model"])
    header = checkpoint_header(checked(manifest["checkpoint"]))
    if header["state_token"] != token or header["source_packet_sha256"] != producer_packet["sha256"]:
        raise ValueError("Checkpoint header differs from bound state")
    return {"manifest": manifest, "header": header, "same_problem_restart": not warm, "native_full_round_trip_required": True}


def refresh_review_binding(cfg, prepared, intermediate_policy=None):
    if "accepted_source_anchor_review" not in cfg:
        raise ValueError("Source review configuration is required")
    review = read(checked(cfg["accepted_source_anchor_review"]))
    checked(review["baseline"])
    if intermediate_policy is not None:
        review["intermediate_policy"] = intermediate_policy
    for key, filename in (("policy", "dense_live_review.py"), ("guard", "dense_fixed_anchor_guard.py"), ("strata", "fixed_local_anchor_audit.py")):
        review[key] = reference(Path(__file__).with_name(filename))
    cfg["accepted_source_anchor_review"] = write_new(Path(prepared) / "review.json", review)


def prepare(spec_path, candidate_dir, executable, prepared_dir, output_dir,
            intermediate_policy="diagnostic", stage_config=None, verify_only=False, code_aliases=None, max_inner_iterations=None, source_withdrawal_packet=None, source_requalification_packet=None, source_append_packet=None, material_append_model=None):
    spec = read(spec_path)
    producer_cfg = read(checked(spec["configuration"]))
    cfg = read(stage_config) if stage_config else deepcopy(producer_cfg)
    cfg.pop("observed_primal_restart_v1", None)
    cfg.pop("observed_primal_warm_start_v1", None)
    cfg["source_uv_capture_construct_problem_only"] = bool(verify_only)
    if max_inner_iterations is not None:
        if isinstance(max_inner_iterations, bool) or int(max_inner_iterations) != max_inner_iterations or max_inner_iterations < 1:
            raise ValueError("max_inner_iterations must be a positive integer")
        cfg["max_inner_iterations"] = int(max_inner_iterations)
    if code_aliases:
        cfg["archived_dependency_aliases"] = reference(code_aliases)
    if cfg.get("source_uv_capture_objective") != OBJECTIVE:
        raise ValueError("Producer run was not the observed-row objective")
    candidate = Path(candidate_dir).resolve()
    prepared = Path(prepared_dir).resolve()
    output = Path(output_dir).resolve()
    if prepared.exists() or output.exists():
        raise FileExistsError("Preparation and run output must both be new directories")
    native = Path(executable).resolve()
    recipe = read(native.parent / "recipe.json")
    if sha256(native) != recipe["executable_sha256"]:
        raise ValueError("Native executable does not match build recipe")
    sources = {k.replace("\\", "/"): v for k, v in recipe["source_files"].items()}
    cpp = Path(__file__).resolve().parents[2] / "cpp"
    for name in ("src/joint_recovery.cpp", "include/rinkfit/joint_observed_primal_restart_v1.hpp"):
        if sources.get(name) != sha256(cpp / name):
            raise ValueError("Rebuild native executable with installed restart patch: " + name)
    producer_ref = spec["configuration"]
    original_packet = spec["packet"]
    if sum(bool(x)for x in (source_withdrawal_packet,source_requalification_packet,source_append_packet))>1:
        raise ValueError("Choose one explicit source membership change")
    if material_append_model and not source_append_packet:
        raise ValueError("Native material append requires its new source packet")
    if source_append_packet:
        if not stage_config: raise ValueError("Native ad append requires explicit new stage")
        spec["packet"]=reference(source_append_packet)
        spec["model"]=reference(material_append_model or candidate/"model.json")
    if source_requalification_packet:
        if not stage_config: raise ValueError("Recovery must be an explicit new stage")
        spec["packet"] = reference(source_requalification_packet)
        validate_source_requalification(original_packet, spec["packet"])
    if source_withdrawal_packet:
        if not stage_config: raise ValueError("Withdrawal must be an explicit new stage")
        spec["packet"] = reference(source_withdrawal_packet)
        validate_source_withdrawal(original_packet, spec["packet"])
    warm = stage_config is not None
    key = "observed_primal_warm_start_v1" if warm else "observed_primal_restart_v1"
    if cfg.get("runtime_atlas_gauge_v1"):
        name = "include/rinkfit/joint_atlas_gauge_v1.hpp"
        if sources.get(name) != sha256(cpp / name):
            raise ValueError("Native recipe lacks the installed atlas-gauge header")
    binding = read(candidate / "observed_row_initialization_state_binding.json")
    manifest = {
        "schema": "rinkfit.observed_primal_warm_start.v1" if warm else "rinkfit.observed_primal_restart.v1", "objective_id": OBJECTIVE,
        "original_configuration": producer_ref, "input_model": spec["model"],
        "source_packet": spec["packet"], "checkpoint": reference(candidate / "engineering_checkpoint.json"),
        "model": reference(candidate / "model.json"),
        "state_binding": reference(candidate / "observed_row_initialization_state_binding.json"),
        "nuisance_manifest": reference(candidate / "explicit_nuisance/manifest.json"),
        "files": {name: reference(candidate / "explicit_nuisance" / name)
                  for name in ("station.f64", "row.f64", "alpha.f64", "path.u32")},
        "expected_state_token": binding["state_token"],
        "optimizer_history_restored": False, "quality_accepted": False,
    }
    if material_append_model:
        manifest["material_append_topology"] = validate_material_append(manifest["model"], spec["model"])
    if warm:
        manifest["allowed_semantic_change"] = "explicit_native_ad_append_v1" if source_append_packet else "explicit_source_requalification_v1" if source_requalification_packet else "explicit_source_withdrawal_v1" if source_withdrawal_packet else ("explicit_engineering_stage_v2" if "engineering_support_budget_v2" in cfg else "runtime_atlas_coordinate_gauge_only")
        if source_withdrawal_packet or source_requalification_packet or source_append_packet: manifest["producer_source_packet"] = original_packet
        if source_append_packet:
            from .joint_metrics import PacketArrays
            expanded = PacketArrays(checked(spec["packet"])).packet
            intermediate = expanded["native_ad_source_append"]["parent_packet"]
            if intermediate != original_packet:
                manifest["reviewed_prefix_source_packet"] = intermediate
                manifest["reviewed_prefix_source_revision"] = PacketArrays(checked(intermediate)).packet["native_material_identity_revision"]
        manifest["changed_semantic_fields"] = sorted(k for k in set(semantic_config(producer_cfg)) | set(semantic_config(cfg)) if semantic_config(producer_cfg).get(k) != semantic_config(cfg).get(k))
    prepared.mkdir(parents=True, exist_ok=False)
    cfg[key] = write_new(prepared / "restart.json", manifest)
    refresh_review_binding(cfg, prepared, intermediate_policy)
    cfg["write_accepted_physical_preview"] = True
    validate_restart_manifest(cfg, spec["model"], spec["packet"])
    new_spec = deepcopy(spec)
    new_spec["native"] = reference(native)
    new_spec["configuration"] = write_new(prepared / "config.json", cfg)
    new_spec["output"] = str(output)
    spec_ref = write_new(prepared / "spec.json", new_spec)
    return {"prepared_spec": spec_ref, "solver_launched": False,
            "restart_semantics": "explicit new engineering stage from complete observed primal" if warm else "exact observed primal; fresh Ceres/LM history",
            "same_problem_restart": not warm, "graph_verification_only": bool(verify_only),
            "intermediate_policy": intermediate_policy, "final_quality_gate_unchanged": True,
            "true_RS_verified": False, "quality_accepted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--native", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--code-aliases", help="Append-only exact Python code archive registry; old source proofs remain byte-identical")
    p.add_argument("--verify-only", action="store_true", help="Construct the native graph and exercise GPU callbacks, then return before Ceres Solve")
    p.add_argument("--stage-config", help="Explicit new engineering stage; source identity and admission cannot change")
    p.add_argument("--max-inner-iterations", type=int, help="Execution budget only; omission preserves the producer value")
    p.add_argument("--intermediate-policy", choices=("strict", "diagnostic"), default="diagnostic")
    a = p.parse_args()
    print(json.dumps(prepare(a.spec, a.candidate, a.native, a.prepared, a.output,
                            a.intermediate_policy, a.stage_config, a.verify_only, a.code_aliases, a.max_inner_iterations), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
