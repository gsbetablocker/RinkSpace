"""Prepare the actual representative-to-full-source native transition; never launch.

The old subset is restored and token-checked first. Native then reactivates only
previously qualified rows, restores original covariance/dependency groups, and
builds a fresh full-source GPU problem. This is still an unaccepted observed-row
primal and requires real complete-shot visual validation and true-RS transition.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from .observed_primal_restart import checked, read, reference, write_new, checkpoint_header, OBJECTIVE

KEY = "representative_full_source_transition_v1"
SCHEMA = "rinkfit.representative_to_full_source_transition.v1"


def validate(cfg, packet_ref, continuation=None):
    transition = read(checked(cfg[KEY]))
    saved = continuation or read(checked(cfg["representative_continuation_v1"]))
    selected = read(checked(cfg["representative_source_selection_v1"]))
    expected = {
        "schema": SCHEMA, "source_packet": packet_ref,
        "continuation": cfg["representative_continuation_v1"],
        "representative_selection": cfg["representative_source_selection_v1"],
        "raw_rows": selected["source_row_count"], "previous_fit_rows": selected["fit_count"],
        "full_fit_rows": selected["fit_count"] + selected["inspection_count"],
        "additional_fit_rows": selected["inspection_count"],
        "old_subset_exact_restore_required": True, "raw_four_auxiliaries_preserved": True,
        "source_identity_or_membership_changed": False, "optimizer_history_restored": False,
        "quality_accepted": False,
    }
    if transition != expected:
        raise ValueError("Full-source transition differs from pinned subset/source/state")
    if (cfg.get("source_uv_capture_objective") != OBJECTIVE
            or cfg.get("reinitialize_native_auxiliaries", True)
            or cfg.get("source_uv_capture_reseed_new_sources", False)
            or cfg.get("source_uv_capture_construct_problem_only", False)
            or cfg.get("initialization_only", False)
            or saved.get("complete_checkpoint") is not True
            or saved.get("source_packet_sha256") != packet_ref["sha256"]
            or selected.get("source_packet") != packet_ref
            or saved.get("representative_source_selection_v1") != cfg["representative_source_selection_v1"]):
        raise ValueError("Full-source transition requires this exact complete representative primal")
    return transition


def prepare(spec_path, candidate_dir, executable, prepared_dir, output_dir):
    spec = read(spec_path)
    cfg = read(checked(spec["configuration"]))
    if KEY in cfg:
        raise ValueError("Producer is already a full-source transition")
    candidate = Path(candidate_dir).resolve()
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if prepared.exists() or output.exists():
        raise FileExistsError("Full-source preparation and output must be new")
    selection_ref = cfg["representative_source_selection_v1"]
    selection = read(checked(selection_ref))
    binding_ref = reference(candidate / "observed_row_initialization_state_binding.json")
    binding = read(checked(binding_ref))
    cp_ref = reference(candidate / "engineering_checkpoint.json")
    header = checkpoint_header(checked(cp_ref))
    nuisance_ref = reference(candidate / "explicit_nuisance/manifest.json")
    nuisance = read(checked(nuisance_ref))
    if (header["source_packet_sha256"] != spec["packet"]["sha256"]
            or binding.get("state_token") != header["state_token"]
            or binding.get("source_packet_sha256") != spec["packet"]["sha256"]
            or binding.get("objective_id") != OBJECTIVE
            or binding.get("true_RS_verified") is not False
            or binding.get("initialization_only") is not True
            or nuisance.get("schema") != "rinkfit.joint_explicit_PHR_nuisance_state.v1"
            or nuisance.get("packet_sha256") != spec["packet"]["sha256"]
            or nuisance.get("raw_rows") != selection["source_row_count"]):
        raise ValueError("Candidate is not the complete current representative primal")
    for name, width in (("station.f64", 8), ("row.f64", 8), ("alpha.f64", 8), ("path.u32", 4)):
        item = nuisance["files"][name]
        file = candidate / "explicit_nuisance" / name
        checked({"file": str(file), "sha256": item["sha256"]})
        if file.stat().st_size != nuisance["raw_rows"] * width or file.stat().st_size != item["size_bytes"]:
            raise ValueError("Incomplete raw nuisance column: " + name)
    continuation = dict(schema="rinkfit.representative_complete_primal_continuation.v1",
        source_packet_sha256=spec["packet"]["sha256"], complete_checkpoint=True,
        model=reference(candidate / "model.json"), checkpoint=cp_ref, state_token=header["state_token"],
        representative_source_selection_v1=selection_ref, explicit_nuisance_manifest=nuisance_ref,
        source_state_binding=binding_ref, quality_accepted=False)
    # Both refs remain unchanged through the native subset restore. The explicit
    # transition is applied only after the native complete-state token matches.
    prepared.mkdir(parents=True, exist_ok=False)
    cfg["representative_continuation_v1"] = write_new(prepared / "continuation.json", continuation)
    cfg["source_uv_capture_construct_problem_only"] = False
    cfg["initialization_only"] = False
    transition = dict(schema=SCHEMA, source_packet=spec["packet"],
        continuation=cfg["representative_continuation_v1"], representative_selection=selection_ref,
        raw_rows=selection["source_row_count"], previous_fit_rows=selection["fit_count"],
        full_fit_rows=selection["fit_count"] + selection["inspection_count"],
        additional_fit_rows=selection["inspection_count"], old_subset_exact_restore_required=True,
        raw_four_auxiliaries_preserved=True, source_identity_or_membership_changed=False,
        optimizer_history_restored=False, quality_accepted=False)
    cfg[KEY] = write_new(prepared / "full_source.json", transition)
    validate(cfg, spec["packet"], continuation)
    new_spec = deepcopy(spec)
    new_spec.update(native=reference(executable), configuration=write_new(prepared / "config.json", cfg), output=str(output))
    spec_ref = write_new(prepared / "spec.json", new_spec)
    from .dense_joint_run import prepare_run
    preparation = prepare_run(new_spec)
    return dict(prepared_spec=spec_ref, raw_rows=transition["raw_rows"],
        previous_fit_rows=transition["previous_fit_rows"], full_fit_rows=transition["full_fit_rows"],
        additional_fit_rows=transition["additional_fit_rows"], solver_launched=False,
        native_command=preparation["command"], quality_accepted=False)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "candidate", "native", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.spec, args.candidate, args.native, args.prepared, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
