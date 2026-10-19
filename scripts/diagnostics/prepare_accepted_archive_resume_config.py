"""Authenticate a saved accepted archive and prepare the original CLI inputs.

This reads JSON and NPZ only. It does not construct a source graph or solve.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_checkpoint_history

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def profile_token(stored, keys=None):
    digest = hashlib.sha256(b"rinkspace.frozen-profile-no-copy.v1")
    for key in stored.files if keys is None else keys:
        if key.startswith("profile__"):
            value = stored[key]
            digest.update(key[9:].encode("utf8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            if value.size:
                digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--accepted-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.report.exists(), "Preserve existing configuration/report")
    config = read_json(args.input_config)
    require(config["schema"] == "rinkspace.reusable-solver-inputs.v1" and config["driver"] == "named66", "Original named66 configuration required")
    directory = args.accepted_directory.resolve()
    checkpoint = directory / "complete_state_UNACCEPTED.npz"
    truth_path = directory / "truth_restoration_state.json"
    identity = read_json(directory / "state_identity.json")
    truth = read_json(truth_path)
    checkpoint_sha = file_sha(checkpoint)
    require(checkpoint_sha == identity["state_npz_sha256"], "Accepted checkpoint identity changed")
    require(truth["complete_state_reference"]["sha256"] == checkpoint_sha, "Truth state references a different checkpoint")
    require(Path(truth["complete_state_reference"]["path"]).resolve() == checkpoint, "Truth checkpoint path changed")
    input_options = config["source_options"]
    initial_truth = read_json(input_options["--truth-restoration-state"])
    resolved_parent_truth=None
    if '--truth-restoration-reviewed-manual-append-audit' in input_options:
        # A completed source append already wrote its actual target-bound
        # initial state. Reuse that authenticated result, rather than trying
        # to apply the old same-q append certificate to a later accepted q.
        parent_manifest=Path(truth['source_input_manifest_reference']['path'])
        resumed=read_json(parent_manifest.parent/'actual_truth_restoration_state_resume.json')
        resolved_parent_truth=Path(resumed['path'])
        require(file_sha(resolved_parent_truth)==resumed['sha256'],
                'Actual parent source append restoration identity changed')
        initial_truth=read_json(resolved_parent_truth)
        require(initial_truth['source_target']==truth['source_target'],
                'Resume must preserve the actual parent mathematical source target')
    with np.load(checkpoint, allow_pickle=False) as stored, np.load(input_options["--camera-seed"], allow_pickle=False) as initial:
        require(stored.files == initial.files, "Complete original NPZ field inventory changed")
        require(all(key in stored.files for key in ("schema", "q", "initial_q", "quadratic", "scales",
                "damping", "history_json", "reviews_json", "next_line_search_fraction")), "Complete original optimizer fields required")
        profile_keys = [key for key in stored.files if key.startswith("profile__")]
        require(profile_keys == [f"profile__{name}_{i}" for i in range(6) for name in ("weights", "present", "robust")], "All eighteen original profiles required")
        # NPZ members are decompressed on every access. Read numerical members
        # once; stream history/reviews rather than materialize growing scalars.
        fields, values, parent_values, profiles = [], {}, {}, {}
        metadata = {"q", "initial_q", "damping", "next_line_search_fraction", "history_json",
                    "reviews_json", "elapsed_seconds", "identity_json"}
        for key in stored.files:
            if key == "history_json":
                last, accepted_count, history_count = None, 0, 0
                for record in iter_checkpoint_history(checkpoint):
                    last = record
                    accepted_count += record.get("accepted") is True
                    history_count += 1
                require(last is not None and last["accepted"] is True, "Checkpoint is not an original accepted state")
                accepted_iteration = last["iteration"]
                fields.append(dict(name=key, shape=[], dtype_kind="U", history_records=history_count,
                    checkpoint_file_sha256=checkpoint_sha, authenticated_history_reader=True,
                    history_scalar_not_materialized=True))
                del last, record
                continue
            if key == "reviews_json":
                reviews_count = sum(1 for _ in iter_checkpoint_history(checkpoint, field=key))
                fields.append(dict(name=key, shape=[], dtype_kind="U", review_records=reviews_count,
                    checkpoint_file_sha256=checkpoint_sha, authenticated_reviews_reader=True,
                    reviews_scalar_not_materialized=True))
                continue
            value = stored[key]
            fields.append(dict(name=key, shape=list(value.shape), dtype=value.dtype.str, sha256=array_sha(value)))
            if key not in metadata or key in ("q", "initial_q"):
                original = initial[key]
                require(value.shape == original.shape and value.dtype == original.dtype,
                        "Original complete field shape/dtype changed: " + key)
                if key.startswith("profile__"):
                    if key.startswith("profile__weights_"):
                        require(value.ndim == 2 and value.shape[1] == 2,
                                "Original frozen motion profile shape changed")
                        require(np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)),
                                "Invalid frozen profile weights")
                    else:
                        require(value.shape == () and value.dtype == np.dtype(bool),
                                "Original profile presence/robust flag changed")
                    profiles[key] = value
                elif key in ("q", "initial_q"):
                    values[key], parent_values[key] = value, original
                else:
                    require(np.array_equal(value, original), "Original fixed source/auxiliary contract changed: " + key)
            elif key in ("damping", "next_line_search_fraction"):
                values[key] = value
        q = values["q"]
        q_sha = array_sha(q)
        require(q.shape == parent_values["q"].shape == (identity["parameter_count"],), "Complete q layout changed")
        require(initial_truth['complete_q_sha256']==array_sha(parent_values['q']),
                'Resolved parent truth state belongs to another initial q')
        require(np.isfinite(q).all(), "Nonfinite complete accepted q")
        require(q_sha == directory.name == identity["q_sha256"] == truth["complete_q_sha256"], "Accepted q/truth/archive identity disagreement")
        require(np.isfinite(values["initial_q"]).all() and
                (np.array_equal(values["initial_q"], parent_values["q"]) or
                 np.array_equal(values["initial_q"], parent_values["initial_q"])),
                "Initial q differs from both authenticated original start states")
        token = profile_token(profiles, profile_keys)
        require(token == identity["frozen_profile_token"], "Frozen profile token does not authenticate all eighteen arrays")
        damping, fraction = float(values["damping"]), float(values["next_line_search_fraction"])
        require(np.isfinite(damping) and damping > 0 and np.isfinite(fraction) and 0 < fraction <= 1, "Invalid original optimizer continuation seed")
        field_count = len(stored.files)
    progress_policy = truth.get("progress_policy", "historical_ratchet")
    require(progress_policy in ("historical_ratchet", "fixed_seed_envelope") and
            progress_policy == initial_truth.get("progress_policy", "historical_ratchet") and
            progress_policy == input_options.get("--truth-restoration-progress-policy", "historical_ratchet"),
            "Original truth restoration progress policy changed")
    if progress_policy == "fixed_seed_envelope":
        require(set(truth.get("fixed_seed_envelopes") or {}) == set(truth["families"]) and
                truth["fixed_seed_envelopes"] == initial_truth.get("fixed_seed_envelopes") and
                truth.get("seed_engineering_scale_source") == initial_truth.get("seed_engineering_scale_source"),
                "Original fixed seed envelope or engineering source changed")
    family_reports = []
    require(truth["families"].keys() == initial_truth["families"].keys(), "Truth family inventory changed")
    for name, family in truth["families"].items():
        original = initial_truth["families"][name]
        require(all(family[key] == original[key] for key in ("authority", "group_ids", "uncertainties")), "Truth source authorities, rows or sigmas changed: " + name)
        limits, old_limits = np.asarray(family["limits"], float), np.asarray(original["limits"], float)
        require(limits.shape == old_limits.shape == (len(family["group_ids"]),), "Truth group limits layout changed")
        require(np.isfinite(limits).all() and np.all(limits >= 0), "Invalid original truth limits: " + name)
        if progress_policy == "fixed_seed_envelope":
            require(np.array_equal(limits, old_limits), "Original fixed seed truth limits changed: " + name)
        else:
            require(np.all(limits <= old_limits), "Saved original truth limits worsened: " + name)
        family_reports.append(dict(name=name, groups=len(limits), authorities_group_ids_uncertainties_unchanged=True,
            limits_monotonic=bool(np.all(limits <= old_limits)), limits_identical_to_parent=bool(np.array_equal(limits, old_limits))))
    group_count=sum(entry["groups"] for entry in family_reports)
    require(group_count==sum(len(r['group_ids'])for r in initial_truth['families'].values()),
            'Complete original truth groups required')
    if progress_policy == "historical_ratchet":
        require(truth["source_rows_uncertainties_and_nonlinear_guards_unchanged"] is True and truth["historical_group_minimum_limits_preserved"] is True, "Saved truth contract changed")
    else:
        require(truth.get("source_rows_uncertainties_and_nonlinear_support_operators_unchanged") is True and
                truth.get("historical_group_minimum_limits_preserved") is False and
                truth.get("source_rows_uncertainties_and_nonlinear_guards_unchanged") is False,
                "Saved fixed seed truth contract changed")
    for filename, key in (("scene_UNACCEPTED.json", "scene_sha256"), ("source_review_UNACCEPTED.json", "source_review_sha256")):
        require(file_sha(directory / filename) == identity[key], "Original accepted review/scene identity changed: " + filename)
    manifest = truth["source_input_manifest_reference"]
    require(file_sha(manifest["path"]) == manifest["sha256"] == identity["input_manifest_sha256"], "Accepted original manifest changed")
    require(file_sha(identity["camera_path"]) == identity["camera_sha256"], "Accepted physical camera identity changed")
    updated = copy.deepcopy(config)
    options = updated["source_options"]
    options["--camera-seed"] = str(checkpoint)
    options["--truth-restoration-state"] = str(truth_path)
    options["--initial-damping"] = damping
    if resolved_parent_truth is not None:options.pop('--truth-restoration-reviewed-manual-append-audit')
    changes = [key for key in set(options)|set(input_options) if options.get(key) != input_options.get(key)]
    require(set(changes) <= {"--camera-seed", "--truth-restoration-state", "--initial-damping",
                            '--truth-restoration-reviewed-manual-append-audit'}, "Original source execution inputs changed")
    state = dict(q_sha256=q_sha, parameters=identity["parameter_count"], damping=damping, next_line_search_fraction=fraction,
        fields=field_count, profiles=len(profile_keys), frozen_profile_token=token, history_records=history_count,
        review_records=reviews_count, original_fixed_numerical_and_auxiliary_fields_verified=True,
        progress_policy=progress_policy)
    updated["continuation"] = dict(complete_accepted_state=str(checkpoint), checkpoint_file_sha256=checkpoint_sha, truth_restoration_state=str(truth_path), truth_restoration_state_sha256=file_sha(truth_path), state=state,
        previous_input_configuration=str(args.input_config.resolve()), accepted_steps_in_completed_run=accepted_count,
        source_observations_roles_limits_and_weights_unchanged=True, seed_is_latest_accepted_not_final_unaccepted_trial=True, truth_restoration_state_is_same_q_as_complete_checkpoint=True,
        complete_saved_group_count=group_count,all_saved_group_limits_and_uncertainties_preserved=True,
        explicit_initial_damping_equals_exact_saved_value=True, next_start_requires_fresh_source_J=True,
        no_cached_J_H_g_or_prefix_imported=True, no_step_or_runtime_limit_added=True,
        expected_complete_active_source_scalars=read_json(manifest['path'])['source_rows'],
        resolved_parent_initial_truth_state=None if resolved_parent_truth is None else
            dict(path=str(resolved_parent_truth),sha256=file_sha(resolved_parent_truth)),
        previous_run_stop_or_failure_does_not_invalidate_saved_acceptance=True, accuracy_accepted=False)
    report = dict(schema="rinkspace.accepted-archive-original-resume-input-check.v1", state=state, archive_directory=str(directory), checkpoint_sha256=checkpoint_sha, all_complete_fields=fields,
        truth_families=family_reports, accepted_iteration=accepted_iteration, changed_source_options=changes, original_source_options_other_than_continuation_unchanged=True,
        source_graph_J_or_QP_loaded_or_recomputed=False, accepted_archive_scene_camera_review_manifest_identities_verified=True,
        previous_run_failure_not_marked_stability_pass=True, configuration=str(args.output.resolve()))
    for path, value in ((args.output, updated), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(dict(configuration=str(args.output.resolve()), report=str(args.report.resolve()), state=state, truth_groups=group_count, changed_source_options=changes)), flush=True)


if __name__ == "__main__":
    main()
