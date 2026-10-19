"""Compare complete accepted NPZ fields without loading a graph or solving."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import (
    iter_checkpoint_history, read_checkpoint_history, read_checkpoint_reviews,
)

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def json_differences(left, right, path=""):
    if type(left) is not type(right):
        return [dict(path=path, difference="type", baseline=left, candidate=right)]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            child = path + "/" + str(key)
            if key not in left:
                result.append(dict(path=child, difference="candidate_only", candidate=right[key]))
            elif key not in right:
                result.append(dict(path=child, difference="baseline_only", baseline=left[key]))
            else:
                result.extend(json_differences(left[key], right[key], child))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [dict(path=path, difference="list_layout", baseline_count=len(left), candidate_count=len(right))]
        result = []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(json_differences(a, b, path + "/" + str(index)))
        return result
    return [] if left == right else [dict(path=path, difference="value", baseline=left, candidate=right)]


def profile_token(stored):
    result = hashlib.sha256(b"rinkspace.frozen-profile-no-copy.v1")
    for name in stored.files:
        if not name.startswith("profile__"):
            continue
        value = stored[name]
        result.update(name[9:].encode("utf8"))
        result.update(value.dtype.str.encode("ascii"))
        result.update(str(value.shape).encode("ascii"))
        if value.size:
            result.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return result.hexdigest()


def state_inventory(run, count):
    entries = []
    for directory in (run / "accepted_states").iterdir():
        path = directory / "complete_state_UNACCEPTED.npz"
        identity = json.loads((directory / "state_identity.json").read_text(encoding="utf8"))
        last = None
        for record in iter_checkpoint_history(path):
            last = record
        if last is None:
            raise ValueError("Accepted checkpoint has no original optimizer history: " + str(path))
        entries.append((int(last["iteration"]), directory, path, identity))
    entries.sort(key=lambda entry: entry[0])
    if len(entries) < count:
        raise ValueError("Requested accepted states are not yet available: " + str(run))
    return entries[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = state_inventory(args.baseline_run, args.count)
    candidate = state_inventory(args.candidate_run, args.count)
    result = dict(schema="rinkspace.independent-complete-accepted-checkpoint-field-comparison.v1",
                  baseline_run=str(args.baseline_run.resolve()), candidate_run=str(args.candidate_run.resolve()),
                  requested_accepted_state_count=args.count, states=[],
                  source_graph_J_or_QP_loaded_or_recomputed=False)
    unexpected = []
    for old, new in zip(baseline, candidate):
        oi, od, op, identity_old = old
        ni, nd, np_, identity_new = new
        old_stat, new_stat = op.stat(), np_.stat()
        old_sha, new_sha = file_sha(op), file_sha(np_)
        if old_sha != identity_old["state_npz_sha256"] or new_sha != identity_new["state_npz_sha256"]:
            raise ValueError("Accepted checkpoint archive differs from its recorded source identity")
        state = dict(iteration_baseline=oi, iteration_candidate=ni,
                     baseline_npz=dict(path=str(op), sha256=old_sha),
                     candidate_npz=dict(path=str(np_), sha256=new_sha), fields=[])
        with np.load(op, allow_pickle=False) as a, np.load(np_, allow_pickle=False) as b:
            state.update(key_order_equal=a.files == b.files, key_sets_equal=set(a.files) == set(b.files),
                         baseline_key_order=a.files, candidate_key_order=b.files)
            if not state["key_order_equal"]:
                unexpected.append(dict(iteration=oi, issue="checkpoint_field_inventory_changed"))
            for key in dict.fromkeys(a.files + b.files):
                if key not in a.files or key not in b.files:
                    state["fields"].append(dict(name=key, baseline_present=key in a.files, candidate_present=key in b.files))
                    continue
                av, bv = a[key], b[key]
                ah, bh = array_sha(av), array_sha(bv)
                same = av.shape == bv.shape and av.dtype == bv.dtype and ah == bh
                field = dict(name=key, baseline_shape=list(av.shape), candidate_shape=list(bv.shape),
                             baseline_dtype=av.dtype.str, candidate_dtype=bv.dtype.str,
                             baseline_sha256=ah, candidate_sha256=bh, bitexact=bool(same))
                if key == "history_json":
                    # Storage dtype/bytes describe the list or reference only;
                    # compare the authenticated records separately from state.
                    histories = [read_checkpoint_history({key: value}) for value in (av, bv)]
                    differences = json_differences(*histories)
                    field.update(category="optimizer_history_with_execution_reports",
                                 history_storage_bitexact=bool(same), semantic_history_equal=not differences,
                                 JSON_differences=differences,
                                 history_lengths=[len(value) for value in histories],
                                 objective_and_damping_endpoints_equal=[
                                     all(x[k] == y[k] for k in ("cost", "next_cost", "damping", "accepted"))
                                     for x, y in zip(*histories)])
                    state["optimizer_history_semantics_equal"] = not differences
                    del histories
                elif key == "reviews_json":
                    reviews = [read_checkpoint_reviews({key: value}) for value in (av, bv)]
                    differences = json_differences(*reviews)
                    field.update(category="source_review_history_with_execution_reports",
                                 review_storage_bitexact=bool(same), semantic_reviews_equal=not differences,
                                 JSON_differences=differences,
                                 review_lengths=[len(value) for value in reviews])
                    state["source_review_semantics_equal"] = not differences
                    del reviews
                elif not same:
                    if key == "elapsed_seconds":
                        field.update(category="execution_timing", baseline_value=float(av), candidate_value=float(bv))
                    elif key == "next_line_search_fraction":
                        field.update(category="optimizer_execution_seed_learning_fix", baseline_value=float(av), candidate_value=float(bv))
                    elif key == "identity_json":
                        field.update(category="actual_run_manifest_provenance",
                                     JSON_differences=json_differences(json.loads(str(av.item())), json.loads(str(bv.item()))))
                    elif key == "aux__current_original_motion_input_compaction_json":
                        changes = json_differences(json.loads(str(av.item())), json.loads(str(bv.item())))
                        timing_only = all(change["path"].startswith("/blocks/")
                                          and change["path"].endswith("/actual_retained_r_J_check_seconds")
                                          for change in changes)
                        field.update(category="retained_source_equivalence_check_execution_timing" if timing_only
                                     else "unclassified_persistent_state_difference",
                                     JSON_differences=changes,
                                     original_compaction_source_arrays_roles_and_row_layout_equal=timing_only)
                        if not timing_only:
                            unexpected.append(dict(iteration=oi, field=key))
                    else:
                        field["category"] = "unclassified_persistent_state_difference"
                        if av.dtype.kind in "US" and av.shape == bv.shape == ():
                            try:
                                field["JSON_differences"] = json_differences(json.loads(str(av.item())), json.loads(str(bv.item())))
                            except (ValueError, TypeError):
                                field.update(baseline_value=str(av.item()), candidate_value=str(bv.item()))
                        elif av.dtype.kind in "bifu" and av.shape == bv.shape:
                            field["maximum_numeric_difference"] = float(np.max(np.abs(av.astype(float) - bv.astype(float))))
                        unexpected.append(dict(iteration=oi, field=key))
                state["fields"].append(field)
            old_token, new_token = profile_token(a), profile_token(b)
            state["frozen_profiles"] = dict(
                array_count=len([key for key in a.files if key.startswith("profile__")]),
                all_profile_shape_dtype_and_bytes_equal=all(field["bitexact"] for field in state["fields"] if field["name"].startswith("profile__")),
                recomputed_baseline_token=old_token, recomputed_candidate_token=new_token,
                baseline_token_matches_state_identity=old_token == identity_old["frozen_profile_token"],
                candidate_token_matches_state_identity=new_token == identity_new["frozen_profile_token"],
                tokens_equal=old_token == new_token)
            if not all(state["frozen_profiles"][key] for key in
                       ("all_profile_shape_dtype_and_bytes_equal", "baseline_token_matches_state_identity",
                        "candidate_token_matches_state_identity", "tokens_equal")):
                unexpected.append(dict(iteration=oi, issue="frozen_profile_or_serialization_identity_changed"))
        final_old_stat, final_new_stat = op.stat(), np_.stat()
        state["archives_stable_while_reading"] = (
            (old_stat.st_size, old_stat.st_mtime_ns) == (final_old_stat.st_size, final_old_stat.st_mtime_ns)
            and (new_stat.st_size, new_stat.st_mtime_ns) == (final_new_stat.st_size, final_new_stat.st_mtime_ns))
        if not state["archives_stable_while_reading"]:
            raise ValueError("Accepted checkpoint archive changed while reading")
        state["differing_fields"] = [field["name"] for field in state["fields"] if not field.get("bitexact", False)]
        result["states"].append(state)
    result["unexpected_persistent_state_differences"] = unexpected
    result["all_complete_NPZ_fields_bitexact"] = all(not state["differing_fields"] for state in result["states"])
    result["optimizer_history_semantics_equal"] = all(state.get("optimizer_history_semantics_equal", False)
                                                       for state in result["states"])
    result["source_review_semantics_equal"] = all(state.get("source_review_semantics_equal", False)
                                                  for state in result["states"])
    result["history_storage_differences_are_not_numerical_state_differences"] = True
    result["review_storage_differences_are_not_numerical_state_differences"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(dict(output=str(args.output.resolve()), unexpected=unexpected,
                         states=[dict(iteration=state["iteration_baseline"],
                                      field_count=len(state["fields"]), different_fields=state["differing_fields"],
                                      profiles=state["frozen_profiles"]) for state in result["states"]])), flush=True)
    if unexpected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
