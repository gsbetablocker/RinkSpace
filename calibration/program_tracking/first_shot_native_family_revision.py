"""Opt-in whole-family native material points on the existing finite-curve kernel.

There is deliberately no production installer.  Preparing or previewing a
candidate does not admit its pixels.  Actual replacement requires a separate
complete independent RGB identity review of every original native readout.

Each saved source profile normal contributes one scalar, in native query
order.  This is the existing conditional source-normal localization model,
not an exact all-q/rolling-row generative likelihood.  In particular the
closest-foot residual and observed-y camera response are not claimed to have
identity UV sensitivity.  No camera, world geometry or sigma is introduced.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import numpy as np

from .official_static_surface_dataset import reference, read_reference
from .first_shot_native_circle import compile_circle_queries
from .first_shot_noise_revision import source_members_digest
from .source_evidence_whitening import SourceWhiteningOperator

SCHEMA = "rinkspace.native-finite-curve-family-candidate.v1"
REVIEW_SCHEMA = "rinkspace.independent-native-family-RGB-review.v1"
NOISE_MODEL = "conditional_saved_source_normal_close_to_current_material_normal"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _read(ref):
    _check_reference(ref)
    return read_reference({k: ref[k] for k in ("path", "sha256")})


def _check_reference(ref):
    if reference(ref["path"]) != {k: ref[k] for k in ("path", "sha256")}:
        raise ValueError("Native family evidence bytes changed: " + ref["path"])


def _flat(frame):
    return [(row, j) for row in frame["rows"] for j in range(len(row["uv"]))]


def _members(frame, group):
    points = _flat(frame)
    return [dict(source_point_id=points[i][0]["source_point_ids"][points[i][1]],
        UV=_plain(points[i][0]["uv"][points[i][1]]),
        sigma=float(points[i][0]["sigma"][points[i][1]]),
        target=deepcopy(points[i][0]["target"]),
        source_feature_id=points[i][0]["source_feature_id"],
        source_segment_index=points[i][0]["source_segment_index"],
        original_vertex_index=points[i][0]["original_vertex_indices"][points[i][1]])
        for i in group["point_indices"]]


def _archived_row_semantics(rows, member_ids):
    """Canonicalize only row batching, retaining every nonpixel row semantic."""
    semantics = {}
    sliced = {"uv", "sigma", "source_point_ids", "original_vertex_indices"}
    for row in rows:
        for index, pid in enumerate(row["source_point_ids"]):
            value = {k: _plain(v) for k,v in row.items() if k not in sliced}
            if "point_support_bounds_xy_px" in value:
                value["point_support_bounds_xy_px"] = [value["point_support_bounds_xy_px"][index]]
            if pid in semantics:
                raise ValueError("Original family row semantics duplicated a source identity")
            semantics[pid] = value
    if set(semantics) != set(member_ids):
        raise ValueError("Original family archive row membership changed")
    return [semantics[pid] for pid in member_ids]


def authenticate_original_diagonal_factor(native, old_operator, house_id="F2"):
    """Reproduce the existing family W from its original D and raw sigma.

    This authenticates the available original diagonal assumption.  It never
    recovers covariance from W or calls that assumption a measured full L.
    """
    from .first_shot_source_evidence import _ellipse_normal, _affine_basis, _influence
    from .source_evidence_whitening import latent_whitening_operator
    features = {f["id"]: f for f in native["features"]
        if f.get("source_circle_fit")
        and f.get("joint_group_navigation", {}).get("house_id") == house_id}
    direct = [r for f in features.values() for r in f["source_circle_fit"]["direct_readouts"]]
    normals = np.asarray([r["source_evidence"]["profile_normal"] for r in direct])
    sigma = np.asarray([r["sigma_px"] for r in direct])
    design = np.stack([n @ _affine_basis(r["representative_uv"])
        for n, r in zip(normals, direct, strict=True)])
    residual = np.asarray([(np.asarray(r["uv"])-r["representative_uv"]) @ n
        for n, r in zip(normals, direct, strict=True)])
    influence = _influence(design, (1+(residual/1.5)**2)**(-1.5))
    original_factor = influence * sigma
    Q, sensitivity = [], []
    ids = old_operator.source_point_ids
    for i, pid in enumerate(ids):
        fid, segment_word, segment, vertex_word, vertex = pid.split("/")
        if segment_word != "segment" or vertex_word != "vertex":
            raise ValueError("Original family representative address changed")
        feature = features[fid]
        uv = feature["segments"][int(segment)]["uv"][int(vertex)]
        normal = _ellipse_normal(feature, [uv])[0]
        projected = np.outer(normal, normal) @ _affine_basis(uv)
        normal = normal / np.linalg.norm(normal)
        row = np.zeros(2*len(ids)); row[2*i:2*i+2] = normal
        Q.append(row); sensitivity.append(normal @ projected)
    rebuilt = latent_whitening_operator(ids, np.asarray(sensitivity),
        explicit_noise_factor=original_factor, physical_basis=np.asarray(Q),
        group_id=old_operator.group_id, covariance_origin=old_operator.covariance_origin)
    if not np.array_equal(rebuilt.matrix, old_operator.matrix):
        raise ValueError("Original complete family W is not reproduced by its archived raw diagonal factor")
    return dict(original_raw_L_shape=[len(direct), len(direct)],
        original_raw_diagonal_sigma_preserved=True, measured_full_L_available=False,
        old_W_rebuilt_from_original_D_and_raw_sigma_bytes_exact=True)


def source_normal_operator(queries, group_id):
    """One original scalar/query, fixed query order; no SVD/Huber rotation."""
    count = len(queries.query_ids)
    matrix = np.zeros((count, 2*count))
    for i, (normal, sigma) in enumerate(zip(queries.normals, queries.sigma, strict=True)):
        matrix[i, 2*i:2*i+2] = normal / sigma
    return SourceWhiteningOperator(matrix, queries.query_ids, group_id,
        "original_native_profile_normal_and_diagonal_sigma_without_template_compression",
        dict(native_noise_dimension=count, physical_scalar_rows=count,
            frozen_basis="original_native_query_order_and_saved_unit_profile_normal",
            covariance_model=NOISE_MODEL, exact_all_q_residual_covariance_claim=False))


def retained_queries(queries, indices):
    """Keep original readout scalars/order; never slice the old compressed W."""
    indices = np.asarray(indices, int)
    if (indices.ndim != 1 or not len(indices) or len(set(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(queries.query_ids))
            or not np.array_equal(indices, np.sort(indices))):
        raise ValueError("A native family requires a nonempty exact ordered retained subset")
    return replace(queries,
        roles=tuple(queries.roles[i] for i in indices),
        query_ids=tuple(queries.query_ids[i] for i in indices),
        **{key:np.asarray(getattr(queries,key))[indices].copy() for key in
           ("observed","origins","normals","support","sigma")})


def prepare_family_candidate(frame, binding_reference, *, house_id="F2", original_archive_reference=None):
    """Archive one exact old group and its complete raw inventory, unadmitted."""
    binding = _read(binding_reference)
    packet = _read(frame["source_packet"])
    native_ref = packet["ice_source_readout_packet"]
    native = _read(native_ref)
    queries = compile_circle_queries(native, house_id=house_id)
    gid = f"{frame['native_pts']}/{house_id}/shared_native_circle_family"
    groups = [g for g in frame["whitening_groups"] if g["operator"].group_id == gid]
    if len(groups) != 1:
        raise ValueError("Exactly one entire original three-ring family is required")
    group = groups[0]
    model = _read(binding["model"]); registry = _read(model["feature_registry"])
    features = [f for f in native["features"] if f.get("source_circle_fit")
        and f.get("joint_group_navigation", {}).get("house_id") == house_id]
    targets = {f["id"]: deepcopy(registry["targets"][f["id"]]) for f in features}
    for target in targets.values():
        if (target.get("kind") != "ice_curve"
                or target.get("residual") != "point_to_named_finite_ice_curve"):
            raise ValueError("Use the existing named finite physical curve residual")
    members = _members(frame, group)
    if any(m["target"] != targets.get(m["source_feature_id"]) for m in members):
        raise ValueError("Original actual family target differs from the complete registry descriptor")
    source_rows = {id(_flat(frame)[i][0]) for i in group["point_indices"]}
    inventory = []
    for feature in features:
        for array_index, raw in enumerate(feature["source_circle_fit"]["direct_readouts"]):
            qi = int(raw["source_evidence"]["query_index"])
            inventory.append(dict(source_point_id=f"{frame['native_pts']}/{feature['id']}/query/{qi}",
                source_point_id_origin="new_explicit_native_packet_query_locator_not_original_vertex_ID",
                source_feature_id=feature["id"], native_readout_array_index=array_index,
                native_query_index=qi, original_native_readout=deepcopy(raw)))
    result = dict(schema=SCHEMA, frame=frame["index"], native_pts=frame["native_pts"],
        house_id=house_id, group_id=gid, base_binding=binding_reference,
        source_packet=frame["source_packet"], native_readout_packet=native_ref,
        source_registry=model["feature_registry"], material_targets=targets,
        old_point_indices=list(group["point_indices"]), old_operator=group["operator"].to_record(),
        old_source_members_sha256=source_members_digest(frame, group["point_indices"]),
        old_members=members,
        archived_original_source_rows=[_plain(r) for r in frame["rows"] if id(r) in source_rows],
        archived_original_group_metadata=_plain({k:v for k,v in group.items() if k!="operator"}),
        native_support_sha256=queries.native_readouts_sha256, native_inventory=inventory,
        original_factor_proof=authenticate_original_diagonal_factor(native, group["operator"], house_id),
        new_operator=source_normal_operator(queries, gid).to_record(),
        independent_RGB_admission_complete=False, production_installed=False,
        noise_model=NOISE_MODEL,
        exact_fixed_query_or_all_q_RS_likelihood=False,
        robust_basis="one original n_i/sigma_i row per query in exact original packet order",
        old_and_new_objective_costs_comparable=False)
    if original_archive_reference is not None:
        original = _read(original_archive_reference)
        # Original revisions can split rows and remap addresses after an
        # ordinary quarantine.  Authorize neither a pixel nor an operator
        # change while explicitly recording that address-only transition.
        invariants = ("schema", "frame", "native_pts", "house_id", "group_id", "base_binding",
            "source_packet", "native_readout_packet", "source_registry", "material_targets",
            "old_operator", "old_source_members_sha256", "old_members", "native_support_sha256",
            "native_inventory", "original_factor_proof", "new_operator", "noise_model",
            "exact_fixed_query_or_all_q_RS_likelihood", "robust_basis")
        if any(original.get(key) != result.get(key) for key in invariants):
            raise ValueError("Address rebase changed original family pixels, identities, targets, factor or native support")
        member_ids = [m["source_point_id"] for m in result["old_members"]]
        if (_archived_row_semantics(original["archived_original_source_rows"], member_ids)
                != _archived_row_semantics(result["archived_original_source_rows"], member_ids)):
            raise ValueError("Address rebase changed original family row semantics or metadata")
        old_metadata = {k:v for k,v in original["archived_original_group_metadata"].items() if k!="point_indices"}
        current_metadata = {k:v for k,v in result["archived_original_group_metadata"].items() if k!="point_indices"}
        if old_metadata != current_metadata:
            raise ValueError("Address rebase changed original family group semantics or metadata")
        result["original_approved_archive_reference"] = deepcopy(original_archive_reference)
    return result


def rebase_family_candidate(frame, original_archive_reference, binding_reference):
    """Use authenticated current row/point addresses; retain the old archive.

    Call after the normal original source/noise/native pipeline, before any
    workers or evaluation.  This does not install or admit the candidate.
    """
    original = _read(original_archive_reference)
    return prepare_family_candidate(frame, binding_reference,
        house_id=original["house_id"], original_archive_reference=original_archive_reference)


def _authenticate(item, frame, binding, review, *, preview):
    if item.get("schema") != SCHEMA:
        raise ValueError("Explicit whole native family candidate required")
    base = _read(item["base_binding"])
    if base != binding:
        raise ValueError("Native family candidate has another authenticated source basis")
    if item["native_pts"] != frame["native_pts"] or item["frame"] != frame["index"]:
        raise ValueError("Native family source exposure changed")
    if item["source_packet"] != frame["source_packet"]:
        raise ValueError("Native family original source packet changed")
    native = _read(item["native_readout_packet"]); packet = _read(item["source_packet"])
    if packet["ice_source_readout_packet"] != item["native_readout_packet"]:
        raise ValueError("Native family packet lineage changed")
    for source in (packet["source"], native["source"]):
        if (source["native_pts"] != frame["native_pts"] or source["actual_pts"] != frame["native_pts"]
                or source["time_base"] != binding["source_clock"]["time_base"]
                or source["source_video_sha256"] != binding["source"]["sha256"]):
            raise ValueError("Native family clock/video changed")
    if packet["source"]["decoded_BGR_sha256"] != native["source"]["decoded_BGR_sha256"]:
        raise ValueError("Native family source pixels changed")
    query = compile_circle_queries(native, house_id=item["house_id"])
    group = next(g for g in frame["whitening_groups"] if g["operator"].group_id == item["group_id"])
    old = SourceWhiteningOperator.from_record(item["old_operator"])
    if (list(group["point_indices"]) != item["old_point_indices"]
            or group["operator"].to_record() != old.to_record()
            or _members(frame, group) != item["old_members"]
            or source_members_digest(frame, group["point_indices"]) != item["old_source_members_sha256"]
            or query.native_readouts_sha256 != item["native_support_sha256"]):
        raise ValueError("Original entire family members/UV/ID/sigma/target/operator differ")
    expected = prepare_family_candidate(frame, item["base_binding"], house_id=item["house_id"],
        original_archive_reference=item.get("original_approved_archive_reference"))
    if expected != item:
        raise ValueError("Complete archived family candidate or native inventory changed")
    retained = list(range(len(query.query_ids))); excluded = []
    if not preview or review is not None:
        if (review is None or review.get("schema") != REVIEW_SCHEMA
                or review.get("all_original_native_queries_independently_reviewed") is not True
                or review.get("world_camera_or_fit_residual_used_for_identity") is not False
                or review.get("native_support_sha256") != item["native_support_sha256"]):
            raise ValueError("All original native queries require independent RGB identity review before FIT")
        checks = review.get("queries", [])
        if len(checks) != len(item["native_inventory"]):
            raise ValueError("Native RGB review must cover the entire original inventory")
        retained = []; excluded = []
        for index, (original, accepted) in enumerate(zip(item["native_inventory"], checks, strict=True)):
            if (accepted.get("native_inventory_entry") != original
                    or accepted.get("authority") not in ("agent_rgb_review", "user_rgb_review")
                    or accepted.get("actually_viewed_original_RGB") is not True
                    or accepted.get("disposition") not in ("retained", "excluded")
                    or not isinstance(accepted.get("reason"), str) or not accepted["reason"].strip()
                    or not accepted.get("evidence_references")):
                raise ValueError("Every native query needs an exact independent RGB disposition and evidence")
            if accepted["disposition"] == "retained":
                if (accepted.get("independent_static_material_identity_confirmed") is not True
                        or accepted.get("outside_person_broadcast_and_unconfirmed_material") is not True
                        or accepted.get("admit_to_fit") is not True):
                    raise ValueError("Unconfirmed native material query cannot enter FIT")
                retained.append(index)
            else:
                if accepted.get("admit_to_fit") is not False:
                    raise ValueError("An excluded native query cannot enter FIT")
                excluded.append(index)
            if accepted["authority"] == "agent_rgb_review" and accepted.get("user_confirmed") is not False:
                raise ValueError("Agent RGB review cannot be labelled human confirmation")
            for ref in accepted["evidence_references"]:
                _check_reference(ref)
    return group, query, retained, excluded


def replace_native_family(frame, item, binding, *, identity_review=None,
                          identity_review_reference=None, diagnostic_preview=False):
    """Atomically retire all old family representatives and append grouped raw UV.

    ``diagnostic_preview`` is explicitly unsuitable for FIT admission.  No
    alias or SourceBundle is installed by this module in either mode.
    """
    if identity_review_reference is not None:
        saved_review = _read(identity_review_reference)
        if identity_review is not None and identity_review != saved_review:
            raise ValueError("Native family review does not match its frozen reference")
        identity_review = saved_review
    group, original_queries, kept, excluded = _authenticate(item, frame, binding,
        identity_review, preview=diagnostic_preview)
    queries = retained_queries(original_queries, kept)
    # This is a row selection of the authenticated *raw* diagonal factor,
    # retaining every original latent column.  It is never a W slice.
    retained_L = np.diag(original_queries.sigma)[kept, :]
    if (np.any(np.count_nonzero(retained_L, axis=1) != 1)
            or not np.array_equal(np.max(np.abs(retained_L), axis=1), queries.sigma)):
        raise ValueError("Retained original raw factor changed its scalar source noise")
    points = _flat(frame); removed = set(group["point_indices"])
    source_rows = {id(points[i][0]) for i in removed}
    if removed != {i for i,(row,_) in enumerate(points) if id(row) in source_rows}:
        raise ValueError("Whole original rows must belong to the one retired family")
    rows, mapping, cursor, old_cursor = [], {}, 0, 0
    for row in frame["rows"]:
        indices = list(range(old_cursor, old_cursor+len(row["uv"]))); old_cursor += len(indices)
        if id(row) in source_rows:
            continue
        rows.append(row)
        for index in indices:
            mapping[index] = cursor; cursor += 1
    new_ids = set(queries.query_ids)
    if new_ids.intersection(pid for row in rows for pid in row["source_point_ids"]):
        raise ValueError("A native query would be counted twice")
    start = cursor
    feature_order = list(dict.fromkeys(v["source_feature_id"] for v in item["native_inventory"]))
    for fid in feature_order:
        target = item["material_targets"][fid]
        entries = [v for i,v in enumerate(item["native_inventory"])
            if i in kept and v["source_feature_id"] == fid]
        if not entries:
            continue
        first = next(row for row in frame["rows"] if row["source_feature_id"] == fid)
        row = {k: deepcopy(first[k]) for k in ("native_pts", "actual_pts", "time_base",
            "source_video_sha256", "source_packet")}
        row.update(target=deepcopy(target), _target_key=first["_target_key"],
            id=f"{frame['native_pts']}/{fid}/native_queries", source_feature_id=fid,
            source_segment_index=-1, point_correspondence=False,
            uv=np.asarray([v["original_native_readout"]["uv"] for v in entries], float),
            sigma=np.asarray([v["original_native_readout"]["sigma_px"] for v in entries], float),
            source_point_ids=[v["source_point_id"] for v in entries],
            original_vertex_indices=[v["native_query_index"] for v in entries],
            original_vertex_indices_are_native_query_locators=True,
            native_query_lineage=deepcopy(entries),
            measurement_origin="original_native_material_interface_without_template_mean")
        rows.append(row); cursor += len(entries)
    native_operator = source_normal_operator(queries, item["group_id"])
    operator = SourceWhiteningOperator(native_operator.matrix, native_operator.source_point_ids,
        native_operator.group_id, native_operator.covariance_origin,
        dict(native_operator.numerical_rank_evidence,
            native_noise_dimension=len(original_queries.query_ids),
            retained_raw_L_shape=list(retained_L.shape),
            retained_raw_L_float64_bytes_sha256=hashlib.sha256(retained_L.tobytes()).hexdigest(),
            original_latent_columns_preserved=True))
    groups, old_formal, new_formal, blocks = [], 0, 0, []
    for previous in frame["whitening_groups"]:
        replacement = previous is group
        updated = (dict(operator=operator, point_indices=list(range(start,cursor)),
            native_family_replacement=dict(schema=SCHEMA, native_support_sha256=item["native_support_sha256"],
                old_operator=item["old_operator"], old_members=deepcopy(item["old_members"]),
                archived_original_source_rows=deepcopy(item["archived_original_source_rows"]),
                archived_original_group_metadata=deepcopy(item["archived_original_group_metadata"]),
                original_factor_proof=deepcopy(item["original_factor_proof"]),
                original_approved_archive_reference=deepcopy(item.get("original_approved_archive_reference")),
                independent_RGB_review_reference=deepcopy(identity_review_reference),
                original_native_inventory=deepcopy(item["native_inventory"]),
                retained_original_inventory_indices=list(kept), excluded_original_inventory_indices=list(excluded),
                complete_independent_RGB_dispositions=deepcopy(identity_review),
                retained_raw_L_shape=[len(kept),len(original_queries.query_ids)],
                retained_original_raw_noise_factor=retained_L.tolist(),
                original_latent_columns_preserved=True, old_W_row_or_column_slice_used=False,
                original_native_readout_packet=item["native_readout_packet"],
                independent_RGB_admitted=not diagnostic_preview,
                conditional_noise_model=NOISE_MODEL)) if replacement else
            dict(previous, point_indices=[mapping[i] for i in previous["point_indices"]]))
        blocks.append(dict(group_id=previous["operator"].group_id, replaced=replacement,
            old_local_formal_slice=[old_formal,old_formal+previous["operator"].rank],
            new_local_formal_slice=[new_formal,new_formal+updated["operator"].rank]))
        old_formal += previous["operator"].rank; new_formal += updated["operator"].rank
        groups.append(updated)
    old_ordinary = 2*(frame["source_point_count"]-sum(len(g["point_indices"]) for g in frame["whitening_groups"]))
    new_ordinary = 2*(cursor-sum(len(g["point_indices"]) for g in groups))
    if old_ordinary != new_ordinary:
        raise ValueError("Native family replacement changed ordinary observations")
    updated = dict(frame, rows=rows, whitening_groups=groups, source_point_count=cursor)
    flattened = _flat(updated)
    for newgroup in groups:
        if tuple(flattened[i][0]["source_point_ids"][flattened[i][1]] for i in newgroup["point_indices"]) != newgroup["operator"].source_point_ids:
            raise ValueError("Retained/new group point-index remap changed membership")
    proof = dict(frame=frame["index"], native_pts=frame["native_pts"],
        diagnostic_preview_only=diagnostic_preview, production_installed=False,
        old_raw_points=frame["source_point_count"], new_raw_points=cursor,
        old_formal_rows=old_formal+old_ordinary, new_formal_rows=new_formal+new_ordinary,
        ordinary_formal_rows_unchanged=old_ordinary, formal_blocks=blocks,
        old_representatives_removed_atomically=True, original_representatives_and_history_archived=True,
        original_native_UV_sigma_profile_normal_and_query_order_preserved=True,
        original_native_queries=len(original_queries.query_ids), retained_native_queries=len(kept),
        excluded_native_queries=len(excluded), retained_original_inventory_indices=list(kept),
        excluded_original_inventory_indices=list(excluded), all_original_readouts_and_dispositions_archived=True,
        retained_raw_L_shape=[len(kept),len(original_queries.query_ids)], original_latent_columns_preserved=True,
        named_finite_curve_and_observed_y_row_camera_unchanged=True,
        new_source_basis_requires_fresh_r_and_J=True, cross_basis_cost_comparable=False,
        noise_model=NOISE_MODEL, exact_fixed_query_or_all_q_RS_likelihood=False)
    updated["native_family_replacement_audit"] = proof
    return updated, proof
