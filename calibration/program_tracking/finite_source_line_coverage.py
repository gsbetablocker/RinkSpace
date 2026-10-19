"""Finite material existence on authenticated native profile query sections.

This is a geometric domain, not another pixel residual. The original source
UV, sigma and complete W remain caller-owned and unchanged. A line profile
measures a normal location at a fixed real native query section; a finite
material model must reach that section. Its unknown actual endpoint is not
measured merely because the last supported section is visible.

Source metadata is compiled once. At evaluation, Brown4 intersections are
solved on the physical world line. With native row timing, both world station
and the intersection's own sensor exposure row are solved together. No
query-center row is substituted for the physical intersection row.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import hashlib
import json

import numpy as np


class FiniteLineCoverageError(ValueError):
    """A declared native material section lacks finite forward model support."""
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class NativeLineSections:
    index: int
    native_pts: int
    target_key: str
    source_feature_id: str
    query_uv: np.ndarray
    profile_normal: np.ndarray
    measured_uv: np.ndarray
    query_ids: tuple[str, ...]
    evidence: Mapping
    end_columns: tuple[int, ...] = ()
    start_columns: tuple[int, ...] = ()

    def __post_init__(self):
        count = len(self.query_ids)
        if not count or len(set(self.query_ids)) != count:
            raise ValueError("Actual native query sections require distinct identities")
        for name in ("query_uv", "profile_normal", "measured_uv"):
            value = np.asarray(getattr(self, name), np.float64).copy()
            if value.shape != (count, 2) or not np.isfinite(value).all():
                raise ValueError("Finite original native section arrays required: " + name)
            if name == "profile_normal":
                length = np.linalg.norm(value, axis=1)
                if np.any(length <= 0.):
                    raise ValueError("Actual source profile direction must be nonzero")
                value /= length[:, None]
            else:
                if np.any(value < 0.) or np.any(value >= [1920., 1080.]):
                    raise ValueError("Actual native query/source rows must be inside the sensor")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _feature_table(packet):
    features = packet.get("features")
    if not isinstance(features, list):
        raise ValueError("Actual native feature list required")
    result = {feature["id"]: feature for feature in features}
    if len(result) != len(features):
        raise ValueError("Duplicate native material feature identity")
    return result


@dataclass(frozen=True)
class NativeLineCoverageInventory:
    sections: tuple[NativeLineSections, ...]
    unsupported: tuple[Mapping, ...]
    requested_extent_columns: tuple[int, ...] | None
    initializer_strategy: str = "minimum_source_coverage"

    def __iter__(self):
        return iter(self.sections)

    def __len__(self):
        return len(self.sections)


def _compile_one(frame, row, feature, native_ref):
    fid = "ice." + row["target"]["curve_id"]
    if feature is None or not feature.get("source_line_fit"):
        raise FiniteLineCoverageError("Actual native straight material readouts are missing: " + fid)
    line = feature["source_line_fit"]
    actual = line.get("direct_readouts")
    if line.get("current_frame_only") is not True or not actual:
        raise FiniteLineCoverageError("Source line has no real current native query sections: " + fid)
    expected_native_group=f"{frame['native_pts']}/{fid}/source_line"
    native_replacement=bool(row.get("native_line_queries")) and any(
        g["operator"].group_id==expected_native_group and g.get("native_line_replacement") is not None
        for g in frame["whitening_groups"])
    queries, normals, measured, ids = [], [], [], []
    for readout in actual:
        evidence = readout.get("source_evidence", {})
        supported_origin=evidence.get("measurement_origin") in (
                "current_native_signed_RGB_material_profiles",
                "current_native_source_identified_thin_paint_two_local_luminance_flanks")
        if native_replacement:
            supported_origin=supported_origin or evidence.get("measurement_origin") in (
                "current_native_source_identified_lane_long_edge_two_luminance_flanks_physical_transverse_section",
                "current_native_physical_width_complete_paint_ice_transitions")
        if (not supported_origin or evidence.get("actual_source_support_outside_person_and_broadcast_masks") is not True):
            raise FiniteLineCoverageError("Unsupported native query geometry/origin: " + fid + ": "
                                          + str(evidence.get("measurement_origin")))
        for name in ("query_uv", "profile_normal", "query_index"):
            if name not in evidence:
                raise FiniteLineCoverageError("Source query metadata missing " + name + ": " + fid)
        queries.append(evidence["query_uv"])
        normals.append(evidence["profile_normal"])
        measured.append(readout["uv"])
        ids.append(f"{frame['native_pts']}/{fid}/native_query/{evidence['query_index']}")
    receipt = dict(source_packet=frame["source_packet"], actual_native_readout_packet=native_ref,
        actual_native_readouts_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        original_native_readout_count=len(actual), model_or_representative_queries_added=0,
        source_profile_origin="actual direct_readouts.source_evidence.query_uv/profile_normal",
        source_measurement_operator=feature.get("measurement_operator"),
        actual_material_endpoint_measured=False, finite_extent_lower_bound_only=True)
    # These readouts must be the support of the *current* complete estimator,
    # not an older candidate family with human-excluded material put back in.
    from .first_shot_source_evidence import _influence
    from .source_evidence_whitening import latent_whitening_operator
    expected_group=f"{frame['native_pts']}/{fid}/source_line"
    matches=[g for g in frame["whitening_groups"]if g["operator"].group_id==expected_group]
    if len(matches)!=1:
        raise FiniteLineCoverageError("Actual native queries do not identify one current full-W line estimator: "+fid)
    group=matches[0];operator=group["operator"]
    if group.get("native_line_replacement") is not None:
        from .first_shot_native_line_revision import certify_native_group
        try:
            native_receipt=certify_native_group(frame,row,group)
        except (ValueError,KeyError) as error:
            raise FiniteLineCoverageError("Active native line factor failed coverage authentication: "+fid) from error
        item=group["native_line_replacement"]["item"]
        if (item["native_readout_packet"]!=native_ref
                or item["old_native_support_canonical_sha256"]!=receipt["actual_native_readouts_sha256"]):
            raise FiniteLineCoverageError("Coverage and native factor do not share the exact final support: "+fid)
        receipt.update(native_receipt,current_formal_estimator_group_id=operator.group_id,
            current_formal_point_ids=list(operator.source_point_ids),
            current_full_W_sha256=hashlib.sha256(operator.matrix.tobytes()).hexdigest(),
            actual_queries_are_existing_full_W_estimator_native_support=True,
            human_excluded_representatives_or_new_readouts_reintroduced=False)
        return NativeLineSections(int(frame["index"]),int(frame["native_pts"]),row["_target_key"],fid,
            queries,normals,measured,tuple(ids),receipt,
            tuple(row.get("_line_end_extent_geometry_columns",())),
            tuple(row.get("_line_start_extent_geometry_columns",())))
    accepted_uv=np.concatenate([r["uv"]for r in frame["rows"]])[group["point_indices"]]
    estimator=group.get("source_estimator_revision")
    if estimator is not None or line.get("estimator_contract") is not None:
        from .first_shot_source_evidence import rebuild_final_native_line_estimator
        original_uv=accepted_uv
        if estimator is not None:
            from .first_shot_revision import ESTIMATOR_SCHEMA
            if (estimator.get("schema")!=ESTIMATOR_SCHEMA
                    or estimator["source_packet"]!=frame["source_packet"]
                    or estimator["native_readout_packet"]!=native_ref
                    or estimator["source_feature_id"]!=fid
                    or tuple(m["source_point_id"]for m in estimator["original_members"])!=operator.source_point_ids):
                raise FiniteLineCoverageError("Derived estimator lacks its same original support/lineage: "+fid)
            original_uv=np.asarray([m["original_UV"]for m in estimator["original_members"]])
        derived=rebuild_final_native_line_estimator(line,fid,operator.source_point_ids,original_uv,
            group_id=operator.group_id)
        if estimator is not None and (estimator["estimator_contract"]!=derived["metadata"]
                or not np.array_equal(accepted_uv,derived["uv"])):
            raise FiniteLineCoverageError("Derived mean and W do not share the actual final native support: "+fid)
        rebuilt=derived["operator"]
        receipt.update(derived_correlated_estimator_replacement=estimator is not None,
            original_approved_UV_preserved_as_lineage=estimator is not None,
            exact_source_mean_differential_normal_factor=derived["metadata"]["physical_sensitivity_normal_projection_factor"])
    else:
        origin=np.asarray(line["point_origin"]);tangent=np.asarray(line["unit_tangent"])
        normal=np.asarray(line["unit_normal"])-line["normal_offset_coefficients"][0]*tangent
        normal/=np.linalg.norm(normal)
        native_uv=np.asarray([r["uv"]for r in actual]);sigma=np.asarray([r["sigma_px"]for r in actual])
        if not np.isfinite(sigma).all()or np.any(sigma<=0.):raise ValueError("Actual estimator readout scales missing")
        station=(native_uv-origin)@tangent
        design=np.column_stack((station,np.ones(len(station))))
        noise=_influence(design,np.ones(len(station)))*sigma
        basis=np.zeros((len(accepted_uv),2*len(accepted_uv)))
        sensitivity=[]
        for j,p in enumerate(accepted_uv):
            basis[j,2*j:2*j+2]=normal
            sensitivity.append([float((p-origin)@tangent),1.])
        rebuilt=latent_whitening_operator(operator.source_point_ids,np.asarray(sensitivity),
            explicit_noise_factor=noise,physical_basis=basis,group_id=operator.group_id,
            covariance_origin=operator.covariance_origin)
    # Native membership and the estimator differential remain fixed when a
    # separately authenticated localization covariance replaces profile width.
    # Certify that original estimator first, then certify the new covariance;
    # do not restore the old W or mistake its scale for material identity.
    support_operator=operator
    noise_revision=group.get("localization_noise_revision")
    if noise_revision is not None:
        from .source_evidence_whitening import SourceWhiteningOperator
        from .first_shot_noise_revision import _noise_operator, source_members_digest
        support_operator=SourceWhiteningOperator.from_record(group["original_support_operator"])
        declared_original=SourceWhiteningOperator.from_record(noise_revision["original_operator"])
        if (support_operator.source_point_ids!=operator.source_point_ids
                or support_operator.group_id!=operator.group_id
                or not np.array_equal(support_operator.matrix,declared_original.matrix)
                or noise_revision["source_members_sha256"]!=source_members_digest(frame,group["point_indices"])):
            raise FiniteLineCoverageError("Localization noise changed finite-coverage source membership: "+fid)
        expected_current=_noise_operator(noise_revision,support_operator)
        if not np.array_equal(expected_current.matrix,operator.matrix):
            raise FiniteLineCoverageError("Finite coverage received a different localization covariance: "+fid)
    support_gram=support_operator.matrix.T@support_operator.matrix
    error=np.linalg.norm(rebuilt.matrix.T@rebuilt.matrix-support_gram)
    relative=error/max(np.linalg.norm(support_gram),np.finfo(float).tiny)
    if relative>1e-10:
        raise FiniteLineCoverageError("Native support does not reproduce the current frozen line W",dict(
            index=frame["index"],source_feature_id=fid,relative_Gram_error=float(relative)))
    receipt.update(current_formal_estimator_group_id=operator.group_id,
        current_formal_point_ids=list(operator.source_point_ids),current_full_W_sha256=hashlib.sha256(operator.matrix.tobytes()).hexdigest(),
        original_estimator_support_Gram_relative_error=float(relative),
        localization_noise_replacement_authenticated=noise_revision is not None,
        current_W_native_support_Gram_relative_error=0. if noise_revision is not None else float(relative),
        actual_queries_are_existing_full_W_estimator_native_support=True,
        human_excluded_representatives_or_new_readouts_reintroduced=False)
    return NativeLineSections(int(frame["index"]), int(frame["native_pts"]), row["_target_key"], fid,
        queries, normals, measured, tuple(ids), receipt,
        tuple(row.get("_line_end_extent_geometry_columns", ())),
        tuple(row.get("_line_start_extent_geometry_columns", ())))


def compile_native_line_sections(frames, *, extent_columns=None):
    """Read actual direct_readouts once; no model/representative adds a section.

    Only explicit continuous ice-line rows with declared movable ends are admitted. Their physical target
    is authenticated by the existing source association, while the native
    query is taken from that material's real current source packet. Missing
    readouts remain explicit unsupported records, never inferred from accepted
    representatives. check_native_line_coverage rejects an incomplete inventory.
    """
    from .official_static_surface_dataset import read_reference
    cache = {}
    def read(ref):
        key = ref["path"], ref["sha256"]
        if key not in cache:
            cache[key] = read_reference(ref)
        return cache[key]
    allowed = None if extent_columns is None else set(map(int, extent_columns))
    compiled = [];unsupported = []
    for frame in frames:
        selected = {}
        for row in frame["rows"]:
            if not row.get("_continuous_world_line", False):
                continue
            columns=(*row.get("_line_end_extent_geometry_columns", ()), *row.get("_line_start_extent_geometry_columns", ()))
            if not columns or (allowed is not None and not allowed.intersection(columns)):
                continue
            target = row["target"]
            if target.get("kind") != "ice_curve" or row["point_correspondence"]:
                raise ValueError("Explicit continuous ice material curve required")
            key = row["_target_key"]
            selected.setdefault(key, row)
        if not selected:
            continue
        try:
            packet = read(frame["source_packet"])
            if int(packet["source"]["native_pts"]) != int(frame["native_pts"]):
                raise ValueError("Native query packet and FIT exposure clocks differ")
            native_ref = packet.get("ice_source_readout_packet")
            if native_ref is None:
                raise FiniteLineCoverageError("Actual ice readout packet is missing")
            native_packet=read(native_ref)
            native_source=native_packet["source"]
            if (int(native_source["native_pts"])!=int(frame["native_pts"])
                    or int(native_source["actual_pts"])!=int(frame["native_pts"])
                    or native_source["time_base"]!="1/12800"
                    or native_source["source_video_sha256"]!=packet["source"]["source_video_sha256"]
                    or native_source["decoded_BGR_sha256"]!=packet["source"]["decoded_BGR_sha256"]):
                raise ValueError("Actual native profile packet has foreign exposure or source pixels")
            features = _feature_table(native_packet)
        except (ValueError,KeyError,OSError) as error:
            for key,row in selected.items():
                unsupported.append(dict(index=frame["index"],target_key=key,reason=str(error),
                    source_packet=frame["source_packet"],source_reference_not_rewritten=True,
                    extent_columns=[*row.get("_line_end_extent_geometry_columns",()),*row.get("_line_start_extent_geometry_columns",())]))
            continue
        for key,row in selected.items():
            fid="ice."+row["target"]["curve_id"]
            try:
                compiled.append(_compile_one(frame,row,features.get(fid),native_ref))
            except (ValueError,KeyError) as error:
                unsupported.append(dict(index=frame["index"],target_key=key,source_feature_id=fid,reason=str(error),
                    source_packet=frame["source_packet"],actual_native_readout_packet=native_ref,
                    extent_columns=[*row.get("_line_end_extent_geometry_columns",()),*row.get("_line_start_extent_geometry_columns",())]))
    return NativeLineCoverageInventory(tuple(compiled),tuple(unsupported),None if allowed is None else tuple(sorted(allowed)))


def _line_chart(world):
    from .first_shot_source_kernel import _line_chart as original_chart
    world = np.asarray(world, float)
    if world.ndim != 2 or world.shape[1:] != (3,) or len(world) < 2 or not np.isfinite(world).all():
        raise ValueError("Unchanged finite ordered world line required")
    origin, unit, length, *_ = original_chart(world, np.empty((*world.shape, 0)), (), ())
    return origin, unit, length


def _central_seeds(camera, origin, unit, sections):
    from .first_shot_source_kernel import _line_polynomials, _interfaces
    if np.asarray(camera).shape != (25,) or np.any(np.asarray(camera)[15:] != 0.):
        raise ValueError("Native finite coverage currently supports the actual Brown4 camera25 only")
    px, py, start, direction, image_unit = _line_polynomials(camera, origin, unit)
    stations, owners = [], []
    tangents = np.column_stack((sections.profile_normal[:, 1], -sections.profile_normal[:, 0]))
    eps = np.finfo(float).eps
    for index, (query, tangent) in enumerate(zip(sections.query_uv, tangents, strict=True)):
        polynomial = tangent[0] * (px-query[0]) + tangent[1] * (py-query[1])
        roots = polynomial.roots()
        real = roots.real[abs(roots.imag) <= 8.*np.sqrt(eps)*np.maximum(1., abs(roots.real))]
        for image_station in real:
            denominator = image_station*direction[2] - image_unit @ direction[:2]
            if denominator == 0.:
                continue
            station = (image_unit @ start[:2] - image_station*start[2]) / denominator
            if np.isfinite(station):
                stations.append(station); owners.append(index)
    station = np.asarray(stations, float)
    owner = np.asarray(owners, int)
    if not len(station):
        return station, owner, np.empty((0, 2))
    uv, depth, valid = _interfaces()[0](camera, origin+station[:, None]*unit, with_jacobian=False)
    physical = valid & (depth > .05) & np.isfinite(uv).all(axis=1)
    return station[physical], owner[physical], uv[physical]


def finite_line_query_intersections(camera25, world, sections, *, row_context=None,
                                    readout_milliseconds=0.):
    """Return verified finite forward intersections and unsupported addresses.

    The only tolerances below are floating point root/optical branch precision.
    There is no material pixel threshold. Nonzero row timing without the actual
    native row context is an error, never silently evaluated as a central frame.
    """
    if not isinstance(sections, NativeLineSections):
        raise TypeError("Once-compiled actual NativeLineSections required")
    if not np.isfinite(readout_milliseconds):
        raise ValueError("Finite signed native readout time required")
    if readout_milliseconds != 0. and row_context is None:
        raise ValueError("Nonzero native row timing requires its actual row-camera context")
    if row_context is not None and float(row_context.readout_milliseconds) != float(readout_milliseconds):
        raise ValueError("Source coverage and physical row timing differ")
    camera = np.asarray(camera25, float)
    world = np.asarray(world, float)
    origin, unit, length = _line_chart(world)
    station, owner, uv = _central_seeds(camera, origin, unit, sections)
    rows = uv[:, 1].copy()
    if readout_milliseconds != 0.:
        # A central-frame root outside the sensor may seed a valid rolling-row
        # root inside it. Only the Newton seed is bounded; final projection,
        # sensor, finite-material, and self-row checks below stay unchanged.
        rows = np.clip(rows, 1e-3, 1079.-1e-3)
    tangents = np.column_stack((sections.profile_normal[:, 1], -sections.profile_normal[:, 0]))
    physical = np.isfinite(rows) & (rows >= 0.) & (rows <= 1079.)
    station, owner, rows = station[physical], owner[physical], rows[physical]
    if readout_milliseconds != 0. and len(station):
        from .row_scene_projection import _batch_project
        active = np.ones(len(station), bool)
        converged = np.zeros(len(station), bool)
        # Central Brown roots are initialization only. The accepted root uses
        # its own y, then verified with the same physical row-camera projector.
        ds = max(1e-5, length*1e-7)
        dy = 1e-3
        for _ in range(18):
            ids = np.flatnonzero(active & ~converged)
            if not len(ids):
                break
            current_y = rows[ids]
            cameras = row_context.camera25_at_rows(current_y, geometry_navigation=False)
            points = origin+station[ids, None]*unit
            pixels, depths, valid = _batch_project(cameras, points)
            g = np.einsum("ij,ij->i", pixels-sections.query_uv[owner[ids]], tangents[owner[ids]])
            h = pixels[:, 1]-current_y
            done = valid & (depths > .05) & (abs(g) <= 1e-7) & (abs(h) <= 1e-7)
            converged[ids[done]] = True
            live = valid & (depths > .05) & ~done & (current_y >= dy) & (current_y <= 1079.-dy)
            active[ids[~live & ~done]] = False
            ids = ids[live]
            if not len(ids):
                continue
            current_y = rows[ids];points = origin+station[ids, None]*unit
            cameras = row_context.camera25_at_rows(current_y, geometry_navigation=False)
            pplus, _, vplus = _batch_project(cameras, points+ds*unit)
            pminus, _, vminus = _batch_project(cameras, points-ds*unit)
            yplus = row_context.camera25_at_rows(current_y+dy, geometry_navigation=False)
            yminus = row_context.camera25_at_rows(current_y-dy, geometry_navigation=False)
            pu, _, vu = _batch_project(yplus, points)
            pd, _, vd = _batch_project(yminus, points)
            ds_uv = (pplus-pminus)/(2.*ds)
            dy_uv = (pu-pd)/(2.*dy)
            tn = tangents[owner[ids]]
            a = np.einsum("ij,ij->i", ds_uv, tn)
            b = np.einsum("ij,ij->i", dy_uv, tn)
            c = ds_uv[:, 1];d = dy_uv[:, 1]-1.
            determinant = a*d-b*c
            good = vplus & vminus & vu & vd & np.isfinite(determinant) & (abs(determinant) > 1e-12)
            active[ids[~good]] = False
            ids = ids[good]
            if not len(ids):
                continue
            a,b,c,d,determinant = [x[good] for x in (a,b,c,d,determinant)]
            gl,hl = g[live][good],h[live][good]
            step_s = (-d*gl+b*hl)/determinant
            step_y = (c*gl-a*hl)/determinant
            next_y = rows[ids]+step_y
            inside = np.isfinite(next_y) & np.isfinite(step_s) & (next_y >= 0.) & (next_y <= 1079.)
            active[ids[~inside]] = False
            station[ids[inside]] += step_s[inside]
            rows[ids[inside]] = next_y[inside]
        from .row_scene_projection import _batch_project
        cameras = row_context.camera25_at_rows(rows, geometry_navigation=False)
        uv, depth, valid = _batch_project(cameras, origin+station[:, None]*unit)
        row_error = abs(uv[:, 1]-rows)
        valid &= active & converged & (row_error <= 1e-6)
    elif len(station):
        from .first_shot_source_kernel import _interfaces
        uv, depth, valid = _interfaces()[0](camera, origin+station[:, None]*unit, with_jacobian=False)
        row_error = np.zeros(len(station))
    else:
        uv=np.empty((0,2));depth=np.empty(0);valid=np.empty(0,bool);row_error=np.empty(0)
    query_error = abs(np.einsum("ij,ij->i", uv-sections.query_uv[owner], tangents[owner]))
    valid &= (depth > .05) & np.isfinite(uv).all(axis=1) & (query_error <= 1e-6)
    valid &= (uv[:, 0] >= 0.) & (uv[:, 0] < 1920.) & (uv[:, 1] >= 0.) & (uv[:, 1] <= 1079.)
    arithmetic = 128.*np.finfo(float).eps*max(1., length, np.max(abs(world)))
    finite = valid & (station >= -arithmetic) & (station <= length+arithmetic)
    records=[];missing=[]
    for j,pid in enumerate(sections.query_ids):
        candidates = np.flatnonzero(owner == j)
        supported = candidates[finite[candidates]]
        if not len(supported):
            missing.append(pid)
        records.append(dict(query_id=pid,query_uv=sections.query_uv[j].tolist(),
            supported=bool(len(supported)),candidates=[dict(station_scene=float(station[k]),
                fraction=float(station[k]/length),UV=uv[k].tolist(),forward_central_valid=bool(valid[k]),
                inside_finite_line=bool(finite[k]),sensor_row_error_px=float(row_error[k]),
                query_section_error_px=float(query_error[k])) for k in candidates if np.isfinite(uv[k]).all()]))
    return dict(passed=not missing,index=sections.index,native_pts=sections.native_pts,
        source_feature_id=sections.source_feature_id,target_key=sections.target_key,
        actual_native_query_count=len(sections.query_ids),missing_queries=missing,
        world_line_length_scene=float(length),readout_milliseconds=float(readout_milliseconds),
        intersection_exposure_row="actual implicit sensor-row root" if readout_milliseconds else "central zero-readout",
        finite_endpoint_observed=False,lower_bound_semantics=True,queries=records)


def check_native_line_coverage(state, q, sections, worlds, *, detailed=False, raise_on_failure=True):
    """Check a prepared source inventory without re-reading source packets/W."""
    from .first_shot_row_camera import prepare_frame_row_camera
    if isinstance(sections,NativeLineCoverageInventory) and sections.unsupported:
        report=dict(passed=False,reason="requested extent source inventory is incomplete",unsupported=list(sections.unsupported))
        if raise_on_failure:raise FiniteLineCoverageError(report["reason"],report)
        return report
    contexts={};camera_cache={};failures=[];results=[]
    for evidence in sections:
        index=evidence.index
        if index not in camera_cache:
            camera_cache[index]=state.camera25(q,index)
            contexts[index]=(None if getattr(state,"row_readout_index",None) is None else
                             prepare_frame_row_camera(state,q,index))
        context=contexts[index]
        readout=0. if context is None else context.readout_milliseconds
        result=finite_line_query_intersections(camera_cache[index],worlds[evidence.target_key],evidence,
                                               row_context=context,readout_milliseconds=readout)
        if not result["passed"]:
            failures.append(result)
        if detailed:results.append(result)
    report=dict(passed=not failures,source_line_groups=len(sections),
        actual_native_queries=sum(len(s.query_ids)for s in sections),failed_groups=len(failures),
        failures=failures,source_UV_sigma_full_W_unchanged=True,extra_statistical_rows=0,
        source_metadata_reparsed=False,finite_endpoint_values_not_claimed_measured=True)
    if detailed:report["results"]=results
    if failures and raise_on_failure:
        raise FiniteLineCoverageError("Finite physical material does not cover actual native query sections",report)
    return report


def read_coverage_revision(ref, state, frames):
    """Opt-in authenticated scope; an old binding creates no coverage domain."""
    from .official_static_surface_dataset import read_reference
    policy=read_reference(ref)
    if (policy.get("schema")!="rinkspace.finite-native-line-coverage-revision.v1"
            or policy.get("source_video_sha256")!=state.binding["source"]["sha256"]
            or policy.get("source_clock")!=state.source_clock
            or policy.get("frame_range")!=[1,state.frame_count]
            or policy.get("domain")!="actual_native_profile_query_forward_finite_intersection"
            or policy.get("initializer") not in ("only_monotonic_existing_endpoint_extension",
                "source_coverage_with_existing_initial_endpoint_values")
            or policy.get("source_UV_sigma_W_modified")is not False
            or policy.get("unknown_terminal_endpoint_is_measured")is not False):
        raise ValueError("Authenticated finite native material coverage revision required")
    columns=policy.get("extent_columns")
    if (not isinstance(columns,list)or not columns or any(type(x)is not int for x in columns)
            or len(set(columns))!=len(columns)):
        raise ValueError("Distinct declared endpoint columns required")
    declared={int(c)for frame in frames for row in frame["rows"]
        for c in (*row.get("_line_end_extent_geometry_columns",()),*row.get("_line_start_extent_geometry_columns",()))}
    # The physical producer owns all four independent ends even when some
    # have no active native source in this shot. Such ends acquire no invented
    # query or initialization change merely by being named in the policy.
    owner=getattr(getattr(state,"provider",None),"current",None)
    declared.update(map(int,getattr(owner,"lane_right_extension_indices",{}).values()))
    if not set(columns).issubset(declared):
        raise ValueError("Coverage policy refers to an undeclared movable line endpoint")
    inventory=compile_native_line_sections(frames,extent_columns=columns)
    if policy["initializer"]=="source_coverage_with_existing_initial_endpoint_values":
        inventory=replace(inventory,initializer_strategy="source_required_and_existing_bound_initial_values")
    proof=dict(reference=ref,extent_columns=columns,source_line_groups=len(inventory),
        actual_native_queries=sum(len(s.query_ids)for s in inventory),unsupported=list(inventory.unsupported),
        source_UV_sigma_full_W_unchanged=True,extra_statistical_rows=0,
        source_metadata_compiled_once=True,actual_terminal_values_measured=False,
        initializer=policy["initializer"],existing_initial_end_values_are_hard_bounds=False,
        columns_without_active_query_support=sorted(set(columns)-{c for s in inventory for c in (*s.end_columns,*s.start_columns)}),
        scope=[dict(index=s.index,native_pts=s.native_pts,source_feature_id=s.source_feature_id,
            target_key=s.target_key,query_ids=list(s.query_ids),evidence=s.evidence)for s in inventory])
    if inventory.unsupported:
        raise FiniteLineCoverageError("Selected finite-coverage scope has unsupported authenticated query evidence",proof)
    return inventory,proof


def initialize_native_line_coverage(state,q,inventory,world_builder):
    """Monotonic affine endpoint lift; no clipping or guessed true end.

    Every other q byte is retained. A required root before a fixed start, after
    a fixed end, on an unsupported optical branch, or beyond the existing ice
    material domain is a reported failure. End-coordinate units are verified by
    their actual physical producer displacement, not a common step in q units.
    An explicitly selected existing-initial-value policy restores that value
    only on endpoints requiring actual coverage extension. It is an initializer,
    never a continuing physical lower bound or additional endpoint measurement.
    """
    before=np.asarray(q,float).copy()
    if isinstance(inventory,NativeLineCoverageInventory)and inventory.unsupported:
        raise FiniteLineCoverageError("Cannot initialize incomplete native query coverage",dict(unsupported=list(inventory.unsupported)))
    worlds=world_builder(before)
    initial=check_native_line_coverage(state,before,inventory,worlds,raise_on_failure=False)
    if initial["passed"]:
        return q,dict(already_covered=True,all_parameter_values_byte_exact=True,
            source_line_groups=len(inventory),actual_native_queries=initial["actual_native_queries"],
            source_UV_sigma_full_W_unchanged=True,terminal_values_measured=False)
    by_group={(s.index,s.target_key):s for s in inventory}
    derivatives={};requirements={};changes=[]
    for failed in initial["failures"]:
        source=by_group[failed["index"],failed["target_key"]]
        original=worlds[source.target_key];origin,unit,length=_line_chart(original)
        for query, measured in zip(failed["queries"],source.measured_uv,strict=True):
            if query["supported"]:continue
            candidates=[p for p in query["candidates"]if p["forward_central_valid"]]
            if not candidates:
                raise FiniteLineCoverageError("Actual native query has no resolved forward physical intersection",failed)
            chosen=min(candidates,key=lambda p:np.linalg.norm(np.asarray(p["UV"])-measured))
            station=chosen["station_scene"]
            side="start" if station < 0. else "end"
            cols=source.start_columns if side=="start"else source.end_columns
            if len(cols)!=1:
                raise FiniteLineCoverageError("Required native material lies beyond a fixed or ambiguous endpoint",dict(query=query,source=source.evidence))
            col=cols[0];key=source.target_key,col
            if key not in derivatives:
                delta=1e-4;probe=before.copy();probe[col]+=delta
                changed=world_builder(probe)[source.target_key]
                endpoint=0 if side=="start"else-1;fixed=-1 if side=="start"else 0
                vector=(changed[endpoint]-original[endpoint])/delta
                velocity=float(vector@unit)
                tolerance=1e-8*max(1.,np.linalg.norm(vector))
                if (np.linalg.norm(changed[fixed]-original[fixed])>1e-10
                        or np.linalg.norm(vector-velocity*unit)>tolerance
                        or(side=="end"and velocity<=0.)or(side=="start"and velocity>=0.)):
                    raise FiniteLineCoverageError("Declared positive q extension does not preserve this physical line chart",dict(column=col,target_key=source.target_key))
                farther=before.copy();farther[col]+=10.*delta
                check=world_builder(farther)[source.target_key]
                if np.linalg.norm((check[endpoint]-original[endpoint])/(10.*delta)-vector)>tolerance:
                    raise FiniteLineCoverageError("Endpoint producer is not affine in its declared q chart",dict(column=col))
                derivatives[key]=velocity
            velocity=derivatives[key]
            needed=(station-length)/velocity if side=="end"else station/velocity
            if needed<0.:raise ValueError("Coverage initialization would shrink a physical material extent")
            requirements[col]=max(requirements.get(col,0.),needed)
            changes.append(dict(column=int(col),source_index=source.index,query_id=query["query_id"],
                side=side,required_positive_q_extension=float(needed),endpoint_velocity_scene_per_q_unit=float(velocity)))
    after=before.copy();selections=[]
    restore=(isinstance(inventory,NativeLineCoverageInventory)
             and inventory.initializer_strategy=="source_required_and_existing_bound_initial_values")
    for col,delta in requirements.items():
        minimum=np.nextafter(before[col]+delta,np.inf)
        original_initial=float(state.initial_q[col]) if restore else None
        after[col]=minimum if original_initial is None else max(minimum,original_initial)
        selections.append(dict(column=int(col),minimum_source_required_q=float(minimum),
            existing_bound_initial_q=original_initial,selected_q=float(after[col]),
            selection="existing_parameter_initializer" if original_initial is not None and original_initial>minimum else "minimum_actual_query_coverage",
            existing_initial_value_is_measured_endpoint=False,existing_initial_value_is_hard_lower_bound=False))
    allowed=np.array(sorted(requirements),int);keep=np.ones(len(before),bool);keep[allowed]=False
    if not np.array_equal(before[keep],after[keep])or np.any(after[allowed]<before[allowed]):
        raise ValueError("Finite material initialization changed unrelated parameters or shortened material")
    state.guard(after)
    final_world=world_builder(after)
    final=check_native_line_coverage(state,after,inventory,final_world,raise_on_failure=False)
    if not final["passed"]:
        raise FiniteLineCoverageError("Minimum endpoint extension did not produce complete physical coverage",dict(initial=initial,final=final,requirements=requirements))
    return after,dict(input_q_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
        output_q_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
        changed_columns=allowed.tolist(),changes=[dict(column=int(col),before=float(before[col]),after=float(after[col]))for col in allowed],
        initializer_strategy="source_required_and_existing_bound_initial_values" if restore else "minimum_source_coverage",
        endpoint_initializer_selections=selections,
        binding_constraints=changes,initial_coverage=initial,final_coverage=final,
        all_non_endpoint_parameter_values_byte_exact=True,only_monotonic_extension=True,
        parameter_values_not_clipped=True,source_UV_sigma_full_W_unchanged=True,
        unknown_terminal_values_are_only_source_supported_lower_bounds=True,accuracy_accepted=False)
