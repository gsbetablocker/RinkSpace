"""Accepted native exposures on the existing physical state and public solver.

No historical pixel factor enters the objective. Source whitening stays frozen;
world finite differences are shared by all exposures, and each target is
projected once per exposure by the exact source kernel.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import time

for _threads in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads,"1")
import numpy as np
from scipy.sparse import csr_matrix, vstack

from .complete_official_fit_dataset import (_direct_dataset, _extension_contract,
    _compatible_prefix_model, _fit_frame)
from .continuous_board_association import with_continuous_board_association
from .official_static_surface_dataset import iter_observations, read_reference, reference
from .first_shot_physical_state import FirstShotPhysicalState, GeometryDomainError
from .first_shot_source_kernel import evaluate_projected_targets


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _target_key(target):
    target = deepcopy(target)
    if target["kind"] == "board_edge" and "association_policy" in target:
        target.pop("part", None)
    return json.dumps(target, sort_keys=True, separators=(",", ":"))


def _continuous_line_metadata(target, scene):
    """Only exact scene primitives declare a continuous finite world line."""
    if target.get("kind") != "ice_curve":
        return {}
    curve = target.get("curve_id", "")
    fid, separator, part = curve.partition(":")
    features = [f for f in scene["ice_features"] if f["id"] == fid]
    if len(features) != 1 or not separator:
        return {}
    feature = features[0]
    straight = False
    if feature["kind"] == "polyline" and part.startswith("segment"):
        index = int(part[len("segment"):])
        segments = feature.get("segments_xyz", [])
        straight = 0 <= index < len(segments) and np.asarray(segments[index]).shape == (2, 3)
    elif feature["kind"] == "polygon" and fid in {"lane"+str(i) for i in range(1, 5)}:
        if part in ("edge0", "edge1", "edge2"):
            index = int(part[-1])
            straight = np.asarray(feature["paint_segments_xyz"][index]).shape == (2, 3)
    if not straight:
        return {}
    result = {"_continuous_world_line": True}
    connections = {f"house_connection_F{i}_R{i}:segment0": 93+i for i in range(1, 5)}
    if curve in connections:
        result["_line_end_extent_geometry_columns"] = [connections[curve]]
    elif fid in {"lane"+str(i) for i in range(1, 5)} and part == "edge0":
        result["_line_end_extent_geometry_columns"] = [22]
    elif fid in {"lane"+str(i) for i in range(1, 5)} and part == "edge2":
        result["_line_start_extent_geometry_columns"] = [22]
    return result


def _model_association_override_table(binding_reference, data):
    """Target identity and its material-normal evidence must bind together."""
    binding = read_reference(binding_reference)
    ref = binding.get("model_association_overrides")
    if ref != data.get("model_association_overrides"):
        raise ValueError("Model association overrides and corrected FIT evidence must share the same reference")
    if ref is None:
        return {}, None
    fix = read_reference(ref)
    if (fix.get("schema") != "rinkspace.accepted-source-point-identity-overrides.v1"
            or fix.get("model") != binding.get("model")
            or fix.get("source_video_sha256") != binding["source"]["sha256"]
            or fix.get("source_time_base") != binding["source_clock"]["time_base"]
            or fix.get("original_source_packets_bindings_UV_IDs_and_approvals_modified") is not False):
        raise ValueError("Authenticated user-confirmed source identity override required")
    registry = read_reference(fix["feature_registry"])["targets"]
    table = {}
    for item in fix["overrides"]:
        key = item["native_pts"], item["source_point_id"]
        if (key in table or item.get("identity_confirmed_by_user") is not True
                or item.get("model_target_descriptor") != registry.get(item["model_target_id"])
                or item.get("point_correspondence") is not False
                or item["time_base"] != fix["source_time_base"]):
            raise ValueError("Duplicate, unconfirmed, or foreign source identity override")
        table[key] = item
    if len(table) != fix["overridden_point_count"]:
        raise ValueError("Confirmed source identity inventory changed")
    return table, ref


def _association_target(item, observation, index):
    if (item["native_pts"] != observation["native_pts"]
            or item["time_base"] != observation["time_base"]
            or item["source_packet"] != observation["source_packet"]
            or item["source_point_id"] != observation["source_point_ids"][index]
            or item["original_source_feature_id"] != observation["source_feature_id"]
            or item["original_source_segment_index"] != observation["source_segment_index"]
            or item["original_source_vertex_index"] != observation["original_vertex_indices"][index]
            or item["original_source_model_target_id"] != observation["target_id"]
            or not np.array_equal(np.asarray(item["approved_source_UV"],float),observation["uv"][index])):
        raise ValueError("Model identity override changed its original native source address or UV")
    return deepcopy(item["model_target_descriptor"])


def _validate_corrected_association_record(record, changed_ids, association_ref):
    if not changed_ids:return
    evidence = record.get("corrected_evidence", {})
    if (evidence.get("model_association_overrides") != association_ref
            or set(evidence.get("source_point_ids", [])) != changed_ids
            or evidence.get("source_estimator_B_and_noise_factor_L_preserved") is not True
            or evidence.get("physical_material_residual_basis_only_changed") is not True):
        raise ValueError("Changed material target lacks the matching corrected source-normal operator evidence")


def _load_frames(binding_reference, policy):
    """Authenticate original source rows without repeatedly building worlds."""
    _, data = _direct_dataset(binding_reference)
    association_table, association_ref = _model_association_override_table(binding_reference, data)
    used_associations = set()
    base_ref, base = _direct_dataset(data["base_dataset"])
    scopes = [read_reference(base["scope_selection"]), read_reference(data["appended_scope_selection"])]
    _extension_contract(base, *scopes)
    scene, templates = _compatible_prefix_model(*scopes)
    if (data["frame_range"] != [1, 1248] or data["fit_points"] != 192725
            or data["parent_complete_state"] != base["parent_complete_state"]
            or data["prefix_fit_points"] != base["fit_points"]
            or data["added_fit_points"] != sum(f["point_count"] for f in data["appended_frames"])
            or data["legacy_objectives_enabled"] is not False
            or data["source_UV_modified"] is not False):
        raise ValueError("Unchanged complete accepted first-shot inventory required")
    frames, descriptors = [], OrderedDict()
    pairs = [(base["scope_selection"], scopes[0], base["frames"]),
             (data["appended_scope_selection"], scopes[1], data["appended_frames"])]
    for scope_ref, scope, records in pairs:
        if len(records) != len(scope["frames"]):
            raise ValueError("Source frame inventory changed")
        by_pts = {}
        for record, source in zip(records, scope["frames"], strict=True):
            if (record["index"] != source["index"] or record["native_pts"] != source["native_pts"]
                    or record["source_packet"] != source["packet"]
                    or record["source_binding"] != source["binding"]
                    or record["point_count"] != source["original_points"]
                    or record["native_pts"] in by_pts):
                raise ValueError("Foreign or duplicate accepted source frame")
            by_pts[record["native_pts"]] = record
        current, rows = None, []
        def finish(pts, rows):
            record = by_pts[pts]
            changed_ids = {pid for exposure,pid in association_table if exposure == pts}
            _validate_corrected_association_record(record, changed_ids, association_ref)
            ids = [pid for r in rows for pid in r["source_point_ids"]]
            if set(ids) != set(record["source_scales_by_point_id"]):
                raise ValueError("Source uncertainty inventory changed")
            frame = _fit_frame(record, rows, scene, templates)
            frame["index"] = record["index"]
            frame.pop("scene"); frame.pop("templates")
            frame["source_packet"] = record["source_packet"]
            frames.append(frame)
        for o in iter_observations(scope_ref, role="all_scope"):
            pts = o["native_pts"]
            if current is not None and current != pts:
                finish(current, rows); rows = []
            current = pts
            sigma = np.array([by_pts[pts]["source_scales_by_point_id"][pid] for pid in o["source_point_ids"]])
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError("Accepted source scale is missing")
            # Split only contiguous runs in the original point order. Frozen
            # group point_indices continue to address the original flattening.
            runs = []
            for j,pid in enumerate(o["source_point_ids"]):
                override_key = pts,pid; item = association_table.get(override_key)
                target = _association_target(item,o,j) if item is not None else deepcopy(o["model_target"])
                if item is not None:used_associations.add(override_key)
                if target["kind"] == "board_edge":
                    target = with_continuous_board_association(target, policy)
                key = _target_key(target); descriptors.setdefault(key, target)
                if runs and runs[-1][2] == key:runs[-1][1] = j+1
                else:runs.append([j,j+1,key,target])
            for first,stop,key,target in runs:
                row = dict(target=target, _target_key=key, id=o["binding_observation_id"],
                    uv=o["uv"][first:stop].copy(), sigma=sigma[first:stop].copy(),
                    point_correspondence=o["point_correspondence"])
                row.update(_continuous_line_metadata(target, scene))
                row.update({k:o[k] for k in ("native_pts", "actual_pts", "time_base", "source_video_sha256",
                    "source_packet", "source_feature_id", "source_segment_index")})
                row.update(source_point_ids=o["source_point_ids"][first:stop],
                    original_vertex_indices=o["original_vertex_indices"][first:stop])
                if any((pts,pid) in association_table for pid in row["source_point_ids"]):
                    row.update(model_association_overrides=association_ref,
                        original_model_target=deepcopy(o["model_target"]),original_model_target_id=o["target_id"],
                        original_binding_observation_id=o["binding_observation_id"])
                if row["point_correspondence"] != (target["kind"] == "ad_corner"):
                    raise ValueError("Physical point/curve correspondence changed")
                rows.append(row)
        if current is not None:
            finish(current, rows)
    if ([f["index"] for f in frames] != list(range(1,1249))
            or sum(f["source_point_count"] for f in frames) != data["fit_points"]):
        raise ValueError("An accepted point or native exposure was lost")
    if used_associations != set(association_table):
        raise ValueError("A confirmed source identity override is missing from the actual FIT point inventory")
    return frames, descriptors


class FrozenWhitening:
    """Same rectangular source matrices, batched without changing membership."""
    def __init__(self, frame):
        n = frame["source_point_count"] * 2
        self.groups, used, cursor = [], set(), 0
        buckets = OrderedDict()
        scales = []
        for group in frame["whitening_groups"]:
            indices = np.ravel([[2*i,2*i+1] for i in group["point_indices"]])
            matrix = group["operator"].matrix
            stop = cursor + len(matrix)
            self.groups.append((indices, matrix, slice(cursor,stop)))
            buckets.setdefault(matrix.shape, []).append((indices, matrix, slice(cursor,stop)))
            # A fixed source direction expressed in pixel-equivalent units.
            # This does not modify source sigma or the normalized objective.
            scales.extend(1. / np.linalg.norm(matrix, axis=1))
            used.update(indices); cursor = stop
        self.buckets = [(np.stack([x[0] for x in values]), np.stack([x[1] for x in values]),
                         [x[2] for x in values]) for values in buckets.values()]
        self.ordinary = np.array([i for i in range(n) if i not in used], int)
        self.ordinary_scales = np.repeat(np.concatenate([r["sigma"] for r in frame["rows"]]),2)[self.ordinary]
        self.ordinary_start = cursor
        self.count = cursor + len(self.ordinary)
        self.scales = np.r_[scales,self.ordinary_scales]
        if not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("Finite frozen source-direction scale required")
    def apply(self, value):
        result = np.empty((self.count,) + value.shape[1:])
        for indices, weights, destinations in self.buckets:
            selected = value[indices]
            if value.ndim == 1:
                selected = selected[...,None]
                whitened = np.matmul(weights,selected)[...,0]
            else:
                whitened = np.matmul(weights,selected)
            for destination, block in zip(destinations,whitened,strict=True):
                result[destination] = block
        ordinary = value[self.ordinary]
        result[self.ordinary_start:] = ordinary / (self.ordinary_scales if value.ndim == 1 else self.ordinary_scales[:,None])
        return result


_geometry_worker = None
_source_worker = None
def _init_geometry(binding_path, descriptors):
    global _geometry_worker
    from calibration.solver_runtime import install,configure_native_runtime
    install(); configure_native_runtime(stage="first_shot_geometry_worker",native_threads=3)
    import cv2
    cv2.setNumThreads(1)
    _geometry_worker = FirstShotPhysicalState.from_binding(binding_path), descriptors


def _geometry_sample(task):
    q, column, shift = task
    state, descriptors = _geometry_worker
    candidate = q.copy(); candidate[column] += shift
    try:
        state.guard_geometry(candidate)
    except GeometryDomainError as error:
        return column, shift, None, str(error)
    # A missing target, invalid world interface, or failed evaluator is not a
    # finite-difference domain side and must propagate as its original error.
    return column, shift, state.world_paths(candidate, descriptors), None


class _CameraState(FirstShotPhysicalState):
    """Only the shared physical camera chart, without a geometry factory."""
    def __init__(self,state):
        self.initial_q=state.initial_q.copy()
        self.count=state.count
        self.frame_count=state.frame_count
        self.view_stop=getattr(state,"view_stop",state.count)
        self.world_parameter_indices=getattr(state,"world_parameter_indices",np.arange(state.geometry_count)).copy()
        self.world_parameter_count=len(self.world_parameter_indices)
        self.scene_unit_working_m=state.scene_unit_working_m
        self.base_calibration=state.base_calibration
        self.zoom_parameter_indices=np.asarray(getattr(state,"zoom_parameter_indices",()),dtype=int).copy()
        self.zoom_configuration=deepcopy(getattr(state,"zoom_configuration",None))
        self.zoom_reference=deepcopy(getattr(state,"zoom_reference",None))
        self.row_readout_index=getattr(state,"row_readout_index",None)
        if self.row_readout_index is not None:
            self.pts=state.pts.copy()
            self.source_clock=deepcopy(state.source_clock)
            self.row_timing_configuration=deepcopy(state.row_timing_configuration)
    def __getstate__(self):
        values=dict(self.__dict__)
        values["base_calibration"]=self.base_calibration.manifest()
        return values
    def __setstate__(self,values):
        from .z190_physical_camera import SharedCalibration
        values=dict(values)
        values["base_calibration"]=SharedCalibration.from_manifest(values["base_calibration"])
        self.__dict__.update(values)


def _init_source(camera_state, frames, whitening):
    global _source_worker
    from calibration.solver_runtime import install,configure_native_runtime
    install()
    camera_state.worker_backend=configure_native_runtime(stage="first_shot_source_worker",native_threads=3)
    import cv2
    cv2.setNumThreads(1)
    _source_worker = camera_state, frames, whitening


def _evaluate_frame(state,q,frame,whitening,worlds,derivatives):
    if getattr(state,"row_readout_index",None) is not None:
        from .first_shot_row_source import evaluate_row_timed_frame
        return evaluate_row_timed_frame(state,q,frame,whitening,worlds,derivatives)
    camera,columns,chain=state.camera15_mapping(q,frame["index"])
    raw,cam,geom=evaluate_projected_targets(camera,frame["rows"],worlds,derivatives,
        jacobian=derivatives is not None)
    residual=whitening.apply(raw)
    if derivatives is None:return residual,None
    local=np.c_[geom,cam@chain]
    whitened=whitening.apply(local)
    column_map=np.r_[getattr(state,"world_parameter_indices",np.arange(state.geometry_count)),columns]
    sparse=csr_matrix(whitened)
    sparse=csr_matrix((sparse.data,column_map[sparse.indices].astype(np.int32),sparse.indptr),
        shape=(len(residual),state.count))
    sparse.sort_indices()
    return residual,sparse


def _source_chunk(task):
    q,worlds,derivatives,indices=task
    state,frames,whitening=_source_worker
    results=[_evaluate_frame(state,q,frames[i],whitening[i],worlds,derivatives) for i in indices]
    r=np.concatenate([x[0] for x in results])
    J=vstack([x[1] for x in results],format="csr") if derivatives is not None else None
    return r,J,dict(pid=os.getpid(),backend=state.worker_backend)


def _raw_source_chunk(task):
    q,worlds,indices=task
    state,frames,_=_source_worker
    values=[]
    for index in indices:
        frame=frames[index]
        if getattr(state,"row_readout_index",None) is not None:
            from .first_shot_row_source import row_timed_raw
            raw=row_timed_raw(state,q,frame,worlds)
        else:
            camera=state.camera25(q,frame["index"])
            raw,_,_=evaluate_projected_targets(camera,frame["rows"],worlds,jacobian=False)
        values.append(np.linalg.norm(raw.reshape(-1,2),axis=1))
    return np.concatenate(values)


class FirstShotSourceBundle:
    def __init__(self, binding_path, policy, *, geometry_workers=8,source_workers=6):
        self.state = FirstShotPhysicalState.from_binding(binding_path)
        if (policy.get("source_binding") != self.state.binding_reference
                or policy.get("frame_range") != [1,self.state.frame_count]
                or policy.get("source_clock") != self.state.source_clock):
            raise ValueError("Continuous board association must bind this exact source dataset and clock")
        self.policy = deepcopy(policy)
        self.frames, self.descriptors = _load_frames(self.state.binding_reference,policy)
        self.source_revision_proof = None
        if self.state.binding.get("source_revision") is not None:
            from .first_shot_revision import revise_frames
            self.frames, self.descriptors, self.source_revision_proof = revise_frames(
                self.frames, self.state.binding["source_revision"], self.state.binding, policy=policy)
        self.source_noise_revision_proof = None
        if self.state.binding.get("source_localization_noise_revision") is not None:
            from .first_shot_noise_revision import revise_localization_noise
            self.frames, self.source_noise_revision_proof = revise_localization_noise(
                self.frames, self.state.binding["source_localization_noise_revision"], self.state.binding)
        self.source_native_line_revision_proof = None
        if self.state.binding.get("source_native_line_revision") is not None:
            from .first_shot_native_line_revision import revise_native_lines
            self.frames, self.source_native_line_revision_proof = revise_native_lines(
                self.frames, self.state.binding["source_native_line_revision"], self.state.binding)
        self.keys = tuple(self.descriptors)
        self.target_descriptors = tuple(self.descriptors.values())
        self.initial_target_proof = self.state.validate_initial_targets(self.target_descriptors)
        self.whitening = [FrozenWhitening(f) for f in self.frames]
        self.scales = np.concatenate([w.scales for w in self.whitening])
        self.quadratic = np.zeros(len(self.scales), bool)
        self.geometry_workers = int(geometry_workers)
        self.source_workers = int(source_workers)
        self._geometry_pool = None
        self._source_pool = None
        self._world_cache = OrderedDict()
        self.timings = []
        self.worker_receipts={}
        self.ledgers = []
        for frame, whitening in zip(self.frames,self.whitening,strict=True):
            h = hashlib.sha256()
            for row in frame["rows"]:
                h.update(json.dumps(row["source_point_ids"],separators=(",",":")).encode())
                h.update(row["uv"].tobytes()); h.update(row["sigma"].tobytes())
            for group in frame["whitening_groups"]:
                h.update(group["operator"].matrix.tobytes())
            self.ledgers.append(dict(family="accepted_official_static_source",programme_index=frame["index"],
                native_pts=frame["native_pts"],time_base="1/12800",source_points=frame["source_point_count"],
                formal_rows=whitening.count,source_inventory_sha256=h.hexdigest(),
                source_packet=frame["source_packet"],association_policy=deepcopy(policy)))
        if len(self.scales) != sum(w.count for w in self.whitening):
            raise ValueError("Frozen FIT operators changed their declared row dimension")
        self.finite_coverage_inventory = None
        self.finite_coverage_revision_proof = None
        self._finite_coverage_cache = OrderedDict()
        coverage_reference = self.state.binding.get("finite_source_line_coverage_revision")
        if coverage_reference is not None:
            from .finite_source_line_coverage import read_coverage_revision
            self.finite_coverage_inventory,self.finite_coverage_revision_proof = read_coverage_revision(
                coverage_reference,self.state,self.frames)

    def close(self):
        if self._geometry_pool is not None:
            self._geometry_pool.shutdown(wait=True); self._geometry_pool = None
        if self._source_pool is not None:
            self._source_pool.shutdown(wait=True); self._source_pool = None

    def world(self,q,jacobian=False):
        world_indices = getattr(self.state,"world_parameter_indices",np.arange(self.state.geometry_count))
        geo = np.asarray(q[world_indices],float)
        token = geo.tobytes()
        if token in self._world_cache and (not jacobian or self._world_cache[token][1] is not None):
            paths, derivatives = self._world_cache[token]
            return paths, derivatives if jacobian else None
        started = time.perf_counter()
        self.state.guard_geometry(geo)
        paths = self.state.world_paths(geo,self.target_descriptors)
        worlds = dict(zip(self.keys,paths,strict=True)); derivatives = None
        if jacobian:
            if self._source_pool is not None:
                self._source_pool.shutdown(wait=True); self._source_pool = None
            h = float(self.state.provider.fd_step)
            samples = {}; failures = {}; ng = len(world_indices); stencils = []
            # Exact existing producer: A=Ftee, B=Rtee+f*(Rtee-Ftee),
            # f=108/945+q[col]. Only these actual terminal-extent columns
            # are analytic; no column is discarded because its J is small.
            expected_connections = {f"house_connection_F{i}_R{i}": 93+i for i in range(1, 5)}
            actual_connections = self.state.provider.current.lane_right_extension_indices
            for identity, column in expected_connections.items():
                if actual_connections.get(identity) != column:
                    raise ValueError("Canonical independent Lane extent producer changed: " + identity)
            analytic_columns = set(expected_connections.values())
            layout_columns = set(range(self.state.geometry_count,ng))
            tasks = [(geo,col,sign*h) for col in range(ng) if col not in analytic_columns for sign in (1.,-1.)]
            tasks = [task for task in tasks if task[1] not in layout_columns]
            if self.geometry_workers > 1:
                if self._geometry_pool is None:
                    self._geometry_pool = ProcessPoolExecutor(self.geometry_workers,
                        mp_context=multiprocessing.get_context("spawn"),initializer=_init_geometry,
                        initargs=(self.state.binding_reference["path"],self.target_descriptors))
                results = self._geometry_pool.map(_geometry_sample,tasks,chunksize=1)
            else:
                def serial():
                    for value,col,shift in tasks:
                        sample=value.copy(); sample[col]+=shift
                        try:
                            self.state.guard_geometry(sample)
                        except GeometryDomainError as error:
                            yield col,shift,None,str(error)
                        else:
                            yield col,shift,self.state.world_paths(sample,self.target_descriptors),None
                results=serial()
            for col,shift,values,failure in results:
                samples[col,shift>0] = values
                if failure is not None:failures[col,shift>0]=failure
            derivatives={key:np.zeros(value.shape+(ng,)) for key,value in worlds.items()}
            for col in range(ng):
                if col in layout_columns:
                    stencils.append(dict(column=col,global_column=int(world_indices[col]),
                        method="analytic_normalized_homography_chain_batched_material_UV_FD",
                        material_UV_fd_step=h,source_or_world_column_removed=False))
                    continue
                if col in analytic_columns:
                    identity = next(k for k,v in expected_connections.items() if v == col)
                    for key, descriptor in self.descriptors.items():
                        if descriptor.get("kind") != "ice_curve" or descriptor.get("curve_id") != identity+":segment0":
                            continue
                        points = worlds[key]
                        fraction = 108./945.+geo[col]
                        station = np.linspace(0.,1.,len(points))
                        # Already in scene units; no additional scale conversion.
                        endpoint_derivative = (points[-1]-points[0])/(1.+fraction)
                        expected = points[0]+station[:,None]*(points[-1]-points[0])
                        roundoff = 128*np.finfo(float).eps*np.maximum(1.,np.abs(points))
                        if not np.all(np.abs(points-expected) <= roundoff):
                            raise ValueError("Declared physical house connection is not an exact continuous world line: " + identity)
                        derivatives[key][...,col] = station[:,None]*endpoint_derivative
                    stencils.append(dict(column=col,method="analytic_current_house_connection_terminal_extent",
                        producer_formula="dA=0;dB=(B-A)/(1+108/945+q[column]);dP(t)=t*dB",
                        source_or_world_column_removed=False))
                    continue
                plus,minus=samples[col,True],samples[col,False]
                if plus is None and minus is None:
                    raise GeometryDomainError("Both original shared-geometry FD sides leave the physical domain: column "
                        + str(col) + "; +h: " + failures[col,True] + "; -h: " + failures[col,False])
                if plus is not None and minus is not None:
                    before,after,denominator,method=minus,plus,2*h,"central_original_step"
                elif plus is not None:
                    before,after,denominator,method=paths,plus,h,"forward_original_step"
                else:
                    before,after,denominator,method=minus,paths,h,"backward_original_step"
                for key,p,m in zip(self.keys,after,before,strict=True):
                    if p.shape!=worlds[key].shape or m.shape!=p.shape:
                        raise ValueError("World finite difference changed the physical target inventory")
                    derivatives[key][...,col]=(p-m)/denominator
                stencils.append(dict(column=col,method=method,original_step=h,
                    plus_domain_failure=failures.get((col,True)),minus_domain_failure=failures.get((col,False))))
            # The original rectangular Lane producer has a common right-end
            # coordinate q22. It moves only the terminal vertex of each long
            # side, while q21 is a transverse width and remains observable.
            # Preserve other targets' q22 finite differences; overwrite only
            # these proved endpoint derivatives with their physical formula.
            for key, descriptor in self.descriptors.items():
                curve = descriptor.get("curve_id", "")
                if descriptor.get("kind") != "ice_curve" or curve not in {
                    f"lane{i}:edge{edge}" for i in range(1, 5) for edge in (0, 2)}:
                    continue
                if self.state.provider.parameter_names[22] != "four_Lane_paint_right_end_from_Rtee_m":
                    raise ValueError("Canonical shared Lane paint-end producer changed")
                points = worlds[key]; station = np.linspace(0., 1., len(points))
                ca = np.array([np.cos(geo[14]), np.sin(geo[14]), 0.]) / self.state.scene_unit_working_m
                edge = int(curve[-1]); signed = 1. if edge == 0 else -1.
                transverse = points[-1]-points[0]-signed*np.linalg.norm(points[-1]-points[0])*ca/np.linalg.norm(ca)
                if np.linalg.norm(transverse) > 128*np.finfo(float).eps*np.maximum(1.,np.abs(points)).max():
                    raise ValueError("Canonical straight Lane paint direction changed: " + curve)
                derivatives[key][...,22] = (station if edge == 0 else 1.-station)[:,None]*ca
            if any(d.get("curve_id", "") in {f"lane{i}:edge{edge}" for i in range(1,5) for edge in (0,2)}
                   for d in self.target_descriptors):
                stencils[22]["exact_target_endpoint_overrides"] = dict(
                    producer="current_structured_Lane_paint", parameter="four_Lane_paint_right_end_from_Rtee_m",
                    edge0="dA=0;dB=ca/scene_unit;dP(t)=t*dB",
                    edge2="dA=ca/scene_unit;dB=0;dP(t)=(1-t)*dA",
                    other_target_derivatives_preserved=True, width_column21_preserved=True)
            if layout_columns:
                layout_derivatives = self.state.print_layout_world_jacobians(geo,self.target_descriptors)
                nlayout = len(layout_columns)
                for key,value in zip(self.keys,layout_derivatives,strict=True):
                    if value.shape != worlds[key].shape+(nlayout,):
                        raise ValueError("Shared print layout derivative changed the physical target inventory")
                    derivatives[key][...,self.state.geometry_count:] = value
        self._world_cache[token]=worlds,derivatives
        while len(self._world_cache)>2:self._world_cache.popitem(last=False)
        self.timings.append(dict(stage="shared_world_derivatives" if jacobian else "shared_world_values",
            seconds=time.perf_counter()-started,geometry_workers=self.geometry_workers,
            original_geometry_fd_step=float(self.state.provider.fd_step),
            geometry_derivative_stencils=stencils if jacobian else []))
        return worlds,derivatives

    def frame_result(self,q,index,worlds,derivatives=None):
        self.finite_coverage_guard(q,worlds)
        frame,whitening=self.frames[index],self.whitening[index]
        return _evaluate_frame(self.state,q,frame,whitening,worlds,derivatives)

    def finite_coverage_guard(self,q,worlds=None):
        """Source domain checked once per exact full q, with prepared metadata."""
        if getattr(self,"finite_coverage_inventory",None) is None:
            return None
        key=np.asarray(q,np.float64).tobytes()
        if key in self._finite_coverage_cache:
            return self._finite_coverage_cache[key]
        from .finite_source_line_coverage import check_native_line_coverage
        if worlds is None:worlds=self.world(q,False)[0]
        started=time.perf_counter()
        result=check_native_line_coverage(self.state,q,self.finite_coverage_inventory,worlds)
        self.timings.append(dict(stage="finite_native_material_coverage",seconds=time.perf_counter()-started,
            source_line_groups=result["source_line_groups"],actual_native_queries=result["actual_native_queries"]))
        self._finite_coverage_cache[key]=result
        while len(self._finite_coverage_cache)>2:self._finite_coverage_cache.popitem(last=False)
        return result

    def initialize_finite_coverage(self,q):
        if getattr(self,"finite_coverage_inventory",None) is None:
            return q,None
        from .finite_source_line_coverage import initialize_native_line_coverage
        return initialize_native_line_coverage(self.state,q,self.finite_coverage_inventory,
                                               lambda value:self.world(value,False)[0])

    def physical_guard(self,q):
        self.state.guard(q)
        self.finite_coverage_guard(q)
        return True

    def source_blocks(self,q,jacobian):
        from z190_streaming_source_normal import SourceRowBlock
        worlds,derivatives=self.world(q,jacobian)
        self.finite_coverage_guard(q,worlds)
        started=time.perf_counter()
        chunks=[list(map(int,c)) for c in np.array_split(np.arange(len(self.frames)),max(1,self.source_workers)) if len(c)]
        if self.source_workers>1:
            if self._source_pool is None:
                self._source_pool=ProcessPoolExecutor(self.source_workers,
                    mp_context=multiprocessing.get_context("spawn"),initializer=_init_source,
                    initargs=(_CameraState(self.state),self.frames,self.whitening))
            results=self._source_pool.map(_source_chunk,[(q,worlds,derivatives,c) for c in chunks],chunksize=1)
        else:
            def serial():
                for indices in chunks:
                    values=[self.frame_result(q,i,worlds,derivatives) for i in indices]
                    yield np.concatenate([x[0] for x in values]),vstack([x[1] for x in values],format="csr") if jacobian else None,None
            results=serial()
        for indices,(r,j,receipt) in zip(chunks,results,strict=True):
            if receipt is not None:self.worker_receipts[receipt["pid"]]=receipt
            scales=np.concatenate([self.whitening[i].scales for i in indices])
            ledger=dict(family="accepted_native50_contiguous_frames",frames=[self.ledgers[i] for i in indices])
            yield SourceRowBlock(r,j,np.zeros(len(r),bool),scales,ledger)
        self.timings.append(dict(stage="source_J" if jacobian else "source_residuals",
            seconds=time.perf_counter()-started,source_workers=self.source_workers,blocks=len(chunks)))

    def raw_errors(self,q):
        """All original points in source order, in native pixels, without W."""
        worlds,_=self.world(q,False)
        chunks=[list(map(int,c)) for c in np.array_split(np.arange(len(self.frames)),max(1,self.source_workers)) if len(c)]
        if self.source_workers>1:
            if self._source_pool is None:
                self._source_pool=ProcessPoolExecutor(self.source_workers,
                    mp_context=multiprocessing.get_context("spawn"),initializer=_init_source,
                    initargs=(_CameraState(self.state),self.frames,self.whitening))
            result=np.concatenate(list(self._source_pool.map(_raw_source_chunk,[(q,worlds,c) for c in chunks],chunksize=1)))
        else:
            values=[]
            for frame in self.frames:
                if getattr(self.state,"row_readout_index",None) is not None:
                    from .first_shot_row_source import row_timed_raw
                    raw=row_timed_raw(self.state,q,frame,worlds)
                else:
                    raw,_,_=evaluate_projected_targets(self.state.camera25(q,frame["index"]),frame["rows"],worlds,jacobian=False)
                values.append(np.linalg.norm(raw.reshape(-1,2),axis=1))
            result=np.concatenate(values)
        if result.shape!=(sum(f["source_point_count"] for f in self.frames),) or not np.isfinite(result).all():
            raise ValueError("Every approved source point requires a finite raw pixel diagnostic")
        return result

    def evaluate(self,q,jacobian=True):
        blocks=list(self.source_blocks(q,jacobian))
        r=np.concatenate([b.r for b in blocks])
        return (r,vstack([b.J for b in blocks],format="csr")) if jacobian else r

    def initialization_report(self):
        report = dict(schema="rinkspace.first-shot-physical-fit-initialization.v1",status="ready",
            physical_state=self.state.proof,source_binding=self.state.binding_reference,
            source_frames=len(self.frames),source_points=sum(f["source_point_count"] for f in self.frames),
            formal_fit_components=len(self.scales),target_paths=len(self.descriptors),
            continuous_board_association=self.policy,initial_targets=self.initial_target_proof,
            all_source_UV_ID_sigma_and_whitening_preserved=self.source_noise_revision_proof is None,
            historical_pixel_factors_connected_to_active_FIT=False,
            scale_semantics="fixed_source_direction_pixel_equivalent_for_numerical_step_only",
            source_scales_are_localization_support_not_calibrated_noise=True,
            source_localization_noise_revision=self.source_noise_revision_proof,
            source_native_line_revision=self.source_native_line_revision_proof,
            source_revision=self.source_revision_proof,
            material_template_revision=self.state.material_revision_proof,
            old_objective_J_H_profile_or_damping_reused=False)
        if self.finite_coverage_revision_proof is not None:
            report["finite_source_line_coverage_revision"]=self.finite_coverage_revision_proof
        return report


def load_seed_state(bundle, path, sha256, association_reference, consumer_snapshot=None):
    """Authenticate a full same-objective state; rebuild all numerical factors."""
    path = Path(path).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Seed checkpoint SHA256 differs from the declared archive")
    with np.load(BytesIO(data), allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "rinkspace.joint-source-LM-checkpoint.v1":
            raise ValueError("Unsupported complete source checkpoint")
        identity = json.loads(str(saved["identity_json"].item()))
        canonical_proof = json.loads(json.dumps(bundle.state.proof))
        if (identity.get("schema") != "rinkspace.first-shot-native50-physical-source-graph.v1"
                or identity.get("input") != bundle.state.binding_reference
                or identity.get("association") != association_reference
                or identity.get("physical_state") != canonical_proof):
            raise ValueError("Seed source, clock, model or physical parameter chart differs")
        expected = {"scales": bundle.scales, "quadratic": bundle.quadratic,
            "aux__native_pts": bundle.state.pts, "aux__parent_q": bundle.state.parent_q,
            "aux__parent_shared_slots": bundle.state.parent_shared_slots}
        baseline_key = "aux__original_8a_initial_q" if "aux__original_8a_initial_q" in saved else "initial_q"
        expected[baseline_key] = bundle.state.initial_q
        for name, value in expected.items():
            actual = saved[name]
            if actual.dtype != np.asarray(value).dtype or not np.array_equal(actual, value):
                raise ValueError("Seed exact source/initialization array changed: " + name)
        q = saved["q"].copy()
        if q.dtype != np.float64 or q.shape != (bundle.state.count,) or not np.isfinite(q).all():
            raise ValueError("Seed must retain all finite float64 physical and layout coordinates")
        damping = float(saved["damping"].item())
        fraction = float(saved["next_line_search_fraction"].item())
        if not np.isfinite(damping) or damping <= 0 or not 0 < fraction <= 1:
            raise ValueError("Seed numerical damping or line-search seed is invalid")
    producers = {Path(item["path"]).name: item for item in identity["producer"]}
    required = {"first_shot_fit.py", "first_shot_physical_state.py", "first_shot_source_kernel.py",
        "first_shot_print_layout.py", "first_shot_alignment_monitor.py", "render_result.py", "ad_landmarks.py"}
    current_binding = getattr(bundle.state, "binding", {})
    if current_binding.get("source_revision") or current_binding.get("material_template_revision"):
        required.add("first_shot_revision.py")
    if current_binding.get("row_timing_model"):
        required.update(("first_shot_row_camera.py", "first_shot_row_source.py",
                         "row_scene_projection.py", "board_grid_overlay.py"))
    if current_binding.get("mount_working_range_revision"):
        required.add("first_shot_mount_revision.py")
    if current_binding.get("finite_source_line_coverage_revision"):
        required.add("finite_source_line_coverage.py")
    if current_binding.get("shared_zoom_camera"):
        required.update(("first_shot_zoom_camera.py", "z190_physical_camera.py"))
    if (getattr(bundle, "source_revision_proof", None) or {}).get("correlated_estimator_replacements", 0):
        required.update(("first_shot_source_evidence.py", "source_evidence_whitening.py"))
    if current_binding.get("source_localization_noise_revision"):
        required.update(("first_shot_noise_revision.py", "source_evidence_whitening.py"))
    if current_binding.get("source_native_line_revision"):
        required.update(("first_shot_native_line_revision.py", "first_shot_native_line_kernel.py"))
    if len(producers) != len(identity["producer"]) or not required.issubset(producers):
        raise ValueError("Seed lacks its source-computation producer contract")
    for name in required - {"first_shot_fit.py", "first_shot_alignment_monitor.py"}:
        if reference(producers[name]["path"]) != producers[name]:
            raise ValueError("Seed source-computation producer changed: " + name)
    original = producers["first_shot_fit.py"]
    if reference(__file__)["sha256"] != original["sha256"]:
        if consumer_snapshot is None or reference(consumer_snapshot)["sha256"] != original["sha256"]:
            raise ValueError("Changed consumer requires the authenticated previous source snapshot")
        def source_nodes(text):
            tree = ast.parse(text)
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name not in ("main", "load_seed_state")}
        if source_nodes(Path(consumer_snapshot).read_text(encoding="utf8")) != source_nodes(Path(__file__).read_text(encoding="utf8")):
            raise ValueError("Seed source evaluation or world/Jacobian consumer changed")
    if getattr(bundle,"finite_coverage_inventory",None) is None:
        bundle.state.guard(q)
    else:
        bundle.physical_guard(q)
    return q, dict(archive=dict(path=str(path),sha256=sha256),
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        original_8a_initial_q_sha256=hashlib.sha256(bundle.state.initial_q.tobytes()).hexdigest(),
        initial_damping=damping, initial_line_search_fraction=fraction,
        old_history_and_source_reviews_used_as_control=False,
        old_J_H_IRLS_replay_or_profile_reused=False,
        previous_producers=identity["producer"],
        authenticated_consumer_snapshot=None if consumer_snapshot is None else reference(consumer_snapshot))



def _initializer_material_chart_continuation(state, identity):
    """Keep fitted layout deltas only on an authenticated identical chart."""
    schema_keys = {
        "rinkspace.first-shot-native50-physical-source-graph.v1": "input",
        "rinkspace.same-q-partial-noise-one-public-step.v1": "binding",
    }
    key = schema_keys.get(identity.get("schema")) if isinstance(identity, dict) else None
    previous_reference = None if key is None else identity.get(key)
    proof = dict(previous_binding=previous_reference, compatible_material_chart=False,
        compared_chart_fields=["model", "material_template_revision", "print_layout"],
        reason="previous_material_chart_identity_unavailable")
    if previous_reference is None:
        return proof
    # A declared identity with altered bytes must fail, rather than silently
    # falling back to a reset and concealing the broken checkpoint provenance.
    previous = read_reference(previous_reference)
    current = read_reference(state.binding_reference)
    changed = [name for name in proof["compared_chart_fields"]
        if current.get(name) is None or previous.get(name) != current[name]]
    same_source = (current.get("source", {}).get("sha256") is not None
        and previous.get("source", {}).get("sha256") == current["source"]["sha256"]
        and current.get("source_clock") is not None
        and previous.get("source_clock") == current["source_clock"])
    proof.update(compatible_material_chart=not changed and same_source,
        changed_chart_fields=changed, same_source_video_and_clock=same_source,
        reason="same_authenticated_material_chart" if not changed and same_source
            else "material_chart_or_source_changed")
    return proof


def load_parameter_initializer(bundle, path, sha256):
    """Reuse a compatible parameter chart, with fresh factors and solver state."""
    archive = reference(path)
    if archive["sha256"] != sha256:
        raise ValueError("Numerical initializer archive changed")
    with np.load(path, allow_pickle=False) as saved:
        for name, expected in (("aux__native_pts", bundle.state.pts),
                               ("aux__parent_q", bundle.state.parent_q),
                               ("aux__parent_shared_slots", bundle.state.parent_shared_slots)):
            if not np.array_equal(saved[name], expected):
                raise ValueError("Initializer physical chart or native clock changed: " + name)
        q = saved["q"].copy()
        identity = (json.loads(str(saved["identity_json"].item()))
            if getattr(bundle.state, "material_revision_proof", None) is not None
            and "identity_json" in saved else None)
    if q.dtype != np.float64 or q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError("Initializer needs the complete finite physical parameter chart")
    source_q_sha256 = hashlib.sha256(q.tobytes()).hexdigest()
    appended = None
    readout_index = getattr(bundle.state,"row_readout_index",None)
    zoom_columns = np.asarray(getattr(bundle.state,"zoom_parameter_indices",()),dtype=int)
    original_chart_count = int(zoom_columns[0]) if len(zoom_columns) else bundle.state.count
    if readout_index is not None and q.shape == (readout_index,) and original_chart_count == readout_index+1:
        q = np.r_[q,bundle.state.initial_q[readout_index]]
        appended = dict(column=int(readout_index),value=float(q[readout_index]),units="signed_milliseconds",
            reference=bundle.state.row_timing_reference,existing_coordinates_byte_exact=True)
    from .first_shot_zoom_camera import append_zoom_initializer
    q, appended_zoom = append_zoom_initializer(bundle.state, q)
    if q.dtype != np.float64 or q.shape != (bundle.state.count,) or not np.isfinite(q).all():
        raise ValueError("Initializer needs the complete finite physical parameter chart")
    revision = getattr(bundle.state, "material_revision_proof", None)
    continuation = (None if revision is None
        else _initializer_material_chart_continuation(bundle.state, identity))
    reset = ([] if revision is None or continuation["compatible_material_chart"]
        else list(revision["reset_layout_advertisement_ids"]))
    for number in reset:
        q[bundle.state.print_layout_global_indices_by_ad[int(number)]] = 0.
    from .first_shot_mount_revision import initialize_restored_mount
    q, mount_restart = initialize_restored_mount(bundle.state, q)
    coverage_restart = None
    if getattr(bundle,"finite_coverage_inventory",None) is not None:
        q, coverage_restart = bundle.initialize_finite_coverage(q)
    bundle.state.guard(q)
    proof = dict(archive=archive, source_q_sha256=source_q_sha256,
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), reset_source_rebuilt_print_layouts=reset,
        material_chart_continuation=continuation,
        appended_shared_row_timing=appended,
        appended_shared_zoom_camera=appended_zoom,
        restored_user_mount_initialization=mount_restart,
        parameter_values_only=True, old_objective_or_history_or_J_H_IRLS_reused=False,
        initial_damping=1e-3, initial_line_search_fraction=1.)
    if coverage_restart is not None:
        proof["finite_material_coverage_initialization"]=coverage_restart
    return q,proof


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--binding",required=True)
    parser.add_argument("--association",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--geometry-workers",type=int,default=8)
    parser.add_argument("--source-workers",type=int,default=6)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--seed-state")
    parser.add_argument("--seed-state-sha256")
    parser.add_argument("--seed-consumer-snapshot")
    parser.add_argument("--initialize-state")
    parser.add_argument("--initialize-state-sha256")
    parser.add_argument("--freeze-rebuilt-material-layouts", action="store_true")
    parser.add_argument("--allow-local-regression",action="store_true")
    parser.add_argument("--defer-diagnostic-stills", action="store_true",
        help="Keep every full native-point numerical review; render complete-model stills explicitly after selecting a candidate")
    args=parser.parse_args(argv)
    if args.seed_state and (args.resume or not args.seed_state_sha256):
        parser.error("--seed-state requires its SHA256 and cannot be combined with --resume")
    if args.initialize_state and (args.resume or args.seed_state or not args.initialize_state_sha256):
        parser.error("--initialize-state requires its SHA256 and a fresh objective branch")
    from calibration.solver_runtime import install,configure_native_runtime,solve_joint_source
    os.environ["RINKSPACE_NORMAL_WORKERS"]="8"
    install(); backend=configure_native_runtime(stage="first_shot_official50_before_factors",native_threads=3)
    output=Path(args.output).resolve()
    if (args.seed_state or args.initialize_state) and (output.exists() and any(output.iterdir())):
        parser.error("Parameter initialization requires a new output branch; preserve the previous checkpoint")
    output.mkdir(parents=True,exist_ok=True)
    policy_ref=reference(args.association); policy=read_reference(policy_ref)
    started=time.perf_counter()
    try:
        bundle=FirstShotSourceBundle(args.binding,policy,
            geometry_workers=args.geometry_workers,source_workers=args.source_workers)
    except Exception as error:
        if getattr(error,"report",None) is not None:
            write_json(output/"finite_coverage_preparation_failure.json",error.report)
        raise
    write_json(output/"initialization.json",bundle.initialization_report())
    write_json(output/"backend.json",backend)
    try:
        fit_start = bundle.state.initial_q
        seed = None
        if args.seed_state:
            fit_start,seed = load_seed_state(bundle,args.seed_state,args.seed_state_sha256,policy_ref,args.seed_consumer_snapshot)
            write_json(output/"seed_verification.json",seed)
        if args.initialize_state:
            if any(p.name not in ("initialization.json", "backend.json") for p in output.iterdir()):
                raise ValueError("Parameter initialization requires a fresh output directory")
            fit_start,seed = load_parameter_initializer(bundle,args.initialize_state,args.initialize_state_sha256)
            write_json(output/"parameter_initialization.json",seed)
        if bundle.finite_coverage_inventory is not None and not args.initialize_state and not args.seed_state:
            fit_start,coverage_restart=bundle.initialize_finite_coverage(fit_start)
            seed=dict(finite_material_coverage_initialization=coverage_restart,
                parameter_values_only=True,old_objective_or_history_or_J_H_IRLS_reused=False,
                initial_damping=1e-3,initial_line_search_fraction=1.)
            write_json(output/"parameter_initialization.json",seed)
        source_count = sum(f["source_point_count"] for f in bundle.frames)
        fixed_layout_columns = []
        if args.freeze_rebuilt_material_layouts:
            material_proof = bundle.state.material_revision_proof
            if material_proof is None:
                raise ValueError("Only source-reconstructed material layouts may be fixed by this option")
            for number in material_proof["reset_layout_advertisement_ids"]:
                fixed_layout_columns.extend(bundle.state.print_layout_global_indices_by_ad[int(number)].tolist())
            fit_start = fit_start.copy()
            fit_start[fixed_layout_columns] = 0.
        source_coverage = dict(all_active_source_points_in_fit=True, source_revision=bundle.source_revision_proof,
            all_original_source_points_in_fit=bundle.source_revision_proof is None
                or bundle.source_revision_proof["quarantined_correspondences"] == 0)
        if args.prepare_only:
            bundle.physical_guard(fit_start)
            write_json(output/"current.json",dict(status="prepared",optimizer_started=False,
                source_frames=len(bundle.frames),source_points=sum(f["source_point_count"] for f in bundle.frames),
                source_clock=bundle.state.source_clock,source_binding=bundle.state.binding_reference,
                active_parameters=bundle.state.count,source_coverage=source_coverage,
                parent_q_sha256=bundle.state.proof["parent_q_sha256"],
                initial_q_sha256=hashlib.sha256(bundle.state.initial_q.tobytes()).hexdigest(),
                accuracy_accepted=False))
            print(json.dumps(dict(status="ready",frames=len(bundle.frames),points=source_count,parameters=bundle.state.count)),flush=True)
            return
        identity=dict(schema="rinkspace.first-shot-native50-physical-source-graph.v1",
            input=bundle.state.binding_reference,association=policy_ref,physical_state=bundle.state.proof,
            numerical_start=seed,allow_local_regression=args.allow_local_regression,
            calibrated_material_layout_columns_fixed_at_zero=fixed_layout_columns,
            diagnostic_stills_deferred=args.defer_diagnostic_stills,
            producer=[reference(Path(__file__)),reference(Path(__file__).with_name("first_shot_source_kernel.py")),
                      reference(Path(__file__).with_name("first_shot_physical_state.py")),
                      reference(Path(__file__).with_name("first_shot_alignment_monitor.py")),
                      reference(Path(__file__).with_name("first_shot_print_layout.py")),
                      reference(Path(__file__).with_name("first_shot_revision.py")),
                      reference(Path(__file__).with_name("render_result.py")),
                      reference(Path(__file__).with_name("ad_landmarks.py"))])
        if getattr(bundle.state,"row_readout_index",None) is not None:
            identity["producer"].extend(reference(Path(__file__).with_name(name))
                for name in ("first_shot_row_camera.py","first_shot_row_source.py",
                             "row_scene_projection.py","board_grid_overlay.py"))
        if getattr(bundle.state,"mount_revision_proof",None) is not None:
            identity["producer"].append(reference(Path(__file__).with_name("first_shot_mount_revision.py")))
        if getattr(bundle.state,"zoom_reference",None) is not None:
            identity["producer"].append(reference(Path(__file__).with_name("first_shot_zoom_camera.py")))
            identity["producer"].append(reference(Path(__file__).with_name("z190_physical_camera.py")))
        if bundle.finite_coverage_inventory is not None:
            identity["producer"].append(reference(Path(__file__).with_name("finite_source_line_coverage.py")))
            identity["finite_source_line_coverage_revision"]=bundle.finite_coverage_revision_proof
        if (bundle.source_revision_proof or {}).get("correlated_estimator_replacements", 0):
            identity["producer"].extend(reference(Path(__file__).with_name(name))
                for name in ("first_shot_source_evidence.py", "source_evidence_whitening.py"))
        if bundle.source_noise_revision_proof is not None:
            identity["source_localization_noise_revision"] = bundle.source_noise_revision_proof
            existing = {Path(ref["path"]).name for ref in identity["producer"]}
            identity["producer"].extend(reference(Path(__file__).with_name(name))
                for name in ("first_shot_noise_revision.py", "source_evidence_whitening.py") if name not in existing)
        if bundle.source_native_line_revision_proof is not None:
            identity["source_native_line_revision"] = bundle.source_native_line_revision_proof
            existing = {Path(ref["path"]).name for ref in identity["producer"]}
            identity["producer"].extend(reference(Path(__file__).with_name(name))
                for name in ("first_shot_native_line_revision.py", "first_shot_native_line_kernel.py") if name not in existing)
        from calibration.program_tracking.first_shot_alignment_monitor import AlignmentMonitor
        monitor=AlignmentMonitor(bundle,output/"raw_alignment",allow_local_regression=args.allow_local_regression)
        def checkpoint(q,event):
            digest=hashlib.sha256(np.asarray(q,float).tobytes()).hexdigest()
            alignment=monitor.evaluate(q,event)
            if not args.defer_diagnostic_stills:
                monitor.render_diagnostics(q,alignment)
            write_json(output/"current.json",dict(
                status="stopped_raw_alignment_deterioration" if alignment["stop_branch"] else "running",
                q_sha256=digest,event=event,raw_alignment=alignment,
                source_frames=len(bundle.frames),source_points=source_count,source_clock=bundle.state.source_clock,
                source_coverage=source_coverage,
                numerical_acceptance_is_not_accuracy=True,
                complete_model_visual_review_required=True,
                diagnostic_stills_deferred=args.defer_diagnostic_stills))
            write_json(output/"stage_timing.json",bundle.timings)
            write_json(output/"worker_backends.json",list(bundle.worker_receipts.values()))
            return dict(source_frames=len(bundle.frames),source_points=source_count,**source_coverage,
                source_UV_modified=bool((bundle.source_revision_proof or {}).get("correlated_estimator_replacements", 0)),
                source_UV_change_scope="explicit correlated derived means only; original native measurements preserved",
                original_source_packets_preserved=True,
                active_derived_source_means_replaced=bool((bundle.source_revision_proof or {}).get("correlated_estimator_replacements", 0)),
                geometry_and_shared_single_installation_fitted=True,
                raw_alignment=alignment,stop_branch=alignment["stop_branch"],
                reason="All-point native projection deteriorated relative to the unchanged 8a28 initial state"
                       if alignment["stop_branch"] else None,accuracy_accepted=False)
        def physicalguard(q):
            return bundle.physical_guard(q)
        zoom_domain = zoom_recovery = None
        if len(bundle.state.zoom_parameter_indices) and bundle.state.mount_bounds_active:
            from .first_shot_zoom_camera import zoom_mount_linear_constraints, recover_zoom_mount_trial
            zoom_domain = lambda q, selected: zoom_mount_linear_constraints(bundle.state, q, selected)
            zoom_recovery = lambda parent, proposed: recover_zoom_mount_trial(bundle.state, parent, proposed)
        print(f"Starting native50 single-installation FIT: {len(bundle.frames)} exposures, {source_count} active points",flush=True)
        result=solve_joint_source(bundle.evaluate,fit_start,bundle.quadratic,bundle.scales,
            output,physicalguard,source_block_factory=bundle.source_blocks,
            checkpoint_identity=identity,checkpoint_arrays=dict(
                native_pts=bundle.state.pts,parent_q=bundle.state.parent_q,
                parent_shared_slots=bundle.state.parent_shared_slots,
                original_8a_initial_q=bundle.state.initial_q),checkpoint_callback=checkpoint,
            resume=args.resume,initial_damping=1e-3 if seed is None else seed["initial_damping"],global_schur_count=bundle.state.global_shared_count,
            global_schur_indices=bundle.state.global_shared_indices,shared_parameter_bounds=bundle.state.shared_parameter_bounds,
            shared_linear_constraints=zoom_domain, trial_feasibility_recovery=zoom_recovery,
            fixed_parameter_indices=fixed_layout_columns,
            cache_native_source_replay=True,reuse_source_jacobian=True,adaptive_source_backtracking=True,
            schur_backtracking=True,schur_view_lm_ratio=1.,schur_backtracking_aware_damping=True,
            source_stage_timing=True,initial_line_search_fraction=1. if seed is None else seed["initial_line_search_fraction"])
        result_path=output/"result.json"
        write_json(result_path,dict(status=result.status,seconds=time.perf_counter()-started,
            solver_seconds=result.seconds,stop_reason=result.stop_reason,
            q_sha256=hashlib.sha256(np.asarray(result.q,float).tobytes()).hexdigest(),
            accepted_iterations=sum(bool(record.get("accepted")) for record in result.history),
            checkpoint=str(result.checkpoint),accuracy_accepted=False))
        current_path=output/"current.json"
        if current_path.exists():
            current=json.loads(current_path.read_text(encoding="utf8"))
            current.update(status=result.status,result=reference(result_path),accuracy_accepted=False)
            write_json(current_path,current)
    except Exception as error:
        result_path=output/"result.json"
        write_json(result_path,dict(status="failed_exception",seconds=time.perf_counter()-started,
            exception_type=type(error).__name__,message=str(error),
            checkpoint_preserved=(output/"latest_accepted_state.npz").exists(),accuracy_accepted=False))
        current_path=output/"current.json"
        if current_path.exists():
            current=json.loads(current_path.read_text(encoding="utf8"))
            current.update(status="failed_exception",result=reference(result_path),accuracy_accepted=False)
            write_json(current_path,current)
        raise
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
