"""All-point development alignment checks independent of source whitening.

These are the accepted training/development pixels, not independent HELD
evidence.  Every point remains in FIT.  The monitor calls the bundle's exact
raw finite-curve/corner evaluator and compares with the unchanged initial
state; weighted cost is never used as an alignment measurement.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def _summary(errors):
    errors = np.asarray(errors, float)
    return dict(points=len(errors), mean=float(np.mean(errors)), median=float(np.median(errors)),
        rms=float(np.sqrt(np.mean(errors**2))), p90=float(np.quantile(errors, .90)),
        p95=float(np.quantile(errors, .95)), p99=float(np.quantile(errors, .99)),
        maximum=float(np.max(errors)))


def _target_label(target):
    kind = target["kind"]
    if kind == "board_edge":
        return "board." + target["role"] + ("." + str(target["part"]) if target.get("part") else "")
    if kind == "ice_curve":
        return "ice." + target["curve_id"]
    prefix = "Ad" + str(target["advertisement_id"]) + "."
    return prefix + ("print." + target["contour_key"] if kind == "ad_print" else target["role"])


class AlignmentMonitor:
    """Audit the bundle's declared point order, then monitor all raw pixels."""
    def __init__(self, bundle, output_dir, *, allow_local_regression=False):
        self.bundle = bundle
        self.allow_local_regression = bool(allow_local_regression)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_q = np.asarray(bundle.state.initial_q, float).copy()
        self.initial_q_sha256 = hashlib.sha256(self.initial_q.tobytes()).hexdigest()
        self.groups = {"kind": defaultdict(list), "target": defaultdict(list)}
        self.frame_ranges, inventory, scales = [], [], []
        cursor = 0
        frame_indices, frame_pts = set(), set()
        for frame in bundle.frames:
            index, pts = int(frame["index"]), int(frame["native_pts"])
            if index in frame_indices or pts in frame_pts:
                raise ValueError("The raw alignment inventory contains a duplicate exposure")
            frame_indices.add(index); frame_pts.add(pts)
            start, ids = cursor, []
            for row in frame["rows"]:
                points = row["source_point_ids"]
                uv, sigma = np.asarray(row["uv"], float), np.asarray(row["sigma"], float)
                if (not points or len(points) != len(uv) or sigma.shape != (len(points),)
                        or not np.isfinite(sigma).all() or (sigma <= 0).any()
                        or any(not isinstance(point, str) or not point for point in points)):
                    raise ValueError("Every native raw point requires its original ID and finite source support")
                positions = list(range(cursor, cursor + len(points)))
                self.groups["kind"][row["target"]["kind"]].extend(positions)
                self.groups["target"][_target_label(row["target"])].extend(positions)
                ids.extend(points); scales.extend(sigma.tolist()); cursor += len(points)
            if (len(ids) != frame["source_point_count"] or len(set(ids)) != len(ids)):
                raise ValueError("The raw alignment monitor lost or repeated a source point")
            self.frame_ranges.append((index, pts, slice(start, cursor)))
            inactive = frame.get("inactive_source_correspondences", [])
            inventory.append(dict(index=index, native_pts=pts, point_offset=start, point_count=len(ids),
                source_point_ids=ids, all_active_ids_covered=True,
                all_original_ids_covered=not inactive,
                quarantined_source_point_ids=[p["source_point_id"] for p in inactive]))
        if not cursor:
            raise ValueError("All-point alignment requires a nonempty source inventory")
        self.point_count = cursor
        self.source_support = np.asarray(scales, float)
        self.groups = {name: {key: np.asarray(values, int) for key, values in mapping.items()}
                       for name, mapping in self.groups.items()}
        self.inventory_path = self.output_dir / "source_inventory.json"
        _write(self.inventory_path, dict(schema="rinkspace.raw-alignment-source-inventory.v1",
            frame_count=len(inventory), point_count=cursor, order="frames_then_rows_then_source_point_ids",
            point_key="native_pts_and_original_source_point_id", frames=inventory))
        self.inventory_reference = _reference(self.inventory_path)
        point_keys = [(frame["native_pts"], point_id) for frame in inventory for point_id in frame["source_point_ids"]]
        self.target_inventory_path = self.output_dir / "target_inventory.json"
        target_inventory = {}
        for target, positions in self.groups["target"].items():
            keys = [point_keys[int(position)] for position in positions]
            target_inventory[target] = dict(points=len(positions), global_point_offsets=positions.tolist(),
                source_point_key_sha256=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
                all_original_IDs_covered=True)
        _write(self.target_inventory_path, dict(schema="rinkspace.raw-alignment-target-inventory.v1",
            source_inventory=self.inventory_reference,
            membership="global_point_offsets_address_the_exact_native_pts_and_original_source_ID_inventory",
            targets=target_inventory))
        self.target_inventory_reference = _reference(self.target_inventory_path)
        self.baseline = self._raw(self.initial_q)
        self.baseline_summary = _summary(self.baseline)
        self.baseline_path = self.output_dir / "initial_raw_errors.npz"
        np.savez_compressed(self.baseline_path, point_errors_px=self.baseline,
            source_support_px=self.source_support, q=self.initial_q)
        self.baseline_reference = _reference(self.baseline_path)
        self.sequence = 0
        self._initial_renderer = None
        self._verified_native_files = {}

    def _raw(self, q):
        errors = np.asarray(self.bundle.raw_errors(q), float)
        if (errors.shape != (self.point_count,) or not np.isfinite(errors).all() or (errors < 0).any()):
            raise ValueError("The native raw evaluator did not return every original finite point error")
        return errors.copy()

    def _comparison(self, errors, positions):
        current, initial = errors[positions], self.baseline[positions]
        support = self.source_support[positions]
        delta = current - initial
        return dict(current=_summary(current), initial=_summary(initial),
            mean_change_px=float(np.mean(delta)), median_paired_change_px=float(np.median(delta)),
            improved_points=int(np.sum(delta < 0)), worsened_points=int(np.sum(delta > 0)),
            worsened_beyond_own_source_support=int(np.sum(delta > support)),
            improved_beyond_own_source_support=int(np.sum(delta < -support)),
            source_support_is_localization_support_not_calibrated_noise=True)

    def _target_comparison(self, target, errors):
        comparison = self._comparison(errors, self.groups["target"][target])
        comparison.update(target_inventory=self.target_inventory_reference, target_selector=target,
            same_original_point_set_as_initial=True, all_original_source_point_IDs_covered=True)
        return comparison

    def evaluate(self, q, event):
        """Save one complete raw check and return compact checkpoint diagnostics.

        Small floating changes never trigger stopping.  One exposure is already
        a counterexample when its mean/tail worsen and a majority of its points
        deteriorate beyond their own measured supports.  The same test applies
        to each physical target so numerous print pixels cannot conceal a bad
        advertisement edge.  Explicit visual counterevidence can also stop a
        branch, independently of global statistics.
        """
        q = np.asarray(q, float)
        errors = self._raw(q)
        self.sequence += 1
        q_digest = hashlib.sha256(q.tobytes()).hexdigest()
        current = _summary(errors)
        global_comparison = self._comparison(errors, slice(None))
        frames = [dict(index=index, native_pts=pts, **self._comparison(errors, section))
                  for index, pts, section in self.frame_ranges]
        global_dominated = all(current[key] > self.baseline_summary[key]
                               for key in ("mean", "median", "p90", "p95", "p99"))
        worse_frames = sum(frame["current"]["mean"] > frame["initial"]["mean"]
                           and frame["current"]["p95"] > frame["initial"]["p95"] for frame in frames)
        support_majority = global_comparison["worsened_beyond_own_source_support"] > self.point_count / 2
        frame_majority = worse_frames > len(frames) / 2
        contradicted_frames = [frame for frame in frames
            if frame["current"]["mean"] > frame["initial"]["mean"]
            and frame["current"]["p95"] > frame["initial"]["p95"]
            and frame["worsened_beyond_own_source_support"] > frame["current"]["points"] / 2]
        targets = {target: self._target_comparison(target, errors) for target in self.groups["target"]}
        contradicted_targets = [target for target, comparison in targets.items()
            if comparison["current"]["mean"] > comparison["initial"]["mean"]
            and comparison["current"]["p95"] > comparison["initial"]["p95"]
            and comparison["worsened_beyond_own_source_support"] > comparison["current"]["points"] / 2]
        mean_and_tail_worse = all(current[key] > self.baseline_summary[key]
                                 for key in ("mean", "p90", "p95", "p99"))
        visual_worse = event.get("visual_alignment_worse") is True
        local_stop = (bool(contradicted_frames) or bool(contradicted_targets)) and not self.allow_local_regression
        stop = (local_stop
                or (global_dominated and support_majority and frame_majority) or visual_worse)
        requires_review = bool(contradicted_frames) or bool(contradicted_targets) or mean_and_tail_worse or visual_worse
        stem = f"raw_{self.sequence:06d}_{q_digest[:12]}"
        errors_path = self.output_dir / (stem + ".npz")
        np.savez_compressed(errors_path, point_errors_px=errors, q=q)
        report_path = self.output_dir / (stem + ".json")
        compact_event = {key: event[key] for key in ("iteration", "accepted", "checkpoint_kind",
            "visual_alignment_worse") if key in event}
        report = dict(schema="rinkspace.all-point-native-raw-alignment.v1", q_sha256=q_digest,
            initial_q_sha256=self.initial_q_sha256, event=compact_event,
            frame_count=len(frames), point_count=self.point_count,
            all_original_source_point_ids_covered=True, source_inventory=self.inventory_reference,
            baseline_raw_errors=self.baseline_reference, current_raw_errors=_reference(errors_path),
            measurement="unweighted_native_pixel_euclidean_finite_curve_or_physical_corner_distance",
            weighted_cost_used_as_alignment=False, development_source_evidence=True,
            independent_HELD=False, source_points_dropped=0, source_UV_or_weights_modified=False,
            global_comparison=global_comparison,
            by_kind={key: self._comparison(errors, positions) for key, positions in self.groups["kind"].items()},
            by_target=targets,
            frames=frames, consistently_worse_frames=int(worse_frames),
            global_mean_median_and_tail_all_worse=bool(global_dominated),
            majority_worse_beyond_their_own_source_support=bool(support_majority),
            local_source_contradiction_frames=[frame["index"] for frame in contradicted_frames],
            physical_target_source_contradictions=contradicted_targets,
            visual_alignment_worse_confirmed=visual_worse, requires_visual_review=bool(requires_review),
            allow_local_regression=self.allow_local_regression,
            stop_branch=bool(stop), accuracy_accepted=False,
            stop_basis=("single_native_exposure_deteriorates_beyond_its_own_measured_source_support"
                        if contradicted_frames and local_stop else "physical_target_deteriorates_beyond_its_own_measured_source_support"
                        if contradicted_targets and local_stop else "explicit_visual_counterevidence"
                        if visual_worse else "raw_global_and_frame_deterioration_beyond_measured_source_support"
                        if stop and support_majority and frame_majority else
                        None))
        revision = getattr(self.bundle, "source_revision_proof", None)
        if revision is not None:
            report.update(source_revision=revision,
                all_original_source_point_ids_covered=revision["quarantined_correspondences"] == 0,
                all_active_source_point_ids_covered=True,
                original_source_point_history_preserved=True,
                source_points_quarantined_with_evidence=revision["quarantined_correspondences"],
                independently_added_native_readouts=revision["added_native_readouts"],
                source_UV_or_weights_modified=bool(revision.get("rebuilt_correlated_source_groups", 0)),
                original_source_UV_modified=False,
                source_weight_groups_rebuilt_from_native_evidence=revision.get("rebuilt_correlated_source_groups", 0))
        _write(report_path, report)
        diagnostics = dict(report=_reference(report_path), source_frames=len(frames),
            source_points=self.point_count, all_source_points_checked=True,
            unweighted_native_pixel_summary=current, initial_unweighted_native_pixel_summary=self.baseline_summary,
            global_mean_median_and_tail_all_worse=bool(global_dominated),
            worsened_beyond_own_source_support=global_comparison["worsened_beyond_own_source_support"],
            consistently_worse_frames=int(worse_frames),
            local_source_contradiction_frames=[frame["index"] for frame in contradicted_frames],
            physical_target_source_contradictions=contradicted_targets,
            requires_visual_review=bool(requires_review),
            allow_local_regression=self.allow_local_regression,
            stop_branch=bool(stop), independent_HELD=False, accuracy_accepted=False)
        _write(self.output_dir / "latest.json", diagnostics)
        return diagnostics

    def render_diagnostics(self, q, diagnostics):
        """Save public-renderer initial/current views, with every approved UV.

        The first, middle, last and most deteriorated exposure are selected.
        The renderer receives the exact scene/camera used by the source bundle;
        all grid paths, advertisements and advertisement numbers stay enabled.
        Nothing is opened or played by this method.
        """
        import cv2
        from .render_result import SceneRenderer, advertisement_review_paths

        report = json.loads(Path(diagnostics["report"]["path"]).read_text(encoding="utf8"))
        q = np.asarray(q, float)
        digest = hashlib.sha256(q.tobytes()).hexdigest()
        if digest != report["q_sha256"]:
            raise ValueError("The rendered state differs from the checked raw alignment state")
        frame_by_index = {frame["index"]: frame for frame in self.bundle.frames}
        worst = max(report["frames"], key=lambda frame: frame["mean_change_px"])["index"]
        indices = sorted({self.bundle.frames[0]["index"],
                          self.bundle.frames[(len(self.bundle.frames)-1)//2]["index"],
                          self.bundle.frames[-1]["index"], worst})
        scene_factory = getattr(self.bundle.state, "review_scene", self.bundle.state.scene)
        def review_scene(state):
            scene = deepcopy(scene_factory(state))
            descriptor = scene.get("source_conditioned_board_prints")
            if descriptor is None:
                return scene
            represented = {int(item["advertisement_id"]) for item in descriptor["patterns"]}
            represented.update(int(path["number"]) for path in advertisement_review_paths(scene)
                if path.get("primitive") == "polyline" and path.get("pattern_kind"))
            missing = [target for target in getattr(self.bundle,"target_descriptors",())
                if target.get("kind") == "ad_print" and int(target["advertisement_id"]) not in represented]
            if missing:
                paths = self.bundle.state.world_paths(state,missing)
                groups = defaultdict(list)
                for target,path in zip(missing,paths,strict=True):
                    groups[int(target["advertisement_id"])].append(dict(id=target["contour_key"],
                        role="actual_FIT_print_"+target["contour_key"],xyz_scene_units=path.tolist()))
                for number,contours in groups.items():
                    descriptor["patterns"].append(dict(advertisement_id=number,external_contours=contours,
                        actual_material_producer="FirstShotPhysicalState.world_paths:same_q_actual_FIT",
                        source_template_modified=False,independent_print_world_measurements_added=False))
            return scene
        if self._initial_renderer is None:
            initial_scene = review_scene(self.initial_q)
            self._initial_render_identity = dict(
                review_scene_sha256=hashlib.sha256(json.dumps(initial_scene, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest(),
                renderer=_reference(Path(__file__).with_name("render_result.py")),
                material_geometry=_reference(Path(__file__).with_name("ad_landmarks.py")),
                source_overlay_producer=_reference(__file__))
            if getattr(self.bundle.state,'row_readout_index',None) is not None:
                self._initial_render_identity['row_world_projection']=_reference(Path(__file__).with_name('row_scene_projection.py'))
                self._initial_render_identity['native_row_camera']=_reference(Path(__file__).with_name('first_shot_row_camera.py'))
            self._initial_render_token = hashlib.sha256(json.dumps(self._initial_render_identity,
                sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()[:12]
            self._initial_renderer = SceneRenderer(initial_scene,
                draw_grid=True, draw_advertisements=True)
        renderers = [self._initial_renderer,
                     SceneRenderer(review_scene(q), draw_grid=True, draw_advertisements=True)]
        images = []
        for index in indices:
            frame = frame_by_index[index]
            packet_bytes = Path(frame["source_packet"]["path"]).read_bytes()
            if hashlib.sha256(packet_bytes).hexdigest() != frame["source_packet"]["sha256"]:
                raise ValueError("The rendered source packet differs from the accepted immutable reference")
            packet = json.loads(packet_bytes)
            source = packet["source"]
            original = cv2.imread(source["image_path"])
            mask = cv2.imread(source["mask_path"], cv2.IMREAD_GRAYSCALE)
            if original is None or mask is None:
                raise ValueError("The accepted native diagnostic source image or mask is missing")
            file_key = (source["image_path"], source["decoded_BGR_sha256"],
                        source["mask_path"], source["mask_sha256"])
            file_stamps = tuple((Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                                for path in (source["image_path"], source["mask_path"]))
            if self._verified_native_files.get(file_key) != file_stamps and (
                    hashlib.sha256(original.tobytes()).hexdigest() != source["decoded_BGR_sha256"]
                    or hashlib.sha256(Path(source["mask_path"]).read_bytes()).hexdigest() != source["mask_sha256"]):
                raise ValueError("The rendered source image or mask differs from the accepted native exposure")
            self._verified_native_files[file_key] = file_stamps
            panels, references, metrics = [], [], []
            for name, state, renderer in zip(("initial", "current"), (self.initial_q, q), renderers, strict=True):
                path = (self.output_dir / f'view_initial_8a_{self.initial_q_sha256[:12]}_{self._initial_render_token}_{index:04d}.png'
                        if name == "initial" else self.output_dir / f'view_{digest[:12]}_{index:04d}_{name}.png')
                cache_path = path.with_suffix(".json")
                source_uv_hash = hashlib.sha256(b"".join(row["uv"].tobytes() for row in frame["rows"])).hexdigest()
                cache_identity = dict(initial_q_sha256=self.initial_q_sha256, source_packet=frame["source_packet"],
                    approved_source_UV_sha256=source_uv_hash, all_grid_ad_borders_numbers=True,
                    initial_render_producer=self._initial_render_identity)
                cached = (json.loads(cache_path.read_text(encoding="utf8"))
                          if name == "initial" and path.exists() and cache_path.exists() else None)
                if cached and cached["identity"] == cache_identity and _reference(path) == cached["image"]:
                    panel = cv2.imread(str(path))
                    if panel is not None:
                        panels.append(panel); references.append(cached["image"]); metrics.append(cached["renderer_metrics"])
                        continue
                camera = self.bundle.state.camera25(state, index)
                from .row_scene_projection import prepare_render_projection
                projection=prepare_render_projection(self.bundle.state,state,index)
                title=f'{name} | frame {index} | native PTS {frame["native_pts"]} | all grid/ad borders/IDs'
                if projection is None:panel,detail=renderer.draw(original,camera,mask,title)
                else:panel,detail=renderer.draw(original,camera,mask,title,projection_context=projection)
                for row in frame["rows"]:
                    for uv in row["uv"]:
                        point = tuple(np.rint(uv).astype(int))
                        cv2.circle(panel, point, 3, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(panel, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
                for record in frame.get('inactive_source_correspondences',[]):
                    cv2.drawMarker(panel,tuple(np.rint(record['UV']).astype(int)),(0,150,255),cv2.MARKER_TILTED_CROSS,7,1,cv2.LINE_AA)
                if not cv2.imwrite(str(path), panel):
                    raise OSError("Could not save native alignment diagnostic: " + str(path))
                image_reference = _reference(path)
                if name == "initial":
                    _write(cache_path, dict(identity=cache_identity, image=image_reference, renderer_metrics=detail))
                panels.append(panel); references.append(image_reference); metrics.append(detail)
            comparison = self.output_dir / f'view_{digest[:12]}_{index:04d}_comparison.png'
            if not cv2.imwrite(str(comparison), np.hstack(panels)):
                raise OSError("Could not save native alignment comparison: " + str(comparison))
            images.append(dict(index=index, native_pts=frame["native_pts"], source_image=source["image_path"],
                initial=references[0], current=references[1], comparison=_reference(comparison), renderer_metrics=metrics,
                approved_source_points_drawn=sum(len(row["uv"]) for row in frame["rows"]),
                same_model_as_fitted=True, all_grid_and_ad_borders_and_numbers=True))
        report["diagnostic_images"] = images
        report["source_UV_overlay"] = "every_accepted_point_drawn_as_white_dot_with_black_outline"
        report["visual_review_complete"] = False
        if getattr(self.bundle.state,'row_readout_index',None) is not None:
            report['raw_residual_measurement']='observed-row conditional native pixel residual; rendered world uses implicit projected-row root on same physical trajectory; same physical zero, different off-zero Euclidean foot'
        _write(diagnostics["report"]["path"], report)
        diagnostics["report"] = _reference(diagnostics["report"]["path"])
        diagnostics["diagnostic_images"] = images
        _write(self.output_dir / "latest.json", diagnostics)
        return images
