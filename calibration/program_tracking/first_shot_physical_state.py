"""Native-50 first-shot state on the saved shared physical model.

The public shared-geometry constructor reuses temporary legacy initialization
helpers; their pixel factors are discarded and never connected to current FIT.
No complete historical programme graph or profile is restored. New native
exposures own R3/zoom1; the installation owns P3, lever3 and the existing
constant PPx/PPy/k1 chart.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .official_static_surface_dataset import read_reference, reference
from .z190_physical_camera import SharedCalibration, ViewState


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINDING = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/fit.json"
DEFAULT_WORLD_INITIALIZER = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json")
DEFAULT_MOUNT_FACTS = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/camera_hardware_20261001/user_camera_mount_and_height_20261006_v4.json")
INSTALLATION = "okada_sp_installation_0_conditional_z190"


class GeometryDomainError(ValueError):
    """The requested shared geometry leaves an existing physical model domain.

    A finite-difference caller may test the opposite original stencil side.
    This class never represents a missing target or a broken world interface.
    """


def _authenticate(ref):
    actual = reference(ref["path"])
    if actual["sha256"] != ref.get("sha256") or Path(actual["path"]) != Path(ref["path"]).resolve():
        raise ValueError("Immutable physical initialization reference changed: " + ref["path"])
    return Path(ref["path"])


def _print_layout_ids(configuration):
    if configuration is None:return ()
    if configuration.get("schema") == "rinkspace.first-shot-ad2-print-layout.v1":
        if configuration.get("advertisement_id") != 2:
            raise ValueError("The original shared print-layout configuration is Ad2 only")
        return (2,)
    if configuration.get("schema") != "rinkspace.first-shot-print-layouts.v1":
        raise ValueError("An explicit shared print-to-outer-canvas layout configuration is required")
    layouts = configuration.get("layouts")
    if not isinstance(layouts,list) or not layouts:
        raise ValueError("At least one explicitly supported shared print layout is required")
    ids = tuple(item.get("advertisement_id") for item in layouts)
    if (any(type(number) is not int or number not in (2,3,4,5,6,7) for number in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Unique explicit Ad2/3/4/5/6/7 print layouts only; other material interfaces remain unchanged")
    return ids


class FirstShotPhysicalState:
    """118 shared material coordinates, 9 installation values, native RF4."""

    geometry_count = 118
    shared_count = 127
    world_slice = slice(0, 100)
    pattern_slice = slice(100, 109)
    circle_slice = slice(109, 114)
    placement_slice = slice(114, 118)
    installation_slice = slice(118, 127)
    view_start = 127

    @classmethod
    def from_binding(cls, binding_path=DEFAULT_BINDING, *, world_initializer=DEFAULT_WORLD_INITIALIZER):
        return cls(binding_path, world_initializer=world_initializer)

    def __init__(self, binding_path=DEFAULT_BINDING, *, world_initializer=DEFAULT_WORLD_INITIALIZER):
        self.binding_reference = reference(binding_path)
        binding = read_reference(self.binding_reference)
        if (binding.get("frame_range") != [1, 1248]
                or binding.get("source_clock") != {"time_base": "1/12800", "fps": "50/1"}
                or binding.get("all_accepted_points_enter_FIT") is not True
                or binding.get("legacy_objectives_enabled") is not False):
            raise ValueError("Only the complete accepted native-50 first-shot FIT binding is supported")
        self.binding = binding
        self.source_clock = deepcopy(binding["source_clock"])
        self.source = deepcopy(binding["source"])
        self.pts = np.arange(binding["first_native_pts"], binding["last_native_pts"] + 1, 256, dtype=np.int64)
        if len(self.pts) != binding["frame_count"]:
            raise ValueError("Native first-shot exposure inventory differs from the accepted dataset")
        self.frame_count = len(self.pts)
        self.view_stop = self.view_start + 4 * self.frame_count
        self.print_layout_reference = binding.get("print_layout")
        self.print_layout_configuration = (None if self.print_layout_reference is None
            else read_reference(self.print_layout_reference))
        self.print_layout_advertisement_ids = _print_layout_ids(self.print_layout_configuration)
        layout_count = 8*len(self.print_layout_advertisement_ids)
        self.print_layout_stop = self.view_stop + layout_count
        self.row_timing_reference = binding.get("row_timing_model")
        self.row_timing_configuration = (None if self.row_timing_reference is None
            else read_reference(self.row_timing_reference))
        if self.row_timing_configuration is not None:
            timing = self.row_timing_configuration
            if (timing.get("schema") != "rinkspace.first-shot-shared-row-timing.v1"
                    or timing.get("source_clock") != self.source_clock
                    or timing.get("source_video_sha256") != self.source["sha256"]
                    or timing.get("positive_is_top_to_bottom") is not True
                    or timing.get("readout_parameter_units") != "signed_milliseconds"
                    or timing.get("manufacturer_measured_readout") is not False):
                raise ValueError("Explicit source-conditioned shared native row timing required")
            initial_readout = float(timing["initial_readout_milliseconds"])
            if not np.isfinite(initial_readout) or not -20 <= initial_readout <= 20:
                raise ValueError("Native 50p readout initializer must be within one frame period")
        else:
            initial_readout = None
        self.row_readout_index = None if initial_readout is None else self.print_layout_stop
        self.count = self.print_layout_stop + int(initial_readout is not None)
        self.zoom_reference = binding.get("shared_zoom_camera")
        self.zoom_configuration = None
        self.zoom_parameter_indices = np.empty(0, dtype=int)
        if self.zoom_reference is not None:
            from .first_shot_zoom_camera import read_zoom_configuration, PARAMETERS
            self.zoom_configuration = read_zoom_configuration(self.zoom_reference, binding)
            self.zoom_parameter_indices = np.arange(self.count, self.count + len(PARAMETERS))
            self.count += len(PARAMETERS)
        self.print_layout_slice = slice(self.view_stop,self.print_layout_stop)
        self.print_layout_tail_slices = {n:slice(8*i,8*i+8) for i,n in enumerate(self.print_layout_advertisement_ids)}
        self.print_layout_world_slices = {n:slice(self.geometry_count+8*i,self.geometry_count+8*i+8)
            for i,n in enumerate(self.print_layout_advertisement_ids)}
        self.print_layout_global_indices_by_ad = {n:np.arange(self.view_stop+8*i,self.view_stop+8*i+8)
            for i,n in enumerate(self.print_layout_advertisement_ids)}
        self.world_parameter_indices = np.r_[np.arange(self.geometry_count),np.arange(self.view_stop,self.print_layout_stop)]
        self.world_parameter_count = len(self.world_parameter_indices)
        self.global_shared_indices = np.r_[np.arange(self.shared_count),np.arange(self.view_stop,self.count)]
        self.global_shared_count = len(self.global_shared_indices)
        state_path = _authenticate(binding["parent"]["complete_state"])
        identity = read_reference(binding["parent"]["state_identity"])
        input_manifest = read_reference(binding["parent"]["input_manifest"])
        keys = ("q", "aux__current_shared_world_geometry_parameter_indices",
                "aux__current_named66_parameter_names", "aux__shared_pattern_start",
                "aux__current_independent_pattern_placement_parameter_indices",
                "aux__installation_xyz_slots", "aux__current_shared_mount_parameter_indices",
                "aux__current_all_installation_optics_parameter_indices",
                "aux__all_installation_optics_installation_order",
                "aux__current_complete_parameter_layout_json",
                "aux__current_camera_pivot_exterior_constraints_json")
        with np.load(state_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in keys}
        self.parent_q = arrays["q"]
        digest = hashlib.sha256(self.parent_q.tobytes()).hexdigest()
        if (digest != binding["parent"]["q_sha256"] or digest != identity["q_sha256"]
                or self.parent_q.shape != (34340,) or self.parent_q.dtype != np.float64):
            raise ValueError("Exact complete 8a28 parent parameter state required")
        self.parent_q.flags.writeable = False
        self.layout_manifest = json.loads(str(arrays["aux__current_complete_parameter_layout_json"].item()))
        from calibration.solver_runtime import install
        install()
        from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout
        self.layout = CompleteProgrammeParameterLayout(
            self.layout_manifest["original_complete_parameter_count"],
            self.layout_manifest["additional_world_parameter_count"])
        if self.layout.manifest() != self.layout_manifest:
            raise ValueError("Saved complete canonical/evaluation permutation changed")
        world_slots = arrays["aux__current_shared_world_geometry_parameter_indices"].astype(int)
        if world_slots.shape != (100,):
            raise ValueError("Exact saved 100-coordinate shared world required")
        pattern_slots = np.arange(int(arrays["aux__shared_pattern_start"]), int(arrays["aux__shared_pattern_start"]) + 9)
        circle_slots = np.asarray(input_manifest["independent_circle_material_parameter_indices"], int)
        placement_slots = arrays["aux__current_independent_pattern_placement_parameter_indices"].astype(int)
        if circle_slots.shape != (6,) or placement_slots.shape != (4,):
            raise ValueError("Existing independent printed circles and placements required")
        self.inactive_circle_slot = int(circle_slots[5])
        order = list(map(str, arrays["aux__all_installation_optics_installation_order"]))
        ii = order.index(INSTALLATION)
        installation_slots = np.r_[arrays["aux__installation_xyz_slots"][ii],
            arrays["aux__current_shared_mount_parameter_indices"].reshape(len(order), 3)[ii],
            arrays["aux__current_all_installation_optics_parameter_indices"].reshape(len(order), 3)[ii]].astype(int)
        self.parent_shared_slots = np.r_[world_slots, pattern_slots, circle_slots[:5], placement_slots, installation_slots]
        if len(self.parent_shared_slots) != self.shared_count or len(np.unique(self.parent_shared_slots)) != self.shared_count:
            raise ValueError("Distinct complete parent material and installation slots required")
        # Explicit round-trip uses canonical slots; no evaluation suffix can
        # accidentally be interpreted as a canonical shared-world coordinate.
        self.evaluation_shared_slots = self.layout.canonical_to_evaluation[self.parent_shared_slots]
        self.geometry_parameter_names = tuple(map(str, arrays["aux__current_named66_parameter_names"])) + (
            "Ad13_station", "Ad13_height", "Ad13_radius", "Ad12_left_station", "Ad12_left_height",
            "Ad12_right_station", "Ad12_right_height", "Ad12_tip_station", "Ad12_tip_height",
            "Ad11_station", "Ad11_height", "Ad11_Ad24_radius", "Ad24_station", "Ad24_height",
            "Ad25_station_offset", "Ad25_height_offset", "Ad26_station_offset", "Ad26_height_offset")
        self.world_parameter_names = self.geometry_parameter_names + tuple(
            "Ad"+str(n)+"_print_layout_"+name for n in self.print_layout_advertisement_ids
            for name in ("a","b","c","d","tx","ty","p","q"))
        self.world_initializer_reference = reference(world_initializer)
        from rinkspace.vision_camera_initialization import current_complete_source_geometry
        self.provider, world, self.geometry_proof = current_complete_source_geometry(state_path, world_initializer)
        if self.provider.parameter_count != 100 or not np.array_equal(world, self.parent_q[world_slots]):
            raise ValueError("Public current complete geometry differs from exact parent world coordinates")
        self.scene_unit_working_m = float(self.provider.scene_unit_m)
        self.model = read_reference(binding["model"])
        self.approved_scene = read_reference(self.model["scene"])
        self.templates = read_reference(self.model["material_templates"])
        self.material_revision_proof = None
        if binding.get("material_template_revision") is not None:
            from .first_shot_revision import material_revision
            self.templates, self.material_revision_proof = material_revision(
                binding["material_template_revision"], binding, self.model["material_templates"])
        self.registry = read_reference(self.model["feature_registry"])
        self.pivot_contract = json.loads(str(arrays["aux__current_camera_pivot_exterior_constraints_json"].item()))
        if self.pivot_contract.get("by_installation", {}).get(INSTALLATION, {}).get("relation") != "pivot_outside_or_on_current_actual_board_course":
            raise ValueError("Saved explicit user outside-pivot fact required")
        _authenticate(self.pivot_contract["facts"])
        camera_reference = dict(path=identity["camera_path"], sha256=identity["camera_sha256"])
        camera_path = _authenticate(camera_reference)
        with np.load(camera_path, allow_pickle=False) as cameras:
            metadata = json.loads(str(cameras["metadata_json"].item()))
            archived_camera = cameras["camera25"].copy()
            archived_view = cameras["view_state4"].copy()
            archived_installation = cameras["installation_id"].copy()
        self.base_calibration = SharedCalibration.from_manifest(metadata["calibrations"][INSTALLATION])
        if self.base_calibration.allow_effective_pp_offset:
            raise ValueError("First-shot shared physical model cannot add free per-frame principal point offsets")
        if any(len(getattr(self.base_calibration, k)) != 1 for k in ("principal_x_fraction", "principal_y_fraction", "k1")):
            raise ValueError("Existing constant shared PPx/PPy/k1 installation chart required")
        views, seed_receipts = [], []
        seen_pts = []
        for seed_ref in binding["navigation_initializers"]:
            seed_path = _authenticate(seed_ref)
            with np.load(seed_path, allow_pickle=False) as seeds:
                seed_pts = seeds["official_pts"].copy() if "official_pts" in seeds else seeds["pts"].copy()
                seed_views = seeds["view_state4"].copy()
                installations = seeds["installation_id"].copy()
                cameras = seeds["camera25"].copy()
                archive_indices = (seeds["inherited_camera_archive_row"].copy()
                    if "inherited_camera_archive_row" in seeds else seeds["inherited_camera_archive_indices"].copy())
                confirmed = seeds["pixel_correspondence_confirmed"].copy() if "pixel_correspondence_confirmed" in seeds else None
            if (seed_views.shape != (len(seed_pts), 4) or not np.isfinite(seed_views).all()
                    or not np.all(installations == INSTALLATION)
                    or not np.array_equal(seed_views, archived_view[archive_indices])
                    or not np.array_equal(cameras, archived_camera[archive_indices])
                    or not np.all(archived_installation[archive_indices] == INSTALLATION)):
                raise ValueError("All native-50 RF4 seeds must retain exact 8a28 single-installation initialization")
            rebuilt = np.asarray([self.base_calibration.camera25(ViewState.from_state4(v)) for v in seed_views])
            if not np.array_equal(rebuilt, cameras):
                raise ValueError("Physical shared calibration does not exactly reproduce inherited RF4 camera seeds")
            views.append(seed_views); seen_pts.extend(map(int, seed_pts))
            seed_receipts.append(dict(reference=deepcopy(seed_ref), frames=len(seed_pts),
                first_native_pts=int(seed_pts[0]), last_native_pts=int(seed_pts[-1]),
                inherited_RF4_and_shared_camera_bytes_exact=True, initializer_only=True,
                original_pixel_match_confirmed_count=None if confirmed is None else int(confirmed.sum())))
        if not np.array_equal(seen_pts, self.pts):
            raise ValueError("Camera seed native clock does not equal the complete accepted first shot")
        self.initial_q = np.r_[self.parent_q[self.parent_shared_slots], np.concatenate(views).ravel(),np.zeros(layout_count),
            [] if initial_readout is None else [initial_readout], np.zeros(len(self.zoom_parameter_indices))]
        if self.initial_q.shape != (self.count,):
            raise ValueError("Complete shared plus native RF4 chart required")
        # The saved 8a28 run explicitly has no installed mount working range.
        # Expose the existing user v4 range without silently changing this
        # fact, clipping its out-of-range q, or inventing a mechanical prior.
        self.mount_facts_reference = reference(DEFAULT_MOUNT_FACTS)
        mount_facts = read_reference(self.mount_facts_reference)
        mount = mount_facts["rigid_mount_offset"]
        if (mount_facts.get("schema_version") != "rinkspace.user-camera-facts.v1"
                or mount.get("authority") != "user_adopted_photo_based_initialization_and_bounds_from_dot"
                or mount.get("bounds_are_user_working_search_ranges_not_measurements") is not True
                or mount.get("coordinate_frame") != "camera-local OpenCV x right, y down, z forward"
                or mount.get("shared_across_same_installation_exposures") is not True
                or mount.get("per_frame_free_optical_centre_enabled") is not False
                or abs(mount_facts["metres_per_original_scene_unit"] - self.scene_unit_working_m) > 1e-15):
            raise ValueError("Same declared scale and explicit shared rigid mount facts required")
        self.mount_absolute_lower = np.asarray(mount["lower_camera_xyz_m"], float)
        self.mount_absolute_upper = np.asarray(mount["upper_camera_xyz_m"], float)
        if (self.mount_absolute_lower.shape != (3,) or self.mount_absolute_upper.shape != (3,)
                or not np.isfinite(self.mount_absolute_lower).all() or not np.isfinite(self.mount_absolute_upper).all()
                or np.any(self.mount_absolute_lower >= self.mount_absolute_upper)):
            raise ValueError("Finite explicit ordered user rigid mount working ranges required")
        initial_lever = np.asarray(self.base_calibration.lever_xyz) * self.scene_unit_working_m
        low = self.initial_q[121:124] + self.mount_absolute_lower - initial_lever
        high = self.initial_q[121:124] + self.mount_absolute_upper - initial_lever
        self.historical_mount_parameter_bounds = {int(k): (float(a), float(b)) for k, a, b in zip(range(121, 124), low, high, strict=True)}
        self.saved_mount_bounds_active = input_manifest.get("camera_mount_working_constraints") is not None
        self.mount_revision_proof = None
        if binding.get("mount_working_range_revision") is not None:
            from .first_shot_mount_revision import read_mount_revision
            self.mount_revision_proof = read_mount_revision(binding["mount_working_range_revision"],
                self.mount_facts_reference, self.scene_unit_working_m)
        self.mount_bounds_active = self.saved_mount_bounds_active or self.mount_revision_proof is not None
        self.shared_parameter_bounds = deepcopy(self.historical_mount_parameter_bounds) if self.mount_bounds_active else {}
        if self.row_readout_index is not None:
            self.shared_parameter_bounds[self.row_readout_index] = (-20., 20.)
        self.active_shared_parameter_bounds = deepcopy(self.shared_parameter_bounds)
        self._scene_cache = OrderedDict()
        self._world_cache = OrderedDict()
        self._base_pattern_geometry = None
        self.proof = dict(schema="rinkspace.first-shot-native50-physical-state.v1",
            parent_q_sha256=digest, parent_complete_state=deepcopy(binding["parent"]["complete_state"]),
            complete_parent_parameter_count=len(self.parent_q), active_parameter_count=self.count,
            shared_geometry_parameter_count=self.world_parameter_count, installation_parameter_count=9,
            original_shared_geometry_parameter_count=self.geometry_count,
            world_parameter_indices=self.world_parameter_indices.tolist(),
            global_shared_parameter_indices=self.global_shared_indices.tolist(),
            view_stop=self.view_stop, shared_print_layout_reference=self.print_layout_reference,
            shared_print_layout_parameter_count=layout_count,
            shared_print_layout_advertisement_ids=list(self.print_layout_advertisement_ids),
            shared_print_layout_global_columns_by_ad={str(n):v.tolist() for n,v in self.print_layout_global_indices_by_ad.items()},
            print_layout_has_no_per_exposure_parameters=True,
            native_RF4_exposures=self.frame_count, source_clock=deepcopy(self.source_clock),
            source_video_sha256=self.source["sha256"], first_native_pts=int(self.pts[0]), last_native_pts=int(self.pts[-1]),
            geometry_canonical_parent_slots=self.parent_shared_slots[:118].tolist(),
            installation_canonical_parent_slots=self.parent_shared_slots[118:].tolist(),
            parent_shared_evaluation_slots=self.evaluation_shared_slots.tolist(),
            inactive_legacy_circle_radius_slot=self.inactive_circle_slot,
            native_view_state_has_only_R3_and_one_effective_focal=True,
            shared_calibration_reference=camera_reference, installation_id=INSTALLATION,
            navigation_initializers=seed_receipts, seed_mapping_is_observation_or_exposure_proof=False,
            historical_pixel_factors_connected_to_active_FIT=False,
            historical_full_programme_graph_loaded=False,
            historical_profiles_loaded=False,
            shared_geometry_constructor_legacy_initialization_helpers_reused=True,
            temporary_legacy_initialization_factor=dict(scalar_rows=4340,
                role="discarded_shared_geometry_constructor_initialization_only",
                included_in_current_FIT=False, native50_source_exposures_affected=0),
            parent_projection_leaves_legacy_view_clock_unchanged=True,
            names_and_working_scale_do_not_establish_verified_metres=True,
            existing_mount_range_reference=deepcopy(self.mount_facts_reference),
            saved_parent_mount_working_range_active=self.saved_mount_bounds_active,
            explicit_mount_working_range_revision=self.mount_revision_proof,
            active_mount_working_range=self.mount_bounds_active,
            historical_optional_mount_working_bounds=self.historical_mount_parameter_bounds,
            initial_existing_mount_range_status=self.mount_bounds_status(self.initial_q),
            existing_mount_range_silently_reenabled_or_q_clipped=False,
            physical_topology_corrections=deepcopy(self.approved_scene.get("physical_topology_corrections", [])))
        if self.row_readout_index is not None:
            self.proof.update(row_timing_reference=self.row_timing_reference,
                row_readout_parameter_index=self.row_readout_index,
                row_readout_parameter_units="signed_milliseconds", row_readout_shared_across_all_native_exposures=True,
                row_timing_is_source_conditioned_not_manufacturer_measured=True,
                row_trajectory_clock="exact native PTS; adjacent SO(3) short-arc and log-focal interpolation",
                camera25_is_mid_row_only=True)
        if self.zoom_reference is not None:
            from .first_shot_zoom_camera import PARAMETERS
            self.proof.update(shared_zoom_camera_reference=self.zoom_reference,
                shared_zoom_parameter_indices=self.zoom_parameter_indices.tolist(),
                shared_zoom_parameter_names=list(PARAMETERS),
                zoom_driver="final_effective_domain_log_focal_ratio",
                zoom_parameters_are_manufacturer_calibrated=False,
                zoom_has_no_per_exposure_extra_parameters=True)

    def _q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.count,) or not np.isfinite(q).all():
            raise ValueError("Finite complete first-shot shared/native RF4 state required")
        return q

    def _geometry(self, q):
        q = np.asarray(q, dtype=float)
        count = getattr(self,"world_parameter_count",self.geometry_count)
        if q.shape == (self.count,):
            q = q[getattr(self,"world_parameter_indices",np.arange(self.geometry_count))]
        if q.shape != (count,) or not np.isfinite(q).all():
            raise ValueError("Finite complete " + str(count) + "-coordinate shared material state required")
        return q

    def templates_at(self, shared_geometry):
        """Same-state runtime template; the immutable source template stays fixed."""
        q = self._geometry(shared_geometry)
        if getattr(self,"print_layout_reference",None) is None or not np.any(q[self.geometry_count:]):
            return self.templates
        from .first_shot_print_layout import apply_print_layout
        templates = self.templates
        for number in self.print_layout_advertisement_ids:
            values = q[self.print_layout_world_slices[number]]
            if np.any(values):templates = apply_print_layout(templates,number,values)
        return templates

    def review_scene(self, shared_geometry):
        """Render the solved print and its real outer border from the same q.

        This review copy leaves the identity scene dictionary intact. Ad5
        retains its named internal graphic; the other selected prints retain
        their existing outer canvases. No new bounding rectangle is created.
        """
        q = self._geometry(shared_geometry)
        scene = deepcopy(self.scene(q))
        if getattr(self,"print_layout_reference",None) is None:return scene
        templates = self.templates_at(q)
        from .framewise_model import feature_world
        from .first_shot_print_layout import print_layout_template
        descriptor = scene["source_conditioned_board_prints"]
        descriptor["retain_outer_canvas_border_advertisement_ids"] = list(self.print_layout_advertisement_ids)
        for number in self.print_layout_advertisement_ids:
            material = print_layout_template(templates,number)
            contours = []
            for key in material["contours"]:
                target = {"kind":"ad_print","advertisement_id":number,"contour_key":key}
                world = feature_world(scene,target,templates)
                contours.append(dict(id=key,role="shared_Ad"+str(number)+"_print_"+key,xyz_scene_units=world.tolist()))
            if any(int(p["advertisement_id"]) == number for p in descriptor["patterns"]):
                raise ValueError("A duplicate print renderer inventory would hide its material interface: Ad"+str(number))
            descriptor["patterns"].append(dict(advertisement_id=number,external_contours=contours,
                retain_outer_canvas_border=True,source_template_modified=False,
                material_frame=("named_internal_graphic_extended" if number == 5 else "existing_outer_canvas"),
                extent_scope=("continuous_board_s_h_relative_to_named_internal_graphic" if number == 5
                    else "ad_rectangle_u_left_to_right_v_top_to_bottom"),
                print_layout_configuration=self.print_layout_reference,
                actual_material_producer="FirstShotPhysicalState.world_paths:same_q_outer_canvas_layout",
                independent_print_world_measurements_added=False))
        return scene

    def parent_projection(self, q):
        q = self._q(q)
        result = self.parent_q.copy()
        result[self.parent_shared_slots] = q[:self.shared_count]
        return result

    def calibration(self, q):
        q = self._q(q)
        delta = q[self.installation_slice] - self.initial_q[self.installation_slice]
        base = self.base_calibration
        calibration = replace(base,
            pivot_xyz=tuple(np.asarray(base.pivot_xyz) + delta[:3] / self.scene_unit_working_m),
            lever_xyz=tuple(np.asarray(base.lever_xyz) + delta[3:6] / self.scene_unit_working_m),
            principal_x_fraction=(base.principal_x_fraction[0] + delta[6] / 1920.,),
            principal_y_fraction=(base.principal_y_fraction[0] + delta[7] / 1080.,),
            k1=(base.k1[0] + .001 * delta[8],))
        from .first_shot_zoom_camera import with_zoom_calibration
        return with_zoom_calibration(self, q, calibration)

    def view_columns(self, programme_index_1based):
        index = int(programme_index_1based)
        if index != programme_index_1based or not 1 <= index <= self.frame_count:
            raise ValueError("An exact first-shot programme frame index is required")
        return self.view_start + 4 * (index - 1) + np.arange(4)

    def camera25(self, q, programme_index_1based):
        q = self._q(q)
        view = ViewState.from_state4(q[self.view_columns(programme_index_1based)])
        return self.calibration(q).camera25(view)

    def camera15_mapping(self, q, programme_index_1based):
        """Same nominal camera and analytical conditioned 15 x 13 chain."""
        q = self._q(q); view_columns = self.view_columns(programme_index_1based)
        cal = self.calibration(q)
        camera, shared, view = cal.camera25_and_jacobians(ViewState.from_state4(q[view_columns]))
        names = cal.shared_state_names()
        shared_columns = [names.index("pivot_" + axis + "_scene_div100") for axis in "xyz"]
        shared_columns += [names.index("lever_" + axis + "_scene_div100") for axis in "xyz"]
        shared_columns += [names.index(name + "_degree_0") for name in ("principal_x_fraction", "principal_y_fraction", "k1")]
        scale = np.r_[np.full(6, 1. / (100. * self.scene_unit_working_m)), 1. / 1920., 1. / 1080., .001]
        chain25 = np.c_[shared[:, shared_columns] * scale, view]
        from .first_shot_zoom_camera import zoom_shared_mapping
        zoom_columns, zoom_selected, zoom_scale = zoom_shared_mapping(self, cal)
        if len(zoom_columns):
            chain25 = np.c_[chain25, shared[:, zoom_selected] * zoom_scale]
        if np.any(abs(chain25[15:]) > 1e-12):
            raise ValueError("Active physical camera coordinate lies outside the original 15-column residual chart")
        inverse_condition = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
        chain = chain25[:15] * inverse_condition[:, None]
        columns = np.r_[np.arange(118, 127), view_columns, zoom_columns]
        return camera, columns, chain

    def scene(self, shared_geometry):
        """Current scene with the approved topology and fixed print templates."""
        q = self._geometry(shared_geometry); token = q.tobytes()
        if token in self._scene_cache:
            return self._scene_cache[token]
        scene = self.provider.source_scene(q[:100])
        retired = self.approved_scene.get("superseded_ice_feature_aliases", {})
        scene["ice_features"] = [f for f in scene["ice_features"] if f["id"] not in retired]
        for key in ("superseded_duplicate_ice_features", "superseded_ice_feature_aliases", "physical_topology_corrections"):
            if key in self.approved_scene:
                scene[key] = deepcopy(self.approved_scene[key])
        # The complete public renderer uses this explicit contour inventory to
        # skip generic rectangles for Ad8/9/10 and their paired instances.
        # Re-evaluate the already frozen normalized materials on current world
        # coordinates; historical pixels and cameras are never consulted.
        print_descriptor = deepcopy(self.approved_scene["source_conditioned_board_prints"])
        for item in print_descriptor["patterns"]:
            number = int(item["advertisement_id"])
            for contour in item["external_contours"]:
                material = np.asarray(contour["frozen_normalized_material_coordinates"], float)
                xyz = self.provider.current.source_template_material(number, *material.T, q[:100])[0]
                contour["xyz_scene_units"] = xyz.tolist()
                contour["conditional_height_scene_units"] = [float(xyz[:, 2].min()), float(xyz[:, 2].max())]
        scene["source_conditioned_board_prints"] = print_descriptor
        from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry
        from structured_programme_source_candidate.course_patterns_independent_placements import IndependentlyPlacedCoursePatterns
        if self._base_pattern_geometry is None:
            self._base_pattern_geometry = SharedPerimeterBoardPatternGeometry(self.provider.current.base,
                self.scene_unit_working_m, facet_authority=self.approved_scene["shared_board_patterns"]["facet_authority"])
        course, cq, _ = self.provider.current.print_course_binding(q[:100])
        patterns = IndependentlyPlacedCoursePatterns(self._base_pattern_geometry, course, cq, q[self.placement_slice])
        scene["shared_board_patterns"] = patterns.descriptor(q[self.pattern_slice])
        from structured_programme_source_candidate.native_board_circle_source import straight_board_circle_world
        circles = deepcopy(self.approved_scene["additional_board_circle_patterns"])
        materials = np.r_[q[self.circle_slice], self.parent_q[self.inactive_circle_slot]]
        circles.update(current_circle6=materials.tolist(), current_course6_working_m=cq.tolist(), shared_radius_working_m=float(materials[2]))
        for number, offset in ((11, 0), (24, 3)):
            material = materials[offset:offset + 3].copy(); material[2] = materials[2]
            world = straight_board_circle_world(cq, material, self.scene_unit_working_m, number, course=course)
            circles["circles"][str(number)].update(circle_outer_stroke_xyz_scene_units=world.tolist(),
                material3_working_m=material.tolist(), radius_scene_units=float(material[2] / self.scene_unit_working_m))
        scene["additional_board_circle_patterns"] = circles
        scene["coordinate_system"] = deepcopy(self.approved_scene["coordinate_system"])
        if getattr(self,"print_layout_reference",None) is not None and np.any(q[self.geometry_count:]):
            scene["shared_print_layouts"] = dict(configuration=self.print_layout_reference,
                advertisement_ids=list(self.print_layout_advertisement_ids),
                shared_parameters=q[self.geometry_count:].tolist(),
                material_coordinates="each_named_material_frame_u_left_to_right_v_top_to_bottom",
                material_frame_by_ad={str(n):("named_internal_graphic_extended" if n == 5 else "existing_outer_canvas")
                    for n in self.print_layout_advertisement_ids},
                extent_scope_by_ad={str(n):("continuous_board_s_h_relative_to_named_internal_graphic" if n == 5
                    else "ad_rectangle_u_left_to_right_v_top_to_bottom") for n in self.print_layout_advertisement_ids},
                original_named_material_frames_modified=False,
                source_template_modified=False,per_exposure_parameters_added=0)
        self._scene_cache[token] = scene
        while len(self._scene_cache) > 2:
            self._scene_cache.popitem(last=False)
        return scene

    def world_paths(self, shared_geometry, descriptors):
        """Exact target paths once per shared state; original identities persist."""
        from .framewise_model import feature_world
        q = self._geometry(shared_geometry)
        key = q.tobytes(), tuple(json.dumps(d, sort_keys=True) for d in descriptors)
        if key not in self._world_cache:
            scene = self.scene(q)
            templates = self.templates_at(q)
            self._world_cache[key] = tuple(feature_world(scene, d, templates) for d in descriptors)
            while len(self._world_cache) > 2:
                self._world_cache.popitem(last=False)
        return self._world_cache[key]

    def print_layout_world_jacobians(self, shared_geometry, descriptors):
        """Eight columns per selected print; each original material map is reused."""
        q = self._geometry(shared_geometry)
        paths = self.world_paths(q,descriptors)
        count = len(q)-self.geometry_count
        result = [np.zeros(path.shape+(count,)) for path in paths]
        if getattr(self,"print_layout_reference",None) is None:return tuple(result)
        from .first_shot_print_layout import material_world_layout_jacobian, print_layout_template
        for number in self.print_layout_advertisement_ids:
            material = print_layout_template(self.templates,number)
            batches = []; locations = []; cursor = 0
            for i,d in enumerate(descriptors):
                if d.get("kind") != "ad_print" or int(d.get("advertisement_id",0)) != number:continue
                contour = material["contours"][d["contour_key"]]
                uv = np.asarray(contour["material_uv"],float)
                locations.append((i,cursor,cursor+len(uv),bool(contour["closed"])))
                cursor += len(uv);batches.append(uv)
            if not batches:continue
            # Four batches per print map, independent of its eight parameters.
            derivatives = material_world_layout_jacobian(self.provider.current,number,np.concatenate(batches),
                q[self.print_layout_world_slices[number]],q[:100],fd_step=float(self.provider.fd_step))
            for i,first,stop,closed in locations:
                value = derivatives[first:stop]
                if len(paths[i]) == len(value)+1 and closed:value = np.concatenate((value,value[:1]))
                if value.shape != paths[i].shape+(8,):
                    raise ValueError("Shared print layout changed the original contour vertex inventory")
                result[i][...,self.print_layout_tail_slices[number]] = value
        return tuple(result)

    def print_material_domain_status(self, shared_geometry, advertisement_id, material_uv):
        """Actual periodic board station and height domain for extended ink.

        Ad5's internal graphic is not an outer bounding rectangle. Its ink
        may have negative u or u>1 and may cross the chosen perimeter seam.
        One connected unwrapped support must fit within one whole perimeter;
        all ink heights must remain between actual ice and board top.
        """
        from .ad_landmarks import AdGeometry
        q = self._geometry(shared_geometry);scene = self.scene(q)
        uv = np.asarray(material_uv,float)
        if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
            raise ValueError("Finite actual printed material coordinates are required")
        geometry = AdGeometry(scene,advertisement_id)
        if geometry._nominal_width is None or geometry._width_sign is None:
            raise ValueError("A named current material frame with both stations is required")
        bottom = np.asarray(scene["board_surface"]["bottom_edge_xyz"],float)
        top = np.asarray(scene["board_surface"]["top_edge_xyz"],float)
        for label,path in (("ice contact",bottom),("board top",top)):
            if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all():
                raise ValueError("Current actual finite " + label + " is required")
            # The existing complete provider has one common contact/top Z.
            # A future varying-height board needs its actual local lifting
            # interface, never a permissive global min/max substitution.
            if np.ptp(path[:,2]) > 64*np.finfo(float).eps*np.maximum(1.,abs(path[:,2])).max():
                raise ValueError("Current printed material requires the actual varying-height board interface: " + label)
        station = geometry._base_s["left"]+geometry._width_sign*geometry._nominal_width*uv[:,0]
        upper = geometry.corner(geometry.initial,"left_top")[2]
        lower = geometry.corner(geometry.initial,"left_bottom")[2]
        z = upper+uv[:,1]*(lower-upper)
        ice_z = float(bottom[0,2]);board_top = float(top[0,2])
        span = float(np.ptp(station));height_tolerance = 1e-8/self.scene_unit_working_m
        station_valid = span < geometry._perimeter
        height_valid = bool((z >= ice_z-height_tolerance).all() and (z <= board_top+height_tolerance).all())
        return dict(valid=bool(station_valid and height_valid),advertisement_id=int(advertisement_id),
            material_frame="named_internal_graphic_extended",actual_material_vertices=len(uv),
            material_u_range=[float(uv[:,0].min()),float(uv[:,0].max())],
            material_v_range=[float(uv[:,1].min()),float(uv[:,1].max())],
            unwrapped_station_range_scene_units=[float(station.min()),float(station.max())],
            unwrapped_station_span_scene_units=span,current_actual_perimeter_scene_units=float(geometry._perimeter),
            station_span_strictly_below_one_actual_perimeter=bool(station_valid),
            station_is_periodic=True,seam_crossing_or_whole_turn_gauge_not_clamped=True,
            absolute_height_range_scene_units=[float(z.min()),float(z.max())],
            actual_ice_z_scene_units=ice_z,actual_board_top_scene_units=board_top,
            actual_height_domain_valid=height_valid,
            height_roundoff_tolerance_working_m=1e-8,
            height_tolerance_authority="existing_public_current_qualified_print_board_Z_domain",
            scene_unit_working_m=self.scene_unit_working_m,verified_metre_scale=False,
            unit_graphic_frame_not_used_as_outer_ink_bounds=True,parameter_cap_or_clamp_added=False)

    def validate_initial_targets(self, descriptors):
        from .framewise_model import feature_world
        actual = self.world_paths(self.initial_q, descriptors)
        records = []
        for descriptor, produced in zip(descriptors, actual, strict=True):
            approved = feature_world(self.approved_scene, descriptor, self.templates)
            if produced.shape != approved.shape:
                raise ValueError("Current dynamic shared geometry changes approved target knots: " + json.dumps(descriptor))
            error = float(np.max(abs(produced - approved)))
            if error > 1e-10:
                raise ValueError("Current dynamic shared geometry changes approved target initialization: " + json.dumps(descriptor) + "; " + str(error))
            records.append(dict(target=deepcopy(descriptor), vertices=len(produced), bytes_identical=np.array_equal(produced, approved), maximum_absolute_difference=error))
        return dict(passed=True, target_count=len(records), targets=records,
            approved_topology_and_templates_preserved=self.material_revision_proof is None,
            material_template_revision=self.material_revision_proof,
            shared_parameters_initialized_from_exact_complete_parent=True,
            maximum_absolute_difference=max((r["maximum_absolute_difference"] for r in records), default=0.))

    def mount_bounds_status(self, q):
        """Existing v4 range status; its working metres are not surveyed units."""
        calibration = self.calibration(q)
        lever = np.asarray(calibration.lever_xyz) * self.scene_unit_working_m
        low_error = np.maximum(self.mount_absolute_lower - lever, 0.)
        high_error = np.maximum(lever - self.mount_absolute_upper, 0.)
        result = dict(valid=bool(np.all(low_error <= 1e-12) and np.all(high_error <= 1e-12)),
            actual_lever_working_m=lever.tolist(), lower_working_m=self.mount_absolute_lower.tolist(),
            upper_working_m=self.mount_absolute_upper.tolist(), lower_violation_working_m=low_error.tolist(),
            upper_violation_working_m=high_error.tolist(), verified_metre_scale=False)
        if len(getattr(self, "zoom_parameter_indices", ())):
            from .first_shot_zoom_camera import zoom_mount_envelope
            result["zoom_optical_offset_envelope"] = zoom_mount_envelope(self, q, calibration)
            result["valid"] = result["valid"] and result["zoom_optical_offset_envelope"]["valid"]
        return result

    def ice_material_containment_status(self, shared_geometry):
        """Every finite ice material path belongs to the same current ice.

        These world vertices are model support, never new source observations.
        The tolerance is the existing float32 footprint roundoff used by the
        parent geometry's paint/goal-crease checks, in original scene units.
        No maximum extension parameter or historical working bound is added.
        """
        import cv2
        q = self._geometry(shared_geometry)
        scene = self.scene(q)
        boundary = np.asarray(scene["ice_boundary"]["points_xyz"], float)
        if (boundary.ndim != 2 or boundary.shape[1] != 3 or len(boundary) < 3
                or not np.isfinite(boundary).all()):
            raise ValueError("Current actual finite ice footprint is required")
        footprint = np.ascontiguousarray(boundary[:, :2], np.float32)
        targets = [(identity, target) for identity, target in self.registry["targets"].items()
                   if target.get("kind") == "ice_curve"]
        paths = self.world_paths(q, [target for _, target in targets])
        records = []; violations = []; total = 0
        tolerance = 2e-5
        for (identity, target), path in zip(targets, paths, strict=True):
            path = np.asarray(path, float)
            if (path.ndim != 2 or path.shape[1] != 3 or not len(path)
                    or not np.isfinite(path).all()):
                raise ValueError("Finite current ice material path required: " + identity)
            distances = np.array([cv2.pointPolygonTest(footprint, tuple(map(float, p[:2])), True)
                                  for p in path])
            index = int(np.argmin(distances)); total += len(path)
            record = dict(target_id=identity, curve_id=target["curve_id"], vertices=len(path),
                minimum_signed_clearance_scene_units=float(distances[index]),
                minimum_signed_clearance_working_m=float(distances[index]*self.scene_unit_working_m),
                worst_vertex_index=index, worst_vertex_xyz_scene_units=path[index].tolist(),
                worst_vertex_xyz_working_m=(path[index]*self.scene_unit_working_m).tolist(),
                worst_vertex_is_endpoint=index in (0, len(path)-1),
                outside_vertices=int(np.count_nonzero(distances < -tolerance)))
            records.append(record)
            if record["outside_vertices"]:
                violations.append(record)
        return dict(valid=not violations, same_current_geometry_for_paint_and_ice=True,
            target_count=len(targets), checked_model_vertices=total,
            footprint_roundoff_tolerance_scene_units=tolerance,
            tolerance_authority="existing_parent_actual_paint_float32_footprint_roundoff",
            scene_unit_working_m=self.scene_unit_working_m, verified_metre_scale=False,
            arbitrary_parameter_cap_or_new_measurement_added=False,
            targets=records, violations=violations)

    def guard_geometry(self, shared_geometry):
        """Check the current material domain without checking P, lever, or RF4.

        Only public geometry-domain failures and the explicitly known legacy
        paint-containment failures become GeometryDomainError. Errors from
        target lookup, world-path shape, or other interfaces propagate intact.
        """
        q = self._geometry(shared_geometry)
        try:
            self.provider.validate_parameters(q[:100])
        except ValueError as error:
            raise GeometryDomainError("Current shared geometry leaves public physical domain: " + str(error)) from error
        layout_status = None
        if getattr(self,"print_layout_reference",None) is not None:
            from .first_shot_print_layout import validate_layout, warp_normalized_material_uv, PrintLayoutDomainError, print_layout_template
            layout_status = {}
            try:
                for number in self.print_layout_advertisement_ids:
                    material = print_layout_template(self.templates,number)
                    values = q[self.print_layout_world_slices[number]]
                    prototype = np.concatenate([np.asarray(c["material_uv"],float) for c in material["contours"].values()])
                    matrix = validate_layout(values,prototype)
                    layout_status[str(number)] = dict(homography_matrix=np.asarray(matrix,float).tolist(),
                        positive_determinant_and_denominator_checked=True,
                        original_outer_canvas_material_domain_checked=True,
                        arbitrary_offset_cap_or_source_prior_added=False)
                    extended = []
                    for key,c in material["contours"].items():
                        warped,_ = warp_normalized_material_uv(np.asarray(c["material_uv"],float),values)
                        if c.get("material_coordinate_domain") == "continuous_board_s_h_relative_to_named_internal_graphic":
                            extended.append(warped);continue
                        if ((warped < 0.) | (warped > 1.)).any():
                            raise GeometryDomainError("Ad"+str(number)+" print leaves its existing normalized outer-canvas material domain: " + key)
                    if extended:
                        domain = self.print_material_domain_status(q,number,np.concatenate(extended))
                        layout_status[str(number)].update(original_outer_canvas_material_domain_checked=False,
                            material_frame="named_internal_graphic_extended",actual_continuous_board_material_domain=domain)
                        if not domain["valid"]:
                            raise GeometryDomainError("Ad"+str(number)+" print leaves current actual continuous board material domain: "
                                + ("unwrapped support spans an entire perimeter" if not domain["station_span_strictly_below_one_actual_perimeter"]
                                   else "ink height outside actual ice/top"))
            except PrintLayoutDomainError as error:
                raise GeometryDomainError("Shared print layout leaves physical homography domain: " + str(error)) from error
        try:
            containment = self.ice_material_containment_status(q)
        except ValueError as error:
            if str(error).startswith(("Qualified Lane painted end leaves actual ice footprint:",
                                      "Complete current structured circle leaves actual ice footprint:",
                                      "Goal-crease paint leaves actual ice footprint:")):
                raise GeometryDomainError(str(error)) from error
            raise
        if not containment["valid"]:
            failed = containment["violations"][0]
            raise GeometryDomainError("Current ice paint leaves actual ice footprint: " + failed["target_id"]
                + "; vertex " + str(failed["worst_vertex_index"])
                + "; outside by " + str(-failed["minimum_signed_clearance_working_m"])
                + " working_m (metric scale unverified)")
        return dict(passed=True, public_shared_geometry_domain_checked=True,
            installation_parameters_checked=False, ice_material_containment=containment,
            shared_print_layout_domain=layout_status)

    def guard(self, q, *, check_mount_bounds=None):
        """Existing finite physical domains and the saved user outside relation."""
        q = self._q(q)
        readout_index = getattr(self, "row_readout_index", None)
        if readout_index is not None and abs(q[readout_index]) > 20.:
            raise ValueError("Shared row readout exceeds the native 50p frame period")
        geometry_guard = self.guard_geometry(q)
        cal = self.calibration(q)
        if cal.pivot_xyz[2] <= 0 or np.linalg.norm(cal.lever_xyz) <= 1e-8:
            raise ValueError("Positive tripod pivot and shared nonzero rigid lever required")
        if check_mount_bounds is None:
            check_mount_bounds = self.mount_bounds_active
        mount_status = self.mount_bounds_status(q)
        if check_mount_bounds and not mount_status["valid"]:
            if not mount_status.get("zoom_optical_offset_envelope", {}).get("valid", True):
                raise ValueError("Zoom-dependent total optical offset exceeds the existing user v4 working range")
            raise ValueError("Shared rigid lever exceeds the existing user v4 working range")
        view_logf = q[self.view_start + 3:getattr(self,"view_stop",self.count):4] + np.log(cal.output_mode.domain_wh[0])
        if np.any((view_logf < np.log(100.)) | (view_logf > np.log(50000.))):
            raise ValueError("Native physical focal domain")
        course, cq, _ = self.provider.current.print_course_binding(q[:100])
        polygon = course.sample_perimeter(cq, with_jacobian=False)[0][:, :2] * self.scene_unit_working_m
        if np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        area = .5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
        from installation_mount_candidate.mount_graph import _pivot_outward_margin
        outside = _pivot_outward_margin(polygon, area, np.asarray(cal.pivot_xyz[:2]) * self.scene_unit_working_m)
        if not outside["valid"]:
            raise ValueError("Fixed physical tripod pivot is inside the current actual board course")
        self.scene(q)
        containment = geometry_guard["ice_material_containment"]
        return dict(passed=True, outside_pivot=outside, nonzero_shared_rigid_lever=True,
            per_frame_only_R3_and_zoom1=True, mount_working_range_active=bool(check_mount_bounds),
            existing_mount_range_status=mount_status, ice_material_containment=containment,
            shared_print_layout_domain=geometry_guard["shared_print_layout_domain"],
            verified_metre_scale=False)
