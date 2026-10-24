"""Typed runtime ownership and scene geometry for RinkSolve 2.

This module has no dependency on the previous solver. d78_import is the *new*
independent data converter; its continuous-board mathematics is reused here.
Initial coordinates are parameters, never evidence or fixed physical facts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import json
import hashlib
import re
import numpy as np
from scipy.spatial.transform import Rotation

from . import schema as S
from .d78_import import ContinuousBoard, load_initialization

VERSION = "rinksolve.runtime_model.v1"
TARGET_MOUNT = "fixed_pivot_constant_camera_local_lever.v1"
REPLAY_MOUNT = "historical_extended_fixed_head_replay.v1"


def readonly(value, *, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    if dtype == np.float64 and not np.isfinite(result).all():
        raise ValueError("nonfinite model parameter")
    result.setflags(write=False)
    return result


def frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(v) for v in value)
    return value


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


def block(identifier, owner_kind, owner_id, component_names, units, coordinate_id):
    if isinstance(units, S.Unit):
        units = [units] * len(component_names)
    return S.VariableBlock(identifier, S.OwnerRef(owner_kind, owner_id),
        tuple(S.ParameterComponent(name, S.UnitRef(unit, coordinate_id if unit == S.Unit.SCENE_UNIT else None))
              for name, unit in zip(component_names, units)), S.VariableMode.FREE)


@dataclass(frozen=True)
class ParameterState:
    """One immutable state with schema-owned blocks, separate from all evidence."""
    id: str
    revision: int
    coordinate_system_id: str
    blocks: tuple[S.VariableBlock, ...]
    values: tuple[S.ParameterValue, ...]

    def __post_init__(self):
        if self.revision < 0:
            raise ValueError("negative parameter revision")
        specs = {b.id: b for b in self.blocks}
        vals = {v.variable_block_id: v for v in self.values}
        if len(specs) != len(self.blocks) or len(vals) != len(self.values) or specs.keys() != vals.keys():
            raise ValueError("parameter block identity mismatch or duplicate")
        for k, v in vals.items():
            if len(v.values) != len(specs[k].components) or not np.isfinite(v.values).all():
                raise ValueError(f"parameter dimension/nonfinite: {k}")
        object.__setattr__(self, "_specs", MappingProxyType(specs))
        object.__setattr__(self, "_arrays", MappingProxyType({k: readonly(v.values) for k, v in vals.items()}))

    def get(self, identifier):
        return self._arrays[identifier]

    def updated(self, changes: Mapping[str, object], *, expected_revision=None, new_id=None):
        if expected_revision is not None and expected_revision != self.revision:
            raise ValueError("parameter revision conflict")
        for k in changes:
            if k not in self._specs:
                raise KeyError(f"unknown parameter block: {k}")
            if self._specs[k].mode != S.VariableMode.FREE:
                raise ValueError(f"cannot update nonfree block: {k}")
        values = tuple(S.ParameterValue(k, tuple(float(x) for x in np.asarray(changes.get(k, self.get(k))).reshape(-1)))
                       for k in self._specs)
        return ParameterState(new_id or f"{self.id.split('@')[0]}@{self.revision+1}", self.revision+1,
                              self.coordinate_system_id, self.blocks, values)

    def digest(self):
        h = hashlib.sha256()
        for spec in self.blocks:
            h.update(spec.id.encode("utf-8"))
            h.update(self.get(spec.id).astype("<f8", copy=False).tobytes())
        return h.hexdigest()

    def inventory(self):
        return {"id": self.id, "revision": self.revision, "coordinate_system_id": self.coordinate_system_id,
                "sha256": self.digest(), "blocks": [S.to_json_value(b) for b in self.blocks],
                "values": [S.to_json_value(v) for v in self.values]}


@dataclass(frozen=True)
class RigModel:
    id: str
    label: str
    coordinate_system_id: str
    mount_family: str = TARGET_MOUNT
    available: bool = True
    source_status: str = "candidate_initialization"
    governing_constraint_ids: tuple[str, ...] = ("rig.installations", "rig.nonzero_offset", "rig.independent_heights", "rig.zoom_offset")
    notes: str = ""
    zoom_reference_focal_px: float = 1.0
    normalized_radius_domain: float = .4
    image_width: int = 1920
    image_height: int = 1080
    reference_rotation_seed: tuple[float,...] = (1.,0.,0.,0.,1.,0.,0.,0.,1.)
    optical_center_reference_block_id: str | None = None
    direction_reference: tuple[float,...] | None = None
    reference_source_pts: int | None = None
    axis_parameterization: str = "raw_source_replay.v1"
    pan_axis_seed: tuple[float,float,float] | None = None
    tilt_axis_world_seed: tuple[float,float,float] | None = None
    pan_tangent_basis: tuple[float,...] | None = None

    def key(self, suffix):
        return f"rig/{self.id}/{suffix}"

    def definition(self, state):
        if not self.available:
            raise ValueError(f"{self.id}: no gauge-compatible camera initialization")
        head_rotation = Rotation.from_rotvec(state.get(self.key("reference_rotation_vector")).copy()).as_matrix()
        r = head_rotation @ np.asarray(self.reference_rotation_seed).reshape(3,3)
        if self.axis_parameterization=="coupled_s2_perpendicular.v1":
            n0=np.asarray(self.pan_axis_seed);w0=np.asarray(self.tilt_axis_world_seed)
            basis=np.asarray(self.pan_tangent_basis).reshape(2,3)
            tangent=state.get(self.key("pan_axis_chart"))@basis
            transport=Rotation.from_rotvec(np.cross(n0,tangent)).as_matrix()
            n=transport@n0;w=transport@w0
            angle=float(state.get(self.key("tilt_axis_chart"))[0])
            world_tilt=np.cos(angle)*w+np.sin(angle)*np.cross(n,w)
            b=r@world_tilt
        elif self.axis_parameterization=="raw_source_replay.v1":
            n, b = state.get(self.key("pan_axis_world")), state.get(self.key("tilt_axis_camera"))
            b = head_rotation @ b
        else:
            raise ValueError("unregistered head-axis parameterization")
        if abs(float(n@n)-1)>1e-10 or abs(float(b@b)-1)>1e-10:
            raise ValueError("head axes require explicit unit-vector equalities; no hidden renormalization")
        if abs(float(b @ r @ n)) > 1e-8:
            raise ValueError("fixed-head axes inconsistent; satisfy explicit head-frame equality")
        o = state.get(self.key("optics"))
        extended = state.get(self.key("extended_mount")) if self.mount_family == REPLAY_MOUNT else np.zeros(3)
        if self.mount_family not in (TARGET_MOUNT, REPLAY_MOUNT):
            raise ValueError("unregistered mount family")
        if np.linalg.norm(state.get(self.key("lever"))) == 0:
            raise ValueError("nonzero shared lever required; do not substitute free centers")
        if state.get(self.key("pivot"))[2] <= 0:
            raise ValueError("pivot_above_ice violated")
        return {"R0": r, "pan_axis_world": n, "tilt_axis_camera": b,
            "pivot_world": state.get(self.key("pivot")), "lever_camera": state.get(self.key("lever")),
            "axis_spacing": float(extended[0]), "pupil_polynomial": extended[1:],
            "f_ref": self.zoom_reference_focal_px, "radius_max": self.normalized_radius_domain,
            "eta": float(o[0]), "principal_point": o[1:3], "principal_zoom_slope": o[3:5],
            "k1_coefficients": o[5:7], "k2_coefficients": o[7:9], "tangential": o[9:11],
            "signed_readout_seconds": float(o[11]), "mount_family": self.mount_family,
            "historical_extensions_are_user_facts": False}

    def inspection(self, state):
        result = {"id": self.id, "label": self.label, "available": self.available,
            "coordinate_system_id": self.coordinate_system_id, "mount_family": self.mount_family,
            "source_status": self.source_status, "notes": self.notes, "pivot_world": None, "lever_camera": None,
            "position_kind": "shared_fixed_pivot_and_rotation_derived_optical_center", "unit": "scene_unit"}
        if self.available:
            result.update(pivot_world=state.get(self.key("pivot")).tolist(), lever_camera=state.get(self.key("lever")).tolist())
        if self.optical_center_reference_block_id is not None:
            result.update(optical_center_reference_world=state.get(self.optical_center_reference_block_id).tolist(),
                reference_position_kind="historical_demo_optical_center_initial_reference_not_pivot",
                reference_only=True,reference_source_pts=self.reference_source_pts,
                reference_direction_world=list(self.direction_reference) if self.direction_reference else None)
        return result


@dataclass(frozen=True)
class ShotMotion:
    id: str
    rig_id: str
    clock_id: str
    first_native_pts: int
    last_native_pts: int
    step_native_pts: int
    time_base_num: int
    time_base_den: int
    first_control_time_seconds: float
    control_dt_seconds: float
    control_count: int
    control_block_id: str
    model_family: str = "cardinal_cubic_two_axis_angles_log_f.v1"
    family_authority: str = "engineering_representation_not_hardware_measurement"

    @property
    def native_pts(self):
        return np.arange(self.first_native_pts, self.last_native_pts+1, self.step_native_pts, dtype=np.int64)

    def frame_time(self, pts):
        data = np.asarray(pts)
        if data.dtype.kind not in "iu" or np.any(data < self.first_native_pts) or np.any(data > self.last_native_pts) or np.any((data-self.first_native_pts)%self.step_native_pts):
            raise ValueError("exact in-scope integer native PTS required")
        return (data-self.first_native_pts) * self.time_base_num / self.time_base_den


@dataclass(frozen=True)
class CurveModel:
    id: str
    label: str
    entity_id: str
    kind: str
    parameter_block_id: str | None
    definition: Mapping
    initial_geometry_only: bool = True
    vertices_are_landmarks: bool = False
    semantic_status: str = "initializer_not_observation_authority"

    def __post_init__(self):
        object.__setattr__(self, "definition", frozen(self.definition))


@dataclass(frozen=True)
class ArtworkGroup:
    """A semantic logo/text group, not independently deformable letters."""
    id: str
    shape_ids: tuple[str, ...]
    layout_block_id: str | None
    transform_family: str = "shared_static_similarity_2d.v1"
    source_template_id: str | None = None
    source_feature_ids: tuple[str,...] = ()
    source_variant_status: str = "unconfirmed"
    # Optional layout values are [tx,ty,log_uniform_scale,angle]; owner=ARTWORK.


@dataclass(frozen=True)
class ArtworkModel:
    id: str
    template: S.ArtworkTemplate | None
    groups: tuple[ArtworkGroup, ...]
    source_status: str
    notes: str
    source_templates: tuple[S.ArtworkTemplate,...] = ()
    layout: S.ArtworkLayout | None = None

    def transform_group(self, group_id, xy, state):
        group = next(g for g in self.groups if g.id == group_id)
        xy = np.asarray(xy, dtype=np.float64)
        if group.layout_block_id is None:
            return xy.copy()
        if group.transform_family != "shared_static_similarity_2d.v1":
            raise ValueError("unregistered artwork-group transform")
        spec = state._specs[group.layout_block_id]
        if spec.owner.kind != S.OwnerKind.ARTWORK or spec.owner.id != group.id:
            raise ValueError("artwork layout must be static artwork-group-owned, never per frame")
        tx, ty, log_scale, angle = state.get(group.layout_block_id)
        c, s = np.cos(angle), np.sin(angle)
        return np.exp(log_scale) * xy @ np.array([[c,s],[-s,c]]) + [tx,ty]


@dataclass(frozen=True)
class InstallationModel:
    """Static rectangle in continuous board station/height, allowed across arcs."""
    id: str
    board_surface_id: str
    template_id: str | None
    placement_block_id: str | None
    size_block_id: str | None
    confirmed_same_size_group_id: str | None = None
    template_match_status: str = "unconfirmed"
    source_status: str = "candidate_initialization"
    mapping_family: str = "unwrapped_rectangle_rotation.v1"
    extent_role: str = "external_material_border"
    shared_layout_block_id: str | None = None
    shared_content_anchor_unwrapped: tuple[float,float] = (0.,0.)
    # placement=[station_at_template_origin,height_at_template_origin,rotation]
    # size=[width,height]; template origin is [0,0], opposite corner [1,1].

    def unwrapped(self, uv, state):
        if self.placement_block_id is None or self.size_block_id is None:
            raise ValueError("installation extent/placement is not recovered; no fake rectangle")
        q = np.asarray(uv, dtype=np.float64)
        if q.shape[-1] != 2:
            raise ValueError("artwork coordinates require last dimension 2")
        s0, h0, angle = state.get(self.placement_block_id)
        width, height = state.get(self.size_block_id)
        if width <= 0 or height <= 0:
            raise ValueError("nonpositive advertisement rectangle size")
        if self.mapping_family != "unwrapped_rectangle_rotation.v1":
            raise ValueError("unregistered installation mapping")
        c, s = np.cos(angle), np.sin(angle)
        result=q * [width,height] @ np.array([[c,s],[-s,c]]) + [s0,h0]
        if self.shared_layout_block_id is not None:
            result=apply_shared_similarity(result,state,self.shared_layout_block_id,self.shared_content_anchor_unwrapped)
        return result

    def boundary_world(self,scene,state,*,samples_per_edge=32):
        if samples_per_edge<2:
            raise ValueError("at least two samples per unwrapped rectangle edge")
        corners=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
        u=np.linspace(0,1,samples_per_edge,endpoint=False)
        points=np.concatenate([corners[i]*(1-u[:,None])+corners[i+1]*u[:,None] for i in range(4)]+[corners[:1]])
        return self.world(points,scene,state)

    def world(self, uv, scene, state):
        q = self.unwrapped(uv, state)
        return scene.board(state).xyz(q[...,0],q[...,1])


@dataclass(frozen=True)
class StructuredIceFamily:
    """Exact confirmed relations; every numeric size remains estimable.

    Geometry generation introduces no evidence. Finite left drawing bounds are
    coverage limits, not physical endpoints; a Lane has edges 0,1,2 only.
    Hockey and curling frames are separate unless a recorded equality ties them.
    """
    id: str
    curling_block_id: str
    lane_block_id: str
    hockey_block_id: str
    ratios_count: int
    hockey_extra_block_id: str | None = None
    hockey_midline_block_id: str | None = None
    constraint_ids: tuple[str,...] = ("curling.centres", "curling.radii", "lane.width", "hockey.centre_blue")
    authority: str = "user_confirmed_relations_with_engineering_parameterization"
    # v1 is retained only to read saved states without silently changing them.
    # New candidates eliminate the independent Lane centres and C0 translation.
    layout_version: int = 1
    board_block_id: str | None = None
    hockey_end_block_id: str | None = None

    def curves(self, state):
        # curling=[xF,xR,yFirst,pitch,rOuter, ratios...]
        p = state.get(self.curling_block_id)
        xF,xR,y0,pitch,radius = p[:5]
        ratios = p[5:]
        if len(ratios) != self.ratios_count or not (xF < xR and pitch > 0 and radius > 0):
            raise ValueError("invalid shared F/R geometry")
        if np.any(ratios <= 0) or np.any(ratios >= 1) or np.any(np.diff(ratios) <= 0):
            raise ValueError("nested shared house radii invalid")
        lane = state.get(self.lane_block_id)
        right, width = lane[:2]
        if self.layout_version == 2:
            if len(lane) != 10:
                raise ValueError("v2 Lane has no independent row-centre parameters")
            centers = y0 + np.arange(4)*pitch
            lefts, carrier_ends = lane[2:6], lane[6:10]
        elif self.layout_version == 1:
            if len(lane) != 14:
                raise ValueError("historical v1 Lane layout mismatch")
            centers, lefts, carrier_ends = lane[2:6], lane[6:10], lane[10:14]
        else:
            raise ValueError("unknown shared-ice layout version")
        if width <= 0 or np.any(lefts >= right) or np.any(carrier_ends <= xR):
            raise ValueError("invalid Lane width or finite coverage")
        result = {}
        layers = [f"ratio{i}" for i in range(len(ratios))] + ["outer"]
        if len(ratios) == 2:
            layers = ["inner", "middle", "outer"]
        elif len(ratios) == 3:
            layers = ["center", "inner", "middle", "outer"]
        for row in range(4):
            y = y0 + row*pitch
            for prefix,x in (("F",xF),("R",xR)):
                for layer, r in zip(layers,np.r_[ratios,1.]*radius):
                    result[f"ice.{prefix}{row+1}:{layer}"]={"kind":"ice_circle","center_xy":[x,y],"radius":float(r),"angle_interval":[0.,2*np.pi]}
            result[f"ice.house_connection_F{row+1}_R{row+1}:segment0"]={"kind":"ice_segment","endpoints_xy":[xF,y,carrier_ends[row],y]}
            low, high = centers[row]-width/2, centers[row]+width/2
            for edge, ends in ((0,[lefts[row],high,right,high]),(1,[right,high,right,low]),(2,[right,low,lefts[row],low])):
                result[f"ice.lane{row+1}:edge{edge}"]={"kind":"ice_segment","endpoints_xy":ends,
                    "left_values_are_drawing_coverage_not_physical_endpoints":True}
        hockey=state.get(self.hockey_block_id)
        if self.layout_version == 2:
            if len(hockey) != 5:
                raise ValueError("v2 hockey has no independent origin parameters")
            hx=hy=0.
            angle,blue,r0,low,high=hockey
        else:
            hx,hy,angle,blue,r0,low,high=hockey
        if blue <= 0 or r0 <= 0 or low >= high:
            raise ValueError("invalid shared hockey geometry")
        ca,sa=np.cos(angle),np.sin(angle)
        transform=np.array([[ca,sa],[-sa,ca]])
        for label,x in (("C",0.),("B1",-blue),("B2",blue)):
            ends=np.array([[x,high],[x,low]])@transform+[hx,hy]
            result[f"ice.{label}:segment0"]={"kind":"ice_segment","endpoints_xy":ends.reshape(-1).tolist()}
        result["ice.C0:outer"]={"kind":"ice_circle","center_xy":[hx,hy],"radius":float(r0),"angle_interval":[0.,2*np.pi]}
        if self.hockey_midline_block_id is not None:
            radius_mid=float(state.get(self.hockey_midline_block_id)[0])
            if radius_mid<=0:
                raise ValueError("nonpositive named C0 material-width midline radius")
            result["ice.C0:middle"]={"kind":"ice_circle","center_xy":[hx,hy],"radius":radius_mid,"angle_interval":[0.,2*np.pi],
                "measurement":"arithmetic_width_midpoint_of_two_actual_material_edges","same_radius_as_A_outer":False}
        if self.hockey_extra_block_id is not None:
            gx_left,gx_right,inward_left,inward_right,yhalf_left,yhalf_right=state.get(self.hockey_extra_block_id)
            if not gx_left<0<gx_right or min(inward_left,inward_right,yhalf_left,yhalf_right)<=0:
                raise ValueError("invalid shared faceoff/goal initial geometry")
            for label,x in (("goal_line_left",gx_left),("goal_line_right",gx_right)):
                ends=np.array([[x,high],[x,low]])@transform+[hx,hy]
                result[f"ice.{label}:segment0"]={"kind":"ice_segment","endpoints_xy":ends.reshape(-1).tolist()}
            for label,x,y in (("A1",gx_right-inward_right,-yhalf_right),("A2",gx_right-inward_right,yhalf_right),
                              ("A3",gx_left+inward_left,yhalf_left),("A4",gx_left+inward_left,-yhalf_left)):
                center=np.array([x,y])@transform+[hx,hy]
                result[f"ice.{label}:outer"]={"kind":"ice_circle","center_xy":center,"radius":float(r0),"angle_interval":[0.,2*np.pi]}
            if self.hockey_end_block_id is not None:
                if self.layout_version != 2 or self.board_block_id is None:
                    raise ValueError("shared end markings require the v2 rink frame")
                rleft,rright,goal_half,board_half=state.get(self.hockey_end_block_id)
                if min(rleft,rright,goal_half,board_half)<=0:
                    raise ValueError("invalid finite end-marking sizes")
                W,rx,ry,p,_=state.get(self.board_block_id)
                def board_end(sign, y):
                    # Exact intersection with this same continuous superellipse
                    # boundary, not a nominal end-plane or a new observation.
                    def membership(x):
                        q=np.abs(np.array([x,y])@transform)
                        return max(0.,(q[0]-(.5-rx))/rx)**p+max(0.,(q[1]-(W/2-ry))/ry)**p-1.
                    if membership(0.)>=0:
                        raise ValueError("trapezoid span does not cross the rink interior")
                    lo,hi=0.,2.
                    for _ in range(55):
                        mid=(lo+hi)/2
                        if membership(sign*mid)>0: hi=mid
                        else: lo=mid
                    return np.array([sign*(lo+hi)/2,y])@transform
                for side,gx,radius,sign in (("left",gx_left,rleft,-1),("right",gx_right,rright,1)):
                    start=-np.pi/2 if sign<0 else np.pi/2
                    result[f"ice.goal_crease_{side}"]={"kind":"ice_circle","center_xy":np.array([gx,0.])@transform,
                        "radius":float(radius),"angle_interval":[float(start+angle),float(start+angle+np.pi)]}
                    for edge,ysign in (("lower",-1),("upper",1)):
                        a=np.array([gx,ysign*goal_half])@transform
                        b=board_end(sign,ysign*board_half)
                        result[f"ice.goal_trapezoid_{side}_{edge}_outer:segment0"]={"kind":"ice_segment",
                            "endpoints_xy":np.r_[a,b],"boundary_endpoint":"derived_from_same_continuous_board"}
        return result


@dataclass(frozen=True)
class SceneModel:
    id: str
    coordinate_system_id: str
    board_block_id: str
    curves: tuple[CurveModel,...]
    geometry_family: str = "historical_independent_initial_geometry.v1"
    structural_family: StructuredIceFamily | None = None
    metric_verified: bool = False
    missing_inventory: tuple[str,...] = ()
    installations: tuple[InstallationModel,...] = ()

    def board(self,state):
        width,rx,ry,p,height=state.get(self.board_block_id)
        return ContinuousBoard({"width":width,"corner_rx":rx,"corner_ry":ry,"corner_exponent":p,"board_height":height})

    def resolved_definition(self, curve, state):
        # A source path alias shares the identical physical curve and layout.
        # It retains source identity without an independent canonical shape.
        visited=set()
        while curve.kind=="curve_alias":
            if curve.id in visited:
                raise ValueError("cyclic physical curve alias")
            visited.add(curve.id)
            target=curve.definition["target_curve_id"]
            curve=next((c for c in self.curves if c.id==target),None)
            if curve is None:
                raise KeyError(f"unknown alias target: {target}")
        d = plain(curve.definition)
        if d.get('kind')=='installed_artwork_polyline' and d.get('template_curve_id'):
            template=next((c for c in self.curves if c.id==d['template_curve_id']),None)
            if template is None or template.definition.get('kind')!='installed_artwork_polyline' or template.definition.get('template_curve_id'):
                raise ValueError('installed artwork requires one canonical static template curve')
            d['normalized_points']=plain(template.definition['normalized_points'])
        if self.structural_family is not None:
            generated=self.structural_family.curves(state)
            if curve.label in generated:
                return generated[curve.label]
        if curve.parameter_block_id is None:
            return d
        q=state.get(curve.parameter_block_id)
        if curve.kind=="ice_segment":
            d["endpoints_xy"]=q
        elif curve.kind=="ice_circle":
            d["center_xy"],d["radius"]=q[:2],q[2]
        elif curve.kind=="board_material_polyline":
            d["unwrapped_station_height"]=q.reshape(-1,2)
        elif curve.kind=="world_polyline":
            d["points_xyz"]=q.reshape(-1,3)
        return d

    def _unwrapped_definition(self,d,s,state,board=None):
        """Unique material-to-board chain, shared by projection and fast batches."""
        kind=d['kind']
        if kind in ('board_material_polyline','installed_artwork_polyline'):
            points=np.asarray(d['unwrapped_station_height'] if kind=='board_material_polyline' else d['normalized_points'])
            if len(points)==1:
                q=np.broadcast_to(points[0],s.shape+(2,))
            else:
                position=s*(len(points)-1)
                index=np.minimum(np.floor(position).astype(np.int64),len(points)-2)
                w=(position-index)[...,None]
                q=points[index]*(1-w)+points[index+1]*w
            if kind=='installed_artwork_polyline':
                installation=next(i for i in self.installations if i.id==d['installation_id'])
                return installation.unwrapped(q,state)
            if d.get('shared_layout_block_id'):
                q=apply_shared_similarity(q,state,d['shared_layout_block_id'],d['shared_content_anchor_unwrapped'])
            return q
        if kind=='board_boundary':
            board=board or self.board(state)
            part=d['course_segment']
            if part==-2:
                u=s*8;index=np.minimum(np.floor(u).astype(np.int64),7)
                coordinate=board.breaks[index]+(u-index)*board.lengths[index]
            elif part>=0:
                coordinate=board.breaks[part]+s*board.lengths[part]
            else:
                coordinate=s*board.perimeter
            return np.stack((coordinate,np.full(s.shape,board.height*d['height_fraction'])),axis=-1)
        raise ValueError(f'curve is not on continuous board: {kind}')

    def unwrapped_path(self,identifier,station,state,*,board=None):
        """Return [...,2] continuous board [course station,height], including static layout and aliases.

        A caller may reuse the same current-state board for boundary course conversion.
        Material curves do not construct/integrate a board at all here.
        """
        curve=next((c for c in self.curves if c.id==identifier),None)
        if curve is None:
            raise KeyError(f'unknown stable curve: {identifier}')
        s=np.asarray(station,dtype=np.float64)
        if not np.isfinite(s).all() or np.any(s<0) or np.any(s>1):
            raise ValueError('finite curve station outside [0,1]; no clamp')
        return self._unwrapped_definition(self.resolved_definition(curve,state),s,state,board)

    def world_path(self, identifier, station, state, *, board=None):
        curve=next((c for c in self.curves if c.id==identifier),None)
        if curve is None:
            raise KeyError(f"unknown stable curve: {identifier}")
        s=np.asarray(station,dtype=np.float64)
        if not np.isfinite(s).all() or np.any(s<0) or np.any(s>1):
            raise ValueError("finite curve station outside [0,1]; no clamp")
        d=self.resolved_definition(curve,state)
        kind=d["kind"]
        if kind=="ice_segment":
            ends=np.asarray(d["endpoints_xy"]).reshape(2,2)
            xy=(1-s[...,None])*ends[0]+s[...,None]*ends[1]
            return np.concatenate((xy,np.zeros(s.shape+(1,))),axis=-1)
        if kind=="world_polyline":
            points=np.asarray(d["points_xyz"])
            position=s*(len(points)-1)
            index=np.minimum(np.floor(position).astype(np.int64),len(points)-2)
            w=(position-index)[...,None]
            return points[index]*(1-w)+points[index+1]*w
        if kind=="ice_circle":
            interval=d["angle_interval"]
            angle=interval[0]+s*(interval[1]-interval[0])
            xy=np.asarray(d["center_xy"])+d["radius"]*np.stack((np.cos(angle),np.sin(angle)),axis=-1)
            return np.concatenate((xy,np.zeros(s.shape+(1,))),axis=-1)
        if kind in ('installed_artwork_polyline','board_material_polyline','board_boundary'):
            board=board or self.board(state)
            q=self._unwrapped_definition(d,s,state,board)
            return board.xyz(q[...,0],q[...,1])
        raise ValueError(f"unregistered geometry family: {kind}")

    def world_rows(self,curve_ids,stations,state):
        ids,s=np.broadcast_arrays(np.asarray(curve_ids),np.asarray(stations,dtype=np.float64))
        shape=s.shape
        ids,s=ids.reshape(-1),s.reshape(-1)
        result=np.empty((len(s),3))
        board=self.board(state)
        for identity in np.unique(ids):
            mask=ids==identity
            result[mask]=self.world_path(str(identity),s[mask],state,board=board)
        return result.reshape(shape+(3,))



    @staticmethod
    def board_station_tangent(board,station):
        s=np.mod(np.asarray(station,dtype=np.float64),board.perimeter);shape=s.shape
        flat=s.reshape(-1);part=np.minimum(np.searchsorted(board.breaks[1:],flat,side='left'),7)
        along=flat-board.breaks[part];out=np.empty((len(flat),2))
        for k in range(8):
            mask=part==k
            if not np.any(mask):continue
            if k%2==0:
                out[mask]={0:[1.,0.],2:[0.,1.],4:[-1.,0.],6:[0.,-1.]}[k];continue
            reverse=k in (1,5);target=board.corner_length-along[mask] if reverse else along[mask]
            theta=board.angle_for_length(target);c=np.maximum(np.cos(theta),0.);ss=np.maximum(np.sin(theta),0.)
            norm=c**board.exponent+ss**board.exponent;rho=norm**(-1/board.exponent)
            logder=(c**(board.exponent-1)*ss-ss**(board.exponent-1)*c)/norm
            deriv=np.stack((board.rx*rho*(-ss+c*logder),board.ry*rho*(c+ss*logder)),axis=-1)
            signs={1:[1,-1],3:[1,1],5:[-1,1],7:[-1,-1]}[k]
            out[mask]=deriv*signs/board.speed(theta)[:,None]*(-1 if reverse else 1)
        return out.reshape(shape+(2,))

    def unwrapped_path_jacobian(self,identifier,station,state,*,board=None):
        """Unique continuous-board q and dq/dstation, before the shared board operator."""
        curve=next(c for c in self.curves if c.id==identifier);d=self.resolved_definition(curve,state)
        s=np.asarray(station,dtype=np.float64);kind=d['kind']
        if kind not in ('board_material_polyline','installed_artwork_polyline','board_boundary'):
            raise ValueError('unwrapped station derivative requires a board material curve')
        if kind=='board_boundary':board=board or self.board(state)
        q=self.unwrapped_path(identifier,s,state,board=board)
        if kind=='board_boundary':
            part=d['course_segment']
            if part==-2:
                index=np.minimum(np.floor(s*8).astype(np.int64),7);speed=8*board.lengths[index]
            elif part>=0:speed=np.full(s.shape,board.lengths[part])
            else:speed=np.full(s.shape,board.perimeter)
            dq=np.stack((speed,np.zeros(s.shape)),axis=-1)
        else:
            points=np.asarray(d['normalized_points'] if kind=='installed_artwork_polyline' else d['unwrapped_station_height'])
            if len(points)==1:dq=np.zeros(s.shape+(2,))
            else:
                position=s*(len(points)-1);index=np.minimum(np.floor(position).astype(np.int64),len(points)-2)
                dq=(points[index+1]-points[index])*(len(points)-1)
                if kind=='installed_artwork_polyline':
                    installation=next(i for i in self.installations if i.id==d['installation_id'])
                    size=state.get(installation.size_block_id);angle=state.get(installation.placement_block_id)[2]
                    c,ss=np.cos(angle),np.sin(angle);dq=dq*size@np.array([[c,ss],[-ss,c]])
                    layout_id=installation.shared_layout_block_id
                else:layout_id=d.get('shared_layout_block_id')
                if layout_id:
                    _,_,scale_log,angle=state.get(layout_id);c,ss=np.cos(angle),np.sin(angle)
                    dq=dq*np.exp(scale_log)@np.array([[c,ss],[-ss,c]])
        return {'q':q,'unwrapped':q,'unwrapped_station_height':q,'jacobian_station':dq,'curve_id':identifier,'station_domain':[0.,1.]}

    def world_path_jacobian(self,identifier,station,state,*,board=None):
        """Exact finite-branch dXYZ/dstation; samples do not become landmarks."""
        curve=next(c for c in self.curves if c.id==identifier);d=self.resolved_definition(curve,state)
        s=np.asarray(station,dtype=np.float64);kind=d['kind']
        if kind in ('board_material_polyline','installed_artwork_polyline','board_boundary'):
            board=board or self.board(state);u=self.unwrapped_path_jacobian(identifier,s,state,board=board)
            q,dq=u['q'],u['jacobian_station'];xyz=board.xyz(q[...,0],q[...,1])
            tangent=self.board_station_tangent(board,q[...,0])
            derivative=np.concatenate((tangent*dq[...,0,None],dq[...,1,None]),axis=-1)
        else:
            xyz=self.world_path(identifier,s,state,board=board)
            if kind=='ice_segment':
                ends=np.asarray(d['endpoints_xy']).reshape(2,2);derivative=np.broadcast_to(np.r_[ends[1]-ends[0],0.],xyz.shape).copy()
            elif kind=='ice_circle':
                a,b=d['angle_interval'];angle=a+s*(b-a)
                derivative=d['radius']*(b-a)*np.stack((-np.sin(angle),np.cos(angle),np.zeros_like(angle)),axis=-1)
            elif kind=='world_polyline':
                points=np.asarray(d['points_xyz'])
                if len(points)==1:derivative=np.zeros_like(xyz)
                else:
                    position=s*(len(points)-1);index=np.minimum(np.floor(position).astype(np.int64),len(points)-2)
                    derivative=(points[index+1]-points[index])*(len(points)-1)
            else:raise ValueError('station derivative unregistered for current geometry')
        return {'xyz':xyz,'jacobian_station':derivative,'curve_id':identifier,'station_domain':[0.,1.]}

    def unwrapped_layout_jacobian(self,identifier,station,state,*,board=None):
        """Unique static group similarity derivative in continuous [course,height]."""
        curve=next(c for c in self.curves if c.id==identifier);d=self.resolved_definition(curve,state)
        if d['kind']!='board_material_polyline' or not d.get('shared_layout_block_id'):
            raise ValueError('curve is not registered to one static artwork-layout group')
        s=np.asarray(station,dtype=np.float64)
        q=self.unwrapped_path(identifier,s,state,board=board)
        key=d['shared_layout_block_id'];anchor=np.asarray(d['shared_content_anchor_unwrapped'])
        tx,ty,_,_=state.get(key);offset=q-anchor-[tx,ty]
        J=np.zeros(s.shape+(2,4));J[...,0,0]=1.;J[...,1,1]=1.;J[...,2]=offset
        J[...,3]=np.stack((-offset[...,1],offset[...,0]),axis=-1)
        return {'q':q,'unwrapped':q,'unwrapped_station_height':q,'jacobian_layout':J,'layout_block_id':key,
            'content_anchor_unwrapped':anchor,'anchor_is_physical_landmark_or_banner_border':False}

    def world_layout_jacobian(self,identifier,station,state,*,board=None):
        u=self.unwrapped_layout_jacobian(identifier,station,state,board=board);q,J=u['q'],u['jacobian_layout']
        board=board or self.board(state);tangent=self.board_station_tangent(board,q[...,0])
        world=np.empty(J.shape[:-2]+(3,4));world[...,:2,:]=tangent[..., :,None]*J[...,0,None,:];world[...,2,:]=J[...,1,:]
        return {'xyz':board.xyz(q[...,0],q[...,1]),'jacobian_layout':world,'layout_block_id':u['layout_block_id'],
            'content_anchor_unwrapped':u['content_anchor_unwrapped'],'anchor_is_physical_landmark_or_banner_border':False}

    def inspection_grid(self,state,*,spacing=.025):
        """World-coordinate diagnostic grid from the same current board shape."""
        width,rx,ry,exponent,_=state.get(self.board_block_id)
        if spacing<=0:
            raise ValueError("positive diagnostic grid spacing required")
        paths=[]
        for i,x in enumerate(np.arange(-.5,.5+spacing*.01,spacing)):
            u=max(0.,abs(float(x))-(.5-rx))/rx
            ymax=width/2 if u<=0 else width/2-ry+ry*max(0.,1-u**exponent)**(1/exponent)
            paths.append({"id":f"inspection_grid_x{i}","xyz":[[float(x),-float(ymax),0.],[float(x),float(ymax),0.]],"role":"engineering_world_grid_not_material"})
        for i,y in enumerate(np.arange(np.ceil(-width/2/spacing)*spacing,width/2+spacing*.01,spacing)):
            v=max(0.,abs(float(y))-(width/2-ry))/ry
            xmax=.5 if v<=0 else .5-rx+rx*max(0.,1-v**exponent)**(1/exponent)
            paths.append({"id":f"inspection_grid_y{i}","xyz":[[-float(xmax),float(y),0.],[float(xmax),float(y),0.]],"role":"engineering_world_grid_not_material"})
        board=self.board(state)
        for i,s in enumerate(np.linspace(0,board.perimeter,65)[:-1]):
            paths.append({"id":f"inspection_board_vertical{i}","xyz":board.xyz(np.array([s,s]),np.array([0.,board.height])).tolist(),"role":"engineering_board_grid_not_material"})
        return paths

    def intersection(self, curve_id_a, curve_id_b, branch_id, state, *, finite=True):
        """Analytic named branch; never choose the root nearest an initial guess."""
        lookup={c.id:c for c in self.curves}
        a,b=[self.resolved_definition(lookup[k],state) for k in (curve_id_a,curve_id_b)]
        if a["kind"]==b["kind"]=="ice_segment":
            if branch_id!="unique_line_line":
                raise ValueError("line/line requires explicit unique_line_line branch")
            pa=np.asarray(a["endpoints_xy"]).reshape(2,2); pb=np.asarray(b["endpoints_xy"]).reshape(2,2)
            matrix=np.column_stack((pa[1]-pa[0],-(pb[1]-pb[0])))
            if abs(np.linalg.det(matrix)) < 1e-13:
                raise ValueError("parallel/degenerate curves do not define unique landmark")
            station=np.linalg.solve(matrix,pb[0]-pa[0])
            if finite and np.any((station<0)|(station>1)):
                raise ValueError("intersection outside finite physical support")
            return np.r_[pa[0]+station[0]*(pa[1]-pa[0]),0.]
        swapped=a["kind"]=="ice_circle" and b["kind"]=="ice_segment"
        if swapped:
            a,b=b,a
        if a["kind"]=="ice_segment" and b["kind"]=="ice_circle":
            if branch_id not in ("lower_oriented_line_station","upper_oriented_line_station"):
                raise ValueError("line/circle branch must be topologically identified by oriented line station")
            ends=np.asarray(a["endpoints_xy"]).reshape(2,2)
            direction=ends[1]-ends[0]; offset=ends[0]-b["center_xy"]
            aa=float(direction@direction); bb=float(2*direction@offset); cc=float(offset@offset-b["radius"]**2)
            disc=bb*bb-4*aa*cc
            if aa<=0 or disc<=0:
                raise ValueError("line/circle branch absent or tangent, not uniquely differentiable")
            sign=-1 if branch_id=="lower_oriented_line_station" else 1
            t=(-bb+sign*np.sqrt(disc))/(2*aa)
            if finite and not 0<=t<=1:
                raise ValueError("line/circle landmark outside finite line support")
            return np.r_[ends[0]+t*direction,0.]
        if a["kind"]==b["kind"]=="ice_circle":
            if branch_id not in ("left_of_center_a_to_b","right_of_center_a_to_b"):
                raise ValueError("circle/circle branch requires oriented center-line side")
            ca,cb=np.asarray(a["center_xy"]),np.asarray(b["center_xy"])
            delta=cb-ca; distance=float(np.linalg.norm(delta))
            if distance==0:
                raise ValueError("concentric circles have no unique intersection")
            along=(a["radius"]**2-b["radius"]**2+distance**2)/(2*distance)
            height2=a["radius"]**2-along**2
            if height2<=0:
                raise ValueError("circle/circle branch absent or tangent")
            direction=delta/distance
            side=1 if branch_id=="left_of_center_a_to_b" else -1
            xy=ca+along*direction+side*np.sqrt(height2)*np.array([-direction[1],direction[0]])
            return np.r_[xy,0.]
        raise ValueError("intersection model not registered for these physical curves")

    def landmark_world(self, landmark: S.LandmarkFeature, state,*,board=None):
        pos=landmark.position
        if isinstance(pos,S.SharedPointPosition):
            owner=state._specs[pos.variable_block_id].owner
            if owner.kind!=S.OwnerKind.FEATURE or owner.id!=landmark.id:
                raise ValueError("fixed landmark coordinates must be owned by the shared semantic feature")
            if pos.coordinate_system_id != self.coordinate_system_id:
                raise ValueError("landmark coordinate mismatch")
            q=state.get(pos.variable_block_id)
            if len(q) not in (2,3):
                raise ValueError("shared world point must have two or three coordinates")
            return np.r_[q,0.] if len(q)==2 else q.copy()
        if isinstance(pos,S.SharedCurvePointPosition):
            owner=state._specs[pos.station_variable_block_id].owner
            if owner.kind!=S.OwnerKind.FEATURE or owner.id!=landmark.id:
                raise ValueError("fixed material-point station must be feature-owned, not observation-local")
            s=float(state.get(pos.station_variable_block_id)[0]);allowed=pos.allowed_interval
            if allowed.unit.unit!=S.Unit.DIMENSIONLESS:
                raise ValueError("this registered finite-branch model uses normalized dimensionless station")
            if (allowed.lower is not None and (s<allowed.lower or (s==allowed.lower and not allowed.lower_closed))) or (allowed.upper is not None and (s>allowed.upper or (s==allowed.upper and not allowed.upper_closed))):
                raise ValueError("shared fixed-point station outside its explicit permitted range")
            return self.world_path(pos.curve_feature_id,s,state,board=board)
        if isinstance(pos,S.KnownPointPosition):
            if pos.coordinate_system_id != self.coordinate_system_id:
                raise ValueError("known point coordinate mismatch")
            q=np.asarray(pos.coordinates)
            return np.r_[q,0.] if len(q)==2 else q
        if isinstance(pos,S.IntersectionPosition):
            return self.intersection(*pos.curve_feature_ids,pos.branch_id,state)
        if isinstance(pos,S.StructuralPointPosition):
            if pos.definition_id!='artwork_contour_corner.v1:'+landmark.id or len(pos.geometry_feature_ids)!=1:
                raise ValueError('unregistered structural landmark definition')
            curve=next(c for c in self.curves if c.id==pos.geometry_feature_ids[0])
            d=self.resolved_definition(curve,state)
            record=d.get('fixed_artwork_landmarks',{}).get(landmark.id)
            if not record or record.get('definition_kind')!='source_glyph_material_boundary_corner':
                raise ValueError('source contour corner lacks a registered physical definition')
            if not pos.definition_evidence_ids or not set(record['evidence_ids'])<=set(pos.definition_evidence_ids):
                raise ValueError('source corner evidence mismatch')
            return self.world_path(curve.id,float(record['station']),state,board=board)
        raise ValueError("structural point definition must have an explicitly registered entity evaluator")


@dataclass(frozen=True)
class ModelBundle:
    id: str
    revision: int
    scene: SceneModel
    rigs: tuple[RigModel,...]
    shots: tuple[ShotMotion,...]
    state: ParameterState
    artworks: tuple[ArtworkModel,...] = ()
    installations: tuple[InstallationModel,...] = ()
    provenance: Mapping = None
    landmarks: tuple[S.LandmarkFeature,...] = ()

    def __post_init__(self):
        if self.state.coordinate_system_id != self.scene.coordinate_system_id:
            raise ValueError("scene and camera parameter coordinates cannot be mixed")
        if len({r.id for r in self.rigs})!=len(self.rigs) or len({s.id for s in self.shots})!=len(self.shots):
            raise ValueError("duplicate rig/shot identity")
        if any(r.coordinate_system_id!=self.scene.coordinate_system_id for r in self.rigs):
            raise ValueError("cannot mix unproved camera gauge with scene")
        object.__setattr__(self,"provenance",frozen(self.provenance or {}))
        if self.scene.installations!=self.installations:
            object.__setattr__(self,"scene",replace(self.scene,installations=self.installations))

    def operator(self,shot_id="S01"):
        from .projector import ProjectionOperator
        return ProjectionOperator(self,shot_id)

    def with_state(self,state):
        if tuple(b.id for b in state.blocks)!=tuple(b.id for b in self.state.blocks):
            raise ValueError("state update changes layout; create an explicit model revision")
        return replace(self,state=state)

    def inventory(self,*,samples=128):
        if samples<2:
            raise ValueError("curve inspection requires at least two samples")
        s=np.linspace(0,1,samples)
        board=self.scene.board(self.state)
        curves=[]
        for c in self.scene.curves:
            # Keep every actual polyline corner. Uniform inspection samples
            # alone can skip short strokes and connect them with false chords.
            definition=self.scene.resolved_definition(c,self.state)
            points=definition.get('unwrapped_station_height',definition.get('normalized_points',definition.get('points_xyz',())))
            knots=np.linspace(0.,1.,len(points)) if len(points)>1 else np.empty(0)
            stations=np.unique(np.r_[s,knots]) if len(knots) else s
            xyz=self.scene.world_path(c.id,stations,self.state,board=board)
            curves.append({"id":c.id,"label":c.label,"entity_id":c.entity_id,"kind":c.kind,
                "xyz":xyz.tolist(),"initial_geometry_only":c.initial_geometry_only,
                "initialization_source":c.definition.get("initialization_source","d78_independent_initialization"),
                "source_path_id":c.definition.get("source_path_id",c.definition.get("original_path_id")),
                "group":"ice" if c.label.startswith("ice.") else "board" if c.label.startswith("board.") else "advertisement_print" if ".print." in c.label else "advertisement",
                "vertices_are_landmarks":False,"semantic_status":c.semantic_status})
        return {"schema":VERSION,"model_id":self.id,"model_revision":self.revision,
            "parameter_state_id":self.state.id,"parameter_revision":self.state.revision,"parameter_sha256":self.state.digest(),
            "coordinate_system_id":self.scene.coordinate_system_id,"unit":"scene_unit","metric_verified":self.scene.metric_verified,
            "geometry_family":self.scene.geometry_family,"curves":curves,"grid":self.scene.inspection_grid(self.state),
            "landmarks":[{"id":p.id,"label":p.label,"position_world":self.scene.landmark_world(p,self.state).tolist(),"position_kind":type(p.position).__name__} for p in self.landmarks],
            "board":{"xyz_bottom":board.xyz(np.linspace(0,board.perimeter,samples),0).tolist(),
                     "xyz_top":board.xyz(np.linspace(0,board.perimeter,samples),board.height).tolist()},
            "rigs":[r.inspection(self.state) for r in self.rigs],
            "installations":[{"id":a.id,"template_id":a.template_id,"template_match_status":a.template_match_status,
                "mapping_family":a.mapping_family,"extent_role":a.extent_role,"boundary_xyz":a.boundary_world(self.scene,self.state,samples_per_edge=max(32,samples//4)).tolist() if a.placement_block_id is not None and a.size_block_id is not None else None} for a in self.installations],
            "missing_items":list(self.scene.missing_inventory)+[f"rig:{r.id}:gauge_compatible_pivot_and_lever" for r in self.rigs if not r.available],
            "provenance":plain(self.provenance)}

    def inspection(self,native_pts=None,*,samples=128):
        """One same-state inspector input, complete scene plus all three rigs."""
        result=self.inventory(samples=samples)
        shot=self.shots[0]
        if native_pts is not None and (isinstance(native_pts,(bool,np.bool_)) or not isinstance(native_pts,(int,np.integer))):
            raise TypeError("inspection requires exact integer native PTS; no rounding")
        pts=shot.first_native_pts if native_pts is None else int(native_pts)
        operator=self.operator(shot.id)
        pose=operator.pose_at(operator.frame_time(np.asarray(pts,dtype=np.int64)))
        result["selected_exposure"]={"native_pts":pts,"source_seconds":pts*shot.time_base_num/shot.time_base_den,
            "shot_id":shot.id,"rig_id":shot.rig_id,"R_cw":pose["R_cw"].tolist(),"C_w":pose["C_w"].tolist(),
            "K":pose["K"].tolist(),"parameter_state_id":self.state.id,"model_revision":self.revision}
        for rig in result["rigs"]:
            if rig["id"]==shot.rig_id:
                rig.update(inspection_position_world=pose["C_w"].tolist(),inspection_direction_world=pose["R_cw"][2].tolist(),
                    inspection_position_kind="current_selected_exposure_rotation_derived_optical_center",inspection_source_pts=pts)
            elif "optical_center_reference_world" in rig:
                rig.update(inspection_position_world=rig["optical_center_reference_world"],inspection_direction_world=rig["reference_direction_world"],
                    inspection_position_kind="historical_demo_initial_reference_not_current_exposure_or_pivot",inspection_source_pts=rig["reference_source_pts"])
        result["inspection_contract"]="all scene geometry and current camera derive from this single model/state; other-rig markers are explicit initial references"
        return result


def from_d78(directory, *, mount_family=REPLAY_MOUNT):
    """Import independent replay data without importing prior optimizer state.

    TARGET_MOUNT explicitly disables extended terms and is a different candidate;
    call projector.compare_mount_families before claiming its equivalence.
    """
    package=load_initialization(directory)
    coordinate="d78_L1"
    blocks=[]; values=[]
    def add(key,owner_kind,owner,names,units,q):
        blocks.append(block(key,owner_kind,owner,names,units,coordinate))
        values.append(S.ParameterValue(key,tuple(float(x) for x in np.asarray(q).reshape(-1))))
    scene=package.scene_definition
    add("scene/board",S.OwnerKind.SCENE,"ankara2026_d78_initial_scene",["width","corner_rx","corner_ry","corner_exponent","board_height"],
        [S.Unit.SCENE_UNIT]*3+[S.Unit.DIMENSIONLESS,S.Unit.SCENE_UNIT],[scene[k] for k in ("width","corner_rx","corner_ry","corner_exponent","board_height")])
    cam=package.camera_definition
    for name,q,unit,names in (
        ("pivot",cam["pivot_world"],S.Unit.SCENE_UNIT,["x","y","z"]),
        ("lever",cam["lever_camera"],S.Unit.SCENE_UNIT,["x","y","z"]),
        ("reference_rotation_vector",[0.,0.,0.],S.Unit.RADIAN,["rx","ry","rz"]),
        ("pan_axis_world",cam["pan_axis_world"],S.Unit.DIMENSIONLESS,["nx","ny","nz"]),
        ("tilt_axis_camera",cam["tilt_axis_camera"],S.Unit.DIMENSIONLESS,["bx","by","bz"]),
        ("extended_mount",[cam["axis_spacing"],*cam["pupil_polynomial"]],S.Unit.SCENE_UNIT,["axis_spacing","pupil_z1","pupil_z2"])):
        add(f"rig/main/{name}",S.OwnerKind.RIG,"main",names,unit,q)
    optics=[cam["eta"],*cam["principal_point"],*cam["principal_zoom_slope"],*cam["k1_coefficients"],*cam["k2_coefficients"],*cam["tangential"],cam["signed_readout_seconds"]]
    add("rig/main/optics",S.OwnerKind.RIG,"main",["eta","cx","cy","cx_z","cy_z","k10","k11","k20","k21","p1","p2","signed_readout"],
        [S.Unit.DIMENSIONLESS]+[S.Unit.PIXEL]*4+[S.Unit.DIMENSIONLESS]*6+[S.Unit.SECOND],optics)
    add("shot/S01/controls",S.OwnerKind.SHOT,"S01",[f"{name}_{i}" for i in range(len(package.controls)) for name in ("alpha","beta","log_f")],
        [u for i in package.controls for u in (S.Unit.RADIAN,S.Unit.RADIAN,S.Unit.DIMENSIONLESS)],package.controls)
    curves=[]
    for p in package.paths:
        kind=p["kind"]; key=None
        if kind=="ice_segment":
            q=p["endpoints_xy"]; names=["ax","ay","bx","by"]
        elif kind=="ice_circle":
            q=[*p["center_xy"],p["radius"]]; names=["cx","cy","radius"]
        elif kind=="board_material_polyline":
            q=np.asarray(p["unwrapped_station_height"]).reshape(-1); names=[f"{axis}{i}" for i in range(len(q)//2) for axis in ("s","z")]
        else:
            q=None
        if q is not None:
            key=f"geometry/{p['id']}"
            add(key,S.OwnerKind.FEATURE,p["id"],names,S.Unit.SCENE_UNIT,q)
        curves.append(CurveModel(p["id"],p["label"],f"entity/{p['label'].split(':')[0]}",kind,key,p))
    state=ParameterState("d78_new_runtime_initial@0",0,coordinate,tuple(blocks),tuple(values))
    rig=RigModel("main","主机位",coordinate,mount_family,True,"independent_d78_initialization",
        notes="历史扩展安装仅复现候选；constant lever目标族另存比较，不把axis/pupil当用户事实。",
        zoom_reference_focal_px=cam["f_ref"],normalized_radius_domain=cam["radius_max"],
        reference_rotation_seed=tuple(np.asarray(cam["R0"]).reshape(-1)))
    rigs=(rig,RigModel("secondary","侧机位",coordinate,available=False,notes="历史demo光心属于另一未证明坐标变换，尚不能合入本共同状态。"),
          RigModel("end","端机位",coordinate,available=False,notes="历史demo光心属于另一未证明坐标变换，尚不能合入本共同状态。"))
    shot=ShotMotion("S01","main","official50_clock",256256,575488,256,1,12800,cam["first_control_time_seconds"],cam["control_dt_seconds"],len(package.controls),"shot/S01/controls")
    sc=SceneModel("ankara2026_d78_initial_scene",coordinate,"scene/board",tuple(curves),
        missing_inventory=("complete_venue_features_not_parameterized_by_d78_replay", "semantic_artwork_installation_objects_pending_explicit_binding"))
    return ModelBundle("ankara2026_S01_runtime",1,sc,rigs,(shot,),state,
        provenance={"independent_initialization":str(Path(directory)),"source_kind":"successful_physical_S01_candidate_data",
            "old_solver_code_imported":False,"old_optimizer_state_imported":False,
            "initial_geometry_is_user_fact":False,"current_mount_family":mount_family,
            "display_samples_are_source_observations":False})


def structured_candidate(model):
    """Create one explicit shared-geometry candidate from current initial values.

    This is not a projection-preserving conversion. Every replaced source curve
    is measured below, and historic independent endpoint blocks are removed from
    the candidate layout rather than left as inactive, rank-zero free variables.
    No nominal metre dimension, extra Lane edge or extra source pixel is added.
    """
    lookup={c.label:c for c in model.scene.curves}
    state=model.state
    defs={k:model.scene.resolved_definition(c,state) for k,c in lookup.items()}
    houses={};outer=[];ratios={k:[] for k in (("center","inner","middle") if any(x.endswith(":center") and x.startswith(("ice.F","ice.R")) for x in defs) else ("inner","middle"))}
    for label,d in defs.items():
        match=re.fullmatch(r"ice\.([FR])([1-4]):(center|inner|middle|outer)",label)
        if match and d["kind"]=="ice_circle":
            key=f"{match[1]}{match[2]}";houses.setdefault(key,{})[match[3]]=d
    if not houses:
        raise ValueError("shared F/R initializer requires named house geometry")
    coords=[];xf=[];xr=[]
    for key, rings in houses.items():
        d=next(iter(rings.values()));coords.append((int(key[1])-1,float(d["center_xy"][1])))
        (xf if key[0]=="F" else xr).append(float(d["center_xy"][0]))
        if "outer" in rings:
            ro=float(rings["outer"]["radius"]);outer.append(ro)
            for layer in ratios:
                if layer in rings:
                    ratios[layer].append(float(rings[layer]["radius"])/ro)
    if not xf or not xr or not outer or not all(ratios.values()):
        raise ValueError("not enough initial named F/R geometry for shared radius/layout")
    design=np.column_stack((np.ones(len(coords)),np.array(coords)[:,0]))
    y0,pitch=np.linalg.lstsq(design,np.array(coords)[:,1],rcond=None)[0]
    curling=[np.mean(xf),np.mean(xr),y0,pitch,np.median(outer),*[np.median(v) for v in ratios.values()]]
    widths=[];rows=[];rights=[];lefts=[]
    for row in range(1,5):
        edges={}
        for edge in (0,2):
            d=defs.get(f"ice.lane{row}:edge{edge}")
            if d:
                ends=np.asarray(d["endpoints_xy"]).reshape(2,2);edges[edge]=ends
                rights.append(float(ends[:,0].max()))
        if not edges:
            raise ValueError("Lane initial drawing support unavailable")
        if 0 in edges and 2 in edges:
            widths.append(abs(float(edges[0][:,1].mean()-edges[2][:,1].mean())))
        rows.append(edges)
        lefts.append(min(float(e[:,0].min()) for e in edges.values()))
    if not widths:
        raise ValueError("no observed initial Lane width support")
    width=float(np.mean(widths))
    carrier_ends=[]
    available=[]
    for row in range(1,5):
        d=defs.get(f"ice.house_connection_F{row}_R{row}:segment0")
        value=float(np.asarray(d["endpoints_xy"]).reshape(2,2)[:,0].max()) if d else None
        carrier_ends.append(value)
        if value is not None:
            available.append(value)
    if not available:
        raise ValueError("house-connection finite drawing support unavailable")
    carrier_ends=[float(np.median(available)) if x is None else x for x in carrier_ends]
    lane=[float(np.mean(rights)),width,*lefts,*carrier_ends]
    c0=defs["ice.C0:outer"];c=defs["ice.C:segment0"]
    ends=np.asarray(c["endpoints_xy"]).reshape(2,2)
    high,low=ends[np.argmax(ends[:,1])],ends[np.argmin(ends[:,1])]
    vec=high-low;angle=float(np.arctan2(-vec[0],vec[1]));axis=np.array([np.cos(angle),np.sin(angle)])
    # The centre is the rink origin by the confirmed relation. The old carrier
    # supplies orientation only; its old translation is deliberately removed.
    origin=np.zeros(2)
    offset=np.mean([abs(float((np.asarray(defs[f"ice.{label}:segment0"]["endpoints_xy"]).reshape(2,2).mean(axis=0)-origin)@axis)) for label in ("B1","B2")])
    transverse=np.array([-np.sin(angle),np.cos(angle)])
    low_y=float((low-origin)@transverse);high_y=float((high-origin)@transverse)
    hockey=[angle,float(offset),float(c0["radius"]),low_y,high_y]
    extra=None
    if all(k in defs for k in ("ice.goal_line_left:segment0","ice.goal_line_right:segment0",*[f"ice.A{i}:outer" for i in range(1,5)])):
        basis=np.stack((axis,transverse),axis=1)
        goals=[(np.asarray(defs[f"ice.goal_line_{side}:segment0"]["endpoints_xy"]).reshape(2,2).mean(axis=0)-origin)@basis for side in ("left","right")]
        faceoffs=[(np.asarray(defs[f"ice.A{i}:outer"]["center_xy"])-origin)@basis for i in range(1,5)]
        extra=[goals[0][0],goals[1][0],np.mean([faceoffs[2][0],faceoffs[3][0]])-goals[0][0],
               goals[1][0]-np.mean([faceoffs[0][0],faceoffs[1][0]]),np.mean([abs(faceoffs[2][1]),abs(faceoffs[3][1])]),
               np.mean([abs(faceoffs[0][1]),abs(faceoffs[1][1])])]
    end_sizes=None
    end_labels=[f"ice.goal_crease_{side}" for side in ("left","right")]
    trap_labels=[f"ice.goal_trapezoid_{side}_{edge}_outer:segment0" for side in ("left","right") for edge in ("lower","upper")]
    if extra is not None and all(k in defs for k in end_labels+trap_labels):
        radii=[]
        for name in end_labels:
            definition=defs[name]
            if definition['kind']=='ice_circle':
                radii.append(float(definition['radius']))
            else:
                q=np.asarray(definition['points_xyz'])[:,:2]
                # The already-saved semicircle's two end points give its
                # diameter. This is initialization from geometry, not fitting.
                radii.append(float(np.linalg.norm(q[-1]-q[0])/2))
        spans=[]
        for label in trap_labels:
            q=np.asarray(defs[label]['endpoints_xy']).reshape(2,2)@basis
            q=q[np.argsort(np.abs(q[:,0]))]
            spans.append(np.abs(q[:,1]))
        end_sizes=[*radii,*np.mean(spans,axis=0)]
    family=StructuredIceFamily("ankara_user_relations.v2","scene/curling_shared","scene/lane_shared","scene/hockey_shared",len(ratios),
        "scene/hockey_faceoff_shared" if extra is not None else None,
        "scene/C0_material_midline_radius" if "ice.C0:middle" in defs else None,
        layout_version=2,board_block_id=model.scene.board_block_id,
        hockey_end_block_id="scene/hockey_ends_shared" if end_sizes is not None else None)
    specs=list(state.blocks);vals=list(state.values)
    def add(key,names,units,q):
        specs.append(block(key,S.OwnerKind.SCENE,model.scene.id,names,units,model.scene.coordinate_system_id))
        vals.append(S.ParameterValue(key,tuple(float(x) for x in q)))
    add(family.curling_block_id,["xF","xR","first_row_y","pitch","outer_radius",*[f"{name}_outer_ratio" for name in ratios]],
        [S.Unit.SCENE_UNIT]*5+[S.Unit.DIMENSIONLESS]*len(ratios),curling)
    add(family.lane_block_id,["right_x","width",*[f"left_drawing_x{i}" for i in range(1,5)],*[f"centreline_right_drawing_x{i}" for i in range(1,5)]],S.Unit.SCENE_UNIT,lane)
    add(family.hockey_block_id,["angle","blue_offset","C0radius","drawing_y_low","drawing_y_high"],
        [S.Unit.RADIAN]+[S.Unit.SCENE_UNIT]*4,hockey)
    if family.hockey_midline_block_id is not None:
        add(family.hockey_midline_block_id,["material_width_midline_radius"],S.Unit.SCENE_UNIT,[defs["ice.C0:middle"]["radius"]])
    if extra is not None:
        add(family.hockey_extra_block_id,["goal_x_left","goal_x_right","faceoff_inward_left","faceoff_inward_right","faceoff_yhalf_left","faceoff_yhalf_right"],S.Unit.SCENE_UNIT,extra)
    if end_sizes is not None:
        add(family.hockey_end_block_id,["crease_radius_left","crease_radius_right","trapezoid_goal_half_span","trapezoid_board_half_span"],S.Unit.SCENE_UNIT,end_sizes)
    preliminary=ParameterState("shared_geometry_initial@1",state.revision+1,state.coordinate_system_id,tuple(specs),tuple(vals))
    generated=family.curves(preliminary)
    aliases={'ice.curling_centerline_row3:segment0':'ice.house_connection_F3_R3:segment0'}
    removed={c.parameter_block_id for c in model.scene.curves if c.label in generated and c.parameter_block_id}
    removed.update(c.parameter_block_id for c in model.scene.curves if c.label in aliases and c.parameter_block_id)
    specs=tuple(b for b in specs if b.id not in removed);vals=tuple(v for v in vals if v.variable_block_id not in removed)
    newstate=ParameterState("shared_geometry_initial@1",state.revision+1,state.coordinate_system_id,specs,vals)
    curves=[];seen=set();displacements=[]
    for curve in model.scene.curves:
        if curve.label in generated:
            curves.append(replace(curve,kind=generated[curve.label]["kind"],parameter_block_id=None,definition={**plain(curve.definition),**generated[curve.label],"structural_family_id":family.id},semantic_status="shared_user_relation_initial_geometry"))
            seen.add(curve.label)
        elif curve.label in aliases:
            target=next(c.id for c in model.scene.curves if c.label==aliases[curve.label])
            curves.append(replace(curve,kind='curve_alias',parameter_block_id=None,
                definition={**plain(curve.definition),'kind':'curve_alias','target_curve_id':target,
                    'identity_basis':'named curling row-3 centreline and F3/R3 carrier are the same user-defined line'},
                semantic_status='shared_row_centreline_alias'))
        else:
            curves.append(curve)
    for label,d in generated.items():
        if label not in seen:
            identifier=f"shared_geometry/{label}"
            curves.append(CurveModel(identifier,label,f"entity/{label.split(':')[0]}",d["kind"],None,d,
                semantic_status="derived_shared_structure_not_new_observation"))
    scene=replace(model.scene,curves=tuple(curves),geometry_family="user_shared_ice_relation_candidate.v2",structural_family=family)
    candidate=replace(model,id=model.id+"_shared_geometry_candidate",revision=model.revision+1,scene=scene,state=newstate)
    stations=np.linspace(0,1,33)
    for curve in model.scene.curves:
        if curve.label in generated:
            delta=np.linalg.norm(candidate.scene.world_path(curve.id,stations,newstate)-model.scene.world_path(curve.id,stations,state),axis=-1)
            displacements.append({"curve_id":curve.id,"label":curve.label,"maximum_scene_units":float(delta.max()),"rms_scene_units":float(np.sqrt(np.mean(delta**2)))})
    report={"schema":"rinksolve.shared_geometry_initialization.v1","source_state":state.id,"candidate_state":newstate.id,
        "projection_preserving":False,"unchanged_camera_parameters":True,"new_source_observations":0,
        "removed_independent_curve_parameter_blocks":sorted(removed),"curve_displacements":displacements,
        "governing_constraint_ids":list(family.constraint_ids),"metre_dimensions_used":False,
        "Lane_left_values":"finite drawing/source-support coverage, not physical endpoints or fourth edges",
        "new_derived_curves_are_evidence":False,"source_visual_acceptance":False}
    report.update(shared_layout_version=2,independent_Lane_centres_removed=4,independent_hockey_translation_removed=2,
        C0_is_rink_origin=True,Lane_centres_are_same_row_coordinates=True,
        end_markings_follow_shared_hockey_frame=end_sizes is not None)
    report['shared_row_centreline_aliases']=aliases
    return candidate,report


def shared_advertisement_candidate(model, pair_facts):
    """One static template and size per explicitly confirmed x/x+13 pair.

    This changes the new scene only. Historic d78 material samples supply the
    template where available; demo geometry supplies missing placement/inventory.
    A material bounding box is a numerical template chart, never a banner edge.
    """
    facts=pair_facts['constraint']
    if facts.get('paired_advertisement_numbers')!='x and x+13' or not facts.get('same_printed_template') or not facts.get('shared_printed_dimensions'):
        raise ValueError('explicit confirmed paired template/size relation required')
    state=model.state;board=model.scene.board(state);P=board.perimeter
    grouped={};untouched=[]
    for c in model.scene.curves:
        match=re.match(r'Ad(\d+)',c.label)
        if not match:untouched.append(c);continue
        n=int(match[1]);d=model.scene.resolved_definition(c,state)
        if d['kind']!='board_material_polyline':raise ValueError('unexpected advertisement input kind')
        q=np.asarray(d['unwrapped_station_height'],dtype=float).copy()
        q[:,0]=np.unwrap(q[:,0]/P*2*np.pi)/(2*np.pi)*P
        grouped.setdefault(n,[]).append((c,q))
    def normalize(parts):
        anchor=float(parts[0][1][0,0]);out=[]
        for c,q in parts:
            q=q.copy();q[:,0]+=np.round((anchor-q[:,0].mean())/P)*P;out.append((c,q))
        return out
    grouped={n:normalize(v) for n,v in grouped.items()}
    removed={c.parameter_block_id for parts in grouped.values() for c,_ in parts if c.parameter_block_id}
    removed.update(b.id for b in state.blocks if b.owner.kind in (S.OwnerKind.INSTALLATION,S.OwnerKind.ARTWORK))
    specs=[b for b in state.blocks if b.id not in removed];values=[v for v in state.values if v.variable_block_id not in removed]
    def add(key,kind,owner,names,units,q):
        specs.append(block(key,kind,owner,names,units,state.coordinate_system_id))
        values.append(S.ParameterValue(key,tuple(float(x) for x in q)))
    def extent(parts):
        q=np.concatenate([q for _,q in parts]);return q.min(axis=0),q.max(axis=0)
    def suffix(c,n):
        s=re.sub(r'^Ad\d+','',c.label)
        s=re.sub(r'Ad\d+:','',s)
        if 'circle' in s and 'print.' in s:return '.print.circle'
        return s
    new_curves=list(untouched);installations=[];artworks=[];reports=[];missing=[]
    previous_installations={a.id:a for a in model.installations}
    for first in range(1,14):
        members=(first,first+13)
        source=next((n for n in members if any(c.id.startswith('d78_') for c,_ in grouped.get(n,()))),None)
        if source is None:source=next((n for n in members if n in grouped),None)
        if source is None:
            missing.extend(members);continue
        parts=grouped[source]
        # A second historic version can have different component names and a
        # different placement. Do not superimpose it on the d78 material object.
        saved=[item for item in parts if item[0].id.startswith('d78_')]
        if saved:parts=saved
        low,high=extent(parts);size=high-low
        if np.any(size<=0):raise ValueError('degenerate static advertisement template')
        pair=f'artwork_pair/Ad{first}-Ad{first+13}'
        size_key=pair+'/size'
        add(size_key,S.OwnerKind.ARTWORK,pair,['material_width','material_height'],S.Unit.SCENE_UNIT,size)
        template_ids=[];missing_members=[];replaced=[]
        for n in members:
            iid=f'installation/Ad{n}'
            current=grouped.get(n)
            old_installation=previous_installations.get(iid)
            role=old_installation.extent_role if old_installation else 'unknown_do_not_infer_banner_from_print_canvas'
            if current is None:
                installations.append(InstallationModel(iid,'board/continuous',pair,None,size_key,pair,
                    source_status='confirmed_template_available_placement_missing',extent_role=role))
                missing.append(n);missing_members.append(n);continue
            if n==source:origin=low
            else:
                target_low,target_high=extent(current)
                # Preserve the current material centre as initialization. Equal
                # dimensions come from the shared template, not another fit.
                origin=(target_low+target_high)/2-size/2
            placement=iid+'/shared_placement'
            add(placement,S.OwnerKind.INSTALLATION,iid,['station','height','angle'],
                [S.Unit.SCENE_UNIT,S.Unit.SCENE_UNIT,S.Unit.RADIAN],[*origin,0.])
            installations.append(InstallationModel(iid,'board/continuous',pair,placement,size_key,pair,
                template_match_status='user_confirmed_paired_print_initial_shape_not_HD_verified',
                source_status='shared_static_pair_candidate',extent_role=role))
            targets={suffix(c,n):c for c,_ in current}
            used=set()
            for c,q in parts:
                tail=suffix(c,source);target=c if n==source else targets.get(tail)
                identifier=target.id if target else f'shared_pair/Ad{n}/{c.id}'
                definition={'kind':'installed_artwork_polyline','installation_id':iid,
                    'template_pair_id':pair,'source_template_curve_id':c.id,
                    'initialization_source':'shared_pair_from_d78_static_template' if saved else 'shared_pair_from_demo8a28_static_template',
                    'initialization_only':True,'vertices_are_landmarks':False,
                    'normalized_extent_is_banner_boundary':False,'shape_is_HD_verified':False}
                if n==source:
                    definition['normalized_points']=(q-low)/size
                    template_ids.append(identifier)
                else:definition['template_curve_id']=c.id
                # Keep the true source identity only where an existing path has
                # that identity; copied template parts are derived geometry.
                if target:
                    definition['historic_definition']=plain(target.definition)
                    used.add(target.id)
                new_curves.append(CurveModel(identifier,f'Ad{n}'+tail,f'entity/Ad{n}',
                    'installed_artwork_polyline',None,definition,
                    semantic_status='shared_pair_static_template_not_new_source_observation'))
            replaced.extend(c.id for c,_ in current if c.id not in used)
        artworks.append(ArtworkModel(pair,None,(ArtworkGroup(pair,tuple(template_ids),None),),
            'user_confirmed_pair_static_initial_shape_not_HD_verified',
            'One canonical curve set and one size block for both installations; origins remain independent.'))
        reports.append({'numbers':members,'template_source_ad':source,'shared_size_block_id':size_key,
            'canonical_template_curves':template_ids,'superseded_initial_curves':replaced,
            'missing_placement':missing_members,'template_bbox_is_not_new_external_border':True})
    newstate=ParameterState('shared_scene_and_artwork@1',state.revision+1,state.coordinate_system_id,tuple(specs),tuple(values))
    scene=replace(model.scene,curves=tuple(new_curves),missing_inventory=tuple(model.scene.missing_inventory)+tuple(f'Ad{n}_placement_unknown' for n in missing))
    candidate=replace(model,id=model.id+'_shared_ads',revision=model.revision+1,scene=scene,state=newstate,
        installations=tuple(installations),artworks=tuple(artworks))
    return candidate,{'schema':'rinksolve.shared_ad_pairs.v1','pairs':reports,
        'missing_placement':missing,'new_source_observations':0,'per_vertex_shape_variables':False,
        'pair_authority':pair_facts.get('authority'),'fitting_performed':False}


@dataclass(frozen=True)
class ParameterView:
    id: str
    semantic_block_id: str
    component_offset: int
    size: int
    vector_offset: int
    owner: S.OwnerRef


def pack_state(state, block_ids=None, *, split_cubic_controls=True):
    """Native numerical views preserve semantic identity/owner and scalar offsets.

    A semantic trajectory array is split into three-value controls for sparse
    factors. This does not introduce per-frame optical centers or new entities.
    """
    selected=tuple(block_ids) if block_ids is not None else tuple(b.id for b in state.blocks)
    if len(set(selected))!=len(selected):
        raise ValueError("duplicate parameter selection")
    vector=[];views=[];cursor=0
    for key in selected:
        spec=state._specs[key];q=state.get(key)
        if spec.mode!=S.VariableMode.FREE:
            raise ValueError("native free-vector selection includes nonfree block")
        width=3 if split_cubic_controls and key.endswith('/controls') else len(q)
        if len(q)%width:
            raise ValueError("control array is not triplet aligned")
        for offset in range(0,len(q),width):
            identity=f"{key}#components{offset}:{width}"
            views.append(ParameterView(identity,key,offset,width,cursor,spec.owner))
            vector.extend(q[offset:offset+width]);cursor+=width
    return np.asarray(vector,dtype=np.float64),tuple(views)


def unpack_state(state, views, vector, *, expected_revision=None, new_id=None):
    flat=np.asarray(vector,dtype=np.float64)
    if flat.ndim!=1 or not np.isfinite(flat).all():
        raise ValueError("finite flat native parameter vector required")
    if sum(v.size for v in views)!=len(flat):
        raise ValueError("native parameter vector/view dimension mismatch")
    changes={};covered=set();cursor=0
    for v in views:
        if v.vector_offset!=cursor or v.semantic_block_id not in state._specs:
            raise ValueError("native view scalar offset or identity mismatch")
        spec=state._specs[v.semantic_block_id]
        if spec.owner!=v.owner or v.size<=0 or v.component_offset<0 or v.component_offset+v.size>len(spec.components):
            raise ValueError("native view owner or component range mismatch")
        q=changes.setdefault(v.semantic_block_id,state.get(v.semantic_block_id).copy())
        for i in range(v.component_offset,v.component_offset+v.size):
            key=(v.semantic_block_id,i)
            if key in covered:
                raise ValueError("overlapping native parameter views")
            covered.add(key)
        q[v.component_offset:v.component_offset+v.size]=flat[cursor:cursor+v.size]
        cursor+=v.size
    return state.updated(changes,expected_revision=expected_revision,new_id=new_id)


def supplement_demo_initialization(model, seed_path):
    """Assemble one explicit new initial state, not two overlaid old models.

    The successful demo supplies only missing inventory and camera-center
    references. Existing d78 semantic labels and parameters are not overridden.
    A known continuous material board is mapped by a documented station chart;
    this numerical initialization is not entity identification or measurement.
    """
    seed=json.loads(Path(seed_path).read_text(encoding="utf-8-sig"))
    if not seed.get("can_supply_unobserved_inventory") or seed.get("can_override_observed_d78_geometry"):
        raise ValueError("demo seed does not authorize this missing-inventory role")
    if seed.get("verified_metre_scale"):
        raise ValueError("this initializer expects explicitly unverified L1 units")
    transform=seed["transform"]
    if not np.allclose(transform["rotation"],np.eye(3),rtol=0,atol=0) or not np.allclose(transform["origin"],0,rtol=0,atol=0):
        raise ValueError("explicit axis/origin correspondence required for this initial gauge composition")
    coordinate="ankara2026_combined_initial_L1"
    state=model.state
    specs=[replace(b,components=tuple(replace(c,unit=replace(c.unit,reference_id=coordinate)) if c.unit.unit==S.Unit.SCENE_UNIT else c for c in b.components)) for b in state.blocks]
    vals=list(state.values);curves=list(model.scene.curves)
    labels={c.label for c in curves};added=[];skipped=[];source_map={c.id:"d78_independent_initialization" for c in curves}
    def add(key,owner_kind,owner,names,units,q):
        specs.append(block(key,owner_kind,owner,names,units,coordinate))
        vals.append(S.ParameterValue(key,tuple(float(x) for x in np.asarray(q).reshape(-1))))
    old_board=next(p for p in seed["paths"] if p["id"]=="ice_board_contact")
    polygon=np.asarray(old_board["xyz"],dtype=np.float64)[:,:2]
    starts=polygon;vectors=np.roll(polygon,-1,axis=0)-polygon
    lens=np.linalg.norm(vectors,axis=1);cumulative=np.r_[0.,np.cumsum(lens)];old_perimeter=float(cumulative[-1])
    current_board=model.scene.board(state)
    # Demo course begins at right straight -> upper-right corner; the new
    # continuous course uses the documented bottom-left -> bottom-right seam.
    seam=float(current_board.breaks[3])
    max_chart_distance=0.
    def chart(xyz):
        nonlocal max_chart_distance
        xyz=np.asarray(xyz,dtype=np.float64);xy=xyz[:,:2]
        # Same declared board entity; distance minimization finds its numeric
        # station only. It does not identify which material/feature was seen.
        difference=xy[:,None,:]-starts[None,:,:]
        fraction=np.clip(np.einsum('nsi,si->ns',difference,vectors)/lens[None,:]**2,0,1)
        error=difference-fraction[...,None]*vectors[None,:,:]
        distance=np.einsum('nsi,nsi->ns',error,error)
        index=np.argmin(distance,axis=1)
        row=np.arange(len(xy));u=fraction[row,index]
        station=cumulative[index]+u*lens[index]
        station=np.unwrap(station/old_perimeter*2*np.pi)/(2*np.pi)
        station=seam+station*current_board.perimeter
        max_chart_distance=max(max_chart_distance,float(np.sqrt(distance[row,index]).max()))
        return np.column_stack((station,xyz[:,2]))
    installation_parts={};known_numbers={int(match[1]) for c in curves if (match:=re.match(r"Ad([0-9]+)",c.label))}
    for p in seed["paths"]:
        group=p["group"];pid=p["id"]
        if group in ("board","grid"):
            skipped.append({"id":pid,"reason":"regenerated from current board/gauge, not an overlaid historic course or grid"})
            continue
        number=re.match(r"Ad(\d+)",p["label"])
        if number:
            known_numbers.add(int(number[1]))
        if group=="ice":
            if re.fullmatch(r"(?:[FR][1-4]|C0|A[1-4]):(?:outer|middle|inner|center)",pid):
                label="ice."+pid;kind="ice_circle"
            elif re.fullmatch(r"lane[1-4]:[012]",pid):
                name,edge=pid.split(':');label=f"ice.{name}:edge{edge}";kind="ice_segment"
            elif pid.endswith(':0'):
                label="ice."+pid[:-2]+":segment0";kind="ice_segment"
            else:
                label="ice."+pid;kind="world_polyline"
        elif group=="advertisement":
            label=pid.replace(':','.').replace('_xyz','');kind="board_material_polyline"
        elif group=="advertisement_print":
            ad,_,component=pid.partition(':contour:');label=f"{ad}.print.{component}";kind="board_material_polyline"
        else:
            skipped.append({"id":pid,"reason":"unregistered inventory group"});continue
        xyz=np.asarray(p["xyz"],dtype=np.float64)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2:
            raise ValueError(f"malformed initial curve {pid}")
        # Preserve known installation material extent for object registration,
        # even where the current S01 material curve is already d78-owned.
        charted=chart(xyz) if kind=="board_material_polyline" else None
        if group=="advertisement" and number:
            installation_parts.setdefault(int(number[1]),[]).append(charted)
        if label in labels:
            skipped.append({"id":pid,"reason":"existing d78/shared semantic curve takes precedence","label":label})
            continue
        labels.add(label)
        definition={"kind":kind,"initialization_source":"demo8a28_normalized","initialization_only":True,
            "source_path_id":pid,"source_coordinate_system_id":seed["coordinate_system_id"],"semantic_binding_required_before_factors":True,
            "vertices_are_landmarks":False}
        identifier=f"demo_initial/{pid}";key=f"geometry/{identifier}"
        if kind=="board_material_polyline":
            q=charted.reshape(-1);definition["unwrapped_station_height"]=charted
            names=[f"{axis}{i}" for i in range(len(q)//2) for axis in ("s","z")]
            definition["chart_mapping"]="old declared board station fraction to current continuous-board course; static initializer, no entity identity inference"
        elif kind=="ice_circle":
            sample=xyz[:-1,:2] if np.linalg.norm(xyz[0]-xyz[-1])<1e-12 else xyz[:,:2]
            A=np.column_stack((2*sample,np.ones(len(sample))))
            solution=np.linalg.lstsq(A,np.sum(sample*sample,axis=1),rcond=None)[0]
            center=solution[:2];radius=float(np.sqrt(solution[2]+center@center))
            q=np.r_[center,radius];names=["cx","cy","radius"]
            definition.update(center_xy=center,radius=radius,angle_interval=[0.,2*np.pi],
                circle_fit_max_radius_deviation=float(np.abs(np.linalg.norm(sample-center,axis=1)-radius).max()))
        elif kind=="ice_segment":
            q=xyz[[0,-1],:2].reshape(-1);names=["ax","ay","bx","by"]
            definition["endpoints_xy"]=q
        else:
            q=xyz.reshape(-1);names=[f"{axis}{i}" for i in range(len(xyz)) for axis in ("x","y","z")]
            definition["points_xyz"]=xyz
        add(key,S.OwnerKind.FEATURE,identifier,names,S.Unit.SCENE_UNIT,q)
        curves.append(CurveModel(identifier,label,f"entity/{label.split(':')[0]}",kind,key,definition,
            semantic_status="demo_missing_inventory_initialization_not_new_source_evidence"))
        added.append({"id":identifier,"label":label,"source_path_id":pid,"kind":kind})
        source_map[identifier]="demo8a28_normalized_missing_inventory"
    rigs=[]
    references={r["id"]:r for r in seed["rig_references"]}
    for rig in model.rigs:
        ref=references.get(rig.id)
        if ref is None:
            rigs.append(replace(rig,coordinate_system_id=coordinate));continue
        key=f"rig/{rig.id}/optical_center_initial_reference"
        add(key,S.OwnerKind.RIG,rig.id,["reference_Cx","reference_Cy","reference_Cz"],S.Unit.SCENE_UNIT,ref["position_xyz"])
        rigs.append(replace(rig,coordinate_system_id=coordinate,optical_center_reference_block_id=key,
            direction_reference=tuple(ref["direction_xyz"]),reference_source_pts=int(ref["source_pts"]),
            notes=rig.notes+" 另有demo同L1数值坐标的初值光心参考；不是pivot，不是已拟合安装。"))
    installation_registry={int(x["user_number"]):x for x in seed.get("advertisement_registry",[])}
    known_numbers.update(installation_registry)
    installations=list(model.installations)
    for number in sorted(known_numbers):
        identifier=f"installation/Ad{number}"
        if any(a.id==identifier for a in installations):
            continue
        parts=installation_parts.get(number)
        placement_key=size_key=None
        if parts:
            anchor=float(parts[0][0,0]);allq=[]
            for q in parts:
                q=q.copy();q[:,0]+=np.round((anchor-q[:,0])/current_board.perimeter)*current_board.perimeter;allq.append(q)
            allq=np.concatenate(allq);low=allq.min(axis=0);high=allq.max(axis=0)
            if np.all(high>low):
                placement_key=f"{identifier}/placement";size_key=f"{identifier}/size"
                add(placement_key,S.OwnerKind.INSTALLATION,identifier,["station","bottom_height","angle"],
                    [S.Unit.SCENE_UNIT,S.Unit.SCENE_UNIT,S.Unit.RADIAN],[*low,0.])
                add(size_key,S.OwnerKind.INSTALLATION,identifier,["width","height"],S.Unit.SCENE_UNIT,high-low)
        installations.append(InstallationModel(identifier,"board/continuous",None,placement_key,size_key,
            source_status="demo_external_border_initialization" if parts else "print_material_present_external_border_unknown",
            extent_role=installation_registry.get(number,{}).get("extent_scope",("internal_graphic_only" if number in (5,18) else "external_material_border") if parts else "unknown_do_not_infer_banner_from_print_canvas")))
    newstate=ParameterState("combined_initial_L1@1",state.revision+1,coordinate,tuple(specs),tuple(vals))
    missing=tuple(x for x in model.scene.missing_inventory if not x.startswith("complete_venue_features"))
    scene=replace(model.scene,coordinate_system_id=coordinate,curves=tuple(curves),missing_inventory=missing)
    provenance=plain(model.provenance)
    provenance.update(initialization_composition="one new candidate state; d78 visible geometry plus explicit demo missing inventory",
        demo_uniform_gauge_seed=str(Path(seed_path)),demo_coordinate_system_id=seed["coordinate_system_id"],
        numerical_gauge_alignment="both sources normalized L=1 with declared center/x-right/y-up/z-up; engineering initialization, not proof of physical same-shape",
        historic_demo_projection_invariance=seed.get("gauge_projection_check"),curve_initialization_sources=source_map)
    candidate=replace(model,id="ankara2026_complete_initial",revision=model.revision+1,scene=scene,rigs=tuple(rigs),state=newstate,
        installations=tuple(installations),provenance=provenance)
    report={"schema":"rinksolve.full_inventory_initialization.v1","candidate_state":newstate.id,
        "initial_state_sha256":newstate.digest(),"added_curves":added,"skipped":skipped,
        "current_d78_source_curve_values_changed":False,"current_d78_camera_values_changed":False,
        "same_complete_scene_as_historical_demo":False,"demonstration_gauge_is_physical_calibration":False,
        "numeric_board_station_chart_max_distance_to_demo_polyline_scene_units":max_chart_distance,
        "installation_count":len(installations),"rig_registry_count":len(rigs),"historical_reference_centers":len(references),
        "unapproved_fake_pivots_created":False,"standard_unobserved_marks_added":False,
        "source_visual_acceptance":False}
    return candidate,report


def build_initial_model(directory, demo_seed_path, *, shared_geometry=True,current_source_registry=None):
    model=from_d78(directory)
    if current_source_registry is None:
        default_registry=Path(directory).parent/'observations/source_registry.json'
        current_source_registry=default_registry if default_registry.exists() else None
    if current_source_registry is not None:
        model,_=apply_current_semantics(model,current_source_registry)
    model,_=supplement_demo_initialization(model,demo_seed_path)
    if shared_geometry:
        model,_=structured_candidate(model)
    return model


def save_model_bundle(model,directory):
    """Self-contained current model/state; no archived code or receipt prerequisite."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays={f"b{i:05d}":model.state.get(b.id) for i,b in enumerate(model.state.blocks)}
    np.savez(directory/'parameters.npz',**arrays)
    scene=model.scene
    metadata={"schema":VERSION,"id":model.id,"revision":model.revision,
        "scene":{"id":scene.id,"coordinate_system_id":scene.coordinate_system_id,"board_block_id":scene.board_block_id,
                 "geometry_family":scene.geometry_family,"structural_family":asdict(scene.structural_family) if scene.structural_family else None,
                 "metric_verified":scene.metric_verified,"missing_inventory":list(scene.missing_inventory),
                 "curves":[{"id":c.id,"label":c.label,"entity_id":c.entity_id,"kind":c.kind,"parameter_block_id":c.parameter_block_id,
                            "definition":plain(c.definition),"initial_geometry_only":c.initial_geometry_only,
                            "vertices_are_landmarks":c.vertices_are_landmarks,"semantic_status":c.semantic_status} for c in scene.curves]},
        "state":{"id":model.state.id,"revision":model.state.revision,"coordinate_system_id":model.state.coordinate_system_id,
                 "sha256":model.state.digest(),"blocks":[S.to_json_value(b) for b in model.state.blocks],
                 "arrays":{b.id:f"b{i:05d}" for i,b in enumerate(model.state.blocks)}},
        "rigs":[asdict(x) for x in model.rigs],"shots":[asdict(x) for x in model.shots],
        "installations":[asdict(x) for x in model.installations],
        "artworks":[{"id":a.id,"template":S.to_json_value(a.template) if a.template else None,
                     "source_templates":[S.to_json_value(t) for t in a.source_templates],
                     "layout":S.to_json_value(a.layout) if a.layout else None,
                     "groups":[asdict(g) for g in a.groups],"source_status":a.source_status,"notes":a.notes} for a in model.artworks],
        "landmarks":[S.to_json_value(p) for p in model.landmarks],
        "provenance":plain(model.provenance),"old_optimizer_state_imported":False}
    (directory/'model.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return directory/'model.json'


def load_model_bundle(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'model.json').read_text(encoding='utf-8-sig'))
    if metadata.get('schema')!=VERSION:
        raise ValueError("current model schema mismatch")
    sm=metadata['state'];arrays=np.load(directory/'parameters.npz',allow_pickle=False)
    try:
        blocks=tuple(S.from_json_value(b,S.VariableBlock) for b in sm['blocks'])
        values=tuple(S.ParameterValue(b.id,tuple(float(x) for x in arrays[sm['arrays'][b.id]])) for b in blocks)
    finally:
        arrays.close()
    state=ParameterState(sm['id'],sm['revision'],sm['coordinate_system_id'],blocks,values)
    if state.digest()!=sm['sha256']:
        raise ValueError("current model/state numerical identity mismatch")
    sc=dict(metadata['scene']);sc['curves']=tuple(CurveModel(**c) for c in sc['curves']);sc['missing_inventory']=tuple(sc['missing_inventory'])
    if sc['structural_family'] is not None:
        f=dict(sc['structural_family']);f['constraint_ids']=tuple(f['constraint_ids']);sc['structural_family']=StructuredIceFamily(**f)
    rigs=[]
    for item in metadata['rigs']:
        r=dict(item)
        for key in ('governing_constraint_ids','reference_rotation_seed','direction_reference','pan_axis_seed','tilt_axis_world_seed','pan_tangent_basis'):
            if r.get(key) is not None:
                r[key]=tuple(r[key])
        rigs.append(RigModel(**r))
    artworks=[]
    for item in metadata['artworks']:
        a=dict(item);a['template']=S.from_json_value(a['template'],S.ArtworkTemplate) if a['template'] else None
        a['source_templates']=tuple(S.from_json_value(t,S.ArtworkTemplate) for t in a['source_templates'])
        a['layout']=S.from_json_value(a['layout'],S.ArtworkLayout) if a['layout'] else None
        groups=[]
        for g in a['groups']:
            g=dict(g);g['shape_ids']=tuple(g['shape_ids']);g['source_feature_ids']=tuple(g['source_feature_ids']);groups.append(ArtworkGroup(**g))
        a['groups']=tuple(groups);artworks.append(ArtworkModel(**a))
    return ModelBundle(metadata['id'],metadata['revision'],SceneModel(**sc),tuple(rigs),
        tuple(ShotMotion(**s) for s in metadata['shots']),state,tuple(artworks),
        tuple(InstallationModel(**a) for a in metadata['installations']),metadata['provenance'],
        tuple(S.from_json_value(p,S.LandmarkFeature) for p in metadata.get('landmarks',[])))


def apply_current_semantics(model, source_registry_path):
    """Apply explicit current source revisions by exact material/path identity.

    Historical names remain provenance. This is not a nearest-curve classifier;
    numeric geometry is preserved while current measurement identity is recorded.
    """
    registry=json.loads(Path(source_registry_path).read_text(encoding="utf-8-sig"))
    current=registry['legacy_material_binding']['materials']
    current_by_path={}
    for item in current:
        for p in item['paths']:
            current_by_path.setdefault(p['id'],[]).append((item,p))
    changes=[];curves=[]
    for c in model.scene.curves:
        d=plain(c.definition);original=d.get('original_path_id')
        candidates=[(item,p) for item,p in current_by_path.get(original,[]) if item['index']==d.get('material_index') and p['index']==d.get('branch_index')]
        if len(candidates)==1 and candidates[0][0]['label']!=c.label:
            item,p=candidates[0];label=item['label']
            d.update(historical_label=c.label,label=label,current_semantic_source=str(Path(source_registry_path)),
                current_material_index=item['index'],current_branch_index=p['index'])
            if label=='ice.C0:middle':
                d.update(measurement='arithmetic_width_midpoint_of_two_actual_material_edges',same_radius_as_A_outer=False)
            changes.append({'curve_id':c.id,'original_path_id':original,'historical_label':c.label,'current_label':label,
                'current_material_index':item['index'],'current_branch_index':p['index'],'numeric_curve_changed':False})
            curves.append(replace(c,label=label,definition=d))
        else:
            curves.append(c)
    provenance=plain(model.provenance);provenance['current_semantic_revisions']=changes
    result=replace(model,revision=model.revision+1,scene=replace(model.scene,curves=tuple(curves)),provenance=provenance)
    report={'schema':'rinksolve.current_semantic_crosswalk.v1','registry_path':str(Path(source_registry_path)),
        'changes':changes,'pixel_or_parameter_values_changed':False,'classification_by_nearest_model':False}
    return result,report


def source_path_crosswalk(model,source_registry_path):
    registry=json.loads(Path(source_registry_path).read_text(encoding="utf-8-sig"))
    by_path={}
    for curve in model.scene.curves:
        original=curve.definition.get('original_path_id')
        if original:
            by_path.setdefault(original,[]).append(curve)
    entries=[];unbound=[]
    for material in registry['legacy_material_binding']['materials']:
        for path in material['paths']:
            possible=by_path.get(path['id'],[])
            exact=[c for c in possible if c.definition.get('material_index')==material['index'] and c.definition.get('branch_index')==path['index']]
            if len(exact)==1:
                chosen=exact[0]
            elif len(possible)==1 and possible[0].label==material['label']:
                chosen=possible[0]
            else:
                unbound.append({'current_material_index':material['index'],'current_path_index':path['index'],
                    'current_path_id':path['id'],'label':material['label'],
                    'reason':'missing_exact_source_path_geometry' if not possible else 'ambiguous_or_semantically_revised_source_path'})
                continue
            entries.append({'current_material_index':material['index'],'current_path_index':path['index'],
                'current_path_id':path['id'],'label':material['label'],'stable_curve_id':chosen.id,'current_curve_label':chosen.label,
                'binding':'explicit_original_source_path_identity','curve_station_domain':[0.,1.],
                'parameter_state_id':model.state.id,'model_revision':model.revision})
    return {'schema':'rinksolve.current_path_crosswalk.v1','registry':str(Path(source_registry_path)),
        'model_id':model.id,'model_revision':model.revision,'parameter_state_id':model.state.id,
        'parameter_sha256':model.state.digest(),'entries':entries,'unbound_paths':unbound,
        'same_label_alone_is_a_binding':False,'nearest_geometry_is_a_binding':False}


def pack_reference_gauge(state,block_ids,*,control_block_id='shot/S01/controls'):
    raw,raw_views=pack_state(state,block_ids)
    views=[];values=[];cursor=0
    for v in raw_views:
        if v.semantic_block_id==control_block_id and v.component_offset==6:
            v=replace(v,id=f"{control_block_id}#components8:1",component_offset=8,size=1)
        q=state.get(v.semantic_block_id)[v.component_offset:v.component_offset+v.size]
        views.append(replace(v,vector_offset=cursor));values.extend(q);cursor+=v.size
    return np.asarray(values,dtype=np.float64),tuple(views)


def unpack_reference_gauge(state,views,vector,*,control_block_id='shot/S01/controls',expected_revision=None,new_id=None):
    intermediate=unpack_state(state,views,vector,expected_revision=expected_revision,new_id=new_id)
    q=intermediate.get(control_block_id).copy().reshape(-1,3)
    q[2,:2]=-.25*(q[1,:2]+q[3,:2])
    values=tuple(S.ParameterValue(v.variable_block_id,tuple(float(x) for x in q.reshape(-1))) if v.variable_block_id==control_block_id else v for v in intermediate.values)
    return ParameterState(intermediate.id,intermediate.revision,intermediate.coordinate_system_id,intermediate.blocks,values)


def reference_gauge_component_mapping(control_index,component):
    if not 0<=component<3 or control_index<0:
        raise ValueError("invalid logical control component")
    if control_index==2 and component<2:
        return ((1,component,-.25),(3,component,-.25))
    return ((control_index,component,1.),)


def apply_shared_similarity(q,state,block_id,anchor):
    q=np.asarray(q,dtype=np.float64);anchor=np.asarray(anchor,dtype=np.float64)
    spec=state._specs[block_id]
    if spec.owner.kind!=S.OwnerKind.ARTWORK:
        raise ValueError('static advertisement layout must be owned by its artwork group')
    tx,ty,log_scale,angle=state.get(block_id);c,s=np.cos(angle),np.sin(angle)
    return (q-anchor)*np.exp(log_scale)@np.array([[c,s],[-s,c]])+anchor+[tx,ty]


def attach_current_print_initialization(model, extracted_path):
    data=json.loads(Path(extracted_path).read_text(encoding='utf-8-sig'))
    if data.get('schema')!='rinksolve.current_print_initialization.v1':
        raise ValueError('wrong current print shape initialization schema')
    state=model.state;curves=list(model.scene.curves);source_ids={c.definition.get('original_path_id') for c in curves}
    for r in data['records']:
        if r['path_id'] in source_ids:
            raise ValueError('requested new finite path already has distinct geometry')
        points=np.asarray(r['normalized_points'],dtype=np.float64);s0,w,z0,z1=r['installation_initial']
        if points.ndim!=2 or points.shape[1]!=2 or w<=0 or z1<=z0:
            raise ValueError('invalid explicit current finite print initializer')
        material=points*[w,z1-z0]+[s0,z0]
        identifier=f"current_material_{r['material_index']:04d}/path{r['branch_index']}"
        definition={'kind':'board_material_polyline','unwrapped_station_height':material,
            'material_index':r['material_index'],'branch_index':r['branch_index'],'original_path_id':r['path_id'],
            'label':r['label'],'closed':r['closed'],'vertices_are_landmarks':False,
            'historical_physical_point_flag':r['physical_point'],'initialization_source':'current_delivered_static_print_vertices',
            'source_model_sha256':data['source_sha256'],'source_provenance_sha256':r['source_provenance_sha256'],
            'station_parameterization':'linear by actual original vertex index; no synthetic closure',
            'shape_is_hd_verified':False,'raw_source_confirmation_is_not_revoked':True}
        curves.append(CurveModel(identifier,r['label'],f"entity/{r['label'].split('.')[0]}",'board_material_polyline',None,definition,
            semantic_status='exact_current_finite_material_path_initial_geometry'))
    # Canonicalize every same-ad material path as a static object, including its
    # old named boundary/corner curves. No optimizer-visible per-vertex variables.
    by_ad={}
    for c in curves:
        match=re.match(r'Ad([0-9]+)',c.label)
        if match and c.kind=='board_material_polyline':
            by_ad.setdefault(int(match[1]),[]).append(c)
    specs=list(state.blocks);values=list(state.values);remove=set();replacements={};artworks=list(model.artworks);installations=list(model.installations)
    for number,parts in sorted(by_ad.items()):
        canonical=[]
        for c in parts:
            d=model.scene.resolved_definition(c,state) if c.parameter_block_id else plain(c.definition)
            q=np.asarray(d['unwrapped_station_height']);canonical.append(q)
            if c.parameter_block_id:remove.add(c.parameter_block_id)
        allq=np.concatenate(canonical);anchor=allq.mean(axis=0)
        gid=f'artwork_group/Ad{number}/whole_material_object';key=f'{gid}/similarity'
        specs.append(block(key,S.OwnerKind.ARTWORK,gid,['station_shift','height_shift','log_uniform_scale','angle'],
            [S.Unit.SCENE_UNIT,S.Unit.SCENE_UNIT,S.Unit.DIMENSIONLESS,S.Unit.RADIAN],state.coordinate_system_id))
        values.append(S.ParameterValue(key,(0.,0.,0.,0.)))
        for c,q in zip(parts,canonical):
            d=plain(c.definition);d.update(unwrapped_station_height=q,shared_layout_block_id=key,shared_content_anchor_unwrapped=anchor,
                shared_group_id=gid,content_anchor_is_advertisement_border=False,shape_vertices_are_optimizer_variables=False,
                object_layout_family='shared_static_similarity_2d.v1')
            replacements[c.id]=replace(c,parameter_block_id=None,definition=d)
        group=ArtworkGroup(gid,tuple(c.id for c in parts),key,source_variant_status='canonical_source_initializer_not_HD_approval')
        artworks.append(ArtworkModel(f'artwork_initial/Ad{number}',None,(group,),
            'static_source_geometry_initializer_pending_HD_component_replacement',
            'One shared object transform; content anchor is only a numeric origin, not an external banner border or aspect constraint.'))
        installations=[replace(a,shared_layout_block_id=key,shared_content_anchor_unwrapped=tuple(float(x) for x in anchor)) if a.id==f'installation/Ad{number}' else a for a in installations]
    specs=tuple(b for b in specs if b.id not in remove);values=tuple(v for v in values if v.variable_block_id not in remove)
    newstate=ParameterState('static_artwork_objects_initial@1',state.revision+1,state.coordinate_system_id,specs,values)
    sc=replace(model.scene,curves=tuple(replacements.get(c.id,c) for c in curves))
    provenance=plain(model.provenance);provenance.update(current_finite_print_initialization=str(Path(extracted_path)),
        current_finite_print_source_sha256=data['source_sha256'],object_layout_family='shared_static_similarity_2d.v1',
        no_per_frame_or_per_vertex_shape_parameters=True)
    candidate=replace(model,id=model.id+'_current_print_objects',revision=model.revision+1,scene=sc,state=newstate,
        installations=tuple(installations),artworks=tuple(artworks),provenance=provenance)
    # Actual identity-layout replay check for every pre-existing material branch.
    samples=np.linspace(0,1,13);worst=0.
    for c in model.scene.curves:
        if c.id in replacements:
            before=model.scene.world_path(c.id,samples,state);after=candidate.scene.world_path(c.id,samples,newstate)
            worst=max(worst,float(np.max(np.abs(before-after))))
    return candidate,{'schema':'rinksolve.current_print_object_import.v1','new_exact_finite_paths':len(data['records']),
        'static_object_groups':len(by_ad),'removed_vertex_geometry_blocks':len(remove),'per_frame_or_per_vertex_variables_added':False,
        'preexisting_world_identity_layout_max_scene_units':worst,'new_source_observations_added':0,
        'shape_is_HD_verified':False,'source_confirmation_revoked':False}


def attach_prepared_landmarks(model,prepared_manifest):
    manifest=json.loads(Path(prepared_manifest).read_text(encoding='utf-8-sig'))
    blocks=list(model.state.blocks);values=list(model.state.values);landmarks=list(model.landmarks)
    for item in manifest.get('new_intersection_landmarks',[]):
        if any(p.id==item['id'] for p in landmarks):continue
        b=S.from_json_value(item['typed_variable_block'],S.VariableBlock);pos=S.from_json_value(item['typed_position'],S.SharedCurvePointPosition)
        if b.owner.kind!=S.OwnerKind.FEATURE or b.owner.id!=item['id']:
            raise ValueError('fixed cross-frame landmark must own one shared station')
        blocks.append(b);values.append(S.ParameterValue(b.id,(float(item['station_seed']),)))
        landmarks.append(S.LandmarkFeature(item['id'],item['entity_id'],'s01_intersections_current_verified',
            item['id'],pos))
    state=ParameterState('current_shared_landmarks_initial@1',model.state.revision+1,model.state.coordinate_system_id,tuple(blocks),tuple(values))
    return replace(model,revision=model.revision+1,state=state,landmarks=tuple(landmarks))


def with_minimal_rig_chart(model,rig_id='main'):
    """Six-DOF reference frame/axis chart; no measured-axis values are frozen.

    Two S2 parameters move the pan axis, one S1 parameter selects the tilt axis
    in the exact perpendicular world plane, and reference R0 retains 3 SO3 DOF.
    Seed vectors are chart bases, not physical constraints or Gaussian priors.
    """
    rig=next(r for r in model.rigs if r.id==rig_id)
    if rig.axis_parameterization!='raw_source_replay.v1':
        raise ValueError('axis chart conversion expects an explicit replay/raw initial state')
    c=rig.definition(model.state);n0=np.asarray(c['pan_axis_world']);n0=n0/np.linalg.norm(n0)
    w0=np.asarray(c['R0']).T@np.asarray(c['tilt_axis_camera']);w0=w0-n0*float(w0@n0);w0/=np.linalg.norm(w0)
    basis0=w0;basis1=np.cross(n0,w0)
    remove={rig.key('pan_axis_world'),rig.key('tilt_axis_camera')}
    if rig.mount_family==TARGET_MOUNT:
        remove.add(rig.key('extended_mount'))
    specs=[b for b in model.state.blocks if b.id not in remove]
    values=[v for v in model.state.values if v.variable_block_id not in remove]
    for key,names,q in ((rig.key('pan_axis_chart'),['pan_tangent_0','pan_tangent_1'],[0.,0.]),
                        (rig.key('tilt_axis_chart'),['tilt_angle_in_pan_perpendicular_plane'],[0.])):
        specs.append(block(key,S.OwnerKind.RIG,rig_id,names,S.Unit.RADIAN,model.state.coordinate_system_id))
        values.append(S.ParameterValue(key,tuple(q)))
    values=[S.ParameterValue(v.variable_block_id,(0.,0.,0.)) if v.variable_block_id==rig.key('reference_rotation_vector') else v for v in values]
    state=ParameterState('minimal_rig_chart_initial@1',model.state.revision+1,model.state.coordinate_system_id,tuple(specs),tuple(values))
    changed=replace(rig,axis_parameterization='coupled_s2_perpendicular.v1',pan_axis_seed=tuple(n0),
        tilt_axis_world_seed=tuple(w0),pan_tangent_basis=tuple(np.stack((basis0,basis1)).reshape(-1)),
        reference_rotation_seed=tuple(np.asarray(c['R0']).reshape(-1)),
        notes=rig.notes+' Pan S2(2), tilt-in-perpendicular-plane S1(1), reference R0 SO3(3); chart seeds are numerical bases, not fixed measured axes.')
    candidate=replace(model,id=model.id+'_minimal_rig_chart',revision=model.revision+1,state=state,
        rigs=tuple(changed if r.id==rig_id else r for r in model.rigs))
    before=model.operator();after=candidate.operator();times=before.frame_time(before.native_pts)
    a,b=before.pose_at(times),after.pose_at(times)
    report={'schema':'rinksolve.exact_head_axis_chart.v1','rig_id':rig_id,'pan_axis_dof':2,'tilt_perpendicular_dof':1,'reference_rotation_dof':3,
        'seed_axes_are_fixed_physical_facts':False,'world_vertical_pan_axis_assumed':False,
        'maximum_initial_rotation_difference':float(np.max(np.abs(a['R_cw']-b['R_cw']))),
        'maximum_initial_center_difference_scene_units':float(np.max(np.abs(a['C_w']-b['C_w']))),
        'removed_raw_axis_parameter_blocks':sorted(remove),'model_revision':candidate.revision,'parameter_state_sha256':state.digest()}
    return candidate,report


def rechart_current_print_initialization(model,extracted_path,target_charts_path):
    """Re-express only source-initialized print branches in one same-Ad d78 chart.

    This is a recorded initializer coordinate conversion, not a physical template
    shape assertion, source-pixel entity admission, or independent world xy scale.
    Official HD replacements are protected.  A content-only chart never creates
    an external banner boundary.
    """
    source=json.loads(Path(extracted_path).read_text(encoding='utf-8-sig'))
    target=json.loads(Path(target_charts_path).read_text(encoding='utf-8-sig'))
    if source['schema']!='rinksolve.current_print_initialization.v1' or target['schema']!='rinksolve.d78_ad_initial_charts.v1':
        raise ValueError('unregistered print initialization chart data')
    if target['coordinate_system_id']!=model.state.coordinate_system_id:
        raise ValueError('target chart gauge differs from current shared scene')
    charts={r['advertisement_number']:np.asarray(r['chart_s0_width_z0_z1'],dtype=float) for r in target['charts']}
    old={c.id:c for c in model.scene.curves};replacements={};by_ad={};protected=[]
    for r in source['records']:
        identifier=f"current_material_{r['material_index']:04d}/path{r['branch_index']}"
        curve=old[identifier];d=plain(curve.definition)
        if d.get('source_template_id') or curve.kind=='curve_alias':
            protected.append(identifier);continue
        if d.get('initialization_source')!='current_delivered_static_print_vertices':
            raise ValueError('new chart conversion must not overwrite an unrelated source or solved geometry')
        number=int(re.match(r'Ad([0-9]+)',r['label']).group(1));chart=charts[number]
        if d.get('shared_layout_block_id') and not np.array_equal(model.state.get(d['shared_layout_block_id']),np.zeros(4)):
            raise ValueError('chart conversion requires explicit identity layout initialization, not silent solved-state replacement')
        xy=np.asarray(r['normalized_points'],dtype=float);s0,width,z0,z1=chart
        q=xy*[width,z1-z0]+[s0,z0]
        d.update(unwrapped_station_height=q,initialization_source='current_static_vertices_recharted_to_same_Ad_d78_numeric_chart',
            previous_source_ad_chart=r['installation_initial'],current_Ad_initial_chart=chart,
            initialization_chart_source=str(Path(target_charts_path)),
            initialization_chart_source_sha256=target['source_data_sha256'],
            chart_conversion_is_only_initializer_coordinates=True,
            chart_conversion_is_world_xy_stretch=False,
            source_shape_is_approved_physical_fact=False,
            source_pixel_entity_admission_proved_by_chart=False,
            chart_extent_is_not_automatically_banner_boundary=True)
        replacements[identifier]=replace(curve,definition=d)
        by_ad.setdefault(number,{'source_chart':r['installation_initial'],'target_chart':chart.tolist(),'curves':[]})['curves'].append(identifier)
    # The actual known d78 outer/material installation chart replaces only the
    # numeric initializer of an existing same-object extent. Unknown print-only
    # extents remain absent; Ad5 keeps internal_graphic_only semantics.
    updates={};extent_updates=[]
    for a in model.installations:
        match=re.fullmatch(r'installation/Ad([0-9]+)',a.id)
        if not match or int(match.group(1)) not in by_ad:continue
        number=int(match.group(1));s0,width,z0,z1=charts[number]
        if a.placement_block_id is not None and a.size_block_id is not None:
            updates[a.placement_block_id]=[s0,z0,0.]
            updates[a.size_block_id]=[width,z1-z0]
            extent_updates.append({'installation_id':a.id,'extent_role':a.extent_role,'source':'same_Ad_d78_numeric_installation_chart_initialization','new_external_boundary_created':False})
    state=model.state.updated(updates,new_id=model.state.id+'/print_charts') if updates else model.state
    provenance=plain(model.provenance);provenance['same_Ad_print_chart_conversion']={'source_initializer':str(Path(extracted_path)),
        'target_chart_source':str(Path(target_charts_path)),'target_chart_source_data_sha256':target['source_data_sha256'],
        'one_static_chart_per_Ad':True,'world_xy_scale_added':False,'per_frame_or_per_curve_transforms_added':False,
        'shape_initializer_is_not_physical_fact':True,'source_association_not_validated_by_chart':True}
    candidate=replace(model,id=model.id+'/same_Ad_print_charts',revision=model.revision+1,
        scene=replace(model.scene,curves=tuple(replacements.get(c.id,c) for c in model.scene.curves)),state=state,provenance=provenance)
    report={'schema':'rinksolve.same_Ad_print_chart_conversion.v1','base_model_revision':model.revision,'candidate_model_revision':candidate.revision,
        'source_records':len(source['records']),'recharted_branches':len(replacements),'protected_HD_branches':protected,
        'per_Ad':by_ad,'extent_initializer_updates':extent_updates,'new_world_xy_scale':False,'new_shape_or_frame_DOF':0,
        'source_UV_ID_PTS_modified':False,'no_new_physical_shape_constraint':True,
        'remaining_limit':'shape variant and per-source entity association still require real native evidence; chart conversion is not acceptance'}
    return candidate,report
