"""Protect the recovered first shot while fitting advertisements.

The reference is already a self-contained current ModelBundle. This module
never opens historical packets, receipts, archives or solver source code.
"""
from __future__ import annotations
from dataclasses import fields,is_dataclass
from enum import Enum
from collections.abc import Mapping
import hashlib,json,re
import numpy as np
from . import schema as S


def _value(value):
    if is_dataclass(value):return {f.name:_value(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Enum):return value.value
    if isinstance(value,Mapping):return {str(k):_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_value(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def _encoded(value):
    return json.dumps(_value(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')


def _is_advertisement(curve):
    # Explicit recovered semantic labels, not world position or residual size.
    return re.match(r'^Ad[0-9]+(?:[.:/]|$)',curve.label) is not None


class OriginalReferenceGuard:
    """Immutable original camera, motion, rink and non-ad material definitions."""
    def __init__(self,reference):
        self.reference_identity=dict(model_id=reference.id,model_revision=reference.revision,
            parameter_state_id=reference.state.id,parameter_sha256=reference.state.digest())
        ad_blocks={c.parameter_block_id for c in reference.scene.curves if _is_advertisement(c)}-{None}
        for installation in reference.scene.installations:
            ad_blocks.update(x for x in (installation.placement_block_id,installation.size_block_id) if x)
        non_ad_blocks={c.parameter_block_id for c in reference.scene.curves if not _is_advertisement(c)}-{None}
        non_ad_blocks.add(reference.scene.board_block_id)
        # A shared block used by any non-ad material remains protected even if
        # an advertisement also refers to it.
        ad_blocks.difference_update(non_ad_blocks)
        self.protected_block_ids=tuple(b.id for b in reference.state.blocks if b.id not in ad_blocks)
        self._specs={b.id:_encoded(b) for b in reference.state.blocks if b.id in self.protected_block_ids}
        self._parameters={k:reference.state.get(k).astype('<f8',copy=False).tobytes() for k in self.protected_block_ids}
        self._curves={c.id:_encoded(c) for c in reference.scene.curves if not _is_advertisement(c)}
        self._scene=_encoded(self._scene_identity(reference))
        self._rigs=_encoded(reference.rigs);self._shots=_encoded(reference.shots)
        h=hashlib.sha256();h.update(self._scene);h.update(self._rigs);h.update(self._shots)
        for k in self.protected_block_ids:h.update(k.encode('utf8'));h.update(self._specs[k]);h.update(self._parameters[k])
        for k in sorted(self._curves):h.update(k.encode('utf8'));h.update(self._curves[k])
        self.protected_sha256=h.hexdigest()

    @staticmethod
    def _scene_identity(model):
        scene=model.scene
        return dict(coordinate_system_id=scene.coordinate_system_id,board_block_id=scene.board_block_id,
            geometry_family=scene.geometry_family,structural_family=scene.structural_family,
            metric_verified=scene.metric_verified,state_coordinate_system_id=model.state.coordinate_system_id)

    def assert_model(self,candidate):
        if _encoded(self._scene_identity(candidate))!=self._scene:raise ValueError('原冰场坐标/围挡/几何定义发生改变')
        if _encoded(candidate.rigs)!=self._rigs:raise ValueError('原相机定义发生改变')
        if _encoded(candidate.shots)!=self._shots:raise ValueError('原镜头时钟或轨迹定义发生改变')
        specs={b.id:b for b in candidate.state.blocks}
        for key in self.protected_block_ids:
            if key not in specs or _encoded(specs[key])!=self._specs[key]:raise ValueError('原非广告参数定义发生改变: '+key)
            if candidate.state.get(key).astype('<f8',copy=False).tobytes()!=self._parameters[key]:raise ValueError('原非广告参数数值发生改变: '+key)
        curves={c.id:c for c in candidate.scene.curves}
        for key,expected in self._curves.items():
            if key not in curves or _encoded(curves[key])!=expected:raise ValueError('原冰线/围挡材料曲线发生改变: '+key)
        return dict(schema='rinksolve.original_first_shot_preservation.v1',preserved=True,
            reference=self.reference_identity,protected_sha256=self.protected_sha256,
            protected_blocks=len(self.protected_block_ids),protected_curves=len(self._curves),
            original_camera_motion_ice_and_board_bitidentical=True)

    def assert_free_blocks(self,candidate,physical_block_ids):
        self.assert_model(candidate)
        specs={b.id:b for b in candidate.state.blocks}
        for key in physical_block_ids:
            if key in self._parameters:raise ValueError('原相机/冰线/围挡不得进入广告自由向量: '+key)
            if key not in specs or specs[key].owner.kind not in (S.OwnerKind.ARTWORK,S.OwnerKind.INSTALLATION):
                raise ValueError('广告物理自由量必须属于全曝光共享对象/安装，不允许逐顶点或逐帧变形: '+key)
        return True


def assert_preserves_original(reference,candidate):
    return OriginalReferenceGuard(reference).assert_model(candidate)


def assert_same_ice_projection(reference,candidate,curve_ids,stations,native_pts):
    """An optional one-time actual forward regression, never a startup history chain."""
    report=assert_preserves_original(reference,candidate)
    ids=np.asarray(curve_ids);stations=np.asarray(stations,dtype=float);pts=np.asarray(native_pts)
    if ids.shape!=stations.shape or ids.shape!=pts.shape or pts.dtype.kind not in 'iu':raise ValueError('冰线回归需要同形有限station和整数原生PTS')
    known={c.id:c for c in reference.scene.curves}
    if any(key not in known or known[key].kind not in ('ice_circle','ice_segment') for key in np.unique(ids)):
        raise ValueError('冰线回归只接收原版冰面实体，不混广告或未知深度')
    ops=[m.operator('S01') for m in (reference,candidate)];pred=[]
    for model,op in zip((reference,candidate),ops):pred.append(op.project_rs(model.scene.world_rows(ids,stations,model.state),pts,root_iterations=64))
    for key in ('uv','row','status_code','numerical_root_valid'):
        if not np.array_equal(pred[0][key],pred[1][key],equal_nan=True):raise ValueError('原冰线真实前向发生改变: '+key)
    return dict(report,ice_rows=int(ids.size),native_exposures=int(len(np.unique(pts))),formal_uv_row_status_bitidentical=True)
