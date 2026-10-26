# Adapted from user-supplied RinkSpace_v014_calibration_companion/code/scene_camera.py.
# Original SHA-256: 74ceae4a77d4fabb821b7d75b33d1bfbd8e39132958967695f7b35a5754bb07f. See provenance.json for exact changes.
"""Headless camera/scene mathematics for the actual v0.14 scene schema.

No renderer, downloads, automatic repository changes or calibration claims.
World: existing X-right/Y-up/Z-above-ice convention. Camera: OpenCV x-right,
y-down,z-forward. Units remain the scene units unless explicitly converted.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def _array(x: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite array of shape {shape}, got {a.shape}')
    return a


@dataclass
class Camera:
    center_xyz: np.ndarray
    rvec_world_to_camera: np.ndarray
    K_px: np.ndarray
    distortion_opencv: np.ndarray

    def __post_init__(self) -> None:
        self.center_xyz = _array(self.center_xyz, (3,), 'center_xyz')
        self.rvec_world_to_camera = _array(self.rvec_world_to_camera, (3,), 'rvec')
        self.K_px = _array(self.K_px, (3, 3), 'K')
        self.distortion_opencv = np.asarray(self.distortion_opencv, dtype=float).reshape(-1)
        if len(self.distortion_opencv) not in (0, 4, 5, 8, 12, 14) or not np.isfinite(self.distortion_opencv).all():
            raise ValueError('distortion must have 0,4,5,8,12 or 14 finite OpenCV coefficients')
        if self.K_px[0, 0] <= 0 or self.K_px[1, 1] <= 0 or not np.allclose(self.K_px[2], [0, 0, 1]):
            raise ValueError('invalid camera intrinsic matrix')
        if abs(self.K_px[0, 1]) > 1e-10 or abs(self.K_px[1, 0]) > 1e-10:
            raise ValueError('This implementation assumes zero skew; do not silently ignore skew.')

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec_world_to_camera)[0]

    @property
    def tvec(self) -> np.ndarray:
        return -self.R @ self.center_xyz

    def project(self, points_xyz: Any) -> tuple[np.ndarray, np.ndarray]:
        points = np.ascontiguousarray(np.asarray(points_xyz, float).reshape(-1, 3))
        if not np.isfinite(points).all():
            raise ValueError('cannot project null/nonfinite 3D coordinates')
        camera_points = (self.R @ (points-self.center_xyz).T).T
        pixels = cv2.projectPoints(points, self.rvec_world_to_camera, self.tvec,
                                  self.K_px, self.distortion_opencv)[0].reshape(-1, 2)
        return pixels, camera_points[:, 2]  # caller MUST reject non-positive depths.

    def ice_homography_undistorted(self) -> np.ndarray:
        """Only for undistorted pixel coordinates. Not a raw distorted-image H."""
        R=self.R
        return self.K_px @ np.column_stack([R[:, 0], R[:, 1], self.tvec])

    def pixel_to_ice(self, pixel_uv: Any, boundary_xy: Any | None = None,
                     min_ray_sine: float = 1e-5) -> dict[str, Any]:
        """Ray/plane intersection, not a body-part's vertical ground projection.

        min_ray_sine is an explicit numerical degeneracy guard, NOT an accuracy
        acceptance threshold. Reject low-angle solutions by an error budget too.
        """
        uv = _array(pixel_uv, (2,), 'pixel_uv')
        if not 0 < min_ray_sine < 1:
            raise ValueError('min_ray_sine must lie between 0 and 1')
        # Iterative inverse distortion with tighter stopping than the generic default.
        try:
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
            if hasattr(cv2, "undistortPointsIter"):
                undist = cv2.undistortPointsIter(uv.reshape(1,1,2), self.K_px,
                    self.distortion_opencv, None, None, criteria).reshape(2)
            else:  # OpenCV 5 exposes criteria on undistortPoints itself.
                undist = cv2.undistortPoints(uv.reshape(1,1,2), self.K_px,
                    self.distortion_opencv, criteria=criteria).reshape(2)
        except cv2.error:
            return {'valid':False, 'reason':'inverse_distortion_failed', 'xyz':None, 'ray_sine':None}
        if not np.isfinite(undist).all():
            return {'valid':False, 'reason':'nonfinite_inverse_distortion', 'xyz':None, 'ray_sine':None}
        ray = self.R.T @ np.array([undist[0], undist[1], 1.0])
        length = np.linalg.norm(ray)
        if not np.isfinite(length) or length <= 0:
            return {'valid':False, 'reason':'nonfinite_inverse_camera_ray', 'xyz':None, 'ray_sine':None}
        ray /= length
        # Fixed-point inverse distortion may terminate without solving the
        # requested pixel, especially outside the physical lens domain. Such
        # a ray must not become a plausible-looking, "valid" ice coordinate.
        projected, _ = self.project([self.center_xyz+ray])
        roundtrip = float(np.linalg.norm(projected[0]-uv))
        if not np.isfinite(roundtrip) or roundtrip > .05:
            return {'valid':False, 'reason':'inverse_distortion_roundtrip_failed',
                    'xyz':None, 'ray_sine':None,
                    'inverse_reprojection_error_px':roundtrip if np.isfinite(roundtrip) else None}
        sine = float(abs(ray[2]))
        if sine < min_ray_sine:
            return {'valid':False, 'reason':'ray_near_parallel_to_ice','xyz':None,'ray_sine':sine}
        distance = -self.center_xyz[2]/ray[2]
        if distance <= 0:
            return {'valid':False, 'reason':'intersection_not_in_front_of_camera','xyz':None,'ray_sine':sine}
        xyz = self.center_xyz+distance*ray
        xyz[2]=0.0
        if not np.isfinite(xyz).all():
            return {'valid':False, 'reason':'nonfinite_ice_intersection', 'xyz':None, 'ray_sine':sine}
        if boundary_xy is not None:
            # Optional contact/boundary operation; calibration needs no Shapely.
            from shapely.geometry import Point, Polygon
            if not Polygon(np.asarray(boundary_xy)[:, :2]).buffer(1e-8).covers(Point(xyz[:2])):
                return {'valid':False,'reason':'intersection_outside_rink','xyz':xyz.tolist(),'ray_sine':sine}
        return {'valid':True,'reason':'ray_intersection_only','xyz':xyz.tolist(),
                'ray_sine':sine,'camera_and_scene_uncertainty_included':False}

    def pixel_sensitivity(self, pixel_uv: Any, step_px: float = 0.25) -> np.ndarray:
        """d(X,Y)/d(u,v), in scene-units/pixel. Not total calibration uncertainty."""
        uv=_array(pixel_uv,(2,),'pixel_uv')
        if not np.isfinite(step_px) or step_px <= 0: raise ValueError('step_px must be positive')
        columns=[]
        for i in range(2):
            d=np.zeros(2);d[i]=step_px
            a=self.pixel_to_ice(uv+d);b=self.pixel_to_ice(uv-d)
            if not a['valid'] or not b['valid']: raise ValueError('sensitivity unavailable near invalid ray')
            columns.append((np.asarray(a['xyz'])[:2]-np.asarray(b['xyz'])[:2])/(2*step_px))
        return np.column_stack(columns)

    def to_dict(self) -> dict[str, Any]:
        return {'center_xyz':self.center_xyz.tolist(), 'rvec_world_to_camera':self.rvec_world_to_camera.tolist(),
                'R_world_to_camera':self.R.tolist(),'t_world_to_camera':self.tvec.tolist(),
                'K_px':self.K_px.tolist(),'distortion_opencv':self.distortion_opencv.tolist(),
                'convention':'X_camera=R_world_to_camera @ (X_world-center_xyz); OpenCV camera axes'}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Camera':
        return cls(d['center_xyz'],d['rvec_world_to_camera'],d['K_px'],d.get('distortion_opencv',[]))


def look_at_rvec(center_xyz: Any, target_xyz: Any, up_world: Any = (0,0,1)) -> np.ndarray:
    """Initial view only; caller supplies center/target hypotheses explicitly."""
    c=_array(center_xyz,(3,),'center');target=_array(target_xyz,(3,),'target')
    z=target-c
    if np.linalg.norm(z)<1e-12: raise ValueError('center and target coincide')
    z/=np.linalg.norm(z)
    x=np.cross(z,_array(up_world,(3,),'up'))
    if np.linalg.norm(x)<1e-10: raise ValueError('look direction parallel to chosen up vector')
    x/=np.linalg.norm(x);y=np.cross(z,x)
    return cv2.Rodrigues(np.vstack([x,y,z]))[0].reshape(3)


class Scene:
    def __init__(self, path: str | Path):
        self.path=Path(path)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.data=json.loads(self.path.read_text(encoding='utf-8-sig'))
        schema=self.data.get('schema_version')
        if schema not in ('rinkspace.scene3d.v1', 'rinkspace.three_frame_refit.scene.v1'):
            raise ValueError('Use the enriched scene.json, not geometry-only scene_geometry.json')
        # A refit scene keeps its source file immutable; the selected release may
        # assign a revision separately through load_active_scene().
        self.revision=self.data.get('revision')
        self.features={f['id']:f for f in self.data['ice_features']}
        if len(self.features)!=len(self.data['ice_features']): raise ValueError('duplicate feature IDs')
        self.boundary=np.asarray(self.data['ice_boundary']['points_xyz'],float)
        self.panels=self.data['board_surface']['panels']
        self.board_vertices=np.asarray(next(m for m in self.data['meshes'] if m['name']=='rink_boards')['positions'],float)
        self.perimeter=float(self.data['board_surface']['perimeter_length'])
        if schema=='rinkspace.scene3d.v1':
            self.ads={a['id']:a for a in self.data['advertisements']}
        else:
            # Candidate advertisement boundaries in the refit have a user number,
            # not the old instance ID or a verified 3D attachment/anchor. Keep
            # those identities explicit without manufacturing physical geometry.
            self.ads={f"user_number:{a['user_number']}":a
                      for a in self.data['advertisements']}
            if len(self.ads)!=len(self.data['advertisements']):
                raise ValueError('duplicate refit advertisement user number')

    def board_point(self, s: float, height: float | None) -> np.ndarray:
        """Point on existing faceted inner surface. s is periodic scene arc length."""
        if height is None or not np.isfinite(height):
            raise ValueError('unknown board/ad height is not zero and cannot be projected')
        if not np.isfinite(s): raise ValueError('nonfinite s')
        s=float(s)%self.perimeter
        for p in self.panels:
            s0,s1=p['s_range']
            if s0 <= s <= s1:
                a,b=self.board_vertices[p['bottom_vertex_indices']]
                xyz=a+(s-s0)/(s1-s0)*(b-a)
                xyz[2]=float(height)
                return xyz
        raise ValueError('s not covered by board panels')

    def point(self, target: dict[str, Any]) -> np.ndarray:
        kind=target['kind']
        if kind=='circle_center':
            return np.asarray(self.features[target['feature_id']]['center_xyz'],float)
        if kind=='polygon_vertex':
            return np.asarray(self.features[target['feature_id']]['polygon_xyz'][int(target['index'])],float)
        if kind=='line_endpoint':
            return np.asarray(self.features[target['feature_id']]['segments_xyz'][int(target.get('segment',0))][int(target['index'])],float)
        if kind=='board_point':
            return self.board_point(float(target['s']),target.get('height'))
        if kind=='advert_anchor':
            advert_id=target['advert_id']
            if advert_id not in self.ads:
                raise ValueError(f'unknown advert identity: {advert_id}')
            a=self.ads[advert_id]
            attachment_s=a.get('attachment_s')
            if attachment_s is None:
                raise ValueError('unknown advert attachment_s cannot be projected')
            anchor=a.get('anchor_xyz')
            if not isinstance(anchor,(list,tuple)) or len(anchor)!=3:
                raise ValueError('unknown advert anchor_xyz cannot be projected')
            return self.board_point(float(attachment_s),anchor[2])
        if kind=='explicit_point':
            if not target.get('provenance'): raise ValueError('explicit point needs provenance')
            return _array(target['xyz'],(3,),'explicit xyz')
        raise ValueError(f'unsupported point target: {kind}')

    def curve(self, target: dict[str, Any], samples: int = 192) -> np.ndarray:
        """Sample one connected semantic curve. No generic unlabeled nearest mark.

        samples controls numerical approximation, not observed evidence count.
        For real use, refine until projected chord error is below observation noise.
        """
        if samples < 8: raise ValueError('curve approximation too coarse: samples must be >=8')
        kind=target['kind']
        if kind=='circle_outline':
            f=self.features[target['feature_id']]
            circle=f
            if 'ring_role' in target:
                circle=next(r for r in f.get('rings',[]) if r['role']==target['ring_role'])
            theta=np.linspace(0,2*np.pi,samples+1)
            c=np.asarray(circle['center_xyz'],float)
            return c+float(circle['radius'])*np.column_stack([np.cos(theta),np.sin(theta),np.zeros_like(theta)])
        if kind=='polygon_edge':
            feature=self.features[target['feature_id']]
            p=np.asarray(feature['polygon_xyz'],float)
            i=int(target['index'])
            if i in feature.get('open_edges',[]):
                raise ValueError('requested polygon edge is an explicitly open side')
            return np.linspace(p[i],p[(i+1)%len(p)],samples)
        if kind=='line_segment':
            seg=self.features[target['feature_id']]['segments_xyz'][int(target.get('segment',0))]
            return np.linspace(seg[0],seg[1],samples)
        if kind=='board_curve':
            s0,s1=map(float,target['s_range'])
            # s1>s0 may go through the perimeter seam. Do not take the wrong short arc.
            if s1 <= s0: raise ValueError('board_curve requires unwrapped increasing s_range')
            return np.asarray([self.board_point(s,target.get('height')) for s in np.linspace(s0,s1,samples)])
        raise ValueError(f'unsupported curve target: {kind}')


def nearest_polyline_vectors(points_uv: Any, polyline_uv: Any) -> np.ndarray:
    """Observed point minus closest point on each projected semantic polyline."""
    q=np.asarray(points_uv,float).reshape(-1,2);p=np.asarray(polyline_uv,float).reshape(-1,2)
    if len(p)<2 or not np.isfinite(q).all() or not np.isfinite(p).all(): raise ValueError('invalid polyline')
    a=p[:-1];d=p[1:]-a
    den=np.sum(d*d,axis=1)
    t=np.einsum('nsi,si->ns',q[:,None,:]-a[None,:,:],d)/np.maximum(den,1e-20)
    closest=a[None,:,:]+np.clip(t,0,1)[:,:,None]*d[None,:,:]
    delta=q[:,None,:]-closest
    idx=np.argmin(np.sum(delta*delta,axis=2),axis=1)
    return delta[np.arange(len(q)),idx]
