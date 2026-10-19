"""Native-pixel camera projection and source/model coordinate bridge.

Camera vectors use the released scene convention (25 values, Brown14).
The scene scale is conditional; this module never calls these units metres.
"""
from __future__ import annotations

from pathlib import Path
import sys

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:  # Scripts in this directory also import projection directly.
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()

import cv2
import numpy as np
try:
    from .projection_reuse import reused_projection
except ImportError:
    from projection_reuse import reused_projection

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT/'src', ROOT/'calibration/work', ROOT/'calibration/shot_tracking/pipeline'):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))
from physical_projection import safe_project
from camera import world_rays

# Opt-in kernels must preserve safe_project's sampled optical-domain rule.
# The default stays the reviewed general Brown14 implementation.
_central_projection_backend = safe_project


def vector_from_scene(record):
    c = record['camera']
    k = np.asarray(c['K_px'],float)
    d = np.zeros(14)
    incoming = np.asarray(c['distortion_opencv'],float)
    d[:len(incoming)] = incoming
    return np.r_[c['center_xyz'],c['rvec_world_to_camera'],
                 np.log(k[0,0]),np.log(k[1,1]),k[0,2],k[1,2],k[0,1],d]


def to_state(camera):
    c = np.asarray(camera,float)
    return np.r_[c[:3]/100.,c[3:8],c[8:11]/1000.]


def from_state(state, template):
    c = np.asarray(template,float).copy()
    q = np.asarray(state,float)
    if q.shape not in ((11,), (15,)):
        raise ValueError('Camera state must have 11 or 15 entries')
    c[:3],c[3:8],c[8:11] = q[:3]*100.,q[3:8],q[8:11]*1000.
    if len(q)==15:
        # The extended physical state owns the shared Brown4. Other Brown
        # terms are explicit fixed zeros, never inherited per-shot templates.
        c[11:25] = 0.
        c[11:15] = q[11:15]
    return c


def project_fast(camera, xyz, with_jacobian=False):
    c = np.asarray(camera,float)
    xyz = np.ascontiguousarray(np.asarray(xyz,float).reshape(-1,3))
    if not with_jacobian:
        reused = reused_projection(c, xyz)
        if reused is not None:
            return reused[:2]
    if len(xyz) == 0:
        if with_jacobian:
            return np.empty((0,2)), np.empty(0), np.empty((0,11))
        return np.empty((0,2)), np.empty(0)
    r, drot = cv2.Rodrigues(c[3:6])
    k = np.array([[np.exp(c[6]),0,c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
    t = -r @ c[:3]
    projected, jac = cv2.projectPoints(xyz,c[3:6],t,k,c[11:25])
    uv = projected.reshape(-1,2)
    yf = (uv[:,1]-c[9])/k[1,1]
    uv[:,0] += c[10]*yf
    depth = (xyz-c[:3]) @ r[2]
    if not with_jacobian:
        return uv, depth
    # OpenCV gives derivatives w.r.t. Rodrigues, translation and K. Convert
    # to world camera centre and the conditioned 11-component vector above.
    j = np.zeros((2*len(xyz),11),float)
    j[:,:3] = jac[:,3:6] @ (-r*100.)
    dr = drot.reshape(3,3,3)
    dt = np.column_stack([-dr[i] @ c[:3] for i in range(3)])
    j[:,3:6] = jac[:,:3]+jac[:,3:6]@dt
    j[:,6] = jac[:,6]*k[0,0]
    j[:,7] = jac[:,7]*k[1,1]
    j[:,8] = jac[:,8]*1000.
    j[:,9] = jac[:,9]*1000.
    # Raw x includes skew * normalized distorted y, separate from OpenCV K.
    yj = j[1::2].copy()
    yj[:,9] -= 1000.
    yj /= k[1,1]
    yj[:,7] -= yf
    j[0::2] += c[10]*yj
    j[0::2,10] = yf*1000.
    return uv,depth,j


def project_central_valid(camera, xyz, *, with_jacobian=False):
    """Return the original projection/Jacobian and its central Brown domain.

    Validity is a property of the world ray, including off-screen rays.  The
    returned UV and Jacobian deliberately come from ``project_fast`` so that
    existing factor values are unchanged whenever the ray is valid.
    """
    c = np.asarray(camera, float).reshape(-1)
    points = np.ascontiguousarray(np.asarray(xyz, float).reshape(-1, 3))
    if c.shape != (25,) or not np.isfinite(c).all():
        raise ValueError('Camera must have 25 finite values')
    if np.any(np.abs(c[23:25]) > 1e-12):
        raise ValueError('Brown sensor tilt must stay fixed to zero')
    if not np.isfinite(points).all():
        raise ValueError('World points must be finite')
    if not with_jacobian:
        reused = reused_projection(c, points)
        if reused is not None:
            return reused
    try:
        projected = project_fast(c, points, with_jacobian=with_jacobian)
    except cv2.error:
        uv = np.full((len(points), 2), np.nan)
        depth = np.full(len(points), np.nan)
        if with_jacobian:
            return uv, depth, np.zeros((2 * len(points), 11)), np.zeros(len(points), bool)
        return uv, depth, np.zeros(len(points), bool)
    uv, depth = projected[:2]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > .05)
    if np.any(np.abs(c[11:23]) > 1e-14) and valid.any():
        # The no-Brown path avoids the nine radial Jacobian stations for the
        # common pinhole case.  Only rays that passed the cheap checks enter
        # the expensive optical-domain check.
        indices = np.flatnonzero(valid)
        try:
            central_uv, _, central = _central_projection_backend(c, points[indices])
            central &= np.isfinite(central_uv).all(axis=1)
            central &= np.linalg.norm(central_uv - uv[indices], axis=1) <= 1e-5
        except (cv2.error, FloatingPointError, OverflowError):
            central = np.zeros(len(indices), bool)
        valid[indices] = central
    if with_jacobian:
        valid &= np.isfinite(projected[2].reshape(-1, 2, 11)).all(axis=(1, 2))
        return uv, depth, projected[2], valid
    return uv, depth, valid


def require_central_projection(camera, xyz, *, stage, family, source_keys,
                               with_jacobian=False):
    """Project physical points or reject a trial with a typed domain error."""
    result = project_central_valid(camera, xyz, with_jacobian=with_jacobian)
    valid = result[-1]
    if not np.all(valid):
        # Lazy import avoids a cycle: reference_board_pair imports projection.
        try:
            from .reference_board_pair import BoardIntersectionError
        except ImportError:
            from reference_board_pair import BoardIntersectionError
        error = BoardIntersectionError('noncentral_projection', np.flatnonzero(~valid))
        error.stage = stage
        error.family = family
        error.evaluation_stage = stage
        error.factor_family = family
        error.source_keys = source_keys
        raise error
    return result[:-1]


def ground_xy(camera, pixels, scene=None):
    rays, valid = world_rays(camera,np.asarray(pixels,float))
    lam = -camera[2]/np.where(np.abs(rays[:,2])>1e-12,rays[:,2],np.nan)
    xyz = camera[:3]+lam[:,None]*rays
    valid &= np.isfinite(xyz).all(axis=1)&(lam>0)
    if scene is not None:
        polygon=np.ascontiguousarray(np.asarray(scene['ice_boundary']['points_xyz'],np.float32)[:,:2])
        for i in np.flatnonzero(valid):
            valid[i] = cv2.pointPolygonTest(polygon,tuple(xyz[i,:2]),False)>=0
    xyz[~valid] = np.nan
    return xyz[:,:2],valid


def attach_visible_surfaces(scene,camera,pixels,*,inverse_tolerance_px=None):
    """First hit on the ice or board wall; nothing above the board is ice."""
    if inverse_tolerance_px is None:
        rays, ray_good=world_rays(camera,np.asarray(pixels,float))
    else:
        # Mapping consumers may require a tighter inverse than the solver's
        # historical .005 px tolerance. Keep that solver path unchanged.
        from types import SimpleNamespace
        from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse
        if not np.isfinite(inverse_tolerance_px) or inverse_tolerance_px <= 0:
            raise ValueError('Inverse tolerance must be finite and positive')
        v = np.asarray(camera, float)
        if np.any(np.abs(v[23:25]) > 1e-12):
            raise ValueError('Brown tilt tauX/tauY must be fixed to zero')
        p = np.asarray(pixels, float).copy()
        fx, fy = np.exp(v[6:8])
        p[:, 0] -= v[10]*(p[:, 1]-v[9])/fy
        k = np.array([[fx, 0., v[8]], [0., fy, v[9]], [0., 0., 1.]])
        inv = brown_central_inverse(
            SimpleNamespace(K_px=k, distortion_opencv=v[11:23]), p,
            tolerance_px=inverse_tolerance_px)
        rays = np.c_[inv['normalized_xy'], np.ones(len(p))] @ cv2.Rodrigues(v[3:6])[0]
        ray_good = np.asarray(inv['valid'], bool)
    centre=np.asarray(camera[:3],float)
    border=np.asarray(scene['ice_boundary']['points_xyz'],float)
    polygon=border[:,:2].astype(np.float32)
    starts=border[:,:2]
    edges=np.roll(starts,-1,axis=0)-starts
    rel=starts-centre[:2]
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)[:,2].max()
    result=np.full((len(pixels),3),np.nan)
    types=np.full(len(pixels),'unavailable',dtype='<U16')
    for index in np.flatnonzero(ray_good):
        direction=rays[index]
        ground_t=-centre[2]/direction[2] if direction[2]<-1e-10 else np.inf
        hit=centre+ground_t*direction if np.isfinite(ground_t) else np.full(3,np.nan)
        if np.isfinite(ground_t) and cv2.pointPolygonTest(polygon,tuple(hit[:2]),False)<0:
            ground_t=np.inf
        den=direction[0]*edges[:,1]-direction[1]*edges[:,0]
        ok=np.abs(den)>1e-10
        distance=np.full(len(edges),np.inf)
        along=np.full(len(edges),np.inf)
        distance[ok]=(rel[ok,0]*edges[ok,1]-rel[ok,1]*edges[ok,0])/den[ok]
        along[ok]=(rel[ok,0]*direction[1]-rel[ok,1]*direction[0])/den[ok]
        heights=centre[2]+distance*direction[2]
        ok &= (distance>0)&(along>=0)&(along<=1)&(heights>=0)&(heights<=top)
        wall_t=np.min(np.where(ok,distance,np.inf))
        if wall_t<ground_t:
            result[index]=centre+wall_t*direction
            types[index]='board_surface'
        elif np.isfinite(ground_t) and ground_t>0:
            result[index]=hit
            types[index]='ice_marking'
    return result,types,np.isfinite(result).all(axis=1)


def scene_paths(scene, grid=True):
    """Return typed 3D polylines without connecting separate physical paths."""
    paths=[]
    for kind, points in (('contact', scene['ice_boundary']['points_xyz']),
                         ('board_top', scene['board_surface']['top_edge_xyz'])):
        ring = np.asarray(points, float)
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack((ring, ring[0]))
        paths.append((kind, ring))
    for f in scene['ice_features']:
        if f['kind']=='circle':
            rings=f.get('rings',[dict(center_xyz=f.get('center_xyz'),radius=f.get('radius'),role='outer')])
            for ring in rings:
                if ring['role']!='outer':
                    continue
                t=np.linspace(0,2*np.pi,181)
                centre=np.asarray(ring['center_xyz'],float)
                paths.append(('marking',centre+np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]*float(ring['radius'])))
        elif f['kind']=='polyline':
            for p in f['segments_xyz']:
                p=np.asarray(p,float)
                paths.append(('marking',np.vstack([np.linspace(a,b,32) for a,b in zip(p[:-1],p[1:])]) if len(p)>1 else p))
        elif f['kind']=='polygon':
            p=np.asarray(f['polygon_xyz'],float)
            paths.append(('marking',np.vstack([p,p[0]])))
    if grid:
        edges={}
        for cell in scene['ice_grid']['cells']:
            p=np.asarray(cell['polygon_xyz'],float)
            for a,b in zip(p,np.roll(p,-1,axis=0)):
                if np.linalg.norm(a-b)<.1:
                    continue
                key=tuple(sorted((tuple(np.round(a,5)),tuple(np.round(b,5)))))
                edges[key]=('grid',np.linspace(a,b,9))
        paths.extend(edges.values())
    return paths
