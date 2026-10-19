"""Current-wall geometry derivatives for verified static board pixel pairs.

The source intersection is eliminated, not promoted to a measured 3-D point.
Only the XY endpoints of the actual first-hit vertical wall panel affect an
interior hit. Bottom/top heights remain part of the unchanged visibility domain.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from .board_motion_factor import evaluate_board_motion
from .ice_motion_factor import _camera_inventory


def _boundary_chain(value, vertex_count):
    if value is None:
        return None
    if issparse(value):
        chain = csr_matrix(value, dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 3 and array.shape[:2] == (vertex_count, 2):
            array = array.reshape(2*vertex_count, array.shape[2])
        if array.ndim != 2:
            raise ValueError('Boundary XY Jacobian needs (V,2,K) or (2V,K) shape')
        chain = csr_matrix(array)
    if chain.shape[0] != 2*vertex_count or not np.isfinite(chain.data).all():
        raise ValueError('Finite boundary XY Jacobian with exactly 2V rows required')
    return chain


def evaluate_board_motion_geometry(surface, camera_a, camera_b, source_uv, target_uv,
                                   *, boundary_xy_jacobian=None, jacobian=True,
                                   inverse_tolerance_px=1e-9):
    """Return r, Ja15, Jb15, Jgeometry, XYZ on the existing current wall.

    All source pairs must already have explicit original-pixel wall qualification.
    This function does not select observations or weaken the first-hit, target
    occlusion, inverse-camera or optical guards in evaluate_board_motion.

    boundary_xy_jacobian is d(bottom_vertices[:, :2])/d(selected_geometry),
    shaped (V,2,K), dense (2V,K), or sparse (2V,K), in actual world units.
    The matching top vertices share these XY coordinates. If omitted, Jgeometry
    is directly with respect to interleaved vertex XY, with 2V columns. The caller
    combines it with both conditioned camera15 mappings and then pulls the sum
    through the exact current physical-to-canonical chain. Camera and surface
    dependencies must both be included, even if their parameter columns overlap.

    This is the derivative of the currently selected first-hit panel. A wall seam,
    visibility boundary or grazing ray is piecewise smooth; finite differences
    must stay on the same valid panel and the original domain checks still apply.
    No fixed XYZ, unknown depth, extra pose anchor, or geometry prior is added.
    """
    vertex_count = len(surface.bottom)
    chain = _boundary_chain(boundary_xy_jacobian, vertex_count)
    r, ja, jb, worlds = evaluate_board_motion(
        surface, camera_a, camera_b, source_uv, target_uv,
        jacobian=jacobian, inverse_tolerance_px=inverse_tolerance_px)
    if not jacobian:
        return r, ja, jb, csr_matrix((len(r), 0)), worlds
    cameras = _camera_inventory(camera_a, len(worlds))
    rows, columns, values = [], [], []
    for i, (camera, world) in enumerate(zip(cameras, worlds, strict=True)):
        # Any nonzero ray scaling is valid. Reusing the already authenticated
        # hit avoids a second inverse Brown camera or another projection.
        ray = world-camera[:3]
        hit = surface._hits_from_rays(camera[:3], ray[None, :])
        surface._require(hit)
        panel = int(hit.panel[0])
        fraction = float(hit.panel_fraction[0])
        edge = surface._edges[panel]
        denominator = ray[0]*edge[1]-ray[1]*edge[0]
        if not np.isfinite(denominator) or denominator == 0.:
            raise ValueError('Current board geometry derivative has a grazing source ray')
        # C+t*d=A+s*(B-A): dt=cross((1-s)dA+s*dB,E)/cross(d,E).
        dt = np.array([edge[1], -edge[0]])/denominator
        # Public conditioned camera-centre columns are C/100, hence dPi/dX.
        world_projection = -jb[2*i:2*i+2, :3]/100.
        pixel_direction = world_projection@ray
        for vertex, weight in ((panel, 1.-fraction),
                               ((panel+1) % vertex_count, fraction)):
            block = np.outer(pixel_direction, weight*dt)
            for coordinate in range(2):
                rows.extend((2*i, 2*i+1))
                columns.extend((2*vertex+coordinate, 2*vertex+coordinate))
                values.extend(block[:, coordinate])
    local = coo_matrix((values, (rows, columns)),
                       shape=(len(r), 2*vertex_count)).tocsr()
    local.eliminate_zeros()
    geometry = local if chain is None else (local@chain).tocsr()
    geometry.eliminate_zeros()
    return r, ja, jb, geometry, worlds
