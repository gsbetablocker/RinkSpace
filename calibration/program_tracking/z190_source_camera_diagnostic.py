"""Small, unaccepted Z190 installation fit on immutable real source keys.

The hard scene, source UV, typed world identities and pre-existing splits stay
fixed. This does not edit any production graph/camera packet or render a video.
Held poses receive same-cut source-key corrections; that interpolation is an
explicit limitation, not a dense source-motion or accuracy acceptance.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

PROGRAM = Path(__file__).resolve().parent
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve
from main7_source_camera_candidate import B, arrays, errors, sha, value_only
from main7_source_bundle_candidate import solve
from projection import project_fast, project_central_valid
from ice_refine import _curve_vector_error, _world_curves, nearest_segments
from reference_board_pair import BoardIntersectionError

HARD_SCENE_SHA = '93c654360712be798e65aa83e587415fa2693008ee320ccbc1972299ead293bd'
RESEARCH_ROOT = Path('D:/CodexData/.codex/visualizations/2026/10/01/private-thread-00011/z190_research')
SHARED_COUNT = 14


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def source_group(raw):
    group = copy.deepcopy(raw)
    for row in group['rows']:
        for name in ('world', 'uv', 'sigma', 'normal', 'normal_interval'):
            if name in row:
                row[name] = np.array(row[name], dtype=float)
        if 'source_uv_float64_sha256' in row:
            import hashlib
            actual = hashlib.sha256(row['uv'].astype(np.float64).tobytes()).hexdigest()
            if actual != row['source_uv_float64_sha256']:
                raise ValueError(f"Source pixels changed at {group['pts']}/{row['id']}")
        row['observation_surface_type'] = (
            'board_surface' if row.get('point_correspondence') or 'top' in row['id']
            else 'ice_board_contact' if row['id'] in ('far_contact', 'ice_board_contact')
            else 'ice_marking')
    return group


def select_keys(groups, installation, per_bin, held_pts):
    """Deterministic zoom/time coverage; never select by projection error."""
    pool = [g for g in groups if g['installation'] == installation
            and int(g['pts']) not in held_pts
            and any(r['split'] == 'fit' for r in g['rows'])]
    pool.sort(key=lambda g: float(np.mean(g['camera25_seed'][6:8])))
    chosen = {}
    bins = np.array_split(np.arange(len(pool)), 3)
    for name, indices in zip(('wide_relative', 'mid_relative', 'tele_relative'), bins):
        if not len(indices):
            continue
        take = indices[np.unique(np.linspace(0, len(indices)-1, min(per_bin, len(indices))).astype(int))]
        for index in take:
            key = source_group(pool[int(index)])
            key['zoom_sampling_bin'] = name
            chosen[key['pts']] = key
    # Preserve each original cut's temporal range for interpolation of untouched
    # checks; no continuity or source pair is invented across cuts.
    for cut in sorted({int(g['cut']) for g in pool}):
        cutkeys = sorted((g for g in pool if int(g['cut']) == cut), key=lambda g: int(g['pts']))
        for raw in (cutkeys[0], cutkeys[-1]):
            if raw['pts'] not in chosen:
                key = source_group(raw)
                key['zoom_sampling_bin'] = 'same_cut_endpoint_support'
                chosen[key['pts']] = key
    for key in chosen.values():
        key['rows'] = [r for r in key['rows'] if r['split'] == 'fit']
    return [chosen[p] for p in sorted(chosen)]


def brown4_world_jac(camera, world):
    """OpenCV derivatives of the four shared residual Brown coefficients."""
    rotation = cv2.Rodrigues(camera[3:6])[0]
    focal = np.exp(camera[6:8])
    intrinsic = np.array([[focal[0], 0., camera[8]], [0., focal[1], camera[9]], [0., 0., 1.]])
    _, raw = cv2.projectPoints(np.ascontiguousarray(world), camera[3:6],
                              -rotation @ camera[:3], intrinsic, camera[11:25])
    jac = raw[:, 10:14].copy()
    jac[0::2] += camera[10] / focal[1] * jac[1::2]
    return jac.reshape(len(world), 2, 4)


def horizontal_extent_value_jac15(camera, row, jacobian=True):
    """Observe a complete projected curve's left/right extent, not a near arc.

    The source y coordinate only identifies the visible source search area.
    An x-extremum is an image support value of the whole named physical ring;
    choosing the nearest lower/upper arc would hide a wrong centre/diameter.
    """
    world, uv = row['world'], row['uv']
    if row['extent_side'] not in ('left','right') or len(world)<4:
        raise ValueError('Expected a complete source-identified curve extent')
    if jacobian:
        projected,depth,j11,central=project_central_valid(camera,world,with_jacobian=True)
    else:
        projected,depth,central=project_central_valid(camera,world)
    if not np.all(central & (depth>0.) & np.isfinite(projected).all(axis=1)):
        raise BoardIntersectionError('complete_extent_curve_not_central_forward',np.flatnonzero(~central | (depth<=0.)))
    choose=np.argmax if row['extent_side']=='right' else np.argmin
    index=int(choose(projected[:,0]))
    residual=np.zeros((len(uv),2))
    residual[:,0]=projected[index,0]-uv[:,0]
    if not jacobian:return residual.ravel()
    block=np.zeros((2*len(uv),15))
    extra=brown4_world_jac(camera,world[index:index+1]).reshape(2,4)
    block[0::2]=np.r_[j11[2*index],extra[0]]
    return residual.ravel(),block


def clipped_extent_value_jac15(camera,row,jacobian=True):
    """One-sided full-ring visibility support, without inventing a tangent UV."""
    if row.get('point_correspondence') or not row.get('source_uv_is_image_boundary_context_only'):
        raise ValueError('Clipped support is an image-boundary inequality, not an observed tangent point')
    side=row['extent_side'];allowance=float(row['clip_blur_allowance_px'])
    if side not in ('left','right')or not np.isfinite(allowance)or allowance<0.:
        raise ValueError('Explicit side and source-blur engineering allowance required')
    limit=float(row['clip_boundary_x'])+(allowance if side=='left'else-allowance)
    proxy=dict(row,uv=np.tile([limit,0.],(len(row['uv']),1)))
    if jacobian:value,block=horizontal_extent_value_jac15(camera,proxy,True)
    else:value=horizontal_extent_value_jac15(camera,proxy,False)
    active=value[0::2]>0. if side=='left'else value[0::2]<0.
    value[0::2]*=active
    if jacobian:
        block[0::2]*=active[:,None]
        return value,block
    return value


def scanline_branch_value_jac15(camera,row,jacobian=True):
    """Preserve a source-identified upper/lower branch at a fixed source x."""
    world,uv=row['world'],row['uv']
    if row.get('scanline_fixed_axis')!='x' or row.get('scanline_branch') not in ('min_y','max_y'):
        raise ValueError('Expected explicitly source-identified vertical scanline branch')
    if row.get('must_preserve_upper_branch') and row['scanline_branch']!='min_y':
        raise ValueError('An explicitly upper source branch cannot select the lower branch')
    if jacobian:
        projected,depth,j11,valid=project_central_valid(camera,world,with_jacobian=True)
        allj=np.c_[j11,brown4_world_jac(camera,world).reshape(-1,4)].reshape(-1,2,15)
    else:projected,depth,valid=project_central_valid(camera,world)
    if not np.all(valid & (depth>0.) & np.isfinite(projected).all(axis=1)):
        raise BoardIntersectionError('complete_scanline_curve_not_central_forward',
            np.flatnonzero(~valid | (depth<=0.) | ~np.isfinite(projected).all(axis=1)))
    start=projected[:-1];delta=np.diff(projected,axis=0)
    admissible=valid[:-1]&valid[1:]&(depth[:-1]>0.)&(depth[1:]>0.)&(np.abs(delta[:,0])>1e-12)
    residual=np.zeros((len(uv),2));block=np.zeros((2*len(uv),15))if jacobian else None
    for k,point in enumerate(uv):
        crossing=admissible&(np.minimum(start[:,0],projected[1:,0])<=point[0])&(np.maximum(start[:,0],projected[1:,0])>=point[0])
        ids=np.flatnonzero(crossing)
        if len(ids):
            tt=(point[0]-start[ids,0])/delta[ids,0]
            yy=start[ids,1]+tt*delta[ids,1]
            selected=int(np.argmin(yy)if row['scanline_branch']=='min_y'else np.argmax(yy))
            i=int(ids[selected]);t=float(tt[selected]);residual[k,1]=yy[selected]-point[1]
            if jacobian:
                ja=allj[i];jd=allj[i+1]-ja
                dt=(-ja[0]-t*jd[0])/delta[i,0]
                block[2*k+1]=ja[1]+t*jd[1]+delta[i,1]*dt
        else:
            # A source sample outside the complete projected ring must remain
            # visible as a lateral/endpoint error, never swap to another arc.
            available=np.flatnonzero(valid&(depth>0.)&np.isfinite(projected).all(axis=1))
            if not len(available):raise BoardIntersectionError('scanline_curve_not_central_forward',[])
            i=int(available[np.argmin(np.abs(projected[available,0]-point[0]))])
            residual[k]=projected[i]-point
            if jacobian:block[2*k:2*k+2]=allj[i]
    return (residual.ravel(),block)if jacobian else residual.ravel()


def absolute_value_jac15(camera, rows, jacobian=True):
    """Existing finite source objective, with Brown4 columns appended."""
    values, blocks = [], []
    for row in rows:
        world, uv = row['world'], row['uv']
        if row.get('curve_observation_kind')=='clipped_horizontal_extent':
            if jacobian:residual,block=clipped_extent_value_jac15(camera,row,True)
            else:residual=clipped_extent_value_jac15(camera,row,False)
        elif row.get('curve_observation_kind')=='horizontal_extent':
            if jacobian:
                residual,block=horizontal_extent_value_jac15(camera,row,True)
            else:residual=horizontal_extent_value_jac15(camera,row,False)
        elif row.get('must_preserve_upper_branch') or row.get('curve_observation_kind')=='axis_scanline_branch':
            if jacobian:residual,block=scanline_branch_value_jac15(camera,row,True)
            else:residual=scanline_branch_value_jac15(camera,row,False)
        elif row.get('point_correspondence'):
            if jacobian:
                projected, depth, j11, central = project_central_valid(camera, world, with_jacobian=True)
                extra = brown4_world_jac(camera, world).reshape(-1, 4)
                block = np.c_[j11, extra]
            else:
                projected, depth, central = project_central_valid(camera, world)
            if not np.all(central & (depth > .05)):
                raise BoardIntersectionError('z190_actual_ad_corner_not_central_forward', np.flatnonzero(~central | (depth <= .05)))
            residual = (projected-uv).ravel()
            if 'point_support_bounds_xy_px' in row:
                bounds=np.asarray(row['point_support_bounds_xy_px'],float)
                if bounds.shape!=(len(uv),2,2)or not np.isfinite(bounds).all()or np.any(bounds[:,:,0]>bounds[:,:,1]):
                    raise ValueError('Each observed corner requires finite ordered native XY support intervals')
                low,high=bounds[:,:,0],bounds[:,:,1]
                residual=(np.minimum(projected-low,0.)+np.maximum(projected-high,0.)).ravel()
                if jacobian:
                    block*=((projected<low)|(projected>high)).ravel()[:,None]
        else:
            if not jacobian:
                residual = _curve_vector_error(camera, world, uv)
            else:
                residual, j11 = _curve_vector_error(camera, world, uv, True)
                projected, depth, valid = project_central_valid(camera, world)
                start, direction = projected[:-1], np.diff(projected, axis=0)
                length2 = np.sum(direction*direction, axis=1)
                midpoint = project_central_valid(camera, .5*(world[:-1]+world[1:]))[-1]
                segments = (valid[:-1] & valid[1:] & midpoint & (depth[:-1] > 0.) & (depth[1:] > 0.)
                            & np.isfinite(start).all(axis=1) & np.isfinite(projected[1:]).all(axis=1)
                            & (length2 > 1e-4))
                index, t, clipped, nearest = nearest_segments(uv, start, direction, length2, segments)
                if not np.allclose(residual, (nearest-uv).ravel(), rtol=0., atol=1e-8):
                    raise ValueError('Brown derivative changed the existing finite-curve objective')
                raw = brown4_world_jac(camera, world)
                ja, jd = raw[index], raw[index+1]-raw[index]
                d, u = direction[index], uv-start[index]
                numerator = -np.einsum('nij,ni->nj', ja, d)+np.einsum('ni,nij->nj', u, jd)
                denominator = 2.*np.einsum('ni,nij->nj', d, jd)
                dt = (numerator-t[:, None]*denominator)/length2[index, None]
                dt[(t <= 0.) | (t >= 1.)] = 0.
                extra = (ja+clipped[:, None, None]*jd+d[:, :, None]*dt[:, None, :]).reshape(-1, 4)
                block = np.c_[j11, extra]
        if 'normal_interval' in row:
            from source_edge_intervals import SourceNormalIntervals
            support = SourceNormalIntervals(row['normal'], row['normal_interval'], np.ones(len(uv), bool))
            if jacobian:
                residual, matrices = support.apply(residual, [block])
                block = matrices[0]
            else:
                residual = support.apply(residual)
        scale = np.repeat(row['sigma'], 2)
        values.append(residual/scale)
        if jacobian:
            blocks.append(block/scale[:, None])
    if not jacobian:
        return np.concatenate(values)
    return np.concatenate(values), np.vstack(blocks)


def physical_camera(shared, view, reference_logf):
    """Adapter to the independent Z190 implementation; see its parameter units."""
    calibration, state = physical_calibration(shared, view, reference_logf)
    return calibration.camera25(state)


def physical_calibration(shared, view, reference_logf):
    import z190_physical_camera as api
    template = api.SharedCalibration('independent_z190_source_key_diagnostic', (0., 0., 1.),
                                    q_reference=reference_logf-np.log(1920.), axial=(0., 0., 0.))
    calibration = template.with_shared_state(np.asarray(shared, float))
    state = api.ViewState(tuple(view[:3]), float(view[3]-np.log(1920.)))
    return calibration, state


def physical_state_jac(shared, view, reference_logf):
    calibration, state = physical_calibration(shared, view, reference_logf)
    _, shared_jac, view_jac = calibration.camera25_and_jacobians(state)
    mapping = np.c_[shared_jac[:15], view_jac[:15]]
    mapping[:3] /= 100.
    mapping[8:11] /= 1000.
    return mapping


class SourceKeys:
    def __init__(self, keys, physical=False, reference_logf=None, physical_shared_columns=None, crew_hypothesis=None, crew_fixed_height=False):
        self.keys, self.physical = keys, physical
        self.stamps = [int(k['pts']) for k in keys]
        self.index = {p: i for i, p in enumerate(self.stamps)}
        self.physical_shared_columns = np.array(list(range(SHARED_COUNT)) if physical_shared_columns is None else physical_shared_columns, int)
        self.view_offset = len(self.physical_shared_columns) if physical else 3
        self.reference_logf = float(reference_logf if reference_logf is not None else
                                   np.median([np.mean(k['camera25_seed'][6:8]) for k in keys]))
        self.native_residual_sigma = np.concatenate([np.repeat(r['sigma'], 2) for k in keys for r in k['rows']])
        self.residual_rows = len(self.native_residual_sigma)
        self.source_residual_rows = self.residual_rows
        self.crew_hypothesis = crew_hypothesis
        self.crew_fixed_height = bool(crew_fixed_height)
        self.prior_wide_focal_px = min(float(np.exp(np.mean(k['camera25_seed'][6:8]))) for k in keys)
        if crew_hypothesis is not None and (not physical or list(self.physical_shared_columns) != [0, 1, 2, 8, 9, 10]):
            raise ValueError('Standing-crew solve uses fixed C, shared PP/k1, zero k2/lever/axial')

    def camera(self, q, p):
        i = self.index[p]
        view = q[self.view_offset+4*i:self.view_offset+4*i+4]
        if self.physical:
            return physical_camera(self.full_shared(q), view, self.reference_logf)
        camera = np.zeros(25)
        camera[:3], camera[3:6] = q[:3]*100., view[:3]
        camera[6:8], camera[8:10] = view[3], [960., 540.]
        return camera

    def full_shared(self, q):
        full = np.zeros(SHARED_COUNT)
        full[8:10] = [.5, .5]
        if self.physical:
            full[self.physical_shared_columns] = q[:self.view_offset]
        else:
            full[:3] = q[:3]
        return full

    def priors(self, q):
        """Working assumptions, in their own units; never source observations.

        Height is counted once per installation. Roll is horizon orientation,
        not Rodrigues z. The weak distortion prior bounds its image effect at
        three radii under a fixed wide seed focal, rather than guessing a
        manufacturer Brown coefficient. These weights are fixed before checks.
        """
        profile = self.crew_hypothesis
        if profile is None:
            return np.empty(0), csr_matrix((0, len(q)))
        values, blocks = [], []
        full = self.full_shared(q)
        first = q[self.view_offset:self.view_offset+4]
        calibration, view = physical_calibration(full, first, self.reference_logf)
        residual, js, jv = profile.height_residual_and_jacobians(calibration, view)
        block = np.zeros(len(q))
        block[:self.view_offset] = js[self.physical_shared_columns]
        block[self.view_offset:self.view_offset+4] = jv
        values.append(residual); blocks.append(block)
        for shared_index, pixels in ((8, 1920.), (9, 1080.)):
            block = np.zeros(len(q))
            column = list(self.physical_shared_columns).index(shared_index)
            block[column] = pixels/32.
            values.append((full[shared_index]-.5)*pixels/32.); blocks.append(block)
        # Eight pixels at the full-image corner is a modelling allowance,
        # not an empirical Sony correction specification.
        for radius_fraction in (.5, .75, 1.):
            radius = radius_fraction*np.hypot(960., 540.)/self.prior_wide_focal_px
            derivative = self.prior_wide_focal_px*radius**3/8.
            block = np.zeros(len(q))
            block[list(self.physical_shared_columns).index(10)] = derivative
            values.append(full[10]*derivative); blocks.append(block)
        roll_sigma = np.deg2rad(.5)
        for i, p in enumerate(self.stamps):
            start = self.view_offset+4*i
            rotation, raw = cv2.Rodrigues(q[start:start+3])
            dr = raw.reshape(3, 3, 3)
            x, y = rotation[0, 2], -rotation[1, 2]
            if x*x+y*y < 1e-8:
                raise ValueError('Horizon roll undefined for a vertical optical axis')
            block = np.zeros(len(q))
            block[start:start+3] = (y*dr[:, 0, 2]+x*dr[:, 1, 2])/(x*x+y*y)/roll_sigma
            values.append(np.arctan2(x, y)/roll_sigma); blocks.append(block)
        return np.asarray(values), csr_matrix(np.asarray(blocks))

    def evaluate(self, q, jacobian=True, include_priors=True):
        values, data, indices, counts = [], [], [], []
        for key in self.keys:
            p = int(key['pts'])
            camera = self.camera(q, p)
            if not .05 < camera[2] < 100.:
                raise ValueError('Numerical camera-height domain; not a measured hardware bound')
            if np.max(np.abs(camera[11:15])) > 2.:
                raise ValueError('Numerical Brown trial domain; not a measured Z190 coefficient bound')
            if jacobian:
                residual, raw = absolute_value_jac15(camera, key['rows'])
                if self.physical:
                    view = q[self.view_offset+4*self.index[p]:self.view_offset+4*self.index[p]+4]
                    full_map = physical_state_jac(self.full_shared(q), view, self.reference_logf)
                    block = raw @ full_map[:, np.r_[self.physical_shared_columns, SHARED_COUNT+np.arange(4)]]
                else:
                    block = np.c_[raw[:, :6], raw[:, 6]+raw[:, 7]]
                cols = np.r_[np.arange(self.view_offset), self.view_offset+4*self.index[p]+np.arange(4)]
                data.append(block.ravel())
                indices.append(np.broadcast_to(cols, block.shape).astype(np.int32).ravel())
                counts.append(np.full(len(residual), len(cols), np.int32))
            else:
                residual = absolute_value_jac15(camera, key['rows'], False)
            values.append(residual)
        result = np.concatenate(values)
        prior, prior_jac = self.priors(q) if include_priors else (np.empty(0), None)
        if not jacobian:
            return np.r_[result, prior]
        pointer = np.r_[0, np.cumsum(np.concatenate(counts), dtype=np.int64)].astype(np.int32)
        source_jac = csr_matrix((np.concatenate(data), np.concatenate(indices), pointer), shape=(len(result), len(q)))
        return (np.r_[result, prior], vstack((source_jac, prior_jac), format='csr')) if len(prior) else (result, source_jac)

    def initial(self):
        shared = np.zeros(self.view_offset)
        shared[:3] = np.median([k['camera25_seed'][:3] for k in self.keys], axis=0)/100.
        if self.physical:
            full = np.zeros(SHARED_COUNT)
            full[:3] = shared[:3]
            full[8:10] = [.5, .5]
            shared = full[self.physical_shared_columns]
        if self.crew_hypothesis is not None:
            shared[2] = self.crew_hypothesis.height_target_scene/100.
        return np.r_[shared, np.concatenate([np.r_[k['camera25_seed'][3:6], np.mean(k['camera25_seed'][6:8])] for k in self.keys])]

    def rank_review(self, q):
        residual, jac = self.evaluate(q, include_priors=False)
        weighted = jac.toarray()*np.sqrt(np.minimum(1., 2./np.maximum(np.abs(residual), 1e-12)))[:, None]
        info = np.zeros((self.view_offset, self.view_offset))
        offset = 0
        for key in self.keys:
            count = 2*sum(len(r['uv']) for r in key['rows'])
            local = weighted[offset:offset+count]
            shared, view = local[:, :self.view_offset], local[:, self.view_offset+4*self.index[key['pts']]:self.view_offset+4*self.index[key['pts']]+4]
            info += shared.T@shared-shared.T@view@np.linalg.pinv(view.T@view, rcond=1e-10)@view.T@shared
            offset += count
        active = np.array([i for i in range(self.view_offset) if not (self.crew_fixed_height and i == 2)], int)
        info = info[active][:, active]
        eigen, vectors = np.linalg.eigh(.5*(info+info.T))
        biggest = max(float(eigen[-1]), 1e-20)
        free = np.r_[active, np.arange(self.view_offset, len(q))]
        singular = np.linalg.svd(weighted[:, free], compute_uv=False)
        return dict(shared_eigenvalues=eigen.tolist(), shared_parameter_count=len(active),
                    ambient_shared_parameter_count=self.view_offset, ambient_parameter_count=len(q),
                    optimized_shared_local_columns=active.tolist(), frozen_shared_local_columns=[2] if self.crew_fixed_height else [],
                    shared_rank_relative_1e_8=int(np.sum(eigen > biggest*1e-8)),
                    shared_weak_direction=vectors[:, 0].tolist(),
                    full_singular_values=singular.tolist(), full_rank_relative_1e_8=int(np.sum(singular > singular[0]*1e-8)),
                    parameter_count=len(free), observation_residual_rows=len(residual),
                    status='local conditional source information, no priors; not hardware covariance')


def solve_with_crew_priors(bundle, q0, output):
    """Sparse LM with robust source pixels and quadratic working priors.

    The existing production solver is untouched. Metre/angle prior rows are
    excluded from pixel step stopping and data-only identifiability review.
    """
    from main7_source_bundle_candidate import huber_cost
    q = q0.copy(); history = []; started = time.perf_counter(); damping = 1e-3
    fixed_height = getattr(bundle, 'fixed_height_scene_units', bundle.crew_hypothesis.height_target_scene)
    if bundle.crew_fixed_height and not np.isclose(q[2]*100., fixed_height, rtol=0., atol=1e-12):
        raise ValueError('Fixed-height initialization must equal the declared standing-height hypothesis')
    n = bundle.source_residual_rows
    free = np.asarray(getattr(bundle, 'optimized_local_columns',
        [i for i in range(len(q)) if not (bundle.crew_fixed_height and i == 2)]), int)
    if len(np.unique(free)) != len(free) or np.any((free < 0) | (free >= len(q))) or (bundle.crew_fixed_height and 2 in free):
        raise ValueError('Invalid explicit optimized source parameter columns')
    def cost(values, robust):
        return huber_cost(values[:n], robust)+.5*float(values[n:]@values[n:])
    for robust in (False, True):
        for iteration in range(45):
            r, j = bundle.evaluate(q)
            before = cost(r, robust)
            weight = np.ones(len(r))
            if robust:
                weight[:n] = np.minimum(1., 2./np.maximum(np.abs(r[:n]), 1e-12))
            jw = j.multiply(np.sqrt(weight)[:, None])
            normal = (jw.T@jw).tocsc(); gradient = j.T@(weight*r)
            diagonal = np.maximum(normal.diagonal(), 1.)
            accepted, reason = False, ''; step = np.zeros_like(q)
            for trial in range(12):
                step = np.zeros_like(q)
                damped = normal+diags(damping*diagonal)
                step[free] = spsolve(damped[free][:, free], -gradient[free], permc_spec='MMD_AT_PLUS_A')
                nextq = q+step
                try:
                    if (not np.isfinite(nextq).all() or
                        np.any((nextq[bundle.view_offset+3::4]<np.log(100)) |
                               (nextq[bundle.view_offset+3::4]>np.log(50000)))):
                        raise ValueError('Candidate numerical focal domain')
                    after = cost(bundle.evaluate(nextq, False), robust)
                    if after < before:
                        q = nextq; accepted = True; damping = max(damping*.3, 1e-12); break
                    reason = 'objective_not_lower'
                except (ValueError, RuntimeError, BoardIntersectionError) as exc:
                    reason = str(exc)[:160]
                damping *= 10.
            source_step = float(np.max(np.abs((j@step)[:n]*bundle.native_residual_sigma)))
            prior_step = float(np.max(np.abs((j@step)[n:])))
            history.append(dict(phase='huber_source_quadratic_prior' if robust else 'linear_source_quadratic_prior',
                iteration=iteration, cost=before, next_cost=after if accepted else before, accepted=accepted,
                trials=trial+1, damping=damping, failure_reason=None if accepted else reason,
                maximum_linearized_source_step_px=source_step, maximum_linearized_prior_step_sigma=prior_step))
            (output/'bundle_iterations.json').write_text(json.dumps(history, indent=2), encoding='utf8')
            if not accepted or (source_step < .05 and prior_step < .001):
                break
    return q, history, time.perf_counter()-started


def held_camera(bundle, q, key):
    nodes = sorted(int(k['pts']) for k in bundle.keys if k['cut'] == key['cut'])
    if not nodes:
        return None, dict(method='no same-cut fitting support; retained unknown')
    p = int(key['pts'])
    right = min(int(np.searchsorted(nodes, p)), len(nodes)-1)
    left = max(0, right-1)
    if p < nodes[0]:
        left = right = 0
    pa, pb = nodes[left], nodes[right]
    weight = 0. if pa == pb else (p-pa)/(pb-pa)
    def correction(stamp):
        seed = np.array(bundle.keys[bundle.index[stamp]]['camera25_seed'], float)
        fitted = bundle.camera(q, stamp)
        return np.r_[fitted[3:6]-seed[3:6], np.mean(fitted[6:8])-np.mean(seed[6:8])]
    delta = (1.-weight)*correction(pa)+weight*correction(pb)
    seed = np.array(key['camera25_seed'], float)
    view = np.r_[seed[3:6], np.mean(seed[6:8])]+delta
    if bundle.physical:
        camera = physical_camera(bundle.full_shared(q), view, bundle.reference_logf)
    else:
        camera = np.zeros(25)
        camera[:3], camera[3:6] = q[:3]*100., view[:3]
        camera[6:8], camera[8:10] = view[3], [960., 540.]
    return camera, dict(method='same-cut interpolation of fit-key R/logf corrections; no held pixels fitted',
                        view_transfer_unverified=True, valid_physical_model_holdout=False,
                        bracketing_pts=[pa, pb], weight=float(weight),
                        bracket_seconds=float((pb-pa)/15360.), outside_fit_time_range=bool(p < nodes[0] or p > nodes[-1]))


def derivative_check(bundle, q):
    # Shared physical parameters and one real-key R/logf. No synthetic target.
    residual, analytic = bundle.evaluate(q)
    columns = list(range(bundle.view_offset))+list(range(bundle.view_offset, bundle.view_offset+4))
    checks = []
    for column in columns:
        h = 1e-7
        plus, minus = q.copy(), q.copy()
        plus[column] += h
        minus[column] -= h
        fd = (bundle.evaluate(plus, False)-bundle.evaluate(minus, False))/(2*h)
        actual = analytic[:, column].toarray().ravel()
        checks.append(dict(column=column, step=h,
                           relative_l2_difference=float(np.linalg.norm(fd-actual)/max(1., np.linalg.norm(fd))),
                           maximum_difference=float(np.max(np.abs(fd-actual)))))
    if max(r['relative_l2_difference'] for r in checks) > 2e-4:
        raise ValueError('Real-source physical model derivative mismatch: '+json.dumps(checks))
    return checks


def future_groups(base, scene, seedfile):
    seeds = arrays(seedfile)
    cameras = dict(zip(map(int, seeds['pts']), seeds['camera25']))
    world = _world_curves(scene)
    result, bindings = [], []
    for cut in (1, 9):
        path = base/f'main{cut}_source_checks_future5_5/source_rows.json'
        bindings.append(dict(path=str(path), sha256=sha(path), role='untouched_unfitted_future_source_checks'))
        rows = json.loads(path.read_text(encoding='utf8'))
        for p in sorted({int(r['pts']) for r in rows}):
            observed = []
            for raw in rows:
                if int(raw['pts']) != p:
                    continue
                row = dict(raw)
                if row['id'] not in world:
                    raise ValueError('Unmodeled future-check identity: '+row['id'])
                row['world'] = world[row['id']]
                row['original_source_split'] = row['split']
                observed.append(row)
            result.append(source_group(dict(pts=p, cut=cut, installation=0,
                                           camera25_seed=cameras[p], rows=observed,
                                           held_role='future5_5_no_pixels_used_in_fitting')))
    return result, bindings


def summarize_source(errors_by_key):
    values = [r['median_px'] for key in errors_by_key for r in key.get('errors', []) if 'median_px' in r]
    return dict(curve_or_point_groups=len(values), median_of_group_medians_px=float(np.median(values)) if values else None,
                worst_group_median_px=float(np.max(values)) if values else None)


def source_errors(camera, rows):
    result = []
    for row in rows:
        try:
            # Point branches have the same forward/central guard as curves.
            supported=absolute_value_jac15(camera, [row], False)
            kind=row.get('curve_observation_kind')
            if kind=='clipped_horizontal_extent':
                outside=np.linalg.norm((supported*np.repeat(row['sigma'],2)).reshape(-1,2),axis=1)
                result.append(dict(pts=row['pts'],id=row['id'],split=row['split'],points=0,
                    observation_kind='one_sided_clipped_complete_ring_visibility',extent_side=row['extent_side'],
                    image_boundary_x=row['clip_boundary_x'],engineering_blur_allowance_px=row['clip_blur_allowance_px'],
                    outside_source_support_median_px=float(np.median(outside)),
                    source_uv_role='image-boundary context only; no measured tangent UV',
                    no_tangent_point_correspondence_invented=True))
            elif kind=='horizontal_extent' or row.get('must_preserve_upper_branch') or kind=='axis_scanline_branch':
                raw=(horizontal_extent_value_jac15(camera,row,False)if kind=='horizontal_extent'else scanline_branch_value_jac15(camera,row,False)).reshape(-1,2)
                distances=np.linalg.norm(raw,axis=1)
                outside=np.linalg.norm((supported*np.repeat(row['sigma'],2)).reshape(-1,2),axis=1)
                result.append(dict(pts=row['pts'],id=row['id'],split=row['split'],points=len(row['uv']),
                    observation_kind='complete_curve_horizontal_extent'if kind=='horizontal_extent'else'identified_source_scanline_branch',extent_side=row.get('extent_side'),
                    median_px=float(np.median(distances)),p95_px=float(np.quantile(distances,.95)),maximum_px=float(np.max(distances)),
                    outside_source_support_median_px=float(np.median(outside)),outside_source_support_p95_px=float(np.quantile(outside,.95)),
                    source_y_role='source search region only, not a point constraint'if kind=='horizontal_extent'else'identified upper/lower source branch ordinate'))
            else:result.extend(errors(camera, [row]))
        except BoardIntersectionError as error:
            result.append(dict(pts=row['pts'], id=row['id'], split=row['split'],
                               invalid_projection=True, reason=getattr(error, 'reason', str(error))))
    return result


def development_group_holdout(keys):
    """Withhold existing ice groups from views restored using boards/ads only.

    These rows retain their original fit/development identity. They have been
    used historically and do not become independent evidence by this split.
    """
    result, held = [], {}
    for key in keys:
        candidate = [r for r in key['rows'] if r['observation_surface_type'] == 'ice_marking']
        support = [r for r in key['rows'] if r['observation_surface_type'] != 'ice_marking']
        if candidate and len(support) >= 2:
            held[int(key['pts'])] = [dict(r, diagnostic_role='development_physical_group_holdout',
                                         consumed_in_this_fit=False, independently_sourced_validation=False) for r in candidate]
            result.append(dict(key, rows=support))
        else:
            result.append(key)
    return result, held


def export_physical_model(bundle, q, folder, installation, source_sha):
    """Exercise the actual adapter save/load/NPZ APIs, including baseline."""
    from dataclasses import replace
    import z190_physical_camera as api
    identifier = f'okada_sp_installation_{installation}_conditional_z190'
    first = q[bundle.view_offset:bundle.view_offset+4]
    calibration, _ = physical_calibration(bundle.full_shared(q), first, bundle.reference_logf)
    settings = calibration.field_settings
    if bundle.crew_hypothesis is not None:
        settings = dict(bundle.crew_hypothesis.create_calibration(calibration.pivot_xyz[:2]).field_settings)
        settings['height_fit_policy'] = 'fixed_at_working_1_60m' if bundle.crew_fixed_height else 'soft_gaussian_working_prior'
    calibration = replace(calibration, installation_id=identifier, field_settings=settings)
    model = api.PhysicalCameraModel({identifier: calibration})
    calibration_file = folder/'physical_calibration_unaccepted.json'
    model.save_calibration_json(calibration_file)
    reloaded = api.PhysicalCameraModel.load_calibration_json(calibration_file)
    canonical = {}
    for p in bundle.stamps:
        i = bundle.index[p]
        view = q[bundle.view_offset+4*i:bundle.view_offset+4*i+4]
        canonical[p] = api.CanonicalView(identifier, api.ViewState(tuple(view[:3]), float(view[3]-np.log(1920.))))
    aliases = {p: p for p in bundle.stamps}
    output_file = folder/'physical_model_camera25_unaccepted.npz'
    reloaded.save_npz(output_file, bundle.stamps, aliases, canonical,
                      source_sha256=source_sha, time_base='1/15360',
                      installation_by_pts={p: identifier for p in bundle.stamps})
    saved = arrays(output_file)
    reference = np.array([bundle.camera(q, p) for p in bundle.stamps])
    difference = float(np.max(np.abs(saved['camera25']-reference)))
    if difference > 1e-9:
        raise ValueError('Saved physical calibration cannot reproduce its camera25 fit')
    return dict(calibration_json=str(calibration_file), camera25_npz=str(output_file),
                canonical_state4_npz=str(output_file),
                state4_provenance='canonical solver state, directly exported',
                max_roundtrip_camera25_difference=difference, shared_parameter_names=list(calibration.shared_state_names()),
                shared_parameters_full=calibration.shared_state().tolist(), accuracy_accepted=False)


def reexport_saved_camera25(profile, legacy_npz, destination):
    """Repackage a declared final-effective numeric fit, without refitting.

    The old v1 archive did not save state4 separately. R3 is copied, and q is
    recovered from its recorded logf under the explicitly recorded identity
    output domain. This is not recovery of hardware zoom, OIS or output crop.
    The v2 loader must reproduce every camera25 element exactly.
    """
    import z190_physical_camera as api
    model = api.PhysicalCameraModel.load_calibration_json(profile)
    legacy = arrays(legacy_npz)
    metadata = json.loads(str(legacy['metadata_json'].item()))
    canonical, aliases, by_pts = {}, {}, {}
    for p, representative, camera, identifier in zip(
            legacy['pts'], legacy['exposure_representative_pts'], legacy['camera25'], legacy['installation_id']):
        p, representative, identifier = int(p), int(representative), str(identifier)
        calibration = model.calibrations[identifier]
        if p != representative or calibration.output_mode.domain != 'final_effective':
            raise ValueError('This sparse re-export supports explicit identity keys in final-effective coordinates only')
        if camera[6] != camera[7] or calibration.allow_effective_pp_offset:
            raise ValueError('State4 cannot be repackaged under the declared equal-focal/zero-offset fit')
        view = api.ViewState(tuple(camera[3:6]), float(camera[6]-np.log(calibration.output_mode.domain_wh[0])))
        canonical[p] = api.CanonicalView(identifier, view)
        aliases[p], by_pts[p] = p, identifier
        if not np.array_equal(model.camera25(canonical[p]), camera):
            raise ValueError('Saved numeric state does not exactly replay its declared physical profile')
    model.save_npz(destination, legacy['pts'], aliases, canonical,
                   source_sha256=metadata['source_sha256'], time_base=metadata['source_time_base'],
                   installation_by_pts=by_pts)
    reloaded, pts, restored_aliases, restored_views = api.PhysicalCameraModel.load_npz(destination)
    replay = np.array([reloaded.camera25(restored_views[restored_aliases[int(p)]]) for p in pts])
    if not np.array_equal(replay, legacy['camera25']):
        raise ValueError('v2 loader did not exactly reproduce the saved numeric fit')
    return dict(camera25_npz=str(destination), canonical_state4_npz=str(destination),
                legacy_camera25_npz=str(legacy_npz), max_roundtrip_camera25_difference=0.,
                state4_provenance='recovered_from_saved_numeric_camera25_R3_and_declared_final_effective_logf',
                source_video_sha256=metadata['source_sha256'], source_time_base=metadata['source_time_base'],
                sparse_identity_aliases_only=True, full_programme_exposure_identity_accepted=False,
                strict_v2_load_verified=True)


def render_comparisons(output, scene, packet, historic, raw_keys, candidate_sets):
    """A few actual source images, all modeled grid/markings/ads enabled."""
    from render_result import SceneRenderer, decode_selected
    renderer = SceneRenderer(scene, draw_grid=True, draw_advertisements=True)
    selected = {}
    for installation in sorted(raw_keys):
        for label in ('wide_relative', 'mid_relative', 'tele_relative'):
            keys = [k for k in raw_keys[installation] if k['zoom_sampling_bin'] == label]
            if keys:
                selected[int(keys[len(keys)//2]['pts'])] = (installation, label)
    rendered = []
    for p, frame in decode_selected(Path(packet['source_path']), sorted(selected)):
        installation, label = selected[p]
        panels, metrics = [], []
        cases = [('historical B + same hard scene', historic[p])]
        cases.extend((name, cameras[p]) for name, cameras in candidate_sets[installation].items())
        for name, camera in cases:
            panel, info = renderer.draw(frame, camera, None,
                                         title=f'{name} | slot {installation} {label} PTS {p} | UNACCEPTED')
            panels.append(cv2.resize(panel, (960, 540)))
            metrics.append(dict(model=name, **info))
        file = output/f'installation{installation}_{label}_{p}_all_markings_ads.jpg'
        if not cv2.imwrite(str(file), np.vstack(panels)):
            raise OSError('Could not save source comparison image')
        rendered.append(dict(pts=p, source_seconds=p/15360., installation=installation,
                             zoom_sampling_bin=label, path=str(file), panels=metrics,
                             whole_cut_review=False, representative_source_image_only=True))
    return rendered


def write_manifest(output, report):
    """Small review entry point; no candidate becomes a programme default."""
    installations = {}
    for installation, entry in report['installations'].items():
        variants = {}
        for name, result in entry['variants'].items():
            exported = result.get('adapter_export', {})
            variants[name] = dict(
                profile=exported.get('calibration_json'),
                profile_sha256=sha(exported['calibration_json']) if exported.get('calibration_json') else None,
                camera25_npz=exported.get('camera25_npz'),
                camera25_npz_sha256=sha(exported['camera25_npz']) if exported.get('camera25_npz') else None,
                canonical_state4_npz=exported.get('canonical_state4_npz'),
                legacy_camera25_npz=exported.get('legacy_camera25_npz'),
                state4_provenance=exported.get('state4_provenance'),
                selected_shared_columns=([0, 1, 2] if name == 'centred_pinhole_fixed_C_refit' else
                                         ([0, 1, 8, 9, 10] if result.get('working_prior_review', {}).get('frozen_shared_columns') else [0, 1, 2, 8, 9, 10]) if name == 'standing_tripod_shared_PP_k1' else
                                         [0, 1, 2, 8, 9, 10, 11] if name == 'fixed_pivot_shared_PP_k1_k2' else list(range(14))),
                fit=result['fit_summary'],
                development_physical_group_holdout=result.get('development_group_holdout_summary'),
                shared_local_rank=result['rank']['shared_rank_relative_1e_8'],
                shared_parameter_count=result['rank']['shared_parameter_count'],
                sparse_pose_transfer=result['held_summary'],
                sparse_pose_transfer_verified=False,
                crew_hypothesis=result.get('crew_hypothesis'),
                working_prior_review=result.get('working_prior_review'),
                accuracy_accepted=False)
        installations[installation] = dict(real_source_key_count=len(entry['selected_keys']), variants=variants)
    manifest = dict(
        schema='rinkspace.z190-source-diagnostic-manifest.v1',
        candidate_accepted=False, accuracy_accepted=False, install_as_default=False,
        scene_sha256=report['scene_sha256'], report=str(output/'report.json'),
        diagnostic_script=str(PROGRAM/'z190_source_camera_diagnostic.py'),
        diagnostic_script_sha256=sha(PROGRAM/'z190_source_camera_diagnostic.py'),
        physical_module=str(PROGRAM/'z190_physical_camera.py'),
        physical_module_sha256=sha(PROGRAM/'z190_physical_camera.py'),
        load_api='z190_physical_camera.PhysicalCameraModel.load_npz(path) -> model, pts, representative_pts_mapping, canonical_views',
        source_video=report['source_video_path'], source_timebase_denominator=15360,
        source_video_sha256=report.get('source_video_sha256'),
        source_video_sha256_verification=report.get('source_video_sha256_verification', 'explicit existing B source-video identity'),
        source_key_packet=report['source_path'], source_key_packet_sha256=report['source_sha256'],
        run_command_argv=report.get('run_command_argv'),
        run_environment=report.get('run_environment'),
        rerun_output_policy='use a fresh output directory; adapter exports refuse overwrite',
        source_display_rate=60, physical_exposure_rate='unknown_in_source_packet',
        crew_assumed_capture_rate_hz=50 if report.get('standing_tripod_priors') else None,
        fixed_scene=True, original_source_split_preserved=True,
        development_holdout_is_independent_validation=False,
        future_pose_transfer='unverified sparse same-cut correction interpolation; no source-motion recovery',
        controlled_shared_variables=(['installation optical-centre XY', 'constant principal point XY', 'constant residual k1']+
                                     ([] if report.get('crew_fixed_height') else ['installation optical-centre Z'])
                                     if report.get('standing_tripod_priors') else
                                     ['shared installation pivot XYZ', 'constant principal point XY', 'constant residual k1', 'constant residual k2']),
        controlled_fixed_zero=['lever XYZ', 'axial zoom shift', 'tangential p1/p2', 'skew', 'higher Brown terms']+
                             (['k2'] if report.get('standing_tripod_priors') else []),
        fixed_height_m=1.60 if report.get('crew_fixed_height') else None,
        view_state_variables=['world-to-camera Rodrigues XYZ', 'log(final effective focal px / 1920)'],
        hardware_zoom_output_OIS_identified=False,
        exposure_alias_scope='selected sparse source keys have identity aliases only; no whole-programme exposure expansion',
        review_images=[{k: item[k] for k in ('pts', 'source_seconds', 'installation', 'path')}
                       for item in report.get('source_image_comparisons', [])],
        image_review_scope='representative source frames only; no whole-cut acceptance',
        installations=installations)
    (output/'manifest.json').write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2), encoding='utf8')
    return manifest


def run(a):
    output = a.output.resolve()
    if not output.is_relative_to(RESEARCH_ROOT.resolve()):
        raise ValueError('Diagnostic output must stay in this chat z190_research directory')
    output.mkdir(parents=True, exist_ok=True)
    packet = json.loads(a.source.read_text(encoding='utf8'))
    scene_path = Path(packet['scene_path']) if a.scene is None else a.scene
    if sha(scene_path) != HARD_SCENE_SHA or packet['scene_sha256'] != HARD_SCENE_SHA:
        raise ValueError('The real-source packet and restored hard scene must both be SHA93c')
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    held = [source_group(g) for g in packet['held_groups']]
    future, future_bindings = future_groups(a.previous_base, scene, a.future_seed)
    held.extend(future)
    held_pts = {int(g['pts']) for g in held}
    original = arrays(B/'candidate_poses.npz')
    historic = dict(zip(map(int, original['pts']), original['camera25']))
    report = dict(schema='rinkspace.z190-source-key-physical-camera-diagnostic.v1',
                  candidate_accepted=False, accuracy_accepted=False, scene_path=str(scene_path), scene_sha256=HARD_SCENE_SHA,
                  run_command_argv=list(sys.orig_argv),
                  run_environment={'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                  source_path=str(a.source), source_sha256=sha(a.source), source_video_path=packet['source_path'],
                  source_timebase_denominator=15360, source_display_rate=60, capture_rate_known=False,
                  source_pixels_modified=False, scene_variables_optimized=0, graph_or_production_camera_files_modified=False,
                  held_rows_promoted_to_fit=False, whole_cut_or_programme_localization=False,
                  source_bindings=packet['input_bindings']+future_bindings,
                  old_B_camera_path=str(B/'candidate_poses.npz'), old_B_camera_sha256=sha(B/'candidate_poses.npz'),
                  old_B_policy='unchanged historical B cameras evaluated against this same fixed hard scene and native source rows',
                  held_pose_limit='view_transfer_unverified: sparse same-cut R/logf correction interpolation is not a valid physical-model holdout test',
                  untouched_future_or_secondary_checks_are_valid_model_holdout=False,
                  development_group_holdout_limit='former fit/development ice groups withheld in this solve; source poses fitted from other physical groups at same PTS; not independent historical validation',
                  zoom_bin_limit='wide/mid/tele are relative effective pixel focal seed bins, not measured optical zoom settings',
                  standing_tripod_priors=a.standing_tripod_priors,
                  crew_fixed_height=a.crew_fixed_height,
                  installations={})
    raw_keys_by_slot, candidate_sets = {}, {}
    source_sha = str(original['source_sha256'].item())
    report['source_video_sha256'] = source_sha
    for installation in a.installations:
        raw_keys = select_keys(packet['groups'], installation, a.keys_per_bin, held_pts)
        keys, development_held = development_group_holdout(raw_keys) if a.development_group_holdout else (raw_keys, {})
        if not keys:
            continue
        raw_keys_by_slot[installation] = raw_keys
        candidate_sets[installation] = {}
        entry = dict(selected_keys=[dict(pts=k['pts'], cut=k['cut'], seed_focal_px=float(np.exp(np.mean(k['camera25_seed'][6:8]))),
                                        zoom_sampling_bin=k['zoom_sampling_bin'], fit_rows=[dict(id=r['id'], split=r['split'],
                                        original_source_split=r.get('original_source_split'), samples=len(r['uv']), surface=r['observation_surface_type'],
                                        origin=r.get('origin')) for r in k['rows']]) for k in keys], variants={})
        fit_keys = [dict(pts=k['pts'], cut=k['cut'], errors=source_errors(historic[int(k['pts'])], k['rows'])) for k in keys]
        entry['old_B_same_hard_scene'] = dict(fit=fit_keys, summary=summarize_source(fit_keys))
        slots = [g for g in held if g['installation'] == installation]
        variants = [(False, 'centred_pinhole_fixed_C_refit', None),
                    (True, 'fixed_pivot_shared_PP_k1_k2', [0, 1, 2, 8, 9, 10, 11])]
        if a.standing_tripod_priors:
            variants[1] = (True, 'standing_tripod_shared_PP_k1', [0, 1, 2, 8, 9, 10])
        if a.full_physical:
            variants.append((True, 'z190_full_physical_shared_optics_refit', None))
        for physical, name, columns in variants:
            import z190_physical_camera as api
            hypothesis = api.Z190StandingTripodHypothesis(installation) if name == 'standing_tripod_shared_PP_k1' else None
            bundle = SourceKeys(keys, physical, physical_shared_columns=columns, crew_hypothesis=hypothesis,
                                crew_fixed_height=a.crew_fixed_height and hypothesis is not None)
            q0 = bundle.initial()
            if physical:
                # Same real-source pinhole fit initializes R/f and T, never creates observations.
                q0[:3] = baseline_q[:3]
                q0[bundle.view_offset:] = baseline_q[3:]
                if hypothesis is not None:
                    q0[2] = hypothesis.height_target_scene/100.
            folder = output/f'installation{installation}'/name
            folder.mkdir(parents=True, exist_ok=True)
            check = derivative_check(bundle, q0) if physical else None
            (folder/'derivative_check.json').write_text(json.dumps(check, indent=2), encoding='utf8')
            q, history, seconds = (solve_with_crew_priors(bundle, q0, folder) if hypothesis is not None else
                                    solve(bundle, q0, folder, verbose=False))
            if not physical:
                baseline_q = q.copy()
            candidates = np.array([bundle.camera(q, p) for p in bundle.stamps])
            candidate_sets[installation][name] = {p: c for p, c in zip(bundle.stamps, candidates)}
            np.savez_compressed(folder/'candidate_key_cameras.npz', pts=np.array(bundle.stamps, np.int64), camera25=candidates,
                                exposure_representative_pts=np.array(bundle.stamps, np.int64), physical_parameters=q[:bundle.view_offset],
                                status=np.full(len(candidates), 'conditional_z190_source_key_diagnostic_unaccepted'),
                                model_sha256=HARD_SCENE_SHA, reference_logf=bundle.reference_logf)
            adapter_export = export_physical_model(bundle, q, folder, installation, source_sha)
            fitted = [dict(pts=k['pts'], cut=k['cut'], errors=source_errors(bundle.camera(q, k['pts']), k['rows'])) for k in keys]
            development_review = [dict(pts=p, original_source_split_preserved=True,
                independently_sourced_validation=False, pose_recovery='only fitted other physical groups at same real source PTS',
                withheld_physical_rows=[dict(id=r['id'], original_source_split=r['split'], origin=r.get('origin')) for r in rows],
                old_B_errors=source_errors(historic[p], rows), errors=source_errors(bundle.camera(q, p), rows))
                for p, rows in development_held.items()]
            held_review, held_cameras = [], []
            for key in slots:
                camera, method = held_camera(bundle, q, key)
                item = dict(pts=key['pts'], cut=key['cut'], prediction=method,
                            source_rows=[dict(id=r['id'], split=r['split'], original_source_split=r.get('original_source_split'),
                                              samples=len(r['uv']), origin=r.get('origin')) for r in key['rows']],
                            old_B_errors=source_errors(historic[int(key['pts'])], key['rows']))
                if camera is not None:
                    item['errors'] = source_errors(camera, key['rows'])
                    held_cameras.append((int(key['pts']), camera))
                held_review.append(item)
            if held_cameras:
                np.savez_compressed(folder/'held_prediction_cameras.npz', pts=np.array([p for p,c in held_cameras], np.int64),
                                    camera25=np.array([c for p,c in held_cameras]),
                                    exposure_representative_pts=np.array([p for p,c in held_cameras], np.int64),
                                    status=np.full(len(held_cameras), 'held_pose_interpolated_no_held_source_fitted_unaccepted'), model_sha256=HARD_SCENE_SHA)
            entry['variants'][name] = dict(seconds=seconds, parameters=q[:bundle.view_offset].tolist(), reference_logf=bundle.reference_logf,
                                          fit=fitted, fit_summary=summarize_source(fitted), held=held_review, held_summary=summarize_source(held_review),
                                          held_view_transfer_unverified=True, development_group_holdout=development_review,
                                          development_group_holdout_summary=summarize_source(development_review),
                                          adapter_export=adapter_export, rank=bundle.rank_review(q), derivative_check=check, history=history)
            if hypothesis is not None:
                prior, _ = bundle.priors(q)
                camera = candidates[0]
                entry['variants'][name]['crew_hypothesis'] = hypothesis.profile()
                entry['variants'][name]['working_prior_review'] = dict(
                    final_camera_height_m=float(camera[2]*hypothesis.scene_unit_m),
                    height_residual_sigma=float(prior[0]),
                    principal_point_final_px=camera[8:10].tolist(), residual_k1=float(camera[11]),
                    fixed_k2=0., wide_seed_focal_for_distortion_prior_px=bundle.prior_wide_focal_px,
                    height_fit_policy='fixed_at_working_1_60m' if bundle.crew_fixed_height else 'soft_gaussian_working_prior',
                    frozen_shared_columns=[2] if bundle.crew_fixed_height else [],
                    prior_rows=len(prior), prior_residuals_sigma=prior.tolist(),
                    prior_policy='quadratic assumed priors; source pixels alone use Huber; original observations untouched')
            print(json.dumps(dict(installation=installation, variant=name, seconds=seconds, fit=entry['variants'][name]['fit_summary'],
                                  held=entry['variants'][name]['held_summary'], rank=entry['variants'][name]['rank']['shared_rank_relative_1e_8'],
                                  development_group_holdout=entry['variants'][name]['development_group_holdout_summary'],
                                  accuracy_accepted=False)), flush=True)
            report['installations'][str(installation)] = entry
            (output/'report.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding='utf8')
    report['physical_model_module_sha256'] = sha(PROGRAM/'z190_physical_camera.py')
    report['diagnostic_script_sha256'] = sha(__file__)
    if a.render_comparisons:
        report['source_image_comparisons'] = render_comparisons(output, scene, packet, historic, raw_keys_by_slot, candidate_sets)
    (output/'report.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding='utf8')
    write_manifest(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'previous-base', 'future-seed'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, default=RESEARCH_ROOT/'model_diagnostic')
    parser.add_argument('--installations', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--keys-per-bin', type=int, default=4)
    parser.add_argument('--development-group-holdout', action='store_true')
    parser.add_argument('--full-physical', action='store_true', help='Exploratory full lever/axial model; default stays fixed pivot')
    parser.add_argument('--standing-tripod-priors', action='store_true', help='Presumed professional Z190 crew settings and conditional-metric standing-height/PP/roll priors; k2 fixed zero')
    parser.add_argument('--crew-fixed-height', action='store_true', help='Controlled comparison at exactly the presumed 1.60m; no inferred floor-height change')
    parser.add_argument('--render-comparisons', action='store_true')
    args = parser.parse_args()
    if args.full_physical and args.standing_tripod_priors:
        parser.error('Full physical exploration and standing-tripod controlled solve are separate hypotheses')
    if args.crew_fixed_height and not args.standing_tripod_priors:
        parser.error('Fixed working height requires the standing-tripod hypothesis')
    run(args)
