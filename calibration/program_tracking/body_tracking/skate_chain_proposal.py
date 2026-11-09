"""Lift native image shoe tasks into attached-leg proposals on the existing curve."""

import numpy as np
from scipy.sparse.linalg import lsmr

from .absolute_motion_curve import _exp, _log
from .skate_physics import batch_features, fk_state


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.e-12)


def swing(a, b):
    """Shortest rotation between two model bone directions."""
    a, b = unit(a), unit(b)
    normal = np.cross(a, b)
    sine = np.linalg.norm(normal, axis=-1)
    cosine = np.einsum('...j,...j->...', a, b)
    if np.any((sine < 1.e-7) & (cosine < 0)):
        raise ValueError('Opposite bone directions need a defined bend branch')
    return _exp(unit(normal) * np.arctan2(sine, cosine)[..., None])


def select_depth(hip, knee, ankle, target, ray, *, slide_support, support, permit_root, ratio):
    """Profile depths of a provisional source pose subject to link reach.

    The pelvis shift is a task-space suggestion only. The native torso, ice,
    motion and anatomical factors decide whether the original state can move.
    """
    upper = np.linalg.norm(knee - hip, axis=-1)
    lower = np.linalg.norm(ankle - knee, axis=-1)
    reach = np.sqrt(upper**2 + lower**2 + 2 * upper * lower * np.cos(np.deg2rad(4.)))
    h = target - hip
    nearest_depth = -np.einsum('nj,nj->n', h, ray)
    discriminant = reach**2 - np.sum((h + nearest_depth[:, None] * ray)**2, axis=1)
    crossing = np.sqrt(np.maximum(0., discriminant))
    reference_depth = np.einsum('nj,nj->n', ankle-target, ray)
    # Move along the optical ray, staying in the near-current branch.
    ray_scale = .22 / ratio
    depths = np.column_stack((np.zeros(len(hip)), reference_depth,
        nearest_depth - crossing, nearest_depth + crossing,
        slide_support, .5*slide_support))
    depths = np.clip(depths, -ray_scale, ray_scale)
    displacement = h[:, None] + depths[..., None] * ray[:, None]
    gap = np.linalg.norm(displacement, axis=-1)
    root = np.maximum(gap-reach[:, None], 0.)[..., None] * unit(displacement)
    lower_bound = np.maximum(upper-lower, lower-upper)
    root_size = np.linalg.norm(root, axis=-1)
    possible = (gap > lower_bound[:, None]+1.e-8)
    if not permit_root:
        possible &= root_size < 1.e-8
        root *= 0.
    # Penalize unnecessary depth and common root motion. Support contributes
    # only where the source has identified the actual supporting foot.
    depth_cost = (depths/ray_scale)**2
    root_cost = (root_size*ratio/.05)**2
    ice_cost = np.where(support[:, None],
        ((depths-slide_support[:, None])*ray[:, None, 2]*ratio/.024)**2, 0.)
    pick = np.argmin(np.where(possible, depth_cost + root_cost + ice_cost, np.inf), axis=1)
    row = np.arange(len(pick))
    slide = depths[row, pick]
    roots = root[row, pick]
    return slide, roots, possible[row, pick], upper, lower


def _support_height(binding, R, O):
    features = batch_features(binding, R[:, None], O[:, None])
    return features['height'][:, 0, np.asarray(features['semantics']) == 'runner'].min(axis=1)


def profiled_leg_proposal(replay, tasks, columns, *, permit_root, fraction=1.):
    """Fit task-space proposals to the current SO(3) and position controls."""
    state = replay.problem.body.state
    frames = np.searchsorted(replay.pts, [p['pts'] for p in tasks])
    feet = np.array([p['foot'] for p in tasks], int)
    times = replay.ref.times[frames]
    x = replay.x[17:]
    d = state.increments(x, times)
    base = replay.ref.at(times)
    s = fk_state(replay.ref, base, d)
    i = np.arange(len(tasks))
    h, k, a = (offset + feet for offset in (1, 4, 7))
    H, K, A = (s['native_world'][i, j] for j in (h,k,a))
    O, R = s['O'][i, feet], s['R'][i, feet]
    binding = replay.problem.binding
    v = np.asarray([p['task_step_xyz_m_and_rot_rad'] for p in tasks]) * fraction
    W = _exp(v[:, 3:])
    new_origin = O + v[:, :3] / binding.ratio
    new_skate = W @ R
    target = new_origin + np.einsum('nij,nj->ni', W, A-O)
    ray = unit(target - replay.camera.camera_centres(frames))
    support = np.zeros(len(frames), bool)
    for e in replay.document['source_episode_review']['episodes']:
        if e['kind'] == 'support_candidate' and e.get('foot') in (0,1):
            support |= ((feet == e['foot']) & (replay.pts[frames] >= e['start_pts']) &
                        (replay.pts[frames] <= e['end_pts_inclusive']))
    slide_support = np.zeros(len(frames))
    if np.any(support):
        before = _support_height(binding, R, O)
        after = _support_height(binding, new_skate, new_origin)
        slide_support = np.divide(before-after, ray[:,2]*binding.ratio,
                                  out=slide_support, where=support & (np.abs(ray[:,2]) > 1.e-4))
    slide, root, feasible, upper, lower = select_depth(H,K,A,target,ray,slide_support=slide_support,
        support=support,permit_root=permit_root,ratio=binding.ratio)
    new_ankle = target + slide[:, None]*ray
    new_hip = H + root
    h_to_a = new_ankle - new_hip
    axis = unit(h_to_a)
    distance = np.linalg.norm(h_to_a, axis=-1)
    along = (upper**2-lower**2+distance**2) / np.maximum(2*distance, 1.e-10)
    center = new_hip + along[:,None]*axis
    offset = K + root - center
    radial = offset - np.einsum('nj,nj->n',offset,axis)[:,None]*axis
    feasible &= np.linalg.norm(radial, axis=-1) > 1.e-8
    knee_radius = np.sqrt(np.maximum(0., upper**2-along**2))
    new_knee = center+knee_radius[:,None]*unit(radial)
    # Unreachable samples have no authority over the spline.
    new_knee[~feasible]=K[~feasible]; new_ankle[~feasible]=A[~feasible]
    new_hip[~feasible]=H[~feasible]; root[~feasible]=0.
    W[~feasible]=np.eye(3)
    world = s['Q'][:,None] @ s['frames']
    hip_swing = swing(K-H, new_knee-new_hip)
    upper_rot = hip_swing @ world[i,h]
    shank = np.einsum('nij,nj->ni', hip_swing, A-K)
    lower_rot = swing(shank, new_ankle-new_knee) @ hip_swing @ world[i,k]
    foot_rot = W @ world[i,a]
    new_local = (s['Q'].swapaxes(-1,-2)@upper_rot,
                 upper_rot.swapaxes(-1,-2)@lower_rot,
                 lower_rot.swapaxes(-1,-2)@foot_rot)
    axes = np.concatenate([np.broadcast_to(np.arange(3),(len(tasks),3)),
                    *[6+3*(j-1)[:,None]+np.arange(3) for j in (h,k,a)]],axis=1)
    tangent = [root]
    for j, rotations in zip((h,k,a), new_local, strict=True):
        tangent.append(_log(base['local_R'][i,j-1].swapaxes(-1,-2) @ rotations) -
                        np.asarray(d)[:,6:].reshape(-1,21,3)[i,j-1])
    target_delta = np.concatenate(tangent, axis=1)
    target_delta[~feasible] = 0.
    local = np.zeros((len(tasks),12,69))
    local[i[:,None], np.arange(12), axes] = 1.
    # Equal normalized task units for translations and rotations. These rows
    # seed an attached pose; they never enter acceptance or add a trajectory.
    normalizer = np.r_[np.full(3,binding.ratio/.08),np.full(9,1./.35)]
    local *= normalizer[None,:,None]
    target_delta *= normalizer
    J = state.chain(local, x, times)[:, columns-17].tocsr()
    useful = np.repeat(feasible,12) & (np.diff(J.indptr)>0)
    result = lsmr(J[useful],target_delta.ravel()[useful],damp=.22,
                  atol=1.e-6,btol=1.e-6)[0]
    report = dict(probed=len(tasks),feasible=int(feasible.sum()),
                  support_profiled=int(np.sum(support&feasible)), root_enabled=permit_root,
                  largest_seed_root_motion_m=float(np.linalg.norm(root,axis=-1).max()*binding.ratio))
    return result,report


def preserve_free_skate(replay, x, columns, frames, foot):
    """Propose a counter-motion of an unmeasured, grasped free skate.

    The saved shoe pose supplies a local continuation prior in source gaps.
    No such prior becomes an observation or overrides a directly traced shoe.
    """
    state = replay.problem.body.state
    time = replay.ref.times[frames]
    base = replay.ref.at(time)
    anchor_joint = 7 + foot
    body_axes = np.r_[6+3*foot:9+3*foot,15+3*foot:18+3*foot,24+3*foot:27+3*foot]
    ratio = replay.problem.binding.ratio
    old = fk_state(replay.ref,base,state.increments(replay.x[17:],time))
    delta = state.increments(x[17:],time)
    old_ankle = old['native_world'][:,anchor_joint]
    old_R = old['R'][:,foot].swapaxes(-1,-2)

    def value(d):
        new = fk_state(replay.ref,base,d)
        return np.c_[(new['native_world'][:,anchor_joint]-old_ankle)*ratio/.10,
                     _log(old_R@new['R'][:,foot])/.35]

    residual = value(delta)
    local = np.zeros((len(frames),6,69))
    for axis in body_axes:
        plus = delta.copy()
        minus = delta.copy()
        plus[:,axis] += 1.e-6
        minus[:,axis] -= 1.e-6
        local[:,:,axis] = (value(plus)-value(minus))/(2.e-6)
    J = state.chain(local,x[17:],time)[:,columns-17].tocsr()
    used = np.diff(J.indptr)>0
    if not np.any(used):
        return np.zeros(len(columns))
    step = lsmr(J[used],-residual.ravel()[used],damp=.45,atol=1.e-6,btol=1.e-6)[0]
    return step/max(1.,np.max(np.abs(step))/.16)
