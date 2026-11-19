"""Transfer a continuous programme body to a frozen, source-clock-matched scene.

Keep the local joints, relative shoes and mounting parameters. Fit one world
root spline across cuts. Parent image positions and lower-skate heights are
preservation targets, including in flight, and carry no new contact evidence.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from source_camera_rebind import (AbsoluteMotionCurve, ProgrammeCamera, bind,
    checked_entry, curve_from_packet, digest, feature_nodes, moving_projection,
    reviewed_cut_weight, stats, write)
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


def transfer(prepared, source, parent, registration, output):
    output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    def log(event, **fields):
        row = dict(event=event, seconds=round(time.perf_counter()-tick, 3), **fields)
        with (output/'transfer.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    log('load_registered_source', registration=str(registration), parent=str(parent))
    old_ref, old_cam, old_info, document, packet = bind(prepared, source, parent)
    camera = ProgrammeCamera(Path.cwd(), old_ref.p, old_ref.p['shot_ids'], record_path=registration)
    info = camera.info
    if camera.scale != old_cam.scale:
        raise ValueError('Prepared solver coordinates differ')
    record = json.loads(registration.read_text(encoding='utf8'))
    if record['source']['sha256'] != document['source_sha256']:
        raise ValueError('Registration and evidence belong to different source media')
    source_new = output/'source.json'
    document['programme_camera_candidate'] = dict(record=str(registration),
        record_sha256=digest(registration), parent_camera_sha256=old_info['state_sha256'],
        registered_programme=dict(path=str(registration), camera_sha256=info['state_sha256'],
                                  scene_values_sha256=info['scene_values_sha256']))
    document['scene_contract'].update(geometry=info['scene_binding'],
        registered_camera_sha256=info['state_sha256'], plane_source=str(registration))
    document['camera_root_transfer'] = dict(parent=checked_entry(parent), parent_source=checked_entry(source),
        programme_registration=checked_entry(registration), source_evidence_rows_unchanged=True,
        calibration_transfer_targets_are_not_observations=True)
    write(source_new, document)
    new_ref, camera, info, _, _ = load_reference(Path.cwd(), prepared, source_new)
    curve = curve_from_packet(packet)
    t = old_ref.times
    frames = np.arange(len(t))
    dt = np.median(np.diff(t))
    B = curve.basis_sparse(t)
    dense = np.linspace(t[0], t[-1], 4*(len(t)-1)+1)
    D2 = curve.basis_sparse(dense, derivative=2).multiply(
        np.sqrt(dt/4*reviewed_cut_weight(old_ref, dense))[:,None]).tocsr()
    relative = camera.midpoint['rotation'].swapaxes(1,2) @ old_cam.midpoint['rotation']
    omega = Rotation.from_matrix(relative).as_rotvec()
    rot_precision = .0002
    # End controls outside the measured midpoints still affect readout nodes.
    # Give their otherwise unobserved directions a weak zero-correction prior.
    guard = eye(B.shape[1],format='csr')*1e-6
    normal = (B.T@B)*(dt/rot_precision**2) + (D2.T@D2)/80**2 + guard
    correction = spsolve(normal.tocsc(), (B.T@omega)*(dt/rot_precision**2))
    rotations = curve.rotation_controls.copy()
    rotations[:,0] = Rotation.from_rotvec(correction).as_matrix() @ rotations[:,0]
    rotated = AbsoluteMotionCurve(curve.knots, curve.position_controls, rotations, curve.edge_reference_quaternion)
    new_ref = absolute_reference(new_ref, rotated, root=prepared)
    new_ref.relative_skate_curve = old_ref.relative_skate_curve
    new_ref.binding.x = packet['x'][:17].copy()
    new_ref.binding._parameter_key = new_ref.binding._packet = None
    new_ref.binding.reference = new_ref
    batches = []
    initial_errors = []
    # Preserve the average change of both lower-skate heights with a finite
    # centimetre-scale prior, to avoid spending monocular depth on root height.
    height_sigma = .015
    target_height_delta = []
    for start in range(0, len(t), 64):
        ix = frames[start:start+64]
        def samples(ref, cam):
            times = t[ix,None] + cam.readout_times(ix)[:,None]*np.linspace(-.5,.5,5)
            return times, feature_nodes(ref,times)[0]
        _, before = samples(old_ref, old_cam)
        times, nodes = samples(new_ref, camera)
        target = moving_projection(old_cam, before, ix)['uv']
        initial_errors.append(np.linalg.norm(moving_projection(camera,before,ix)['uv']-target,axis=-1))
        height_shift = (before[:,2,22:,2]*old_ref.binding.ratio-nodes[:,2,22:,2]*new_ref.binding.ratio).mean(axis=1)
        target_height_delta.extend(height_shift/new_ref.binding.ratio)
        basis = curve.basis_sparse(times.ravel()).tocsr()
        batches.append((ix, nodes, target, basis))
        if start%1024 == 0:log('native_projection_targets', frames=int(ix[-1]+1))
    n = len(curve.position_controls)
    penalty = kron((D2.T@D2)/12**2 + (B.T@B)*(dt*(.3/.75)**2) + guard, eye(3),format='csr')
    height_matrix = kron(B*(np.sqrt(dt)*new_ref.binding.ratio/height_sigma),[[0.,0.,1.]],format='csr')
    h = np.asarray(target_height_delta)*np.sqrt(dt)*new_ref.binding.ratio/height_sigma
    normal_prior = penalty + height_matrix.T@height_matrix
    rhs_prior = height_matrix.T@h
    x = np.zeros(3*n)
    history = []
    def evaluate(x, with_jacobian):
        cost_geometry = 0.
        normal = normal_prior.copy()
        rhs = rhs_prior - normal_prior@x
        errors = []
        for ix,nodes,target,basis in batches:
            change = (basis@x.reshape(n,3)).reshape(len(ix),5,1,3)
            projected = moving_projection(camera,nodes+change,ix,jacobian=with_jacobian)
            err = projected['uv']-target
            errors.append(np.linalg.norm(err,axis=-1))
            pixels = err.size//(2*len(ix))
            weight = dt/pixels
            cost_geometry += np.sum(err**2)*weight
            if with_jacobian:
                first = basis.indices.min()
                last = basis.indices.max()+1
                # Each exposure touches only the four controls at each of five
                # readout nodes. Emit those nonzeros in the local spline block.
                compact = basis[:,first:last].toarray().reshape(len(ix),5,last-first)
                j = projected['pixel_jac_nodes']
                values = np.einsum('fpuqa,fqc->fpuca',j,compact,optimize=True)
                J = coo_matrix(values.reshape(err.size,-1)).tocsr()
                block = (J.T@J).tocoo()
                H = coo_matrix((block.data*weight,(block.row+3*first,block.col+3*first)),shape=normal.shape)
                normal = normal+H
                rhs[3*first:3*last] -= (J.T@err.ravel())*weight
        temporal = float(x@(penalty@x))
        height_cost = float(np.sum((height_matrix@x-h)**2))
        return (cost_geometry+temporal+height_cost)/2, normal, rhs, np.concatenate(errors), dict(
            image_cost=cost_geometry/2,continuity_cost=temporal/2,height_cost=height_cost/2)

    while True:
        cost, normal, rhs, errors, terms = evaluate(x,True)
        entry = dict(iteration=len(history), cost=cost, gradient_max=float(abs(rhs).max()),
                     transfer_px=stats(errors), **terms)
        history.append(entry)
        log('linearization', **entry)
        if entry['gradient_max'] < 1e-6:break
        step = spsolve(normal.tocsc(),rhs)
        if not np.isfinite(step).all():raise ValueError('Singular root correction')
        alpha = 1.
        while True:
            trial = x+alpha*step
            next_cost = evaluate(trial,False)[0]
            if next_cost<cost or np.max(abs(alpha*step))<1e-10:break
            alpha *= .5
        if next_cost>=cost:break
        x = trial
        if cost-next_cost < 1e-9*max(cost,1.):break
    final_cost,_,_,errors,_ = evaluate(x,False)
    position = curve.position_controls+x.reshape(n,3)
    final = AbsoluteMotionCurve(curve.knots,position,rotations,curve.edge_reference_quaternion)
    final.check_continuous_edge_update(curve)
    per_shot = {s:stats(errors[camera.shot_ids==s]) for s in camera.cameras}
    report = dict(parent=checked_entry(parent), parent_source=checked_entry(source),registration=checked_entry(registration),
        camera=record['camera'], geometry=record['geometry_binding'], source=record['source'],
        scale_before=old_ref.binding.ratio, scale_after=new_ref.binding.ratio,
        source_frames=len(t), source_pts_limits=old_ref.p['pts'][[0,-1]].tolist(),time_base=[1,12800],
        initial_px=stats(np.concatenate(initial_errors)), final_px=stats(errors), per_shot_px=per_shot,
        worst_seconds=float(old_ref.p['pts'][errors.max(1).argmax()]/12800),
        height_change_m=stats(abs((B@x.reshape(n,3))[:,2]-np.asarray(target_height_delta))*new_ref.binding.ratio),
        height_prior_sigma_m=height_sigma, history=history, final_cost=final_cost, independent_accuracy_verified=False)
    saved = dict(packet, absolute_position_controls=position, absolute_rotation_controls=rotations,
        absolute_seed_position_controls=position, absolute_seed_rotation_controls=rotations,
        camera_binding_json=np.array(json.dumps(info,sort_keys=True)),
        source_evidence_sha256=np.array(digest(source_new)),camera_root_transfer_json=np.array(json.dumps(report)))
    pending = output/'candidate.pending.npz'
    with pending.open('wb') as stream: np.savez_compressed(stream,**saved)
    pending.replace(output/'candidate.npz')
    report['candidate'] = checked_entry(output/'candidate.npz')
    write(output/'transfer.json',report)
    log('candidate_saved', candidate=report['candidate'], final_px=report['final_px'], height_change_m=report['height_change_m'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','parent','registration','output'):
        parser.add_argument(name,type=lambda x:Path(x).resolve())
    transfer(**vars(parser.parse_args()))
