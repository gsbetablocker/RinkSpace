"""Transfer the saved continuous body to a revised, frozen programme camera.

The old saved projections are calibration-transfer targets, NOT new image
observations or independent accuracy evidence. Local joint and shoe controls
stay byte-identical. Only the continuous world root is adapted; source time,
metric scale, and the shared ice coordinate frame are retained.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, eye, kron, vstack
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation

from compare import bind
from source_review import digest
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve
from calibration.program_tracking.body_tracking.continuous_motion_factors import reviewed_cut_weight
from calibration.program_tracking.body_tracking.footwear_binding import transform
from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection


def write(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    pending.replace(path)


def checked_entry(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def camera_changed_frames(before, after):
    np.testing.assert_array_equal(before.shot_ids, after.shot_ids)
    mask = np.zeros(len(before.shot_ids), bool)
    for shot in before.cameras:
        a, b = before.cameras[shot], after.cameras[shot]
        equal = a.core.library_path == b.core.library_path
        equal &= all(np.array_equal(a.p[k], b.p[k]) for k in
            ('rig', 'controls', 'motion_times', 'first_control_time', 'control_dt'))
        mask[before.shot_ids == shot] = not equal
    return mask


def feature_nodes(ref, times):
    from calibration.program_tracking.body_tracking.skate_physics import fk_state
    state = fk_state(ref, ref.at(times.ravel()), np.zeros((times.size, 69)))
    body = np.einsum('nij,nkj->nki', state['Q'], state['native'][:, :22])+state['P'][:, None]
    vertices = np.asarray(ref.binding.packet()['template']['vertices'])
    indices = np.unique(np.r_[vertices.argmin(0), vertices.argmax(0)])
    shoes = transform(state['R'], state['O'], vertices[indices]).reshape(times.size, -1, 3)
    return np.concatenate((body, shoes), axis=1).reshape(len(times), 5, -1, 3), state


def stats(value):
    value = np.asarray(value)
    return dict(median=float(np.median(value)), p95=float(np.percentile(value, 95)),
                maximum=float(np.max(value)))


def run(args):
    work = args.work
    output = work/'whole_continuity'
    logpath = output/'camera_rebind_progress.jsonl'
    destination = output/'camera_1117_candidate.npz'
    if destination.exists():
        raise FileExistsError(destination)
    tick = time.perf_counter()
    with logpath.open('a', encoding='utf8', buffering=1) as log:
        def emit(event, **values):
            entry = dict(event=event, elapsed_seconds=time.perf_counter()-tick, **values)
            line = json.dumps(entry, ensure_ascii=False)
            log.write(line+'\n'); print(line, flush=True)

        ref, old_camera, old_info, document, packet = bind(args.prepared, args.source, args.parameter)
        camera = ProgrammeCamera(Path.cwd(), ref.p, ref.p['shot_ids'], record_path=args.registration)
        if camera.scale != old_camera.scale or camera.info['length_export_ratio'] != old_info['length_export_ratio']:
            raise ValueError('This root transfer requires the same scene gauge and metric scale')
        changed = camera_changed_frames(old_camera, camera)
        curve = curve_from_packet(packet)
        g = greville_times(curve.knots)
        # Use the already-declared live-edit neighbourhood; do not split the
        # saved body curve or invent an independent per-shot alignment.
        cut_policy = document['scoped_candidate_import']['camera_cut_continuity']
        halo = float(cut_policy['half_width_s'])
        selected = np.zeros(len(g), bool)
        shots = []
        for shot in camera.cameras:
            ix = np.flatnonzero((camera.shot_ids == shot) & changed)
            if not len(ix):
                continue
            lo, hi = ref.times[ix[[0,-1]]]
            selected |= (g >= lo-halo) & (g <= hi+halo)
            shots.append(dict(id=shot, frames=len(ix), source_seconds=(ref.p['pts'][ix[[0,-1]]]/12800.).tolist()))
        controls = np.flatnonzero(selected)
        all_frames = np.arange(len(ref.times))
        all_times = ref.times[:,None]+camera.readout_times(all_frames)[:,None]*np.linspace(-.5,.5,5)
        affected = np.asarray(curve.basis_sparse(all_times.ravel())[:,controls].getnnz(axis=1)>0).reshape(-1,5).any(1)
        frames = np.flatnonzero(affected | changed)
        times = all_times[frames]
        dt = float(np.median(np.diff(ref.times)))
        quadrature = np.unique(np.concatenate([
            np.linspace(ref.times[f[0]], ref.times[f[-1]], 4*(len(f)-1)+1)
            for f in np.split(frames, np.flatnonzero(np.diff(frames)>1)+1)]))
        weights = np.sqrt(dt/4*reviewed_cut_weight(ref, quadrature))
        D2 = curve.basis_sparse(quadrature, derivative=2)[:,controls].multiply(weights[:,None]).tocsr()
        B = curve.basis_sparse(ref.times[frames])[:,controls]
        relative = camera.midpoint['rotation'][frames].swapaxes(-1,-2)@old_camera.midpoint['rotation'][frames]
        angular = Rotation.from_matrix(relative).as_rotvec()
        angular_precision = .0002
        angular_system = vstack((B*np.sqrt(dt)/angular_precision, D2/80.), format='csr')
        angle_controls = np.column_stack([lsmr(angular_system,
            np.r_[angular[:,j]*np.sqrt(dt)/angular_precision, np.zeros(len(quadrature))],
            atol=1e-11, btol=1e-11)[0] for j in range(3)])
        rotations = curve.rotation_controls.copy()
        rotations[controls,0] = Rotation.from_rotvec(angle_controls).as_matrix()@rotations[controls,0]
        rotated = AbsoluteMotionCurve(curve.knots, curve.position_controls, rotations, curve.edge_reference_quaternion)
        nodes, state = feature_nodes(ref, times)
        target = moving_projection(old_camera, nodes, frames)['uv']
        unadapted = moving_projection(camera, nodes, frames)['uv']
        new_pose = evaluate_curve(rotated, times.ravel())
        A = new_pose['Q']@state['Q'].swapaxes(-1,-2)
        rotated_nodes = (np.einsum('nij,nkj->nki', A, nodes.reshape(times.size,-1,3)-state['P'][:,None])
                         +state['P'][:,None]).reshape(nodes.shape)
        basis = curve.basis_sparse(times.ravel()).tocsr()
        node_basis = basis[:,controls]
        # Analytic world-root derivatives through the native moving row solve.
        indices = np.clip(np.searchsorted(curve.knots,times.ravel(),side='right')-4,0,len(g)-4)[:,None]+np.arange(4)
        coefficients = basis[np.arange(times.size)[:,None],indices]
        coefficients = (coefficients.toarray() if hasattr(coefficients,'toarray') else np.asarray(coefficients)).reshape(len(frames),5,4)
        indices = indices.reshape(len(frames),5,4)
        global_to_active = np.full(len(g),-1,int);global_to_active[controls] = np.arange(len(controls))
        active_indices = global_to_active[indices]
        points = nodes.shape[2]
        pixel_scale = np.sqrt(dt/points)  # one pixel RMS transfer precision
        regularizer = vstack((kron(D2/12.,eye(3),format='csr'),
            kron(B*np.sqrt(dt)*.3/.75,eye(3),format='csr')),format='csr')
        row_index = np.broadcast_to(np.arange(len(frames)*points*2).reshape(len(frames),points,2,1,1,1),
                                   (len(frames),points,2,5,4,3)).ravel()
        column_index = np.broadcast_to((active_indices*3)[...,None]+np.arange(3),
                                      (len(frames),5,4,3))
        column_index = np.broadcast_to(column_index[:,None,None],(len(frames),points,2,5,4,3)).ravel()
        valid = column_index >= 0
        cached = [None,None]
        iterations = [0]
        emit('camera_root_transfer_start', parent=checked_entry(args.parameter),
             old_camera_sha256=old_info['state_sha256'],new_camera_sha256=camera.info['state_sha256'],
             affected_shots=shots,source_frames=len(frames),camera_changed_frames=int(changed.sum()),
             active_root_controls=len(controls),local_joints_frozen=True,shoe_controls_frozen=True,
             targets='saved parent projections; calibration transfer only, not source-image observations',
             initial_pixel_displacement=stats(np.linalg.norm(unadapted-target,axis=-1)),
             continuity=dict(existing_cut_policy=cut_policy,root_correction_acceleration_scale=12.,
                root_correction_prior_sigma=.75,root_correction_prior_weight=.3,
                root_rotation_acceleration_scale=80.,angular_transfer_precision_rad=angular_precision))

        def evaluate(value, jacobian=False):
            if cached[0] is not None and np.array_equal(value,cached[0]) and (not jacobian or cached[1][1] is not None):
                return cached[1]
            delta = (node_basis@value.reshape(-1,3)).reshape(len(frames),5,3)
            projection = moving_projection(camera,rotated_nodes+delta[:,:,None],frames,jacobian=jacobian)
            pixel = (projection['uv']-target).ravel()*pixel_scale
            prior = regularizer@value
            residual = np.r_[pixel,prior]
            J = None
            if jacobian:
                values = (projection['pixel_jac_nodes'][...,None,:]*coefficients[:,None,None,:,:,None]).ravel()*pixel_scale
                J = coo_matrix((values[valid],(row_index[valid],column_index[valid])),
                               shape=(len(pixel),len(value))).tocsr()
                J = vstack((J,regularizer),format='csr')
                iterations[0] += 1
                emit('camera_root_transfer_linearization',iteration=iterations[0],
                     cost=float(residual@residual/2),pixel_cost=float(pixel@pixel/2),continuity_cost=float(prior@prior/2))
            cached[:] = [value.copy(),(residual,J,projection['uv'])]
            return cached[1]

        zero = np.zeros(3*len(controls))
        initial = evaluate(zero,True)
        # Check the actual native implicit-row derivative before this new root
        # transfer path is used, independent of fitting the desired result.
        direction=np.random.default_rng(1117).normal(size=len(zero));direction/=np.linalg.norm(direction)
        epsilon=1e-5
        numerical=(evaluate(epsilon*direction)[0]-evaluate(-epsilon*direction)[0])/(2*epsilon)
        derivative_error=float(np.max(np.abs(numerical-initial[1]@direction)))
        emit('camera_root_derivative_check',maximum_absolute_error=derivative_error)
        if derivative_error>2e-5:
            raise AssertionError('Camera root transfer analytic derivative mismatch')
        result = least_squares(lambda x:evaluate(x)[0],zero,jac=lambda x:evaluate(x,True)[1],
            x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-6,tr_solver='lsmr',
            tr_options=dict(atol=1e-10,btol=1e-10))
        position = curve.position_controls.copy();position[controls] += result.x.reshape(-1,3)
        final = AbsoluteMotionCurve(curve.knots,position,rotations,curve.edge_reference_quaternion)
        final.check_continuous_edge_update(curve)
        np.testing.assert_array_equal(final.rotation_controls[:,1:],curve.rotation_controls[:,1:])
        new_document = copy.deepcopy(document)
        geometry = camera.info['scene_binding']
        new_document['programme_camera_candidate'] = dict(record=str(args.registration),
            record_sha256=digest(args.registration),parent_camera_sha256=old_info['state_sha256'],
            registered_programme=dict(path=str(args.registration),camera_sha256=camera.info['state_sha256'],
                                      scene_values_sha256=camera.info['scene_values_sha256']))
        new_document['scene_contract'] = dict(document['scene_contract'],geometry=geometry,
            registered_camera_sha256=camera.info['state_sha256'])
        source_path=work/'input/source_camera_1117.json'
        report=dict(schema='rinkspace.continuous_body_camera_transfer.v1',
            parent=checked_entry(args.parameter),parent_source=checked_entry(args.source),
            programme_registration=checked_entry(args.registration),
            old_camera=old_info,new_camera=camera.info,
            source_seconds=[float(ref.p['pts'][0]/12800),float(ref.p['pts'][-1]/12800)],
            source_time_base=[1,12800],source_frames=len(ref.times),affected_shots=shots,
            source_pts=ref.p['pts'][frames].tolist(),local_joints_and_relative_shoes_exactly_preserved=True,
            no_new_image_observations=True,new_visual_accuracy_claim=False,
            scope='camera calibration adaptation of current fitted pose; not a new accuracy claim',
            finite_continuous_root_transfer=True,shared_scale_unchanged=True,
            initial_transfer_error_px=stats(np.linalg.norm(unadapted-target,axis=-1)),
            final_transfer_error_px=stats(np.linalg.norm(evaluate(result.x)[2]-target,axis=-1)),
            solver=dict(status=int(result.status),message=result.message,optimality=float(result.optimality),
                        evaluations=int(result.nfev),cost=float(result.cost),
                        gradient_tolerance=1e-6,gradient_converged=bool(result.optimality<=1e-6)),
            derivative_maximum_error=derivative_error,user_visual_accepted=False)
        new_document['camera_root_transfer'] = dict(parent=report['parent'],
            parent_source=report['parent_source'],programme_registration=report['programme_registration'],
            source_evidence_rows_unchanged=True,calibration_transfer_targets_are_not_observations=True)
        write(source_path,new_document)
        saved=dict(packet,camera_binding_json=np.array(json.dumps(camera.info,sort_keys=True)),
            source_evidence_sha256=np.array(digest(source_path)),
            absolute_position_controls=position,absolute_rotation_controls=rotations,
            absolute_seed_position_controls=position,absolute_seed_rotation_controls=rotations,
            camera_root_transfer_json=np.array(json.dumps(report,sort_keys=True)))
        pending=destination.with_suffix('.pending.npz')
        with pending.open('xb') as stream:np.savez_compressed(stream,**saved)
        pending.replace(destination)
        report.update(candidate=checked_entry(destination),source_evidence=checked_entry(source_path))
        write(work/'source_review/camera_1117_review.json',report)
        emit('camera_root_transfer_complete',candidate=report['candidate'],
             transfer_error_before=report['initial_transfer_error_px'],transfer_error_after=report['final_transfer_error_px'],
             solver=report['solver'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','prepared','source','parameter','registration'):
        parser.add_argument('--'+name,type=lambda p:Path(p).resolve(),required=True)
    run(parser.parse_args())
