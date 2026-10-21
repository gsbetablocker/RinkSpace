"""Source line supported wide Brown candidate; all fitted UV remain original.

Only reviewed finite static ice line entities are used. Their global template
station is unnecessary for the plumb-line operator. This offline candidate has
frozen radial extension coefficients; it is not a hardware calibration or a
replacement for full shared source fitting.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .io_utils import read_json, write_json, file_sha256
from .joint_model import as_event_view
from .screen_to_native_seed import _stats
from rinkfit_start.native_camera import rotation
from rinkfit_start.native_scene import camera_from_event


def _inverse_brown(target, coefficients):
    """Newton inverse on the positive-Jacobian branch; failed trials stay failed."""
    xy = target.copy()
    old_error = np.inf
    while True:
        x, y = xy.T
        k1, k2, p1, p2 = coefficients.T
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2
        pred = np.c_[x*radial+2*p1*x*y+p2*(r2+2*x*x),
                     y*radial+p1*(r2+2*y*y)+2*p2*x*y]
        residual = pred-target
        error = np.max(abs(residual))
        if error < 2e-13:
            return xy
        if not np.isfinite(error) or error >= old_error:
            raise ValueError('Brown inverse trial did not converge on its retained branch')
        old_error = error
        slope = k1+2*k2*r2
        a = radial+2*x*x*slope+2*p1*y+6*p2*x
        b = 2*x*y*slope+2*p1*x+2*p2*y
        d = radial+2*y*y*slope+6*p1*y+2*p2*x
        determinant = a*d-b*b
        if np.any(a <= 0) or np.any(determinant <= 0):
            raise ValueError('Brown inverse trial crossed a local fold')
        delta = np.c_[(d*residual[:, 0]-b*residual[:, 1])/determinant,
                      (a*residual[:, 1]-b*residual[:, 0])/determinant]
        xy -= delta


def _camera_geometry(camera, t):
    q = camera.spline(t)
    alpha, beta, logf = q.T
    z = logf-np.log(camera.optics['f_ref'])
    R = rotation(camera.b, beta)@camera.R0@rotation(camera.n, alpha)
    Rt = np.transpose(R, (0, 2, 1))
    normal = np.cross(camera.n, np.einsum('nij,j->ni', Rt, camera.b))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    lever = np.broadcast_to(camera.d, (len(t), 3)).copy()
    lever[:, 2] += camera.head.get('pupil_linear', 0.)*z+camera.head.get('pupil_quadratic', 0.)*z*z
    C = camera.P+np.einsum('nij,nj->ni', Rt, lever)+camera.a*normal
    return R, C, logf, z


def build_wide_candidate(model_file, reviewed_lines_file, output, *, shot_id='s07', fit_tangential=False):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable optical candidate generation')
    model = read_json(model_file)
    review = read_json(reviewed_lines_file)
    if review['source_video_sha256'] != model['source_clock']['source_video_sha256'] or not review['source_UV_unchanged']:
        raise ValueError('Reviewed source identity or original UV changed')
    reading_path = Path(review['source_reading']['path'])
    if reading_path.is_file():
        reading_sha = file_sha256(reading_path)
        reading_archive = None
    elif Path(str(reading_path)+'.gz').is_file():
        # The current task's large profile JSON may be losslessly compressed.
        # Preserve its original byte identity and explicitly bind the archive.
        archive = Path(str(reading_path)+'.gz')
        digest = hashlib.sha256()
        with gzip.open(archive, 'rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20), b''):
                digest.update(chunk)
        reading_sha = digest.hexdigest()
        reading_archive = dict(path=str(archive), sha256=file_sha256(archive),
                               original_path=str(reading_path), uncompressed_sha256=reading_sha)
    else:
        raise FileNotFoundError('Reviewed source reading and its lossless gzip archive are absent')
    if reading_sha != review['source_reading']['sha256']:
        raise ValueError('Source reading changed after review')
    sid = shot_id
    rid = model['shots'][sid]['rig_id']
    if rid != 'main':
        raise ValueError('This candidate preserves the confirmed main S01 interval')
    camera = camera_from_event(as_event_view(model, sid))
    angles = model['blocks'][model['shots'][sid]['reference_angles_block']]['values']
    camera.R0 = rotation(camera.b, angles[1])@camera.R0@rotation(camera.n, angles[0])
    first = camera_from_event(as_event_view(model, 's01'))
    boundary = float(first.controls[:, 2].min())
    source_groups = []
    for entity in review['confirmed_finite_source_entities']:
        if not entity['physical_surface_verified'] or entity['surface'] != 'ice' or not entity['straight_finite_source_entity']:
            raise ValueError('Plumb-line fitting cannot admit an unverified static surface')
        if len(entity['rows']) < 3:
            raise ValueError('A finite line needs three source readings to constrain curvature')
        uv = np.asarray([r['current_uv'] for r in entity['rows']], float)
        if any(r.get('model_UV_used_as_observation') for r in entity['rows']):
            raise ValueError('Projected or interpolated UV cannot be an optical observation')
        source_groups.append((entity, uv))
    pts = int(review['native_pts'])
    frame_t = (pts-model['shots'][sid]['time_origin_pts'])/12800.
    R0, C0, focal0, _ = _camera_geometry(camera, np.array([frame_t]))
    all_uv = np.concatenate([uv for _, uv in source_groups])
    times = frame_t+camera.tau*(all_uv[:, 1]/1079.-.5)
    R, C, logf, z = _camera_geometry(camera, times)
    eta = camera.optics.get('eta', 0.)
    focal = np.c_[np.exp(logf+eta/2), np.exp(logf-eta/2)]
    pp = np.asarray(camera.optics['principal_point'])+z[:, None]*np.asarray(camera.optics['principal_linear'])
    distorted = (all_uv-pp)/focal
    basis = np.maximum(boundary-logf, 0.)**2
    base = np.c_[np.polynomial.polynomial.polyval(z, camera.optics['k1']),
                 np.polynomial.polynomial.polyval(z, camera.optics['k2']),
                 np.full(len(z), camera.optics['p1']), np.full(len(z), camera.optics['p2'])]
    offsets = np.r_[0, np.cumsum([len(uv) for _, uv in source_groups])]

    def corrected(extension):
        coefficients = base+basis[:, None]*extension
        xy = _inverse_brown(distorted, coefficients)
        direction = np.einsum('nji,nj->ni', R, np.c_[xy, np.ones(len(xy))])
        distance = -C[:, 2]/direction[:, 2]
        if np.any(distance <= 0):
            raise ValueError('Static ice plumb-line ray has no positive plane intersection')
        world = C+distance[:, None]*direction
        central = (world-C0[0])@R0[0].T
        if np.any(central[:, 2] <= 0):
            raise ValueError('Plumb-line trial has nonpositive central depth')
        return central[:, :2]/central[:, 2:3]*np.exp(focal0[0])

    def line_residual(points):
        result = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            q = points[lo:hi]
            centered = q-q.mean(0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            normal = Vt[-1]
            if normal[1] < 0:
                normal = -normal
            result.append(centered@normal)
        return result

    def residual(parameters):
        extension = parameters if fit_tangential else np.r_[parameters, 0., 0.]
        try:
            pieces = line_residual(corrected(extension))
        except ValueError:
            return np.full(len(all_uv), 1e6)
        # Equal physical entities avoid counting dense correlated line samples
        # as independent camera evidence. This is an initialization criterion;
        # no calibrated covariance is claimed for these optical candidates.
        return np.concatenate([r/np.sqrt(len(r)) for r in pieces])

    initial = -base[:, :2].mean(0)/basis.mean()
    if fit_tangential:
        initial = np.r_[initial, 0., 0.]
    fit = least_squares(residual, initial, ftol=1e-11, xtol=1e-11, gtol=1e-9)
    extension = fit.x if fit_tangential else np.r_[fit.x, 0., 0.]
    after_points = corrected(extension)
    after = line_residual(after_points)
    # Keep the all-source denominator even if the old Brown inverse is invalid.
    try:
        before = line_residual(corrected(np.zeros(4)))
        old_inverse_failed = False
    except ValueError:
        before = [np.full(len(uv), np.nan) for _, uv in source_groups]
        old_inverse_failed = True
    pinhole = line_residual(corrected(np.r_[-base[:, :2].mean(0)/basis.mean(), 0., 0.]))
    support = dict(path=str(Path(reviewed_lines_file).resolve()), sha256=file_sha256(reviewed_lines_file),
                   source_RGB_sha256=review['source_RGB_sha256'], native_pts=pts,
                   source_entity_ids=[e['source_entity_id'] for e, _ in source_groups])
    if reading_archive is not None:
        support['source_reading_archive'] = reading_archive
    metadata = dict(schema='rinkfit.source_supported_wide_brown_hinge.v1', power=2,
                    activation_log_f_native=boundary, coefficients=extension.tolist(),
                    activation_focal_native_px=float(np.exp(boundary)),
                    preserved_interval='entire S01 allocated cubic log-f hull and RS halo',
                    source_support=[support], calibration_status=('candidate_frozen_offline_radial_and_tangential_fit' if fit_tangential else 'candidate_frozen_offline_radial_fit'),
                    hardware_calibration_claimed=False, tangential_extension_identified=False,
                    tangential_extension_fitted=bool(fit_tangential),
                    outside_observed_wide_neighborhood='unverified extrapolation; domain audit required')
    candidate = copy.deepcopy(model)
    candidate['rigs'][rid]['camera']['brown_wide_extension'] = metadata
    # Declare the actual native sensor working rays for the new candidate,
    # rather than enlarge a disk by an arbitrary hand-picked radius. This is a
    # mathematical working domain, not calibration evidence for every ray.
    sensor_corner_radii = []
    for source_sid, shot in candidate['shots'].items():
        if shot['rig_id'] != rid:
            continue
        cc = camera_from_event(as_event_view(candidate, source_sid))
        stamps = np.arange(shot['scope']['first_pts'], shot['scope']['last_pts']+1, 256)
        frame_times = (stamps-shot['time_origin_pts'])/12800.
        corners = np.broadcast_to(np.array([[0.,0.],[1919.,0.],[0.,1079.],[1919.,1079.]]),
                                  (len(stamps),4,2)).reshape(-1,2)
        tt = np.repeat(frame_times,4)+cc.tau*(corners[:,1]/1079.-.5)
        qq = cc.spline(tt)
        lf = qq[:,2]
        zz = lf-np.log(cc.optics['f_ref'])
        principal = np.asarray(cc.optics['principal_point'])+zz[:,None]*np.asarray(cc.optics['principal_linear'])
        fxy = np.c_[np.exp(lf+cc.optics['eta']/2),np.exp(lf-cc.optics['eta']/2)]
        bb = np.maximum(boundary-lf,0.)**2
        coefficients = np.c_[np.polynomial.polynomial.polyval(zz,cc.optics['k1']),
                              np.polynomial.polynomial.polyval(zz,cc.optics['k2']),
                              np.full(len(lf),cc.optics['p1']),np.full(len(lf),cc.optics['p2'])]+bb[:,None]*extension
        rays = _inverse_brown((corners-principal)/fxy,coefficients)
        sensor_corner_radii.extend(np.linalg.norm(rays,axis=1).tolist())
    working_radius = float(np.nextafter(max(sensor_corner_radii),np.inf))
    candidate['rigs'][rid]['camera']['radius_max'] = working_radius
    candidate['wide_optical_candidate'] = dict(input_model_sha256=file_sha256(model_file), accepted=False,
                                               source_support=support, main_source_refinement_required=True,
                                               working_ray_radius=dict(value=working_radius,
                                                   previous=model['rigs'][rid]['camera']['radius_max'],
                                                   method='inverse_new_Brown_at_four_native_sensor_corners_all_main_frames_and_corner_row_times',
                                                   working_domain_only_not_source_calibration_certificate=True))
    summaries = []
    for (entity, uv), b, a, p in zip(source_groups, before, after, pinhole):
        summaries.append(dict(source_entity_id=entity['source_entity_id'], source_rows=len(uv),
                               global_template_identity_verified=entity['global_template_instance_correspondence_verified'],
                               old_Brown_RS_compensated_plumbline_error_px=_stats(abs(b)),
                               provisional_radial_cancel_plumbline_error_px=_stats(abs(p)),
                               candidate_Brown_RS_compensated_plumbline_error_px=_stats(abs(a))))
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    out.mkdir(parents=True)
    write_json(out/'model.json', candidate)
    np.savez_compressed(out/'plumbline.npz', source_uv=all_uv, source_entity_offsets=offsets,
                        corrected_candidate_pixels=after_points, native_pts=np.full(len(all_uv), pts))
    report = dict(schema='rinkfit.source_wide_optical_candidate_fit.v1',
                   model=dict(path=str(out/'model.json'), sha256=file_sha256(out/'model.json')),
                   metadata=metadata, entities=summaries, fitted_extension=extension.tolist(),
                   local_Brown_at_frame=(base.mean(0)+basis.mean()*extension).tolist(),
                   source_row_focal_range_px=[float(np.exp(logf).min()), float(np.exp(logf).max())],
                   Jacobian_singular_values=singular.tolist(), optimizer_success=bool(fit.success),
                   old_inverse_branch_failed=old_inverse_failed, source_UV_changed=False,
                   robust_loss_used=False, rows_discarded_from_reviewed_set=0,
                   RS_compensation='provisional native main head at observed source sensor row; ice plane only',
                   source_optical_precision_calibrated=False, source_fit_accepted=False,
                   working_ray_radius=candidate['wide_optical_candidate']['working_ray_radius'],
                   whole_nine_shot_domain_and_S01_regression_required=True)
    write_json(out/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source-lines', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(build_wide_candidate(a.model, a.source_lines, a.out)))


if __name__ == '__main__':
    main()
