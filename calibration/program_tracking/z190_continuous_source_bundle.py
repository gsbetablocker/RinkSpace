"""Dense source-exposure bridge for the confirmed rinkside Z190 installations.

Geometry and observed pixels remain immutable. One installation has fixed Cz,
shared Cxy/PP/k1, and R3/effective logf per evidenced exposure. Constant lever,
axial, k2, tangential, skew and higher Brown terms are exactly zero. The pure
ray factors are consequently valid for this *conditional fixed-C branch*;
they do not establish that a physical pan head has no centre displacement.

StandingSourceBundle accepts the existing main/secondary/low-end Bundle data
contract. It retains the cached exposure rotations, chunked native pixel
factors, fixed CSR pattern and sparse analytic LM. Nothing switches production.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

PROGRAM = Path(__file__).resolve().parent
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import csr_matrix, vstack

from main7_source_camera_candidate import arrays, sha
from z190_candidate_entry import DEFAULT_CONFIG, load_standing_camera_candidate
from z190_physical_camera import CanonicalView, PhysicalCameraModel, ViewState, Z190StandingTripodHypothesis
from z190_source_camera_diagnostic import absolute_value_jac15, solve_with_crew_priors, source_errors, future_groups

HARD_SCENE_SHA = '93c654360712be798e65aa83e587415fa2693008ee320ccbc1972299ead293bd'
SCENE_UNIT_M = 0.31353729994266055
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'


def radial_inverse(distorted, k1):
    """Invert radial k1 on the central branch, with its implicit derivative.

    The cubic radial inverse is exact on its monotone branch. Negative k1
    images outside that branch are rejected; they never get a different root.
    """
    distorted = np.asarray(distorted, float)
    rd = np.linalg.norm(distorted, axis=1)
    if not np.isfinite(distorted).all() or not np.isfinite(k1):
        raise ValueError('Nonfinite source radial ray')
    if k1 > 0.:
        scale = np.sqrt(3.*k1)
        radius = 2./scale*np.sinh(np.arcsinh(1.5*scale*rd)/3.)
    elif k1 < 0.:
        scale = np.sqrt(-3.*k1)
        argument = 1.5*scale*rd
        if np.any(argument >= 1.):
            raise ValueError('Source radial ray has no central inverse')
        radius = 2./scale*np.sin(np.arcsin(argument)/3.)
    else:
        radius = rd
    factor = 1.+k1*radius*radius
    rays = distorted/factor[:, None]
    r2 = np.einsum('ni,ni->n', rays, rays)
    radial_eigen = 1.+3.*k1*r2
    if np.any(factor <= 1e-7) or np.any(radial_eigen <= 1e-7):
        raise ValueError('Source radial inverse leaves the central optical domain')
    inverse = np.broadcast_to(np.eye(2), (len(rays), 2, 2)).copy()/factor[:, None, None]
    inverse -= (2.*k1/(factor*radial_eigen))[:, None, None]*rays[:, :, None]*rays[:, None, :]
    return rays, inverse


def standing_relative_kernel(fa, fb, ra, rb, da, db, a, b, optics, *, jacobian=True):
    """Both-endpoint source ray transfer with shared PP fractions and k1.

    Optics=[cx/1920,cy/1080,k1]; view logf here is log(final effective px).
    Both inverse and forward Brown responses enter the shared derivatives.
    """
    a, b, optics = np.asarray(a, float), np.asarray(b, float), np.asarray(optics, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or optics.shape != (3,):
        raise ValueError('Expected actual native Nx2 pairs and shared PP/k1')
    fa, fb = np.broadcast_to(fa, (len(a),)), np.broadcast_to(fb, (len(a),))
    pp = optics[:2]*[1920., 1080.]
    distorted = (a-pp)/fa[:, None]
    local_xy, inverse = radial_inverse(distorted, optics[2])
    local = np.c_[local_xy, np.ones(len(a))]
    world_ray = np.einsum('ni,nij->nj', local, ra)
    target = np.einsum('ni,nji->nj', world_ray, rb)
    if not np.isfinite(target).all() or np.any(target[:, 2] <= .05):
        raise ValueError('Native ray left target forward domain')
    z = target[:, 2]
    xy = target[:, :2]/z[:, None]
    r2 = np.einsum('ni,ni->n', xy, xy)
    factor = 1.+optics[2]*r2
    if np.any(factor <= 1e-7) or np.any(1.+3.*optics[2]*r2 <= 1e-7):
        raise ValueError('Native ray left target central radial domain')
    projected = fb[:, None]*xy*factor[:, None]+pp
    residual = (projected-b).ravel()
    if not jacobian:
        return residual
    forward = factor[:, None, None]*np.eye(2)+2.*optics[2]*xy[:, :, None]*xy[:, None, :]
    normal = np.zeros((len(a), 2, 3))
    normal[:, 0, 0] = normal[:, 1, 1] = 1./z
    normal[:, :, 2] = -xy/z[:, None]
    d_uv = fb[:, None, None]*np.einsum('nij,njk->nik', forward, normal)
    ja, jb, shared = np.zeros((len(a), 2, 4)), np.zeros((len(a), 2, 4)), np.zeros((len(a), 2, 3))

    def source_chain(dlocal):
        dworld = np.einsum('ni,nij->nj', dlocal, ra)
        dtarget = np.einsum('ni,nji->nj', dworld, rb)
        return np.einsum('nij,nj->ni', d_uv, dtarget)

    for axis in range(3):
        dworld = np.einsum('ni,nij->nj', local, da[:, axis])
        dtarget = np.einsum('ni,nji->nj', dworld, rb)
        ja[:, :, axis] = np.einsum('nij,nj->ni', d_uv, dtarget)
        dtarget = np.einsum('ni,nji->nj', world_ray, db[:, axis])
        jb[:, :, axis] = np.einsum('nij,nj->ni', d_uv, dtarget)
    dxy = -np.einsum('nij,nj->ni', inverse, distorted)
    ja[:, :, 3] = source_chain(np.c_[dxy, np.zeros(len(a))])
    jb[:, :, 3] = fb[:, None]*xy*factor[:, None]
    for axis, scale in enumerate((1920., 1080.)):
        dxy = inverse[:, :, axis]*(-scale/fa[:, None])
        shared[:, :, axis] = source_chain(np.c_[dxy, np.zeros(len(a))])
        shared[:, axis, axis] += scale
    source_r2 = np.einsum('ni,ni->n', local_xy, local_xy)
    dxy = -np.einsum('nij,nj->ni', inverse, local_xy*source_r2[:, None])
    shared[:, :, 2] = source_chain(np.c_[dxy, np.zeros(len(a))])+fb[:, None]*xy*r2[:, None]
    return residual, ja.reshape(-1, 4), jb.reshape(-1, 4), shared.reshape(-1, 3)


class StandingSourceBundle:
    """Generic dense installation objective; prior rows stay separate from UV.

    Each bundle belongs to one physical installation. Its height is shared
    across that installation's own cuts only, with no equality to other cameras.

    Six ambient shared slots preserve the existing fixed-height LM contract:
    [Cx/100,Cy/100,fixed Cz/100,cx/W,cy/H,k1]. Only five are free. Per-exposure
    slots use log(f_px); the PhysicalCameraModel export converts to log(f/W).
    """
    def __init__(self, base, calibration, *, scene_unit_m=SCENE_UNIT_M, prior_wide_focal_px, fixed_height=True, fixed_height_scene_units=None):
        self.stamps, self.index = base.stamps, base.index
        self.a, self.b, self.ia, self.ib = base.a, base.b, base.ia, base.ib
        self.motion_sigma, self.motion_rows = base.motion_sigma, 2*len(base.a)
        self.abs_groups = base.abs_groups
        self.native_residual_sigma = base.native_residual_sigma
        self.source_residual_rows = len(self.native_residual_sigma)
        hypothesis = Z190StandingTripodHypothesis.from_calibration(calibration)
        self.crew_hypothesis = replace(hypothesis, scene_unit_m=scene_unit_m)
        settings = dict(calibration.field_settings)
        settings['working_hypothesis'] = self.crew_hypothesis.profile()
        self.crew_fixed_height = bool(fixed_height)
        if fixed_height_scene_units is not None and (not fixed_height or not np.isfinite(fixed_height_scene_units)):
            raise ValueError('An explicit frozen saved height requires fixed-height mode and a finite value')
        self.fixed_height_scene_units = self.crew_hypothesis.height_target_scene if fixed_height_scene_units is None else float(fixed_height_scene_units)
        settings['height_fit_policy'] = ('fixed_saved_physical_height' if fixed_height_scene_units is not None else
                                         'fixed_at_working_1_60m') if fixed_height else 'soft_gaussian_working_prior'
        if fixed_height:
            settings['height_fixed_scene_units'] = self.fixed_height_scene_units
        else:
            settings.pop('height_fixed_scene_units', None)
        self.calibration = replace(calibration, pivot_xyz=(*calibration.pivot_xyz[:2], self.fixed_height_scene_units), field_settings=settings)
        self.view_offset = 6
        self.prior_wide_focal_px = float(prior_wide_focal_px)
        if (calibration.output_mode.domain != 'final_effective' or calibration.output_mode.final_wh != (1920, 1080)
                or any(calibration.lever_xyz) or any(calibration.axial) or calibration.aspect_fx_over_fy != 1.
                or any(calibration.k2) or any(calibration.tangential_p1_p2) or calibration.allow_effective_pp_offset
                or len(calibration.principal_x_fraction) != 1 or len(calibration.principal_y_fraction) != 1 or len(calibration.k1) != 1):
            raise ValueError('Standing fixed-C source-ray bridge cannot represent this calibration')
        ia2, ib2 = np.repeat(self.ia, 2), np.repeat(self.ib, 2)
        self.swap_rows = ia2 > ib2
        lo, hi = np.minimum(ia2, ib2), np.maximum(ia2, ib2)
        columns = [np.c_[np.broadcast_to(np.arange(3, 6), (self.motion_rows, 3)),
                         6+4*lo[:, None]+np.arange(4), 6+4*hi[:, None]+np.arange(4)].astype(np.int32).ravel()]
        counts = [np.full(self.motion_rows, 11, np.int32)]
        for p, rows in self.abs_groups:
            count = 2*sum(len(row['uv']) for row in rows)
            cols = np.r_[np.arange(6), 6+4*self.index[p]+np.arange(4)]
            columns.append(np.broadcast_to(cols, (count, 10)).astype(np.int32).ravel())
            counts.append(np.full(count, 10, np.int32))
        self.indices = np.concatenate(columns)
        self.indptr = np.r_[0, np.cumsum(np.concatenate(counts), dtype=np.int64)].astype(np.int32)
        if len(self.indptr)-1 != self.source_residual_rows:
            raise ValueError('Existing source residual inventory changed')
        self.residual_rows = self.source_residual_rows

    def initial_shared(self):
        c = self.calibration
        return np.r_[np.asarray(c.pivot_xyz)/100., c.principal_x_fraction[0], c.principal_y_fraction[0], c.k1[0]]

    def current_calibration(self, q):
        height = self.fixed_height_scene_units if self.crew_fixed_height else 100.*q[2]
        return replace(self.calibration, pivot_xyz=(100.*q[0], 100.*q[1], height),
                       principal_x_fraction=(q[3],), principal_y_fraction=(q[4],), k1=(q[5],))

    def camera(self, q, p):
        start = self.view_offset+4*self.index[p]
        view = q[start:start+4]
        return self.current_calibration(q).camera25(ViewState(tuple(view[:3]), float(view[3]-np.log(1920.))))

    def priors(self, q, rotations=None):
        # Identical PP/distortion/physical roll assumptions as the sparse entry.
        if rotations is None:
            rotations = [cv2.Rodrigues(v[:3]) for v in q[6:].reshape(-1, 4)]
        values = [(q[3]-.5)*1920./32., (q[4]-.5)*1080./32.]
        data, columns, counts = [1920./32., 1080./32.], [3, 4], [1, 1]
        if not self.crew_fixed_height:
            height = self.crew_hypothesis
            values.append((100.*q[2]*height.scene_unit_m-height.height_target_m)/height.height_sigma_m)
            data.append(100.*height.scene_unit_m/height.height_sigma_m); columns.append(2); counts.append(1)
        for fraction in (.5, .75, 1.):
            radius = fraction*np.hypot(960., 540.)/self.prior_wide_focal_px
            derivative = self.prior_wide_focal_px*radius**3/8.
            values.append(q[5]*derivative); data.append(derivative); columns.append(5); counts.append(1)
        sigma = np.deg2rad(.5)
        for index, (rotation, raw) in enumerate(rotations):
            dr = raw.reshape(3, 3, 3)
            x, y = rotation[0, 2], -rotation[1, 2]
            norm2 = x*x+y*y
            if norm2 < 1e-8:
                raise ValueError('Physical horizon roll undefined')
            values.append(np.arctan2(x, y)/sigma)
            data.extend((y*dr[:, 0, 2]+x*dr[:, 1, 2])/norm2/sigma)
            columns.extend(6+4*index+np.arange(3)); counts.append(3)
        pointer = np.r_[0, np.cumsum(counts)]
        return np.asarray(values), csr_matrix((data, columns, pointer), shape=(len(values), len(q)))

    def evaluate(self, q, jacobian=True, include_priors=True):
        views = q[6:].reshape(-1, 4)
        rotations = [cv2.Rodrigues(v[:3]) for v in views]
        rot = np.array([r[0] for r in rotations])
        drot = np.array([r[1].reshape(3, 3, 3) for r in rotations]) if jacobian else None
        focal = np.exp(views[:, 3])
        values, data = [], []
        for lo in range(0, len(self.a), 65536):
            hi = min(lo+65536, len(self.a)); ia, ib = self.ia[lo:hi], self.ib[lo:hi]
            sigma = self.motion_sigma[2*lo:2*hi]
            result = standing_relative_kernel(focal[ia], focal[ib], rot[ia], rot[ib],
                        None if drot is None else drot[ia], None if drot is None else drot[ib],
                        self.a[lo:hi], self.b[lo:hi], q[3:6], jacobian=jacobian)
            if jacobian:
                residual, ja, jb, shared = result
                endpoints = np.c_[ja, jb]; swap = self.swap_rows[2*lo:2*hi]
                endpoints[swap] = np.c_[jb[swap], ja[swap]]
                data.append((np.c_[shared, endpoints]/sigma[:, None]).ravel())
            else:
                residual = result
            values.append(residual/sigma)
        for p, rows in self.abs_groups:
            result = absolute_value_jac15(self.camera(q, p), rows, jacobian)
            if jacobian:
                residual, raw = result
                # project_fast J already uses C/100 and PP/1000. Convert
                # only PP/1000 -> PP image fraction; do not condition twice.
                block = np.c_[raw[:, :2], np.zeros(len(raw)) if self.crew_fixed_height else raw[:, 2], raw[:, 8]*1.920, raw[:, 9]*1.080,
                              raw[:, 11], raw[:, 3:6], raw[:, 6]+raw[:, 7]]
                data.append(block.ravel())
            else:
                residual = result
            values.append(residual)
        result = np.concatenate(values)
        prior, prior_jac = self.priors(q, rotations) if include_priors else (np.empty(0), None)
        if not jacobian:
            return np.r_[result, prior]
        jac = csr_matrix((np.concatenate(data), self.indices, self.indptr), shape=(self.source_residual_rows, len(q)))
        return (np.r_[result, prior], vstack((jac, prior_jac), format='csr')) if len(prior) else (result, jac)


def check_source_derivatives(bundle, q):
    """Check real pixels at nonzero k1, endpoints and absolute Brown mapping."""
    checks = []
    # Check one whole actual pair, including every correspondence, not a frame
    # projection synthesized from the model. This avoids a second huge CSR.
    ia, ib = int(bundle.ia[0]), int(bundle.ib[0])
    selected = (bundle.ia == ia) & (bundle.ib == ib)
    a, b = bundle.a[selected], bundle.b[selected]
    first, second = q[6+4*ia:10+4*ia], q[6+4*ib:10+4*ib]
    def pair(local, jacobian=False):
        x, y, optics = local[:4], local[4:8], local[8:]
        ra, da = cv2.Rodrigues(x[:3]); rb, db = cv2.Rodrigues(y[:3])
        return standing_relative_kernel(np.exp(x[3]), np.exp(y[3]), np.broadcast_to(ra, (len(a), 3, 3)),
                   np.broadcast_to(rb, (len(a), 3, 3)), np.broadcast_to(da.reshape(3, 3, 3), (len(a), 3, 3, 3)),
                   np.broadcast_to(db.reshape(3, 3, 3), (len(a), 3, 3, 3)), a, b, optics, jacobian=jacobian)
    local = np.r_[first, second, q[3:6]]
    result = pair(local, True); analytic = np.c_[result[1], result[2], result[3]]
    for column in range(len(local)):
        h = 1e-6 if column in (3, 7) else 1e-7
        plus, minus = local.copy(), local.copy(); plus[column] += h; minus[column] -= h
        fd = (pair(plus)-pair(minus))/(2.*h)
        checks.append(dict(kind='actual_pair', source_pts=[bundle.stamps[ia], bundle.stamps[ib]], source_points=len(a), column=column,
                           relative_l2_difference=float(np.linalg.norm(fd-analytic[:, column])/max(1., np.linalg.norm(fd))),
                           maximum_difference=float(np.max(np.abs(fd-analytic[:, column])))))
    for p, rows in bundle.abs_groups[::max(1, len(bundle.abs_groups)//8)]:
        camera = bundle.camera(q, p)
        _, raw = absolute_value_jac15(camera, rows)
        derivatives = [raw[:, 0], raw[:, 1], raw[:, 8]*1.920, raw[:, 9]*1.080, raw[:, 11]]
        for column, derivative in zip((0, 1, 3, 4, 5), derivatives):
            h = 1e-7
            plus, minus = q.copy(), q.copy(); plus[column] += h; minus[column] -= h
            fd = (absolute_value_jac15(bundle.camera(plus, p), rows, False)-absolute_value_jac15(bundle.camera(minus, p), rows, False))/(2.*h)
            checks.append(dict(kind='actual_absolute', pts=p, column=column,
                               relative_l2_difference=float(np.linalg.norm(fd-derivative)/max(1., np.linalg.norm(fd))),
                               maximum_difference=float(np.max(np.abs(fd-derivative)))))
    if max(row['relative_l2_difference'] for row in checks) > 2e-4:
        raise ValueError('Actual source standing Brown analytic derivative mismatch: '+json.dumps(checks))
    return checks


def check_exposure_aliases(streams):
    """Exact lookup against frozen official pixel-match alias evidence."""
    import low_end_confirmed_exposure_aliases as proof
    path = proof.CADENCE/'exposure_groups.json'
    if sha(path) != proof.GROUPS_SHA256:
        raise ValueError('Frozen source exposure grouping changed')
    packet = json.loads(path.read_text(encoding='utf8'))
    groups = packet['groups']
    alias = {int(p): min(map(int, g['full_pts'])) for g in groups for p in g['full_pts']}
    for cut, stream in streams.items():
        for p, representative in zip(stream['pts'], stream['exposure_representative_pts']):
            if alias[int(p)] != int(representative):
                raise ValueError(f'Dense source alias differs from real official exposure proof: cut{cut}, PTS{p}')
    return dict(path=str(path), sha256=sha(path), exact_source_alias_comparison=True,
                frame_rate_based_deduplication=False, source_native_time_base='1/15360')


def run(a):
    from main_installation_source_bundle_candidate import prepare
    from continuous_central_brown4_optin import enable_projection_backend
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'input_manifest.json').exists():
        raise ValueError('Use a fresh diagnostic output directory')
    if sha(a.scene) != HARD_SCENE_SHA:
        raise ValueError('Dense Z190 bridge requires unchanged hard scene 93c654...')
    started = time.perf_counter()
    candidate = load_standing_camera_candidate(a.config)
    identifier = 'okada_sp_installation_0_conditional_z190'
    calibration = candidate.model.calibrations[identifier]
    scene, streams, canonical, cuts, rows, dense, bindings, suppressed, base, qbase = prepare(
        a.base, a.previous, a.reviewed_bridge_directory, a.scene, a.additional_fit_source)
    aliases = check_exposure_aliases(streams)
    bundle = StandingSourceBundle(base, calibration, prior_wide_focal_px=np.exp(np.min(qbase[6::4])))
    q0 = np.r_[bundle.initial_shared(), qbase[3:]]
    initializer = dict(dense_pose_seed='previous_same_hard_scene_source_candidate_R3_effective_logf_only',
                       dense_pose_seed_path=str(a.previous/'all_main_candidate_cameras.npz'),
                       dense_pose_seed_sha256=sha(a.previous/'all_main_candidate_cameras.npz'),
                       accepted_camera_values=False, source_observations_added=False, sparse_correction_used=False)
    if a.sparse_correction_seed:
        # The sparse correction supplies an initial guess only. Every emitted
        # dense result is subsequently optimized from real full-cut motion.
        previous = arrays(a.previous/'all_main_candidate_cameras.npz')
        old = {canonical[int(p)]: c for p, c in zip(previous['pts'], previous['camera25'])}
        count = 0
        for cut in streams:
            supported = sorted(p for p, v in candidate.canonical_views.items()
                               if v.installation_id == identifier and p in canonical and cuts[canonical[p]] == cut)
            if not supported:
                continue
            delta = np.array([np.r_[candidate.camera25(p)[3:6]-old[canonical[p]][3:6],
                                     np.mean(candidate.camera25(p)[6:8])-np.mean(old[canonical[p]][6:8])] for p in supported])
            targets = [p for p in bundle.stamps if cuts[p] == cut]
            for index, p in enumerate(targets):
                start = 6+4*bundle.index[p]
                q0[start:start+4] += [np.interp(p, supported, delta[:, component]) for component in range(4)]
            count += len(supported)
        initializer.update(sparse_correction_used=True, sparse_correction_fit_keys=count,
                           correction_role='same-cut interpolated numerical initialization only; never delivered as recovered source pose')
    backend = enable_projection_backend()
    manifest = dict(schema='rinkspace.z190-dense-standing-source-input.v1', accuracy_accepted=False,
        scene_path=str(a.scene), scene_sha256=HARD_SCENE_SHA, scene_frozen=True,
        native_scene_unit_m=SCENE_UNIT_M, metric_scale_policy='uniform_user_latest_scale_no_geometry_change',
        source_pixels_unchanged=True, source_bindings=bindings, source_exposure_proof=aliases,
        latest_sparse_entry=dict(path=str(a.config), sha256=sha(a.config), accuracy_accepted=False),
        canonical_exposures=len(bundle.stamps), native_frames=len(canonical), matches=len(bundle.a),
        actual_free_unknowns=5+4*len(bundle.stamps), ambient_fixed_height_slot=2,
        shared_free_variables=['C_x_scene_div100','C_y_scene_div100','principal_x_fraction','principal_y_fraction','residual_k1'],
        fixed_Cz_scene=bundle.crew_hypothesis.height_target_scene, fixed_Cz_m=bundle.crew_hypothesis.height_target_m,
        fixed_terms=['lever','axial','k2','tangential','skew','higher_Brown','fx_over_fy_minus1'],
        per_exposure_variables=['world_to_camera_Rodrigues3','log_final_effective_focal_px'],
        export_per_exposure_focal='log_final_effective_focal_px_div1920',
        source_residual_rows=bundle.source_residual_rows, priors_not_source_observations=True,
        cut_counts={str(n):dict(native=len(s['pts']),actual_exposures=len(set(s['exposure_representative_pts']))) for n,s in streams.items()},
        initializer=initializer, sparse_pattern='fixed sorted CSR; exposure-cached Rodrigues; 65536-source-match blocks',
        prior_policy='same sparse standing PP32px, wide seed radial displacement8px, physical horizon roll0.5deg; height fixed',
        model_motion_is_conditional_not_hardware_identified=True, source_projection_backend=backend,
        preparation_seconds=time.perf_counter()-started, run_command_argv=[sys.executable, '-X', 'utf8', *sys.argv])
    (a.output/'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='prepared_dense_standing_main', canonical_exposures=len(bundle.stamps), matches=len(bundle.a),
                          fixed_Cz_scene=manifest['fixed_Cz_scene'], seconds=manifest['preparation_seconds'])), flush=True)
    check = check_source_derivatives(bundle, q0)
    (a.output/'actual_source_derivative_check.json').write_text(json.dumps(check, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='actual_source_derivative_check_passed', maximum_relative_difference=max(r['relative_l2_difference'] for r in check))), flush=True)
    if a.prepare_only:
        return manifest
    q, history, seconds = solve_with_crew_priors(bundle, q0, a.output)
    if sha(a.scene) != HARD_SCENE_SHA:
        raise ValueError('Frozen hard scene changed during source-only solve')
    cameras = {p:bundle.camera(q,p) for p in bundle.stamps}
    for cut, stream in streams.items():
        np.savez_compressed(a.output/f'main{cut}_candidate_cameras.npz', pts=stream['pts'],
            camera25=np.array([cameras[canonical[int(p)]] for p in stream['pts']]),
            exposure_representative_pts=stream['exposure_representative_pts'],
            status=np.full(len(stream['pts']), 'dense_standing_source_camera_candidate_unvalidated'))
    pts = np.sort(np.concatenate([s['pts'] for s in streams.values()]))
    np.savez_compressed(a.output/'all_main_candidate_cameras.npz', pts=pts,
        camera25=np.array([cameras[canonical[int(p)]] for p in pts]), exposure_representative_pts=np.array([canonical[int(p)] for p in pts]),
        shared_standing_state=q[:6])
    current = bundle.current_calibration(q)
    model = PhysicalCameraModel({identifier:current})
    views = {p:CanonicalView(identifier, ViewState(tuple(q[6+4*bundle.index[p]:9+4*bundle.index[p]]),
                                  float(q[9+4*bundle.index[p]]-np.log(1920.)))) for p in bundle.stamps}
    model.save_npz(a.output/'physical_dense_main_candidate_unaccepted.npz', pts, canonical, views,
                   source_sha256=SOURCE_SHA, time_base='1/15360')
    report = dict(schema='rinkspace.z190-dense-standing-source-report.v1', accuracy_accepted=False,
        fixed_scene=True, scene_sha256=HARD_SCENE_SHA, seconds=seconds, history=history,
        shared_C_scene=current.pivot_xyz, shared_C_m=(np.asarray(current.pivot_xyz)*SCENE_UNIT_M).tolist(),
        principal_px=[q[3]*1920.,q[4]*1080.], shared_k1=float(q[5]), shared_k2=0.,
        source_absolute_errors=[dict(native_pts=r['pts'], cut=cuts[canonical[r['pts']]], id=r['id'], origin=r['origin'], split=r['split'],
            errors=source_errors(cameras[canonical[r['pts']]],[r])) for r in rows])
    future, future_binding = future_groups(a.base, scene, a.previous/'all_main_candidate_cameras.npz')
    report['untouched_future_source_checks'] = dict(bindings=future_binding, sparse_pose_transfer_used=False,
        rows=[dict(pts=key['pts'],cut=key['cut'],errors=source_errors(cameras[canonical[key['pts']]],key['rows'])) for key in future])
    (a.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='dense_standing_main_complete', seconds=seconds, C=current.pivot_xyz,
                         principal=report['principal_px'], k1=report['shared_k1'], accuracy_accepted=False)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','previous','scene','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--reviewed-bridge-directory', type=Path)
    parser.add_argument('--additional-fit-source', type=Path, action='append', default=[])
    parser.add_argument('--sparse-correction-seed', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
