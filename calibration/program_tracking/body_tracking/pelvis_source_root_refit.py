"""Refine the existing programme root with explicit pelvis source priority.

This reuses the complete body spline, saved articulation, qualified pixels and
50 Hz exposure evidence. The root remains SMPL-X joint0, not the COCO hip mean.
The two original qualified hip factors receive a declared positive gain.
Opening mask containment requires source evidence of a visible pelvis; a
predicted root entering the image does not supply that evidence. Separately
reviewed clothed-hip means remain weak conditional proxies, not joint0 pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr

from . import fullprogramme_body_sequence as sequence
from .fullprogramme_camera_rebind import checked_camera, recompose_root_prior
from .camera_bridge import project_world
from ..render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE, DEFAULT_SHOTS
from rinkspace.metric_scale import build_uniform_scale_manifest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def binding(path):
    with Path(path).open('rb') as stream:
        token = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(Path(path).resolve()), sha256=token)


class OpeningRootMaskSupport:
    """Existing selected SAM silhouette, in its original half-resolution grid.

    No inferred off-image part is added. A source with cropped/unqualified hips
    cannot constrain root visibility just because its fitted UV is in frame.
    A supported root is compared with the visible silhouette, never with a
    detected or synthesized anatomical centre.
    """
    def __init__(self, packet):
        masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
        opening = ((packet['shot_index'] == 1)
                   & packet['body_opening_temporal_inference_only']
                   & (packet['actual_pts'] < packet['opening_near_static_user_source_end_pts']))
        # These are original source gates, independent of fitted P/camera.
        # A partial clothed-hip proxy does not establish joint0 visibility.
        source_visible = (packet['source_COCO17_qualified'][:, [11, 12]].all(axis=1)
                          & ~packet['source_original_truncated'])
        active = opening & source_visible
        self.source_visibility_excluded_rows = np.flatnonzero(opening & ~source_visible).tolist()
        self.contours = {}
        self.references = {}
        self.failures = []
        packed_path, packed = None, None
        for i in np.flatnonzero(active):
            pts = int(packet['source_native_pts'][i])
            if pts not in masks.lookup:
                self.failures.append(dict(row=int(i), pts=pts, error='source mask unavailable'))
                continue
            path, index, status = masks.lookup[pts]
            if status != 'measured_mask':
                self.failures.append(dict(row=int(i), pts=pts, error=status))
                continue
            if path != packed_path:
                with np.load(path, allow_pickle=False) as z:
                    packed = z['masks'].copy()
                packed_path = path
            mask = np.unpackbits(packed[index], axis=1)[:, :960].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.failures.append(dict(row=int(i), pts=pts, error='empty measured mask'))
                continue
            # The tracked programme person is the largest source-mask component.
            contour = max(contours, key=cv2.contourArea)
            source_hips = packet['source_COCO17_native_uv'][i, 11:13] / 2.
            if not all(cv2.pointPolygonTest(contour, tuple(q), True) >= 0.
                       for q in source_hips):
                self.failures.append(dict(row=int(i), pts=pts,
                    error='source hips not contained by selected person component'))
                continue
            self.contours[int(i)] = cv2.approxPolyDP(contour, .25, True).astype(np.float32)
            self.references[str(path)] = binding(path)

    def value_and_pixel_gradient(self, row, uv):
        contour = self.contours.get(int(row))
        uv = np.asarray(uv, float)
        if (contour is None or not np.isfinite(uv).all()
                or np.any(uv < 0.) or np.any(uv >= [1920., 1080.])):
            return 0., np.zeros(2)
        point = uv / 2.
        distance = cv2.pointPolygonTest(contour, tuple(point), True) * 2.
        if distance >= 0.:
            return 0., np.zeros(2)
        # Signed Euclidean distance is differentiable away from nearest-edge
        # ties. Source-polygon corners retain their ordinary piecewise gradient.
        gradient = np.asarray([
            -(cv2.pointPolygonTest(contour, tuple(point + d), True)
              - cv2.pointPolygonTest(contour, tuple(point - d), True))
            for d in (np.array([.25, 0.]), np.array([0., .25]))]) * 2.
        return -distance, gradient


class ReviewedClothedHipProxies:
    """Retained raw COCO hip means supported by source-only clothing review.

    The four original mean candidates can condition inferred pelvis geometry.
    Individual hip qualification and original truncation gates remain intact.
    Neither a clothing rectangle nor its centre becomes an observed root UV.
    """
    def __init__(self, packet, authority_path=None):
        self.rows = {}
        self.references = {}
        if authority_path is None:
            return
        authority = read(authority_path)
        source = read(DEFAULT_SHOTS)
        if (authority.get('schema') != 'rinkspace.opening-clothed-pelvis-source-candidates.v1'
                or authority.get('authority') != 'actual_source_visibility_review_of_retained_coco_candidates'
                or authority.get('source_video_sha256') != source['source_sha256_from_timeline']
                or authority.get('source_time_base') != source['source_time_base']):
            raise ValueError('Clothed hip proxies require matching actual source-only review')
        self.references[str(authority_path)] = binding(authority_path)
        for record in authority['records']:
            if not (record.get('conditional_source_proxy_candidate_valid')
                    and record.get('limited_source_qualification_restore')
                    and record.get('candidate_reviewable')
                    and record.get('independently_measured_pelvis_uv') is None
                    and record.get('exact_joint0_measured') is False):
                raise ValueError('Only reviewed conditional original hip means may be fitted')
            pts = int(record['source_observation_pts'])
            matches = np.flatnonzero(packet['source_native_pts'] == pts)
            if len(matches) != 1:
                raise ValueError('Reviewed proxy source exposure is missing or duplicated')
            i = int(matches[0])
            uv = np.asarray(record['raw_source_coco11_12_uv'], float)
            if (packet['shot_index'][i] != 1
                    or not packet['body_opening_temporal_inference_only'][i]
                    or not packet['source_opening_reviewed_hip_only_override'][i]
                    or not packet['source_hip_proxy_valid'][i]
                    or not np.array_equal(packet['source_COCO17_native_uv'][i, 11:13], uv)
                    or not np.isfinite(uv).all()):
                raise ValueError('Clothed proxy differs from retained original source pixels/gates')
            mean = uv.mean(axis=0)
            roi = np.asarray(record['source_only_manually_reviewed_visible_clothing_region_xyxy'], float)
            if not (np.all(mean >= roi[:2]) and np.all(mean <= roi[2:])
                    and np.all(mean >= 0.) and np.all(mean < [1920., 1080.])):
                raise ValueError('Conditional proxy is not on the reviewed visible clothing')
            evidence = record['source_evidence']
            actual = binding(evidence['path'])
            if actual['sha256'] != evidence['sha256']:
                raise ValueError('Reviewed original source image changed')
            self.references[actual['path']] = actual
            if i in self.rows:
                raise ValueError('Duplicate conditional proxy exposure')
            self.rows[i] = dict(source_native_pts=pts, uv=mean,
                original_hip_uv=uv, source_visibility=record['source_visibility'])

    def specification(self):
        return dict(bindings=list(self.references.values()),
            source_rows=[dict(row=i, source_native_pts=r['source_native_pts'],
                original_hip_native_uv=r['original_hip_uv'].tolist(),
                original_hipmean_proxy_native_uv=r['uv'].tolist(),
                source_visibility=r['source_visibility']) for i, r in sorted(self.rows.items())],
            operator='Mean of two projected model COCO hips minus retained original raw COCO hip mean',
            individual_original_hip_qualification_unchanged=True,
            inferred_local_body_geometry_condition=True,
            independently_measured_joint0_pixel=False)


class PelvisSourceRootProblem(sequence.FullProgrammeSplineProblem):
    """The original root graph plus explicit source-driven pelvis factors."""
    def __init__(self, packet, *, support, warm_body, hip_gain=2., mask_sigma_px=6.,
                 clothed_proxies=None, clothed_proxy_sigma_px=48., conditional_face_gain=1., **kwargs):
        source_gain = np.ones((len(packet['actual_pts']), len(sequence.SOURCE_INDICES)))
        source_gain[:, np.isin(sequence.SOURCE_INDICES, [11, 12])] = float(hip_gain)
        opening = ((packet['shot_index'] == 1) & packet['body_opening_temporal_inference_only']
                   & (packet['actual_pts'] < packet['opening_near_static_user_source_end_pts']))
        # A borrowed local face cannot represent the real head turning while
        # its torso bends. Keep all original face UV/gates, but declare their
        # optional weaker influence on an inferred coarse torso/root.
        source_gain[np.ix_(opening, np.isin(sequence.SOURCE_INDICES, np.arange(5)))] = float(conditional_face_gain)
        super().__init__(packet, source_factor_gain=source_gain, **kwargs)
        self.support = support
        self.mask_sigma_px = float(mask_sigma_px)
        self.hip_gain = float(hip_gain)
        self.clothed_proxies = clothed_proxies or ReviewedClothedHipProxies(packet)
        self.clothed_proxy_sigma_px = float(clothed_proxy_sigma_px)
        self.conditional_face_gain = float(conditional_face_gain)
        weights = np.asarray([self.hip_gain, self.mask_sigma_px,
                              self.clothed_proxy_sigma_px, self.conditional_face_gain])
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Declared positive pelvis/mask engineering weights required')
        # Retain current camera-local P/Q only as a numerical warm start. The
        # replacement camera defines world coordinates; no old world root is a
        # measurement and no orientation/articulation network runs here.
        old_r = np.stack([cv2.Rodrigues(c[3:6])[0] for c in warm_body['camera25']])
        old_unit = float(warm_body['metres_per_scene_unit'])
        local_p = np.einsum('nij,nj->ni', old_r,
                           warm_body['P_world_m'] - warm_body['camera25'][:, :3] * old_unit)
        initial_p = self.C + np.einsum('nji,nj->ni', self.R, local_p)
        initial_q = np.einsum('nji,njk,nkl->nil', self.R, old_r, warm_body['Q_world'])
        correction = initial_q @ np.transpose(self.qprior, (0, 2, 1))
        delta = np.stack([cv2.Rodrigues(q)[0].ravel() for q in correction])
        self.x0 = np.r_[np.column_stack([
            lsqr(self.B, initial_p[:, j], atol=1e-9, btol=1e-9)[0] for j in range(3)]).ravel(),
            np.column_stack([lsqr(self.Bq, delta[:, j], atol=1e-9, btol=1e-9)[0]
                             for j in range(3)]).ravel()]
        self._pelvis_cached_x = None
        self._pelvis_cached = None

    def infer_unobserved_position_control_warm_start(self):
        return dict(current_body_camera_local_root_used_only_as_numerical_initialization=True,
                    new_camera_world_transform_recomputed=True,
                    no_old_world_root_measurement=True)

    def evaluate(self, x):
        if self._pelvis_cached_x is not None and np.array_equal(x, self._pelvis_cached_x):
            return self._pelvis_cached
        result = super().evaluate(x)
        residual, jacobian, P = result[:3]
        values, rows, columns, extra = [], [], [], []
        for k, i in enumerate(sorted(self.support.contours)):
            campt = self.R[i] @ (P[i] - self.C[i])
            root_uv, j = cv2.projectPoints(campt[None], np.zeros(3), np.zeros(3),
                                          self.K[i], self.cameras[i, 11:25])
            raw, pixel_gradient = self.support.value_and_pixel_gradient(i, root_uv[0, 0])
            extra.append(raw / self.mask_sigma_px)
            pixel_world = j[:, 3:6] @ self.R[i]
            derivative = pixel_gradient @ pixel_world / self.mask_sigma_px
            cp = (self.bcols[i, :, None] * 3 + np.arange(3)).ravel()
            rows.extend([k] * 12)
            columns.extend(cp.tolist())
            values.extend((self.bvals[i, :, None] * derivative[None]).ravel().tolist())
        delta = self.Bq @ np.asarray(x)[self.m * 3:].reshape(self.qm, 3)
        for i, proxy in sorted(self.clothed_proxies.rows.items()):
            correction, dcorrection = cv2.Rodrigues(delta[i])
            q = correction @ self.qprior[i]
            dq = np.einsum('rjk,kl->rjl', dcorrection.reshape(3, 3, 3), self.qprior[i])
            local = self.local[i, 11:13]
            world = local @ q.T + P[i]
            campts = (world - self.C[i]) @ self.R[i].T
            uv, j = cv2.projectPoints(campts, np.zeros(3), np.zeros(3),
                                     self.K[i], self.cameras[i, 11:25])
            jp = np.einsum('npk,kj->npj', j[:, 3:6].reshape(2, 2, 3), self.R[i])
            dw = np.einsum('rjk,nk->njr', dq, local)
            jq = np.einsum('npj,njr->npr', jp, dw)
            jp, jq = jp.mean(axis=0) / self.clothed_proxy_sigma_px, jq.mean(axis=0) / self.clothed_proxy_sigma_px
            row0 = len(extra)
            extra.extend(((uv.reshape(2, 2).mean(axis=0) - proxy['uv'])
                          / self.clothed_proxy_sigma_px).tolist())
            rr = np.repeat(np.arange(row0, row0 + 2), 12)
            cp = (self.bcols[i, :, None] * 3 + np.arange(3)).ravel()
            cq = (self.qcols[i, :, None] * 3 + np.arange(3) + self.m * 3).ravel()
            rows.extend(np.r_[rr, rr].tolist())
            columns.extend(np.r_[np.tile(cp, 2), np.tile(cq, 2)].tolist())
            values.extend(np.r_[(jp[:, None, :] * self.bvals[i, None, :, None]).ravel(),
                               (jq[:, None, :] * self.qvals[i, None, :, None]).ravel()].tolist())
        block = coo_matrix((values, (rows, columns)), shape=(len(extra), len(x))).tocsr()
        self._pelvis_cached_x = np.asarray(x).copy()
        self._pelvis_cached = (np.r_[residual, extra], vstack([jacobian, block]).tocsr(), *result[2:])
        return self._pelvis_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-body-directory', type=Path, required=True)
    parser.add_argument('--camera', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hip-gain', type=float, default=2.)
    parser.add_argument('--mask-sigma-px', type=float, default=6.)
    parser.add_argument('--reviewed-clothed-hip-proxies', type=Path)
    parser.add_argument('--clothed-proxy-sigma-px', type=float, default=48.)
    parser.add_argument('--conditional-face-gain', type=float, default=1.)
    parser.add_argument('--max-nfev', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing body evidence must be retained')
    args.output.mkdir(parents=True)
    source_snapshot = args.output / 'pelvis_source_root_refit.used.py'
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    previous = args.previous_body_directory
    packet = arrays(previous / 'fullprogramme_body_source_input.npz')
    warm = arrays(previous / 'actual_body_root_fit.npz')
    preparation = read(previous / 'camera_rebind_prepare.json')
    report = read(previous / 'report.json')
    if binding(previous / 'fullprogramme_body_source_input.npz')['sha256'] != report['input']['sha256']:
        raise ValueError('Previous source packet changed')
    if binding(previous / 'actual_body_root_fit.npz')['sha256'] != report['output']['sha256']:
        raise ValueError('Previous full body root changed')
    camera, actual_rows, _ = checked_camera(args.camera, packet)
    scale = build_uniform_scale_manifest(args.scene,
        Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json'),
        scale_id='pelvis-current-scene-uniform-reference')
    old_packet = {k: v.copy() for k, v in packet.items()}
    raw = arrays(preparation['official_pose_clock_authority']['path'])
    old_native = arrays(preparation['replacement_camera']['path'])
    replay = recompose_root_prior(packet, old_native['camera25'],
        raw['raw_orientation_source_pts'], raw['raw_incam_global_orient'])
    if np.max(np.abs(replay - packet['raw_root_prior_Q_world'])) > 2e-13:
        raise ValueError('Effective saved raw-root endpoints do not replay')
    packet['camera25'] = camera['camera25'][actual_rows]
    packet['raw_root_prior_Q_world'] = recompose_root_prior(packet, camera['camera25'],
        raw['raw_orientation_source_pts'], raw['raw_incam_global_orient'])
    packet['metres_per_scene_unit'] = np.asarray(scale['metres_per_scene_unit'])
    old_unit = float(old_packet['metres_per_scene_unit'])
    for i, (old, new) in enumerate(zip(old_packet['camera25'], packet['camera25'])):
        local_initial = cv2.Rodrigues(old[3:6])[0] @ (
            old_packet['initial_P_m'][i] - old[:3] * old_unit)
        packet['initial_P_m'][i] = (new[:3] * float(packet['metres_per_scene_unit'])
                                  + cv2.Rodrigues(new[3:6])[0].T @ local_initial)
    allowed = {'camera25', 'raw_root_prior_Q_world', 'initial_P_m', 'metres_per_scene_unit'}
    preserved = {k: np.array_equal(old_packet[k], packet[k], equal_nan=True)
                 if old_packet[k].dtype.kind in 'fc' else np.array_equal(old_packet[k], packet[k])
                 for k in packet if k not in allowed}
    if not all(preserved.values()):
        raise ValueError('Root source, local articulation or source clock changed')
    np.savez_compressed(args.output / 'fullprogramme_body_source_input.npz', **packet)
    (args.output / 'uniform_scale_v2.json').write_text(json.dumps(scale, ensure_ascii=False, indent=2), encoding='utf-8')
    support = OpeningRootMaskSupport(packet)
    clothed_proxies = ReviewedClothedHipProxies(packet, args.reviewed_clothed_hip_proxies)
    specification = dict(schema='rinkspace.pelvis-priority-native-root-refit.v1',
        previous_body=binding(previous / 'actual_body_root_fit.npz'),
        previous_source_packet=binding(previous / 'fullprogramme_body_source_input.npz'),
        camera=binding(args.camera), scene=binding(args.scene),
        scale=binding(args.output / 'uniform_scale_v2.json'),
        preserved_source_local_pose_clock_fields=preserved,
        pelvis_qualified_original_hip_residual_gain=args.hip_gain,
        gain_is_root_priority_engineering_weight_not_independent_new_evidence=True,
        opening_source_mask_containment_sigma_native_px=args.mask_sigma_px,
        mask_is_necessary_containment_not_anatomical_root_measurement=True,
        source_mask_containment_rows=sorted(support.contours),
        source_mask_visibility_policy='Original qualified source hips and no original truncation; predicted root UV never establishes pelvis visibility',
        source_mask_visibility_excluded_rows=support.source_visibility_excluded_rows,
        source_mask_bindings=list(support.references.values()), failures=support.failures,
        reviewed_clothed_hipmean_proxies=clothed_proxies.specification(),
        clothed_hipmean_proxy_sigma_native_px=args.clothed_proxy_sigma_px,
        clothed_proxy_sigma_is_declared_weak_engineering_uncertainty_not_measured_noise=True,
        inferred_opening_face_source_factor_gain=args.conditional_face_gain,
        conditional_face_gain_is_local_geometry_uncertainty_not_source_identity_rejection=True,
        root_world_vertical_projection_only=True, blade_contact_factor=False,
        new_pose_or_articulation=False, accuracy_accepted=False)
    (args.output / 'pelvis_refit_inputs.json').write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding='utf-8')
    next_fit = dict(preparation['next_root_fit'])
    for k in ('module', 'note'):
        next_fit.pop(k, None)
    next_fit.update(camera=str(args.camera), output=str(args.output),
        body_samples=report['complete_body_samples_binding']['path'], warm_controls=None,
        max_nfev=args.max_nfev, opening_planar_binding='soft-velocity',
        opening_source_support='conditional-upper-body', pose_inference_body_samples=None)
    current_preparation = dict(preparation)
    current_preparation.update(replacement_camera=binding(args.camera),
        output_prepared_input=binding(args.output / 'fullprogramme_body_source_input.npz'),
        previous_root_refit_directory=str(previous.resolve()),
        scale_source=binding(args.output / 'uniform_scale_v2.json'),
        metres_per_scene_unit=float(packet['metres_per_scene_unit']))
    current_preparation['next_root_fit'] = dict(next_fit)
    (args.output / 'camera_rebind_prepare.json').write_text(
        json.dumps(current_preparation, ensure_ascii=False, indent=2), encoding='utf-8')
    # This process-local dispatch preserves the original fit/export code and
    # its output schema; no persistent solver/source package is changed.
    base_problem = sequence.FullProgrammeSplineProblem
    try:
        sequence.FullProgrammeSplineProblem = lambda p, **kw: PelvisSourceRootProblem(
            p, support=support, warm_body=warm, hip_gain=args.hip_gain,
            mask_sigma_px=args.mask_sigma_px, clothed_proxies=clothed_proxies,
            clothed_proxy_sigma_px=args.clothed_proxy_sigma_px,
            conditional_face_gain=args.conditional_face_gain, **kw)
        sequence.run(SimpleNamespace(**next_fit))
    finally:
        sequence.FullProgrammeSplineProblem = base_problem
    final_report = read(args.output / 'report.json')
    final_report['pelvis_source_root_refinement'] = specification
    final_report['additional_code'] = binding(source_snapshot)
    (args.output / 'report.json').write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
