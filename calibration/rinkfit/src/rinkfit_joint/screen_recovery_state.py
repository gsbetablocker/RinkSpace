"""One hash-bound projection provider for registration, ROI and preview.

This is an image-plane recovery state, never a native physical checkpoint.
S01 uses the unchanged complete d78 forward model. S02--S09 consume the exact
compact H state in its own explicit chart. An ice H cannot navigate board XYZ.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .compact_screen_seed import ScreenSeed, object_sha
from .io_utils import file_sha256, read_json, write_json

SCHEMA = 'rinkfit.screen_recovery_state.v1'


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path))


def verified(ref, base=None):
    path = Path(ref.get('path', ref.get('file', '')))
    if not path.is_absolute(): path = Path(base) / path
    path = path.resolve()
    if file_sha256(path) != ref['sha256']:
        raise ValueError('Screen dependency changed: ' + str(path))
    return path


def initialize(seed_file, s01_complete_state, output):
    """Create only small references, with no physical-model fallback."""
    seed = ScreenSeed(seed_file)
    complete = Path(s01_complete_state).resolve()
    doc = read_json(complete)
    if file_sha256(complete) != seed.meta['s01']['complete_auxiliary_state_ref']['sha256']:
        raise ValueError('S01 complete state differs from the seed-bound d78 baseline')
    if doc['model']['sha256'] != seed.meta['s01']['projection_model']['sha256']:
        raise ValueError('S01 complete state model mismatch')
    verified(doc['model']); verified(doc['parent_complete_checkpoint'])
    for ref in doc['row_state_files'].values(): verified(ref)
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError('New immutable screen-state generation required')
    out.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, state_kind='image_plane_recovery',
        compact_seed=reference(seed_file), s01_complete_state=reference(complete),
        source_video_sha256=seed.meta['source_video_sha256'],
        time_base=[1, 12800], native_step_pts=256, image_size=[1920, 1080],
        s02_s09_world_chart_sha256=seed.world_chart_sha256,
        corrections={}, source_generations=[],
        initial_state='exact_compact_H_and_complete_d78',
        physical_checkpoint_initialized=False, source_UV_created_from_projection=False,
        s01_projection_dependencies_frozen=True,
        consumer_entry='rinkfit_joint.screen_registration',
        implementation={n: reference(Path(__file__).with_name(n)) for n in
            ('screen_recovery_state.py', 'compact_screen_seed.py', 'screen_registration.py', 'screen_tracklet_registration.py')})
    write_json(out / 'state.json', manifest)
    state = ScreenRecoveryState(out / 'state.json')
    report = check_consumers(state)
    write_json(out / 'consumer_check.json', report)
    return dict(state=reference(out/'state.json'), consumer_check=reference(out/'consumer_check.json'), **report)


class ScreenRecoveryState:
    def __init__(self, manifest_file):
        self.path = Path(manifest_file).resolve()
        self.manifest = read_json(self.path)
        if self.manifest.get('schema') != SCHEMA: raise ValueError('Unsupported screen recovery state')
        self.sha256 = file_sha256(self.path)
        self.seed = ScreenSeed(verified(self.manifest['compact_seed'], self.path.parent))
        if self.manifest['source_video_sha256'] != self.seed.meta['source_video_sha256']:
            raise ValueError('Screen source-video identity mismatch')
        if self.manifest.get('s02_s09_world_chart_sha256') != self.seed.world_chart_sha256:
            raise ValueError('Reference/current template chart confusion')
        complete = read_json(verified(self.manifest['s01_complete_state'], self.path.parent))
        if complete['model']['sha256'] != self.seed.meta['s01']['projection_model']['sha256']:
            raise ValueError('S01 complete baseline changed')
        self.s01_complete_state = complete
        model_file = verified(self.seed.meta['s01']['projection_model'], self.seed.path.parent)
        from rinkfit_start.native_scene import camera_from_event
        self.s01_model = read_json(model_file)
        self.s01_camera = camera_from_event(self.s01_model)
        self.s01_chart_sha256 = object_sha(dict(kind='d78_world_L1', projection_model_sha256=file_sha256(model_file)))
        self.world_chart_sha256 = self.seed.world_chart_sha256
        self.corrections = self.manifest.get('corrections', {})
        if 's01' in self.corrections: raise ValueError('S01 cannot acquire image-plane correction controls')
        verified_tracklets = set()
        for sid, correction in self.corrections.items():
            if sid not in self.seed.meta['per_shot']: raise ValueError('Unknown shot correction')
            segments = correction.get('segments', [correction])
            previous = None
            for segment in segments:
                pts = np.asarray(segment['native_pts'], dtype=np.int64)
                delta = np.asarray(segment['delta'], dtype=float)
                if len(pts) < 2 or np.any(np.diff(pts) <= 0) or delta.shape != (len(pts), 8) or not np.isfinite(delta).all():
                    raise ValueError('Invalid supported screen correction basis')
                for stamp in pts: self.seed.frame_index(int(stamp), sid)
                if previous is not None and pts[0] <= previous:
                    raise ValueError('Overlapping source-supported screen correction intervals')
                previous = pts[-1]
                if not segment.get('source_tracklets') or segment.get('basis') != 'linear_within_source_supported_interval':
                    raise ValueError('Corrections require real source support and a declared temporal basis')
                for ref in segment['source_tracklets']:
                    key = (ref.get('path', ref.get('file')), ref['sha256'])
                    if key not in verified_tracklets:
                        verified(ref, self.path.parent); verified_tracklets.add(key)
        self._s01_inventory = None

    def shot_id(self, native_pts):
        if isinstance(native_pts, bool) or not isinstance(native_pts, (int, np.integer)):
            raise ValueError('Native PTS must be an integer')
        scope = self.seed.meta['s01']['scope']
        if scope['first_pts'] <= native_pts <= scope['last_pts']:
            if (native_pts-scope['first_pts']) % 256: raise ValueError('Off-native PTS')
            return 's01'
        i = int(np.searchsorted(self.seed.pts, native_pts))
        if i == len(self.seed.pts) or int(self.seed.pts[i]) != native_pts: raise ValueError('Missing native exposure')
        return f's{int(self.seed.shots[i]):02d}'

    def delta(self, native_pts, shot_id):
        if self.shot_id(native_pts) != shot_id: raise ValueError('Shot/PTS mismatch')
        control = self.corrections.get(shot_id)
        if control is None: return np.zeros(8)
        for segment in control.get('segments', [control]):
            pts = np.asarray(segment['native_pts'], np.int64)
            if pts[0] <= native_pts <= pts[-1]:
                d = np.asarray(segment['delta'], float)
                return np.array([np.interp(native_pts, pts, d[:, j]) for j in range(8)])
        # No fabricated precision across source/role gaps. This is the exact
        # inherited seed there; the motion report accounts these as unknown.
        return np.zeros(8)

    def correction_status(self, native_pts, shot_id):
        if self.shot_id(native_pts) != shot_id: raise ValueError('Shot/PTS mismatch')
        control = self.corrections.get(shot_id)
        if control:
            for segment in control.get('segments', [control]):
                if segment['native_pts'][0] <= native_pts <= segment['native_pts'][-1]:
                    return 'source_supported_correction'
        return 'preserved_d78' if shot_id == 's01' else 'inherited_screen_seed_motion_unknown'

    def matrix(self, native_pts, shot_id):
        if shot_id == 's01': raise ValueError('Full Brown/RS d78 forward has no substitute ice H')
        return self.seed.matrix(native_pts, shot_id, self.delta(native_pts, shot_id))

    def project_ice(self, native_pts, shot_id, xy, *, world_chart_sha256):
        if shot_id == 's01':
            if world_chart_sha256 != self.s01_chart_sha256: raise ValueError('S01 requires its own d78 chart')
            xy = np.asarray(xy, float)
            return self.project_s01_world(native_pts, np.c_[xy, np.zeros(len(xy))])
        return self.seed.project_reference_xy(native_pts, shot_id, xy,
            world_chart_sha256=world_chart_sha256, delta=self.delta(native_pts, shot_id))

    def project_s01_world(self, native_pts, xyz):
        if self.shot_id(native_pts) != 's01': raise ValueError('S01 forward called outside S01')
        result = self.s01_camera.project_rs(np.asarray(xyz, float),
            (native_pts-self.seed.meta['s01']['time_origin_pts'])/12800.)
        return result['uv'], result['valid']

    def inverse_ice(self, native_pts, shot_id, uv):
        if shot_id == 's01': raise ValueError('S01 Brown/RS inverse needs native ray/plane solver; no H fallback')
        return self.seed.inverse_native_uv(native_pts, shot_id, uv, delta=self.delta(native_pts, shot_id))

    def navigation_roi(self, native_pts, shot_id, xy, *, world_chart_sha256, margin_px):
        """Ice search rectangle only; never certifies visibility or source UV."""
        if not np.isfinite(margin_px) or margin_px < 0: raise ValueError('Explicit nonnegative search margin required')
        uv, valid = self.project_ice(native_pts, shot_id, xy, world_chart_sha256=world_chart_sha256)
        points = uv[valid]
        if not len(points): return dict(roi_xyxy=None, status='no_valid_navigation', state_sha256=self.sha256)
        lo = np.maximum([0, 0], np.floor(points.min(axis=0)-margin_px)).astype(int)
        hi = np.minimum([1920, 1080], np.ceil(points.max(axis=0)+margin_px+1)).astype(int)
        roi = [*lo.tolist(), *hi.tolist()] if np.all(hi > lo) else None
        return dict(roi_xyxy=roi, status='ice_search_domain_only' if roi else 'outside_sensor',
            state_sha256=self.sha256, world_chart_sha256=world_chart_sha256,
            source_UV_created=False, visibility_verified=False)

    def project_inventory(self, native_pts, shot_id):
        if self.shot_id(native_pts) != shot_id: raise ValueError('Wrong preview shot')
        if shot_id != 's01':
            return self.seed.project_inventory(native_pts, shot_id, delta=self.delta(native_pts, shot_id))
        if self._s01_inventory is None:
            from rinkfit_start.native_scene import NativeScene
            scene = NativeScene(self.s01_model)
            paths = [scene.world(mi, pi, np.linspace(0, 1, 101))
                for mi, material in enumerate(self.s01_model['materials'])
                for pi, _ in enumerate(material['paths'])]
            self._s01_inventory = (np.concatenate(paths), np.r_[0, np.cumsum([len(x) for x in paths])], np.ones(len(paths), np.uint8))
        world, offsets, kinds = self._s01_inventory
        uv, valid = self.project_s01_world(native_pts, world)
        return uv, valid, offsets, kinds

    def draw(self, native_bgr, native_pts, shot_id):
        """Preview uses the exact provider consumed by optimization and ROI."""
        import cv2
        if native_bgr.shape != (1080, 1920, 3) or native_bgr.dtype != np.uint8:
            raise ValueError('Original 1920x1080 BGR required')
        uv, valid, offsets, kinds = self.project_inventory(native_pts, shot_id)
        image = native_bgr.copy(); drawn = 0
        for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            color = (85, 220, 65) if kinds[i] == 0 else (220, 55, 205)
            for j in range(a, b-1):
                if not (valid[j] and valid[j+1]): continue
                # Clip each valid segment and never join separate paths/gaps.
                q = uv[j:j+2]
                if np.max(np.abs(q)) > np.iinfo(np.int32).max/4: continue
                ok, x, y = cv2.clipLine((0, 0, 1920, 1080), tuple(np.rint(q[0]).astype(int)), tuple(np.rint(q[1]).astype(int)))
                if ok: cv2.line(image, x, y, color, 1, cv2.LINE_AA); drawn += 1
        cv2.putText(image, f'{shot_id} PTS={native_pts} SCREEN {self.sha256[:12]}', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2, cv2.LINE_AA)
        return image, dict(state_sha256=self.sha256, shot_id=shot_id, native_pts=native_pts,
            valid_samples=int(valid.sum()), drawn_segments=drawn, physical_checkpoint_claimed=False)


def check_consumers(state):
    """Small connection check; do not re-render the audited 188m vertices."""
    from .screen_registration import RegistrationProblem
    maximum = 0.; inverse = 0.; equal = True; count = 0
    checked = []
    for sid in state.seed.meta['per_shot']:
        ids = np.flatnonzero(state.seed.shots == int(sid[1:]))
        # Check actual production registration/ROI/preview on shot endpoints
        # and one interior exposure, plus every original H byte at zero delta.
        for index in ids:
            pts = int(state.seed.pts[index])
            equal &= np.array_equal(state.matrix(pts, sid), state.seed.matrix(pts, sid, state.delta(pts, sid)))
        for index in np.unique([ids[0], ids[len(ids)//2], ids[-1]]):
            pts = int(state.seed.pts[index]); xy = state.seed.a['verification_world_xy'][index]
            target = state.seed.a['verification_target_uv'][index] if not state.corrections else state.seed.project_reference_xy(pts,sid,xy,world_chart_sha256=state.world_chart_sha256,delta=state.delta(pts,sid))[0]
            problem = RegistrationProblem(state, sid, [dict(native_pts=pts,
                reference_xy=xy.tolist(), source_uv=target.tolist())], world_chart_sha256=state.world_chart_sha256)
            pred = problem.predict(np.zeros(8))
            maximum = max(maximum, float(np.max(np.linalg.norm(pred-target, axis=1))))
            back, valid = state.inverse_ice(pts, sid, pred)
            if not valid.all(): raise ValueError('Consumer inverse invalid')
            inverse = max(inverse, float(np.max(np.abs(back-xy))))
            roi = state.navigation_roi(pts, sid, xy, world_chart_sha256=state.world_chart_sha256, margin_px=0.)
            invuv, good, _, _ = state.project_inventory(pts, sid)
            if not np.isfinite(invuv[good]).all(): raise ValueError('Preview consumer nonfinite')
            if roi['state_sha256'] != problem.state.sha256: raise ValueError('ROI/registration state differs')
            count += len(xy); checked.append([sid, pts])
    # Full d78 function, including its original head/optics/RS/time and all
    # scene dependencies, is evaluated at three native S01 exposures.
    from rinkfit_start.native_scene import camera_from_event
    independent = camera_from_event(state.s01_model)
    s01max = 0.; s01valid = True
    world = np.array([[0.,0.,0.], [.1,.1,0.], [-.1,-.1,0.], [.2,0.,.01]])
    for pts in [256256, 415744, 575488]:
        a, valid = state.project_s01_world(pts, world)
        ref = independent.project_rs(world, (pts-256256)/12800.)
        s01max = max(s01max, float(np.max(np.abs(a-ref['uv']))))
        s01valid &= bool(np.array_equal(valid, ref['valid']))
    if not equal or maximum > 1e-8 or s01max != 0 or not s01valid: raise ValueError('Screen consumer transfer failed')
    return dict(schema='rinkfit.screen_recovery_consumer_check.v1',
        actual_consumer='screen_registration.RegistrationProblem.predict',
        state_sha256=state.sha256, checked_exposures=checked,
        all_7186_current_H_bytes_equal_to_seed_plus_declared_correction=equal,
        all_7186_H_bytes_equal_at_zero=True if not state.corrections else None, actual_registration_points=count,
        maximum_registration_transfer_error_px=maximum, maximum_inverse_error_L=inverse,
        s01_full_forward_max_abs_difference_px=s01max, s01_validity_equal=s01valid,
        ROI_preview_optimizer_use_same_provider=True, giant_cache_rebuilt=False,
        source_truth_accuracy_certified=False, physical_checkpoint_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('initialize'); p.add_argument('--seed', required=True); p.add_argument('--s01-complete-state', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('check'); p.add_argument('--state', required=True)
    a = parser.parse_args()
    result = initialize(a.seed, a.s01_complete_state, a.out) if a.command == 'initialize' else check_consumers(ScreenRecoveryState(a.state))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
