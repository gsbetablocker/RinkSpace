"""Source-image registration whose numerical initial state is ScreenRecoveryState.

The physical solver is deliberately not called by this entry. Registration
consumes independent native source UV in the seed's explicit ice-plane chart;
ROI navigation and preview open exactly the same state.json. There are no
virtual four-point penalties and no fallback to a historical physical camera.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .io_utils import file_sha256, read_json, write_json


class RegistrationProblem:
    """Constant eight-coordinate SL(3) increment of the actual input state.

A caller may use this local image registration at a source-supported keypose.
Persisting it as a motion correction additionally requires the temporal support
contract checked by register_points. Source UV never become target grid probes.
"""
    def __init__(self, state, shot_id, rows, *, world_chart_sha256):
        if shot_id == 's01': raise ValueError('S01 complete baseline is frozen')
        if world_chart_sha256 != state.world_chart_sha256: raise ValueError('Explicit reference chart required')
        self.state = state; self.shot_id = shot_id; self.rows = list(rows)
        self.xy = []; self.target = []; self.matrices = []; self.front = []
        for row in self.rows:
            pts = row['native_pts']; i = state.seed.frame_index(pts, shot_id)
            xy = np.asarray(row['reference_xy'], float).reshape(-1, 2)
            uv = np.asarray(row['source_uv'], float).reshape(-1, 2)
            if xy.shape != uv.shape or not len(xy) or not np.isfinite(xy).all() or not np.isfinite(uv).all():
                raise ValueError('Registration needs finite same-size chart points and native source UV')
            self.xy.append(xy); self.target.append(uv)
            self.matrices.append(state.matrix(pts, shot_id)); self.front.append(float(state.seed.a['front_sign'][i]))
        if not self.xy: raise ValueError('No registration source rows')
        self.target_uv = np.concatenate(self.target)
        self.N = np.array([[2/1920., 0., -1.], [0., 2/1080., -1.], [0., 0., 1.]])

    def predict(self, delta):
        delta = np.asarray(delta, float)
        if delta.shape != (8,) or not np.isfinite(delta).all(): raise ValueError('Eight finite correction coordinates required')
        if np.any(delta):
            from scipy.linalg import expm
            d = delta; A = np.array([[d[0],d[1],d[2]], [d[3],d[4],d[5]], [d[6],d[7],-d[0]-d[4]]])
            T = np.linalg.solve(self.N, expm(A) @ self.N)
        else: T = None
        result = []
        for xy, H, front in zip(self.xy, self.matrices, self.front):
            # Do not even multiply by I at zero: the existing H is the state.
            h = H if T is None else T @ H
            q = np.c_[xy, np.ones(len(xy))] @ h.T
            if np.any(front*q[:,2] <= 0): raise ValueError('Trial leaves the source projective branch')
            uv = q[:,:2]/q[:,2:3]
            if not np.isfinite(uv).all(): raise ValueError('Trial projection nonfinite')
            result.append(uv)
        return np.concatenate(result)

    def residual(self, delta):
        return (self.predict(delta)-self.target_uv).ravel()


def register_points(state_file, measurements_file, output):
    """Fit a real source-supported common screen increment, preserving sources.

The manifest must supply a connected static tracklet temporal support covering
its shot and fixed chart coordinates with independent physical identity. For
unanchored texture tracks use fit_tracklets instead of declaring chart truth.
"""
    from scipy.optimize import least_squares
    from .screen_recovery_state import ScreenRecoveryState, reference
    state = ScreenRecoveryState(state_file); doc = read_json(measurements_file)
    if doc.get('source_video_sha256') != state.manifest['source_video_sha256']:
        raise ValueError('Registration source-video identity mismatch')
    if doc.get('surface_class') != 'ice' or doc.get('static_surface_verified') is not True:
        raise ValueError('Only source-verified static ice points belong in the ice H')
    if doc.get('chart_points_independently_anchored') is not True:
        raise ValueError('H-inverted navigation is not an independently known chart point; use tracklet latent variables')
    source_refs = doc.get('source_tracklets')
    if not source_refs: raise ValueError('Persistent correction requires actual source tracklet evidence')
    from .screen_recovery_state import verified
    for ref in source_refs: verified(ref, Path(measurements_file).parent)
    sid = doc['shot_id']; rows = doc['rows']
    if sid in state.corrections: raise ValueError('Rebuild point correction from the inherited state and complete new source set')
    for row in rows:
        if row.get('current_pixels_measured') is not True or not row.get('decoded_rgb_sha256'):
            raise ValueError('Every fit exposure requires current source-pixel provenance')
        if row.get('role') != 'development': raise ValueError('Protected source pixels cannot enter registration')
        if not state.seed.a['navigation_role_allowed'][state.seed.frame_index(row['native_pts'], sid)]:
            raise ValueError('Protected/navigation-disabled native exposure')
    problem = RegistrationProblem(state, sid, rows, world_chart_sha256=doc['world_chart_sha256'])
    initial = problem.residual(np.zeros(8))
    result = least_squares(problem.residual, np.zeros(8), method='trf', x_scale='jac')
    singular = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(result.jac))
    if rank < 8: raise ValueError('Source geometry does not identify an eight-coordinate screen increment')
    final = problem.residual(result.x)
    if not result.success or np.dot(final, final) >= np.dot(initial, initial):
        raise ValueError('No source-supported improvement; input state retained')
    indices = np.flatnonzero(state.seed.shots == int(sid[1:]))
    first, last = int(state.seed.pts[indices[0]]), int(state.seed.pts[indices[-1]])
    support = np.asarray(doc.get('supported_adjacent_native_pts', []), dtype=np.int64)
    if not np.array_equal(np.unique(support), np.arange(first, last, 256)):
        raise ValueError('Whole-shot constant correction lacks continuous actual source-motion support')
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError('New immutable registration generation required')
    out.mkdir(parents=True)
    manifest = dict(state.manifest)
    manifest['parent_screen_state'] = reference(state_file)
    manifest['corrections'] = dict(state.corrections)
    manifest['corrections'][sid] = dict(basis='linear_within_source_supported_interval',
        native_pts=[first, last], delta=[result.x.tolist(), result.x.tolist()],
        source_tracklets=source_refs, source_measurements=reference(measurements_file))
    manifest['source_generations'] = list(manifest.get('source_generations', [])) + [reference(measurements_file)]
    write_json(out/'state.json', manifest)
    candidate = ScreenRecoveryState(out/'state.json')
    # Saved state and actual optimizer must produce the same source predictions.
    saved = RegistrationProblem(candidate, sid, rows, world_chart_sha256=state.world_chart_sha256).predict(np.zeros(8))
    transfer = float(np.max(np.abs(saved-problem.predict(result.x))))
    if transfer > 1e-8: raise ValueError('Serialized registration candidate lost the optimizer state')
    report = dict(schema='rinkfit.actual_screen_registration.v1', parent_state=reference(state_file),
        candidate_state=reference(out/'state.json'), source_measurements=reference(measurements_file),
        source_uv_sha256=__import__('hashlib').sha256(problem.target_uv.astype('<f8').tobytes()).hexdigest(),
        optimizer_initial_state_sha256=state.sha256, initialized_from_physical_966=False,
        virtual_grid_residuals=0, real_source_points=len(problem.target_uv),
        initial_rms_px=float(np.sqrt(np.mean(initial.reshape(-1,2)**2)*2)),
        final_rms_px=float(np.sqrt(np.mean(final.reshape(-1,2)**2)*2)),
        correction=result.x.tolist(), jacobian_rank=rank, singular_values=singular.tolist(),
        serialized_candidate_transfer_max_abs_px=transfer, physical_checkpoint_initialized=False)
    write_json(out/'registration.json', report)
    return report


def preview(state_file, video_file, native_pts, output):
    """Decode exactly one native exposure for same-state comparison/ROI review."""
    from fractions import Fraction
    from .screen_recovery_state import ScreenRecoveryState, reference
    import av, cv2
    state = ScreenRecoveryState(state_file)
    if file_sha256(video_file) != state.manifest['source_video_sha256']: raise ValueError('Wrong native source video')
    sid = state.shot_id(native_pts); raw = None
    with av.open(str(video_file)) as source:
        stream = source.streams.video[0]
        if Fraction(stream.time_base) != Fraction(1,12800): raise ValueError('Native source clock mismatch')
        source.seek(native_pts, stream=stream, backward=True, any_frame=False)
        for frame in source.decode(stream):
            if frame.pts == native_pts: raw = frame.to_ndarray(format='bgr24'); break
            if frame.pts > native_pts: break
    if raw is None: raise ValueError('Exact native source exposure unavailable')
    image, report = state.draw(raw, native_pts, sid)
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError('Immutable comparison preview required')
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image): raise OSError('Cannot save native comparison preview')
    report.update(state=reference(state_file), preview=reference(out), source_video=reference(video_file),
        source_seconds=native_pts/12800., all_projections_from_same_provider=True,
        screen_state_is_physical_checkpoint=False)
    write_json(out.with_suffix('.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit'); p.add_argument('--state', required=True); p.add_argument('--measurements', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('fit-tracks'); p.add_argument('--state', required=True); p.add_argument('--generation', required=True); p.add_argument('--out', required=True); p.add_argument('--qualification')
    p = sub.add_parser('preview'); p.add_argument('--state', required=True); p.add_argument('--video', required=True); p.add_argument('--native-pts', required=True, type=int); p.add_argument('--out', required=True)
    p = sub.add_parser('roi'); p.add_argument('--state', required=True); p.add_argument('--request', required=True); p.add_argument('--out', required=True)
    a = parser.parse_args()
    if a.command == 'fit': result = register_points(a.state, a.measurements, a.out)
    elif a.command == 'fit-tracks':
        from .screen_tracklet_registration import fit_tracklets
        result = fit_tracklets(a.state, a.generation, a.out, a.qualification)
    elif a.command == 'preview': result = preview(a.state, a.video, a.native_pts, a.out)
    else:
        from .screen_recovery_state import ScreenRecoveryState
        state = ScreenRecoveryState(a.state); doc = read_json(a.request)
        if doc.get('surface_class') != 'ice': raise ValueError('Ice H cannot navigate board/advertisement surfaces')
        result = state.navigation_roi(doc['native_pts'], doc['shot_id'], doc['reference_xy'],
            world_chart_sha256=doc['world_chart_sha256'], margin_px=doc['margin_px'])
        write_json(a.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
