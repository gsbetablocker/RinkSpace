"""Review complete shared-transverse geometry with its native camera trajectory.

Default output is requested PNGs. --video is an explicit complete-shot render;
this entry point never runs an optimizer or constructs a world Jacobian.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .first_shot_review import DATA, FirstShotReview, render_video
from . import row_scene_projection


def _json_sha(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf8')
    return hashlib.sha256(payload).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _review_bundle(adapter, review_state_mixin, review_bundle_type):
    class ActualTransverseReviewState(review_state_mixin):
        # CompleteTransverseBundle.bundle.state is an unshifted algebraic
        # intermediate. Only adapter.state applies the shared5 before the
        # public provider's geometry/material containment and review paths.
        def review_scene(self, q):
            return self.adapter.state.review_scene(q)

        def world_paths(self, q, targets):
            return self.adapter.state.world_paths(q, targets)

    proxy = review_bundle_type(adapter)
    review_type = type('ExactTransverseReviewState',
                       (ActualTransverseReviewState, type(adapter.state)), {})
    proxy.state = review_type(adapter)
    return proxy


def _geometry_identity(adapter, proxy, q):
    actual_scene = adapter.state.review_scene(q)
    scene_sha = _json_sha(actual_scene)
    if _json_sha(proxy.state.review_scene(q)) != scene_sha:
        raise ValueError('Review scene differs from the actual shared5 state')
    descriptors = adapter.bundle.target_descriptors
    direct = adapter.state.world_paths(q, descriptors)
    reviewed = proxy.state.world_paths(q, descriptors)
    fit_world, derivatives = adapter.world(q, False)
    if derivatives is not None:
        raise ValueError('Still review must not construct a world Jacobian')
    base_world, base_derivatives = adapter.bundle.world(q[:adapter.state.old_count], False)
    if base_derivatives is not None:
        raise ValueError('Still review must not request intermediate derivatives')
    maximum_fit_difference = 0.
    changed_paths = 0
    maximum_shared5_displacement = 0.
    for key, actual, rendered in zip(adapter.bundle.keys, direct, reviewed, strict=True):
        if not np.array_equal(actual, rendered):
            raise ValueError('Review path differs from actual shared5 world path: ' + key)
        expected = fit_world[key]
        if actual.shape != expected.shape:
            raise ValueError('Actual scene and FIT world path shapes differ: ' + key)
        difference = float(np.max(np.abs(actual - expected), initial=0.))
        maximum_fit_difference = max(maximum_fit_difference, difference)
        if difference > 1e-9:
            raise ValueError('Actual scene and FIT world values differ: ' + key)
        displacement = float(np.max(np.abs(actual - base_world[key]), initial=0.))
        maximum_shared5_displacement = max(maximum_shared5_displacement, displacement)
        changed_paths += int(displacement > 1e-12)
    if np.any(adapter.state.deviations(q)) and not changed_paths:
        raise ValueError('Nonzero shared5 did not move any actual review geometry')
    return dict(actual_scene_sha256=scene_sha, target_paths=len(descriptors),
                proxy_paths_byte_exact_to_actual_state=True,
                maximum_actual_vs_fit_world_difference=maximum_fit_difference,
                fit_world_comparison_tolerance=1e-9,
                paths_moved_from_unshifted_intermediate=changed_paths,
                maximum_shared5_world_displacement=maximum_shared5_displacement,
                world_units='inherited_scene_units_not_verified_metres',
                shared5=adapter.state.deviations(q).tolist(),
                shared5_columns=adapter.state.deviation_indices.tolist(),
                review_producer='adapter.state.review_scene(q) and adapter.state.world_paths(q, targets)',
                algebraic_intermediate_rendered=False, world_jacobian_requested=False)


def _camera_identity(adapter, proxy, q, frames):
    checks = []
    rows = np.linspace(0., 1079., 17)
    for index in frames:
        expected_camera = adapter.state.camera25(q, index)
        actual_camera = proxy.state.camera25(q, index)
        if expected_camera.shape != (25,) or not np.array_equal(expected_camera, actual_camera):
            raise ValueError('Review camera25 differs from actual complete camera')
        fitted = adapter.model.prepare(adapter.state, q, index)
        rendered = row_scene_projection.prepare_render_projection(proxy.state, q, index)
        expected_rows = fitted.camera25_at_rows(rows)
        actual_rows = (np.repeat(actual_camera[None, :], len(rows), axis=0)
                       if rendered is None else rendered.context.camera25_at_rows(rows))
        if not np.array_equal(expected_rows, actual_rows):
            raise ValueError('Review row exposures differ from the fitted native trajectory')
        checks.append(dict(frame=index, native_pts=int(adapter.state.pts[index - 1]),
                           camera25_sha256=_array_sha(actual_camera),
                           sampled_row_camera_sha256=_array_sha(actual_rows),
                           sampled_rows=rows.tolist(),
                           camera25_byte_exact_to_actual_state=True,
                           all_17_row_cameras_byte_exact_to_fit=True,
                           row_projection_enabled=rendered is not None))
    return checks



def _timed_draw(review, native, pts, profile_path=None):
    """Time source checks before the first camera call separately from raster.

    FirstShotReview.draw validates packet/native pixels/mask before requesting
    camera25. Timed method wrappers leave all those public checks unchanged.
    """
    timing = {}
    cache_before = review.renderer.geometry_cache_info()
    state = review.bundle.state
    camera25 = state.camera25
    renderer_draw = review.renderer.draw
    started = time.perf_counter()

    def camera(*args, **kwargs):
        if 'source_validation_seconds' not in timing:
            timing['source_validation_seconds'] = time.perf_counter() - started
        return camera25(*args, **kwargs)

    def raster(*args, **kwargs):
        before = time.perf_counter()
        timing['camera_preparation_seconds'] = before - started - timing['source_validation_seconds']
        try:
            return renderer_draw(*args, **kwargs)
        finally:
            timing['renderer_draw_seconds'] = time.perf_counter() - before

    state.camera25 = camera
    review.renderer.draw = raster
    profiler = cProfile.Profile() if profile_path is not None else None
    try:
        if profiler is not None:
            profiler.enable()
        canvas = review.draw(native, pts)
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(profile_path))
        state.camera25 = camera25
        review.renderer.draw = renderer_draw
    timing['complete_draw_seconds'] = time.perf_counter() - started
    timing['source_points_and_plan_overlay_seconds'] = (timing['complete_draw_seconds']
        - timing['source_validation_seconds'] - timing['camera_preparation_seconds']
        - timing['renderer_draw_seconds'])
    timing['profiled'] = profiler is not None
    cache_after = review.renderer.geometry_cache_info()
    timing['projection_geometry_cache_misses'] = cache_after['misses']-cache_before['misses']
    timing['projection_geometry_cache_hits'] = cache_after['hits']-cache_before['hits']
    return canvas, timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prototype-dir', 'binding', 'association', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[1, 201, 385, 842, 1239, 1241, 1248])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Preserve existing PNGs without redrawing them')
    parser.add_argument('--profile-next-draw', action='store_true')
    parser.add_argument('--fast-projection', action='store_true')
    parser.add_argument('--projection-audit-frames', type=int, nargs='*', default=[])
    parser.add_argument('--after-audit-frames', type=int, nargs='*', default=[])
    args = parser.parse_args(argv)
    frames = sorted(set(args.frames))
    followup_frames = [i for i in sorted(set(args.after_audit_frames)) if i not in frames]
    frames.extend(followup_frames)
    if followup_frames and not args.projection_audit_frames:
        parser.error('--after-audit-frames requires explicit projection audits')
    if any(index < 1 or index > 1248 for index in frames):
        parser.error('--frames must address the actual first-shot frames 1..1248')
    if args.projection_audit_frames and not args.fast_projection:
        parser.error('--projection-audit-frames requires --fast-projection')
    if not set(args.projection_audit_frames).issubset(frames):
        parser.error('Projection audit frames must be included in --frames')
    output = Path(args.output).resolve()
    if not output.is_relative_to(DATA):
        raise ValueError('Review diagnostics and media must stay in Data')
    if output.exists() and not args.resume:
        raise FileExistsError('Preserve the existing review directory: ' + str(output))
    prototype = Path(args.prototype_dir).resolve()
    for name in ('native_transverse_bundle.py', 'native_camera_review.py'):
        if not (prototype / name).is_file():
            raise FileNotFoundError(prototype / name)
    sys.path.insert(0, str(prototype))
    from native_transverse_bundle import CompleteTransverseBundle
    from native_camera_review import ReviewStateMixin, ReviewBundle
    from .official_static_surface_dataset import reference

    output.mkdir(parents=True, exist_ok=args.resume)
    started = time.perf_counter()
    adapter = CompleteTransverseBundle(Path(args.binding), Path(args.association),
                                       Path(args.checkpoint), geometry_workers=1, source_workers=1)
    try:
        q = adapter.start.copy()
        if q.shape != (7674,) or _array_sha(q) != adapter.seed_receipt['seed_qaug_sha256']:
            raise ValueError('The complete7674 checkpoint identity changed')
        proxy = _review_bundle(adapter, ReviewStateMixin, ReviewBundle)
        adapter.model.install_dispatch()
        geometry = _geometry_identity(adapter, proxy, q)
        cameras = _camera_identity(adapter, proxy, q, frames)
        identity = dict(schema='rinkspace.complete-transverse-review-identity.v1',
                        archive=reference(args.checkpoint), seed=adapter.seed_receipt,
                        q_sha256=_array_sha(q), parameter_count=len(q),
                        binding=reference(args.binding), association=reference(args.association),
                        source_producers=adapter.source_producers,
                        wrapper=reference(__file__), geometry=geometry)
        review = FirstShotReview(proxy, q, identity)
        fast_module = None
        fast_provenance = None
        if args.fast_projection:
            from . import row_scene_projection_fast as fast_module
            fast_provenance = fast_module.install()
            fast_provenance['producer'] = reference(fast_module.__file__)
        (output / 'run_identity.json').write_text(json.dumps(dict(identity=identity, camera_checks=cameras, fast_projection=fast_provenance),
            ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
        checks = []
        profile_next = args.profile_next_draw
        passed_audit_frames = set()
        for index in frames:
            if index in followup_frames and not set(args.projection_audit_frames).issubset(passed_audit_frames):
                raise ValueError('Supplemental stills require successful full-call and PNG projection audits first')
            image = output / f'frame_{index:04d}.png'
            if image.exists():
                if not args.resume:
                    raise FileExistsError(image)
                checks.append(dict(frame=index, image=reference(image), rendered_this_run=False,
                    retained_prior_png=True, timing=None, prior_render_not_repeated=True))
                continue
            frame = adapter.bundle.frames[index - 1]
            pts = int(frame['native_pts'])
            read_started = time.perf_counter()
            native = cv2.imread(review.sources[pts]['image_path'])
            read_seconds = time.perf_counter() - read_started
            if native is None:
                raise ValueError('Original native source image is missing')
            profile = output / f'draw_{index:04d}.pstats' if profile_next else None
            projection_audit = None
            reference_canvas = None
            if index in args.projection_audit_frames:
                audit = fast_module.ProjectionAudit(output / f'projection_inputs_{index:04d}.zip')
                try:
                    fast_module.install(audit)
                    reference_canvas, reference_timing = _timed_draw(review, native, pts)
                finally:
                    projection_audit = audit.close()
                    fast_module.install()
                reference_image = output / f'frame_{index:04d}_original.png'
                if not cv2.imwrite(str(reference_image), reference_canvas):
                    raise OSError(reference_image)
                projection_audit.update(reference_image=reference(reference_image),
                                        reference_draw_timing=reference_timing,
                                        reference_timing_includes_input_capture_and_both_kernels=True)
                # The camera/row key deliberately describes physics, not the
                # projection implementation. Reusing its projected geometry
                # would make the independent fast PNG comparison vacuous.
                review.renderer._geometry_cache.clear()
                projection_audit['reference_projected_geometry_cache_cleared'] = True
            canvas, timing = _timed_draw(review, native, pts, profile)
            if reference_canvas is not None and timing['projection_geometry_cache_misses'] < 1:
                raise ValueError('Independent fast PNG must freshly project the entire geometry inventory')
            if reference_canvas is not None:
                difference = np.abs(canvas.astype(np.int16)-reference_canvas.astype(np.int16))
                projection_audit['complete_PNG_pixel_comparison'] = dict(
                    identical=bool(np.array_equal(canvas, reference_canvas)),
                    differing_pixels=int(np.any(difference != 0, axis=2).sum()),
                    maximum_channel_difference=int(difference.max(initial=0)))
                if (projection_audit['all_valid_depth_and_finite_masks_identical']
                        and projection_audit['maximum_UV_difference_px'] <= 1e-9
                        and projection_audit['complete_PNG_pixel_comparison']['identical']):
                    passed_audit_frames.add(index)
            timing['source_image_read_seconds'] = read_seconds
            write_started = time.perf_counter()
            if not cv2.imwrite(str(image), canvas):
                raise OSError(image)
            timing['png_write_seconds'] = time.perf_counter() - write_started
            checks.append(dict(frame=index, image=reference(image), rendered_this_run=True,
                               timing=timing, profile=None if profile is None else reference(profile),
                               metrics=review.last_metrics, projection_audit=projection_audit))
            profile_next = False
            print(json.dumps(dict(stage='same_transverse_model_still', frame=index, image=str(image), timing=timing)), flush=True)
        video = (render_video(review, output / f'first_shot_{review.q_sha256[:12]}_candidate.mp4')
                 if args.video else None)
        for ref in adapter.source_producers:
            if reference(ref['path']) != ref:
                raise ValueError('Model/source producer changed during review: ' + ref['path'])
        if fast_provenance is not None and reference(fast_provenance['producer']['path']) != fast_provenance['producer']:
            raise ValueError('Fast review projection producer changed during review')
        result = dict(schema='rinkspace.complete-transverse-same-model-review.v1',
                      identity=identity, camera_checks=cameras, frames=checks,
                      full_grid_advertisement_boundaries_numbers_and_same_model_plan=True,
                      fast_projection=fast_provenance,
                      projection_audit_passed_frames=sorted(passed_audit_frames),
                      source_points=review.active_source_points,
                      source_and_plan_inventory=review.renderer.path_counts,
                      advertisement_ids=review.plan.advertisement_ids, ad12=review.ad12_proof,
                      optimizer_started=False, world_jacobian_requested=False,
                      historical_quarantined_points_displayed=False,
                      accuracy_accepted=False, stills_do_not_validate_complete_shot=True,
                      video=video, elapsed_seconds=time.perf_counter() - started)
        receipt = output / 'review_receipt.json'
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
        print(json.dumps(dict(stage='same_transverse_model_review_finished',
                              receipt=str(receipt), q_sha256=review.q_sha256,
                              static_checks=len(checks), video_generated=video is not None,
                              seconds=result['elapsed_seconds'])), flush=True)
        return result
    finally:
        if args.fast_projection:
            from .row_scene_projection_fast import uninstall
            uninstall()
        adapter.close()


if __name__ == '__main__':
    main()
