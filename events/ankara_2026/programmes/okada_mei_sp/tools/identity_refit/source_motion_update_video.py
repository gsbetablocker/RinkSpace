"""Update affected native exposures in the one existing comparison video.

Unchanged spline support is proved from both saved models, including all five
readout states. Those frames reuse the previous verified render; changed frames
are projected from the actual new mesh. The complete video clock is rechecked.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

import source_motion_delivery as delivery
from source_review import digest
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve
from calibration.program_tracking.body_tracking.native_mesh_overlay import open_encoder, encode
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity


def changed_exposures(before, after, reference, camera, previous_camera=None):
    camera_mask=np.zeros(len(reference.times),bool)
    if previous_camera is None:
        validate_checkpoint_identity(after, before)
    else:
        from source_camera_rebind import camera_changed_frames
        # The two states must each bind their actual camera. Only that declared
        # identity may differ; geometry/mount/model identities remain checked.
        validate_checkpoint_identity(after,dict(before,camera_binding_json=after['camera_binding_json']))
        if json.loads(str(before['camera_binding_json']))['state_sha256'] != previous_camera.info['state_sha256']:
            raise ValueError('Previous video camera does not bind its saved model')
        if json.loads(str(after['camera_binding_json']))['state_sha256'] != camera.info['state_sha256']:
            raise ValueError('New video camera does not bind its saved model')
        camera_mask=camera_changed_frames(previous_camera,camera)
    for key in ('relative_skate_knots', 'relative_skate_rotation_controls'):
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(before['x'][:17], after['x'][:17])
    a, b = curve_from_packet(before), curve_from_packet(after)
    frames = np.arange(len(reference.times))
    times = reference.times[:, None] + camera.readout_times(frames)[:, None] * np.linspace(-.5, .5, 5)
    if not np.array_equal(a.knots, b.knots):
        # Knot insertion changes the control layout. Reuse a rendered exposure
        # only when both curves evaluate exactly alike at every readout node.
        # Approximate preservation during refinement still requires rendering.
        affected = np.zeros(len(frames), dtype=bool)
        for start in range(0, len(frames), 128):
            t = times[start:start+128].ravel()
            old, new = evaluate_curve(a, t), evaluate_curve(b, t)
            changed = np.any(old['P'] != new['P'], axis=1)
            changed |= np.any(old['rotations'] != new['rotations'], axis=(1, 2, 3))
            affected[start:start+128] = changed.reshape(-1, 5).any(1)
        return affected | camera_mask
    changed = np.any(a.position_controls != b.position_controls, axis=1)
    changed |= np.any(a.rotation_controls != b.rotation_controls, axis=(1, 2, 3))
    edge = np.any(a._check_log_chart() != b._check_log_chart(), axis=(1, 2))
    changed[:-1] |= edge
    changed[1:] |= edge
    affected = np.asarray(a.basis_sparse(times.ravel())[:, np.flatnonzero(changed)].getnnz(axis=1) > 0)
    affected = affected.reshape(len(frames), 5).any(1) | camera_mask
    # Check the actual rotations as well as translation at every reused node.
    reused_times = times[~affected].ravel()
    for start in range(0, len(reused_times), 1024):
        t = reused_times[start:start+1024]
        old, new = evaluate_curve(a, t), evaluate_curve(b, t)
        np.testing.assert_array_equal(old['P'], new['P'])
        np.testing.assert_array_equal(old['rotations'], new['rotations'])
    return affected


def update(args):
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    path = args.output/'source_motion_comparison.mp4'
    clock_path = args.output/'source_clock.json'
    clock = json.loads(clock_path.read_text(encoding='utf8'))
    crop_ledger=clock.get('body_observations')
    args.body_observations=None if crop_ledger is None else Path(crop_ledger)
    if crop_ledger is not None and digest(crop_ledger)!=clock['body_observations_sha256']:
        raise ValueError('Physical source ankle crop identities changed; a full render is required')
    base_path = Path(clock['after'])
    if digest(base_path) != clock['after_sha256']:
        raise ValueError('The previous rendered model was changed; cannot reuse its frames')
    if digest(args.before) != clock['before_sha256']:
        raise ValueError('The displayed before model changed')
    base = dict(np.load(base_path))
    args.foot_detail = False
    args.both_feet_detail = True
    delivery.initialize_renderer(args)
    current = delivery._render_state[0][1]
    reference, camera, _, _, after = current
    pts = reference.p['pts']
    np.testing.assert_array_equal(pts, clock['source_pts'])
    np.testing.assert_array_equal(clock['time_base'], [1, 12800])
    registration = json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    if Path(clock['source_video']).resolve() != Path(registration['path']).resolve():
        raise ValueError('Original media changed')
    previous_source=Path(clock.get('after_source') or getattr(args,'previous_source',None)
                         or getattr(args,'before_source',None) or args.source)
    if clock.get('after_source_sha256') and digest(previous_source)!=clock['after_source_sha256']:
        raise ValueError('Previous video source/camera document changed')
    previous=delivery.bind(args.prepared,previous_source,base_path)
    np.testing.assert_array_equal(previous[0].p['pts'],pts)
    affected = changed_exposures(base, after, reference, camera, previous[1])
    selected = np.flatnonzero(affected)
    if not len(selected):
        raise ValueError('No mesh changes to render')
    prior = dict(path=str(path), sha256=digest(path), source_clock=clock)
    if 'full_resume_checkpoint' in base:
        saved = Path(str(base['full_resume_checkpoint']))
        prior['replay_checkpoint'] = dict(path=str(saved), sha256=digest(saved))
        if str(base['candidate_sha256']) != prior['replay_checkpoint']['sha256']:
            raise ValueError('Archived replay checkpoint differs from rendered model')
    pending = path.with_name('source_motion_comparison.pending.mp4')
    if pending.exists():
        raise FileExistsError(pending)
    print(json.dumps(dict(event='update_video_start', total=len(pts),
        changed_native_frames=len(selected), unchanged_native_frames=int((~affected).sum()),
        exact_reused_readout_states=int((~affected).sum())*5)), flush=True)
    pool = None
    if getattr(args, 'workers', 1) > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers,
            initializer=delivery.initialize_renderer, initargs=(args,))
    def replacement_frames():
        chunks = iter(selected[start:start+3] for start in range(0, len(selected), 3))
        if pool is None:
            for chunk in chunks:
                yield from delivery.render_chunk(chunk)
            return
        pending_frames = deque()
        for _ in range(2*args.workers):
            chunk = next(chunks, None)
            if chunk is not None:
                pending_frames.append(pool.submit(delivery.render_chunk, chunk))
        while pending_frames:
            frames = pending_frames.popleft().result()
            chunk = next(chunks, None)
            if chunk is not None:
                pending_frames.append(pool.submit(delivery.render_chunk, chunk))
            yield from frames
    replacement = iter(replacement_frames())
    writer = open_encoder(pending, 1920, 1080)
    count = rendered = 0
    try:
        with av.open(str(path)) as old:
            for i, frame in enumerate(old.decode(video=0)):
                if i >= len(pts) or abs(float(frame.pts*frame.time_base)-i/50.) > 1e-10:
                    raise ValueError('Previous video source clock mismatch')
                if frame.width != 1920 or frame.height != 1080:
                    raise ValueError('Comparison layout changed')
                if affected[i]:
                    stamp, pixels = next(replacement)
                    if stamp != pts[i]:
                        raise ValueError('Updated mesh and original exposure differ')
                    image = Image.fromarray(pixels)
                    rendered += 1
                else:
                    image = frame.to_image()
                encode(*writer, image, i)
                count += 1
                if count % 500 == 0 or count == len(pts):
                    print(json.dumps(dict(event='update_video_progress',frames=count,
                        total=len(pts),rerendered=rendered)),flush=True)
        if count != len(pts) or rendered != len(selected):
            raise ValueError('Video coverage mismatch')
        if next(replacement, None) is not None:
            raise ValueError('Unexpected extra updated frame')
        for packet in writer[1].encode():
            writer[0].mux(packet)
    finally:
        writer[0].close()
        delivery._render_state[2].close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    with av.open(str(pending)) as video:
        times = np.array([float(f.pts*f.time_base) for f in video.decode(video=0)])
    np.testing.assert_allclose(times, np.arange(len(pts))/50., atol=1e-10, rtol=0.)
    result = dict(clock, before=str(args.before), after=str(args.after),
        before_sha256=digest(args.before), after_sha256=digest(args.after),
        before_source=str(getattr(args,'before_source',None) or args.source),
        before_source_sha256=digest(getattr(args,'before_source',None) or args.source),
        after_source=str(args.source),after_source_sha256=digest(args.source),
        before_camera=delivery._render_state[0][0][2],after_camera=current[2],
        replaces_previous_delivery=prior, native_frames_verified=count,
        updated_source_pts=pts[affected].tolist(),
        unchanged_render_reuse=dict(frames=int((~affected).sum()),
            actual_five_readout_states_exactly_equal=True,
            complete_camera_spline_rig_and_readout_exactly_equal=True,
            image_operation='Decode/reencode the same rendered frame; no alignment or image warp'),
        projection='Actual saved body and both skates; changed native exposures rerendered, unchanged five-node poses verified identical',
        visual_acceptance='pending')
    pending_clock = clock_path.with_suffix('.pending.json')
    pending_clock.write_text(json.dumps(result,indent=2),encoding='utf8')
    pending.replace(path)
    pending_clock.replace(clock_path)
    print(json.dumps(dict(event='updated_video_saved', path=str(path),
        sha256=digest(path), native_frames=count, updated_native_frames=rendered)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','registration','before','after','before-result','output'):
        parser.add_argument('--'+name, type=lambda s:Path(s).resolve(), required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--before-source',type=lambda s:Path(s).resolve())
    parser.add_argument('--previous-source',type=lambda s:Path(s).resolve())
    update(parser.parse_args())
