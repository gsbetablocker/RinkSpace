"""Screen every native exposure and review the complete programme in time order.

Source-mask changes flag turns; mask disagreement flags candidates, not truth.
The review boards use the already encoded, checkpoint-bound comparison video.
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from source_review import digest


def spans(mask, pts, shots):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = (np.diff(indices) > 1) | (shots[indices[1:]] != shots[indices[:-1]])
    return [dict(first_pts=int(pts[a[0]]), last_pts=int(pts[a[-1]]),
                 seconds=(pts[a[[0, -1]]] / 12800.).tolist(), frames=len(a))
            for a in np.split(indices, np.flatnonzero(cuts) + 1)]


def index(work, *, source_outline=None, preserve_report=False):
    ledger_path=work/'input/shoe_source.npz' if source_outline is None else source_outline
    ledger = dict(np.load(ledger_path))
    screen = dict(np.load(work / 'source_review/shoe_screen_after.npz'))
    clock = json.loads((work / 'delivery/source_clock.json').read_text(encoding='utf8'))
    pts, shots = ledger['source_pts'], ledger['shot_ids']
    np.testing.assert_array_equal(pts, screen['pts'])
    np.testing.assert_array_equal(pts, clock['source_pts'])
    if source_outline is None:
        assert str(screen['parameter_sha256']) == clock['after_sha256']
    else:
        assert str(screen['source_outline_sha256']) == digest(ledger_path)
    parameter_sha256=str(screen['parameter_sha256'])
    visible = ledger['visible'] & (ledger['role'] == 1) & np.isfinite(ledger['region']).all(axis=(2, 3))
    partial = ledger.get('partial_boundary', np.zeros_like(visible))
    centre = np.full((len(pts), 2, 2), np.nan)
    axis = np.full((len(pts), 2), np.nan)
    anisotropy = np.full_like(axis, np.nan)
    size = np.full_like(axis, np.nan)
    for frame, foot in zip(*np.nonzero(visible)):
        p = ledger['region'][frame, foot]
        m = cv2.moments(p.astype(np.float32))
        if m['m00'] <= 1.e-6:
            continue
        centre[frame, foot] = [m['m10'] / m['m00'], m['m01'] / m['m00']]
        covariance = np.array([[m['mu20'], m['mu11']], [m['mu11'], m['mu02']]]) / m['m00']
        values, vectors = np.linalg.eigh(covariance)
        axis[frame, foot] = np.arctan2(vectors[1, -1], vectors[0, -1])
        anisotropy[frame, foot] = (values[-1] - values[0]) / max(values[-1], 1.e-9)
        size[frame, foot] = np.sqrt(m['m00'])
        if partial[frame, foot]:
            # A fragment of an occluded shoe cannot establish its full axis.
            axis[frame, foot] = np.nan
            anisotropy[frame, foot] = np.nan
    # Compare 80 ms source intervals, retaining cuts and missing masks explicitly.
    dt = (pts[4:] - pts[:-4]) / 12800.
    change = axis[4:] - axis[:-4]
    speed = np.full_like(axis, np.nan)
    same = shots[4:] == shots[:-4]
    valid = visible[4:] & visible[:-4] & same[:, None]
    valid &= (anisotropy[4:] > .25) & (anisotropy[:-4] > .25)
    rate = np.abs(np.arctan2(np.sin(2 * change), np.cos(2 * change))) / 2 / dt[:, None]
    speed[2:-2] = np.where(valid, np.degrees(rate), np.nan)
    error = screen['error']
    relative_error = error / np.maximum(size, 1.)
    rapid = (speed > 120.) & visible
    mismatch = visible & ((screen['iou'] < .60) | (relative_error > .10))
    windows = []
    # All chronological bins are reviewed, including bins without a high score.
    for lo in range(0, len(pts), 100):
        ids = np.arange(lo, min(lo + 100, len(pts)))
        samples = np.unique(np.rint(np.linspace(ids[0], ids[-1], 6)).astype(int))
        windows.append(dict(index=len(windows), seconds=(pts[ids[[0, -1]]] / 12800).tolist(),
                            sample_indices=samples.tolist(), sample_pts=pts[samples].tolist(),
                            rapid_frames=rapid[ids].sum(0).tolist(),
                            disagreement_frames=mismatch[ids].sum(0).tolist(),
                            missing_frames=(~visible[ids]).sum(0).tolist(),
                            review='pending'))
    report = dict(schema='rinkspace.native_turn_audit.v1', source_frames=len(pts),
                  source_seconds=(pts[[0, -1]] / 12800).tolist(), parameter_sha256=parameter_sha256,
                  source_outline_sha256=digest(ledger_path),
                  encoded_video_parameter_sha256=clock['after_sha256'],
                  video_sha256=digest(work / 'delivery/source_motion_comparison.mp4'),
                  all_frames_screened=True, all_chronological_bins_included=True,
                  visibility_counts=visible.sum(0).tolist(),
                  missing_counts=(~visible).sum(0).tolist(),
                  interpretation='Source shape-change and contour disagreement screen only; physical turn direction, identity, occlusion and correctness require the original sequence.',
                  rapid_source_shape_spans={str(f): spans(rapid[:, f], pts, shots) for f in (0, 1)},
                  disagreement_spans={str(f): spans(mismatch[:, f], pts, shots) for f in (0, 1)},
                  unknown_spans={str(f): spans(~visible[:, f], pts, shots) for f in (0, 1)},
                  windows=windows)
    if preserve_report:
        previous=json.loads((work/'source_review/whole_review.json').read_text(encoding='utf8'))
        previous.update(report);report=previous
    (work / 'source_review/whole_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    np.savez_compressed(work / 'source_review/turn_screen.npz', pts=pts, shot_ids=shots,
                        centre=centre, principal_axis=axis, anisotropy=anisotropy,
                        area_length_px=size, source_axis_rate_deg_s=speed,
                        relative_error=relative_error, rapid=rapid, mismatch=mismatch, visible=visible,
                        parameter_sha256=np.array(parameter_sha256),source_outline_sha256=np.array(digest(ledger_path)))
    print(json.dumps(dict(frames=len(pts), windows=len(windows), rapid_foot_frames=rapid.sum(0).tolist(),
                          disagreeing_foot_frames=mismatch.sum(0).tolist(), missing=(~visible).sum(0).tolist())), flush=True)
    return report


def boards(work, report):
    def decoded():
        with av.open(str(work / 'delivery/source_motion_comparison.mp4')) as video:
            for i,frame in enumerate(video.decode(video=0)):
                yield i,frame.to_image()
    compose_body_boards(work,report,decoded())


def compose_body_boards(work, report, images):
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    windows = report['windows']
    groups = {i: r for r in windows for i in r['sample_indices']}
    pages = {}
    counts = {}
    for i, image in images:
            if i not in groups:
                continue
            row = groups[i]
            page, slot = divmod(row['index'], 4)
            col = row['sample_indices'].index(i)
            if page not in pages:
                pages[page] = Image.new('RGB', (3360, 1840), (19, 24, 30))
                counts[page] = 0
            tile = Image.new('RGB', (560, 460), (19, 24, 30))
            draw = ImageDraw.Draw(tile)
            stamp = report['windows'][row['index']]['sample_pts'][col]
            draw.text((4, 3), f'W{row["index"]:02d} {stamp/12800:.2f}s  SOURCE / CURRENT', font=font, fill='white')
            for n, x in enumerate((0, 1280)):
                tile.paste(image.crop((x, 35, x+640, 625)).resize((280, 258), Image.Resampling.LANCZOS), (n*280, 27))
                tile.paste(image.crop((x, 660, x+640, 990)).resize((280, 144), Image.Resampling.LANCZOS), (n*280, 290))
            pages[page].paste(tile, (col*560, slot*460))
            counts[page] += 1
            rows = windows[page*4:page*4+4]
            if counts[page] == sum(len(r['sample_indices']) for r in rows):
                path = work / f'source_review/whole_review_{page}.jpg'
                pages.pop(page).save(path, quality=91)
                print(json.dumps(dict(page=page, path=str(path), seconds=[rows[0]['seconds'][0], rows[-1]['seconds'][1]])), flush=True)
    assert not pages


def native_boards(args):
    """Review the actual full mesh across the programme before video encoding."""
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    from types import SimpleNamespace
    import source_motion_delivery as delivery
    report_path=args.work/'source_review/whole_review.json'
    report=json.loads(report_path.read_text(encoding='utf8'))
    frames=np.array(sorted({i for w in report['windows'] for i in w['sample_indices']}))
    render_args=SimpleNamespace(prepared=args.prepared,source=args.source,before=args.before,
        after=args.parameter,registration=args.registration,
        before_result=args.work/'source_resume/fitted/result.npz',
        body_observations=args.work/'input/body_track_source.npz',foot_detail=False,both_feet_detail=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=delivery.initialize_renderer,
                             initargs=(render_args,)) as pool:
        chunks=iter([frames[i:i+3] for i in range(0,len(frames),3)])
        pending=deque()
        def rendered():
            for _ in range(2*args.workers):
                indices=next(chunks,None)
                if indices is not None:pending.append((indices,pool.submit(delivery.render_chunk,indices)))
            while pending:
                indices,future=pending.popleft();values=future.result()
                more=next(chunks,None)
                if more is not None:pending.append((more,pool.submit(delivery.render_chunk,more)))
                for i,(stamp,pixels) in zip(indices,values):
                    if stamp!=283136+256*i:raise ValueError('Native body-review exposure mismatch')
                    yield int(i),Image.fromarray(pixels)
        compose_body_boards(args.work,report,rendered())
    report['native_body_review_pages']=dict(after_sha256=digest(args.parameter),
        before_sha256=digest(args.before),source_frames=frames.tolist(),
        samples=len(frames),chronological_windows=len(report['windows']),
        projection='Same actual full-body and both-skate mesh renderer as final video, at five native readout nodes')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def project_all(prepared, source, parameter, work, output, source_outline=None):
    """Exact shoe surfaces at every native readout; retain compact silhouettes."""
    from scipy.spatial import cKDTree
    from compare import bind
    from experiments.skate_video_v1.body_skate_fit import Assembly, SCALE
    from experiments.skate_video_v1.inputs import distribute
    from calibration.program_tracking.body_tracking.skate_physics import fk_state
    from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection

    ref, camera, _, _, packet = bind(prepared, source, parameter)
    # Screen the same complete shoe and blade that the delivery renders.
    # The reduced runner assembly can miss end-on blade/heel disagreement.
    model = Assembly(ref.binding, exact_template=True)
    outline_path = work / 'input/shoe_turn_source.npz' if source_outline is None else source_outline
    observations = dict(np.load(outline_path))
    admitted = observations['visible'] & (observations['role'] == 1)
    partial = observations.get('partial_boundary', np.zeros_like(admitted))
    pts = ref.p['pts']; n = len(pts)
    np.testing.assert_array_equal(pts, observations['source_pts'])
    error = np.full((n, 2), np.nan); iou = error.copy()
    contour = np.full((n, 2, 80, 2), np.nan, np.float32)
    shape = np.full((n, 2, 2), np.nan); source_shape = shape.copy()
    rotations = np.empty((n, 2, 3, 3)); origins = np.empty((n, 2, 3))

    def descriptor(poly):
        m = cv2.moments(np.asarray(poly, np.float32))
        if m['m00'] < 1.e-9:
            return [np.nan, np.nan]
        trace = max(m['mu20'] + m['mu02'], 1.e-9)
        return [(m['mu20'] - m['mu02']) / trace, 2*m['mu11']/trace]

    for first in range(0, n, 64):
        frames = np.arange(first, min(n, first+64))
        tau = camera.readout_times(frames)
        t = (ref.times[frames, None] + tau[:, None]*np.linspace(-.5, .5, 5)).ravel()
        state = fk_state(ref, ref.at(t), np.zeros((len(t), 69)))
        rotations[frames] = state['R'].reshape(len(frames), 5, 2, 3, 3)[:, 2]
        origins[frames] = state['O'].reshape(len(frames), 5, 2, 3)[:, 2]
        for foot in (0, 1):
            nodes = model.points(state, foot).reshape(len(frames), 5, -1, 3)
            projected = moving_projection(camera, nodes, frames)['uv']
            for frame, uv in zip(frames, projected):
                # An absent source measurement is not proof the foot is hidden.
                # Keep in-frame model contours for visual review of these gaps.
                if not np.isfinite(uv).all():
                    continue
                lo_uv,hi_uv=uv.min(0),uv.max(0)
                outside=np.any(hi_uv<[0,0]) or np.any(lo_uv>[1920,1080])
                too_large=np.max((np.ceil(hi_uv-np.floor(lo_uv-3)+3))*SCALE)>700
                if outside or too_large:
                    if admitted[frame,foot]:raise ValueError('A measured shoe projects outside the review domain')
                    continue
                outlines = model.raster(uv)
                boundary = np.concatenate(outlines)
                longest = max(outlines, key=len)
                contour[frame, foot] = distribute(longest, 80, closed=True)
                shape[frame, foot] = descriptor(longest)
                if not admitted[frame, foot]:
                    continue
                raw = observations['region'][frame, foot]
                if not np.isfinite(raw).all():
                    continue
                if 'boundary_valid' in observations:
                    source_points = raw[observations['boundary_valid'][frame, foot]]
                else:
                    source_points = distribute(raw, 80, closed=True)
                if not len(source_points):
                    continue
                distances = cKDTree(boundary).query(source_points)[0]
                if not partial[frame, foot]:
                    source_shape[frame, foot] = descriptor(raw)
                    distances = np.r_[distances, cKDTree(source_points).query(contour[frame, foot])[0]]
                error[frame, foot] = np.median(distances)
                if partial[frame, foot]:
                    # Occluded source fragments have no full-shoe IoU target.
                    continue
                all_points = np.r_[boundary, raw]
                lo = np.floor(all_points.min(0)-2)
                wh = np.ceil(all_points.max(0)-lo+3).astype(int)
                if np.max(wh) > 2000:
                    raise ValueError('Unbounded projected shoe extent')
                a = np.zeros(tuple(wh[::-1]), np.uint8); b = a.copy()
                for outline in outlines:
                    cv2.fillPoly(a, [np.rint(outline-lo).astype(np.int32)], 1)
                cv2.fillPoly(b, [np.rint(raw-lo).astype(np.int32)], 1)
                iou[frame, foot] = np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b))
        if first % 512 == 0:
            print(json.dumps(dict(event='turn_projection', frames=min(n, first+64), total=n)), flush=True)
    np.savez_compressed(output, pts=pts, shot_ids=ref.p['shot_ids'], error=error, iou=iou,
                        visible=observations['visible'], role=observations['role'],
                        contour=contour, shape_descriptor=shape, source_shape_descriptor=source_shape,
                        rotation=rotations, origin=origins,
                        parameter_sha256=np.array(digest(parameter)),
                        partial_boundary=partial,
                        source_outline_sha256=np.array(digest(outline_path)),
                        source_outline_path=np.array(str(outline_path)),
                        exact_rendered_skate_template=np.array(True),
                        projected_vertex_count=np.array(len(model.vertices)))
    print(json.dumps(dict(event='turn_projection_complete', output=str(output), frames=n,
                         median_px=np.nanmedian(error, axis=0).tolist(),
                         p95_px=np.nanpercentile(error, 95, axis=0).tolist())), flush=True)


def compare_screens(work):
    """Compare identical native exposures, including every chronological bin."""
    before = dict(np.load(work/'source_review/shoe_screen_before.npz'))
    after = dict(np.load(work/'source_review/shoe_screen_after.npz'))
    source = dict(np.load(work/'source_review/turn_screen.npz'))
    np.testing.assert_array_equal(before['pts'], after['pts'])
    np.testing.assert_array_equal(before['source_outline_sha256'], after['source_outline_sha256'])
    np.testing.assert_array_equal(before.get('exact_rendered_skate_template', False),
                                  after.get('exact_rendered_skate_template', False))
    valid = (before['role'] == 1) & before['visible'] & after['visible']
    valid &= np.isfinite(before['error']) & np.isfinite(after['error'])
    size = np.maximum(source['area_length_px'], 1.)

    def metrics(cache, selected):
        error = cache['error'][selected]; iou = cache['iou'][selected]
        if not len(error):
            return dict(count=0)
        model = cache['shape_descriptor'][selected]
        raw = cache['source_shape_descriptor'][selected]
        defined = (np.linalg.norm(model, axis=1) > .25) & (np.linalg.norm(raw, axis=1) > .25)
        angle = np.degrees(np.abs(np.arctan2(model[:, 0]*raw[:, 1]-model[:, 1]*raw[:, 0],
                                            np.sum(model*raw, axis=1))))/2
        return dict(count=len(error), median_px=float(np.median(error)),
                    p95_px=float(np.percentile(error, 95)),
                    complete_outline_count=int(np.isfinite(iou).sum()),
                    median_iou=float(np.nanmedian(iou)) if np.isfinite(iou).any() else None,
                    low_overlap_count=int(np.count_nonzero(iou < .60)),
                    relative_error_p95=float(np.percentile((cache['error']/size)[selected], 95)),
                    elongated_shape_axis_median_deg=float(np.median(angle[defined])) if defined.any() else None,
                    elongated_shape_axis_p95_deg=float(np.percentile(angle[defined], 95)) if defined.any() else None)

    def pair(selected):
        selected = selected & valid
        a, b = metrics(before, selected), metrics(after, selected)
        old_bad = selected & ((before['iou'] < .60) | (before['error']/size > .10))
        new_bad = selected & ((after['iou'] < .60) | (after['error']/size > .10))
        return dict(before=a, after=b, old_disagreements=int(old_bad.sum()),
                    new_disagreements=int(new_bad.sum()), resolved=int((old_bad & ~new_bad).sum()),
                    introduced=int((new_bad & ~old_bad).sum()))

    rows=[]
    for lo in range(0, len(valid), 100):
        mask=np.zeros_like(valid);mask[lo:lo+100]=True
        rows.append(dict(seconds=(before['pts'][[lo,min(lo+99,len(valid)-1)]]/12800.).tolist(),
                         all_visible=pair(mask), rapid_source_shape=pair(mask & source['rapid'])))
    result=dict(before_sha256=str(before['parameter_sha256']), after_sha256=str(after['parameter_sha256']),
                exact_rendered_skate_template=bool(after.get('exact_rendered_skate_template', False)),
                source_outline_sha256=str(before['source_outline_sha256']),
                source_frames=len(valid), known_foot_exposures=int(valid.sum()),
                unavailable_or_rejected_foot_exposures=int(valid.size-valid.sum()),
                all_visible=pair(np.ones_like(valid)), rapid_source_shape=pair(source['rapid']),
                feet={str(f):pair(np.broadcast_to(np.arange(2)==f,valid.shape)) for f in (0,1)},
                shots={str(shot):pair(np.broadcast_to((before['shot_ids']==shot)[:,None],valid.shape))
                       for shot in np.unique(before['shot_ids'])}, windows=rows,
                interpretation='Image agreement on the same source masks, not independent truth or a certificate for occluded feet. Shape axes are unsigned image directions, not physical skate yaw.')
    report_path=work/'source_review/whole_review.json'
    report=json.loads(report_path.read_text(encoding='utf8'))
    report['turn_follow_comparison']=result
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('windows','shots')}),flush=True)
    return result


def foot_boards(work, registration, pages=None, dense=False):
    """All chronological review windows, source/before/after for both feet."""
    from source_review import SourceVideo
    report=json.loads((work/'source_review/whole_review.json').read_text(encoding='utf8'))
    old=dict(np.load(work/'source_review/shoe_screen_before.npz'))
    new=dict(np.load(work/'source_review/shoe_screen_after.npz'))
    raw=dict(np.load(str(new['source_outline_path']) if 'source_outline_path' in new else work/'input/shoe_turn_source.npz'))
    windows=report['windows'];prefix='foot_overview'
    if dense:
        screen=dict(np.load(work/'source_review/turn_screen.npz'))
        valid=(new['role']==1)&new['visible']
        bad=valid&screen['rapid']&((new['iou']<.60)|(new['error']/np.maximum(screen['area_length_px'],1.)>.10))
        centres={int(r['pts']) for r in report.get('direction_source_revision',{}).get('added_readings',[])}
        centres.update(map(int,report.get('source_ambiguity_review_pts',[])))
        # Previous problem locations remain navigation targets even after the
        # old acceptance claims have been archived out of the current report.
        previous_path=work/'source_review/whole_review_before_occlusion.json'
        if previous_path.exists():
            previous=json.loads(previous_path.read_text(encoding='utf8'))
            centres.update(int(r['centre_pts']) for r in previous.get('dense_turn_review',{}).get('windows',[]))
        occlusion_path=work/'source_review/occlusion_review.json'
        if occlusion_path.exists():
            occlusion=json.loads(occlusion_path.read_text(encoding='utf8'))
            centres.update(int(new['pts'][r['centre_index']]) for r in occlusion['rows'])
        residual_path=work/'source_review/direction_residuals.json'
        if residual_path.exists():
            residual=json.loads(residual_path.read_text(encoding='utf8'))
            centres.update(int(new['pts'][r['centre_index']]) for r in residual['rows'])
        foreground_review=work/'source_review/occlusion_geometry.json'
        if foreground_review.exists():
            foreground=json.loads(foreground_review.read_text(encoding='utf8'))
            if foreground['results']['after']['parameter_sha256']==str(new['parameter_sha256']):
                centres.update(int(r['pts']) for r in foreground['probes'] if r['after_max_px']>10.)
        centres.update(int(r['pts']) for r in report.get('direction_source_revision',{}).get('source_ankle_association_revision',[]))
        centres.update(int(r['pts']) for r in report.get('unmeasured_source_revision',{}).get('added_points',[]))
        for key in ('final_review_source_probes','final_review_extra_source_probes',
                    'residual_source_probes','residual_extra_source_probes',
                    'orientation_source_probes','orientation_extra_source_probes',
                    'whole_body_foot_issue_probes','remaining_shoe_source_probes'):
            centres.update(int(r['pts']) for r in report.get(key,[]))
        old_bad=valid&((old['iou']<.60)|(old['error']/np.maximum(screen['area_length_px'],1.)>.10))
        new_bad=valid&((new['iou']<.60)|(new['error']/np.maximum(screen['area_length_px'],1.)>.10))
        # Inspect newly introduced disagreements even outside rapid turns.
        bad |= new_bad & ~old_bad
        # Include rapid source changes throughout the entire programme, even
        # where the new outline score is already good. A good silhouette alone
        # cannot establish the directed toe/heel orientation during a turn.
        for first in range(0,len(new['pts']),100):
            stop=min(first+100,len(new['pts']))
            selected=screen['rapid'][first:stop]
            if selected.any():
                rate=np.where(selected,screen['source_axis_rate_deg_s'][first:stop],-np.inf)
                frame=first+np.unravel_index(np.argmax(rate),rate.shape)[0]
                centres.add(int(new['pts'][frame]))
        # A partial or overlapping source mask has no trustworthy full axis.
        # Inspect the fastest saved rotations in every chronological bin too;
        # these model rates select images only, never certify source turns.
        from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
        for candidate in (old,new):
            rotation=candidate['rotation']
            step=np.degrees(np.linalg.norm(_log(rotation[:-1].swapaxes(-1,-2)@rotation[1:]),axis=-1))/.02
            step[candidate['shot_ids'][1:]!=candidate['shot_ids'][:-1]]=np.nan
            for first in range(0,len(step),100):
                block=step[first:first+100]
                if np.isfinite(block).any() and np.nanmax(block)>120.:
                    frame=first+np.unravel_index(np.nanargmax(block),block.shape)[0]+1
                    centres.add(int(new['pts'][frame]))
        for foot in (0,1):
            ids=np.flatnonzero(bad[:,foot])
            if len(ids):
                cut=(np.diff(ids)>6)|(new['shot_ids'][ids[1:]]!=new['shot_ids'][ids[:-1]])
                for block in np.split(ids,np.flatnonzero(cut)+1):
                    rank=np.where(np.isfinite(new['iou'][block,foot]),1-new['iou'][block,foot],
                                  new['error'][block,foot]/np.maximum(screen['area_length_px'][block,foot],1.))
                    frame=block[np.argmax(rank)];centres.add(int(new['pts'][frame]))
        windows=[]
        for stamp in sorted(centres):
            if windows and stamp-windows[-1]['centre_pts']<1280:continue
            frame=int(np.searchsorted(new['pts'],stamp));samples=np.unique(np.clip(frame+np.array([-4,-2,0,1,3,5]),0,len(new['pts'])-1))
            samples=samples[new['shot_ids'][samples]==new['shot_ids'][frame]]
            windows.append(dict(index=len(windows),centre_pts=stamp,sample_indices=samples.tolist(),sample_pts=new['pts'][samples].tolist(),seconds=(new['pts'][samples[[0,-1]]]/12800.).tolist()))
        prefix='turn_detail';report['dense_turn_review']=dict(after_sha256=str(new['parameter_sha256']),
            sampling='20 to 40 ms original-source steps around source changes, previous problem locations, source-identity conflicts, current foreground violations and the fastest old/new model rotations in each 2 s bin; model rates are navigation only; nearby centres share one window',windows=windows)
        (work/'source_review/whole_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    info=json.loads(registration.read_text(encoding='utf8'))['primary_source']
    video=SourceVideo(info['path'],np.array([1,12800]))
    video.stream.codec_context.thread_count=1
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
    for first in range(0,len(windows),4):
        if pages is not None and first//4 not in pages:continue
        canvas=Image.new('RGB',(2880,1920),(19,24,30))
        for slot,window in enumerate(windows[first:first+4]):
            for sample,frame in enumerate(window['sample_indices']):
                original=Image.fromarray(video.read(int(raw['source_pts'][frame])))
                tile=Image.new('RGB',(960,240),(19,24,30));draw=ImageDraw.Draw(tile)
                draw.text((4,2),f"W{window['index']:02d} {raw['source_pts'][frame]/12800.:.2f}s    SOURCE                  BEFORE                   AFTER",font=font,fill='white')
                for foot in (0,1):
                    curves=[raw['region'][frame,foot],old['contour'][frame,foot],new['contour'][frame,foot]]
                    finite=[p[np.isfinite(p).all(1)] for p in curves]
                    finite=[p for p in finite if len(p)]
                    if not finite:
                        for panel in range(3):draw.text((panel*320+foot*160+2,105),'unmeasured',font=font,fill=(255,190,100))
                        continue
                    points=np.concatenate(finite);lo=np.maximum(0,np.floor(points.min(0)-18)).astype(int)
                    hi=np.minimum([1920,1080],np.ceil(points.max(0)+18)).astype(int)
                    if np.any(hi<=lo):continue
                    for panel,curve in enumerate((None,curves[1],curves[2])):
                        crop=np.array(original.crop((*lo,*hi)))
                        if curve is not None and np.isfinite(curve).all():
                            cv2.polylines(crop,[np.rint(curve-lo).astype(np.int32)],True,(255,155,35),1,cv2.LINE_AA)
                        # Allow enlargement while retaining the same bounds for all three panels.
                        scale=min(155/crop.shape[1],205/crop.shape[0])
                        image=Image.fromarray(crop).resize((max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),Image.Resampling.LANCZOS)
                        x=panel*320+foot*160
                        tile.paste(image,(x+(160-image.width)//2,28+(205-image.height)//2))
                        if not raw['visible'][frame,foot]:draw.text((x+2,221),'unmeasured',font=font,fill=(255,190,100))
                canvas.paste(tile,((sample%3)*960,(slot*2+sample//3)*240))
        path=work/f'source_review/{prefix}_{first//4}.jpg'
        canvas.save(path,quality=93)
        print(json.dumps(dict(event='turn_foot_board',page=first//4,path=str(path))),flush=True)
    video.close()


def motion_scan(prepared, source, before, parameter, work):
    from compare import bind
    from calibration.program_tracking.body_tracking.skate_physics import fk_state
    from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
    results={}
    for name,path in (('before',before),('after',parameter)):
        ref,_,_,_,_=bind(prepared,source,path)
        times=np.linspace(ref.times[0],ref.times[-1],4*(len(ref.times)-1)+1)
        local=[];feet=[];last=None;finite=True
        for first in range(0,len(times),512):
            chunk=times[first:first+512]
            state=fk_state(ref,ref.at(chunk),np.zeros((len(chunk),69)))
            R=np.concatenate([state['Q'][:,None],state['local_R']],axis=1);F=state['R']
            finite &= bool(np.isfinite(R).all() and np.isfinite(F).all() and np.isfinite(state['O']).all())
            if last is not None:R=np.r_[last[0][None],R];F=np.r_[last[1][None],F]
            local.append(np.degrees(np.linalg.norm(_log(R[:-1].swapaxes(-1,-2)@R[1:]),axis=-1)))
            feet.append(np.degrees(np.linalg.norm(_log(F[:-1].swapaxes(-1,-2)@F[1:]),axis=-1)))
            last=(R[-1],F[-1])
        local=np.concatenate(local);feet=np.concatenate(feet);clock=times[1:]+ref.p['pts'][0]/12800.
        def describe(value,selected=None):
            ix=np.arange(len(value)) if selected is None else np.flatnonzero(selected)
            v=value[ix];peak=np.unravel_index(np.argmax(v),v.shape)
            return dict(p95_degrees_per_5ms=float(np.percentile(v,95)),
                        maximum_degrees_per_5ms=float(v[peak]),maximum_source_seconds=float(clock[ix[peak[0]]]),joint_or_foot=int(peak[1]))
        windows=[]
        for lo in np.arange(22.12,188.68,2.):
            selected=(clock>=lo)&(clock<lo+2.)
            windows.append(dict(seconds=[float(lo),min(float(lo+2),188.68)],
                                local_rotations=describe(local,selected),foot_world_rotations=describe(feet,selected)))
        results[name]=dict(parameter=str(path),sha256=digest(path),source_frames=len(ref.times),
                           diagnostic_samples=len(times),all_FK_finite=finite,
                           local_rotations=describe(local),foot_world_rotations=describe(feet),windows=windows)
        print(json.dumps(dict(event='whole_turn_motion_scan',candidate=name,
                              **{k:v for k,v in results[name].items() if k!='windows'})),flush=True)
    result=dict(results=results,interpretation='5 ms trajectory diagnostics are not additional source observations or physical accuracy certification')
    (work/'source_review/continuity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def point_scan(prepared, source, before, parameter, points, work):
    from types import SimpleNamespace
    from compare import bind
    from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
    from calibration.program_tracking.body_tracking.source_motion_refit import NativeShoePoints, NativeBodyPoints, NativeBodyRegions, NativeSurfacePoints
    document=json.loads(points.read_text(encoding='utf8'));results={}
    for name,path in (('before',before),('after',parameter)):
        ref,camera,_,_,packet=bind(prepared,source,path)
        state=SequenceState(ref,packet['seconds'])
        problem=SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,
                                body=SimpleNamespace(state=state,size=state.size),size=17+state.size)
        zero=np.zeros(state.size)
        shoe_factor=NativeShoePoints(problem,document)
        shoe_metrics=shoe_factor.evaluate(zero)['metrics']['native_heel_toe_pixels']
        shoe=shoe_metrics['pixel_error']
        inferred={r['id']:r['foot'] for r in shoe_metrics['inferred_foot_association']}
        shoe_rows=[r for r in document['samples'] if r['role']=='fit']
        propagated=np.array([r.get('source_kind')=='adjacent_source_flow' for r in shoe_rows])
        temporal_inference=np.array([r.get('source_kind')=='temporal_inference_region' for r in shoe_rows])
        shoe_region=np.array(['pixel_box' in r for r in shoe_rows],bool)
        body=NativeBodyPoints(problem,document).evaluate(zero)['metrics']['reviewed_native_body_pixels']
        surface=NativeSurfacePoints(problem,document).evaluate(zero)['metrics']['reviewed_native_surface_pixels'] if document.get('surface_samples') else []
        surface_region=np.array(['pixel_box' in r for r in document.get('surface_samples',[]) if r['role']=='fit'],bool)
        surface_exclusion=np.array([r.get('hidden_joint_not_measured',False)
            for r in document.get('surface_samples',[]) if r['role']=='fit'],bool)
        body_regions=NativeBodyRegions(problem,document).evaluate(zero)['metrics']['reviewed_native_body_regions'] if document.get('body_regions') else []
        shoe_regions=NativeShoePoints(problem,dict(samples=document['shoe_regions'])).evaluate(zero)['metrics']['native_heel_toe_pixels']['pixel_error'] if document.get('shoe_regions') else []
        def stats(values):
            a=np.asarray(values)
            if not len(a):return dict(count=0)
            return dict(count=len(a),median_px=float(np.median(a)),p95_px=float(np.percentile(a,95)),maximum_px=float(np.max(a)))
        results[name]=dict(parameter=str(path),sha256=digest(path),shoe=stats(shoe),
                           shoe_direct=stats(np.asarray(shoe)[~propagated&~shoe_region]),
                           shoe_visible_region_violation=stats(np.asarray(shoe)[~propagated&shoe_region&~temporal_inference]),
                           shoe_temporal_inference_region_violation=stats(np.asarray(shoe)[temporal_inference]),
                           shoe_correlated_source_flow=stats(np.asarray(shoe)[propagated]),
                           body=stats([r['distance_px'] for r in body]),
                           surface_chords=stats([r['distance_px'] for r in surface]),
                           surface_point_chords=stats(np.asarray([r['distance_px'] for r in surface])[~surface_region]),
                           surface_region_violation=stats(np.asarray([r['distance_px'] for r in surface])[surface_region&~surface_exclusion]),
                           surface_hidden_exclusion_violation=stats(np.asarray([r['distance_px'] for r in surface])[surface_exclusion]),
                           body_exclusion_violation=stats([r['distance_px'] for r in body_regions]),
                           shoe_exclusion_violation=stats(shoe_regions),
                           new_turn_points=[dict(id=r.get('id'),pts=r['pts'],foot=inferred.get(r.get('id'),r['foot']),
                                            anatomical_identity_inferred=bool(r.get('foot_candidates')),feature=r['feature'],error_px=float(e))
                                            for r,e in zip(shoe_rows,shoe)
                                            if r.get('source_kind')!='adjacent_source_flow' and r.get('source_review_batch') in ('whole_turn_1116','whole_unmeasured_1116','whole_postreview_1116','whole_residual_1116','source_occlusion_1116','source_blade_1116','source_residual_1116','source_occlusion_final_1116','source_cuff_continuity_1116','source_lift_cuff_continuity_1116','source_lift56_continuity_1116','source_support_turn159_1116','source_temporal_boot_inference_1116')])
    path=work/'source_review/whole_review.json';report=json.loads(path.read_text(encoding='utf8'))
    report['directed_point_comparison']=results
    report['source_ambiguity_review_pts']=sorted({int(r['pts']) for r in document['samples']
        if r.get('role')=='fit' and r.get('source_review_batch') in ('source_occlusion_final_1116','source_cuff_continuity_1116','source_lift_cuff_continuity_1116','source_lift56_continuity_1116','source_support_turn159_1116','source_temporal_boot_inference_1116')})
    arc_paths={r['replacement_source']['path'] for r in document.get('source_outline_review_exclusions',[])}
    arc_pts={int(f['source_pts']) for p in arc_paths for f in json.loads(Path(p).read_text(encoding='utf-8'))['frames']}
    report['source_ambiguity_review_pts']=sorted(set(report['source_ambiguity_review_pts'])|arc_pts)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{q:v for q,v in r.items() if q!='new_turn_points'} for k,r in results.items()}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--boards', action='store_true')
    p.add_argument('--native-boards', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--project', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--foot-boards', action='store_true')
    p.add_argument('--registration', type=Path)
    p.add_argument('--pages', nargs='+', type=int)
    p.add_argument('--dense', action='store_true')
    p.add_argument('--motion-scan', action='store_true')
    p.add_argument('--before', type=Path)
    p.add_argument('--point-scan', action='store_true')
    p.add_argument('--points', type=Path)
    p.add_argument('--source-outline', type=Path)
    for name in ('prepared', 'source', 'parameter', 'output'):
        p.add_argument('--'+name, type=Path)
    args = p.parse_args()
    if args.native_boards:
        native_boards(args)
    elif args.point_scan:
        point_scan(args.prepared,args.source,args.before,args.parameter,args.points,args.work)
    elif args.motion_scan:
        motion_scan(args.prepared,args.source,args.before,args.parameter,args.work)
    elif args.foot_boards:
        foot_boards(args.work,args.registration,args.pages,args.dense)
    elif args.compare:
        compare_screens(args.work)
    elif args.project:
        project_all(args.prepared, args.source, args.parameter, args.work, args.output, args.source_outline)
    else:
        result = index(args.work, source_outline=args.source_outline)
        if args.boards:
            boards(args.work, result)
