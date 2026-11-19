"""Review heel material points on the saved native motion and original source."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import bind
from motion_overlay import project
from source_review import SourceVideo, digest
from calibration.program_tracking.body_tracking.native_mesh_overlay import silhouette
from calibration.program_tracking.body_tracking.scoped_motion_step import skate_projection
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState


def review(args):
    w = args.work
    document = json.loads(args.points.read_text(encoding='utf8'))
    source = json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    refs = [bind(args.prepared, w/'input/source.json', p) for p in (args.before, args.after)]
    hashes = [digest(p) for p in (args.before, args.after)]
    batch = [r for r in document['samples'] if r.get('source_batch') == 'source_heel_semantics_1117']
    new = [r for r in batch if r['role'] == 'fit']
    scores = []
    for ref, camera, _, _, packet in refs:
        state = SequenceState(ref, packet['seconds'])
        problem = SimpleNamespace(reference=ref, camera=camera, binding=ref.binding,
            body=SimpleNamespace(state=state), analytic_rigid_shoe_derivatives=True,
            fixed_relative_skates=True)
        uv, _ = skate_projection(problem, np.r_[ref.binding.x, np.zeros(state.size)], new)
        boxes = np.asarray([r['pixel_box'] for r in new])
        error = np.linalg.norm(uv-np.clip(uv, boxes[:, 0], boxes[:, 1]), axis=-1)
        scores.append((uv, error))
    report = dict(before_sha256=hashes[0], after_sha256=hashes[1],
        points_sha256=digest(args.points), source_video_sha256=source['sha256'],
        superseded_source_rows_excluded_from_fit_statistics=[r.get('id') for r in batch if r['role'] != 'fit'],
        rows=[dict(id=r['id'], pts=r['pts'], foot=r['foot'], feature=r['feature'],
            source_box=r['pixel_box'], before_prediction=scores[0][0][i].tolist(),
            after_prediction=scores[1][0][i].tolist(), before_region_error=float(scores[0][1][i]),
            after_region_error=float(scores[1][1][i])) for i, r in enumerate(new)],
        interpretation='Development source-region fit, not independent visual or physical accuracy',
        visual_review='pending', human_confirmed=False)
    for name, (_, errors) in zip(('before', 'after'), scores):
        report[name] = dict(count=len(errors), median_px=float(np.median(errors)),
            p95_px=float(np.percentile(errors, 95)), maximum_px=float(np.max(errors)))

    navigation = json.loads((w/'source_review/current_support_review.json').read_text(encoding='utf8'))
    frames = {(r['pts'], r['foot']): r for r in navigation['raw_source_frames']}
    # Add adjacent source exposures around the newly read rear views elsewhere.
    for row in new:
        if not (81.6 <= row['pts']/12800 <= 93.4):
            for d in (-1024, 0, 1024):
                frames.setdefault((row['pts']+d, row['foot']), dict(pts=row['pts']+d, foot=row['foot']))
    selected = sorted(frames)
    if args.seconds:
        selected = [r for r in selected if args.seconds[0] <= r[0]/12800 <= args.seconds[1]]
    obs = dict(np.load(w/'input/body_track_source.npz'))
    pts = refs[0][0].p['pts']
    reader = SourceVideo(source['path'], [1, 12800])
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 17)
    pages = []
    # The fast foot-only call must return exactly the same actual meshes.
    check_frames = np.searchsorted(pts, np.array([1058560, 1142784, 2095360]))
    for ref, camera, *_ in refs:
        full = project(ref, camera, check_frames)
        feet = project(ref, camera, check_frames, include_body=False)
        for a, b in zip(full[2], feet[2]):
            np.testing.assert_array_equal(a, b)
    report['foot_only_projection_matches_full_renderer'] = True
    try:
        for first in range(0, len(selected), 24):
            picks = selected[first:first+24]
            indices = np.searchsorted(pts, [p for p, _ in picks])
            np.testing.assert_array_equal(pts[indices], [p for p, _ in picks])
            projected = [project(ref, camera, indices, include_body=False) for ref, camera, *_ in refs]
            canvas = Image.new('RGB', (2304, 2320), (19, 24, 30))
            draw = ImageDraw.Draw(canvas)
            for j, ((stamp, foot), index) in enumerate(zip(picks, indices)):
                original = reader.read(stamp)
                q = np.flatnonzero(obs['model_joint_ids'][index] == 15+foot)
                centre = obs['uv'][index, q[0]]
                extents = [centre+[[-72., -40.], [72., 110.]]]
                # Source material readings can lie beyond a missed detector
                # ankle. Keep those original pixels in the common crop too.
                extents.extend(np.asarray(r['pixel_box'], float).reshape(2, 2)
                    for r in new if r['pts'] == stamp and r['foot'] == foot)
                extents.extend(np.asarray(r['extent_box'], float).reshape(2, 2)
                    for r in document.get('source_free_entry_frames', [])
                    if r['pts'] == stamp and r.get('foot', 0) == foot
                    and r.get('extent_box') is not None)
                for _, _, shoes, _ in projected:
                    extents.extend(s[j] for s in shoes[2*foot:2*foot+2])
                xy = np.concatenate(extents); xy = xy[np.isfinite(xy).all(1)]
                lo = np.maximum([0, 0], np.floor(xy.min(0)-5)).astype(int)
                hi = np.minimum([1920, 1080], np.ceil(xy.max(0)+5)).astype(int)
                raw = original[lo[1]:hi[1], lo[0]:hi[0]]
                panels = [Image.fromarray(raw)]
                for _, _, shoes, faces in projected:
                    rgb = raw.copy()
                    for vertices, triangles in zip(shoes[2*foot:2*foot+2], faces[2*foot:2*foot+2]):
                        mask = silhouette(vertices[j]-lo, triangles, rgb.shape[:2])
                        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rgb, contour, -1, (255, 155, 35), 1)
                    panels.append(Image.fromarray(rgb))
                x, y = (j % 3)*768, (j//3)*290
                draw.text((x+5, y+3), f'{stamp/12800:.2f}s foot {foot}  SOURCE / BEFORE / AFTER', font=font, fill='white')
                for k, panel in enumerate(panels):
                    scale = min(252/panel.width, 260/panel.height)
                    panel = panel.resize((round(panel.width*scale), round(panel.height*scale)),
                                         Image.Resampling.LANCZOS)
                    canvas.paste(panel, (x+k*256+(256-panel.width)//2, y+27))
            page = first//24
            path = w/f'source_review/current_support_comparison_{page}.jpg'
            canvas.save(path, quality=95)
            pages.append(dict(path=str(path), sha256=digest(path), samples=[dict(pts=p, foot=f) for p, f in picks]))
            print(json.dumps(dict(page=page, samples=len(picks), path=str(path))), flush=True)
    finally:
        reader.close()
    assert hashes == [digest(p) for p in (args.before, args.after)], 'Candidate changed during review'
    report['pages'] = pages
    (w/'source_review/current_support_point_comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ('before_sha256', 'after_sha256', 'before', 'after')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'prepared', 'points', 'registration', 'before', 'after'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--seconds', type=float, nargs=2)
    review(parser.parse_args())
