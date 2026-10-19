"""Preserve a user's video annotation and bind it to exact source/render frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av
import cv2
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_frame(path: Path, pts: int):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if str(stream.time_base) != '1/15360':
            raise ValueError('Expected the explicitly bound source time base')
        container.seek(pts, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts == pts:
                return frame.to_ndarray(format='bgr24')
            if frame.pts > pts:
                break
    raise ValueError(f'Exact presentation {pts} is missing from {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--issue-id', required=True)
    parser.add_argument('--pts', type=int, required=True)
    parser.add_argument('--time', required=True)
    parser.add_argument('--quote', required=True)
    parser.add_argument('--feature', action='append', required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding='utf-8-sig'))
    if any(row['issue_id'] == args.issue_id for row in ledger['issues']):
        raise ValueError('Issue already registered; preserve its original evidence')
    folder = Path(ledger['issues'][0]['attachment']['path']).parent
    attachment = folder / f'{args.issue_id}_user_arrows.png'
    if attachment.exists():
        if digest(attachment) != digest(args.image):
            raise ValueError('Existing attachment differs')
    else:
        shutil.copyfile(args.image, attachment)
    image = cv2.imread(str(attachment))
    if image is None:
        raise ValueError('Unreadable attachment')
    native_path = folder / f'{args.issue_id}_source_native_{args.pts}.png'
    render_path = folder / f'{args.issue_id}_current_c134_video_frame.png'
    if not cv2.imwrite(str(native_path), exact_frame(args.source, args.pts)):
        raise OSError('Cannot preserve original RGB frame')
    video = Path(ledger['reviewed_video'])
    receipt = json.loads((video.parent / 'render.json').read_text(encoding='utf-8-sig'))
    local_pts = args.pts - receipt['video_production']['source_pts_first']
    frame = exact_frame(video, local_pts)
    if not cv2.imwrite(str(render_path), frame):
        raise OSError('Cannot preserve rendered frame')
    resized = cv2.resize(frame, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    difference = np.abs(resized.astype(np.int16) - image.astype(np.int16))
    row = dict(issue_id=args.issue_id, user_quote=args.quote,
        displayed_source_time=args.time, native_source_pts=args.pts,
        time_base='1/15360', reported_model_features=args.feature,
        attachment=dict(path=str(attachment), sha256=digest(attachment),
            original_path=str(args.image), pixel_size_wh=[image.shape[1], image.shape[0]]),
        exact_original_RGB=dict(path=str(native_path), sha256=digest(native_path)),
        exact_current_video_frame=dict(path=str(render_path), sha256=digest(render_path),
            source_pts=args.pts), human_visual_report_authority=True,
        new_FIT_UV_from_screenshot=False,
        arrows_are_issue_locations_not_exact_boundary_measurements=True,
        common_cause_not_yet_proven=True, status='unresolved_user_report')
    ledger['issues'].append(row)
    ledger.setdefault('screenshot_current_video_pixel_comparisons', []).append(dict(
        issue_id=args.issue_id,
        linear_resize_mean_absolute_pixel_difference=float(difference.mean()),
        median=float(np.median(difference)),
        includes_user_red_arrows_and_resize_compression=True))
    args.ledger.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(issue_id=args.issue_id, source_pts=args.pts,
        pixel_difference_median=float(np.median(difference))), ensure_ascii=False))


if __name__ == '__main__':
    main()
