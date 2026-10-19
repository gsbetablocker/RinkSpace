"""Propose broadcast edit boundaries from the aligned 30 fps source proxy.

This is a source-only scout. Its candidates are not reviewed cuts or camera
assignments. All new frame evidence is written under the caller's --out path.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PROXY = _resolve_asset(ROOT, ROOT / 'data/input/oxXu1avhrvs.f134.mp4')
EXISTING_SAMPLES = _resolve_asset(ROOT, ROOT / 'outputs/calibration/ankara_2026_v015_joint/full_event_1000_candidates/candidates.json')
TIME_BASE = Fraction(1, 15360)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clock(pts: int) -> str:
    millis = round(pts * 1000 / 15360)
    hours, rem = divmod(millis, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, ms = divmod(rem, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}'


def decode_range(source: Path, start_pts: int, end_pts: int):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 3
        if stream.time_base != TIME_BASE:
            raise ValueError(f'Unexpected time base: {stream.time_base}')
        container.seek(max(0, start_pts - 15360), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < start_pts:
                continue
            if frame.pts >= end_pts:
                break
            yield frame


def metrics(source: Path, start_pts: int, end_pts: int):
    pts, mads, hists = [], [], []
    mask = np.ones((144, 256), np.uint8)
    mask[:38, :75] = 0
    mask[125:, 215:] = 0
    valid = mask.astype(bool)
    previous_gray = previous_hist = None
    for frame in decode_range(source, start_pts, end_pts):
        rgb = frame.reformat(width=256, height=144, format='rgb24').to_ndarray()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
        hist = cv2.calcHist([rgb], [0, 1, 2], mask, [8, 8, 8], [0, 256] * 3).ravel()
        hist /= max(float(hist.sum()), 1)
        pts.append(int(frame.pts))
        mads.append(0. if previous_gray is None else float(np.abs(gray - previous_gray)[valid].mean()))
        hists.append(0. if previous_hist is None else float(np.abs(hist - previous_hist).sum() / 2))
        previous_gray, previous_hist = gray, hist
    return np.asarray(pts, dtype=np.int64), np.asarray(mads), np.asarray(hists)


def propose(pts, mads, hists, core_start_pts: int, core_end_pts: int):
    potential = []
    for index in range(1, len(pts) - 1):
        if not core_start_pts <= pts[index] < core_end_pts:
            continue
        nearby = np.r_[mads[max(1, index - 15):index], mads[index + 1:min(len(mads), index + 16)]]
        ratio = float(mads[index] / max(float(np.median(nearby)), .35))
        if ((mads[index] > 6 and ratio > 2.5) or
                (mads[index] > 10 and hists[index] > .10) or
                (hists[index] > .20 and mads[index] > 4)):
            potential.append((index, ratio))
    clusters = []
    for item in potential:
        if clusters and pts[item[0]] - pts[clusters[-1][-1][0]] < round(.22 * 15360):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    candidates = []
    for cluster in clusters:
        index, ratio = max(cluster, key=lambda row: mads[row[0]])
        candidates.append({
            'proposal_pts': int(pts[index]),
            'before_pts': int(pts[index - 1]),
            'after_pts': int(pts[index]),
            'proposal_time': clock(int(pts[index])),
            'transition_extent_pts': [int(pts[cluster[0][0]]), int(pts[cluster[-1][0]])],
            'score': {'mad_256x144': float(mads[index]), 'rgb_hist_distance': float(hists[index]),
                      'local_mad_ratio': ratio},
            'source_track': 'f134',
            'refinement_status': 'proxy_proposal_unreviewed',
        })
    return candidates


def save_target_frames(source: Path, start_pts: int, end_pts: int, targets: set[int], frame_dir: Path):
    frame_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    remaining = set(targets)
    for frame in decode_range(source, start_pts, end_pts):
        if frame.pts not in remaining:
            continue
        path = frame_dir / f'pts_{frame.pts:012}.jpg'
        image = frame.to_ndarray(format='bgr24')
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 87]):
            raise OSError(f'Cannot write {path}')
        found[int(frame.pts)] = str(path)
        remaining.remove(frame.pts)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(f'Missing target frames: {sorted(remaining)[:8]}')
    return found


def contact_sheets(candidates: list[dict], out: Path):
    font_path = Path('C:/Windows/Fonts/consola.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    pages = []
    sheet_dir = out / 'contacts'
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for page in range((len(candidates) + 11) // 12):
        subset = candidates[page * 12:(page + 1) * 12]
        canvas = Image.new('RGB', (1280, 392 * len(subset)), '#14202b')
        draw = ImageDraw.Draw(canvas)
        for row, item in enumerate(subset):
            y = row * 392
            draw.text((9, y + 5), f"C{page * 12 + row + 1:04}  {item['proposal_time']}   MAD {item['score']['mad_256x144']:.1f}   H {item['score']['rgb_hist_distance']:.2f}", font=font, fill='white')
            with Image.open(item['before_image']) as before, Image.open(item['after_image']) as after:
                canvas.paste(before, (0, y + 32))
                canvas.paste(after, (640, y + 32))
        path = sheet_dir / f'cuts_{page + 1:03}.jpg'
        canvas.save(path, quality=88)
        pages.append(str(path))
    return pages


def existing_samples(start_pts: int, end_pts: int, source: Path, chunk_out: Path):
    existing = json.loads(EXISTING_SAMPLES.read_text(encoding='utf-8')) if EXISTING_SAMPLES.is_file() else {'candidates': []}
    rows = [
        {'source_pts': row['source_pts'], 'image_path': str(ROOT / row['image_path']),
         'source_track': 'f134', 'authority': 'existing_source_proxy_sample'}
        for row in existing['candidates']
        if start_pts <= row['source_pts'] < end_pts and (ROOT / row['image_path']).is_file()
    ]
    present = {row['source_pts'] for row in rows}
    needed = {second * 15360 for second in range(start_pts // 15360, end_pts // 15360, 4)} - present
    missing_files = {pts for pts in needed if not (chunk_out / 'samples' / f'pts_{pts:012}.jpg').is_file()}
    if missing_files:
        save_target_frames(source, start_pts, end_pts, missing_files, chunk_out / 'samples')
    rows.extend({'source_pts': pts,
                 'image_path': str(chunk_out / 'samples' / f'pts_{pts:012}.jpg'),
                 'source_track': 'f134', 'authority': 'new_source_proxy_sample_for_unsampled_start'}
                for pts in sorted(needed))
    return sorted(rows, key=lambda row: row['source_pts'])


def chunks(start: int, end: int, chunk_size: int):
    cursor = start
    while cursor < end:
        next_boundary = ((cursor // chunk_size) + 1) * chunk_size
        stop = min(end, next_boundary)
        yield cursor, stop
        cursor = stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, required=True, help='source seconds, inclusive')
    parser.add_argument('--end', type=int, required=True, help='source seconds, exclusive')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=PROXY)
    parser.add_argument('--chunk-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 19560:
        parser.error('Invalid source time interval')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_hash = sha256(args.source)
    source_size = args.source.stat().st_size
    settings = {'source_sha256': source_hash, 'start': args.start, 'end': args.end,
                'chunk_seconds': args.chunk_seconds, 'detector_version': 1}
    index = {'schema_version': 'rinkspace.broadcast_cut_scan.v1', 'source_id': 'oxXu1avhrvs',
             'source_path': str(args.source.resolve()), 'source_sha256': source_hash,
             'source_size_bytes': source_size, 'time_base': str(TIME_BASE),
             'source_track': 'f134', 'range_pts': [args.start * 15360, args.end * 15360],
             'chunk_boundary_is_not_a_cut': True,
             'method': '30 fps proxy all-frame adjacent differences; proposals only; 4-second periodic source frames reused; full-resolution 60 fps review required for final boundaries.',
             'chunks': []}
    for core_start, core_end in chunks(args.start, args.end, args.chunk_seconds):
        chunk_out = out / 'chunks' / f'chunk_{core_start:06}_{core_end:06}'
        chunk_file = chunk_out / 'scan.json'
        if chunk_file.exists():
            old = json.loads(chunk_file.read_text(encoding='utf-8'))
            if old.get('settings') == settings and old.get('core_start_pts') == core_start * 15360 and old.get('core_end_pts') == core_end * 15360 and all(Path(c['before_image']).is_file() and Path(c['after_image']).is_file() for c in old['candidates']):
                refreshed_samples = existing_samples(old['core_start_pts'], old['core_end_pts'], args.source, chunk_out)
                if old['samples'] != refreshed_samples:
                    old['samples'] = refreshed_samples
                    save_json(chunk_file, old)
                index['chunks'].append({'scan_path': str(chunk_file), 'core_start_pts': old['core_start_pts'],
                                        'core_end_pts': old['core_end_pts'], 'candidate_count': len(old['candidates'])})
                print(json.dumps({'chunk': [core_start, core_end], 'reused': True, 'candidates': len(old['candidates'])}), flush=True)
                continue
        scan_start_pts = max(0, (core_start - 1) * 15360)
        scan_end_pts = min(19560, core_end + 1) * 15360
        pts, mads, hists = metrics(args.source, scan_start_pts, scan_end_pts)
        candidates = propose(pts, mads, hists, core_start * 15360, core_end * 15360)
        target_pts = {c[key] for c in candidates for key in ('before_pts', 'after_pts')}
        images = save_target_frames(args.source, scan_start_pts, scan_end_pts, target_pts, chunk_out / 'frames') if target_pts else {}
        for candidate in candidates:
            candidate['before_image'] = images[candidate['before_pts']]
            candidate['after_image'] = images[candidate['after_pts']]
        pages = contact_sheets(candidates, chunk_out)
        scan = {'schema_version': 'rinkspace.broadcast_cut_scan.v1', 'settings': settings,
                'source_path': str(args.source.resolve()), 'source_sha256': source_hash,
                'time_base': str(TIME_BASE), 'core_start_pts': core_start * 15360,
                'core_end_pts': core_end * 15360, 'scan_start_pts': scan_start_pts,
                'scan_end_pts': scan_end_pts, 'decoded_frames': len(pts),
                'chunk_boundary_is_not_a_cut': True,
                'pts_step_counts': {str(int(step)): int(count) for step, count in zip(*np.unique(np.diff(pts), return_counts=True))},
                'sample_interval_seconds': 4, 'samples': existing_samples(core_start * 15360, core_end * 15360, args.source, chunk_out),
                'candidates': candidates, 'contact_sheets': pages,
                'authority': 'numeric_cut_proposals_only_no_camera_identity'}
        save_json(chunk_file, scan)
        index['chunks'].append({'scan_path': str(chunk_file), 'core_start_pts': scan['core_start_pts'],
                                'core_end_pts': scan['core_end_pts'], 'candidate_count': len(candidates)})
        save_json(out / 'index.json', index)
        print(json.dumps({'chunk': [core_start, core_end], 'decoded': len(pts),
                          'candidates': len(candidates), 'samples_reused': len(scan['samples']),
                          'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    index['complete'] = True
    index['candidate_count'] = sum(c['candidate_count'] for c in index['chunks'])
    index['elapsed_seconds'] = round(time.monotonic() - started, 1)
    save_json(out / 'index.json', index)
    print(json.dumps({'complete': True, 'candidates': index['candidate_count'],
                      'elapsed_seconds': index['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
