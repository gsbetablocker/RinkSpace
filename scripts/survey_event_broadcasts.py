"""Acquire bounded official proxies and continuously survey the eight broadcasts."""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
from pathlib import Path
import shutil
import time
import urllib.request
from datetime import datetime, timezone

import av
import cv2
import numpy as np
from fractions import Fraction

from rinkspace.event_survey import dump, file_hash, survey_chunk, summarize_source, coverage_intervals, covers_interval
from rinkspace.storage import AssetStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = AssetStore(ROOT).directory(
    'outputs/event_survey_v11', 'survey/event_survey_v11',
)


class QuietLogger:
    def debug(self, message): pass
    def warning(self, message): pass
    def error(self, message): pass


def extract_highres(source: dict, output: Path, seconds: list[float]) -> dict:
    """Decode requested official HD PTS from hashed public byte ranges.

    Partial range provenance deliberately does not claim a whole-source SHA256.
    """
    import yt_dlp
    import hashlib
    from PIL import Image
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'logger': QuietLogger(),
                          'socket_timeout': 25, 'retries': 1}) as downloader:
        info = downloader.extract_info(source['url'], download=False)
    if info.get('channel_id') != source['channel_id']:
        raise ValueError('Returned video channel does not match the verified event source registry')
    formats = [f for f in info['formats'] if f.get('height') == 1080 and f.get('filesize') and f.get('acodec') == 'none']
    fmt = next((f for f in formats if f['format_id'] == '399'), formats[0])
    folder = output / source['source_id'] / f'highres_f{fmt["format_id"]}'
    folder.mkdir(parents=True, exist_ok=True)
    total = int(fmt['filesize'])
    headers = info.get('http_headers', {})

    class RangeReader(io.RawIOBase):
        def __init__(self):
            self.position = 0
            self.used = set()
            self.chunk_size = 2 * 1024 * 1024
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.position
        def seek(self, offset, whence=0):
            self.position = offset if whence == 0 else self.position + offset if whence == 1 else total + offset
            return self.position
        def read(self, size=-1):
            end = min(total, self.position + (size if size >= 0 else total))
            blocks = []
            while self.position < end:
                base = self.position // self.chunk_size * self.chunk_size
                last = min(base + self.chunk_size, total) - 1
                path = folder / f'bytes_{base:012d}.part'
                if not path.exists() or path.stat().st_size != last - base + 1:
                    for attempt in range(3):
                        try:
                            request = urllib.request.Request(fmt['url'], headers={**headers, 'Range': f'bytes={base}-{last}'})
                            with urllib.request.urlopen(request, timeout=18) as response:
                                body = response.read(last - base + 2)
                                content_range = response.headers.get('Content-Range', '')
                            if len(body) != last - base + 1 or content_range != f'bytes {base}-{last}/{total}':
                                raise RuntimeError('Official HD range response mismatch')
                            path.write_bytes(body)
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                self.used.add(base)
                data = path.read_bytes()
                offset = self.position - base
                count = min(end - self.position, len(data) - offset)
                blocks.append(data[offset:offset + count])
                self.position += count
            return b''.join(blocks)
    reader = RangeReader()
    previous_manifest = folder / 'manifest.json'
    old = json.loads(previous_manifest.read_text(encoding='utf-8')) if previous_manifest.exists() else None
    if old and (old['original_size_bytes'] != total or old['format_id'] != fmt['format_id']):
        raise ValueError('Official HD representation changed; existing range evidence was not overwritten')
    rows = []
    with av.open(reader, format='mp4') as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        for wanted in sorted(seconds):
            container.seek(int(wanted / stream.time_base), stream=stream, backward=True)
            frame = next(f for f in container.decode(stream) if f.pts is not None and float(f.pts * f.time_base) >= wanted)
            image_path = folder / f'pts_{frame.pts:012d}.jpg'
            frame.to_image().save(image_path, quality=98)
            rows.append({'source_id': source['source_id'], 'source_url': source['url'],
                         'session_id': source['session_id'], 'background_epoch': source['background_epoch'],
                         'format_id': fmt['format_id'], 'requested_seconds': wanted,
                         'source_pts': frame.pts, 'time_base': str(frame.time_base),
                         'source_time_seconds': float(frame.pts * frame.time_base),
                         'image_size': [frame.width, frame.height], 'image_path': image_path.as_posix(),
                         'source_sha256': None, 'whole_source_hash_status': 'unavailable_partial_byte_ranges_only',
                         'image_sha256': file_hash(image_path), 'original_time_offset_seconds': 0,
                         'geometry_eligibility': 'surveyed_withheld_for_geometry' if source['source_id'] == 'Rd66CneD9io' else 'requires_same_venue_visual_review'})
            print(f'HD_VIEW {source["source_id"]} {float(frame.pts * frame.time_base)} {image_path}', flush=True)
    if old:
        rows = sorted({row['source_pts']: row for row in old['frames'] + rows}.values(), key=lambda r: r['source_pts'])
        reader.used.update(row['byte_offset'] for row in old['validated_original_byte_ranges'])
    ranges = [{'byte_offset': start, 'byte_length': (folder / f'bytes_{start:012d}.part').stat().st_size,
               'sha256': file_hash(folder / f'bytes_{start:012d}.part'),
               'path': (folder / f'bytes_{start:012d}.part').as_posix()} for start in sorted(reader.used)]
    record = {'schema_version': 'rinkspace.partial-original-track-frames.v1', 'source_id': source['source_id'],
              'source_url': source['url'], 'format_id': fmt['format_id'], 'original_size_bytes': total,
              'whole_source_download_complete': False, 'source_sha256': None,
              'range_set_sha256': hashlib.sha256(json.dumps(ranges, sort_keys=True).encode()).hexdigest(),
              'validated_original_byte_ranges': ranges, 'frames': rows,
              'limitations': ['Exact decoded source PTS from original HD track; complete original-source SHA256 is not available.',
                              'JPEG is an extracted visualization; immutable original byte ranges are retained for repeat decode.']}
    dump(folder / 'manifest.json', record)
    return record


def acquire(source: dict, output: Path, workers: int = 4) -> Path | None:
    """Public original format-133 bytes; only signed URLs are kept in RAM."""
    import yt_dlp
    folder = output / source['source_id']
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{source["source_id"]}.f133.mp4'
    record = {'source_id': source['source_id'], 'url': source['url'],
              'authorization': 'User authorized full official event sources for this local research task.',
              'format_id': '133', 'status': 'acquiring', 'input_modified': False}
    record['attempt_started_at_utc'] = datetime.now(timezone.utc).isoformat()
    previous_record = folder / 'acquisition.json'
    if previous_record.exists():
        previous = json.loads(previous_record.read_text(encoding='utf-8'))
        record['previous_attempts'] = previous.get('previous_attempts', []) + [
            {key: previous.get(key) for key in ['attempt_started_at_utc', 'status', 'error_type', 'error']}]
    dump(previous_record, record)
    print(f'ACQUIRE_START {source["source_id"]}', flush=True)
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'logger': QuietLogger(),
                              'socket_timeout': 25, 'retries': 1}) as downloader:
            info = downloader.extract_info(source['url'], download=False)
        if info.get('channel_id') != source['channel_id']:
            raise ValueError('Returned video channel does not match the verified event source registry')
        fmt = next(row for row in info['formats'] if row['format_id'] == '133')
        total = int(fmt['filesize'])
        record.update({k: info.get(k) for k in ['title', 'channel_id', 'duration', 'upload_date']})
        record.update({'size_bytes': total, 'width': fmt['width'], 'height': fmt['height'],
                       'fps': fmt['fps'], 'vcodec': fmt['vcodec']})
        if target.exists() and target.stat().st_size != total:
            raise ValueError('Existing proxy differs from official byte count; not overwritten')
        if not target.exists():
            parts = folder / 'download_parts'
            parts.mkdir(exist_ok=True)
            size = 4 * 1024 * 1024
            headers = info.get('http_headers', {})
            def get_range(start):
                end = min(start + size, total) - 1
                part = parts / f'{start:012d}.part'
                if part.exists() and part.stat().st_size == end - start + 1:
                    return part
                for attempt in range(4):
                    try:
                        request = urllib.request.Request(fmt['url'], headers={**headers, 'Range': f'bytes={start}-{end}'})
                        with urllib.request.urlopen(request, timeout=35) as response:
                            body = response.read(end - start + 2)
                            content_range = response.headers.get('Content-Range', '')
                        if len(body) != end - start + 1 or content_range != f'bytes {start}-{end}/{total}':
                            raise ValueError('Unexpected public byte-range response')
                        part.write_bytes(body)
                        return part
                    except Exception as exc:
                        if attempt == 3:
                            reason = str(getattr(exc, 'reason', '')).split('https://')[0][:160]
                            raise RuntimeError(f'Public video byte range {start}-{end} failed after 4 attempts: {type(exc).__name__}, HTTP status {getattr(exc, "code", None)}, reason {reason}') from None
                        time.sleep(1)
            starts = list(range(0, total, size))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for index, _ in enumerate(pool.map(get_range, starts), 1):
                    if index % 10 == 0 or index == len(starts):
                        print(f'ACQUIRE {source["source_id"]} {index}/{len(starts)} ranges', flush=True)
            pending = target.with_suffix('.assembled')
            with pending.open('wb') as writer:
                for start in starts:
                    with (parts / f'{start:012d}.part').open('rb') as reader:
                        shutil.copyfileobj(reader, writer)
            pending.replace(target)
        record.update({'status': 'download_complete_not_yet_visually_surveyed',
                       'local_path': target.as_posix(), 'source_sha256': file_hash(target)})
    except Exception as exc:
        record.update({'status': 'failed', 'error_type': type(exc).__name__,
                       'error': str(exc).split('https://')[0][:500]})
        target = None
    dump(folder / 'acquisition.json', record)
    print(f'ACQUIRE_RESULT {source["source_id"]} {record["status"]}', flush=True)
    return target


def run_source(source: dict, output: Path, media: Path, chunk_seconds: int) -> dict:
    with av.open(str(media)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
    digest = file_hash(media)
    existing = sorted((output / source['source_id'] / 'chunks').glob('*/manifest.json'))
    if existing:
        first = json.loads(existing[0].read_text(encoding='utf-8'))
        first_start, first_end = first['requested_interval_seconds']
        if first['source_sha256'] != digest:
            raise ValueError('Existing survey uses different source bytes; choose a new output directory')
        if first_start != 0:
            raise ValueError('Existing survey chunk layout does not begin at zero; use a separate output directory')
        # Resume the established grid; changing 300s to 600s would leave overlapping
        # old chunks and double-count PTS in the source summary.
        established_step = int(first_end - first_start)
        if first_end < duration and established_step != chunk_seconds:
            print(f'RESUME_GRID {source["source_id"]} retaining {established_step}s chunks', flush=True)
            chunk_seconds = established_step
    for start in range(0, math.ceil(duration), chunk_seconds):
        result = survey_chunk(source, media, output, float(start), min(start + chunk_seconds, duration), digest)
        print(f'SURVEY {source["source_id"]} {start}-{min(start + chunk_seconds, duration)} frames={result["decoded_frame_count"]} candidates={result["candidate_count"]} complete={result["complete"]}', flush=True)
        summarize_source(source, output, duration, digest)
    return summarize_source(source, output, duration, digest)


def audit_sources(sources: list[dict], output: Path) -> dict:
    """Recompute actual coverage from every saved decoded PTS, independently of flags."""
    rows = []
    for source in sources:
        folder = output / source['source_id']
        paths = sorted((folder / 'chunks').glob('*/manifest.json'))
        arrays, chunk_audits = [], []
        if not paths:
            rows.append({'source_id': source['source_id'], 'all_pts_coverage_complete': False, 'reason': 'No continuous decode chunks'})
            continue
        first_manifest = json.loads(paths[0].read_text(encoding='utf-8'))
        media = Path(first_manifest['source_path'])
        with av.open(str(media)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
            step = 1 / float(stream.average_rate)
            tb = stream.time_base
        for path in paths:
            chunk = json.loads(path.read_text(encoding='utf-8'))
            pts_path = path.parent / 'decoded_frame_pts_and_cut_scores.npz'
            with np.load(pts_path) as archive:
                pts = archive['source_pts']
                scores = archive['cut_score']
            intervals = coverage_intervals(pts, Fraction(chunk['source_time_base']), step)
            complete = chunk['error'] is None and covers_interval(intervals, *chunk['requested_interval_seconds'])
            matches = bool(len(pts) == chunk['decoded_frame_count'] and len(scores) == len(pts)
                           and intervals == chunk['decoded_intervals_seconds'] and np.isfinite(scores).all())
            arrays.append(pts)
            chunk_audits.append({'manifest': path.as_posix(), 'pts_archive_sha256': file_hash(pts_path),
                                 'decoded_count': len(pts), 'record_matches_saved_pts': matches,
                                 'recomputed_complete': complete})
            # Repair only the completion boolean/limitations, retaining all observed inputs.
            chunk['complete'] = complete
            limitations = chunk['limitations']
            note = 'Candidate extraction ranks one-second samples and detected cuts; sub-second views without a detected cut can be absent.'
            if note not in limitations:
                limitations.append(note)
            dump(path, chunk)
        all_pts = np.concatenate(arrays)
        coverage = coverage_intervals(all_pts, tb, step)
        summary = summarize_source(source, output, duration, first_manifest['source_sha256'])
        rows.append({'source_id': source['source_id'], 'duration_seconds': duration,
                     'decoded_frame_count': len(all_pts), 'source_time_base': str(tb),
                     'first_decoded_pts': int(all_pts[0]), 'last_decoded_pts': int(all_pts[-1]),
                     'decoded_intervals_seconds': coverage,
                     'all_pts_coverage_complete': covers_interval(coverage, 0, duration),
                     'all_chunk_records_match_saved_pts': all(c['record_matches_saved_pts'] for c in chunk_audits),
                     'one_second_feature_count': summary['one_second_feature_count'],
                     'cut_candidate_count': summary['cut_candidate_count'], 'candidate_count': summary['candidate_count'],
                     'geometry_eligibility': summary['geometry_eligibility'], 'chunks': chunk_audits})
    result = {'schema_version': 'rinkspace.all-frame-survey-audit.v1', 'sources': rows,
              'all_eight_sources_continuously_decoded': len(rows) == 8 and all(r['all_pts_coverage_complete'] for r in rows),
              'decoded_frame_count': sum(r.get('decoded_frame_count', 0) for r in rows),
              'actual_source_duration_seconds': sum(r.get('duration_seconds', 0) for r in rows),
              'human_viewing_complete': False, 'all_useful_views_found': False,
              'limitations': ['Low-cost per-frame cut screening plus one-second image features is algorithmic visual screening, not full human viewing.',
                             'Physical camera identity and geometric usefulness require candidate-specific review.',
                             'Entire women FS broadcast is withheld from geometry because CETIN holdout overlap is not yet excluded.']}
    dump(output / 'coverage_audit.json', result)
    return result


def build_review_panels(sources: list[dict], output: Path) -> list[dict]:
    """Show actual, diverse candidate pixels from every source for human review.

    Appearance clusters propose review diversity, never physical camera identity.
    The withheld source is sampled at fixed times without image-based selection.
    """
    from PIL import Image, ImageDraw
    review = output / 'all_source_review_v2'
    review.mkdir(exist_ok=True)
    results = []
    cv2.setRNGSeed(20260921)
    for source in sources:
        reviewed_path = review / f'{source["source_id"]}_review.json'
        if reviewed_path.exists():
            reviewed = json.loads(reviewed_path.read_text(encoding='utf-8'))
            if reviewed.get('visual_reviewed_count', 0) > 0:
                # Do not overwrite observations/acceptance decisions when a user
                # reruns candidate-panel generation. New selection needs a new revision.
                results.append(reviewed)
                continue
        folder = output / source['source_id']
        summary_path = folder / 'summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if not summary.get('continuous_decode_complete'):
            continue
        candidates = json.loads((folder / 'candidates.json').read_text(encoding='utf-8'))
        if source['source_id'] == 'Rd66CneD9io':
            first = summary['chunks'][0]
            media = Path(first['source_path'])
            selected = []
            with av.open(str(media)) as container:
                stream = container.streams.video[0]
                for index in range(16):
                    wanted = (index + .5) * summary['duration_seconds'] / 16
                    container.seek(int(wanted / stream.time_base), stream=stream, backward=True)
                    frame = next(f for f in container.decode(stream) if float(f.pts * f.time_base) >= wanted)
                    path = review / f'{source["source_id"]}_{frame.pts}.jpg'
                    frame.to_image().save(path, quality=93)
                    selected.append({'source_id': source['source_id'], 'source_sha256': first['source_sha256'],
                                     'source_path': str(media.as_posix()), 'source_pts': frame.pts,
                                     'time_base': str(frame.time_base), 'source_time_seconds': float(frame.pts * frame.time_base),
                                     'image_path': path.as_posix(), 'image_size': [frame.width, frame.height],
                                     'selection_reason': 'Fixed 16 equal-time stratum midpoints; no visual optimization',
                                     'geometry_eligibility': 'surveyed_withheld_for_geometry'})
        else:
            shot_spans = {}
            for path in sorted((folder / 'chunks').glob('*/one_second_features.json')):
                for feature_row in json.loads(path.read_text(encoding='utf-8')):
                    sid = feature_row['shot_id']
                    time_s = feature_row['source_time_seconds']
                    span = shot_spans.setdefault(sid, [time_s, time_s])
                    span[0], span[1] = min(span[0], time_s), max(span[1], time_s)
            pool = [r for r in candidates if r['source_time_seconds'] >= 600
                    and .4 < r['features']['ice_candidate_fraction'] < .985
                    and r['features']['sharpness'] > 25
                    and shot_spans.get(r['shot_id'], [0, 0])[1] - shot_spans.get(r['shot_id'], [0, 0])[0] >= 2]
            vectors = []
            filtered = []
            for row in pool:
                image = cv2.imread(row['image_path'])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
                neutral = ((gray > 160) & (chroma < 42)).astype(np.uint8)
                neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                _, components, stats, _ = cv2.connectedComponentsWithStats(neutral)
                bottom_labels = np.unique(components[int(.88 * gray.shape[0]), int(.05*gray.shape[1]):int(.95*gray.shape[1])])
                floor_fraction = sum(stats[label, cv2.CC_STAT_AREA] for label in bottom_labels if label != 0) / gray.size
                if floor_fraction < .36:
                    continue
                row = dict(row)
                row['floor_connected_brightness_fraction_ranking_only'] = float(floor_fraction)
                filtered.append(row)
                spatial = cv2.resize(gray, (24, 14), interpolation=cv2.INTER_AREA).reshape(-1) / 255.
                background = cv2.resize(image[:image.shape[0] // 2], (12, 5), interpolation=cv2.INTER_AREA).reshape(-1) / 255.
                vectors.append(np.concatenate([spatial, background * .6]))
            pool = filtered
            data = np.asarray(vectors, np.float32)
            k = min(32, len(data))
            _, labels, centers = cv2.kmeans(data, k, None,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .02), 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(-1)
            reps = []
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                distances = np.linalg.norm(data[members] - centers[cluster], axis=1)
                idx = int(members[np.argmin(distances)])
                row = dict(pool[idx])
                row['appearance_cluster_id'] = int(cluster)
                row['appearance_cluster_candidate_count'] = len(members)
                reps.append((row, centers[cluster]))
            # Keep all view families visible: start with common scene, then farthest
            # appearance centers rather than selecting sixteen near-duplicate scores.
            chosen = [int(np.argmax([r[0]['appearance_cluster_candidate_count'] for r in reps]))]
            while len(chosen) < min(16, len(reps)):
                distance = np.min(np.linalg.norm(centers[:, None] - centers[chosen][None, :], axis=2), axis=1)
                distance[chosen] = -1
                for index, rep in enumerate(reps):
                    if any(abs(rep[0]['source_time_seconds'] - reps[chosen_index][0]['source_time_seconds']) < 30 for chosen_index in chosen):
                        distance[index] = -1
                if np.max(distance) < 0:
                    break
                chosen.append(int(np.argmax(distance)))
            selected = [reps[index][0] for index in chosen]
            if len(selected) < 16:
                # Fill with time-separated useful scene candidates, without undoing
                # the floor/persistence filters or duplicating occlusion bursts.
                for row in sorted(pool, key=lambda r: r['features']['discovery_score'], reverse=True):
                    if all(abs(row['source_time_seconds'] - prior['source_time_seconds']) >= 30 for prior in selected):
                        selected.append(dict(row))
                    if len(selected) == 16:
                        break
            for row in selected:
                row['selection_reason'] = 'Persistent shot with visible connected neutral floor candidate; diverse appearance and >=30s separation; camera identity unverified'
                row['geometry_eligibility'] = 'requires_same_venue_and_geometric_review'
            selected.sort(key=lambda row: row['source_time_seconds'])
        canvas = Image.new('RGB', (1920, 1200), '#101821')
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(selected):
            x, y = index % 4 * 480, index // 4 * 300
            picture = Image.open(row['image_path']).convert('RGB')
            picture.thumbnail((480, 270))
            canvas.paste(picture, (x, y))
            time = row['source_time_seconds']
            label = f'{index+1:02d} | {int(time//3600):02d}:{int(time%3600//60):02d}:{time%60:06.3f} | {source["source_id"]}'
            draw.text((x+8, y+274), label, fill='white')
            row['panel_index'] = index + 1
            row['camera_group_status'] = 'unverified_appearance_family_only'
            row['visual_review_status'] = 'pending_explicit_human_review'
        panel = review / f'{source["source_id"]}_16_views.jpg'
        canvas.save(panel, quality=94)
        record = {'source_id': source['source_id'], 'session_id': source['session_id'], 'title': source['title'],
                  'panel_path': panel.as_posix(), 'decoded_frame_count': summary['decoded_frame_count'],
                  'one_second_feature_count': summary['one_second_feature_count'],
                  'cut_candidate_count': summary['cut_candidate_count'], 'candidate_count': summary['candidate_count'],
                  'selected_for_review_count': len(selected), 'human_reviewed_count': 0,
                  'visual_reviewed_count': 0, 'reviewer_kind': None,
                  'selected_views': selected,
                  'camera_grouping_status': 'Appearance diversification only; physical camera groups require review and registration.'}
        dump(review / f'{source["source_id"]}_review.json', record)
        results.append(record)
    dump(review / 'index.json', results)
    return results


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('configs/rink_model/source_registry.json'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--acquire', action='store_true')
    parser.add_argument('--acquire-only', action='store_true')
    parser.add_argument('--source', action='append')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--chunk-seconds', type=int, default=600)
    parser.add_argument('--extract-highres', action='append', metavar='SOURCE_ID:SECONDS')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--review-panels', action='store_true')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    sources = [s for s in registry['sources'] if s.get('duration_seconds_from_listing', 0) > 3000]
    if args.review_panels:
        result = build_review_panels(sources, args.output)
        print(json.dumps([{'source_id': row['source_id'], 'panel_path': row['panel_path']} for row in result]))
        return
    if args.audit_only:
        result = audit_sources(sources, args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ['sources', 'limitations']}, ensure_ascii=False))
        return
    if args.extract_highres:
        groups = {}
        for value in args.extract_highres:
            sid, sec = value.rsplit(':', 1)
            groups.setdefault(sid, []).append(float(sec))
        for sid, seconds in groups.items():
            for attempt in range(3):
                try:
                    extract_highres(next(s for s in sources if s['source_id'] == sid), args.output, seconds)
                    break
                except Exception as exc:
                    record = {'source_id': sid, 'requested_seconds': seconds, 'attempt': attempt + 1,
                              'at_utc': datetime.now(timezone.utc).isoformat(),
                              'error_type': type(exc).__name__, 'error': str(exc).split('https://')[0][:400]}
                    with (args.output / 'highres_acquisition_failures.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(f'HD_RETRY {sid} attempt={attempt+1} {type(exc).__name__}', flush=True)
                    if attempt == 2:
                        raise
        return
    if args.source:
        sources = [s for s in sources if s['source_id'] in args.source]
    def job(source):
        local = [Path(r['local_path']) for r in source['representations'] if Path(r['local_path']).exists()]
        local = sorted(local, key=lambda p: p.stat().st_size)
        proxy = args.output / source['source_id'] / f'{source["source_id"]}.f133.mp4'
        media = local[0] if local else proxy if proxy.exists() else None
        if media is None and args.acquire:
            media = acquire(source, args.output)
        if args.acquire_only:
            return {'source_id': source['source_id'], 'local_path': str(media) if media else None}
        if media is None:
            result = {'source_id': source['source_id'], 'status': 'media_unavailable',
                      'unprocessed_intervals_seconds': [[0, source['duration_seconds_from_listing']]]}
            dump(args.output / source['source_id'] / 'summary.json', result)
            return result
        return run_source(source, args.output, media, args.chunk_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, sources))
    dump(args.output / ('acquisition_run.json' if args.acquire_only else 'survey_run.json'), results)


if __name__ == '__main__':
    main()
