"""Render compact, source-PTS-labelled review boards for proxy cut proposals.

Each row compares the frames around one proposed cut and samples the interval
until the next proposed cut. It never converts proposals into reviewed shots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def clock(pts: int) -> str:
    milliseconds = round(pts * 1000 / 15360)
    h, remain = divmod(milliseconds, 3600000)
    m, remain = divmod(remain, 60000)
    s, ms = divmod(remain, 1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'


def sample_interval(samples: list[dict], start_pts: int, end_pts: int):
    available = [row for row in samples if start_pts <= row['source_pts'] < end_pts]
    if not available:
        return [None, None, None]
    targets = [start_pts, (start_pts + end_pts) // 2, end_pts - 1]
    chosen = []
    for target in targets:
        unused = [row for row in available if row not in chosen]
        if not unused:
            chosen.append(None)
            continue
        nearest = min(unused, key=lambda row: (abs(row['source_pts'] - target), row['source_pts']))
        chosen.append(nearest)
    return chosen


def write_board(rows: list[dict], page: int, out: Path, font, small):
    item_height = 670
    canvas = Image.new('RGB', (1280, item_height * len(rows)), '#13202b')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * item_height
        cut = row['candidate']
        draw.text((8, y + 4),
                  f"{row['candidate_id']}  {clock(cut['before_pts'])} | {clock(cut['after_pts'])}"
                  f"   MAD {cut['score']['mad_256x144']:.1f}  H {cut['score']['rgb_hist_distance']:.2f}",
                  font=font, fill='white')
        for side, x in [('before', 0), ('after', 640)]:
            with Image.open(cut[f'{side}_image']) as image:
                canvas.paste(image.convert('RGB'), (x, y + 36))
        interval = row['interval_after_pts']
        draw.text((8, y + 402), f"Proposal interval {clock(interval[0])} to {clock(interval[1])} (end exclusive) | existing 4 s samples: first / middle / last", font=small, fill='#c6e1e9')
        for position, sample in enumerate(row['interval_samples']):
            x = position * 426
            if sample is None:
                draw.rectangle((x, y + 428, x + 426, y + 668), fill='#283846')
                draw.text((x + 10, y + 530), 'No distinct 4 s sample', font=small, fill='#c2ced4')
                continue
            with Image.open(sample['image_path']) as image:
                canvas.paste(image.convert('RGB').resize((426, 240), Image.Resampling.LANCZOS), (x, y + 428))
            draw.rectangle((x, y + 636, x + 426, y + 669), fill='#13202b')
            draw.text((x + 7, y + 642), f"{clock(sample['source_pts'])}  PTS {sample['source_pts']}", font=small, fill='white')
    path = out / f'review_{page:03}.jpg'
    canvas.save(path, quality=86)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.scan_root.resolve()
    scan_index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    font_file = Path('C:/Windows/Fonts/consola.ttf')
    font = ImageFont.truetype(str(font_file), 20) if font_file.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_file), 16) if font_file.exists() else ImageFont.load_default()
    all_boards = []
    ranked = []
    for chunk in scan_index['chunks']:
        scan_path = Path(chunk['scan_path'])
        scan = json.loads(scan_path.read_text(encoding='utf-8'))
        candidates = scan['candidates']
        samples = sorted(scan['samples'], key=lambda row: row['source_pts'])
        board_dir = scan_path.parent / 'review_boards'
        board_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, candidate in enumerate(candidates):
            interval_end = (candidates[index + 1]['proposal_pts'] if index + 1 < len(candidates)
                            else scan['core_end_pts'])
            rows.append({'candidate_id': f"{scan_path.parent.name}:C{index + 1:03}",
                         'candidate': candidate,
                         'interval_after_pts': [candidate['after_pts'], interval_end],
                         'interval_samples': sample_interval(samples, candidate['after_pts'], interval_end)})
        boards = [write_board(rows[offset:offset + 4], offset // 4 + 1, board_dir, font, small)
                  for offset in range(0, len(rows), 4)]
        for index, row in enumerate(rows):
            candidate = row['candidate']
            ranked.append({'candidate_id': row['candidate_id'],
                           'proposal_pts': candidate['proposal_pts'],
                           'proposal_time': candidate['proposal_time'],
                           'rgb_hist_distance': candidate['score']['rgb_hist_distance'],
                           'mad_256x144': candidate['score']['mad_256x144'],
                           'chronological_board_path': boards[index // 4],
                           'board_row_index': index % 4})
        manifest = {'schema_version': 'rinkspace.broadcast_cut_review.v1',
                    'scan_path': str(scan_path), 'time_base': scan['time_base'],
                    'authority': 'review_layout_only_not_camera_decisions',
                    'chunk_boundary_is_not_a_cut': True,
                    'leading_interval_pts': [scan['core_start_pts'], candidates[0]['proposal_pts'] if candidates else scan['core_end_pts']],
                    'leading_interval_samples': sample_interval(samples, scan['core_start_pts'], candidates[0]['proposal_pts'] if candidates else scan['core_end_pts']),
                    'boards': boards, 'rows': rows}
        manifest_path = board_dir / 'review_index.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        all_boards.extend(boards)
        print(json.dumps({'chunk': scan_path.parent.name, 'candidates': len(rows), 'boards': len(boards)}, ensure_ascii=False), flush=True)
    overall = {'schema_version': 'rinkspace.broadcast_cut_review.v1', 'scan_index_path': str(root / 'index.json'),
               'boards': all_boards, 'board_count': len(all_boards),
               'chunk_boundary_is_not_a_cut': True,
               'authority': 'numeric_proposals_and_source_samples_for_human_visual_review'}
    (root / 'review_boards.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ranked.sort(key=lambda row: (-row['rgb_hist_distance'], -row['mad_256x144'], row['proposal_pts']))
    (root / 'hist_ranked_review_index.json').write_text(json.dumps({
        'schema_version': 'rinkspace.broadcast_cut_review_rank.v1',
        'authority': 'review_priority_only_all_numeric_proposals_retained',
        'sort': 'descending_rgb_hist_distance_then_mad',
        'candidates': ranked,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
