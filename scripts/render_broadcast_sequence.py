"""Create one-second source-proxy sequence contact sheets without frame files."""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = _resolve_asset(ROOT, ROOT / 'data/input/oxXu1avhrvs.f134.mp4')
TB = Fraction(1, 15360)


def stamp(pts: int) -> str:
    ms = round(pts * 1000 / 15360)
    hour, rest = divmod(ms, 3600000)
    minute, rest = divmod(rest, 60000)
    sec, milli = divmod(rest, 1000)
    return f'{hour:02}:{minute:02}:{sec:02}.{milli:03}'


def write_page(frames, page: int, out: Path, font):
    sheet = Image.new('RGB', (1920, 2060), '#17232c')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for cell, (target_second, pts, image) in enumerate(frames):
        x, y = (cell % 6) * 320, (cell // 6) * 206
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + 183), f'{stamp(pts)}   PTS {pts}', font=font, fill='white')
        rows.append({'target_second': target_second, 'source_pts': pts,
                     'source_time': stamp(pts), 'cell_index': cell})
    page_path = out / f'sequence_{page:03}.jpg'
    sheet.save(page_path, quality=87)
    return {'page_path': str(page_path), 'samples': rows, 'columns': 6, 'rows': 10,
            'cell_image_wh': [320, 180], 'cell_label_height': 26}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 19560:
        parser.error('Invalid source interval')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    font_file = Path('C:/Windows/Fonts/consola.ttf')
    font = ImageFont.truetype(str(font_file), 16) if font_file.exists() else ImageFont.load_default()
    pages, selected = [], []
    target = args.start
    with av.open(str(args.source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 3
        if stream.time_base != TB:
            raise ValueError(f'Unexpected time base: {stream.time_base}')
        container.seek(max(0, (args.start - 1) * 15360), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < target * 15360:
                continue
            if target >= args.end:
                break
            image = frame.to_image().resize((320, 180), Image.Resampling.LANCZOS)
            selected.append((target, int(frame.pts), image))
            target += 1
            if len(selected) == 60:
                pages.append(write_page(selected, len(pages) + 1, out, font))
                selected = []
            if target >= args.end:
                break
    if selected:
        pages.append(write_page(selected, len(pages) + 1, out, font))
    if target != args.end:
        raise RuntimeError(f'Missing samples: stopped at {target}, expected {args.end}')
    manifest = {'schema_version': 'rinkspace.broadcast_sequence.v1',
                'source_path': str(args.source.resolve()), 'time_base': str(TB),
                'range_pts': [args.start * 15360, args.end * 15360],
                'sample_interval_seconds': 1, 'sample_count': sum(len(page['samples']) for page in pages),
                'pages': pages, 'authority': 'ordered_source_proxy_samples_only_not_camera_decisions'}
    (out / 'sequence.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'range': [args.start, args.end], 'samples': manifest['sample_count'],
                      'pages': len(pages), 'out': str(out)}), flush=True)


if __name__ == '__main__':
    main()
