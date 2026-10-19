"""Make one small contact sheet from actual source-reading images, unchanged."""
from pathlib import Path
from fractions import Fraction
import argparse
import json
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    report = json.loads((directory / 'actual_representative_application_review.json').read_text(encoding='utf-8'))
    if report['status'] != 'representative_keyframes_complete' or not report['code_SHA_start_end_equal']:
        raise ValueError('Actual representative review is incomplete or mixed source versions')
    representatives = report['sources'][:9]
    if [row[0] for row in representatives] != list(range(1, 10)):
        raise ValueError('The sheet must include every complete-shot representative in order')
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 19)
    title = ImageFont.truetype(str(font_path), 28)
    sheet = Image.new('RGB', (1920, 1332), '#101923')
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), '新视觉读取：整段九镜头代表关键帧', font=title, fill='white')
    draw.text((18, 54), '均为原 RGB 实际取点；未知、缺测与误读保留。代表帧不代替完整连续序列验收。', font=font, fill='#c9d4df')
    cells = []
    for position, (shot, pts) in enumerate(representatives):
        image_path = directory / f'shot_{shot:02d}_source_{pts}' / f'source_{pts}_reading.png'
        image = Image.open(image_path).convert('RGB')
        if image.width != 1920 or image.height < 1080:
            raise ValueError('Source overlay dimensions differ from the native programme')
        # The original full-resolution file retains the reader's footer. The
        # overview resizes the full visible source frame, without erasing any
        # source-measurement marks or selecting only successful features.
        tile = image.crop((0, 0, 1920, 1080)).resize((640, 360), Image.Resampling.LANCZOS)
        x, y = position % 3 * 640, 100 + position // 3 * 410
        sheet.paste(tile, (x, y))
        milliseconds = round(Fraction(pts, 15360) * 1000)
        total_seconds, ms = divmod(milliseconds, 1000)
        hour, remaining = divmod(total_seconds, 3600)
        minute, second = divmod(remaining, 60)
        stamp = f'{hour:02}:{minute:02}:{second:02}.{ms:03}'
        draw.text((x+12, y+366), f'镜头 {shot} · 原片 {stamp} · PTS {pts}', font=font, fill='white')
        cells.append(dict(shot_index=shot, source_pts=pts, time_base='1/15360',
                          source_time=stamp, full_resolution_reading_image=str(image_path)))
    output = directory / 'nine_shots_actual_application_points.png'
    sheet.save(output, optimize=True)
    metadata = dict(schema='rinkspace.actual-application-nine-shot-contact-sheet.v1',
                    image=str(output), image_bytes=output.stat().st_size, actual_reading_images=cells,
                    application_data_bytes=report['application_data_total_bytes'],
                    actual_all_keyframes=report['actual_completed_keyframes'],
                    full_programme_complete=False, pixel_accuracy_accepted=False,
                    overlays_not_filtered_by_apparent_fit=True,
                    original_source_points_not_replaced_by_projection=True)
    (directory/'nine_shots_actual_application_points.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(image=str(output), bytes=metadata['image_bytes'], application_bytes=report['application_data_total_bytes'])))


if __name__ == '__main__':
    main()
