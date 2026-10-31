"""Prepare the fixed S07-S09 source bootstrap representatives in external Data.

Recovered from the 2026-10-31 preparation driver. Exact native PTS and source
identity checks are retained; representative frames do not prove full coverage.
"""
from pathlib import Path
from fractions import Fraction
import sys, json, math, hashlib, datetime
import av
from PIL import Image, ImageDraw, ImageFont

import argparse

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from rinkspace.storage import AssetStore


def prepare(scope_path, output):
    OUT = Path(output).resolve()
    if OUT.is_relative_to(PROJECT.resolve()):
        raise ValueError("Source media outputs must stay outside the repository")
    OUT.mkdir(parents=True, exist_ok=True)
    scope = json.loads(scope_path.read_text(encoding='utf-8'))
    source = dict(scope['source'])
    clock = dict(scope['source_clock'])
    source_path = AssetStore(PROJECT).resolve(source['registered_path'], must_exist=True)
    actual_hash = hashlib.file_digest(source_path.open('rb'), 'sha256').hexdigest()
    assert actual_hash == source['sha256'], 'registered source SHA256 mismatch'
    tb = Fraction(clock['time_base_num'], clock['time_base_den'])
    step = int(clock['native_pts_step'])
    shots = [s for s in scope['shots'] if s['id'] in ('s07','s08','s09')]
    assert [(s['id'], s['rig_id'], s['processing']['first_pts'], s['processing']['last_pts']) for s in shots] == [
        ('s07','main',1425664,1742592),('s08','end',1742848,1837824),('s09','main',1838080,2415104)]

    def stamp(pts):
        # Source time comes from actual native PTS multiplied by the rational time base.
        ms = round(Fraction(int(pts)) * tb * 1000)
        hour, rem = divmod(ms, 3600000)
        minute, rem = divmod(rem, 60000)
        second, milli = divmod(rem, 1000)
        return f'{hour:02d}:{minute:02d}:{second:02d}.{milli:03d}'

    font_path = Path('C:/Windows/Fonts/consola.ttf')
    font = ImageFont.truetype(str(font_path), 25) if font_path.is_file() else ImageFont.load_default()
    result = {
        'schema': 'rinkspace.source_bootstrap.v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'First independent scene-recognition representative frames only; not point selection or full-shot visual coverage.',
        'image_content_reviewed': False,
        'inputs_read': ['AGENTS.md', 'MAINTENANCE.md', 'docs/VALIDATION.md', 'docs/rink_model/BUILD_SPEC.md',
                        'programme README source/range entry', str(scope_path), 'AssetStore implementation/configuration', str(source_path)],
        'prohibited_prior_observations_or_camera_states_read': False,
        'source': {**source, 'resolved_path': str(source_path), 'actual_sha256': actual_hash,
                   'bytes': source_path.stat().st_size, 'stream_index': 0},
        'source_clock': clock,
        'sampling': 'Per shot: inclusive first/middle/last plus equal index intervals at most approximately 3 source seconds, rounded to registered native exposures and verified by actual decode.',
        'original_image_size': [1920,1080],
        'contact_sheet_cell_image_size': [960,540],
        'contact_sheet_max_frames': 6,
        'shots': []
    }
    for shot in shots:
        shot_id = shot['id']; rig = shot['rig_id']; window = shot['processing']
        first = int(window['first_pts']); last = int(window['last_pts'])
        count = int(window['native_frame_count'])
        assert last == first + (count-1)*step
        intervals = max(1, math.ceil(float((last-first)*tb)/3.0))
        sample_indices = {round(i*(count-1)/intervals) for i in range(intervals+1)}
        sample_indices.update([0, (count-1)//2, count-1])
        selected = {first+i*step for i in sample_indices}
        shot_out = OUT / shot_id
        shot_out.mkdir(exist_ok=False)
        originals = shot_out / 'originals'; originals.mkdir()
        images = {}; rows = []; actual_pts = []
        with av.open(str(source_path)) as container:
            stream = container.streams.video[0]
            assert stream.time_base == tb
            assert (stream.width, stream.height) == (1920,1080)
            container.seek(first, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                assert frame.pts is not None, 'decoded frame has no PTS'
                if frame.pts < first:
                    continue
                if frame.pts > last:
                    break
                assert frame.time_base == tb
                pts = int(frame.pts)
                expected = first + len(actual_pts)*step
                assert pts == expected, f'{shot_id}: actual source PTS mismatch {pts} != {expected}'
                actual_pts.append(pts)
                if pts in selected:
                    image = frame.to_image().convert('RGB')
                    assert image.size == (1920,1080)
                    path = originals / f'{shot_id}_pts{pts}.png'
                    image.save(path, compress_level=4)
                    images[pts] = image
                    rows.append({'shot_id': shot_id, 'known_rig_id': rig,
                                 'actual_pts': pts, 'time_base_num': tb.numerator, 'time_base_den': tb.denominator,
                                 'source_time': stamp(pts), 'source_time_seconds': float(pts*tb),
                                 'original_path': str(path), 'original_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        assert len(actual_pts) == count and actual_pts[-1] == last
        assert set(images) == selected
        sheets = []
        for start in range(0,len(rows),6):
            group = rows[start:start+6]
            sheet_index = start//6+1
            sheet_path = shot_out / f'{shot_id}_sheet_{sheet_index:02d}.jpg'
            cols = 2 if len(group)>1 else 1
            sheet_rows = math.ceil(len(group)/cols)
            cell_h = 604
            sheet = Image.new('RGB', (cols*960, sheet_rows*cell_h), (20,20,20))
            draw = ImageDraw.Draw(sheet)
            for j,row in enumerate(group):
                x = (j%cols)*960; y = (j//cols)*cell_h
                tile = images[row['actual_pts']].resize((960,540), Image.Resampling.LANCZOS)
                sheet.paste(tile,(x,y))
                label = f'{shot_id.upper()} | rig={rig} | PTS {row["actual_pts"]} | {row["source_time"]}'
                draw.text((x+12,y+548),label,fill=(250,250,250),font=font)
                row['sheet_path'] = str(sheet_path)
                row['sheet_cell_index'] = j
            sheet.save(sheet_path,quality=95,subsampling=0)
            sheets.append({'path':str(sheet_path),'frame_count':len(group),
                           'actual_pts':[r['actual_pts'] for r in group], 'size':list(sheet.size)})
        row = {
            'shot_id': shot_id, 'known_rig_id': rig,
            'rig_binding_status': shot['rig_binding_status'],
            'processing': window,
            'decoded_pts_verification': {'all_registered_pts_matched':True, 'decoded_count':len(actual_pts),
                                         'first_actual_pts':actual_pts[0], 'last_actual_pts':actual_pts[-1],
                                         'step':step, 'full_shot_image_cache_created':False},
            'representative_count':len(rows), 'frames':rows,'sheets':sheets,
            'visual_coverage_claim':'Representative frames only; image content has not been inspected.'
        }
        result['shots'].append(row)
        (OUT/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'shot':shot_id,'representative_frames':len(rows),'sheets':[s['path'] for s in sheets]},ensure_ascii=False),flush=True)
    print(str(OUT/'index.json'),flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=PROJECT / "events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.scope, args.out)


if __name__ == "__main__":
    main()
