"""Complete native-PTS replay through the same compiled scene/camera forward.

This module never optimizes, interpolates source exposures or smooths state.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import time
import av
import cv2
import numpy as np
from ..core.native import NativeCore
from ..compat.d78_entities import sample_entities
from ..session.dataset import load_prepared, load_array, relative
from ..storage import AssetStore
from .draw import top_down, overlay


@dataclass(frozen=True)
class ReviewReceipt:
    review: dict
    review_path: str


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def reference(base, value):
    if isinstance(value, dict):
        value = value['path']
    return relative(Path(base), value)


def shot_selection(manifest, shot_ids):
    selected = manifest['selected_shot_ids']
    requested = selected if shot_ids == 'all' else (shot_ids.split(',') if isinstance(shot_ids, str) else list(shot_ids))
    if not requested or len(requested) != len(set(requested)) or any(s not in selected for s in requested):
        raise ValueError('review shots must be unique IDs contained in the prepared selection')
    return requested


def load_state(state_path, core, shot_ids):
    path = Path(state_path).resolve(strict=True)
    state = read(path)
    if state.get('schema') != 'rinkspace.state.v1':
        raise ValueError('review requires rinkspace.state.v1')
    scene = read(reference(path.parent, state.get('scene_ref', state.get('replay_scene_ref'))))
    if scene.get('schema') not in {'rinkspace.scene.replay.v1', 'rinkspace.scene.shared.v1'}:
        raise ValueError('unsupported scene family: '+str(scene.get('schema')))
    samples = sample_entities(scene, core)
    # The board's complete course is part of the same compiled geometry. Its
    # two display edges supplement finite observed paths without inventing
    # observations or changing the physical path inventory.
    for height, name in ((0., 'continuous_board_bottom'), (1., 'continuous_board_top')):
        xyz = np.asarray(core.sample_path(scene['board'], 3, [-1., height], [], np.linspace(0., 1., 4097)))
        begin = len(samples['xyz'])
        samples['xyz'] = np.vstack((samples['xyz'], xyz))
        samples['cuts'].append({'id': name, 'entity_id': 'continuous_board', 'label': name,
            'layer': 'board', 'closed': True, 'begin': begin, 'end': begin+len(xyz),
            'kind': 'derived_continuous_board_edge', 'support': [0., 1.]})
    samples['inventory']['derived_display_paths'] = ['continuous_board_bottom', 'continuous_board_top']
    cameras = {}
    by_id = {s.get('id', s.get('shot_id')): s for s in state['shots']}
    for shot_id in shot_ids:
        camera_ref = by_id.get(shot_id, {}).get('camera_ref')
        if not camera_ref:
            raise ValueError('state has no camera definition for '+shot_id)
        camera_path = reference(path.parent, camera_ref)
        camera = read(camera_path)
        camera['controls'] = np.load(reference(camera_path.parent, camera['controls_ref']), allow_pickle=False)
        cameras[shot_id] = camera
    return state, scene, samples, cameras


def project(core, camera, xyz, pts):
    # Subtract native int64 identities before conversion to floating seconds.
    frame_time = (int(pts)-int(camera['reference_pts']))*camera['time_base_num']/camera['time_base_den']
    return core.project_motion(camera['rig'], camera['controls'], camera['first_control_time'],
                               camera['control_dt'], camera['image_height'], frame_time, xyz)


def source_frames(path, source, expected_pts):
    expected = np.asarray(expected_pts, dtype=np.int64)
    if not len(expected) or np.any(np.diff(expected) <= 0):
        raise ValueError('review requires a strictly increasing actual PTS index')
    tb = Fraction(source['time_base_num'], source['time_base_den'])
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base != tb or (stream.width,stream.height) != (source['width'],source['height']):
            raise ValueError('source dimensions/time_base differ from the registered source')
        container.seek(int(expected[0]), stream=stream, backward=True, any_frame=False)
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('decoded source exposure has no PTS')
            if frame.pts < expected[0]:
                continue
            if index == len(expected):
                break
            if frame.pts != int(expected[index]) or frame.time_base != tb:
                raise ValueError(f'actual source PTS gap: expected {expected[index]}, decoded {frame.pts}')
            yield int(frame.pts), frame.to_ndarray(format='bgr24')
            index += 1
        if index != len(expected):
            raise ValueError(f'source ended after {index}/{len(expected)} required exposures')


def validate_video(path, expected_pts, time_base):
    expected = np.asarray(expected_pts, dtype=np.int64)
    found = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            # Keep rational arithmetic to reject altered cadence or dropped frames.
            mapped = Fraction(frame.pts)*frame.time_base/time_base
            if mapped.denominator != 1:
                raise RuntimeError('encoded output has non-native PTS cadence')
            found.append(int(mapped)+int(expected[0]))
    if found != expected.tolist():
        raise RuntimeError('encoded output PTS do not exactly match every selected source exposure')
    return {'decoded_frames': len(found), 'native_PTS_equal': True, 'display_speed': 1}


def render(prepared_dir, state_path, assets_path, shot_ids, library_path, out_dir, compare_path=None):
    prepared_dir = Path(prepared_dir).resolve(strict=True)
    manifest, arrays = load_prepared(prepared_dir, purpose="review")
    requested = shot_selection(manifest, shot_ids)
    out = Path(out_dir).resolve()
    project_root = Path(__file__).resolve().parents[3]
    if out == project_root or project_root in out.parents:
        raise ValueError('review videos, frame images and logs must be outside the OneDrive project')
    if out.exists():
        raise FileExistsError('review output already exists; preserve it and choose a new directory: '+str(out))
    core = NativeCore(library_path)
    state, scene, samples, cameras = load_state(state_path, core, requested)
    if state.get('prepared_revision') != manifest['revision']:
        raise ValueError('state/prepared revision mismatch')
    current_scene = read(reference(prepared_dir, manifest['scene_ref']))
    if current_scene != scene:
        raise ValueError('state scene differs from the current prepared scene')
    comparison = None
    if compare_path is not None:
        other_state, other_scene, other_samples, other_cameras = load_state(compare_path, core, requested)
        if other_samples['inventory'] != samples['inventory']:
            raise ValueError('comparison requires the same entity inventory, not its intersection')
        comparison = {'state_id': other_state['state_id'], 'path': str(Path(compare_path).resolve()),
                      'meaning': 'same source PTS and full entity inventory', 'maximum_uv_difference_px': 0., 'status_mismatches': 0}
    assets = read(assets_path)
    store = AssetStore(assets.get('project_root', project_root))
    source_lookup = {s['id']: s for s in manifest['sources']}
    shot_lookup = {s['id']: s for s in manifest['shots']}
    all_shot_index = {s['id']: i for i,s in enumerate(manifest['shots'])}
    # Resolve and check source assets before creating any candidate output.
    source_paths = {}
    for shot_id in requested:
        source = source_lookup[shot_lookup[shot_id]['source_id']]
        supplied = assets.get('sources', {}).get(source['id'], source['asset_path'])
        if isinstance(supplied, dict):
            supplied = supplied['path']
        path = store.source(supplied)
        if not path.is_file():
            raise FileNotFoundError(path)
        source_paths[source['id']] = path
    out.mkdir(parents=True)
    overview = top_down(samples['xyz'], samples['cuts'])
    cv2.imwrite(str(out/'model_top.png'), overview)
    write(out/'entity_inventory.json', {**samples['inventory'], 'cuts': samples['cuts'], 'missing_entities': samples['missing_entities']})
    artifacts, covered, timing = [], {}, {}
    log_path = out/'main.log'
    started = time.perf_counter()
    try:
        with log_path.open('w', encoding='utf-8', buffering=1) as log:
            log.write('Native source replay; fit_executed=false; cuda_executed=false\n')
            for shot_id in requested:
                shot = shot_lookup[shot_id]; camera = cameras[shot_id]; source = source_lookup[shot['source_id']]
                if camera['source_id'] != source['id'] or (camera['time_base_num'],camera['time_base_den']) != (source['time_base_num'],source['time_base_den']):
                    raise ValueError('camera/source identity or native time base differs for '+shot_id)
                pts_index = load_array(prepared_dir, source['pts_ref'])
                window = shot['processing']
                expected = pts_index[(pts_index >= window['first_pts']) & (pts_index < window['stop_pts_exclusive'])]
                if len(expected) != window['native_frame_count']:
                    raise ValueError('prepared actual PTS index does not cover '+shot_id)
                if camera['first_pts'] > int(expected[0]) or camera['stop_pts_exclusive'] < window['stop_pts_exclusive']:
                    raise ValueError('camera domain does not cover '+shot_id)
                tb = Fraction(source['time_base_num'], source['time_base_den'])
                path = out/(shot_id+'_native_overlay.mp4')
                records, compare_records = [], []
                output = av.open(str(path), 'w', options={'movflags': '+faststart', 'video_track_timescale': str(tb.denominator)})
                rate = Fraction(tb.denominator, tb.numerator*int(expected[1]-expected[0])) if len(expected)>1 else Fraction(50)
                stream = output.add_stream('libx264', rate=rate)
                stream.width,stream.height,stream.pix_fmt = source['width']+640,source['height'],'yuv420p'
                stream.time_base = tb;stream.codec_context.time_base = tb
                stream.options = {'crf': '18', 'preset': 'fast'}
                uv_rows = np.flatnonzero((arrays['active'] != 0) & (arrays['shot_index'] == all_shot_index[shot_id]))
                by_pts = {}
                for row in uv_rows:
                    by_pts.setdefault(int(arrays['pts'][row]), []).append(int(row))
                try:
                    for i,(pts,raw) in enumerate(source_frames(source_paths[source['id']],source,expected)):
                        before = time.perf_counter()
                        prediction = project(core,camera,samples['xyz'],pts)
                        elapsed_projection = time.perf_counter()-before
                        if compare_path is not None:
                            baseline = project(core,other_cameras[shot_id],other_samples['xyz'],pts)
                            mismatch = int(np.count_nonzero(prediction['status'] != baseline['status']))
                            good = (prediction['status'] == 0) & (baseline['status'] == 0)
                            error = float(np.max(np.linalg.norm(prediction['uv'][good]-baseline['uv'][good],axis=1))) if good.any() else None
                            comparison['status_mismatches'] += mismatch
                            if error is not None:
                                comparison['maximum_uv_difference_px'] = max(comparison['maximum_uv_difference_px'],error)
                            compare_records.append({'pts': pts,'status_mismatches': mismatch,'maximum_uv_difference_px': error})
                        rows = by_pts.get(pts, [])
                        measured = arrays['uv'][rows] if rows else np.empty((0,2))
                        canvas,record = overlay(raw,prediction,samples['cuts'],overview,pts,tb,measured,shot_id,i,len(expected))
                        record['projection_seconds'] = elapsed_projection
                        record['active_source_row_indices'] = rows
                        records.append(record)
                        if i in {0,len(expected)//2,len(expected)-1}:
                            cv2.imwrite(str(out/f'{shot_id}_{pts}_overlay.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
                            cv2.imwrite(str(out/f'{shot_id}_{pts}_source.jpg'),raw,[cv2.IMWRITE_JPEG_QUALITY,97])
                        frame = av.VideoFrame.from_ndarray(canvas,format='bgr24')
                        frame.pts = pts-int(expected[0]);frame.time_base=tb
                        for packet in stream.encode(frame):
                            output.mux(packet)
                        if i % 50 == 0 or i == len(expected)-1:
                            log.write(json.dumps({'shot':shot_id,'frame':i+1,'total':len(expected),'pts':pts,'elapsed_s':time.perf_counter()-started,'failed_samples':record['failed_samples']})+'\n')
                    for packet in stream.encode():
                        output.mux(packet)
                finally:
                    output.close()
                check = validate_video(path,expected,tb)
                write(out/(shot_id+'_frames.json'),records)
                if compare_path is not None:
                    write(out/(shot_id+'_comparison.json'),compare_records)
                covered[shot_id] = {'first_pts':int(expected[0]),'last_pts':int(expected[-1]),'stop_pts_exclusive':window['stop_pts_exclusive'],'count':len(expected),'status':'complete_selected_shot'}
                artifacts.append({'kind':'complete_shot_overlay','path':str(path),'shot_id':shot_id,'source_id':source['id'],'first_pts':int(expected[0]),'last_pts':int(expected[-1]),'stop_pts_exclusive':window['stop_pts_exclusive'],'time_base_num':tb.numerator,'time_base_den':tb.denominator,'source_first_seconds':float(expected[0]*tb),'source_last_seconds':float(expected[-1]*tb),'frames':len(expected),'layers':['all_ice','all_board','all_advertisement_contours','entity_numbers','current_active_source_uv','same_model_top_view'],'verification':check})
                timing[shot_id] = {'projection_seconds':sum(r['projection_seconds'] for r in records)}
        result = {'schema':'rinkspace.review.v1','state_id':state['state_id'],'prepared_revision':manifest['revision'],'source_ids':list(source_paths),'required_shot_ids':manifest['required_shot_ids'],'selected_shot_ids':requested,'scope_status':'complete' if set(requested)==set(manifest['required_shot_ids']) else 'subset','unselected_shot_ids':[s for s in manifest['required_shot_ids'] if s not in requested],'covered_pts':covered,'missing_pts':{},'entity_inventory':samples['inventory'],'missing_entities':samples['missing_entities'],'artifacts':artifacts,'comparison':comparison,'user_approval':'not_reviewed','fit_executed':False,'cuda_executed':False,'purpose':'replay_and_model_migration_review','state_path':str(Path(state_path).resolve()),'prepared_path':str(prepared_dir),'assets_path':str(Path(assets_path).resolve()),'log_path':str(log_path),'timings':{**timing,'total_seconds':time.perf_counter()-started},'external_d78_baseline':assets.get('d78_baseline')}
        write(out/'review.json',result)
        return ReviewReceipt(result,str(out/'review.json'))
    except BaseException as exc:
        write(out/'error.json',{'schema':'rinkspace.error.v1','code':'review_failed','message':str(exc),'log_path':str(log_path),'fit_executed':False})
        raise