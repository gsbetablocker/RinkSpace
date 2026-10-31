"""Draw selected native-PTS stills from an explicit current problem/snapshot.

Extracted from current_grid_stills/render.py: complete ice, boards and ads use
sample_native_review_scene and the solver's ResidentResidualProblem projection.
Only the current PreparedProblem source UV is drawn in red. No fit or smoothing.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import collections
import hashlib
import json
import re
import time

import cv2
import numpy as np

import prepare_native_multicamera as prepared
from export_native_six_review import (
    PTS_STEP, TIME_BASE, _name, _source_selections, _verify_native_library_binding,
    _manifest_for_library,
)
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.scene_assets import scene_content_identity
from rinkspace.path_policy import guard_project_write
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.review.render import source_frames
from rinkspace.review.draw import COLORS, text
from rinkspace.storage import configured_data_root

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def parse_shot_pts(value):
    """Parse s07:1446912,1460992;s08:1806080 without implicit frame selection."""
    selected = {}
    for group in value.split(';'):
        parts = group.strip().split(':')
        if len(parts) != 2:
            raise ValueError('Use shot:PTS,PTS;shot:PTS with integer native PTS')
        shot_id = _name(parts[0].strip())
        if shot_id in selected:
            raise ValueError('Shot is repeated in --shot-pts: '+shot_id)
        values = [item.strip() for item in parts[1].split(',')]
        if any(re.fullmatch(r'[0-9]+', item) is None for item in values):
            raise ValueError('Every requested PTS must be a nonnegative integer')
        pts = [int(item) for item in values]
        if len(set(pts)) != len(pts) or any(item > np.iinfo(np.int64).max for item in pts):
            raise ValueError('Requested native PTS must be distinct int64 values')
        selected[shot_id] = pts
    return selected


def _validate_source(manifest, mapping, points, scope):
    """Reuse native-source validation without the video concat adjacency rule."""
    clock = scope['source_clock']
    if (Fraction(clock['time_base_num'],clock['time_base_den']) != TIME_BASE
            or clock['native_pts_step'] != PTS_STEP
            or Fraction(clock['native_fps_num'],clock['native_fps_den']) != 50):
        raise ValueError('Current renderer requires the registered 50 fps, TB 1/12800 source')
    source = {**scope['source'],**clock}
    for record in (manifest,mapping,*manifest.get('source_inputs',{}).values()):
        if record.get('source_video_sha256',source['sha256']) != source['sha256']:
            raise ValueError('Prepared source video identity differs from registered source')
    declared_clock = manifest.get('source_clock',{})
    for key in ('time_base_num','time_base_den','native_pts_step','native_fps_num','native_fps_den'):
        if key in declared_clock and declared_clock[key] != clock[key]:
            raise ValueError('Prepared native source clock differs: '+key)
    by_id, row_stop = {}, 0
    for shot in manifest['shots']:
        shot_id = _name(shot['id'])
        if shot_id in by_id:
            raise ValueError('Current manifest has duplicate shot IDs')
        begin, stop = int(shot['row_begin']), int(shot['row_stop'])
        if begin != row_stop or stop < begin or stop > len(points['pts']):
            raise ValueError('Invalid current observation rows: '+shot_id)
        if Fraction(shot['time_base_num'],shot['time_base_den']) != TIME_BASE:
            raise ValueError('Current shot source time base differs: '+shot_id)
        local_shot = dict(shot,row_begin=0,row_stop=stop-begin)
        _source_selections(dict(shots=[local_shot]),dict(pts=points['pts'][begin:stop]),scope,
                           legacy_six=False,allow_sparse_observations=True)
        by_id[shot_id] = shot
        row_stop = stop
    if not by_id or row_stop != len(points['pts']):
        raise ValueError('Current observations are not partitioned by distinct manifest shots')
    for key in ('point_input_sha256','input_sha256'):
        if key in mapping and mapping[key] != manifest['point_input_sha256']:
            raise ValueError('Mapping point identity differs: '+key)
    return source, by_id


def render(run_root, state_path, output, selected, *, native_library=None):
    started = time.monotonic()
    root = Path(run_root).resolve(strict=True)
    state_path = Path(state_path).resolve(strict=True)
    out = guard_project_write(output)
    if not out.is_relative_to(configured_data_root(REPO)):
        raise ValueError('Native review image outputs must stay inside registered RinkSpace_Data')
    if out.exists():
        raise FileExistsError('Preserve existing still output: '+str(out))
    if isinstance(selected,str):
        selected = parse_shot_pts(selected)
    if not selected:
        raise ValueError('Request at least one explicit shot/PTS')
    manifest_path, mapping_path = root/'joint_manifest.json', root/'mapping.json'
    points_path, scope_path = root/'point_observations.npz', HERE/'rebuild_scope.json'
    prepared_manifest, mapping, scope = [prepared.read(path) for path in (manifest_path,mapping_path,scope_path)]
    manifest = _manifest_for_library(prepared_manifest,native_library)
    library = Path(manifest['native_library']).resolve(strict=True)
    scene_path = Path(manifest['scene_path']).resolve(strict=True)
    source_path = Path(scope['source']['registered_path']).resolve(strict=True)
    state_sha256 = prepared.digest(state_path)
    expected_inputs = {str(path):expected for path,expected in (
        (state_path,state_sha256),(points_path,manifest['point_input_sha256']),
        (library,manifest['native_library_sha256']),(scene_path,manifest['scene_file_sha256']),
        (source_path,scope['source']['sha256']))}
    for path,expected in expected_inputs.items():
        if prepared.digest(path) != expected:
            raise ValueError('Input SHA mismatch: '+path)
    script_path = Path(__file__).resolve()
    metadata_hashes = {str(path):prepared.digest(path) for path in
                       (manifest_path,mapping_path,scope_path,script_path)}
    points, state = prepared.npz(points_path), prepared.npz(state_path)
    _verify_native_library_binding(manifest,state)
    source, by_id = _validate_source(manifest,mapping,points,scope)
    for shot_id,requested in selected.items():
        if shot_id not in by_id:
            raise ValueError('Requested shot absent from current manifest: '+shot_id)
        shot = by_id[shot_id]
        for pts in requested:
            if not (shot['first_pts'] <= pts <= shot['last_pts']
                    and (pts-shot['first_pts']) % source['native_pts_step'] == 0):
                raise ValueError('Requested PTS outside registered native shot: '+shot_id+':'+str(pts))
            guard_project_write(out/f'{shot_id}_{pts}.png')
    guard_project_write(out/'manifest.json')
    core = NativeCore(library)
    raw_scene = prepared.read(scene_path)
    model = SceneParameterModel(raw_scene,scene_path,optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,manifest,points)
    parameter_state = ParameterState(model.initial_state.layout,state['scene_values'])
    scene_identity = scene_content_identity(raw_scene,scene_path)
    native = prepared.NativeSceneGPU(scene_path,core,parameter_state=parameter_state)
    timeline = ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    samples = sample_native_review_scene(raw_scene,core,scene_path,parameter_state=parameter_state,
        observed_path_ids={mapping['path_ids'][int(i)] for i in np.unique(points['path_indices'])},native=native)
    if samples.get('native_paths_resolved') is not True:
        raise ValueError('Native path table unresolved')
    world, cuts = samples['xyz'], samples['cuts']
    n = len(world)
    ads = collections.defaultdict(list)
    for cut in cuts:
        if cut['layer'] == 'advertisement':
            ads[cut['id'].split('::',1)[0]].extend(range(cut['begin'],cut['end']))
    ads = {key:np.asarray(indices,dtype=np.int32) for key,indices in ads.items()}
    width, height = int(source['width']), int(source['height'])
    tb = Fraction(source['time_base_num'],source['time_base_den'])
    records = []
    out.mkdir(parents=True,exist_ok=False)
    for shot_id,requested in selected.items():
        shot = by_id[shot_id]
        rig = state['rig'][shot['camera_index']]
        shot_rows = np.arange(shot['row_begin'],shot['row_stop'])
        for pts in requested:
            frame_path = out/f'{shot_id}_{pts}.png'
            if frame_path.exists():
                raise FileExistsError('Preserve existing still: '+str(frame_path))
            rows = shot_rows[points['pts'][shot_rows] == pts]
            raw_frames = list(source_frames(source_path,source,np.array([pts],dtype=np.int64)))
            if len(raw_frames) != 1 or raw_frames[0][0] != pts:
                raise ValueError('Exact source frame missing')
            raw = raw_frames[0][1]
            motion_time = ((pts-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']
                           +timeline.offset(shot))
            with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
                    options=native.scene['shared_geometry_options'],rig=rig,controls=state['controls'],
                    first_control_time=timeline.first,control_dt=timeline.dt,image_height=height,
                    codes=np.zeros(n,dtype=np.int32),stations=np.linspace(0,1,n),
                    times=np.full(n,motion_time),observed=np.zeros((n,2))) as resident:
                resident.configure_geometry([dict(kind=4,vertices=world,parameters=[])])
                resident.configure_projection_time_support(half_seconds=manifest['max_readout']/2)
                projected = resident.evaluate('cuda')
            uv = projected['values'][:,3:5]
            good = (projected['status'] == 0) & np.isfinite(uv).all(axis=1)
            pixels = np.rint(np.clip(np.nan_to_num(uv),-1e7,1e7)).astype(np.int32)
            picture = raw.copy()
            for cut in cuts:
                a,b = cut['begin'],cut['end']
                color = (220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
                for lo,hi in np.flatnonzero(np.diff(np.r_[False,good[a:b],False])).reshape(-1,2):
                    if hi-lo > 1:
                        cv2.polylines(picture,[pixels[a+lo:a+hi]],
                            bool(cut.get('closed') and lo == 0 and hi == b-a),color,1,cv2.LINE_AA)
            for identity,indices in ads.items():
                visible = indices[good[indices] & (uv[indices,0] >= 0) & (uv[indices,0] < width)
                                  & (uv[indices,1] >= 0) & (uv[indices,1] < height)]
                if len(visible):
                    at = np.rint(np.median(uv[visible],axis=0)).astype(int)
                    text(picture,identity.rsplit('/',1)[-1],at,.45,COLORS['advertisement'])
            for row in rows:
                cv2.circle(picture,tuple(np.rint(points['uv'][row]).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
            header = np.full((64,width,3),15,dtype=np.uint8)
            text(header,f'{shot_id.upper()} | Source {float(pts*tb):.3f}s | PTS {pts} | {len(rows)} points',
                 (15,26),.65,(60,220,255))
            text(header,'Frozen native state | Complete ice / boards / print IDs | Current source UV: red | No new fit',
                 (15,52),.48)
            canvas = np.vstack((header,picture))
            if not cv2.imwrite(str(frame_path),canvas):
                raise OSError('PNG write failed')
            records.append(dict(shot=shot_id,pts=pts,source_seconds=float(pts*tb),source_points=len(rows),
                path=str(frame_path),sha256=prepared.digest(frame_path),
                source_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                source_size=[width,height],output_size=[int(canvas.shape[1]),int(canvas.shape[0])]))
            print(json.dumps(dict(shot=shot_id,pts=pts,points=len(rows),path=str(frame_path),
                                  elapsed_s=round(time.monotonic()-started,2))),flush=True)
    for path,expected in {**expected_inputs,**metadata_hashes}.items():
        if prepared.digest(path) != expected:
            raise ValueError('Input changed while drawing: '+path)
    if scene_content_identity(prepared.read(scene_path),scene_path) != scene_identity:
        raise ValueError('Scene assets changed')
    receipt = dict(schema='rinkspace.native_review_stills.v1',run_root=str(root),
        state=dict(path=str(state_path),sha256=state_sha256),
        prepared_manifest=dict(path=str(manifest_path),sha256=metadata_hashes[str(manifest_path)]),
        mapping=dict(path=str(mapping_path),sha256=metadata_hashes[str(mapping_path)]),
        points=dict(path=str(points_path),sha256=manifest['point_input_sha256']),
        scene=dict(path=str(scene_path),sha256=manifest['scene_file_sha256'],content_sha256=scene_identity),
        native_library=dict(path=str(library),sha256=manifest['native_library_sha256']),
        prepared_native_library=prepared_manifest['native_library'],
        prepared_native_library_sha256=prepared_manifest['native_library_sha256'],
        selected_native_library=str(library),native_library_override_explicit=native_library is not None,
        state_native_library_provenance_sha256=str(state['native_library_provenance_sha256']),
        source=dict(path=str(source_path),sha256=source['sha256'],time_base_num=tb.numerator,
                    time_base_den=tb.denominator,native_pts_step=source['native_pts_step'],size=[width,height]),
        scope=dict(path=str(scope_path),sha256=metadata_hashes[str(scope_path)]),
        script=dict(path=str(script_path),sha256=metadata_hashes[str(script_path)]),
        selected_shot_pts=selected,fit_executed=False,smoothing_applied=False,
        current_source_points_only=True,full_source_resolution_preserved=True,header_height=64,frames=records)
    with (out/'manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True,frames=len(records),manifest=str(out/'manifest.json'))),flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True,help='Current PreparedProblem directory')
    parser.add_argument('--state',type=Path,required=True,help='Explicit numeric snapshot; never inferred from a run')
    parser.add_argument('--output',type=Path,required=True,help='New directory inside registered RinkSpace_Data')
    parser.add_argument('--shot-pts',type=parse_shot_pts,required=True,
                        help='Explicit native PTS, e.g. s07:1446912,1460992;s08:1806080')
    parser.add_argument('--native-library',type=Path,
                        help='Explicit actual DLL; its SHA256 must match the frozen state')
    args = parser.parse_args()
    render(args.run_root,args.state,args.output,args.shot_pts,native_library=args.native_library)


if __name__ == '__main__':
    main()
