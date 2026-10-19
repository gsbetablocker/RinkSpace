"""Complete official-50 first-shot review of one authenticated physical state.

Preparation and requested stills are the default. Video is an explicit action;
it always covers all 1248 native exposures and never runs an optimizer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np

from ..solver_runtime import install, configure_native_runtime
install()

from .first_shot_fit import FirstShotSourceBundle, load_seed_state
from .official_static_surface_dataset import reference, read_reference
from .render_result import SceneRenderer, advertisement_review_paths, _ffmpeg_executable, _available_encoder
from .board_grid_overlay import make_board_grid_renderer
from .review_metric_grid import prepare_metric_review_scene, validate_metric_review_scene
from .current_source_points import draw_current_source_pixel_markers
from .body_tracking.pelvis_proxy_programme import PlanView
from .row_scene_projection import prepare_render_projection

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data')
SIZE = (2580, 1080)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def check_onedrive_destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Model and RF4 JSON must stay in the OneDrive project')
    if len(str(path).encode('utf-16-le')) // 2 >= 260 or any(len(p.encode('utf-16-le')) // 2 > 255 for p in path.parts):
        raise ValueError('OneDrive destination exceeds the Windows path domain: ' + str(path))
    return path


def complete_review_scene(bundle, q):
    """Add missing actual FIT ink; existing point anchors remain point anchors."""
    scene = deepcopy(bundle.state.review_scene(q))
    descriptor = scene['source_conditioned_board_prints']
    represented = {int(p['advertisement_id']) for p in descriptor['patterns']}
    represented.update(int(p['number']) for p in advertisement_review_paths(scene)
                       if p.get('primitive') == 'polyline' and p.get('pattern_kind'))
    missing = [d for d in bundle.target_descriptors if d.get('kind') == 'ad_print'
               and int(d['advertisement_id']) not in represented]
    grouped = defaultdict(list)
    for target, path in zip(missing, bundle.state.world_paths(q, missing), strict=True):
        grouped[int(target['advertisement_id'])].append(dict(id=target['contour_key'],
            role='actual_FIT_print_' + target['contour_key'], xyz_scene_units=path.tolist()))
    for number, contours in grouped.items():
        descriptor['patterns'].append(dict(advertisement_id=number, external_contours=contours,
            actual_material_producer='FirstShotPhysicalState.world_paths:same_q_actual_FIT',
            source_template_modified=False, independent_print_world_measurements_added=False))
    return scene


def verify_source_clock(source, pts, *, verify_video=True):
    expected = np.arange(256256, 575489, 256, dtype=np.int64)
    if (source['native_fps'] != '50/1' or source['time_base'] != '1/12800'
            or not np.array_equal(pts, expected)):
        raise ValueError('Complete official 1248-exposure native-50 clock required')
    path = Path(source['path']).resolve()
    if reference(path)['sha256'] != source['sha256']:
        raise ValueError('Official source video bytes changed')
    if verify_video:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Exactly one official source video stream required')
            stream = container.streams.video[0]
            if (stream.time_base != Fraction(1, 12800) or stream.average_rate != Fraction(50, 1)
                    or (stream.width, stream.height) != (1920, 1080)):
                raise ValueError('Official source dimensions or native clock changed')
    return reference(path)


def review_source_uv(frame):
    """Show every retained source fact, including explicitly quarantined IDs."""
    active = np.concatenate([row['uv'] for row in frame['rows']])
    inactive = frame.get('inactive_source_correspondences', [])
    if inactive:
        return np.vstack((active, np.asarray([p['UV'] for p in inactive], float)))
    return active


class FirstShotReview:
    def __init__(self, bundle, q, checkpoint_receipt):
        self.bundle = bundle
        self.q = np.asarray(q, np.float64).copy()
        self.q.setflags(write=False)
        self.q_sha256 = hashlib.sha256(self.q.tobytes()).hexdigest()
        self.checkpoint_receipt = deepcopy(checkpoint_receipt)
        bundle.state.guard(self.q)
        binding = read_reference(bundle.state.binding_reference)
        self.source = binding['source']
        self.video_reference = verify_source_clock(self.source, bundle.state.pts)
        original_count = sum(f.get('original_approved_source_point_count', f['source_point_count']) for f in bundle.frames)
        self.active_source_points = sum(f['source_point_count'] for f in bundle.frames)
        self.review_source_points = sum(len(review_source_uv(f)) for f in bundle.frames)
        revision = getattr(bundle, 'source_revision_proof', None)
        added_count = 0 if revision is None else revision['added_native_readouts']
        native_revision = getattr(bundle, 'source_native_line_revision_proof', None)
        if native_revision is not None:
            if native_revision.get('old_representatives_removed_atomically') is not True:
                raise ValueError('Native review requires authenticated whole-group replacement')
            added_count += sum(item['new_raw_points'] - item['old_raw_points']
                               for item in native_revision['frames'] if not item['unchanged'])
        self.native_line_revision_proof = native_revision
        if (len(bundle.frames) != 1248 or original_count != 192725
                or self.review_source_points != original_count + added_count):
            raise ValueError('The review must retain every approved first-shot point')
        self.scene = prepare_metric_review_scene(complete_review_scene(bundle, self.q))
        self.review_grid_conversion = validate_metric_review_scene(self.scene)
        renderer_type = make_board_grid_renderer(
            SceneRenderer, scene_unit_m=self.review_grid_conversion['metres_per_scene_unit'])
        self.renderer = renderer_type(self.scene, draw_grid=True, draw_advertisements=True)
        self.plan = PlanView(self.scene, 0., board_paths=self.renderer.board_grid.paths)
        ids = sorted({int(p['number']) for p in self.renderer.advertisement_paths})
        if self.renderer.path_counts != self.plan.path_counts or ids != self.plan.advertisement_ids:
            raise ValueError('Source view and plan do not share the complete geometry inventory')
        ad12 = [p for p in advertisement_review_paths(self.scene) if int(p['number']) == 12]
        anchors = sum(len(p['xyz']) for p in ad12 if p.get('primitive') == 'points')
        actual = [d for d in bundle.target_descriptors if d.get('kind') == 'ad_print' and d['advertisement_id'] == 12]
        patterns = [p for p in self.scene['source_conditioned_board_prints']['patterns'] if p['advertisement_id'] == 12]
        contours = {c['id']: np.asarray(c['xyz_scene_units'], float) for p in patterns for c in p['external_contours']}
        if anchors != 3 or len(actual) != 2:
            raise ValueError('Ad12 review requires its actual two contours and three independent anchors')
        for target, path in zip(actual, bundle.state.world_paths(self.q, actual), strict=True):
            if not np.array_equal(contours.get(target['contour_key']), path):
                raise ValueError('Ad12 displayed contour differs from its actual FIT world path')
        self.ad12_proof = dict(actual_contour_keys=[d['contour_key'] for d in actual], actual_curves=2,
            actual_world_paths_byte_exact=True, independent_anchor_points=anchors, anchors_connected_as_triangle=False)
        self.right = self.plan.base.copy()
        self.right[510:] = 248  # reuse only PlanView's geometry, not its body explanatory text
        for i, text in enumerate(('冰面/围挡1米格 · 60×30统一换算', '全部广告边界、实际印刷轮廓与编号',
                                  '左侧显示原审批与本次核实取点' if revision else '左侧显示这一帧审批过的取点',
                                  '官方单节目 50 帧原视频', '左右显示同一拟合模型',
                                  '修订模型，完整镜头审阅' if revision else '成本平台停止，等待审阅')):
            self.right = self.plan.text(self.right, (20, 535 + 58*i), text, small=True)
        self.frames = {int(f['native_pts']): f for f in bundle.frames}
        self.sources = {}
        for frame in bundle.frames:
            packet = read_reference(frame['source_packet'])
            source = packet['source']
            if (source['native_pts'] != frame['native_pts'] or source['actual_pts'] != frame['native_pts']
                    or source['time_base'] != '1/12800' or source['source_video_sha256'] != self.source['sha256']):
                raise ValueError('Approved packet belongs to another native exposure')
            self.sources[int(frame['native_pts'])] = source
        self.last_metrics = None

    @classmethod
    def from_checkpoint(cls, binding, association, checkpoint, sha256, consumer_snapshot=None):
        configure_native_runtime(stage='first_shot_official50_review',native_threads=3)
        association_ref = reference(association)
        bundle = FirstShotSourceBundle(binding, read_reference(association_ref), geometry_workers=1, source_workers=1)
        try:
            q, receipt = load_seed_state(bundle, checkpoint, sha256, association_ref, consumer_snapshot)
            return cls(bundle, q, receipt)
        except BaseException:
            bundle.close()
            raise

    def close(self):
        self.bundle.close()

    def draw(self, native, pts, *, include_quarantined=False):
        pts = int(pts)
        frame, source = self.frames[pts], self.sources[pts]
        if reference(frame['source_packet']['path']) != frame['source_packet']:
            raise ValueError('Approved source packet changed during review')
        if not isinstance(native, np.ndarray) or native.shape != (1080, 1920, 3) or native.dtype != np.uint8:
            raise ValueError('Original native BGR pixels required')
        if hashlib.sha256(native.tobytes()).hexdigest() != source['decoded_BGR_sha256']:
            raise ValueError('Decoded pixels differ from this approved native exposure')
        if reference(source['mask_path'])['sha256'] != source['mask_sha256']:
            raise ValueError('Native source foreground mask changed')
        mask = cv2.imread(source['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (1080, 1920):
            raise ValueError('Native source mask missing or dimension changed')
        camera = self.bundle.state.camera25(self.q, frame['index'])
        projection=prepare_render_projection(self.bundle.state,self.q,frame['index'])
        if projection is None:left, metrics = self.renderer.draw(native,camera,mask != 0)
        else:left, metrics = self.renderer.draw(native,camera,mask != 0,projection_context=projection)
        uv = review_source_uv(frame)
        if len(uv) != frame['source_point_count'] + len(frame.get('inactive_source_correspondences', [])):
            raise ValueError('Approved current-frame source UV inventory changed')
        inactive=frame.get('inactive_source_correspondences',[])
        if inactive:
            draw_current_source_pixel_markers(left,np.concatenate([row['uv'] for row in frame['rows']]))
            if include_quarantined:
                for record in inactive:
                    cv2.drawMarker(left,tuple(np.rint(record['UV']).astype(int)),(0,150,255),cv2.MARKER_TILTED_CROSS,7,1,cv2.LINE_AA)
        else:draw_current_source_pixel_markers(left, uv)
        right = self.plan.text(self.right.copy(), (20, 940), '第一镜头：完整原片审阅', small=True)
        if include_quarantined and inactive:
            right=self.plan.text(right,(20,900),'橙叉：保留历史、不入拟合',small=True)
        right = self.plan.text(right, (20, 980), f'原片 {pts/12800:.3f}s | PTS {pts} | frame {frame["index"]}', small=True)
        displayed_points=frame['source_point_count']+(len(inactive) if include_quarantined else 0)
        right = self.plan.text(right, (20, 1020), f'当前曝光取点：{displayed_points}；拟合：{frame["source_point_count"]}', small=True)
        canvas = np.hstack((left,right))
        self.last_metrics = dict(index=frame['index'], native_pts=pts, reviewed_points=displayed_points,
            approved_points=frame.get('original_approved_source_point_count', frame['source_point_count']),
            active_fit_points=frame['source_point_count'],
            source_revision=frame.get('source_revision_reference'),
            source_native_line_revision=(None if self.native_line_revision_proof is None
                                         else self.native_line_revision_proof['reference']),
            source_packet=frame['source_packet'], native_BGR_sha256=source['decoded_BGR_sha256'],
            native_mask_sha256=source['mask_sha256'], camera25_sha256=hashlib.sha256(camera.tobytes()).hexdigest(),
            approved_UV_sha256=hashlib.sha256(uv.tobytes()).hexdigest(), renderer=metrics,
            same_q_geometry_RF4_and_plan=True, q_sha256=self.q_sha256)
        if inactive:self.last_metrics['quarantined_historical_points_displayed']=bool(include_quarantined)
        if projection is not None:self.last_metrics['row_projection']=projection.provenance()
        return canvas

    def parameters(self):
        state = self.bundle.state
        shared = np.asarray(state.global_shared_indices, int)
        exposures = []
        reconstructed = np.empty(state.count, np.float64)
        covered = np.zeros(state.count,bool)
        reconstructed[shared] = self.q[shared]
        covered[shared] = True
        for frame in self.bundle.frames:
            columns = state.view_columns(frame['index'])
            if covered[columns].any():raise ValueError('RF4 export repeats a physical coordinate')
            reconstructed[columns] = self.q[columns]
            covered[columns] = True
            exposures.append(dict(index=frame['index'], native_pts=int(frame['native_pts']),
                RF4_columns=columns.tolist(), RF4=self.q[columns].tolist(),
                camera25=state.camera25(self.q, frame['index']).tolist()))
        if not covered.all() or reconstructed.tobytes() != self.q.tobytes():
            raise ValueError('Small parameter JSON omitted a physical or RF4 coordinate')
        result = dict(schema='rinkspace.first-shot-same-q-review-parameters.v1', q_sha256=self.q_sha256,
            checkpoint=self.checkpoint_receipt['archive'], source_binding=state.binding_reference,
            source=self.video_reference, source_clock=state.source_clock, parameter_count=state.count,
            view_stop=state.view_stop, global_shared_columns=shared.tolist(), global_shared_values=self.q[shared].tolist(),
            world_parameter_indices=state.world_parameter_indices.tolist(), exposures=exposures,
            native_RF4_names=['rotation_x_rad', 'rotation_y_rad', 'rotation_z_rad', 'log_effective_focal'],
            print_layout_advertisement_ids=list(state.print_layout_advertisement_ids),
            scene_unit_working_m=state.scene_unit_working_m, verified_metric_scale=False,
            review_grid_conversion=deepcopy(self.review_grid_conversion),
            every_coordinate_reconstructed_byte_exact=True, accuracy_accepted=False)
        row_index=getattr(state,'row_readout_index',None)
        if row_index is not None:
            from .first_shot_row_camera import prepare_frame_row_camera
            result['row_camera_model']=dict(row_timing_reference=state.row_timing_reference,
                shared_readout_parameter_index=int(row_index),signed_readout_milliseconds=float(self.q[row_index]),
                positive_is_top_to_bottom=True,camera25_values_are_midrow_only=True,
                complete_row_trajectory_reconstructed_from_all_native_RF4_and_shared_installation_and_tau=True,
                trajectory=prepare_frame_row_camera(state,self.q,1).provenance())
        return result

    def export_model(self, destination):
        destination = Path(destination).resolve()
        paths = [check_onedrive_destination(destination/name) for name in ('scene_candidate.json', 'parameters_candidate.json', 'model_entry.json')]
        if any(p.exists() for p in paths):
            raise FileExistsError('Preserve existing model exports; choose a new short destination')
        destination.mkdir(parents=True, exist_ok=True)
        _write(paths[0], self.scene)
        _write(paths[1], self.parameters())
        entry = dict(schema='rinkspace.first-shot-same-q-review-model.v1', q_sha256=self.q_sha256,
            scene=reference(paths[0]), parameters=reference(paths[1]), checkpoint=self.checkpoint_receipt['archive'],
            source_binding=self.bundle.state.binding_reference, ad12=self.ad12_proof, accuracy_accepted=False)
        _write(paths[2], entry)
        return reference(paths[2])


def verify_video(path, expected_count):
    count = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if len(video.streams.video) != 1 or (stream.width, stream.height) != SIZE or stream.average_rate != Fraction(50,1):
            raise ValueError('Review video dimensions or native-50 cadence changed')
        for frame in video.decode(stream):
            if frame.pts is None or frame.pts * stream.time_base != Fraction(count,50):
                raise ValueError('Review output exposure timing changed')
            count += 1
    if count != expected_count:
        raise ValueError('Review video does not cover every native first-shot frame')
    return dict(decoded_frames=count, size_wh=list(SIZE), fps='50/1', duration_seconds=count/50)


def render_video(view, output):
    """Explicit complete-shot action; reusable I/O, exact official native clock."""
    output = Path(output).resolve()
    if not output.is_relative_to(DATA) or output.exists():
        raise ValueError('Use a new Data video path')
    temporary = output.with_name(output.stem + '.partial.mp4')
    if temporary.exists():
        raise FileExistsError('Preserve existing review video intermediates')
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_executable(); encoder, encoder_name = _available_encoder(ffmpeg)
    command = [ffmpeg, '-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{SIZE[0]}x{SIZE[1]}','-framerate','50','-i','pipe:0','-an',*encoder,
        '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    count = 0
    coverage = output.with_suffix('.coverage.jsonl')
    if coverage.exists():raise FileExistsError(coverage)
    with output.with_suffix('.encoding.txt').open('wb') as log, coverage.open('x',encoding='utf8') as ledger:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for frame, pts in zip(view.bundle.frames, view.bundle.state.pts, strict=True):
                if int(frame['native_pts']) != int(pts):raise ValueError('Native packet/render clock changed')
                # Lossless original-source frames are already authenticated by
                # the accepted packet. No duplicate frame library is produced.
                native=cv2.imread(view.sources[int(pts)]['image_path'])
                canvas = view.draw(native,pts)
                process.stdin.write(memoryview(canvas).cast('B'));count += 1
                ledger.write(json.dumps(view.last_metrics,ensure_ascii=False,allow_nan=False)+'\n')
                if count % 50 == 0:
                    print(json.dumps(dict(rendered_frames=count, native_pts=int(pts))),flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Review encoder failed; see the preserved Data encoding log')
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
    if count != 1248:raise ValueError('Complete first-shot render requires exactly 1248 frames')
    verification = verify_video(temporary,count)
    temporary.replace(output)
    return dict(video=reference(output),verification=verification,encoder=encoder_name,coverage=reference(coverage),
        source_start_pts=int(view.bundle.state.pts[0]), source_end_pts_exclusive=int(view.bundle.state.pts[-1])+256,
        native_pts_sha256=hashlib.sha256(view.bundle.state.pts.tobytes()).hexdigest(),
        q_sha256=view.q_sha256, source_clock='1/12800', accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binding','association','checkpoint','checkpoint-sha256','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--consumer-snapshot')
    parser.add_argument('--check-frames',type=int,nargs='*',default=[])
    parser.add_argument('--export-model')
    parser.add_argument('--video',action='store_true')
    args = parser.parse_args(argv)
    output=Path(args.output).resolve()
    if not output.is_relative_to(DATA):raise ValueError('Review media and diagnostics stay in Data')
    if any(i<1 or i>1248 for i in args.check_frames):raise ValueError('Static checks must address actual first-shot frames')
    output.mkdir(parents=True,exist_ok=True)
    view=FirstShotReview.from_checkpoint(args.binding,args.association,args.checkpoint,args.checkpoint_sha256,args.consumer_snapshot)
    try:
        checks=[]
        for index in sorted(set(args.check_frames)):
            frame_started=time.perf_counter()
            pts=int(view.bundle.state.pts[index-1]);source=view.sources[pts]
            native=cv2.imread(source['image_path'])
            path=output/f'review_{view.q_sha256[:12]}_{index:04d}.png'
            if path.exists():raise FileExistsError(path)
            canvas=view.draw(native,pts,include_quarantined=True)
            if not cv2.imwrite(str(path),canvas):raise OSError(path)
            checks.append(dict(image=reference(path),seconds=time.perf_counter()-frame_started,**view.last_metrics))
        exported=None if args.export_model is None else view.export_model(args.export_model)
        video=None if not args.video else render_video(view,output/f'first_shot_{view.q_sha256[:12]}_candidate.mp4')
        result=dict(schema='rinkspace.complete-first-shot-same-q-review.v1',q_sha256=view.q_sha256,
            checkpoint=view.checkpoint_receipt,source=view.video_reference,source_clock=view.bundle.state.source_clock,
            review_producers=[reference(__file__), reference(Path(__file__).with_name('board_grid_overlay.py')),
                reference(Path(__file__).with_name('current_source_points.py')),
                reference(Path(__file__).parent/'body_tracking/pelvis_proxy_programme.py')],
            frames=1248,source_points=view.review_source_points,active_fit_points=view.active_source_points,
            source_revision=getattr(view.bundle, 'source_revision_proof', None),
            all_geometry_and_advertisement_numbers_enabled=True,
            left_and_right_same_scene=True,source_and_plan_inventory=view.renderer.path_counts,
            advertisement_ids=view.plan.advertisement_ids,ad12=view.ad12_proof,
            working_scale_is_verified_metres=False,model_export=exported,static_checks=checks,video=video,
            optimizer_started=False,source_or_weights_modified=False,accuracy_accepted=False,
            static_checks_do_not_validate_the_complete_shot=True)
        entry=output/f'review_{view.q_sha256[:12]}.json';_write(entry,result)
        print(json.dumps(dict(entry=reference(entry),static_checks=len(checks),video_generated=video is not None)),flush=True)
        return result
    finally:view.close()


if __name__=='__main__':main()
