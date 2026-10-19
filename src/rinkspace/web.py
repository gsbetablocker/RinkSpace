"""Loopback-only workbench. Source files are read-only; edits are versioned JSON."""
from __future__ import annotations

import copy
import json
import math
import mimetypes
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .io import dump_json, load_json
from .storage import AssetStore


def byte_range(header, size):
    """A single HTTP byte range for seekable, read-only local media."""
    if not header:
        return 0, size - 1
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError('Unsupported byte range')
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError('Empty suffix range')
        return max(0, size-suffix), size-1
    start, end = int(first), min(int(last), size-1) if last else size-1
    if start > end or start >= size:
        raise ValueError('Unsatisfiable byte range')
    return start, end


class Workbench:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.storage = AssetStore(self.root)
        self.directory = self.storage.directory('outputs/workbench', 'workbench')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'project.json'
        self.lock = threading.RLock()
        from .player_cache import LocalizationCache
        self.localization_cache = LocalizationCache(self.root)
        self.jobs = {}
        self.project = load_json(self.path) if self.path.exists() else {
            'id': 'rinkspace-local', 'name': 'Ankara · 冰面参照工作台',
            'revision': 0, 'selected_video_id': None, 'annotations': [], 'history': [], 'shot_edits': {}}

    def persist(self):
        temp = self.path.with_suffix('.tmp')
        dump_json(temp, self.project)
        temp.replace(self.path)

    def snapshot(self, action):
        prior = {k: copy.deepcopy(v) for k, v in self.project.items() if k != 'history'}
        self.project['history'].append({'action': action, 'at_unix_s': time.time(), 'previous': prior})
        self.project['revision'] += 1

    def state(self):
        from .workspace import load_sources
        videos = load_sources(self.root)
        shots = []
        for video in videos:
            meta = video.get('metadata') or {}
            video.update(id=video['key'], label=video.get('title', video['key']), indexed=video['prepared'],
                         frame_count=meta.get('frame_count', 0), fps=meta.get('declared_average_rate'),
                         duration_s=meta.get('duration_s'), width=meta.get('width'), height=meta.get('height'),
                         media_url='/api/media?video_id=' + video['key'] if video['exists'] else None)
            for raw in video.get('shots', []):
                role = raw.get('role', 'unknown').replace('_candidate', '')
                shots.append({**raw, 'id': video['key'] + ':' + raw['shot_id'], 'video_id': video['key'],
                    'end_frame': raw['end_frame_exclusive']-1,
                    'label': 'non_programme' if role == 'nonprogramme' else role,
                    'status': raw.get('review_status'), 'confidence': raw.get('confidence')})
        for shot in shots:
            shot.update(self.project['shot_edits'].get(shot['id'], {}))
        public = {k: v for k, v in self.project.items() if k != 'history'}
        public.update(videos=videos, shots=shots, can_undo=bool(self.project['history']))
        return {'project': public, 'runs': self.runs(), 'jobs': list(self.jobs.values()),
                'capabilities': {'source_frames': True, 'mask_edit': True, 'landmarks': True,
                    'shared_rink_model': True, 'original_video_grid': True,
                    'registration': True, 'metric': False, 'trajectory': False,
                    'registration_backends': ['photometric_affine', 'rotation_zoom_experimental', 'baseline'],
                    'default_registration_backend': 'photometric_affine', 'default_every_n': 1,
                    'note': '共同模型定位与原片坐标网可浏览；模型几何与独立精度状态分别保留，米制未默认成立。'}}

    def submit(self, name, function):
        job_id = uuid.uuid4().hex[:12]
        job = {'id': job_id, 'job_id': job_id, 'name': name, 'status': 'running', 'started_at': time.time()}
        self.jobs[job_id] = job
        def worker():
            try:
                result = function()
                job.update(status='complete', result=result)
            except Exception as error:
                job.update(status='failed', error=str(error))
                (self.directory / f'error-{job_id}.txt').write_text(traceback.format_exc(), encoding='utf-8')
            job['finished_at'] = time.time()
        threading.Thread(target=worker, daemon=True).start()
        return job.copy()

    def runs(self):
        result = []
        directories = {self.storage.resolve('outputs/runs'), self.storage.output('runs')}
        for path in sorted((path for directory in directories for path in directory.glob('*/run.json')), reverse=True):
            row = load_json(path)
            row['report_url'] = self.asset_url(path.parent / 'index.html')
            bundle = path.parent / 'review.zip'
            row['review_url'] = self.asset_url(bundle) if bundle.exists() else None
            result.append(row)
        return result

    def asset_url(self, path):
        return self.storage.asset_url(path)

    def _evidence_url(self, directory, relative):
        """Expose only existing derived evidence, including relative model links."""
        if not isinstance(relative, str) or not relative:
            return None
        try:
            path = self.storage.resolve(relative, base=directory)
            return self.asset_url(path) if path.is_file() else None
        except ValueError:
            return None

    def rink_models(self):
        result = []
        preferred_path = self.root / 'configs/rink_model/active_delivery.json'
        preferred = load_json(preferred_path) if preferred_path.is_file() else {}
        active_model = self.storage.resolve(preferred['model_path']) if isinstance(preferred.get('model_path'), str) else None
        for path in sorted(self.storage.resolve('outputs/rink_model').glob('*/*/rink_model.json')):
            row = load_json(path)
            if path.resolve() == active_model:
                row['preferred_delivery'] = True
                row['preferred_run_ids'] = {k: v for k, v in preferred.get('preferred_run_ids', {}).items()
                                            if isinstance(k, str) and isinstance(v, str)}
            review_path = path.parent / 'review_status.json'
            if review_path.is_file():
                review = load_json(review_path)
                row['review_status'] = review.get('status')
                row['review_label_zh'] = review.get('label_zh')
                if isinstance(review.get('reference_source_count'), int):
                    row['reference_source_count'] = review['reference_source_count']
            row['document_url'] = self.asset_url(path)
            row['evidence_url'] = self._evidence_url(path.parent, 'index.html')
            whole = row.get('schema_version') == 'rinkspace.whole-rink.v1'
            row['overview_url'] = self._evidence_url(path.parent, row.get('overview_path', 'whole_rink.png' if whole else 'model_overview.jpg'))
            if whole and not row['evidence_url']:
                row['evidence_url'] = self._evidence_url(path.parent, 'render_manifest.json')
            for view in row.get('view_poses', []):
                view['image_url'] = self._evidence_url(self.root, view.get('image'))
            for view in row.get('reference_views', []):
                view['id'] = view.get('id', view.get('reference_id', view.get('view_id')))
                view['image_url'] = self._evidence_url(path.parent, view.get('image_path'))
            for landmark in row.get('landmarks', []):
                for observation in landmark.get('observations', []):
                    observation['image_url'] = self._evidence_url(path.parent, observation.get('image_path'))
            result.append(row)
        return result

    def model_localizations(self, video_id=None):
        result = []
        paths = [(folder, path) for folder in ('model_localization', 'whole_rink_localization')
                 for path in sorted(self.storage.resolve('outputs/' + folder).glob('*/localization.json'))]
        for folder, path in paths:
            with self.lock:
                row = self.localization_cache.summary(path)
            if video_id is not None and row.get('source_id', row.get('video_id')) != video_id:
                continue
            row.update(run_id=('whole:' if folder == 'whole_rink_localization' else '') + path.parent.name,
                       storage_folder=folder, storage_run_name=path.parent.name,
                       document_url=self.asset_url(path),
                       report_url=self._evidence_url(path.parent, 'index.html'))
            result.append(row)
        return result

    def model_localization(self, run_id):
        # Bounded run-name lookup avoids scanning every large evidence file.
        whole=isinstance(run_id,str) and run_id.startswith('whole:')
        name=run_id.removeprefix('whole:') if isinstance(run_id,str) else ''
        if not name or name in ('.','..') or '/' in name or '\\' in name or ':' in name:
            raise ValueError('Unknown model localization run')
        folder=self.storage.resolve('outputs/'+('whole_rink_localization' if whole else 'model_localization'))
        path=folder/name/'localization.json'
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Unknown model localization run')
        with self.lock:
            row=self.localization_cache.player(path)
        row.update(run_id=run_id,document_url=self.asset_url(path))
        return row

    def frame_meta(self, video_id, index, run_id=None):
        from .workspace import source_frame_meta
        row = source_frame_meta(self.root, video_id, index)
        row['image_url'] = f'/api/frame?video_id={video_id}&index={index}'
        keys = [a for a in self.project['annotations'] if a.get('video_id') == video_id and a['kind'] == 'keyframe']
        reference = keys[-1]['source_frame_index'] if keys else 0
        row.update(reference_url=f'/api/frame?video_id={video_id}&index={reference}',
                   reference_frame_index=reference, reference_width=row['width'], reference_height=row['height'])
        row['registration'] = {'status': 'unavailable', 'reason': '此源帧未生成配准结果，选择短区间运行。'}
        for run in self.runs():
            if run.get('video_id') != video_id:
                continue
            if run_id and run['id'] != run_id:
                continue
            path = next((directory/run['id']/'registration.json' for directory in
                (self.storage.output('runs'), self.storage.resolve('outputs/runs'))
                if (directory/run['id']/'registration.json').is_file()), self.storage.output('runs')/run['id']/'registration.json')
            if not path.exists():
                continue
            registration = load_json(path)
            samples = registration.get('samples', registration.get('frames', []))
            valid_frames = sorted(s['source_frame_index'] for s in samples if s.get('source_frame_index') is not None)
            row['previous_evidence_frame'] = next((n for n in reversed(valid_frames) if n < index), None)
            row['next_evidence_frame'] = next((n for n in valid_frames if n > index), None)
            match = next((s for s in samples if s.get('source_frame_index') == index), None)
            if match:
                # Review conclusions describe the saved run without rewriting
                # its original numerical candidate status or fitted matrices.
                row['run_id'] = run['id']
                row['backend'] = run.get('backend', 'baseline')
                row['registration'] = copy.deepcopy(match)
                if match.get('photometric_audit_path'):
                    row['photometric_audit_url'] = self.asset_url(path.parent / match['photometric_audit_path'])
                    row['photometric_audit_manifest_url'] = self.asset_url(path.parent / 'footprints/manifest.json')
                for field in ('review_disposition', 'review_note'):
                    if run.get(field):
                        row[field] = run[field]
                        row['registration'][field] = run[field]
                if match.get('reference_frame_only'):
                    row['error_reason'] = '这是参考帧自身；其误差不计入独立验收统计。'
                if run.get('review_disposition') == 'rejected_plane_contamination':
                    row['diagnostic_preview_only'] = True
                    row['fixed_reason'] = '已否定的污染基线；此图仅保留作失败诊断，不是有效冰面补偿。'
                    row['error_reason'] = '此运行已确认非冰面污染与漂移；原数值仅作失败诊断，不能作为独立验收通过。'
                reference_row = next((s for s in samples if s['sample_index'] == match.get('reference_sample_index')), None)
                if reference_row:
                    reference = reference_row['source_frame_index']
                    row.update(reference_frame_index=reference, reference_url=f'/api/frame?video_id={video_id}&index={reference}')
                for key, out_key in [('fixed_image', 'fixed_url'), ('mask_image', 'mask_url'), ('error_image', 'error_url')]:
                    if match.get(key):
                        row[out_key] = self.asset_url(path.parent / match[key])
                break
        return row

    def edit(self, data):
        action = data['action']
        with self.lock:
            if action == 'undo':
                if self.project['history']:
                    entry = self.project['history'].pop()
                    history = self.project['history']
                    revision = self.project['revision'] + 1
                    self.project = entry['previous']
                    self.project.update(history=history, revision=revision)
            elif action == 'save':
                pass
            elif action == 'select_video':
                self.snapshot(action)
                self.project['selected_video_id'] = data['video_id']
            elif action == 'edit_shot':
                changes = data.get('changes', data.get('shot', {}))
                if not changes:
                    changes = {k: data[k] for k in ('label', 'start_s', 'end_s', 'start_frame', 'end_frame') if k in data}
                if changes.get('label') and changes['label'] not in ('programme', 'replay', 'non_programme', 'unknown', 'transition', 'live', 'off_ice'):
                    raise ValueError('Unknown shot label')
                if 'start_s' in changes and 'end_s' in changes and changes['end_s'] <= changes['start_s']:
                    raise ValueError('镜头结束必须晚于开始')
                if 'start_frame' in changes or 'end_frame' in changes:
                    from .workspace import load_index
                    frames = load_index(self.root, data['video_id'])['frames']
                    start_frame, end_frame = int(changes['start_frame']), int(changes['end_frame'])
                    if not 0 <= start_frame <= end_frame < len(frames):
                        raise ValueError('镜头边界超出完整源帧索引')
                    changes.update(start_frame=start_frame, end_frame=end_frame,
                                   start_s=frames[start_frame]['source_time_s'],
                                   end_s=frames[end_frame+1]['source_time_s'] if end_frame+1 < len(frames) else frames[end_frame]['source_time_s'])
                self.snapshot(action)
                shot_id = data.get('shot_id', changes.get('id'))
                if 'start_frame' in changes:
                    shot_rows = sorted((s for s in self.state()['project']['shots'] if s['video_id'] == data['video_id']), key=lambda s:s['start_frame'])
                    pos = next(i for i,s in enumerate(shot_rows) if s['id'] == shot_id)
                    if (pos == 0 and start_frame != 0) or (pos == len(shot_rows)-1 and end_frame != len(frames)-1):
                        self.project['history'].pop(); self.project['revision'] -= 1
                        raise ValueError('首尾镜头需保留源视频首尾边界')
                    if pos > 0:
                        previous = shot_rows[pos-1]
                        if start_frame <= previous['start_frame']:
                            self.project['history'].pop(); self.project['revision'] -= 1
                            raise ValueError('开始边界不能吞掉前一镜头')
                        self.project['shot_edits'].setdefault(previous['id'], {}).update(
                            end_frame=start_frame-1, end_s=frames[start_frame]['source_time_s'], status='corrected')
                    if pos+1 < len(shot_rows):
                        following = shot_rows[pos+1]
                        if end_frame >= following['end_frame']:
                            self.project = self.project['history'].pop()['previous'] | {'history': self.project['history']}
                            raise ValueError('结束边界不能吞掉后一镜头')
                        self.project['shot_edits'].setdefault(following['id'], {}).update(
                            start_frame=end_frame+1, start_s=frames[end_frame+1]['source_time_s'], status='corrected')
                self.project['shot_edits'].setdefault(shot_id, {}).update(changes, status='corrected')
            elif action in ('add_mask', 'add_landmark', 'add_keyframe'):
                annotation = copy.deepcopy(data.get('annotation', data))
                annotation.pop('action', None)
                annotation.update(id=uuid.uuid4().hex[:12], kind=action.removeprefix('add_'),
                                  revision=self.project['revision'] + 1, provenance='user_review', created_at=time.time())
                self.snapshot(action)
                self.project['annotations'].append(annotation)
            elif action == 'delete_annotation':
                self.snapshot(action)
                self.project['annotations'] = [a for a in self.project['annotations'] if a['id'] != data['annotation_id']]
            else:
                raise ValueError(f'Unknown project action: {action}')
            self.persist()
        return self.state()

    def run(self, data):
        from .workspace import sample_source_window
        from .review import build_run_report, attach_independent_audit
        backend = data.get('backend', 'photometric_affine')
        if backend not in ('photometric_affine', 'rotation_zoom_experimental', 'baseline'):
            raise ValueError('Unknown registration backend')
        if backend in ('photometric_affine', 'rotation_zoom_experimental'):
            from .photometric_sequence import register_photometric_sequence as register_sequence
        else:
            from .local_registration import register_sequence
        video_id = data['video_id']
        start, end = float(data['start_s']), float(data['end_s'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end - start > 60:
            raise ValueError('短镜头需满足 0 <= 开始 < 结束，且长度不超过60秒')
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        out = self.storage.output('runs') / run_id
        samples = out / 'source'
        every_n = int(data.get('every_n', 1))
        if every_n < 1 or float(data.get('every_n', 1)) != every_n:
            raise ValueError('every_n must be a positive integer')
        with self.lock:
            annotations = copy.deepcopy([a for a in self.project['annotations'] if a.get('video_id') == video_id])
            project_revision = self.project['revision']
            shots = [s for s in self.state()['project']['shots'] if s['video_id'] == video_id]
        sample_source_window(self.root, video_id, start, end, every_n, 960, samples)
        source = load_json(samples / 'samples.json')
        exclusions = {}
        corrections = {}
        for sample in source['samples']:
            i = str(sample['sample_index'])
            scale = sample.get('image_scale_from_source', 1)
            for a in annotations:
                if a.get('source_frame_index') != sample.get('source_frame_index'):
                    continue
                if a['kind'] == 'mask':
                    points = [[float(x)*scale, float(y)*scale] for x,y in a.get('points', a.get('polygon', []))]
                    if len(points) >= 3:
                        if a.get('reason') == 'heldout_protection':
                            exclusions.setdefault(i, []).append({'polygon': points, 'physical_landmark_id': a.get('physical_landmark_id', a.get('group_id'))})
                        else:
                            field = 'include_polygons' if a.get('operation') == 'include' else 'exclude_polygons'
                            corrections.setdefault(i, {}).setdefault(field, []).append(points)
                elif a['kind'] == 'landmark' and a.get('role') != 'training':
                    xy = [float(v)*scale for v in a['source_xy']]
                    exclusions.setdefault(i, []).append({'xy': xy, 'radius_px': float(a.get('radius_px', 40))*scale,
                        'physical_landmark_id': a.get('physical_landmark_id')})
        config = copy.deepcopy(data.get('config', {}))
        # Protection belongs to the saved project snapshot. A configuration
        # override must not erase its independent physical-group exclusions.
        config.pop('exclusions', None)
        config['corrections'] = corrections
        boundaries = sorted(s['start_frame'] for s in shots)
        shot_breaks = []
        for previous, current in zip(source['samples'], source['samples'][1:]):
            if any(previous['source_frame_index'] < cut <= current['source_frame_index'] for cut in boundaries):
                shot_breaks.append(current['sample_index'])
        options = {'config': config, 'exclusions': exclusions, 'shot_breaks': shot_breaks}
        freeze_path = self.root / 'configs/continuous_holdout_lock.json'
        if backend == 'photometric_affine' and freeze_path.exists():
            options['freeze_lock'] = freeze_path
        if backend == 'rotation_zoom_experimental':
            options['backend'] = backend
        register_sequence(samples, out, **options)
        info = {'id': run_id, 'video_id': video_id, 'start_s': start, 'end_s': end,
                'first_frame_index': source['samples'][0]['source_frame_index'],
                'role': data.get('role', 'development'), 'project_revision': project_revision,
                'backend': backend, 'every_n': every_n,
                'created_at': time.time(), 'annotations': annotations, 'status': 'candidate'}
        if backend == 'photometric_affine':
            info.update(review_disposition='candidate_not_independently_validated',
                review_note='Affine 光度候选，尚未独立验收。拟合采样不是 RANSAC 内点；手工标注只作用于对应源帧，掩膜仍需目视检查。完整拟合足迹保存在本机 footprints，紧凑 review.zip 不含全部足迹。',
                full_fitting_audit_url=self.asset_url(out / 'footprints/manifest.json'))
        elif backend == 'rotation_zoom_experimental':
            info.update(review_disposition='experimental_candidate_not_independently_validated',
                review_note='旋转与变焦约束实验：假定光心固定、主点居中、初始焦距为图宽4倍，尚未标定。拟合仅使用允许冰面；须看独立检查，不能用较平滑的画面证明精度。',
                full_fitting_audit_url=self.asset_url(out / 'footprints/manifest.json'))
        dump_json(out / 'run.json', info)
        observations = []
        by_source = {s['source_frame_index']: s for s in source['samples']}
        for annotation in annotations:
            if annotation['kind'] != 'landmark' or annotation.get('role') == 'training':
                continue
            observed = by_source.get(annotation.get('source_frame_index'))
            reference = by_source.get(annotation.get('reference_frame_index'))
            if observed is None or reference is None:
                continue
            observed_scale = observed.get('image_scale_from_source', 1)
            reference_scale = reference.get('image_scale_from_source', 1)
            observations.append({'sample_index': observed['sample_index'], 'reference_sample_index': reference['sample_index'],
                'source_xy': [float(v)*observed_scale for v in annotation['source_xy']],
                'reference_xy': [float(v)*reference_scale for v in annotation['reference_xy']],
                'physical_landmark_id': annotation['physical_landmark_id'],
                'point_id': annotation.get('point_id', annotation['physical_landmark_id']),
                'visible': annotation.get('visible', True), 'provenance': annotation['provenance']})
        if observations:
            attach_independent_audit(out, observations)
        build_run_report(out)
        return info


def serve(root='.', port=8765, open_browser=False):
    app = Workbench(root)
    static = Path(__file__).parent / 'static'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def json(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def file(self, path):
            if not path.is_file():
                raise FileNotFoundError(str(path))
            size = path.stat().st_size
            partial = self.headers.get('Range', '')
            try:
                start, end = byte_range(partial, size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                if url.path == '/api/state':
                    return self.json(app.state())
                if url.path == '/api/runs':
                    return self.json(app.runs())
                if url.path == '/api/rink-models':
                    return self.json(app.rink_models())
                if url.path == '/api/model-localizations':
                    return self.json(app.model_localizations(query.get('video_id')))
                if url.path == '/api/model-localization':
                    return self.json(app.model_localization(query['run_id']))
                if url.path.startswith('/api/jobs/'):
                    return self.json(app.jobs[url.path.rsplit('/', 1)[-1]])
                if url.path == '/api/frame-meta':
                    return self.json(app.frame_meta(query['video_id'], int(query.get('index', 0)), query.get('run_id')))
                if url.path == '/api/frame':
                    from .workspace import get_frame
                    return self.file(get_frame(app.root, query['video_id'], int(query.get('index', 0))))
                if url.path == '/api/media':
                    # Resolve only configured source keys, never a caller-supplied path.
                    from .workspace import source_path
                    return self.file(source_path(app.root, query['video_id']))
                if url.path.startswith('/api/run/'):
                    return self.json(next(r for r in app.runs() if r['id'] == url.path.rsplit('/', 1)[-1]))
                if url.path.startswith('/assets/'):
                    path = app.storage.public_asset(unquote(url.path[len('/assets/'):]))
                elif url.path.startswith('/data/'):
                    path = app.storage.public_asset('data/' + unquote(url.path[len('/data/'):]))
                elif url.path in ('/', '/START_HERE.html') and (app.root / 'START_HERE.html').is_file():
                    path = app.root / 'START_HERE.html'
                else:
                    relative = unquote(url.path).lstrip('/')
                    if relative == 'workbench.html':
                        relative = 'index.html'
                    if relative.startswith('static/'):
                        relative = relative[len('static/'):]
                    path = (static / (relative or 'index.html')).resolve()
                    if not path.is_relative_to(static.resolve()):
                        raise ValueError('Invalid static path')
                self.file(path)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as exc:
                self.json({'error': str(exc)}, 400)

        def do_HEAD(self):
            if urlsplit(self.path).path == '/api/media':
                return self.do_GET()
            self.send_error(405)

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                    raise ValueError('Only same-origin local requests are accepted')
                if 'application/json' not in self.headers.get('Content-Type', ''):
                    raise ValueError('Expected application/json')
                length = int(self.headers.get('Content-Length', 0))
                if length > 2_000_000:
                    raise ValueError('JSON payload too large')
                data = json.loads(self.rfile.read(length) or b'{}')
                route = urlsplit(self.path).path
                if route == '/api/project':
                    return self.json(app.edit(data))
                if route == '/api/config':
                    from .workspace import import_source
                    import_source(app.root, data['path'], key=data.get('video_id'), url=data.get('url'))
                    return self.json(app.state())
                if route == '/api/prepare':
                    from .workspace import prepare_source
                    return self.json(app.submit('源PTS索引与镜头候选', lambda: prepare_source(app.root, data['video_id'])))
                if route == '/api/run':
                    return self.json(app.submit('局部配准与报告', lambda: app.run(data)))
                if route == '/api/pick':
                    # Dialog runs in its own short-lived process, avoiding Tk thread reuse.
                    import subprocess, sys
                    picked = subprocess.run([sys.executable, '-c',
                        'import tkinter as t;from tkinter.filedialog import askopenfilename;r=t.Tk();r.withdraw();print(askopenfilename(filetypes=[("Video","*.mp4 *.mkv *.webm *.mov"),("All","*.*")]));r.destroy()'],
                        capture_output=True, text=True, timeout=180)
                    return self.json({'path': picked.stdout.strip()})
                raise ValueError('Unknown endpoint')
            except Exception as exc:
                self.json({'error': str(exc)}, 400)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'RinkSpace: http://127.0.0.1:{server.server_port}', flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{server.server_port}')
    server.serve_forever()
