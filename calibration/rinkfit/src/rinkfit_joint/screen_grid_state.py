"""Immutable Demo screen-grid initialization, separate from physical camera state.

The S02--S09 UV arrays are consumed directly.  S01 always remains the complete
existing physical baseline; this module neither projects nor optimizes a camera.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import file_sha256, read_json, write_json

SCHEMA = 'rinkfit.direct_screen_grid_initial_state.v1'
STATE_SCHEMA = 'rinkfit.accepted_direct_screen_grid_reference.v1'
COLORS = {'grid': (85, 220, 65), 'marking': (220, 55, 205)}


def _reference(ref, base):
    if not isinstance(ref, dict) or not ref.get('sha256'):
        raise ValueError('screen-grid dependency needs a file and SHA256')
    path = Path(ref.get('file', ref.get('path', '')))
    if not str(path) or str(path) == '.':
        raise ValueError('screen-grid dependency file is missing')
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve()
    if file_sha256(path) != ref['sha256']:
        raise ValueError('screen-grid dependency SHA changed: ' + str(path))
    return path


def _physical_reference(path):
    """Check serialized identity/domain, without claiming physical acceptance."""
    model = read_json(path)
    if model.get('schema') != 'rinkfit.joint_event_model.v1':
        raise ValueError('current physical reference must remain a joint model')
    clock = model['source_clock']
    if clock['time_base'] != [1, 12800] or clock['frame_step_pts'] != 256 or clock['image_size'] != [1920, 1080]:
        raise ValueError('current physical reference has a different native clock')
    if set(model['shots']) != {f's{i:02d}' for i in range(1, 10)}:
        raise ValueError('screen-grid initialization requires the complete nine-shot reference')
    names = set()
    for block in model['blocks']:
        value, lower, upper, scale = (np.asarray(block[key], np.float64) for key in ('values', 'lower', 'upper', 'scales'))
        if block['name'] in names or not len(value) or not (value.shape == lower.shape == upper.shape == scale.shape):
            raise ValueError('invalid physical reference block identity/shape')
        if not (np.isfinite(value).all() and np.isfinite(scale).all() and (scale > 0).all() and (value >= lower).all() and (value <= upper).all()):
            raise ValueError('physical reference outside its serialized parameter domain')
        names.add(block['name'])
    scopes = [model['shots'][f's{i:02d}']['scope'] for i in range(1, 10)]
    pts = np.concatenate([np.arange(s['first_pts'], s['last_pts'] + 1, 256, dtype=np.int64) for s in scopes])
    if not np.array_equal(pts, np.arange(256256, 2415105, 256, dtype=np.int64)):
        raise ValueError('physical reference does not cover the original 8434 exposures')
    return model


class ScreenGridState:
    """Read-only, hash-bound arrays; only one requested frame is materialized."""

    def __init__(self, manifest_file):
        self.path = Path(manifest_file).resolve()
        self.manifest = read_json(self.path)
        self.sha256 = file_sha256(self.path)
        doc = self.manifest
        if doc.get('schema') != SCHEMA:
            raise ValueError('unsupported direct screen-grid state schema')
        self.producer_physical_reference = dict(doc['current_physical_reference'])
        _reference(self.producer_physical_reference, self.path.parent)
        self.arrays = {}
        self.verified_files = {}
        for name in ('screen_uv', 'depth', 'vertex_mask_bits', 'path_offsets', 'segment_vertices', 'frame_native_pts', 'frame_shot_number', 'path_kind'):
            if name not in doc['fields']:
                raise ValueError('missing direct screen-grid array: ' + name)
        for name, ref in doc['fields'].items():
            path = Path(ref.get('file', ref.get('path', '')))
            if not path.is_absolute():
                path = self.path.parent / path
            path = path.resolve()
            if path not in self.verified_files:
                _reference(ref, self.path.parent)
                self.verified_files[path] = ref['sha256']
            elif self.verified_files[path] != ref['sha256']:
                raise ValueError('conflicting array dependency SHA')
            dtype, shape = np.dtype(ref['dtype']), tuple(ref['shape'])
            if path.suffix.lower() == '.npy':
                arr = np.load(path, mmap_mode='r', allow_pickle=False)
                if arr.dtype != dtype or arr.shape != shape:
                    raise ValueError('screen-grid NPY dtype/shape mismatch: ' + name)
            else:
                if path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
                    raise ValueError('screen-grid raw array byte count mismatch: ' + name)
                arr = np.memmap(path, dtype=dtype, shape=shape, mode='r')
            self.arrays[name] = arr
        a = self.arrays
        F, V, _ = a['screen_uv'].shape
        if (F, V, a['screen_uv'].shape[-1]) != (7186, 26238, 2) or a['screen_uv'].dtype != np.dtype('<f8'):
            raise ValueError('screen-grid must retain every S02--S09 native UV vertex in float64')
        required = {'depth': ((F, V), '<f8'), 'vertex_mask_bits': ((F, V), 'u1'),
                    'path_offsets': ((2131,), '<i8'), 'segment_vertices': ((24108, 2), '<i4'),
                    'frame_native_pts': ((F,), '<i8'), 'frame_shot_number': ((F,), 'u1'), 'path_kind': ((2130,), 'u1')}
        for name, (shape, dtype) in required.items():
            if a[name].shape != shape or a[name].dtype != np.dtype(dtype):
                raise ValueError('screen-grid array contract mismatch: ' + name)
        if not np.array_equal(a['frame_native_pts'], np.arange(575744, 2415105, 256, dtype=np.int64)):
            raise ValueError('direct grid must cover only S02--S09; S01 cannot be replaced')
        offsets = a['path_offsets']
        if offsets[0] != 0 or offsets[-1] != V or np.any(np.diff(offsets) < 2):
            raise ValueError('screen-grid topology does not preserve all original paths/vertices')
        expected = np.arange(V - 1, dtype=np.int32)
        expected = np.delete(expected, offsets[1:-1] - 1)
        if not np.array_equal(a['segment_vertices'][:, 0], expected) or not np.array_equal(a['segment_vertices'][:, 1], expected + 1):
            raise ValueError('screen-grid adjacency changed, reparameterized or implicitly closed')
        if np.any(a['path_kind'] > 1) or int(np.count_nonzero(a['path_kind'] == 0)) != 2065:
            raise ValueError('screen-grid complete grid/marking inventory changed')
        topology_file = _reference(doc['topology'], self.path.parent)
        topology = read_json(topology_file)
        self.paths = topology if isinstance(topology, list) else topology['paths']
        if len(self.paths) != 2130:
            raise ValueError('incomplete screen-grid path metadata')
        for i, path in enumerate(self.paths):
            if path['vertex_begin'] != int(offsets[i]) or path['vertex_end'] != int(offsets[i + 1]):
                raise ValueError('path metadata differs from immutable vertex topology')
            kind = ('grid', 'marking')[int(a['path_kind'][i])]
            if path['kind'] != kind or tuple(path['color_bgr']) != COLORS[kind]:
                raise ValueError('path metadata differs from original renderer kind/color')
        shots = doc['shots']
        if shots['s01']['state_kind'] != 'preserved_complete_physical_baseline':
            raise ValueError('S01 must route to its original complete physical baseline')
        self.s01_references = {}
        for name in ('model', 'complete_state_manifest'):
            ref = shots['s01'][name]
            path = _reference(ref, self.path.parent)
            self.s01_references[name] = dict(file=str(path), sha256=ref['sha256'])
        self.s01_references['row_state_files'] = {}
        for name, ref in shots['s01'].get('row_state_files', {}).items():
            path = _reference(ref, self.path.parent)
            self.s01_references['row_state_files'][name] = dict(file=str(path), sha256=ref['sha256'])
        if not {'station.f64', 'path.u32', 'initial_row.f64', 'initial_alpha.f64'}.issubset(self.s01_references['row_state_files']):
            raise ValueError('S01 complete original auxiliary references are missing')
        ref = shots['s01']['parent_complete_checkpoint']
        self.s01_references['parent_complete_checkpoint'] = dict(file=str(_reference(ref, self.path.parent)), sha256=ref['sha256'])
        for i in range(2, 10):
            if shots[f's{i:02d}']['state_kind'] != 'direct_screen_grid':
                raise ValueError('every remaining shot must explicitly route to direct UV')
        self.segment_kind = a['path_kind'][np.searchsorted(offsets[1:], a['segment_vertices'][:, 0], side='right')]

    def bind_physical_reference(self, model_file):
        model = _physical_reference(model_file)
        if file_sha256(model_file) != self.producer_physical_reference['sha256']:
            raise ValueError('initial screen-grid receiver must retain the unchanged parent physical reference, including S01')
        clock = self.manifest.get('source_clock', self.manifest.get('clock', {}))
        identity = self.manifest['source_identity']
        clock = dict(clock, time_base=([1, 12800] if clock.get('time_base') == '1/12800' else clock.get('time_base')),
                     frame_step_pts=clock.get('frame_step_pts', clock.get('native_pts_step')),
                     image_size=clock.get('image_size', [identity['width'], identity['height']]))
        if identity['id'] != 'O6j8L8TmfKo' or clock.get('native_fps', clock.get('fps')) != '50/1':
            raise ValueError('direct grid must retain the official native 50fps source identity')
        for key in ('time_base', 'frame_step_pts', 'image_size', 'source_video_sha256'):
            if clock.get(key) != model['source_clock'].get(key):
                raise ValueError('screen-grid/current physical native source mismatch: ' + key)
        for i in range(2, 10):
            scope = model['shots'][f's{i:02d}']['scope']
            sel = self.arrays['frame_shot_number'] == i
            if not np.array_equal(self.arrays['frame_native_pts'][sel], np.arange(scope['first_pts'], scope['last_pts'] + 1, 256)):
                raise ValueError('screen-grid shot identity/scope mismatch')
        return dict(file=str(Path(model_file).resolve()), sha256=file_sha256(model_file),
                    role='current_physical_reference_only', projection_equal_to_direct_grid_claimed=False,
                    serialized_schema_bounds_clock_checked=True, physical_acceptance_certified=False)

    def frame(self, native_pts):
        idx = int(np.searchsorted(self.arrays['frame_native_pts'], native_pts))
        if idx >= 7186 or int(self.arrays['frame_native_pts'][idx]) != native_pts:
            raise ValueError('no direct UV at this exposure; S01 must use the preserved baseline function')
        return {name: self.arrays[name][idx] for name in ('screen_uv', 'depth', 'vertex_mask_bits')}

    def draw(self, native_bgr, native_pts, *, labels=True):
        """Direct native-UV rasterization. No current camera or projection call."""
        import cv2
        if native_bgr.shape != (1080, 1920, 3) or native_bgr.dtype != np.uint8:
            raise ValueError('direct grid drawing requires an original native 1920x1080 BGR frame')
        frame = self.frame(int(native_pts)); uv = frame['screen_uv']; bits = frame['vertex_mask_bits']
        valid = ((bits & 3) == 3) & np.isfinite(uv).all(axis=1)
        seg = self.arrays['segment_vertices']; xy = np.asarray(uv[seg], np.float64)
        keep = valid[seg].all(axis=1)
        start, delta = xy[:, 0], xy[:, 1] - xy[:, 0]
        lo, hi = np.zeros(len(seg)), np.ones(len(seg))
        # Display-only viewport clipping; immutable raw UV/validity stay intact.
        for axis, maximum in ((0, 1919.), (1, 1079.)):
            parallel = np.abs(delta[:, axis]) < 1e-30
            keep &= ~parallel | ((start[:, axis] >= 0) & (start[:, axis] <= maximum))
            denominator = np.where(parallel, 1., delta[:, axis])
            t0, t1 = -start[:, axis] / denominator, (maximum - start[:, axis]) / denominator
            lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(t0, t1)))
            hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(t0, t1)))
        keep &= lo <= hi
        xy = np.stack((start[keep] + lo[keep, None] * delta[keep], start[keep] + hi[keep, None] * delta[keep]), axis=1)
        clipped = np.rint(xy * 256.).astype(np.int32)
        kinds = self.segment_kind[keep]; layer = native_bgr.copy()
        for code, kind in enumerate(('grid', 'marking')):
            paths = list(clipped[kinds == code].reshape(-1, 2, 1, 2))
            if paths:
                cv2.polylines(layer, paths, False, COLORS[kind], 1 if code == 0 else 2, cv2.LINE_AA, shift=8)
        image = cv2.addWeighted(layer, .78, native_bgr, .22, 0)
        if labels:
            for path in self.paths:
                if path['kind'] != 'marking' or not path.get('label'):
                    continue
                begin, end = path['vertex_begin'], path['vertex_end']
                ids = np.flatnonzero(valid[begin:end] & ((bits[begin:end] & 4) != 0))
                if len(ids):
                    point = tuple(np.rint(uv[begin + ids[len(ids) // 2]]).astype(int))
                    cv2.putText(image, path['label'], point, 0, .45, COLORS['marking'], 1, cv2.LINE_AA)
        return image, dict(native_pts=int(native_pts), total_vertices=len(uv), total_paths=2130,
                           total_segments=len(seg), finite_front_vertices=int(valid.sum()),
                           viewport_drawn_segments=int(keep.sum()), all_vertices_retained=True,
                           screen_grid_manifest_sha256=self.sha256, current_camera_projection_called=False,
                           physical_camera_CP_initialized=False, optimizer_started=False)


def initialize_screen_grid(manifest_file, model_file, packet_file, output, *, config_file):
    """Receive the exact screen state after the driver's existing source guard."""
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('immutable screen-grid initialization generation exists')
    state = ScreenGridState(manifest_file)
    model_ref = state.bind_physical_reference(model_file)
    binding = dict(schema=STATE_SCHEMA, state_kind='screen_grid',
                   manifest=dict(file=str(state.path), sha256=state.sha256),
                   complete_scope=dict(native_frames=8434, s01_preserved_frames=1248, direct_s02_s09_frames=7186),
                   s01_preserved_complete_physical_baseline=state.s01_references,
                   producer_physical_reference_only=state.producer_physical_reference,
                   current_physical_reference=model_ref,
                   source_packet=dict(file=str(Path(packet_file).resolve()), sha256=file_sha256(packet_file)),
                   configuration=dict(file=str(Path(config_file).resolve()), sha256=file_sha256(config_file)),
                   screen_grid_initialized=True, physical_camera_CP_initialized=False, optimizer_started=False,
                   direct_UV_is_source_observation=False, source_UV_W_Query_or_qualification_changed=False,
                   existing_s01_parameters_grid_and_auxiliaries_replaced=False,
                   subsequent_physical_candidate_policy=dict(new_immutable_generation_required=True,
                       same_initial_grid_pixel_comparison_required=True, overwrite_initial_screen_grid_permitted=False),
                   input_arrays_copied=False, direct_grid_vertices_per_frame=26238, direct_grid_paths=2130,
                   source_identity_guard='joint_solver.assert_source_identity_current executed before receiver',
                   physical_solver_and_RS_domain_acceptance_claimed=False)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'initial_screen_grid_state.json', binding)
    write_json(out / 'execution.json', dict(schema='rinkfit.screen_grid_initialization_execution.v1',
        returncode=0, screen_grid_initialized=True, physical_camera_CP_initialized=False, optimizer_started=False,
        initial_screen_grid_state=dict(file=str(out / 'initial_screen_grid_state.json'),
                                       sha256=file_sha256(out / 'initial_screen_grid_state.json')),
        current_physical_reference=model_ref))
    print(json.dumps(binding, ensure_ascii=False))
    return 0
