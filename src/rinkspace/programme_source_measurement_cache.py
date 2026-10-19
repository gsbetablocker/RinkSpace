"""Reuse authenticated material readouts in one ordered original RGB replay.

The cache contains pending measurements, never source identity approval.
Frame-local candidate/query indices stay frame-local when ZIP chunks are read.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ProgrammeSourceMeasurementCache:
    def __init__(self, manifest_path, *, source_sha256, time_base,
                 expected_actual_pts, follow=False):
        self.path = Path(manifest_path).resolve()
        self.source_sha256, self.time_base = source_sha256, time_base
        self.expected_pts = np.asarray(expected_actual_pts, np.int64)
        self.follow = bool(follow)
        self.chunks, self.starts, self.loaded = [], [], OrderedDict()
        self.refresh()

    def refresh(self):
        try:
            packet = json.loads(self.path.read_text(encoding='utf8'))
        except (FileNotFoundError, json.JSONDecodeError):
            if self.follow and self.chunks:
                return False  # A producer may currently be replacing its JSON.
            raise
        if (packet['source_sha256'] != self.source_sha256
                or packet['time_base'] != self.time_base):
            raise ValueError('Cached measurements belong to another original source')
        chunks = packet['chunks']
        if chunks[:len(self.chunks)] != self.chunks:
            raise ValueError('Producer changed an authenticated measurement prefix')
        if hasattr(self, 'inventory') and packet['inventory'] != self.inventory:
            raise ValueError('Source readout inventory changed while following chunks')
        self.inventory = packet['inventory']
        self.packet = packet
        self.chunks = chunks
        self.starts = [int(c['first_actual_pts']) for c in chunks]
        return True

    def _load_chunk(self, number):
        if number in self.loaded:
            self.loaded.move_to_end(number)
            return self.loaded[number]
        reference = self.chunks[number]
        path = Path(reference['path'])
        if _sha(path) != reference['sha256']:
            raise ValueError('Saved source measurement chunk changed')
        with np.load(path, allow_pickle=False) as saved:
            # Only actual readout arrays are needed; no cached masks/images or
            # past auto-selected identities are consumed as approval.
            names = ('actual_pts', 'native_pts', 'BGR_sha256', 'RGB_sha256', 'shot_id',
                'query_offsets', 'query_curve', 'query_uv', 'query_normal',
                'query_interval', 'query_status', 'candidate_offsets',
                'candidate_uv', 'candidate_sides', 'candidate_query',
                'candidate_flags', 'curve_status')
            arrays = {name: saved[name] for name in names}
            if 'candidate_side_quantiles_uv' in saved.files:
                arrays['candidate_side_quantiles_uv'] = saved['candidate_side_quantiles_uv']
        pts = arrays['actual_pts']
        start = int(np.searchsorted(self.expected_pts, int(pts[0])))
        if (not np.array_equal(pts, self.expected_pts[start:start+len(pts)])
                or not np.array_equal(pts, arrays['native_pts'])
                or int(pts[0]) != reference['first_actual_pts']
                or int(pts[-1]) != reference['last_actual_pts']):
            raise ValueError('Saved measurement source exposure clock differs')
        for prefix in ('query', 'candidate'):
            offsets = arrays[prefix+'_offsets']
            if (len(offsets) != len(pts)+1 or offsets[0] != 0
                    or np.any(np.diff(offsets)<0)
                    or offsets[-1] != len(arrays[prefix+'_uv'])):
                raise ValueError('Saved source readout offsets differ')
        self.loaded[number] = arrays
        while len(self.loaded) > 2:
            self.loaded.popitem(last=False)
        return arrays

    def frame(self, actual_pts, *, source_binding=None, wait_callback=None):
        pts = int(actual_pts)
        if pts not in self.expected_pts:
            raise ValueError('Undeclared source exposure requested')
        announced = False
        while True:
            number = int(np.searchsorted(self.starts, pts, side='right')-1)
            if number >= 0 and pts <= int(self.chunks[number]['last_actual_pts']):
                break
            if (not self.follow or self.packet.get('all_actual_exposures_processed') is True
                    or self.packet.get('actual_source_profiles_ready') is True):
                raise ValueError(f'Original source measurement not yet available: {pts}')
            if wait_callback and not announced:
                wait_callback(pts)
                announced = True
            # Hold the current original RGB frame; do not decode another frame
            # or seek while its producer readout is not published yet.
            time.sleep(2.)
            self.refresh()
        arrays = self._load_chunk(number)
        index = int(np.searchsorted(arrays['actual_pts'], pts))
        if index == len(arrays['actual_pts']) or int(arrays['actual_pts'][index]) != pts:
            raise ValueError('Cache omitted an original source exposure')
        binding = dict(source_sha256=self.source_sha256, time_base=self.time_base,
            source_path=self.packet['source_path'], source_pts=int(arrays['native_pts'][index]),
            native_pts=int(arrays['native_pts'][index]), actual_pts=pts,
            shot_id=str(arrays['shot_id'][index]),
            decoded_native_BGR_sha256=str(arrays['BGR_sha256'][index]),
            source_resolution=[1920,1080], native_hw=[1080,1920],
            image_domain='native_brown_pixels')
        if source_binding is not None:
            for key in ('source_sha256','time_base','native_pts','actual_pts',
                        'shot_id','decoded_native_BGR_sha256'):
                if binding[key] != source_binding[key]:
                    raise ValueError(f'Cached and replayed exact source differ: {key}')
        result = dict(source_binding=binding,
            decoded_native_RGB_sha256=str(arrays['RGB_sha256'][index]),
            curve_status=arrays['curve_status'][index].copy(),
            source_readout_quantization_step_px=1., source_interpolation_grid_px=1./32.,
            source_RGB_measured=True, fitting_admissible=False,
            source_cache_reference=dict(**self.chunks[number],frame_index=index))
        for prefix, keys in (('query',('curve','uv','normal','interval','status')),
                             ('candidate',('uv','sides','query','flags'))):
            lo, hi = arrays[prefix+'_offsets'][index:index+2]
            for key in keys:
                result[prefix+'_'+key] = arrays[prefix+'_'+key][lo:hi].copy()
        lo, hi = arrays['candidate_offsets'][index:index+2]
        if 'candidate_side_quantiles_uv' in arrays:
            result['candidate_side_quantiles_uv'] = arrays['candidate_side_quantiles_uv'][lo:hi].copy()
        query = result['candidate_query']
        if np.any(query<0) or np.any(query>=len(result['query_uv'])):
            raise ValueError('Candidate indices ceased to be frame-local')
        from .feedback_withdrawal import measurement_withdrawal_applicability
        result['human_withdrawal_applicability']=measurement_withdrawal_applicability(result)
        return result


def recheck_source_candidate_exclusions(measurement, exclusion_mask):
    """Re-read physical exclusions; old yellow-band flags cannot veto good UV.

    Material truncation/competition/identity-pending bits are preserved. This
    clears only the old scope bit, then measures the complete source context
    at the center and both actual material sides against this frame's mask.
    """
    mask = np.asarray(exclusion_mask, bool)
    centers = np.asarray(measurement['candidate_uv'],float)
    sides = np.asarray(measurement['candidate_sides'],float)
    normals = np.asarray(measurement['query_normal'],float)[measurement['candidate_query']]
    points = np.concatenate((centers[:,None],sides),axis=1)
    flags = np.asarray(measurement['candidate_flags'],np.uint8).copy() & np.uint8(253)
    directions = np.c_[np.repeat(np.arange(-3,4),17),np.tile(np.arange(-8,9),7)]
    # Bound working memory, not coverage: every candidate is visited.
    for first in range(0,len(points),512):
        last = min(first+512,len(points)); normal = normals[first:last]
        tangent = np.c_[-normal[:,1],normal[:,0]]
        context = (points[first:last,:,None]
            +tangent[:,None,None]*directions[None,None,:,0,None]
            +normal[:,None,None]*directions[None,None,:,1,None])
        finite = np.isfinite(context).all(axis=(1,2,3))
        pixels = np.rint(np.nan_to_num(context,nan=-1.,posinf=-1.,neginf=-1.)).astype(np.int64)
        inside = (pixels>=0).all(-1) & (pixels<[mask.shape[1],mask.shape[0]]).all(-1)
        excluded = (~finite | ~inside.all(axis=(1,2))
            | mask[np.clip(pixels[...,1],0,mask.shape[0]-1),
                   np.clip(pixels[...,0],0,mask.shape[1]-1)].any(axis=(1,2)))
        flags[first:last] |= excluded.astype(np.uint8)*2
    measurement['candidate_flags'] = flags
    measurement['current_exact_source_scope_rechecked'] = True
    return measurement


def supplement_selected_profile_quantiles(rgb, measurement, indices, inventory, *, backend='cuda'):
    """Reread only selected current source profiles whose old cache lacks blur.

    Exact float32 readouts must agree with the earlier cache; no proximity or
    model prediction substitutes for the original profile's material sides.
    """
    from .vision_source_shapes import _MaterialProfileFrame
    from .vision_programme_ice_review import _actual_profile_candidates, _batched_actual_profiles
    ids = np.unique(np.asarray(indices,np.int64))
    quantiles = measurement.setdefault('candidate_side_quantiles_uv',
        np.full((len(measurement['candidate_uv']),2,3,2),np.nan))
    ids = ids[~np.isfinite(quantiles[ids]).all(axis=(1,2,3))]
    if not len(ids):return 0
    queries = np.unique(measurement['candidate_query'][ids])
    roles = [inventory[int(measurement['query_curve'][q])] for q in queries]
    frame = _MaterialProfileFrame(rgb, measurement['source_binding'],
        expected_digest=measurement['decoded_native_RGB_sha256'])
    sampler = _batched_actual_profiles
    if backend=='cuda':
        from .vision_source_cuda_profiles import batched_actual_profiles_cuda
        sampler=batched_actual_profiles_cuda
    offsets,profiles,smooth,valid=sampler(frame,measurement['query_uv'][queries],
        measurement['query_normal'][queries],measurement['query_interval'][queries],
        [r['material_signal'] for r in roles])
    filled=0
    for local,(query,role) in enumerate(zip(queries,roles)):
        readings=_actual_profile_candidates(measurement['query_uv'][query],
            measurement['query_normal'][query],offsets[local,valid[local]],
            profiles[local,valid[local]],smooth[local,valid[local]],
            operator=role['operator'],signal=role['material_signal'],transition=role.get('material_transition'),
            with_quantiles=True)
        for candidate in ids[measurement['candidate_query'][ids]==query]:
            exact=[r for r in readings if np.array_equal(np.asarray(r[0],np.float32),measurement['candidate_uv'][candidate])
                and np.array_equal(np.asarray(r[1],np.float32),measurement['candidate_sides'][candidate])]
            if len(exact)==1:
                quantiles[candidate]=exact[0][3];filled+=1
    return filled


def restore_exact_navigation_queries(measurement, reader, camera25):
    """Recover original profile query precision without reading new profiles.

    The cache packs query windows as float32. Reproducing their original
    float64 coordinates is necessary to reread exactly the same source
    profiles; camera coordinates remain search locations, never output UV.
    """
    from calibration.program_tracking.projection import project_fast
    from .vision_programme_ice_review import _source_queries
    curves,points,normals,intervals=[],[],[],[]
    for number,path in enumerate(reader.paths):
        projected,depth=project_fast(camera25,path['xyz'])
        search=dict(path)
        if 'center_xyz' in path:
            search['_projected_center']=project_fast(camera25,path['center_xyz'][None])[0][0]
        uv,normal,interval=_source_queries(search,projected,depth,(1080,1920),reader.spacing,reader.radius)
        points.extend(uv);normals.extend(normal);intervals.extend(interval);curves.extend([number]*len(uv))
    guided=measurement.get('source_guided_queries')
    for name,value in (('query_curve',np.asarray(curves,np.int16)),
                       ('query_uv',np.asarray(points,float).reshape(-1,2)),
                       ('query_normal',np.asarray(normals,float).reshape(-1,2)),
                       ('query_interval',np.asarray(intervals,float).reshape(-1,2))):
        saved=np.asarray(measurement[name])
        prefix=saved
        if guided is not None:
            if (guided.get('schema')!='rinkspace.original-source-guided-query-overlay.v1'
                    or guided['original_query_prefix_length']!=len(value)):
                raise ValueError('Guided source overlay changed its original frozen-query prefix')
            extra=np.asarray(guided[name],dtype=value.dtype).reshape((-1,*value.shape[1:]))
            prefix=saved[:len(value)]
            if not np.array_equal(saved[len(value):],extra):
                raise ValueError('Appended actual source-guide search stations changed')
        if not np.array_equal(value.astype(prefix.dtype),prefix):
            raise ValueError('Frozen navigation no longer reproduces original source search windows')
        measurement[name]=value if guided is None else np.concatenate([value,extra])
    return measurement
