"""Read archived source fields without constructing cameras or a solver graph.

Only NumPy's numeric reconstruction and ordinary containers run. Archived
project classes/functions become inert state holders. These holders must
never be passed to a solver; this reader is for original observation review.
"""
from __future__ import annotations

import collections
import copyreg
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ArchivedSourceState:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self._archive_constructor_arguments = args
        # The legacy geometry restore functions take (class, state).
        if len(args) == 2 and isinstance(args[1], dict):
            self.__dict__.update(args[1])

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2:
            for value in state:
                if isinstance(value, dict):
                    self.__dict__.update(value)
        else:
            self._archive_state = state


class SourceReviewUnpickler(pickle.Unpickler):
    """An intentionally inert source reader, not the production graph loader."""
    def __init__(self, stream):
        super().__init__(stream)
        self._holders = {}

    def find_class(self, module, name):
        known = {
            ('numpy', 'dtype'): np.dtype,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy._core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('numpy.core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('collections', 'OrderedDict'): collections.OrderedDict,
            ('collections', 'defaultdict'): collections.defaultdict,
            ('types', 'SimpleNamespace'): SimpleNamespace,
            ('pathlib', 'Path'): Path,
            ('pathlib', 'WindowsPath'): Path,
            ('pathlib', 'PosixPath'): Path,
            ('copyreg', '_reconstructor'): copyreg._reconstructor,
            ('builtins', 'object'): object,
            ('builtins', 'set'): set,
            ('builtins', 'frozenset'): frozenset,
        }
        if (module, name) in known:
            return known[(module, name)]
        if module.startswith(('numpy', 'scipy', 'builtins', 'os', 'sys', 'subprocess')):
            raise pickle.UnpicklingError('Unsupported archived runtime global: ' + module + '.' + name)
        key = (module, name)
        if key not in self._holders:
            self._holders[key] = type(name, (ArchivedSourceState,),
                dict(_archive_original_type=module + '.' + name))
        return self._holders[key]


def read_source_state_archive(path):
    with Path(path).open('rb') as stream:
        return SourceReviewUnpickler(stream).load()


def archived_motion_blocks(path):
    """Yield only already-compacted immutable endpoint fields and original maps."""
    payload = read_source_state_archive(path)
    if payload.get('schema') != 'rinkspace.final-original-source-state-skeleton.v1':
        raise ValueError('Original source-state skeleton archive required')
    state = payload['static_locals']['state']
    for block in state.blocks:
        bundle = block.bundle
        n = len(bundle.a)
        active_to_original = np.asarray(getattr(bundle,
            'original_motion_compaction_active_to_old', np.arange(n)), np.int64)
        if active_to_original.shape != (n,):
            raise ValueError('Archived original motion index mapping differs')
        yield dict(block=block.name, a=bundle.a, b=bundle.b,
            ia=bundle.ia, ib=bundle.ib, stamps=np.asarray(bundle.stamps, np.int64),
            sigma=np.asarray(bundle.motion_sigma, float).reshape(-1, 2),
            active_to_original=active_to_original)


def write_strict_archived_motion_inputs(archive_path, review_path, output, *, input_manifest):
    """Prepare exact compact FIT inputs without any source/J/solver execution."""
    from calibration.program_tracking.strict_model_scope import (
        ExactSourceScopeReview, file_sha, write_motion_partition,
    )
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    receipt_path = archive_path.with_name('source_state_skeleton_receipt.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    actual_hash = file_sha(archive_path)
    if actual_hash != receipt['pickle_sha256']:
        raise ValueError('Original source-state archive bytes changed')
    input_manifest = Path(input_manifest).resolve()
    manifest = json.loads(input_manifest.read_text(encoding='utf8'))
    expected_counts = dict(zip((r['block'] for r in manifest['original_motion_input_compaction']['blocks']),
                               manifest['original_motion_edge_counts']))
    review = ExactSourceScopeReview.load(review_path)
    references = [dict(path=str(archive_path), sha256=actual_hash),
        dict(path=str(receipt_path), sha256=file_sha(receipt_path)),
        dict(path=str(input_manifest), sha256=file_sha(input_manifest))]
    partitions = []
    for block in archived_motion_blocks(archive_path):
        if len(block['a']) != expected_counts[block['block']]:
            raise ValueError('Current actual compact source point inventory changed')
        name = block.pop('block')
        partitions.append(write_motion_partition(output / (name + '_strict_motion.npz'),
            block=name, review=review, original_references=references, **block))
    if {r['block'] for r in partitions} != set(expected_counts):
        raise ValueError('Strict source admission omitted an original programme block')
    binding = dict(schema='rinkspace.strict-model-scope-original-motion-inputs.v1',
        source=review.source, review=review.reference, original_references=references,
        blocks=partitions, unknown_or_unreviewed_default='excluded',
        all_frames_visually_reviewed=False, solver_started=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'strict_motion_inputs.json'
    if target.exists():
        raise ValueError('Do not overwrite a previous strict source partition binding')
    target.write_text(json.dumps(binding, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    return dict(path=str(target), sha256=file_sha(target), blocks=partitions)


def _frame_uv_indices(uv, endpoint_indices, stamps, frozen, frozen_lookup):
    """Locate exact old motion endpoints in the existing frozen UV pool."""
    result = np.full(len(uv), -1, np.int64)
    dtype = np.dtype([('x', '<f8'), ('y', '<f8')])
    order = np.argsort(endpoint_indices, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(endpoint_indices[order])) + 1, len(order)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        selected = order[lo:hi]
        if not len(selected):
            continue
        stamp = int(stamps[int(endpoint_indices[selected[0]])])
        frame = frozen_lookup.get(stamp)
        if frame is None:
            continue
        start, stop = map(int, frozen['offsets'][frame:frame + 2])
        values = np.ascontiguousarray(frozen['uv'][start:stop]).view(dtype).reshape(-1)
        query = np.ascontiguousarray(uv[selected], dtype=np.float64).view(dtype).reshape(-1)
        positions = np.searchsorted(values, query)
        valid = positions < len(values)
        valid[valid] &= values[positions[valid]] == query[valid]
        result[selected[valid]] = start + positions[valid]
    return result


def write_additional_source_review_inventory(archive_path, *, input_manifest, frozen_current_pixels, output):
    """Enumerate the other original pixel families without factor construction."""
    from calibration.program_tracking.strict_model_scope import file_sha
    manifest_path, output = Path(input_manifest).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    payload = read_source_state_archive(archive_path)
    state = payload['static_locals']['state']
    aliases = {int(k):int(v) for k,v in state.original.aliases.items()}
    with np.load(frozen_current_pixels,allow_pickle=False) as z:
        frozen={k:z[k].copy() for k in ('actual_pts','offsets','uv')}
        receipts=json.loads(str(z['metadata_json'].item()))['family_receipts']
    lookup={int(p):i for i,p in enumerate(frozen['actual_pts'])}
    rows=[]; chunks=[]; point_pts=[]; row_ids=[]; point_ids=[]; pool_indices=[]
    references=[]
    def add(family,row,uv,pts,locator,*,endpoint=None):
        values=np.asarray(uv,np.float64).reshape(-1,2)
        actual=aliases.get(int(pts),int(pts))
        if not np.isfinite(values).all():raise ValueError('Original native UV is nonfinite')
        header={k:row[k] for k in ('id','feature_id','split','role','kind','measurement_role',
            'curve_observation_kind','point_correspondence','endpoint_observation','physical_identity_status')
            if k in row and isinstance(row[k],(str,int,bool,float,type(None)))}
        curve=row.get('curve')
        if isinstance(curve,dict):header['curve']={k:v for k,v in curve.items() if isinstance(v,(str,int,bool,float,type(None)))}
        header.update(family=family,actual_pts=actual,native_pts=int(pts),original_locator=locator,
            endpoint=endpoint,point_count=len(values),scope_status='requires_current_source_review')
        ri=len(rows);rows.append(header);chunks.append(values)
        point_pts.extend([actual]*len(values));row_ids.extend([ri]*len(values));point_ids.extend(range(len(values)))
        pool_indices.extend(_frame_uv_indices(values,np.zeros(len(values),int),np.array([actual]),frozen,lookup).tolist())
    simple={
        'additional_real_native_ice_curves':'native_ice_factor',
        'additional_real_native_board_edges':'native_board_factor',
        'additional_real_native_board_material_points':'native_board_material_point_factor',
        'additional_real_native_board_circles':'circle_factor'}
    native_rows=[]
    for key,family in simple.items():
        contract=manifest.get(key)
        if contract is None:continue
        ref=dict(path=contract['packet_path'],sha256=contract['packet_sha256'])
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Original native source packet changed')
        references.append(ref);packet=json.loads(Path(ref['path']).read_text(encoding='utf8'))
        for index,row in enumerate(packet['rows']):
            if 'uv' not in row:continue
            add(family,row,row['uv'],row.get('pts',row.get('actual_pts')),dict(packet=ref,row_index=index))
            native_rows.append((family,index,row,ref))
    contract=manifest['additional_native_static_board_tracks'];ref=dict(path=contract['packet_path'],sha256=contract['packet_sha256'])
    references.append(ref);packet=json.loads(Path(ref['path']).read_text(encoding='utf8'))
    for index,row in enumerate(packet['rows']):
        for field,clock in (('uv0','source_pts'),('uv1','target_pts')):
            add('native_board_track_factor',row,row[field],row[clock],dict(packet=ref,row_index=index),endpoint=field)
    for index,row in enumerate(packet.get('source_anchor_observations',[])):
        add('native_board_track_factor',row,row['uv'],row.get('native_pts',row['pts']),
            dict(packet=ref,collection='source_anchor_observations',row_index=index),endpoint='anchor')
    contract=manifest['additional_real_native_persistent_ice_long_baseline'];ref=dict(path=contract['packet_path'],sha256=contract['packet_sha256'])
    references.append(ref);packet=json.loads(Path(ref['path']).read_text(encoding='utf8'))
    for index,row in enumerate(packet['new_rows']):
        for endpoint in ('anchor','target'):
            value=row[endpoint]
            add('persistent_ice_factor',row,value['uv'],value['source_native_pts'],dict(packet=ref,row_index=index),endpoint=endpoint)
    patterns=getattr(state,'patterns',None)
    if patterns is not None:
        for split in ('fit','held'):
            for fi,frame in enumerate(getattr(patterns,split)):
                key='source_curve_support' if frame['physical_ad_id'] in (13,26) else 'source_pattern_points'
                for ri,row in enumerate(frame[key]):
                    add('shared_board_patterns',dict(row,split=split,feature_id='Ad'+str(frame['physical_ad_id'])),
                        row['uv'],frame['source_actual_pts'],dict(source_archive=str(archive_path),split=split,frame_index=fi,row_index=ri))
    external=getattr(state,'external',None)
    if external is not None:
        for index,row in enumerate(external.track_rows):
            add('external_main_low',row,row['target_uv_native'],row['actual_pts'],dict(source_archive=str(archive_path),collection='track_rows',row_index=index))
        for index,row in enumerate(external.track_sources):
            add('external_main_low',dict(row,split='fit'),row['anchor_uv_native'],row['anchor_actual_pts'],dict(source_archive=str(archive_path),collection='track_sources',row_index=index))
    authority=manifest.get('highest_direct_human_hard_truth',{}).get('native',{}).get('authority')
    if authority is not None:
        references.append(authority);packet=json.loads(Path(authority['path']).read_text(encoding='utf8'))
        manual_ids={r['id'] for r in packet['records']}
        for family,index,row,ref in native_rows:
            if row['id'] in manual_ids:
                add('direct_manual_centre_secondary',row,row['uv'],row.get('pts',row.get('actual_pts')),
                    dict(packet=ref,row_index=index,alias_of_family=family,manual_authority=authority))
    uv=np.concatenate(chunks) if chunks else np.empty((0,2));pts=np.asarray(point_pts,np.int64);order=np.argsort(pts,kind='stable')
    output.mkdir(parents=True,exist_ok=True);target=output/'additional_absolute_source_review_inventory.npz'
    if target.exists():raise ValueError('Do not overwrite an original pixel family inventory')
    np.savez_compressed(target,actual_pts=pts[order],uv=uv[order],row_index=np.asarray(row_ids,np.int32)[order],
        original_point_index=np.asarray(point_ids,np.int32)[order],current_UV_pool_index=np.asarray(pool_indices,np.int64)[order],
        rows_json=json.dumps(rows,ensure_ascii=False,separators=(',',':')),
        metadata_json=json.dumps(dict(schema='rinkspace.original-additional-pixel-source-review-inventory.v1',
            original_references=references,original_pixel_family_receipts=receipts,
            reviewed=False,wide_source_extracted=False,unknown_or_unreviewed_not_admitted=True),separators=(',',':')))
    # The original programme inventory also contains deliberate nonpixel
    # placeholders and one historical retired point. Expose the exact mask.
    core_path=output/'original_absolute_source_review_inventory.npz'
    with np.load(core_path,allow_pickle=False) as z:
        core_rows=json.loads(str(z['rows_json'].item()));core_row_ids=z['row_index'].copy();core_point_ids=z['original_point_index'].copy()
    retired=set(manifest['original_source_role_retirement'].get('active_runtime_global_scalar_slots',[]))
    masks={};cursor=0
    compactblocks={b['block']:b for b in manifest['original_absolute_input_compaction']['blocks']}
    maps={(b['block'],r['group_order'],r['row_order']):r for b in manifest['original_absolute_input_compaction']['blocks'] for r in b['row_maps']}
    for block in state.blocks:
        cursor=compactblocks[block.name]['compact_start']+2*len(block.bundle.a)
        for gi,(stamp,group) in enumerate(block.groups):
            for ri,row in enumerate(group):
                ids=np.arange(len(row['uv']));mapping=maps.get((block.name,gi,ri))
                if mapping is not None:ids=np.asarray(mapping['kept_original_point_indices'])
                nonpixel=(row.get('source_uv_is_image_boundary_context_only') is True or row.get('no_visible_tangent_UV_created') is True or row.get('curve_observation_kind') in ('horizontal_extent','clipped_horizontal_extent'))
                for ci,pi in enumerate(ids):masks[(block.name,gi,ri,int(pi))]=not nonpixel and not {cursor+2*ci,cursor+2*ci+1}.issubset(retired)
                cursor+=2*len(ids)
    mask=[]
    for ri,pi in zip(core_row_ids,core_point_ids):
        r=core_rows[int(ri)];mask.append(masks.get((r['block'],r['group_order'],r['row_order'],int(pi)),False))
    mask=np.asarray(mask,bool)
    maskpath=output/'original_absolute_current_pixel_mask.npz'
    np.savez_compressed(maskpath,current_fit_pixel=mask,original_inventory_reference_json=json.dumps(dict(path=str(core_path),sha256=file_sha(core_path))))
    return dict(path=str(target),sha256=file_sha(target),points=len(uv),rows=len(rows),
        core_current_pixel_mask=dict(path=str(maskpath),sha256=file_sha(maskpath),current_pixels=int(mask.sum())))


def write_source_review_inventory(archive_path, *, input_manifest, frozen_current_pixels, output):
    """Expose old absolute identities and motion correspondences for one stream.

    The original UV pool is reused for motion endpoints. Only small missing
    historical endpoint exceptions are stored here. No review or admission is
    inferred from this inventory.
    """
    from calibration.program_tracking.strict_model_scope import file_sha
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    manifest = json.loads(Path(input_manifest).read_text(encoding='utf8'))
    current = Path(frozen_current_pixels).resolve()
    with np.load(current, allow_pickle=False) as z:
        frozen = {key: z[key].copy() for key in ('actual_pts', 'offsets', 'uv')}
    lookup = {int(p): i for i, p in enumerate(frozen['actual_pts'])}
    references = [dict(path=str(p.resolve()), sha256=file_sha(p)) for p in
                  (archive_path, Path(input_manifest), current)]
    payload = read_source_state_archive(archive_path)
    state = payload['static_locals']['state']
    absolute_maps = {(b['block'], r['group_order'], r['row_order']): r
        for b in manifest['original_absolute_input_compaction']['blocks'] for r in b['row_maps']}
    rows, absolute_uv, absolute_pts, absolute_row, absolute_point = [], [], [], [], []
    output.mkdir(parents=True, exist_ok=True)
    motion = []
    for block in state.blocks:
        bundle = block.bundle
        n = len(bundle.a)
        stamps = np.asarray(bundle.stamps, np.int64)
        a_indices = _frame_uv_indices(bundle.a, bundle.ia, stamps, frozen, lookup)
        b_indices = _frame_uv_indices(bundle.b, bundle.ib, stamps, frozen, lookup)
        missing_a, missing_b = np.flatnonzero(a_indices < 0), np.flatnonzero(b_indices < 0)
        motion_path = output / (block.name + '_motion_review_index.npz')
        if motion_path.exists():
            raise ValueError('Do not overwrite a previous original-source review index')
        np.savez_compressed(motion_path, source_pts=stamps[bundle.ia], target_pts=stamps[bundle.ib],
            source_uv_pool_index=a_indices, target_uv_pool_index=b_indices,
            active_to_original=np.asarray(getattr(bundle,'original_motion_compaction_active_to_old',np.arange(n)),np.int64),
            missing_source_active_indices=missing_a, missing_source_uv=bundle.a[missing_a],
            missing_target_active_indices=missing_b, missing_target_uv=bundle.b[missing_b],
            metadata_json=json.dumps(dict(block=block.name, current_frozen_UV_reference=references[2],
                original_source_references=references, reviewed=False),separators=(',', ':')))
        motion.append(dict(block=block.name,path=str(motion_path),sha256=file_sha(motion_path),points=n))
        for gi, (pts, groups) in enumerate(block.groups):
            for ri, row in enumerate(groups):
                uv = np.asarray(row['uv'],np.float64).reshape(-1,2)
                indices = np.arange(len(uv),dtype=np.int64)
                corrected = row.get('active_source_pixel_correction', {})
                original_points = np.asarray(corrected.get('point_indices',indices),np.int64)
                if len(original_points)!=len(uv):
                    original_points=indices
                compact=absolute_maps.get((block.name,gi,ri))
                if compact is not None:
                    indices=np.asarray(compact['kept_original_point_indices'],np.int64)
                header={key:row.get(key) for key in ('id','curve_id','split','origin','method',
                    'measurement_role','curve_observation_kind','geometry_identity','physical_edge_role',
                    'point_correspondence','operator','source_native_pts') if row.get(key) is not None}
                header.update(family='programme:'+block.name+':absolute',block=block.name,
                    group_order=gi,row_order=ri,actual_pts=int(pts),
                    feature_id=str(row.get('curve_id',row.get('id','unknown'))),
                    point_count=len(indices),scope_status='requires_current_source_review')
                # Preserve source identity certificates as references, not the
                # enormous historical RGB/polyline/world arrays embedded in them.
                for key in ('world_scene_binding','geometry_identity','material_identity'):
                    value=row.get(key)
                    if isinstance(value,(str,list)):
                        header[key]=value
                    elif isinstance(value,dict):
                        header[key]={k:v for k,v in value.items() if isinstance(v,(str,bool,int,float,type(None)))}
                row_index=len(rows);rows.append(header)
                absolute_uv.append(uv[indices]);absolute_pts.extend([int(pts)]*len(indices))
                absolute_row.extend([row_index]*len(indices));absolute_point.extend(original_points[indices].tolist())
    flat_uv=np.concatenate(absolute_uv) if absolute_uv else np.empty((0,2))
    pts=np.asarray(absolute_pts,np.int64);sort=np.argsort(pts,kind='stable')
    exposure_pts=frozen['actual_pts'];counts=np.asarray([np.count_nonzero(pts==p) for p in exposure_pts],np.int64)
    absolute_path=output/'original_absolute_source_review_inventory.npz'
    if absolute_path.exists():
        raise ValueError('Do not overwrite a previous original absolute review inventory')
    np.savez_compressed(absolute_path,actual_pts=pts[sort],uv=flat_uv[sort],
        row_index=np.asarray(absolute_row,np.int32)[sort],original_point_index=np.asarray(absolute_point,np.int32)[sort],
        exposure_pts=exposure_pts,exposure_offsets=np.r_[0,np.cumsum(counts)],
        rows_json=json.dumps(rows,ensure_ascii=False,separators=(',',':')),
        metadata_json=json.dumps(dict(schema='rinkspace.original-source-absolute-review-inventory.v1',
            original_references=references,reviewed=False,programme_original_families_complete=True,
            additional_native_pixel_families_complete=False),separators=(',',':')))
    result=dict(absolute=dict(path=str(absolute_path),sha256=file_sha(absolute_path),
        points=len(flat_uv),rows=len(rows)),motion=motion,current_frozen_UV_reference=references[2])
    (output/'source_review_inventory.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    return result
