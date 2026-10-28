"""Continuous native RGB verification through the existing trained source reader."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from ..path_policy import guard_project_write
from ..vision_native import accepted_packet_guides, file_ref


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    from ..vision import _source_json
    path = guard_project_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_source_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _original_query_segments(row, indices):
    """Read saved connectivity; older material rows may store it per measured run."""
    count=int(row['metadata']['source_point_count'])
    segments=row['guide'].get('segment_ids')
    if segments is not None:
        segments=np.asarray(segments)
        if segments.shape!=(count,) or not np.issubdtype(segments.dtype,np.integer):
            raise ValueError('Original guide segment IDs do not match its source queries')
        return segments[indices].tolist()
    known={}
    def remember(query,segment):
        query,segment=int(query),int(segment)
        if query<0 or query>=count or (query in known and known[query]!=segment):
            raise ValueError('Original source segment records conflict')
        known[query]=segment
    saved=row.get('segment_ids')
    if saved is not None:
        if len(saved)!=len(row['source_indices']):raise ValueError('Original measured segment IDs lost their query indices')
        for query,segment in zip(row['source_indices'],saved):remember(query,segment)
    for run in row.get('measured_runs',()):
        for query in run['source_indices']:remember(query,run['segment_id'])
    if any(index not in known for index in indices):
        raise ValueError('Original guide segment IDs are missing; cannot invent delayed-query connectivity')
    if len(known)==count:row['guide']['segment_ids']=[known[i] for i in range(count)]
    row['metadata']['guide_segments_recovered_from_original_readouts']=True
    return [known[index] for index in indices]


def apply_delayed_source_readouts(record, revisions):
    """Apply graph evidence to its original automatic frame; never to today's UV."""
    from copy import deepcopy
    from ..vision_geometry import _same_frame
    result=deepcopy(record)
    binding=result['source_binding']
    for revision in revisions:
        _same_frame(binding,revision['source_binding'])
        evidence=revision['evidence_source_binding']
        if (any(evidence[k]!=binding[k] for k in ('source_sha256','time_base','shot_id'))
                or int(evidence['source_pts'])<=int(binding['source_pts'])):
            raise ValueError('Delayed readout must be supported by a later exposure of this shot')
        if revision['native_scene']!=result['native_geometry']['scene']:
            raise ValueError('Delayed readout uses another native geometry')
        if revision['decoded_native_RGB_sha256']!=binding['decoded_native_rgb_sha256']:
            raise ValueError('Delayed readout uses another original RGB')
        row=result['guided']['readings'][revision['guide_index']]
        if (row['expected_id']!=revision['expected_id'] or row['native_path_id']!=revision['native_path_id']):
            raise ValueError('Delayed readout changed its original native identity')
        current=dict(zip(map(int,row['source_indices']),row['actual_source_uv']))
        indices=list(map(int,revision['source_indices']));points=np.asarray(revision['actual_source_uv'],float)
        count=row['metadata']['source_point_count']
        if (points.shape!=(len(indices),2) or not np.isfinite(points).all()
                or len(set(indices))!=len(indices) or any(i<0 or i>=count for i in indices)):
            raise ValueError('Delayed readout lost original query indices')
        current.update(zip(indices,points.tolist()));order=sorted(current)
        uv=[current[i] for i in order];segments=_original_query_segments(row,order)
        runs=[];start=0
        for end in range(1,len(order)+1):
            if end==len(order) or order[end]!=order[end-1]+1 or segments[end]!=segments[end-1]:
                runs.append(dict(actual_source_uv=uv[start:end],source_indices=order[start:end],segment_id=segments[start]))
                start=end
        quality=[i in current for i in range(count)]
        row.update(actual_source_uv=uv,source_indices=order,coherent_source_uv=uv,coherent_source_indices=order,
            segment_ids=segments,measured_runs=runs,coherent_measured_runs=runs,source_quality_mask=quality,
            source_quality_weights=list(map(float,quality)),source_peak_candidates=None,shape_characterization=None)
        row['metadata']['actual_measured_count']=len(order)
        row.setdefault('delayed_source_identity_evidence',[]).append(revision)
    result['automatic_readout_revised_from_later_actual_RGB']=True
    result['original_accepted_UV_rewritten']=False
    return result


def publish_delayed_source_readouts(result, reports, output, video):
    """Publish derived revisions while preserving the original frame files."""
    import av
    from ..vision import _source_json
    from ..vision_storage import hydrate_source_record_arrays,pack_source_record_arrays
    grouped={}
    for revision in (result.get('native_material') or {}).get('delayed_source_readouts',[]):
        b=revision['source_binding'];grouped.setdefault((b['shot_id'],int(b['source_pts'])),[]).append(revision)
    for (shot,pts),revisions in grouped.items():
        report,=[r for r in reports if r['shot_id']==shot and r['source_pts']==pts]
        original=output/report.get('effective_result',report['result'])
        record=hydrate_source_record_arrays(load(original),output)
        updated=apply_delayed_source_readouts(record,revisions)
        name=f"{shot}_{pts}_revised_at_{result['source_binding']['source_pts']}"
        compact=pack_source_record_arrays(updated,output/(name+'_arrays.npz'),minimum_array_bytes=16384,storage_layout='byte_pool')
        write(output/(name+'.json'),_source_json(compact))
        report['initial_result']=report['result'];report['effective_result']=name+'.json'
        report.setdefault('delayed_evidence_pts',[]).append(int(result['source_binding']['source_pts']))
        for item,row in zip(report['readings'],updated['guided']['readings']):
            item['read_count']=len(row.get('coherent_source_uv',row['actual_source_uv']))
        with av.open(str(video)) as container:
            stream=container.streams.video[0];container.seek(pts,stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts<pts:continue
                if frame.pts!=pts:raise ValueError('Original revision RGB missing')
                rgb=frame.to_ndarray(format='rgb24')
                if hashlib.sha256(rgb.tobytes()).hexdigest()!=report['decoded_native_RGB_sha256']:
                    raise ValueError('Original revision RGB changed')
                image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
                for row in updated['guided']['readings']:
                    for u,v in row.get('coherent_source_uv',row['actual_source_uv']):
                        draw.ellipse((u-2,v-2,u+2,v+2),fill=(255,180,0))
                image.save(output/(name+'.png'))
                report.setdefault('initial_image',report['image']);report['image']=name+'.png'
                break
            else:raise ValueError('Original revision source ended early')


def _validated_sequences(request, scope):
    """Validate native intervals before loading models or writing frame results."""
    step = scope['source_clock']['native_pts_step']
    if type(step) is not int or step <= 0:
        raise ValueError('Native PTS step must be a positive integer')
    shots = {shot['id']: shot for shot in scope['shots']}
    if len(shots) != len(scope['shots']):
        raise ValueError('Programme scope repeats a shot identity')
    batches = request['sequences']
    if not isinstance(batches, list) or not batches:
        raise ValueError('At least one explicit source sequence is required')
    selected, intervals = [], {}
    for batch in batches:
        shot = shots.get(batch['shot_id'])
        if shot is None:
            raise ValueError('Requested shot is absent from the programme scope')
        start, stop = batch['first_pts'], batch['stop_pts_exclusive']
        first, limit = shot['processing']['first_pts'], shot['processing']['stop_pts_exclusive']
        if (type(start) is not int or type(stop) is not int or stop <= start
                or start < first or stop > limit):
            raise ValueError('Requested sequence is outside the original shot scope')
        if (start-first) % step or (stop-first) % step:
            raise ValueError('Requested sequence must be aligned to original native PTS')
        intervals.setdefault(shot['id'], []).append((start, stop))
        selected.append((batch, shot))
    for ranges in intervals.values():
        ordered = sorted(ranges)
        if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError('Source sequences overlap and would overwrite the same frame result')
    return selected


def run_source_read(request_path, output):
    import av
    from ..vision import SourceRinkReader, _source_json
    request = load(request_path)
    output = guard_project_write(Path(output).resolve())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Source read output contains prior evidence; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    scope = load(request['programme_scope'])
    sequences = _validated_sequences(request, scope)
    video = Path(scope['source']['registered_path'])
    expected_sha = scope['source']['sha256']
    if file_ref(video)['sha256'] != expected_sha:
        raise ValueError('Original source video hash differs from programme scope')
    clock = str(Fraction(scope['source_clock']['time_base_num'],scope['source_clock']['time_base_den']))
    step = scope['source_clock']['native_pts_step']
    reader = SourceRinkReader(request['vision_config'], source_reidentification=request.get('source_reidentification'), programme_scope=scope)
    if reader.native_geometry is None:
        raise ValueError('Current native geometry is required by this execution entry')
    geometry_before = reader.native_geometry.refs
    navigation = None
    if request.get('navigation') is not None:
        from ..vision_navigation import NativeNavigation
        spec = request['navigation']
        seed_set = Path(spec['seed_set'])
        if not seed_set.is_absolute():
            seed_set = Path(request_path).resolve().parent/seed_set
        navigation = NativeNavigation(reader.native_geometry, seed_set, scope=scope,
            search_radius_px=spec.get('search_radius_px',24.),
            sample_spacing_px=spec.get('sample_spacing_px',8.))
    reports = []
    started = time.monotonic()
    for batch, shot in sequences:
        start, stop = batch['first_pts'], batch['stop_pts_exclusive']
        acceptance = load(batch['acceptance']) if batch.get('acceptance') else None
        decoded = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            if str(stream.time_base) != clock:
                raise ValueError('Original source time base differs from scope')
            container.seek(start, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts < start:
                    continue
                if frame.pts >= stop:
                    break
                pts = int(frame.pts)
                if pts != start + len(decoded)*step:
                    raise ValueError('Missing native frame: never fabricate PTS')
                rgb = frame.to_ndarray(format='rgb24')
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                binding = dict(source_sha256=expected_sha, source_pts=pts, time_base=clock,
                    shot_id=shot['id'], native_hw=list(rgb.shape[:2]),
                    evaluation_role='development', decoded_native_rgb_sha256=digest)
                from ..source_binding import canonical_source_binding
                binding = canonical_source_binding(binding, scope)
                guides, reference, masks = None, None, {}
                accepted_guides = []
                if acceptance:
                    receipts = [r for r in acceptance['frames'] if r.get('native_pts') == pts]
                    if len(receipts) != 1:
                        raise ValueError('Accepted reference requires one exact native PTS receipt')
                    index = receipts[0]['index']
                    packet_path = Path(batch['packet_directory'])/(str(index).zfill(6)+'.json')
                    binding_path = Path(batch['binding_directory'])/packet_path.name
                    packet, original_binding = load(packet_path), load(binding_path)
                    accepted_guides, reference = accepted_packet_guides(packet, original_binding, acceptance,
                        packet_path=packet_path, binding_path=binding_path, acceptance_path=batch['acceptance'],
                        source_binding=binding, programme_scope=scope)
                    source = packet['source']
                    if source['decoded_RGB_sha256'] != digest:
                        raise ValueError('Accepted packet differs from the actual decoded native RGB')
                    mask_path = Path(source['mask_path'])
                    if file_ref(mask_path)['sha256'] != source['mask_sha256']:
                        raise ValueError('Current source exclusion mask changed')
                    masks['ice_paint'] = np.asarray(Image.open(mask_path).convert('L')) != 0
                    # First frame anchors identity. Every following frame uses the
                    # reader's continuous optical flow and current RGB rereading.
                    if not decoded:
                        guides = accepted_guides
                elif not decoded:
                    guides = []
                extra_guides, navigation_report = None, None
                if navigation is not None:
                    extra_guides, navigation_report = navigation.project(binding)
                result = reader.read(rgb, binding, guides_2d=guides, reset=not decoded,
                    current_source_masks_by_kind=masks,
                    extract_source_components=not decoded,
                    additional_search_guides=extra_guides,
                    additional_source_guides=accepted_guides if decoded else None)
                if navigation_report is not None:
                    result['native_navigation'] = navigation_report
                from ..vision_storage import pack_source_record_arrays
                name = f"{shot['id']}_{pts}"
                # Reuse the existing lossless array store before JSON conversion.
                compact = pack_source_record_arrays(result, output/(name+'_arrays.npz'),
                    minimum_array_bytes=16384, storage_layout='byte_pool')
                clean = _source_json(compact)
                rows = []
                # The packed record contains sidecar references for large arrays.
                # Statistics and overlays must consume the original measurements.
                for reading in result['guided']['readings']:
                    actual = reading.get('coherent_source_uv', reading['actual_source_uv'])
                    valid = reading.get('temporal_flow_supported')
                    rows.append(dict(id=reading.get('expected_id'), primitive=reading.get('primitive'),
                        role=reading.get('physical_role'), read_count=len(actual),
                        tracked_count=None if valid is None else int(sum(_source_json(valid))),
                        queried_count=reading.get('metadata',{}).get('source_point_count'),
                        continuity_status=reading.get('coherent_flags',{}).get('status')))
                clean['preserved_accepted_reference'] = reference
                clean['accepted_reference_used_for_guides'] = guides is not None and bool(guides)
                clean['accepted_reference_used_for_gap_queries'] = bool(decoded and accepted_guides)
                clean['evaluation_role'] = 'regression' if acceptance else 'development'
                name = f"{shot['id']}_{pts}"
                write(output/(name+'.json'), clean)
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                for reading in result['guided']['readings']:
                    # These are measured material samples. The fitted diagnostic
                    # and immutable native scene remain separately identifiable.
                    for u,v in _source_json(reading.get('coherent_source_uv', reading['actual_source_uv'])):
                        draw.ellipse((u-2,v-2,u+2,v+2),fill=(255,180,0))
                near=result.get('near_board',{})
                near_count=sum(r.get('source_point_count',0) for r in near.get('readings',[]))
                for reading in near.get('readings',[]):
                    for u,v in _source_json(reading.get('actual_source_uv',[])):
                        draw.ellipse((u-2,v-2,u+2,v+2),fill=(0,220,80))
                image.save(output/(name+'.png'))
                report = dict(shot_id=shot['id'], source_pts=pts, source_time_seconds=float(pts*Fraction(clock)),
                    decoded_native_RGB_sha256=digest, memory_update=result['memory_update'],
                    new_observation=result['new_observation'], original_acceptance_preserved=reference is not None,
                    guide_origin='saved_accepted_source' if guides else 'continuous_current_RGB_tracking' if decoded else 'new_shot_search',
                    readings=rows, near_board_points=near_count,
                    near_board_source_reader=near.get('source_reader'),
                    near_board_excluded_pixels=near.get('exclusion_pixel_count',0),
                    source_feedback=clean['source_feedback'],
                    navigation_dispatch=clean.get('navigation_dispatch'),
                    navigation_query_points=(navigation_report['query_point_count']
                                             if navigation_report is not None else 0),
                    source_reidentification={key:clean['source_reidentification'].get(key) for key in
                        ('status','verified_physical_ids','guide_count','native_path_ids','board_search')} if clean.get('source_reidentification') else None,
                    native_geometry=clean['native_geometry'], result=name+'.json', image=name+'.png')
                reports.append(report); decoded.append(pts)
                publish_delayed_source_readouts(result,reports,output,video)
                print(json.dumps(dict(shot=shot['id'], pts=pts, memory=result['memory_update'],
                    measured=sum(r['read_count'] for r in rows),near_board_points=near_count, elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
                write(output/'progress.json',dict(frames_completed=len(reports),last=report))
        if len(decoded)*step != stop-start:
            raise ValueError('Source sequence ended before requested native PTS')
    reader.native_geometry.assert_unchanged()
    if navigation is not None:
        navigation.assert_unchanged()
    receipt = dict(schema='rinkspace.continuous_source_reader_run.v1',request=file_ref(request_path),
        programme_scope=file_ref(request['programme_scope']), required_shots=scope['required_shot_ids'],
        validated_frames=len(reports), frames=reports, model_usage=reader.vision.model_usage(),
        native_geometry_refs=geometry_before, native_geometry_unchanged=True,
        navigation_refs=navigation.refs if navigation is not None else [],
        original_source_UV_and_acceptance_files_modified=False,
        full_programme_accuracy_accepted=False, fit_executed=False,
        elapsed_seconds=time.monotonic()-started)
    write(output/'result.json', receipt)
    return dict(result_path=str(output/'result.json'), frames=len(reports),
                native_geometry_unchanged=True, fit_executed=False)
