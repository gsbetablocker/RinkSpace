"""Continuous native-pixel rink review, with visible failures and source identity.

This is an output adapter for the existing reader, not another solver.  It does
not turn a class probability, transported guide or SAM2 mask into pixel truth.
Optional prompts are source-bound search aids.  Independent mode never injects
annotations from a frame being evaluated.  Outputs belong in the Data tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .vision import ROOT, SourceRinkReader, read_json, sha256, write_json


def _canonical_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def source_shots(path, source_sha256, *, shot_ids=(), boundary_overlay_path=None):
    """Normalize existing programme/timeline cuts without guessing cameras."""
    packet = read_json(path)
    digest = packet.get("source_sha256", packet.get("source_sha256_from_timeline"))
    if digest != source_sha256:
        raise ValueError("Shot classification belongs to a different source video")
    clock = Fraction(packet.get("time_base", packet.get("source_time_base")))
    selected = set(shot_ids)
    shots, original_rows = [], {}
    for index, row in enumerate(packet.get("shots", packet.get("intervals", ()))):
        identity = str(row.get("id", row.get("shot_id", index)))
        if identity in original_rows:
            raise ValueError("Duplicate classified source shot ID")
        original_rows[identity] = row
        start = int(row["start_pts"])
        end = int(row["end_pts_exclusive"] if "end_pts_exclusive" in row else row["end_pts"])
        if start < 0 or end <= start:
            raise ValueError("Invalid classified half-open source shot")
        shots.append(dict(id=identity, index=index, start_pts=start,
                          end_pts_exclusive=end, camera_id=row.get("camera_id"),
                          interval_type=row.get("interval_type", "shot"),
                          camera_authority=row.get("authority"),
                          review_status=row.get("review_status"), time_base=str(clock),
                          expected_native_frame_count=row.get("source_native_frame_count")))
    shots.sort(key=lambda s: s["start_pts"])
    if boundary_overlay_path:
        overlay = read_json(boundary_overlay_path)
        if (overlay.get("schema") != "rinkspace.native-shot-boundary-correction.v1" or
                overlay.get("source_sha256") != source_sha256 or
                overlay.get("original_timeline_sha256") != sha256(path) or
                Fraction(str(overlay.get("time_base", "0"))) != clock):
            raise ValueError("Boundary overlay source, clock, schema or original timeline SHA differs")
        by_id = {shot["id"]: shot for shot in shots}
        overrides = overlay.get("shot_overrides", ())
        if not overrides or len({row["shot_id"] for row in overrides}) != len(overrides):
            raise ValueError("Boundary overlay needs distinct, existing source shot rows")
        changes = {}
        for correction in overrides:
            identity = correction["shot_id"]
            if identity not in by_id:
                raise ValueError("Boundary overlay references an absent source shot")
            base, original = by_id[identity], original_rows[identity]
            if (correction["original_start_pts"] != base["start_pts"] or
                    correction["original_end_pts_exclusive"] != base["end_pts_exclusive"] or
                    correction["original_row"] != original or
                    correction["original_row_json_sha256"] != _canonical_json_sha256(original) or
                    correction["corrected_row_json_sha256"] != _canonical_json_sha256(correction["corrected_row"])):
                raise ValueError("Boundary overlay original row/range or corrected row integrity differs")
            new = correction["corrected_row"]
            if (new.get("id", new.get("shot_id")) != identity or
                    new["start_pts"] != correction["start_pts"] or
                    new.get("end_pts_exclusive", new.get("end_pts")) != correction["end_pts_exclusive"]):
                raise ValueError("Boundary overlay corrected row and range disagree")
            for field, overlay_field in (("camera_id", "camera_id"), ("authority", "camera_authority"),
                                         ("review_status", "camera_review_status")):
                if new.get(field) != original.get(field) or correction.get(overlay_field) != original.get(field):
                    raise ValueError("Boundary correction cannot change camera taxonomy or authority")
            start, end = correction["start_pts"], correction["end_pts_exclusive"]
            if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                    or not isinstance(end, int) or not 0 <= start < end):
                raise ValueError("Corrected source shot needs a valid integer half-open range")
            changes[identity] = correction
        for index, shot in enumerate(shots):
            correction = changes.get(shot["id"])
            if not correction:
                continue
            for side, neighbor_index, original_key, new_key, neighbor_key in (
                ("start", index-1, "original_start_pts", "start_pts", "end_pts_exclusive"),
                ("end", index+1, "original_end_pts_exclusive", "end_pts_exclusive", "start_pts")):
                if correction[original_key] == correction[new_key]:
                    continue
                if not 0 <= neighbor_index < len(shots):
                    raise ValueError("Changed cut has no adjacent source shot")
                neighbor = shots[neighbor_index]
                adjacent = changes.get(neighbor["id"])
                if (adjacent is None or neighbor[neighbor_key] != correction[original_key] or
                        adjacent[neighbor_key] != correction[new_key]):
                    raise ValueError("Both adjacent source shot boundaries must be corrected together")
        overlay_meta = dict(path=str(Path(boundary_overlay_path).resolve()), sha256=sha256(boundary_overlay_path),
            original_timeline_sha256=overlay["original_timeline_sha256"], source_sha256=source_sha256,
            time_base=str(clock), source_cut_identity_authority=overlay.get("source_cut_identity_authority"),
            ranges=[{key: row[key] for key in ("shot_id", "original_start_pts", "original_end_pts_exclusive",
                                              "start_pts", "end_pts_exclusive", "camera_id", "camera_authority")}
                    for row in overrides], camera_taxonomy_and_authority_preserved=True)
        for shot in shots:
            shot["source_boundary_overlay"] = overlay_meta
            if shot["id"] in changes:
                correction = changes[shot["id"]]
                shot.update(start_pts=correction["start_pts"], end_pts_exclusive=correction["end_pts_exclusive"],
                            original_classified_start_pts=correction["original_start_pts"],
                            original_classified_end_pts_exclusive=correction["original_end_pts_exclusive"],
                            expected_native_frame_count=None)
    shots = [shot for shot in shots if shot["interval_type"] == "shot"]
    if selected:
        shots = [shot for shot in shots if shot["id"] in selected or str(shot["index"]) in selected]
    if not shots or (selected and not selected.issubset(
            {s["id"] for s in shots} | {str(s["index"]) for s in shots})):
        raise ValueError("Requested classified source shots are absent")
    shots.sort(key=lambda s: s["start_pts"])
    for previous, following in zip(shots, shots[1:]):
        if previous["end_pts_exclusive"] > following["start_pts"]:
            raise ValueError("Classified source shots overlap")
    return shots


def clip_diagnostic_shots(shots, start_pts, end_pts):
    """Clip a native diagnostic window across verified adjacent source cuts."""
    if (start_pts is None or end_pts is None or not shots or
            not shots[0]["start_pts"] <= start_pts < end_pts <= shots[-1]["end_pts_exclusive"]):
        raise ValueError("A diagnostic window needs native bounds within its selected source shots")
    clipped = []
    for shot in shots:
        start, end = max(start_pts, shot["start_pts"]), min(end_pts, shot["end_pts_exclusive"])
        if start >= end:
            continue
        clipped.append(dict(shot, classified_full_start_pts=shot["start_pts"],
            classified_full_end_pts_exclusive=shot["end_pts_exclusive"],
            start_pts=start, end_pts_exclusive=end, expected_native_frame_count=None))
    if any(previous["end_pts_exclusive"] != following["start_pts"]
           for previous, following in zip(clipped, clipped[1:])):
        raise ValueError("A diagnostic window cannot bridge unselected or unclassified source gaps")
    return clipped


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_json(value):
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        return _finite_json(_array(value).tolist())
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class SourceRecordMemberStream:
    """Publish each complete lossless record as a readable gzip member.

    The small commit index permits concurrent source checking by byte range.
    Only a finished member is published; the arrays remain in their sidecar.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.progress_path = self.path.with_name('readings_committed_frames.jsonl')
        self._records = self._progress = None
        self.record_count = 0

    def __enter__(self):
        self._records = self.path.open('xb')
        try:
            self._progress = self.progress_path.open('x', encoding='utf-8', newline='\n')
        except BaseException:
            self._records.close()
            raise
        return self

    def __exit__(self, *args):
        if self._progress is not None:
            self._progress.close()
        if self._records is not None:
            self._records.close()

    def commit_source_record(self, encoded, source_binding, *, array_sidecar_path=None):
        member = gzip.compress(encoded, compresslevel=1, mtime=0)
        offset = self._records.tell()
        self._records.write(member)
        self._records.flush()
        self.record_count += 1
        entry = dict(schema='rinkspace.committed-native-source-record.v1',
            frame_ordinal=self.record_count, source_binding=source_binding,
            record_member_offset=offset, record_member_compressed_bytes=len(member),
            record_member_sha256=hashlib.sha256(member).hexdigest(),
            uncompressed_UTF8_bytes=len(encoded), uncompressed_UTF8_sha256=hashlib.sha256(encoded).hexdigest(),
            array_sidecar=str(Path(array_sidecar_path).name) if array_sidecar_path is not None else None,
            complete_record_readable=True, original_arrays_copied_for_progress=False)
        self._progress.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        self._progress.flush()


def load_committed_source_record(readings_path, entry, *, expected_source_binding=None):
    """Read one published native frame without scanning a still-open programme."""
    from .vision_storage import hydrate_source_record_arrays
    offset, length = entry['record_member_offset'], entry['record_member_compressed_bytes']
    if isinstance(offset, bool) or isinstance(length, bool) or offset < 0 or length <= 0:
        raise ValueError('Invalid committed source record byte range')
    with Path(readings_path).open('rb') as stream:
        stream.seek(offset)
        member = stream.read(length)
    if len(member) != length or hashlib.sha256(member).hexdigest() != entry['record_member_sha256']:
        raise ValueError('Committed source record member changed or is incomplete')
    encoded = gzip.decompress(member)
    if len(encoded) != entry['uncompressed_UTF8_bytes'] or hashlib.sha256(encoded).hexdigest() != entry['uncompressed_UTF8_sha256']:
        raise ValueError('Committed source record content changed')
    row = json.loads(encoded.decode('utf-8'))
    if row['source_binding'] != entry['source_binding']:
        raise ValueError('Committed source record binding differs from its progress index')
    return hydrate_source_record_arrays(row, Path(readings_path).parent,
        expected_source_binding=expected_source_binding or entry['source_binding'])


def _write_source_record(records, row, content_hash, *, array_sidecar_path=None, storage_metadata=None,
                         minimum_array_bytes=16384, storage_layout='individual_arrays'):
    """Hash the same UTF-8 LF record bytes written by the explicit LF stream."""
    if array_sidecar_path is not None:
        from .vision_storage import ARCHIVE_FIELD, pack_source_record_arrays
        row=pack_source_record_arrays(row,array_sidecar_path,minimum_array_bytes=minimum_array_bytes,
                                     storage_layout=storage_layout)
        archive=row.get(ARCHIVE_FIELD)
        if archive and storage_metadata is not None:
            for key,value in (('sidecar_count',1),('sidecar_compressed_bytes',archive['compressed_bytes']),
                    ('unique_array_count',archive['unique_array_count']),
                    ('array_reference_count',archive['reference_count']),
                    ('duplicate_array_references',archive['duplicate_array_references'])):
                storage_metadata[key]=storage_metadata.get(key,0)+value
    line = json.dumps(_finite_json(row), ensure_ascii=False, allow_nan=False) + "\n"
    encoded = line.encode("utf-8")
    if hasattr(records, 'commit_source_record'):
        records.commit_source_record(encoded, row['source_binding'], array_sidecar_path=array_sidecar_path)
    else:
        records.write(line)
    content_hash.update(encoded)
    return len(encoded)


def load_saved_source_record(readings_path,source_pts,*,expected_source_binding=None):
    """Read one exact native record and validate its optional lossless arrays."""
    from .vision_storage import hydrate_source_record_arrays
    with gzip.open(readings_path,'rt',encoding='utf-8') as records:
        row=next((item for line in records
                  if (item:=json.loads(line))['source_binding']['source_pts']==source_pts),None)
    if row is None:
        raise ValueError('Saved continuous source timestamp is absent')
    return hydrate_source_record_arrays(row,Path(readings_path).parent,
        expected_source_binding=expected_source_binding)


def application_source_record(row, *, result=None):
    """Retain fitting observations and explicit failures, not inference workspaces.

    Actual UV, measured intervals, identity authority and roles are copied from
    existing readings. No source mask, profile or fitted/projection UV is saved.
    Legacy decoded lists stay lists; their original tensor dtype is not guessed.
    """
    def selected(value, indices, count):
        if hasattr(value, 'detach') or isinstance(value, np.ndarray):
            value = _array(value)
            if value.ndim and len(value) != count and len(indices) and len(value) > int(indices.max()):
                return value[indices]
            return value
        return value
    def flags(value, indices, count):
        output = {}
        for key, item in (value or {}).items():
            if isinstance(item, dict):
                output[key] = flags(item, indices, count)
            elif hasattr(item, 'detach') or isinstance(item, np.ndarray):
                array = _array(item)
                if array.ndim <= 1:
                    output[key] = selected(array, indices, count)
                else:
                    output[key] = dict(observed_true_count=int(np.count_nonzero(array)),
                        intermediate_array_not_saved=True)
            elif not isinstance(item, (list, tuple)):
                output[key] = item
        return output
    groups=[]
    def group(reading, identifier, *, context=None, primary=False, disposition='source_measurement_hypothesis'):
        uv = reading.get('primary_source_uv') if primary else reading.get('coherent_source_uv',reading.get('actual_source_uv',[]))
        if uv is None:
            uv=[]
        uv = _array(uv) if hasattr(uv,'detach') or isinstance(uv,np.ndarray) else uv
        indices = reading.get('primary_source_search_indices') if primary else reading.get('coherent_source_indices',reading.get('source_indices',reading.get('original_guide_indices',[])))
        indices = _array(indices).astype(np.int64,copy=False).reshape(-1) if indices is not None else np.empty(0,np.int64)
        runs = reading.get('primary_measured_runs',()) if primary else reading.get('coherent_measured_runs',reading.get('source_measured_runs',reading.get('measured_runs',())))
        compact_runs=[]
        for run in runs:
            if isinstance(run,dict):
                compact_runs.append({key: _array(item) if hasattr(item,'detach') or isinstance(item,np.ndarray) else item
                    for key,item in run.items() if key in ('actual_source_uv','source_indices','original_source_search_indices','segment_id','closed')})
            else:
                compact_runs.append(dict(actual_source_uv=run))
        metadata=reading.get('metadata',{})
        search_count=metadata.get('source_point_count',reading.get('expected_search_sample_count'))
        observation=dict(group_id=str(identifier),actual_source_uv=uv,source_indices=indices,measured_runs=compact_runs,
            measurement_method='current_original_RGB_profile_reread',RGB_recheck_executed=True,
            expected_id=reading.get('expected_id'),physical_id=reading.get('physical_id'),
            source_group_id=reading.get('source_group_id'),
            physical_curve_id=reading.get('physical_curve_id',reading.get('source_group_id')),
            expected_curve_id=reading.get('expected_id'),
            identity_authority=reading.get('identity_authority'),geometry_role=reading.get('physical_role',reading.get('boundary_role')),
            kind=reading.get('kind'),disposition=disposition,measured_count=len(uv),
            expected_search_sample_count=search_count,
            missing_search_sample_count=reading.get('missing_search_sample_count',max(0,search_count-len(np.unique(indices))) if search_count is not None else None),
            physical_pixel_accuracy_accepted=False,projection_or_mask_UV_used_as_output=False,
            source_flags=flags(reading.get('coherent_flags',reading.get('flags',{})),indices,len(uv)),
            original_measurement_flags=flags(reading.get('flags',{}),indices,len(uv)),
            model_disagreement=flags(reading.get('model_disagreement',{}),indices,len(uv)),
            source_flag_counts=reading.get('source_flags',{}))
        for key in ('physical_boundary_verified','print_material_scope','internal_hole_scope','selected_material_hole',
                    'whole_printed_pattern_hole_verified','solver_outer_boundary_eligible','source_reference_component_scope',
                    'whole_advertisement_outer_boundary_eligible','material_role_authority','source_component_id'):
            if key in reading:
                observation[key]=reading[key]
        support=reading.get('coherent_support')
        if support:
            observation['observed_uncertainty']={key:_array(value) if hasattr(value,'detach') or isinstance(value,np.ndarray) else value
                for key,value in support.items() if key in ('fwhm_px','normal_interval_px','left_border_uv','right_border_uv','band_width_px','source_quality_weights')}
        elif 'support_fwhm_px' in reading:
            observation['observed_uncertainty']=dict(support_fwhm_px=selected(reading['support_fwhm_px'],indices,len(uv)))
        else:
            observation['observed_uncertainty']=dict(localization_interval_not_available=True,calibrated_sigma_not_invented=True)
        interval=reading.get('observed_RGB_transition_intervals')
        if interval:
            observation['observed_RGB_transition_intervals']={key:_array(value) if hasattr(value,'detach') or isinstance(value,np.ndarray) else value
                for key,value in interval.items() if key in ('normal_interval_native_px','same_material_signed_transition_normal_interval_native_px',
                    'actual_RGB_interval_uv','source_indices','interval_profile_clipped','interval_is_exact_physical_boundary')}
        if context:
            observation.update(context)
        groups.append(observation)
    named = result['guided']['readings'] if result is not None else row.get('named_readings',())
    for index, reading in enumerate(named):
        group(reading,reading.get('source_group_id') or reading.get('physical_id') or reading.get('expected_id') or f'guided_{index}')
    foot=row.get('source_board_foot_material_reading',{})
    for index,reading in enumerate(foot.get('readings',())):
        group(reading,f'board_foot_material_{index}',primary=True,context=dict(
            geometry_role=reading.get('physical_role'),role_is_material_hypothesis=True))
        support=foot.get('source_material_transition_support',{})
        if support.get('observed_native_warm_colour_fall_support_endpoints_uv') is not None:
            indices=_array(reading.get('primary_source_search_indices',())).astype(np.int64).reshape(-1)
            groups[-1]['observed_uncertainty']=dict(
                actual_RGB_descending_material_support_uv=_array(support['observed_native_warm_colour_fall_support_endpoints_uv'])[indices],
                observed_support=_array(support['observed_support'])[indices],
                interval_is_exact_physical_contact=False,calibrated_sigma_not_invented=True)
    top=row.get('source_board_top_role_assignment',{}).get('primary_source_top')
    if top:
        uv=top.get('actual_source_uv')
        if uv is None:
            uv=_run_points(top.get('source_pixel_runs_y_x_length',())).astype(np.float32)
        group(dict(actual_source_uv=uv,physical_role=top.get('physical_role'),kind='board_top'),
              'board_upper_rim',context=dict(role_is_material_hypothesis=True))
    states=[]
    for item in row.get('instances',()):
        state={key:item[key] for key in ('object_id','kind','physical_id','expected_id','candidate_physical_ids','content_id',
            'identity_authority','expected_identity_authority','status','missing_reason','prompt_origin',
            'source_material_proposal_unverified','source_print_visibility_status') if key in item}
        states.append(state)
        if item.get('kind')!='advertisement_panel':
            continue
        identifiable=bool(item.get('physical_id') or item.get('expected_id') or item.get('candidate_physical_ids'))
        if not identifiable:
            state['application_geometry_usable']=False
            state['application_geometry_missing_reason']='source_material_has_no_established_print_instance_identity'
            state['unidentified_material_measurements']={collection:dict(
                measured_points=sum(len(reading.get('actual_source_uv',())) for reading in item.get(collection,())),
                reading_count=len(item.get(collection,())),actual_measurement_UV_not_used_as_named_geometry=True,
                reproduction_source_binding=row['source_binding']) for collection in
                ('boundary_readings','source_print_role_conflict_readings')}
            continue
        for collection,disposition in (('boundary_readings','printed_source_hypothesis'),
                ('source_print_role_conflict_readings','source_role_conflict')):
            for index,reading in enumerate(item.get(collection,())):
                group(reading,f"object_{item['object_id']}:{collection}:{index}",context=state,disposition=disposition)
    states.extend(row.get('tracked_instance_identity_states',()))
    return dict(schema='rinkspace.application-source-observations.v1',source_binding=row['source_binding'],
        RGB_sha256=row['RGB_sha256'],source_time=row.get('source_time'),model_sha256=row['model_sha256'],
        model_usage=row.get('model_usage'),original_FIT_source_revision=row.get('original_FIT_source_revision'),
        source_clock_evidence=row.get('source_clock_evidence'),source_clock_scope=row.get('source_clock_scope'),
        memory_update=row.get('memory_update'),new_observation=row.get('new_observation'),
        source_motion_metadata=row.get('source_motion_metadata'),
        measurement_groups=groups,object_states=states,issues=row.get('issues',()),
        existing_model_search=row.get('existing_model_search'),
        application_QA=dict(source_role_reading_status=row.get('source_role_reading_status'),
            component_counts_by_class=row.get('component_counts_by_class'),
            source_edge_class_overlap=row.get('source_edge_class_overlap'),
            current_person_and_print_region_reidentification_performed=row.get('current_person_and_print_region_reidentification_performed',True),
            full_frame_component_extraction_performed=row.get('full_frame_component_extraction_performed',True),
            physical_ids_automatically_inferred=False,intermediate_profiles_masks_and_raw_pools_not_saved=True,
            failed_reading_reproduction_source=row['source_binding'],fitted_points_used_as_output=False),
        source_reidentification_keyframe=row.get('source_reidentification_keyframe',True),tracking_only=row.get('tracking_only',False),
        last_source_reidentification_binding=row.get('last_source_reidentification_binding'),
        source_seed_injected=row.get('source_seed_injected',False),
        instance_source_frame_index=row.get('instance_source_frame_index'),
        instance_source_shot_reset_requested=row.get('instance_source_shot_reset_requested'))


def draw_application_record(rgb, record):
    """Show saved actual observations, without masks, candidate pools or a grid."""
    from PIL import Image, ImageDraw, ImageFont
    image=Image.fromarray(np.asarray(rgb).copy())
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    colors=dict(ice_paint='#ff45c4',ice_contact='#ff991a',board_top='#ffe34b',ad_boundary='#22dcf6')
    for group in record['measurement_groups']:
        points=_array(group['actual_source_uv']).reshape(-1,2)
        role=str(group.get('geometry_role') or '')
        kind=group.get('kind')
        if 'ice_contact' in role:
            kind='ice_contact'
        elif 'upper_rim' in role:
            kind='board_top'
        elif kind=='advertisement_panel':
            kind='ad_boundary'
        color='#f44848' if group['disposition']=='source_role_conflict' else colors.get(kind,'#a1d496')
        finite=points[np.isfinite(points).all(1)]
        for x,y in finite:
            draw.ellipse((float(x)-1,float(y)-1,float(x)+1,float(y)+1),fill=color)
        if len(finite):
            x,y=finite[len(finite)//2]
            name=group.get('physical_id') or group.get('expected_id') or group['group_id']
            if not group.get('physical_id'):
                name=str(name)+'?'
            draw.text((float(x)+3,float(y)+2),str(name),font=font,fill=color,stroke_width=1,stroke_fill='black')
    h,w=np.asarray(rgb).shape[:2]
    draw.rectangle((0,h-48,w,h),fill='black')
    draw.text((10,h-45),f"Source {record.get('source_time')} PTS {record['source_binding']['source_pts']} | actual RGB points; ? identity unverified",font=font,fill='white')
    draw.text((10,h-23),'Paint pink | contact orange | upper rim yellow | print cyan | red role conflict | gaps are missing; accuracy not accepted',font=font,fill='white')
    return np.asarray(image)


def retain_same_shot_installation_identity(binding, previous_keyframe_binding):
    """Keep a verified installation alias across one classified source shot.

    This carries a source identity, not projected coordinates or a camera pose.
    The original classification remains visible in the binding metadata.
    """
    output=dict(binding)
    if previous_keyframe_binding is None:
        return output
    for field in ('source_sha256','time_base','shot_id','native_hw'):
        if output.get(field)!=previous_keyframe_binding.get(field):
            raise ValueError('Installation identity cannot cross a source, clock, shot or image domain')
    if output['source_pts']<=previous_keyframe_binding['source_pts']:
        raise ValueError('Installation identity carry requires a later native source timestamp')
    output['source_classification_installation_id']=output.get('installation_id')
    output['installation_id']=previous_keyframe_binding['installation_id']
    output['installation_identity_origin']='same_classified_shot_from_previous_exact_source_keyframe'
    output['installation_identity_source_pts']=previous_keyframe_binding['source_pts']
    return output


def carry_confirmed_application_guides(reader, binding):
    """Transport established source identities as search, then reread RGB."""
    import torch
    from torch.nn import functional as F
    from .vision_geometry import warp
    guides=[]
    pair=reader.vision.last_motion
    h,w=binding['native_hw']
    for original in reader.active_guides:
        if not original.get('identity_authority'):
            continue
        guide=dict(original)
        points=torch.as_tensor(guide['points_uv'],device=reader.vision.device,dtype=torch.float32)
        if pair is not None:
            grid=points*points.new_tensor([2/(w-1),2/(h-1)])-1
            movement=F.grid_sample(pair.forward,grid[None,:,None],align_corners=True)[0,:,:,0].T
            guide['temporal_flow_supported']=F.grid_sample(pair.valid_forward.float(),grid[None,:,None],
                mode='nearest',align_corners=True)[0,0,:,0]>.5
            points=points+movement
            if guide.get('source_exclusion_mask') is not None:
                mask,inside=warp(guide['source_exclusion_mask'].float(),pair.backward)
                guide['source_exclusion_mask']=(mask>.5)&inside
        guide.update(points_uv=points,source_binding=binding,origin='temporal_source_search_hint',
            identity_confirmed=True,hint_previous_source_binding=reader.active_binding,hint_is_source_measurement=False)
        if guide.get('source_exclusion_mask') is not None:
            guide['source_exclusion_binding']=binding
        guides.append(guide)
    return guides


def retain_current_print_identity_guides(reader, instances, binding):
    """Keep valid matched print runs for subsequent current-RGB remeasurement."""
    import torch
    added=[]
    for instance in instances:
        physical=instance.get('physical_id')
        authority=instance.get('identity_authority')
        if instance.get('kind')!='advertisement_panel' or not physical or not authority:
            continue
        by_role={}
        for reading in instance.get('boundary_readings',()):
            role=reading.get('boundary_role')
            if not role or reading.get('missing_reason'):
                continue
            for run in reading.get('measured_runs',()):
                points=run['actual_source_uv']
                if len(points)<2:
                    continue
                by_role.setdefault(role,[]).append(points)
        for role,runs in by_role.items():
            points=torch.cat([torch.as_tensor(points,device=reader.vision.device,dtype=torch.float32)for points in runs])
            segments=torch.cat([torch.full((len(points),),i,device=reader.vision.device,dtype=torch.int64)for i,points in enumerate(runs)])
            added.append(dict(id=physical,kind='ad_boundary',primitive='free_curve',physical_role=role,
                points_uv=points,segment_ids=segments,closed=False,source_binding=binding,
                origin='temporal_source_search_hint',identity_confirmed=True,identity_authority=authority,
                source_group_id=f"{physical}:{role}:actual_source_print_runs",search_radius_px=8.,
                source_current_print_reference_object_id=instance['object_id'],
                hint_previous_source_binding=binding,hint_is_source_measurement=False))
    if added:
        current_ids={guide['source_group_id']for guide in added}
        reader.active_guides=[guide for guide in reader.active_guides if guide.get('source_group_id')not in current_ids]+added
    return added


def format_source_time(source_pts, time_base):
    stamp = Fraction(int(source_pts)) * Fraction(time_base)
    milliseconds = (stamp.numerator * 1000 + stamp.denominator // 2) // stamp.denominator
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def _flag_counts(flags):
    counts = {}
    for name, value in flags.items():
        if isinstance(value, (bool, str)):
            counts[name] = value
        elif hasattr(value, "shape"):
            array = _array(value)
            if array.dtype == bool:
                counts[name] = int(array.sum())
    return counts


def _pixel_runs(points):
    """Lossless row runs of actual integer source pixels; never fill a gap."""
    xy = np.asarray(points, dtype=np.int64)
    if not len(xy):
        return []
    xy = xy[np.lexsort((xy[:, 0], xy[:, 1]))]
    breaks = np.flatnonzero((np.diff(xy[:, 1]) != 0) | (np.diff(xy[:, 0]) != 1)) + 1
    return [[int(run[0, 1]), int(run[0, 0]), len(run)]
            for run in np.split(xy, breaks)]


def source_diagnostics(result):
    """Expose overlap, missing source samples and unestablished identities."""
    source = result["source"]
    h, w = source["native_hw"]
    components, memberships = [], {}
    for component in source["components"]:
        kind = component["typed_class_candidate"]
        points = _array(component["source_support_uv"]).astype(np.int64)
        memberships.setdefault(kind, []).append(points[:, 1] * w + points[:, 0])
        components.append(dict(kind=kind, source_pixel_runs_y_x_length=_pixel_runs(points),
            source_point_count=len(points), native_bbox_xyxy=component["source_bbox_xyxy"],
            physical_id=component.get("physical_instance_id"),
            candidate_kind=component["candidate_kind"], identity_verified=False,
            boundary_accuracy_verified=False))
    memberships = {k: np.unique(np.concatenate(v)) for k, v in memberships.items()}
    confusion = []
    names = tuple(memberships)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = np.intersect1d(memberships[first], memberships[second], assume_unique=True)
            if len(shared):
                confusion.append(dict(classes=[first, second], shared_actual_pixel_count=len(shared),
                    fraction_of_smaller_support=len(shared) / min(len(memberships[first]), len(memberships[second])),
                    interpretation="same RGB edge admitted under multiple semantic hypotheses"))
    boundary_order = None
    if "ice_contact" in memberships and "board_top" in memberships:
        by_column = {}
        for name in ("ice_contact", "board_top"):
            indices = memberships[name]
            x, y = indices % w, indices // w
            by_column[name] = {int(column): float(np.median(y[x == column])) for column in np.unique(x)}
        common = sorted(set(by_column["ice_contact"]) & set(by_column["board_top"]))
        gaps = np.asarray([by_column["ice_contact"][x] - by_column["board_top"][x] for x in common])
        boundary_order = dict(shared_source_columns=len(common),
            contact_not_below_top_columns=[common[i] for i in np.flatnonzero(gaps <= 0)],
            median_contact_minus_top_y_native_px=float(np.median(gaps)) if len(gaps) else None,
            interpretation="candidate-class vertical-order diagnostic; multiple edges in one column remain ambiguous")
    named, issues = [], []
    for reading in result["guided"]["readings"]:
        actual = _array(reading.get("coherent_source_uv", reading["actual_source_uv"]))
        source_indices = _array(reading.get("coherent_source_indices", reading["source_indices"]))
        runs = reading.get("coherent_measured_runs", reading["measured_runs"])
        expected_count = len(reading["guide"]["uv"])
        flags = dict(_flag_counts(reading.get("flags", {})),
                     **_flag_counts(reading.get("coherent_flags", {})))
        flow = reading.get("temporal_flow_supported")
        invalid_flow = int((~_array(flow).astype(bool)).sum()) if flow is not None else None
        identity = reading.get("physical_id")
        row = dict(expected_id=reading.get("expected_id"), physical_id=identity,
            source_group_id=reading.get("source_group_id"), kind=reading["kind"],
            physical_role=reading.get("physical_role"), identity_authority=reading.get("identity_authority"),
            actual_source_uv=actual.tolist(), original_guide_indices=source_indices.tolist(),
            measured_runs=[_array(run["actual_source_uv"]).tolist() for run in runs],
            expected_search_sample_count=expected_count, measured_count=len(actual),
            missing_search_sample_count=expected_count - len(np.unique(source_indices)),
            invalid_flow_hint_sample_count=invalid_flow, source_flags=flags,
            hint_pixels_used_as_output=False, pixel_accuracy_accepted=False)
        named.append(row)
        reasons = []
        if not identity:
            reasons.append("physical_identity_unconfirmed")
        if row["missing_search_sample_count"]:
            reasons.append("source_samples_missing_do_not_join_or_extrapolate")
        if invalid_flow:
            reasons.append("transported_search_hint_has_invalid_forward_backward_flow")
        if flags.get("competing_image_shape_hypothesis") or flags.get("multiple_source_peaks"):
            reasons.append("competing_source_edge_or_shape")
        if reasons:
            issues.append(dict(feature_id=identity or row["expected_id"], reasons=reasons,
                               measured_count=len(actual), missing_count=row["missing_search_sample_count"]))
    unidentified = Counter(c["kind"] for c in components if not c["physical_id"])
    if confusion:
        issues.append(dict(feature_id="semantic_edge_assignment", reasons=["source_edge_class_overlap"],
                           overlaps=confusion))
    if boundary_order and boundary_order["contact_not_below_top_columns"]:
        issues.append(dict(feature_id="ice_contact_vs_board_top", reasons=["candidate_boundary_order_conflict"],
                           diagnostic=boundary_order))
    if unidentified.get("ice_paint") or unidentified.get("ad_boundary"):
        issues.append(dict(feature_id="unidentified_source_components",
            reasons=["actual_source_edges_present_without_established_marker_or_ad_instance"],
            component_counts=dict(unidentified)))
    rejected_regions = {}
    for name, mask in source.get("source_region_rejection_masks", {}).items():
        yx = np.argwhere(_array(mask)[0, 0])
        rejected_regions[name] = dict(source_pixel_count=len(yx),
            source_pixel_runs_y_x_length=_pixel_runs(yx[:, [1, 0]]),
            reason="actual_RGB_support_outside_source_region_search_scope_not_erased_from_audit")
    rejected_quality = {}
    for field, reason in (("blur_rejection_mask", "source_edge_wider_than_current_reader_quality_limit"),
                          ("shadow_rejection_mask", "model_shadow_or_reflection_hypothesis_veto_unverified")):
        if field not in source:
            continue
        yx = np.argwhere(_array(source[field])[0, 0])
        rejected_quality[field] = dict(source_pixel_count=len(yx),
            source_pixel_runs_y_x_length=_pixel_runs(yx[:, [1, 0]]), reason=reason,
            rejection_is_source_accuracy_proof=False)
    role_status = {name: dict(actual_RGB_candidate_pixel_count=sum(
        c["source_point_count"] for c in components if c["kind"] == name),
        status="source_RGB_candidates_present_unverified" if any(c["kind"] == name for c in components) else
               "no_source_RGB_candidate_output_visibility_or_reading_unresolved")
        for name in ("ice_contact", "board_top", "ice_paint", "ad_boundary")}
    return dict(source_components=components, named_readings=named,
                source_edge_class_overlap=confusion, issues=issues,
                boundary_role_order_diagnostic=boundary_order,
                source_region_rejections=rejected_regions,
                source_quality_rejections=rejected_quality,
                source_role_reading_status=role_status,
                component_counts_by_class=dict(Counter(c["kind"] for c in components)),
                physical_ids_automatically_inferred=False)


def _compact_source_boundary_reading(reading):
    """Keep the complete measurement, with the existing display-run format."""
    row=dict(reading)
    def original(value):
        return _array(value) if isinstance(value,np.ndarray) or hasattr(value,'detach') else value
    row['actual_source_uv']=original(reading['actual_source_uv'])
    row['source_measured_runs']=reading.get('measured_runs',())
    row['measured_runs']=[original(run['actual_source_uv']) for run in reading.get('measured_runs',())]
    row['source_flags']=_flag_counts(reading.get('flags',{}))
    row['original_unsigned_profile_flag_counts']=_flag_counts(reading.get('original_unsigned_profile_flags') or {})
    return row


def compact_instances(packet):
    result = []
    for instance in packet.get("instances", ()):
        row = dict(instance)
        for collection in ('boundary_readings','source_print_raw_boundary_readings','source_print_role_conflict_readings'):
            row[collection]=[_compact_source_boundary_reading(reading) for reading in instance.get(collection,())]
        row["region_outline_is_source_measurement"] = False
        unlinked = instance.get("unlinked_source_foreground_candidate_mask")
        if unlinked is not None:
            yx = np.argwhere(_array(unlinked))
            row["unlinked_source_foreground_candidates"] = dict(source_pixel_count=len(yx),
                source_pixel_runs_y_x_length=_pixel_runs(yx[:, [1, 0]]),
                reason="source_foreground_not_linked_to_actual_matched_print_material_keypoints",
                rejection_is_pixel_accuracy_proof=False)
        result.append(row)
    return result


def current_source_pixel_exclusion(instance_packet, binding, *, native_hw, device):
    """Reuse this exposure's SAM person/graphic pixels, never detector rectangles."""
    import torch
    from .vision_geometry import _same_frame
    _same_frame(binding,instance_packet['source_binding'])
    mask=torch.zeros(tuple(native_hw),device=device,dtype=torch.bool)
    evidence=[]
    for instance in instance_packet.get('instances',()):
        if instance.get('kind') not in ('occluder','broadcast_overlay'):continue
        region=instance.get('region_mask')
        if region is None:continue
        _same_frame(binding,instance['source_binding'])
        region=torch.as_tensor(region,device=device)
        if tuple(region.shape)!=tuple(native_hw) or region.dtype!=torch.bool:
            raise ValueError('Source occlusion requires a boolean mask on the current native frame')
        mask|=region
        evidence.append(dict(object_id=instance.get('object_id'),kind=instance['kind'],pixel_count=int(region.sum())))
    reviewed=instance_packet.get('reviewed_source_exclusion_mask')
    if reviewed is not None:
        reviewed=torch.as_tensor(reviewed,device=device)
        if tuple(reviewed.shape)!=tuple(native_hw) or reviewed.dtype!=torch.bool:
            raise ValueError('Reviewed source exclusion differs from the current native frame')
        mask|=reviewed
    return mask,dict(source_binding=dict(binding),SAM_model_sha256=instance_packet.get('model_sha256'),
        regions=evidence,pixel_count=int(mask.sum()),person_detector_rectangles_used_as_exclusion=False,
        mask_is_pixel_ground_truth=False,human_acceptance_modified=False)


def reread_source_advertisements(rgb, instance_packet, provider, *, near_board=None):
    """Read printed contours from RGB without enabling unrelated SAM outlines."""
    from .vision_near_board import read_near_board, exclude_top_face_prints
    near = near_board if near_board is not None else read_near_board(rgb, instance_packet["source_binding"])
    from .vision_instances import same_frame
    if near_board is not None:same_frame(instance_packet["source_binding"], near["source_binding"])
    from .vision_instances import reread_region_edges
    cache_function=getattr(provider,'source_RGB_profile_cache',None)
    source_rgb_cache=cache_function(rgb,instance_packet['source_binding']) if cache_function else None
    exclusion=instance_packet.get('current_source_exclusion_mask',instance_packet.get('reviewed_source_exclusion_mask'))
    for instance in instance_packet.get("instances", ()):
        if instance["kind"] != "advertisement_panel" or provider.reread_rgb:
            continue
        contours = instance.get("printed_pattern_contours", ())
        readings = reread_region_edges(rgb, contours, instance_packet["source_binding"],
            device=provider.device, search_radius_px=provider.search_radius_px,
            printed_contour_search_radius_px=provider.printed_contour_search_radius_px,
            source_exclusion_mask=exclusion,source_rgb_cache=source_rgb_cache)
        count = sum(len(reading["actual_source_uv"]) for reading in readings)
        instance.update(boundary_readings=readings,
            boundary_reading_mode="printed_source_RGB_profile_read_no_type_probability_veto",
            status="printed_source_RGB_boundary_candidates" if count else "missing_printed_source_RGB_boundary",
            missing_reason=None if count else "matched_print_region_but_RGB_boundary_weak_ambiguous_or_occluded")
        conflicts = instance.get("source_print_role_conflict_contours", ())
        instance["source_print_role_conflict_readings"] = reread_region_edges(
            rgb, conflicts, instance_packet["source_binding"], device=provider.device,
            search_radius_px=provider.search_radius_px,
            printed_contour_search_radius_px=provider.printed_contour_search_radius_px,
            source_exclusion_mask=exclusion,source_rgb_cache=source_rgb_cache) if conflicts else []
        raw_contours=instance.get('source_print_unfiltered_contours',())
        instance['source_print_raw_boundary_readings']=reread_region_edges(
            rgb,raw_contours,instance_packet['source_binding'],device=provider.device,
            search_radius_px=provider.search_radius_px,
            printed_contour_search_radius_px=provider.printed_contour_search_radius_px,
            source_exclusion_mask=exclusion,source_rgb_cache=source_rgb_cache) if raw_contours else []
        instance['source_print_raw_reading_mode']='unfiltered_source_contour_search_remeasured_on_current_RGB'
        instance['source_print_unfiltered_contours_are_measured_UV']=False
    return exclude_top_face_prints(instance_packet, near)


def retain_same_source_refinement_evidence(previous, refined):
    """Keep this exposure's refresh ledger when SAM is refined with print prompts."""
    first,last=previous['source_binding'],refined['source_binding']
    if (first['source_sha256']!=last['source_sha256'] or first['source_pts']!=last['source_pts'] or
            Fraction(first['time_base'])!=Fraction(last['time_base'])):
        raise ValueError('Source prompt refinement cannot inherit another native exposure\'s evidence')
    result=dict(refined)
    for key,value in previous.items():
        if key not in result and key!='instances':
            result[key]=value
    result['initial_source_read_memory_update']=previous.get('initial_source_read_memory_update',previous.get('memory_update'))
    return result


def bootstrap_source_print_materials(rgb, instance_packet, top_role_packet, provider, *,
                                    person_boxes=(), overlay_boxes=()):
    """Navigate unnamed material searches from current measured rail/foot support.

    This refines the current SAM exposure, then remeasures original RGB. The
    measured top/foot points stay unchanged and every proposal remains separate
    from print identity, physical boundary verification and source measurement.
    """
    from .vision_instances import (source_print_material_search_proposals, same_frame,
                                  material_prompts_with_shared_board_search)
    binding=instance_packet['source_binding']
    navigation=top_role_packet.get('source_board_material_search_navigation')
    if navigation is None:
        return instance_packet,dict(source_binding=binding,
            status='missing_current_actual_top_foot_material_navigation',prompts=[],
            source_material_proposal_is_physical_identity_proof=False)
    same_frame(binding,navigation['source_binding'])
    if navigation.get('RGB_hard_veto_allowed') or not navigation.get('mask_is_navigation_only'):
        raise ValueError('Source material bootstrap requires navigation-only board support')
    parents=[instance for instance in instance_packet.get('instances',())
             if instance['kind']=='advertisement_panel' and not instance.get('source_material_proposal_unverified')]
    proposals=source_print_material_search_proposals(rgb,binding,print_parents=parents,
        board_search=navigation,person_boxes=person_boxes,overlay_boxes=overlay_boxes,
        first_object_id=300000,device='cpu')
    proposals['source_top_foot_role_packets_used_as_search_only']=True
    proposals['source_material_proposal_is_physical_identity_proof']=False
    before=provider.frame_index
    if proposals['prompts']:
        effective_navigation=proposals.get('effective_source_material_search_navigation',navigation)
        board=next((item for item in instance_packet.get('instances',())if item['kind']=='board_region'),None)
        shared=material_prompts_with_shared_board_search(rgb,binding,proposals['prompts'],
            board_search=effective_navigation,board_instance=board)
        proposals['shared_source_board_material_search']=shared
        refined=retain_same_source_refinement_evidence(instance_packet,
            provider.read(rgb,binding,prompts=shared['prompts'],reset=False,
                          material_search_navigation=effective_navigation))
        if provider.frame_index!=before:
            raise ValueError('Same-PTS material refinement advanced the native source exposure index')
        instance_packet=reread_source_advertisements(rgb,refined,provider)
    proposals['SAM_source_frame_index_before']=before
    proposals['SAM_source_frame_index_after']=provider.frame_index
    proposals['actual_source_RGB_remeasurement_after_prompt_refinement']=bool(proposals['prompts'])
    proposals['navigation_boxes_or_masks_used_as_measured_UV']=False
    return instance_packet,proposals


def source_advertisement_inventory(matcher, object_ids, instances, last_match_status):
    """Keep the appearance bank's fixed objects and missing reads in every frame."""
    by_object = {instance["object_id"]: instance for instance in instances}
    inventory = []
    for template in matcher.templates:
        key = template["template_id"]
        instance = by_object.get(object_ids.get(key))
        points = sum(len(reading["actual_source_uv"]) for reading in instance.get("boundary_readings", ())) if instance else 0
        inventory.append(dict(template_id=key, object_id=object_ids.get(key),
            reference_source_physical_id=template["reference_physical_id"],
            reference_identity_is_query_identity_truth=False,
            content_id=template["content_id"], source_component_id=template["source_component_id"],
            repeated_print_physical_candidates=[f"Ad{n}" for n in template["candidate_numbers"]],
            current_expected_id=instance.get("expected_id") if instance else None,
            current_physical_id=instance.get("physical_id") if instance else None,
            current_actual_source_RGB_boundary_point_count=points,
            current_source_region_present=bool(instance["region_mask"].any()) if instance else False,
            status=instance["status"] if instance else "reference_content_not_matched_visibility_or_reading_unresolved",
            missing_reason=instance.get("missing_reason") if instance else "no_current_source_content_correspondence",
            last_source_feature_match=last_match_status.get(key)))
    return inventory


def _run_points(runs):
    return np.asarray([[x, y] for y, first, count in runs
                       for x in range(first, first + count)], dtype=np.int64).reshape(-1, 2)


def source_board_top_roles(rgb, source_components, proposal, *, source_binding=None,
        foot_material_packet=None, source_instances=()):
    """Separate a source-supported board-top hypothesis from stray typed edges.

    The selected top/contact pair only supplies navigation. Its robust source
    trend is extended as a search hint, never as an output curve. Every primary
    UV below is an unchanged, individually RGB-read pixel. SAM envelopes and
    human evaluation coordinates are not inputs to this role assignment.
    """
    import cv2
    height, width = rgb.shape[:2]
    top_points = [ _array(component["source_support_uv"]).astype(np.int64)
                  for component in source_components if component["typed_class_candidate"] == "board_top"]
    actual = np.unique(np.concatenate(top_points), axis=0) if top_points else np.empty((0, 2), np.int64)
    if foot_material_packet is not None:
        from .vision_source_topology import (source_board_top_foot_role_pairs,
            read_source_board_rail_material_profiles, source_board_material_search_navigation)
        binding = source_binding or foot_material_packet.get('source_binding')
        source = dict(source_binding=binding, RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            components=[dict(component, candidate_kind=component.get('candidate_kind',
                'unclassified_source_edge_component')) for component in source_components])
        paired = source_board_top_foot_role_pairs(rgb, source, foot_material_packet,
            advertisement_instances=source_instances)
        material = read_source_board_rail_material_profiles(rgb, paired, source_instances=source_instances)
        navigation = source_board_material_search_navigation(material, foot_material_packet, native_hw=rgb.shape[:2])
        entry = material['actual_source_blue_rail_entry_uv']
        primary = entry[material['primary_role_samples']]
        return dict(status='source_RGB_upper_rail_entry_and_internal_exit_separated' if len(entry) else
                paired['status'], selected_pair=None, separate_source_top_foot_pair_evidence=paired,
            actual_source_rail_material_reading=material, source_board_material_search_navigation=navigation,
            primary_source_top=dict(source_pixel_count=len(primary), actual_source_uv=primary,
                source_pixel_runs_y_x_length=[], physical_role='source_RGB_board_upper_rim_material_entry_hypothesis',
                role_accuracy_accepted=False),
            unresolved_typed_edges=dict(source_pixel_count=len(actual), source_pixel_runs_y_x_length=_pixel_runs(actual),
                physical_role=None, reason='original_typed_edges_retained_separately_from_fresh_RGB_material_roles'),
            physical_structure_accuracy_verified=False, SAM_region_outline_used_as_truth=False,
            human_test_annotations_used=False, primary_points_are_existing_actual_RGB_readings=False,
            fresh_original_RGB_profile_measurement_performed=True, navigation_fit_pixels_used_as_output=False,
            blue_to_white_exit_role='board_internal_material_interface_not_board_upper_rim',
            source_person_candidates_are_competing_role_evidence_not_RGB_veto=True,
            all_original_typed_board_top_pixels_retained=True)
    pairs = (proposal or {}).get("board_contact_pair_hypotheses", ())
    preferred = [pair for pair in pairs if pair["top_hypothesis"].get("preferred_region_search_hypothesis")]
    selected = max(preferred, key=lambda pair: pair["candidate_pair_score"]) if preferred else None
    primary = np.zeros(len(actual), bool)
    metadata = dict(status="missing_same_contact_board_source_pair", selected_pair=None,
                    physical_structure_accuracy_verified=False,
                    SAM_region_outline_used_as_truth=False, human_test_annotations_used=False,
                    primary_points_are_existing_actual_RGB_readings=True,
                    navigation_fit_pixels_used_as_output=False)
    if selected and len(actual):
        top = selected["top_hypothesis"]
        seed = _run_points(top.get("source_gradient_pixel_runs_y_x_length", ()))
        if not len(seed):
            # Compatibility for saved source observations predating the exact
            # proposal ledger. This uses only their actual read pixels.
            x0, y0, x1, y1 = top["source_bbox_xyxy"]
            seed = actual[(actual[:, 0] >= x0) & (actual[:, 0] < x1) &
                          (actual[:, 1] >= y0) & (actual[:, 1] < y1)]
        if len(seed) and selected.get("board_print_source_gradient_pixels", 0) > 0:
            columns = np.unique(seed[:, 0])
            center = np.asarray([np.median(seed[seed[:, 0] == x, 1]) for x in columns])
            design = np.column_stack((columns / max(1, width-1), np.ones(len(columns))))
            keep = np.ones(len(columns), bool)
            for _ in range(4):
                if keep.sum() < 2:
                    break
                fit = np.linalg.lstsq(design[keep], center[keep], rcond=None)[0]
                residual = np.abs(center-design @ fit)
                keep = residual <= max(2., 3.*float(np.median(residual[keep])))
            if keep.sum() >= 2:
                fit = np.linalg.lstsq(design[keep], center[keep], rcond=None)[0]
                spread = float(np.quantile(np.abs(center[keep]-design[keep] @ fit), .95))
                radius = min(12., max(4., spread+2.))
                trend = actual[:, 0]/max(1, width-1)*fit[0]+fit[1]
                gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
                x, y = actual.T
                # Rail-like transverse RGB support, not vertical spectators,
                # letters or equipment edges that happen to receive top logits.
                transverse = np.abs(gy[y, x]) >= np.abs(gx[y, x])*.7
                # A long rail may be fragmented by people or print, so the
                # preferred bootstrap component can cover only one side. Read
                # the same nearby RGB seam across the image before deciding
                # its role. This refines navigation using real support; it
                # never creates a pixel across an empty or occluded interval.
                nearby = (np.abs(y-trend) <= 12.) & transverse
                extension = actual[nearby]
                extension_columns = np.unique(extension[:, 0])
                if len(extension_columns) > len(columns[keep]):
                    extension_y = np.asarray([np.median(extension[extension[:, 0] == column, 1])
                                              for column in extension_columns])
                    extension_design = np.column_stack((extension_columns/max(1, width-1),
                                                        np.ones(len(extension_columns))))
                    extension_keep = np.ones(len(extension_columns), bool)
                    for _ in range(4):
                        if extension_keep.sum() < 2:
                            break
                        refined = np.linalg.lstsq(extension_design[extension_keep],
                                                 extension_y[extension_keep], rcond=None)[0]
                        residual = np.abs(extension_y-extension_design @ refined)
                        extension_keep = residual <= max(2., 3.*float(np.median(residual[extension_keep])))
                    if extension_keep.sum() >= 2:
                        fit = np.linalg.lstsq(extension_design[extension_keep],
                                             extension_y[extension_keep], rcond=None)[0]
                        spread = float(np.quantile(np.abs(extension_y[extension_keep]-
                                                         extension_design[extension_keep] @ fit), .95))
                        radius = min(12., max(4., spread+2.))
                        trend = actual[:, 0]/max(1, width-1)*fit[0]+fit[1]
                        metadata["source_RGB_seam_navigation_refined"] = True
                        metadata["source_RGB_seam_refinement_point_count"] = len(extension)
                primary = (np.abs(y-trend) <= radius) & transverse
                metadata.update(status="same_contact_board_pair_consistent_source_top_hypothesis",
                    selected_pair=selected, source_navigation_seed_point_count=len(seed),
                    navigation_fit_y_equals_slope_x_plus_intercept=[float(fit[0]/max(1, width-1)), float(fit[1])],
                    navigation_fit_source_x_range=[int(columns[keep].min()), int(columns[keep].max())],
                    navigation_fit_extension_is_search_only=True, search_radius_native_px=radius,
                    role_assigned_by="same_source_long_top_contact_pair_print_parent_and_transverse_RGB_support",
                    physical_identity_verified=False)
    return dict(**metadata,
        primary_source_top=dict(source_pixel_count=int(primary.sum()),
            source_pixel_runs_y_x_length=_pixel_runs(actual[primary]),
            physical_role="same_contact_board_structure_consistent_top",
            role_accuracy_accepted=False),
        unresolved_typed_edges=dict(source_pixel_count=int((~primary).sum()),
            source_pixel_runs_y_x_length=_pixel_runs(actual[~primary]),
            physical_role=None, reason="typed_edge_outside_source_board_pair_search_or_structure_unresolved"),
        all_original_typed_board_top_pixels_retained=True)


def source_region_constraints(instance_packet, *, native_hw, device, radius_px=12):
    """Candidate source topology narrows edge searches, with missing roles explicit.

    The bands below are search masks.  No mask boundary becomes an observation,
    and rejected RGB support must remain available in the reader's audit ledger.
    """
    import torch
    from torch.nn import functional as F
    h, w = native_hw
    masks, unverified_exclusions = {}, []
    for instance in instance_packet.get("instances", ()):
        mask = instance.get("region_mask")
        if mask is None:
            continue
        mask = torch.as_tensor(mask, device=device).bool()
        if tuple(mask.shape) != (h, w):
            raise ValueError("Source region and actual original image domains differ")
        kind = instance["kind"]
        if kind in ("occluder", "broadcast_overlay"):
            unverified_exclusions.append(dict(object_id=instance.get("object_id"), kind=kind,
                source_candidate_pixel_count=int(mask.sum()),
                reason="current_SAM_pixel_exclusion_applied_by_reader" if instance_packet.get("current_source_pixel_exclusion_evidence") else
                       "candidate_exclusion_not_used_without_provider_source_review_authority"))
            continue
        masks[kind] = masks.get(kind, torch.zeros_like(mask)) | mask

    def dilate(mask):
        return F.max_pool2d(mask.float()[None, None], 2 * radius_px + 1,
                            stride=1, padding=radius_px)[0, 0] > .5

    def band(mask):
        return dilate(mask) & dilate(~mask)

    ice, board = masks.get("ice_region"), masks.get("board_region")
    exclusion = torch.zeros((h, w), device=device, dtype=torch.bool)
    reviewed_exclusion = instance_packet.get("current_source_exclusion_mask",instance_packet.get("reviewed_source_exclusion_mask"))
    if reviewed_exclusion is not None:
        exclusion = torch.as_tensor(reviewed_exclusion, device=device).bool()
        if tuple(exclusion.shape) != (h, w):
            raise ValueError("Reviewed source exclusion and native image domains differ")
    roles, missing, soft_roles = {}, [], {}
    proposal = instance_packet.get("bootstrap_source_proposals") or {}
    same_board_verified = instance_packet.get("same_board_structure_verified") is True or proposal.get("same_board_structure_verified") is True
    if ice is not None and bool(ice.any()):
        roles["ice_paint"] = ice & ~exclusion
        if board is not None and bool(board.any()):
            if same_board_verified:
                roles["ice_contact"] = band(ice) & dilate(board) & ~exclusion
            else:
                soft_roles["ice_contact"] = band(ice) & ~exclusion
                missing.append("ice_contact_board_structure_unverified_RAW_RGB_candidates_preserved")
        else:
            soft_roles["ice_contact"] = band(ice) & ~exclusion
            missing.append("ice_contact_requires_visible_board_structure_RAW_RGB_candidates_preserved")
    else:
        missing.extend(("ice_contact_region_absent", "ice_paint_region_absent"))
    if board is not None and bool(board.any()):
        if same_board_verified:
            roles["ad_boundary"] = board & ~exclusion
        else:
            soft_roles["ad_boundary"] = board & ~exclusion
            missing.append("ad_boundary_board_region_is_search_candidate_not_verified_structure")
        if ice is not None and bool(ice.any()):
            yy = torch.arange(h, device=device)[:, None].expand(h, w)
            top = torch.where(board, yy, h).amin(dim=0)
            # Internal holes/panel boundaries are not the board's upper side.
            # This envelope only declares where to inspect original RGB; its
            # integer samples never enter a measurement or training target.
            upper_search = (top[None] < h) & ((yy - top[None]).abs() <= radius_px)
            top_search = upper_search & ~dilate(ice) & ~exclusion
            if same_board_verified:
                roles["board_top"] = top_search
            else:
                soft_roles["board_top"] = top_search
        else:
            missing.append("board_top_region_needs_contact_side_disambiguation")
    else:
        missing.extend(("board_top_region_absent", "ad_boundary_region_absent"))
    counts = {name: int(mask.sum()) for name, mask in roles.items()}
    metadata = dict(missing_roles=missing, source_role_search_pixel_counts=counts,
        unverified_exclusion_candidates=unverified_exclusions,
        source_role_status={name: "source_region_search_scope_available" if name in roles else
                           "missing_source_region_kept_explicit"
                           for name in ("ice_contact", "board_top", "ice_paint", "ad_boundary")},
        source_region_constraints_are_search_scope_only=True, radius_native_px=radius_px)
    metadata["same_contact_board_structure_verified"] = same_board_verified
    metadata["unverified_top_envelope_hard_veto_disabled"] = not same_board_verified
    metadata["unverified_board_region_hard_veto_disabled"] = not same_board_verified
    metadata["model_prompt_region_name_establishes_physical_structure"] = False
    metadata["soft_role_search_pixel_counts"] = {name: int(mask.sum()) for name, mask in soft_roles.items()}
    for name in soft_roles:
        metadata["source_role_status"][name] = "RGB_type_evidence_preserved_region_scope_unresolved"
    metadata["board_top_search_rule"] = "upper_source_region_envelope_search_band_internal_mask_holes_not_top"
    original_reviewed=instance_packet.get("reviewed_source_exclusion_mask")
    metadata["reviewed_exclusion_source_pixel_count"] = 0 if original_reviewed is None else int(torch.as_tensor(original_reviewed).sum())
    metadata["current_source_exclusion_pixel_count"] = int(exclusion.sum())
    metadata["current_source_pixel_exclusion_evidence"] = instance_packet.get("current_source_pixel_exclusion_evidence")
    return dict(source_binding=instance_packet.get("source_binding"),
                masks={name: mask[None, None] for name, mask in roles.items()},
                soft_search_masks={name: mask[None, None] for name, mask in soft_roles.items()},
                missing_roles=missing, source_role_search_pixel_counts=counts,
                source_region_constraints_are_search_scope_only=True,
                source_region_outline_used_as_measurement=False,
                world_or_camera_projection_used=False, radius_native_px=radius_px,
                source_role_status={name: "source_region_search_scope_available" if name in roles else
                                    "missing_source_region_kept_explicit"
                                    for name in ("ice_contact", "board_top", "ice_paint", "ad_boundary")},
                metadata=metadata, world_or_camera_used=False, accuracy_accepted=False)


def draw_review(rgb, result, diagnostics, instances=()):
    """Draw measured pixels only; anonymous support remains visibly unnamed."""
    import cv2
    canvas = rgb.copy()
    for component in result["source"]["components"]:
        if component["typed_class_candidate"] in ("ice_paint", "ad_boundary"):
            uv = _array(component["source_support_uv"]).astype(np.int64)
            canvas[uv[:, 1], uv[:, 0]] = ((220, 55, 205) if component["typed_class_candidate"] == "ice_paint"
                                        else (145, 110, 155) if any(instance.get("kind") == "advertisement_panel" for instance in instances)
                                        else (35, 215, 255))
    display_result = dict(result)
    foot=diagnostics.get('source_board_foot_material_reading')
    structure = diagnostics.get("source_board_top_role_assignment")
    if structure:
        # Domain conflicts remain visible with their own unresolved color; they
        # must not be rendered as the physical board's yellow top.
        unresolved = _run_points(structure["unresolved_typed_edges"]["source_pixel_runs_y_x_length"])
        if len(unresolved):
            canvas[unresolved[:, 1], unresolved[:, 0]] = (145, 110, 155)
        display_result["source"] = dict(result["source"], components=[component
            for component in result["source"]["components"]
            if component["typed_class_candidate"] != "board_top"])
        primary = _run_points(structure["primary_source_top"]["source_pixel_runs_y_x_length"])
        if len(primary):
            canvas[primary[:, 1], primary[:, 0]] = (255, 235, 40)
        material = structure.get('actual_source_rail_material_reading')
        if material:
            points = _array(material['actual_source_blue_rail_entry_uv'])
            primary = _array(material['primary_role_samples']).astype(bool)
            for point, accepted in zip(points, primary):
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 1,
                    (255,235,40) if accepted else (145,110,155), -1)
            # The observed lower blue/white side is an internal board interface,
            # so it never receives the yellow board-top role.
            for point in _array(material['actual_source_blue_to_white_exit_uv']):
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 1, (170,150,210), -1)
    if foot:
        display_result['source']=dict(display_result.get('source',result['source']),components=[component
            for component in display_result.get('source',result['source'])['components']
            if component['typed_class_candidate']!='ice_contact'])
    display_result["guided"] = dict(result["guided"])
    display_readings = []
    for reading in result["guided"]["readings"]:
        shown = dict(reading)
        if not shown.get("physical_id"):
            shown["display_label"] = str(shown.get("display_label") or shown.get("expected_id")) + "?"
        display_readings.append(shown)
    display_result["guided"]["readings"] = display_readings
    canvas = SourceRinkReader.draw(canvas, display_result)
    if foot:
        for reading in foot.get('readings',()):
            unresolved=_array(reading.get('unresolved_source_uv',np.empty((0,2)))).reshape(-1,2)
            for point in unresolved[np.isfinite(unresolved).all(1)]:
                xy=tuple(np.rint(point).astype(int))
                cv2.circle(canvas,xy,1,(110,110,110),-1)
            points=_array(reading.get('primary_source_uv',np.empty((0,2)))).reshape(-1,2)
            color=(255,90,0) if reading['physical_role'].startswith('ice_contact') else (210,190,255)
            for point in points:
                cv2.circle(canvas,tuple(np.rint(point).astype(int)),2,color,-1)
    for instance in instances:
        unverified_material=bool(instance.get('source_material_proposal_unverified'))
        for conflict in instance.get("source_print_role_conflict_readings", ()):
            for run in conflict.get("measured_runs", ()):
                points = _array(run["actual_source_uv"])
                if len(points) > 1:
                    cv2.polylines(canvas, [np.rint(points*256).astype(np.int32)], False,
                                  (145, 110, 155), 1, cv2.LINE_AA, shift=8)
        anchor = None
        for reading in instance.get("boundary_readings", ()):
            for run in reading.get("measured_runs", ()):
                points = _array(run["actual_source_uv"])
                if not len(points):
                    continue
                anchor = points[len(points) // 2]
                color = (35, 215, 255) if instance.get("kind") == "advertisement_panel" else (90, 240, 90)
                if reading.get("internal_hole"):
                    color = (220, 55, 205)
                if unverified_material:
                    color = (170,125,165) if reading.get('internal_hole') else (135,165,100)
                if len(points) > 1:
                    cv2.polylines(canvas, [np.rint(points * 256).astype(np.int32)], False, color, 2,
                                  cv2.LINE_AA, shift=8)
                else:
                    cv2.circle(canvas, tuple(np.rint(points[0]).astype(int)), 2, color, -1)
        if anchor is not None:
            label = str(instance.get("physical_id") or
                        (str(instance["expected_id"])+"?" if instance.get("expected_id") else
                         f"material?#{instance['object_id']}" if unverified_material else
                         f"unknown#{instance['object_id']}"))
            location = tuple(np.rint(anchor).astype(int))
            cv2.putText(canvas, label, location, cv2.FONT_HERSHEY_SIMPLEX, .5, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, location, cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (135,165,100) if unverified_material else (35,215,255), 1, cv2.LINE_AA)
    binding = result["source_binding"]
    banner = np.zeros((66, rgb.shape[1], 3), np.uint8)
    stamp = format_source_time(binding["source_pts"], binding["time_base"])
    text = f"source {stamp}   PTS {binding['source_pts']}   shot {binding['shot_id']}"
    cv2.putText(banner, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
    text = (f"RGB: yellow=board-pair top, muted purple=unresolved typed edges, pink=paint candidates, cyan=print readings | issues {len(diagnostics['issues'])}"
            " | no grid | accuracy not accepted")
    if foot:
        text=f"RGB: yellow=upper?, orange=foot lower?, pale purple=internal | cyan=reference print?, muted green=material? | issues {len(diagnostics['issues'])} | accuracy not accepted"
    cv2.putText(banner, text, (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 185, 80), 1, cv2.LINE_AA)
    return np.concatenate((canvas, banner), axis=0)


def redraw_saved_source(readings_path, source_image_path, source_pts, output_path):
    """Pure CPU redraw of saved actual observations, with exact original RGB proof."""
    import torch
    from PIL import Image
    row=load_saved_source_record(readings_path,source_pts)
    rgb = np.asarray(Image.open(source_image_path).convert("RGB"))
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["RGB_sha256"]:
        raise ValueError("Redraw source PNG differs from the original decoded RGB")
    if row.get('schema')=='rinkspace.application-source-observations.v1':
        image=draw_application_record(rgb,row)
        output_path=Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(output_path)
        return dict(source_pts=source_pts,source_RGB_sha256=row['RGB_sha256'],output=str(output_path),
            actual_observations_reused=True,inference_rerun=False,model_sha256=row['model_sha256'])
    components = []
    for component in row["source_components"]:
        points = [[x, y] for y, start, length in component["source_pixel_runs_y_x_length"]
                  for x in range(start, start + length)]
        components.append(dict(typed_class_candidate=component["kind"],
                               source_support_uv=torch.tensor(points, dtype=torch.float32).reshape(-1, 2)))
    readings = []
    for reading in row["named_readings"]:
        readings.append(dict(reading,
            actual_source_uv=torch.tensor(reading["actual_source_uv"], dtype=torch.float32).reshape(-1, 2),
            measured_runs=[dict(actual_source_uv=torch.tensor(run, dtype=torch.float32).reshape(-1, 2))
                           for run in reading["measured_runs"]]))
    instances = []
    for instance in row.get("instances", ()):
        restored=dict(instance)
        for collection in ('boundary_readings','source_print_raw_boundary_readings','source_print_role_conflict_readings'):
            restored[collection]=[dict(reading,measured_runs=[
                dict(run,actual_source_uv=_array(run['actual_source_uv']).reshape(-1,2)) if isinstance(run,dict)
                else dict(actual_source_uv=_array(run).reshape(-1,2))
                for run in reading.get('source_measured_runs',reading.get('measured_runs',()))])
                for reading in instance.get(collection,())]
        instances.append(restored)
    result = dict(source_binding=row["source_binding"], source=dict(components=components),
                  guided=dict(readings=readings))
    if "source_board_top_role_assignment" not in row:
        row["source_board_top_role_assignment"] = source_board_top_roles(
            rgb, components, row.get("automatic_source_region_proposal"))
    image = draw_review(rgb, result, row, instances)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return dict(source_pts=source_pts, source_RGB_sha256=row["RGB_sha256"], output=str(output_path),
                actual_observations_reused=True, inference_rerun=False, model_sha256=row["model_sha256"])


def compare_confirmed_source(readings_paths, sample):
    """Compare actual edge coverage near unchanged human source marks on CPU.

    Native 2/4/8px bands are descriptive measurements, not acceptance limits.
    A hand mark's width/precision and a candidate edge's identity remain separate
    from this coverage test.  It cannot establish complete visual accuracy.
    """
    pts, digest = int(sample["source_pts"]), sample["source_hash"]
    models = []
    expected_rgb = None
    for path in readings_paths:
        row=load_saved_source_record(path,pts,expected_source_binding=dict(
            source_sha256=digest,source_pts=pts,time_base=sample['time_base']))
        if row["source_binding"]["source_sha256"] != digest:
            raise ValueError("Coverage comparison requires the exact human source video and PTS")
        if Fraction(row["source_binding"]["time_base"]) != Fraction(sample["time_base"]):
            raise ValueError("Coverage comparison clocks differ")
        if expected_rgb is not None and row["RGB_sha256"] != expected_rgb:
            raise ValueError("Compared models decoded different source pixels")
        expected_rgb = row["RGB_sha256"]
        support = {}
        for component in row["source_components"]:
            support.setdefault(component["kind"], []).extend(
                [x, y] for y, first, count in component["source_pixel_runs_y_x_length"]
                for x in range(first, first + count))
        support = {kind: np.asarray(points, dtype=float) for kind, points in support.items() if points}
        primary_top = row.get("source_board_top_role_assignment", {}).get("primary_source_top")
        primary_top_points = None
        if primary_top:
            primary_top_points = (_array(primary_top["actual_source_uv"]).reshape(-1, 2)
                if "actual_source_uv" in primary_top else _run_points(primary_top["source_pixel_runs_y_x_length"]))

        def nearest_distance(uv, points):
            values = []
            for start in range(0, len(uv), 128):
                batch = uv[start:start+128]
                squared = np.full(len(batch), np.inf)
                for offset in range(0, len(points), 4096):
                    candidates = points[offset:offset+4096]
                    difference = batch[:, None] - candidates[None]
                    squared = np.minimum(squared, np.square(difference).sum(axis=-1).min(axis=1))
                values.extend(np.sqrt(squared))
            return np.asarray(values)
        groups = []
        for annotation in sample["annotations"]:
            if annotation.get("state") != "user_confirmed_visual_supervision":
                continue
            uv = np.asarray(annotation["uv"], dtype=float)
            if not len(uv):
                continue
            distances = {kind: nearest_distance(uv, points) for kind, points in support.items()}
            correct = distances.get(annotation["kind"], np.full(len(uv), np.inf))
            others = [distance for kind, distance in distances.items() if kind != annotation["kind"]]
            wrong = np.minimum.reduce(others) if others else np.full(len(uv), np.inf)
            primary_distance = (nearest_distance(uv, primary_top_points) if annotation["kind"] == "board_top"
                                and primary_top_points is not None and len(primary_top_points) else None)
            groups.append(dict(feature_id=annotation["feature_id"], group_id=annotation["group_id"],
                expected_kind=annotation["kind"], human_mark_point_count=len(uv), authority=annotation["authority"],
                correct_class_distance_px_p50=float(np.median(correct)) if np.isfinite(correct).all() else None,
                correct_class_distance_px_p95=float(np.quantile(correct, .95)) if np.isfinite(correct).all() else None,
                correct_class_coverage_by_native_px={str(radius): float((correct <= radius).mean()) for radius in (2, 4, 8)},
                any_wrong_class_coverage_by_native_px={str(radius): float((wrong <= radius).mean()) for radius in (2, 4, 8)},
                correct_missing_and_wrong_present_at_4px_count=int(((correct > 4) & (wrong <= 4)).sum()),
                correct_and_wrong_classes_overlap_at_4px_count=int(((correct <= 4) & (wrong <= 4)).sum()),
                primary_same_board_structure_top_coverage_by_native_px={str(radius): float((primary_distance <= radius).mean())
                    for radius in (2, 4, 8)} if primary_distance is not None else None,
                primary_same_board_structure_top_distance_px_p50=float(np.median(primary_distance))
                    if primary_distance is not None else None,
                original_human_quality_metadata=annotation.get("original_quality")))
        models.append(dict(model_sha256=row["model_sha256"], readings_path=str(Path(path).resolve()), groups=groups,
            source_region_rejected_pixels={kind: item["source_pixel_count"] for kind, item in row["source_region_rejections"].items()},
            source_quality_rejected_pixels={kind: item["source_pixel_count"] for kind, item in row.get("source_quality_rejections", {}).items()}))
    return dict(schema="rinkspace.human-source-edge-coverage-comparison.v1", source_pts=pts,
        source_time=format_source_time(pts, sample["time_base"]), source_sha256=digest,
        exact_same_decoded_RGB_sha256=expected_rgb, models=models, test_annotations_injected=False,
        distance_is_to_human_source_mark=True, distance_is_verified_boundary_error=False,
        band_widths_are_descriptive_not_acceptance_thresholds=True, full_visual_accuracy_accepted=False)


def _source_seeds(path, source_sha256):
    if path is None:
        return {}
    packet = read_json(path)
    rows = packet.get("frames", packet.get("samples", ()))
    seeds = {}
    for row in rows:
        binding = row.get("source_binding", {})
        digest = binding.get("source_sha256", row.get("source_hash"))
        if digest != source_sha256:
            raise ValueError("A source search seed belongs to another video")
        pts = int(binding.get("source_pts", row.get("source_pts")))
        if pts in seeds:
            raise ValueError("Duplicate source search seed PTS")
        seeds[pts] = row
    return seeds


def _bind_seed(seed, binding):
    """Join exact source pixels to the existing classified shot, retaining history."""
    original = dict(seed.get("source_binding", {}))
    original.setdefault("source_sha256", seed.get("source_hash"))
    original.setdefault("source_pts", seed.get("source_pts"))
    original.setdefault("time_base", seed.get("time_base"))
    for key in ("source_sha256", "source_pts"):
        if original.get(key) != binding[key]:
            raise ValueError("Source seed and decoded original frame differ")
    if Fraction(str(original.get("time_base"))) != Fraction(binding["time_base"]):
        raise ValueError("Source seed and decoded frame clocks differ")
    dimensions = original.get("native_hw")
    if dimensions is None and seed.get("frame_size_wh"):
        dimensions = list(reversed(seed["frame_size_wh"]))
    if dimensions is not None and list(dimensions) != binding["native_hw"]:
        raise ValueError("Source seed canvas differs from the original frame")
    result = dict(seed, original_source_annotation_binding=original)
    for name in ("annotations", "prompts"):
        rebound = []
        for entry in seed.get(name, ()):
            record = dict(entry)
            old = record.get("source_binding")
            if old:
                for key in ("source_sha256", "source_pts"):
                    if old.get(key) != binding[key]:
                        raise ValueError("A source seed child belongs to another original frame")
                if Fraction(old["time_base"]) != Fraction(binding["time_base"]):
                    raise ValueError("A source seed child uses another original clock")
                record["original_source_annotation_binding"] = old
            record["source_binding"] = dict(binding)
            rebound.append(record)
        result[name] = rebound
    return result


def review(config_path, shots_path, output_directory, *, video=None, shot_ids=(),
           mode="independent", seed_path=None, instance_checkpoint=None,
           snapshot_pts=(), issue_image_count=6, render_video=False,
           diagnostic_start_pts=None, diagnostic_end_pts=None,
           automatic_regions=True, person_weights=None, person_vendor=None, model_path=None,
           model_search_root=None, paint_role_contract=None, evaluation_role="development",
           advertisement_source_paths=(), event_id=None, boundary_overlay_path=None,
           profile_stage_times=False, profile_cuda_sync=False,
           board_foot_material_contract=None,person_dependencies=(),source_material_proposals=False,
           source_array_storage_layout='byte_pool',output_mode='evidence',application_keyframe_interval_seconds=None,
           original_fit_request_path=None):
    """Process every original frame; full videos are an explicit caller choice.

    Generate a complete video only after real source evidence shows an actual
    model improvement.  Diagnostic images and complete numerical coverage do
    not themselves establish visual accuracy or complete autonomous capability.
    """
    import av
    from PIL import Image
    output_directory = Path(output_directory).resolve()
    if output_mode not in ('evidence','application'):
        raise ValueError('Output mode must explicitly choose evidence or application observations')
    if application_keyframe_interval_seconds is not None and (output_mode!='application' or application_keyframe_interval_seconds<=0):
        raise ValueError('Sparse reidentification requires application output and a positive source-time interval')
    allowed = Path("D:/CodexData/RinkSpace_Data").resolve()
    if not output_directory.is_relative_to(allowed):
        raise ValueError("Continuous review outputs must stay in the RinkSpace Data tree")
    if mode not in ("independent", "assisted"):
        raise ValueError("Unknown review mode")
    if evaluation_role not in ("development", "held_out"):
        raise ValueError("Unknown evaluation role")
    if source_array_storage_layout not in ('individual_arrays','byte_pool'):
        raise ValueError('Unknown lossless source array storage layout')
    if mode == "independent" and seed_path:
        raise ValueError("Independent evaluation cannot inject source test-frame annotations")
    reader = SourceRinkReader(config_path, model_path=model_path, original_fit_request_path=original_fit_request_path)
    material_authority=None
    if board_foot_material_contract:
        from .vision_source_topology import board_foot_material_authority
        material_authority=board_foot_material_authority(board_foot_material_contract)
    video = Path(video or reader.config["source_video"]).resolve()
    digest = sha256(video)
    shots = source_shots(shots_path, digest, shot_ids=shot_ids, boundary_overlay_path=boundary_overlay_path)
    partial = diagnostic_start_pts is not None or diagnostic_end_pts is not None
    if partial:
        shots = clip_diagnostic_shots(shots, diagnostic_start_pts, diagnostic_end_pts)
    seeds = _source_seeds(seed_path, digest)
    provider = person_detector = search_adapter = advertisement_matcher = None
    if model_search_root:
        from .vision_instances import SceneGuideSearchAdapter
        if paint_role_contract is None:
            paint_role_contract = reader.data / "paint_role_contract_current.json"
        search_adapter = SceneGuideSearchAdapter.from_okada_review(model_search_root,
                                                                  role_contract_path=paint_role_contract)
    if instance_checkpoint:
        from .vision_instances import Sam2SourceInstances, SourcePersonBoxes
        provider = Sam2SourceInstances(instance_checkpoint, device=reader.config["device"],
                                      reread_rgb=mode == "assisted")
        if person_weights:
            person_detector = SourcePersonBoxes(person_weights, vendor_path=person_vendor,
                dependency_paths=person_dependencies,runtime_directory=reader.data/'vision_instance_yolo',
                device=reader.config["device"])
    if advertisement_source_paths:
        if not provider or not event_id:
            raise ValueError("Source advertisement matching needs the SAM instance provider and an explicit event_id")
        from .vision_instances import SourceAdvertisementMatcher
        advertisement_matcher = SourceAdvertisementMatcher(advertisement_source_paths, event_id=event_id,
            cache_directory=reader.data / "advertisement_feature_bank")
    if source_material_proposals and (provider is None or material_authority is None):
        raise ValueError('Source material proposals require SAM and current actual top/foot material readings')
    output_directory.mkdir(parents=True, exist_ok=True)
    rows_path = output_directory / "readings.jsonl.gz"
    summary_path = output_directory / "review.json"
    requested_snapshots = set(map(int, snapshot_pts))
    training_scope = reader.vision.training_scope
    training_pts = set(map(int, training_scope.get("training_source_pts", ())))
    source_is_training = training_scope.get("source_video_sha256") == digest
    summary = dict(schema="rinkspace.native-continuous-vision-review.v1", status="running", mode=mode,
        evaluation_role=evaluation_role, complete_shot_used_for_diagnosis=evaluation_role == "development",
        source_video=str(video), source_sha256=digest, source_video_sha256_verified=True,
        source_code_sha256_at_run_start={name: sha256(ROOT / "src/rinkspace" / name) for name in
            ("vision.py", "vision_network.py", "vision_geometry.py", "vision_review.py", "vision_instances.py",
             "vision_storage.py", "vision_source_shapes.py", "camera_context_reuse.py", "camera_reuse_localization.py", "camera_classification.py")
            + (("vision_source_fit.py",) if reader.original_fit_reader is not None else ())
            + (("vision_source_topology.py",) if material_authority else ())},
        classification_path=str(Path(shots_path).resolve()), classification_sha256=sha256(shots_path),
        source_classification_boundary_overlay=shots[0].get("source_boundary_overlay"),
        model_path=str(Path(model_path).resolve() if model_path else ROOT / reader.config["final_model"]),
        model_sha256=reader.vision.model_sha256,
        model_usage=reader.vision.model_usage(),
        original_FIT_request_reference=reader.original_fit_reader.request_reference if reader.original_fit_reader is not None else None,
        training_step=training_scope.get("latest_step"), frame_count=0, source_pts_complete=False,
        all_decoded_original_frames_processed=True, snapshots=[], shots=[], issue_counts={},
        source_clock_evidence=[],
        prediction_is_pixel_truth=False, accuracy_accepted=False,
        actual_RGB_is_only_output_geometry=True, grid_shown=False,
        world_or_camera_optimized=False, independent_mode_has_no_test_annotation_injection=mode == "independent",
        output_rows=str(rows_path), complete_video_requested=render_video,
        readings_storage=dict(format="gzip_UTF8_JSONL", gzip_compresslevel=1,
            newline="LF",
            gzip_member_scope='one_complete_native_source_record',
            committed_frame_progress_path=str(rows_path.with_name('readings_committed_frames.jsonl')),
            concurrent_record_checks_without_array_copy=True,
            large_array_storage='source_bound_lossless_NPZ',large_array_minimum_bytes=1,
            NPZ_array_storage_layout=source_array_storage_layout,
            typed_array_dtype_shape_and_empty_arrays_preserved=True,
            sidecar_directory='beside_readings_JSONL',sidecar_count=0,sidecar_compressed_bytes=0,
            original_array_dtype_shape_and_nonfinite_values_preserved=True,
            uncompressed_UTF8_bytes=0, uncompressed_UTF8_sha256=None,
            complete_candidate_missing_and_conflict_ledger_retained=True),
        coverage_scope="declared_diagnostic_subinterval" if partial else "complete_classified_shots",
        source_seed_path=str(Path(seed_path).resolve()) if seed_path else None,
        source_seed_sha256=sha256(seed_path) if seed_path else None,
        automatic_source_region_initialization=bool(provider and automatic_regions),
        event_id=event_id, source_advertisement_matching_enabled=bool(advertisement_matcher),
        advertisement_appearance_source_packets=[dict(path=str(Path(path).resolve()), sha256=sha256(path))
                                                 for path in advertisement_source_paths],
        advertisement_query_pixels_are_not_reference_human_labels=True,
        person_source_model_sha256=person_detector.weights_sha256 if person_detector else None,
        existing_model_search_used=bool(search_adapter),
        existing_model_search_scope="exact existing Okada source camera only; outside its PTS range remains missing" if search_adapter else None,
        limitation="semantic edge candidates do not establish physical marker/ad IDs; source masks are search aids",
        full_shot_pixel_accuracy_validated=False)
    summary['source_print_material_proposals_enabled']=bool(source_material_proposals)
    summary['output_mode']=output_mode
    summary['intermediate_profiles_masks_and_raw_pools_saved']=output_mode=='evidence'
    summary['application_keyframe_interval_source_seconds']=application_keyframe_interval_seconds
    summary['intermediate_tracking_method']='native_RAFT_search_then_current_RGB_measurement_no_SAM_or_full_components' if application_keyframe_interval_seconds else None
    summary['source_print_material_proposals_schedule']='shot_reset_shared_board_then_current_native_navigation_and_RGB_every_frame' if source_material_proposals else 'disabled'
    summary['source_board_foot_material_authority']=material_authority
    summary['source_board_foot_annotation_pixels_used_as_runtime_input']=False
    profile_stage_times = profile_stage_times or profile_cuda_sync
    if profile_stage_times:
        summary["stage_timing"] = dict(enabled=True, timer="host_perf_counter_wallclock",
            cuda_execution_synchronized=profile_cuda_sync,
            CUDA_event_kernel_time_measured=False,
            default_pipeline_adds_CUDA_synchronization=False,
            asynchronous_CUDA_completion_may_be_charged_to_later_stages=not profile_cuda_sync,
            diagnostic_windows_start_with_context_reset=True, frames=[])
    write_json(summary_path, summary)
    counts, seen_issue_keys = Counter(), set()
    started = last_report = time.monotonic()
    snapshots_used = 0
    encoder = None
    readings_hash = hashlib.sha256()
    try:
        with av.open(str(video)) as container, SourceRecordMemberStream(rows_path) as records:
            stream = container.streams.video[0]
            clock = Fraction(stream.time_base)
            for shot in shots:
                if Fraction(shot["time_base"]) != clock:
                    raise ValueError("Classified shot time base differs from the native video")
                shot_summary = dict(shot, processed_frames=0, first_pts=None, last_pts=None,
                    first_excluded_end_pts=None, strictly_increasing_pts=True,
                    contains_training_source_frame=source_is_training and any(
                        shot["start_pts"] <= p < shot["end_pts_exclusive"] for p in training_pts),
                    issues=Counter(), pts_gaps=Counter(), source_pts_complete=False)
                shot_summary["source_clock_scope_frame_counts"] = Counter()
                summary["shots"].append(shot_summary)
                reader.vision.reset()
                reader.active_guides, reader.active_binding, reader.active_global_exclusion = [], None, None
                advertisement_object_ids, previously_missing_ads, last_ad_match_status = {}, set(), {}
                previous_pts = None
                last_keyframe_pts = None
                last_keyframe_binding = None
                tracked_instance_states = []
                last_duration = None
                encoder = encoder_stream = None
                container.seek(shot["start_pts"], stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("Original decoded source frame has no PTS")
                    pts = int(frame.pts)
                    if pts < shot["start_pts"]:
                        continue
                    if pts >= shot["end_pts_exclusive"]:
                        shot_summary["first_excluded_end_pts"] = pts
                        break
                    if previous_pts is not None:
                        gap = pts - previous_pts
                        shot_summary["pts_gaps"][str(gap)] += 1
                        if gap <= 0:
                            raise ValueError("Original source PTS is not strictly increasing")
                    rgb = frame.to_ndarray(format="rgb24")
                    if frame.time_base != clock:
                        raise ValueError("Decoded source frame changed its native time base")
                    last_duration = frame.duration
                    binding = dict(source_sha256=digest, source_pts=pts, time_base=str(clock),
                        shot_id=shot["id"], installation_id=shot["camera_id"] or "unclassified_installation",
                        native_hw=list(rgb.shape[:2]), image_domain="native_brown_pixels")
                    reset = shot_summary["processed_frames"] == 0
                    keyframe=(application_keyframe_interval_seconds is None or reset or pts in seeds or last_keyframe_pts is None
                        or float((pts-last_keyframe_pts)*clock)>=application_keyframe_interval_seconds)
                    if not keyframe:
                        binding=retain_same_shot_installation_identity(binding,last_keyframe_binding)
                    model_search = search_adapter.guides_for(binding) if search_adapter and keyframe else None
                    if model_search and model_search["guides"]:
                        binding = model_search["source_binding"]
                    stage_times = {}
                    def timed(name, operation):
                        if not profile_stage_times:
                            return operation()
                        import torch
                        if profile_cuda_sync:
                            torch.cuda.synchronize(reader.vision.device)
                        tick = time.perf_counter()
                        value = operation()
                        if profile_cuda_sync:
                            torch.cuda.synchronize(reader.vision.device)
                        stage_times[name] = stage_times.get(name, 0.)+time.perf_counter()-tick
                        return value
                    if profile_stage_times:
                        import torch
                        torch.cuda.reset_peak_memory_stats(reader.vision.device)
                    precomputed = timed("predict", lambda: reader.vision.predict(rgb, binding, reset=reset)) if provider or material_authority else None
                    if precomputed is not None:
                        binding = precomputed["source_binding"]
                    seed = _bind_seed(seeds[pts], binding) if pts in seeds else None
                    guides = None
                    if seed and seed.get("annotations"):
                        from .vision import confirmed_source_guides
                        sample=dict(seed,source_hash=binding['source_sha256'],source_pts=binding['source_pts'],
                            time_base=binding['time_base'],shot_id=binding['shot_id'],frame_size_wh=list(reversed(rgb.shape[:2])))
                        guides = confirmed_source_guides(sample,role_definitions=seed.get('role_definitions'),return_packet=True)
                        for guide in guides['guides']:
                            guide['source_binding']=dict(binding)
                        unresolved = guides["unresolved"]
                        guides = guides["guides"]
                    else:
                        unresolved = []
                    if keyframe and not reset and application_keyframe_interval_seconds is not None:
                        confirmed=carry_confirmed_application_guides(reader,binding)
                        supplied={guide['id']for guide in guides or ()}
                        guides=list(guides or ())+[guide for guide in confirmed if guide['id']not in supplied]
                    if model_search and model_search["guides"]:
                        manual_groups = {guide["id"] for guide in guides or ()}
                        guides = list(guides or ()) + [guide for guide in model_search["guides"]
                                                      if guide["id"] not in manual_groups]
                    instances, constraints, person_rows, advertisement_match = {}, None, [], None
                    if provider and keyframe:
                        if person_detector:
                            person_rows = person_detector.read(rgb, binding)
                        if automatic_regions:
                            instances = timed("SAM2_regions_and_object_memory", lambda: provider.read_auto(rgb, binding,
                                class_probabilities=precomputed["class_probabilities"], classes=reader.vision.classes,
                                probabilities_conditioned_on_geometry=False, person_boxes=person_rows,
                                reset=reset or application_keyframe_interval_seconds is not None))
                            if seed and seed.get("prompts"):
                                instances = retain_same_source_refinement_evidence(instances,
                                    provider.read(rgb, binding, prompts=seed["prompts"], reset=False))
                        else:
                            instances = provider.read(rgb, binding, prompts=seed.get("prompts", ()) if seed else (), reset=reset)
                        if automatic_regions and not instances.get("bootstrap_source_proposals"):
                            from .vision_instances import automatic_source_region_prompts
                            instances["bootstrap_source_proposals"] = timed("native_source_pair_topology", lambda: automatic_source_region_prompts(
                                rgb, binding, class_probabilities=precomputed["class_probabilities"],
                                classes=reader.vision.classes, probabilities_conditioned_on_geometry=False))
                        missing_ads = {item["object_id"] for item in instances.get("instances", ())
                                       if item["kind"] == "advertisement_panel" and not bool(item["region_mask"].any())}
                        new_missing_ads = missing_ads-previously_missing_ads
                        if advertisement_matcher and (reset or new_missing_ads or application_keyframe_interval_seconds is not None):
                            advertisement_match = timed("advertisement_source_feature_matcher", lambda:
                                advertisement_matcher.read(rgb, binding, model_search=model_search))
                            advertisement_match["trigger"] = "source_shot_reset" if reset else "application_source_reidentification_keyframe" if application_keyframe_interval_seconds is not None else "tracked_source_region_lost"
                            last_ad_match_status = {item["template_id"]: dict(source_pts=pts,
                                appearance_match=item.get("appearance_match", False),
                                identity_status=item["identity_status"], reason=item.get("reason"))
                                for item in advertisement_match["matches"]}
                            prompts = []
                            for original in advertisement_match["prompts"]:
                                prompt = dict(original)
                                key = prompt["appearance_correspondence_template_id"]
                                if key not in advertisement_object_ids:
                                    advertisement_object_ids[key] = 2000+len(advertisement_object_ids)
                                prompt["object_id"] = advertisement_object_ids[key]
                                prompts.append(prompt)
                            if prompts:
                                initial_instances = instances
                                instances = timed("SAM2_advertisement_prompt_refinement", lambda:
                                    retain_same_source_refinement_evidence(initial_instances,
                                        provider.read(rgb, binding, prompts=prompts, reset=False)))
                            previously_missing_ads = missing_ads
                        elif advertisement_matcher:
                            previously_missing_ads = missing_ads
                        if advertisement_matcher:
                            instances = timed("advertisement_RGB_reread", lambda:
                                reread_source_advertisements(rgb, instances, provider))
                        constraints = source_region_constraints(instances, native_hw=rgb.shape[:2],
                                                                device=reader.vision.device)
                    result = timed("source_and_guided_reader", lambda: reader.read(rgb, binding, guides_2d=guides,
                        ignored_regions=seed.get("ignored_regions", ()) if seed else (), reset=reset,
                        source_region_constraints=constraints, precomputed_prediction=precomputed,
                        extract_source_components=keyframe))
                    diagnostics = timed("source_role_diagnostics_and_ledger", lambda: source_diagnostics(result))
                    if not keyframe:
                        for status in diagnostics['source_role_reading_status'].values():
                            status['status']='full_frame_component_search_not_performed_current_guided_RGB_is_separate'
                    if material_authority and keyframe:
                        from .vision_source_topology import source_board_fragment_topology,read_board_foot_material_profiles
                        topology=timed('source_material_fragment_navigation',lambda: source_board_fragment_topology(
                            rgb,result['source'],advertisement_instances=instances.get('instances',())))
                        material_packet=timed('native_RGB_board_foot_material_reading',lambda: read_board_foot_material_profiles(
                            rgb,instances or dict(source_binding=binding,instances=[]),topology,
                            material_role_authority=material_authority,prediction=precomputed,classes=reader.vision.classes,
                            device=reader.vision.device))
                        diagnostics['source_board_foot_material_reading']=material_packet
                        lower=material_packet['readings'][0] if material_packet.get('readings') else None
                        diagnostics['source_role_reading_status']['ice_contact'].update(
                            actual_lower_material_source_point_count=len(lower['primary_source_uv']) if lower else 0,
                            raw_optical_source_point_count=len(lower['all_observed_optical_source_uv']) if lower else 0,
                            physical_role_status='source_RGB_board_foot_lower_material_hypothesis',
                            unverified_region_scope_used_as_RGB_veto=False)
                        diagnostics['issues'].append(dict(feature_id='board_foot_source_material_roles',
                            reasons=['upper_material_entry_precision_unaccepted_and_occluded_source_not_fabricated'],
                            unresolved_lower_source_points=len(lower['unresolved_source_uv']) if lower else 0))
                    diagnostics["source_board_top_role_assignment"] = timed("source_top_role_assignment", lambda: source_board_top_roles(
                        rgb, result["source"]["components"], instances.get("bootstrap_source_proposals"),
                        source_binding=binding, foot_material_packet=diagnostics.get('source_board_foot_material_reading'),
                        source_instances=instances.get('instances',()))) if keyframe else dict(
                            status='material_reidentification_not_performed_on_intermediate_tracking_frame',
                            primary_source_top=dict(source_pixel_count=0,actual_source_uv=np.empty((0,2),np.float32)),
                            unresolved_typed_edges=dict(source_pixel_count=0))
                    top_roles = diagnostics["source_board_top_role_assignment"]
                    diagnostics["source_role_reading_status"]["board_top"].update(
                        primary_same_board_source_top_pixel_count=top_roles["primary_source_top"]["source_pixel_count"],
                        unresolved_typed_edge_pixel_count=top_roles["unresolved_typed_edges"]["source_pixel_count"],
                        physical_role_status=top_roles["status"])
                    if top_roles["unresolved_typed_edges"]["source_pixel_count"]:
                        diagnostics["issues"].append(dict(feature_id="board_top_physical_role",
                            reasons=["typed_source_edges_have_unresolved_same_contact_board_role"],
                            unresolved_source_pixel_count=top_roles["unresolved_typed_edges"]["source_pixel_count"]))
                    if source_material_proposals and keyframe:
                        if reset or application_keyframe_interval_seconds is not None:
                            instances,material_proposals=timed('unnamed_source_material_bootstrap_and_RGB_reread',lambda:
                                bootstrap_source_print_materials(rgb,instances,top_roles,provider,person_boxes=person_rows))
                            diagnostics['source_print_material_search_proposals']=material_proposals
                        else:
                            frame_index=provider.frame_index
                            initial_instances=instances
                            instances=timed('shared_material_current_navigation_RGB_reread',lambda:
                                retain_same_source_refinement_evidence(initial_instances,provider.read(
                                    rgb,binding,material_search_navigation=top_roles.get('source_board_material_search_navigation'))))
                            if provider.frame_index!=frame_index:
                                raise ValueError('Current material navigation advanced the native source frame index')
                            instances=timed('shared_material_original_RGB_edges',lambda:
                                reread_source_advertisements(rgb,instances,provider))
                            diagnostics['source_print_material_search_proposals']=dict(source_binding=binding,
                                status='shared_board_unnamed_layers_current_native_navigation_and_RGB_remeasured',
                                current_source_material_object_ids=[item['object_id']for item in instances.get('instances',())
                                    if item.get('source_material_proposal_unverified')],
                                current_source_frame_index=provider.frame_index,
                                fresh_source_material_reidentification_performed=False,
                                source_current_top_foot_navigation_used_as_search_only=True,
                                navigation_boxes_or_masks_used_as_measured_UV=False)
                        unknown=[item['object_id']for item in instances.get('instances',())
                            if item.get('source_material_proposal_unverified')]
                        if unknown:
                            diagnostics['issues'].append(dict(feature_id='unnamed_source_print_materials',
                                reasons=['current_RGB_material_layers_have_unverified_print_identity_and_physical_roles'],
                                source_material_object_ids=unknown))
                    from .vision_near_board import exclude_top_face_prints
                    if result.get("near_board"):
                        instances.setdefault("source_binding",binding)
                        instances=exclude_top_face_prints(instances,result["near_board"])
                    if constraints and constraints["soft_search_masks"]:
                        conflicts = {}
                        for kind, mask in constraints["soft_search_masks"].items():
                            support = result["source"]["typed_source_edge_masks"].get(kind)
                            if support is None:
                                continue
                            yx = np.argwhere(_array(support & ~mask)[0, 0])
                            conflicts[kind] = dict(source_pixel_count=len(yx),
                                source_pixel_runs_y_x_length=_pixel_runs(yx[:, [1, 0]]),
                                reason="SAM_region_scope_conflicts_with_RGB_type_evidence_preserved_not_vetoed")
                        diagnostics["source_region_soft_scope_conflicts"] = conflicts
                        if any(item["source_pixel_count"] for item in conflicts.values()):
                            diagnostics["issues"].append(dict(feature_id="same_contact_board_structure",
                                reasons=["unverified_region_envelope_multiple_top_hypotheses_preserved"],
                                conflicting_source_pixels={kind: item["source_pixel_count"] for kind, item in conflicts.items()}))
                    instance_rows = timed("compact_instance_readings_and_conflicts", lambda: compact_instances(instances))
                    for instance, original_instance in zip(instance_rows, instances.get("instances", ())):
                        if provider and not provider.reread_rgb and original_instance["kind"] != "advertisement_panel":
                            present = bool(original_instance["region_mask"].any())
                            instance["boundary_reading_mode"] = "source_RGB_remeasured_by_SourceRinkReader"
                            instance["missing_reason"] = None if present else "source_region_absent"
                        if instance.get("missing_reason"):
                            diagnostics["issues"].append(dict(feature_id=instance.get("physical_id") or instance["object_id"],
                                reasons=[instance["missing_reason"]]))
                    if constraints and constraints["missing_roles"]:
                        diagnostics["issues"].append(dict(feature_id="source_region_roles",
                            reasons=constraints["missing_roles"]))
                    if model_search:
                        if model_search["unresolved"]:
                            diagnostics["issues"].append(dict(feature_id="existing_model_search_roles",
                                reasons=["source_search_role_unresolved"], details=model_search["unresolved"]))
                        if model_search["status"].startswith("missing"):
                            diagnostics["issues"].append(dict(feature_id="existing_model_search_camera",
                                reasons=[model_search["status"]], details=model_search.get("missing_reason")))
                    if unresolved:
                        diagnostics["issues"].append(dict(feature_id="source_seed_roles", reasons=["unresolved_source_roles"],
                                                          details=unresolved))
                    clock_evidence = result["source_clock_evidence"]
                    if clock_evidence not in summary["source_clock_evidence"]:
                        summary["source_clock_evidence"].append(clock_evidence)
                    shot_summary["source_clock_evidence"] = clock_evidence
                    clock_scope = ("explicit_proof_covered_native_PTS" if binding.get("exposure_proof_sha256")
                                   else "native_PTS_without_verified_alias")
                    shot_summary["source_clock_scope_frame_counts"][clock_scope] += 1
                    row = dict(source_binding=binding, RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        source_time=format_source_time(pts, clock),
                        source_clock_evidence=clock_evidence, source_clock_scope=clock_scope,
                        model_sha256=result["model_sha256"], memory_update=result["memory_update"],
                        model_usage=result.get('model_usage'),original_FIT_source_revision=result.get('original_FIT_source_revision'),
                        new_observation=result["new_observation"], source_motion_metadata=result["source_motion_metadata"],
                        source_seed_injected=seed is not None, instance_model_sha256=instances.get("model_sha256"),
                        instance_memory_update=instances.get("memory_update"),
                        source_instance_provider_evidence={key:value for key,value in instances.items() if key!='instances'},
                        source_person_reidentification=instances.get('source_person_reidentification'),
                        source_region_constraints=constraints["metadata"] if constraints else None,
                        automatic_source_region_proposal=instances.get("bootstrap_source_proposals"),
                        advertisement_source_matching=advertisement_match,
                        advertisement_source_object_inventory=source_advertisement_inventory(
                            advertisement_matcher, advertisement_object_ids, instances.get("instances", ()), last_ad_match_status)
                            if advertisement_matcher else None,
                        instance_source_frame_index=provider.frame_index if provider else None,
                        instance_source_shot_reset_requested=reset if provider else None,
                        advertisement_source_matching_status=(advertisement_match["trigger"] if advertisement_match else
                            "SAM2_source_object_memory_continued_no_query_keypoints_fabricated") if advertisement_matcher else "disabled",
                        independent_source_person_boxes=person_rows,
                        existing_model_search={k: model_search[k] for k in (
                            "status", "event_id", "scene_sha256", "camera_sha256", "search_inventory",
                            "model_curve_inventory_count", "projected_pixels_are_search_only", "missing_reason")
                            if k in model_search} if model_search else None,
                        instances=instance_rows, **diagnostics)
                    row['source_reidentification_keyframe']=keyframe
                    row['tracking_only']=not keyframe
                    row['last_source_reidentification_binding']=binding if keyframe else last_keyframe_binding
                    row['current_person_and_print_region_reidentification_performed']=bool(provider and keyframe)
                    row['full_frame_component_extraction_performed']=keyframe
                    row['source_calibration_anchor_used']=bool(seed and seed.get('annotations'))
                    row['source_calibration_anchor_is_independent_holdout']=False if row['source_calibration_anchor_used'] else None
                    if keyframe:
                        last_keyframe_pts,last_keyframe_binding=pts,binding
                        tracked_instance_states=[{key:instance[key] for key in ('object_id','kind','physical_id','expected_id',
                            'identity_authority','expected_identity_authority','candidate_physical_ids','missing_reason','status')if key in instance}
                            for instance in instance_rows]
                        if application_keyframe_interval_seconds is not None:
                            retain_current_print_identity_guides(reader,instances.get('instances',()),binding)
                    else:
                        row['tracked_instance_identity_states']=[dict(state,
                            status='not_reidentified_on_intermediate_tracking_frame',
                            current_region_or_person_occlusion_not_verified=True,
                            source_identity_evidence_binding=last_keyframe_binding)for state in tracked_instance_states]
                        row['instance_source_frame_index']=None
                        row['instance_source_shot_reset_requested']=None
                        row['advertisement_source_matching_status']='not_reidentified_on_intermediate_frame_current_RGB_guided_reading_is_separate'
                    stored_row=application_source_record(row,result=result) if output_mode=='application' else row
                    summary["readings_storage"]["uncompressed_UTF8_bytes"] += timed(
                        "serialize_and_compressed_write", lambda: _write_source_record(records, stored_row, readings_hash,
                            array_sidecar_path=output_directory/f'source_{pts}_arrays.npz',
                            storage_metadata=summary['readings_storage'],minimum_array_bytes=1,
                            storage_layout=source_array_storage_layout))
                    if profile_stage_times:
                        import torch
                        free_bytes, total_bytes = torch.cuda.mem_get_info(reader.vision.device)
                        summary["stage_timing"]["frames"].append(dict(source_pts=pts, shot_id=shot["id"],
                            context_reset_requested=reset, elapsed_seconds=stage_times,
                            GPU_memory_bytes=dict(allocated=torch.cuda.memory_allocated(reader.vision.device),
                                reserved=torch.cuda.memory_reserved(reader.vision.device),
                                peak_allocated=torch.cuda.max_memory_allocated(reader.vision.device),
                                peak_reserved=torch.cuda.max_memory_reserved(reader.vision.device),
                                device_free=free_bytes, device_total=total_bytes)))
                    issue_keys = [(str(i["feature_id"]), reason) for i in diagnostics["issues"] for reason in i["reasons"]]
                    for _, reason in issue_keys:
                        counts[reason] += 1
                        shot_summary["issues"][reason] += 1
                    novel = next((key for key in issue_keys if (shot["id"], *key) not in seen_issue_keys), None)
                    requested = pts in requested_snapshots
                    capture = requested or (novel is not None and snapshots_used < issue_image_count)
                    if capture or render_video:
                        image = draw_application_record(rgb,stored_row) if output_mode=='application' else draw_review(rgb, result, diagnostics, instances.get("instances", ()))
                    if capture:
                        prefix = f"source_{pts}"
                        source_path = output_directory / f"{prefix}.png"
                        reading_path = output_directory / f"{prefix}_reading.png"
                        if output_mode!='application':
                            Image.fromarray(rgb).save(source_path)
                        Image.fromarray(image).save(reading_path)
                        summary["snapshots"].append(dict(source_pts=pts, time_base=str(clock), shot_id=shot["id"],
                            source_time_seconds=float(pts * clock), original_image=str(source_path) if output_mode!='application' else None,
                            original_RGB_reproduction=dict(source_video=str(video),source_binding=binding,RGB_sha256=stored_row['RGB_sha256']),
                            source_time=format_source_time(pts, clock),
                            reading_image=str(reading_path), issues=diagnostics["issues"],
                            reason="requested_source_timestamp" if requested else "new_failure_kind",
                            training_source_frame=source_is_training and pts in training_pts))
                        requested_snapshots.discard(pts)
                        snapshots_used += not requested
                    seen_issue_keys.update((shot["id"], *key) for key in issue_keys)
                    if render_video:
                        if encoder is None:
                            video_path = output_directory / f"shot_{shot['index']:04d}_actual_readings.mp4"
                            encoder = av.open(str(video_path), "w")
                            encoder_stream = encoder.add_stream("libx264", rate=stream.average_rate)
                            encoder_stream.width, encoder_stream.height = image.shape[1], image.shape[0]
                            encoder_stream.pix_fmt = "yuv420p"
                            encoder_stream.options = {"crf": "18", "preset": "fast"}
                            encoder_stream.time_base = clock
                            shot_summary["video"] = str(video_path)
                        rendered = av.VideoFrame.from_ndarray(image, format="rgb24")
                        rendered.pts, rendered.time_base = pts - shot["start_pts"], clock
                        for packet in encoder_stream.encode(rendered):
                            encoder.mux(packet)
                    summary["frame_count"] += 1
                    shot_summary["processed_frames"] += 1
                    shot_summary["first_pts"] = shot_summary["first_pts"] if previous_pts is not None else pts
                    shot_summary["last_pts"] = previous_pts = pts
                    if time.monotonic() - last_report >= 15:
                        summary["issue_counts"] = dict(counts)
                        summary["elapsed_seconds"] = time.monotonic() - started
                        write_json(summary_path, _finite_json(summary))
                        print(json.dumps(dict(frames=summary["frame_count"], shot=shot["id"], source_pts=pts,
                                              elapsed_seconds=summary["elapsed_seconds"])), flush=True)
                        last_report = time.monotonic()
                    if output_mode=='application':
                        # Current measured guide state lives in reader.active_guides.
                        # Workspaces and unneeded SAM histories are not retained.
                        if keyframe and application_keyframe_interval_seconds is not None and provider:
                            provider.reset()
                        diagnostics=result=row=stored_row=instances=constraints=precomputed=instance_rows=None
                        topology=material_packet=material_proposals=top_roles=lower=None
                if encoder is not None:
                    for packet in encoder_stream.encode():
                        encoder.mux(packet)
                    encoder.close()
                    encoder = None
                exact_eof_boundary = (shot_summary["first_excluded_end_pts"] is None
                    and shot_summary["last_pts"] is not None and last_duration is not None
                    and shot_summary["last_pts"] + last_duration == shot["end_pts_exclusive"])
                expected_count = shot["expected_native_frame_count"]
                shot_summary["source_pts_complete"] = bool(
                    shot_summary["processed_frames"] and shot_summary["first_pts"] == shot["start_pts"]
                    and (shot_summary["first_excluded_end_pts"] == shot["end_pts_exclusive"] or exact_eof_boundary)
                    and (expected_count is None or shot_summary["processed_frames"] == expected_count))
                shot_summary["native_last_frame_duration_pts"] = last_duration
                shot_summary["exact_end_of_stream_boundary"] = exact_eof_boundary
                shot_summary["issues"] = dict(shot_summary["issues"])
                shot_summary["pts_gaps"] = dict(shot_summary["pts_gaps"])
                shot_summary["source_clock_scope_frame_counts"] = dict(shot_summary["source_clock_scope_frame_counts"])
        summary["source_pts_complete"] = all(s["source_pts_complete"] for s in summary["shots"])
        summary["status"] = "source_range_complete" if summary["source_pts_complete"] else "source_clock_review_required"
        summary["unobserved_requested_snapshot_pts"] = sorted(requested_snapshots)
        summary["independent_shots_not_containing_training_source_frames"] = [
            s["id"] for s in summary["shots"] if not s["contains_training_source_frame"]]
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error),
                       source_pts_complete=False, all_decoded_original_frames_processed=False)
        raise
    finally:
        if encoder is not None:
            encoder.close()
        summary["issue_counts"] = dict(counts)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["readings_storage"]["uncompressed_UTF8_sha256"] = readings_hash.hexdigest()
        summary["readings_storage"]["compressed_bytes"] = rows_path.stat().st_size if rows_path.exists() else None
        summary['source_code_sha256_at_run_end']={name:sha256(ROOT/'src/rinkspace'/name)
            for name in summary['source_code_sha256_at_run_start']}
        summary['source_code_SHA_start_end_equal']=(summary['source_code_sha256_at_run_start']==summary['source_code_sha256_at_run_end'])
        write_json(summary_path, _finite_json(summary))
    print(json.dumps(dict(status=summary["status"], frame_count=summary["frame_count"],
                         output=str(summary_path), accuracy_accepted=False)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/rink_vision.json"))
    parser.add_argument("--model", help="explicit Data candidate weights; defaults to configured final model")
    parser.add_argument("--shots", required=True)
    parser.add_argument("--boundary-overlay", help="hash-bound verified native cut correction; preserve camera authority")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video")
    parser.add_argument("--shot-id", action="append", default=[])
    parser.add_argument("--mode", choices=("independent", "assisted"), default="independent")
    parser.add_argument("--evaluation-role", choices=("development", "held_out"), default="development")
    parser.add_argument("--source-seeds")
    parser.add_argument("--instance-checkpoint")
    parser.add_argument("--disable-auto-regions", action="store_true")
    parser.add_argument("--person-weights", help="existing local model used solely for source occluder boxes")
    parser.add_argument("--person-vendor", help="optional existing local Ultralytics source path")
    parser.add_argument('--person-dependency',action='append',default=[],help='append an existing dependency directory without replacing the active GPU runtime')
    parser.add_argument('--board-foot-material-contract',help='permanent user material-role contract; never load its annotation pixels into runtime')
    parser.add_argument('--source-material-proposals',action='store_true',
        help='bootstrap unnamed print materials from actual top/foot navigation at shot reset; subsequent native RGB is remeasured')
    parser.add_argument('--array-storage-layout',choices=('individual_arrays','byte_pool'),default='byte_pool',
        help='lossless typed evidence layout; byte_pool avoids thousands of independent ZIP members')
    parser.add_argument('--output-mode',choices=('evidence','application'),default='evidence',
        help='application keeps actual fitting observations and failure provenance; inference masks/profiles stay in memory')
    parser.add_argument('--application-keyframe-seconds',type=float,
        help='sparse source-time reidentification; intermediate native RAFT plus current RGB measurements')
    parser.add_argument("--model-search-okada-root", help="optional exact existing source-bound camera/scene search adapter")
    parser.add_argument("--paint-role-contract", help="source role contract for the optional model search")
    parser.add_argument('--original-fit-request', help='prepare original packet/geometry once; reread exact FIT locators during native streaming')
    parser.add_argument("--event-id", help="event scope for the source advertisement bank")
    parser.add_argument("--profile-stage-times", action="store_true", help="diagnostic host wallclock stage timing, CUDA may be asynchronous")
    parser.add_argument("--profile-cuda-sync", action="store_true", help="diagnostic-only completed-stage wallclock timing; explicitly synchronizes CUDA")
    parser.add_argument("--advertisement-source", action="append", default=[],
                        help="authenticated original RGB appearance packet; never inject query-frame annotations")
    parser.add_argument("--snapshot-pts", type=int, action="append", default=[])
    parser.add_argument("--issue-image-count", type=int, default=6,
                        help="adjustable diagnostic image budget, not a processing or task limit")
    parser.add_argument("--render-video", action="store_true",
                        help="complete source-reading video; use after demonstrated source improvement")
    parser.add_argument("--start-pts", type=int, help="optional diagnostic subinterval; requires one selected shot")
    parser.add_argument("--end-pts", type=int, help="diagnostic exclusive end; not a complete-shot delivery")
    args = parser.parse_args()
    review(args.config, args.shots, args.output, video=args.video, shot_ids=args.shot_id,
        mode=args.mode, seed_path=args.source_seeds, instance_checkpoint=args.instance_checkpoint,
        snapshot_pts=args.snapshot_pts, issue_image_count=args.issue_image_count, render_video=args.render_video,
        diagnostic_start_pts=args.start_pts, diagnostic_end_pts=args.end_pts,
        automatic_regions=not args.disable_auto_regions, person_weights=args.person_weights, person_vendor=args.person_vendor,
        model_path=args.model, model_search_root=args.model_search_okada_root, paint_role_contract=args.paint_role_contract,
        evaluation_role=args.evaluation_role, event_id=args.event_id,
        advertisement_source_paths=args.advertisement_source, boundary_overlay_path=args.boundary_overlay,
        profile_stage_times=args.profile_stage_times, profile_cuda_sync=args.profile_cuda_sync,
        board_foot_material_contract=args.board_foot_material_contract,person_dependencies=args.person_dependency,
        source_material_proposals=args.source_material_proposals,source_array_storage_layout=args.array_storage_layout,
        output_mode=args.output_mode,application_keyframe_interval_seconds=args.application_keyframe_seconds,
        original_fit_request_path=args.original_fit_request)


if __name__ == "__main__":
    main()
