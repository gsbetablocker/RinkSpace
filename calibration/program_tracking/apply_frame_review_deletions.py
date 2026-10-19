"""Apply exact reviewed point deletions, retaining evidence and untouched UVs.

The specification is a review receipt with pre-edit hashes, source addresses,
and exact native UVs. It cannot select points from a geometric threshold.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import cv2

from .finalize_continuation_frame_batch import DATA, OUT
from .validate_opening_frame_batch import read, sha, reference, require
from .extract_opening_frame_batch import write


def apply(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = read(spec_path)
    revision = spec['revision_id']
    archive = DATA/revision/'before'
    archive.mkdir(parents=True, exist_ok=True)
    receipt = OUT/(revision + '.json')
    require(not receipt.exists(), 'Review revision already applied; inspect its receipt instead of applying again')
    changes = spec['frames']
    # Preflight all exact source coordinates before writing any candidate.
    for item in changes:
        index = item['index']
        require(101 <= index < 200, 'This correction is limited to continuation candidate frames')
        path = DATA/f'ice_candidates/{index:06d}.json'
        require(sha(path) == item['before_sha256'], 'Candidate changed since review')
        packet = read(path)
        features = {f['id']: f for f in packet['features']}
        addresses = set()
        for point in item['remove']:
            address = (point['feature_id'], point['segment_index'], point['vertex_index'])
            require(address not in addresses, 'Duplicate deletion address')
            addresses.add(address)
            segment = features[address[0]]['segments'][address[1]]
            require(segment['uv'][address[2]] == point['uv'], 'Reviewed native coordinate does not match')
    for name in ['ice_extraction_manifest.json', 'release_check.json', 'model_entry_validation.json']:
        shutil.copy2(DATA/name, archive/name)
    for name in ['batch_manifest.json', 'review.html']:
        shutil.copy2(OUT/name, archive/name)
    immutable_before = {}
    for path in sorted((OUT/'frames').glob('*.json')):
        immutable_before[int(path.stem)] = {'packet': reference(path), 'binding': reference(OUT/'bindings'/path.name)}
    report = deepcopy(spec)
    report.update(schema='rinkspace.user-reviewed-point-deletions.v1',
        optimization_performed=False, retained_coordinates_changed=False,
        source_images_changed=False, user_acceptance_of_corrected_batch=False,
        input_specification=reference(spec_path), frames=[])
    for item in changes:
        index = item['index']; path = DATA/f'ice_candidates/{index:06d}.json'
        before = read(path); packet = deepcopy(before)
        for folder, source in [('ice_candidates', path), ('frames', OUT/f'frames/{index:06d}.json'),
                               ('bindings', OUT/f'bindings/{index:06d}.json'),
                               ('overlays', DATA/f'overlays/{index:06d}.png'),
                               ('ice_overlays', DATA/f'ice_overlays/{index:06d}.png')]:
            destination=archive/folder/source.name; destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
        deleted = {(p['feature_id'], p['segment_index'], p['vertex_index']) for p in item['remove']}
        retained_old, retained_new, history = [], [], []
        for feature in packet['features']:
            fid = feature['id']; segments = []
            changed = any(address[0] == fid for address in deleted)
            for si, segment in enumerate(feature['segments']):
                uv = segment['uv']; runs = []; active = []
                for vi, point in enumerate(uv):
                    if (fid,si,vi) in deleted:
                        if active: runs.append(active); active=[]
                        history.append(dict(feature_id=fid,segment_index=si,vertex_index=vi,uv=point,
                            source_evidence=segment.get('source_evidence', [None]*len(uv))[vi]))
                    else:
                        active.append(vi);retained_old.append((fid,point))
                if active: runs.append(active)
                if len(runs)==1 and runs[0]==list(range(len(uv))):
                    segments.append(segment)
                    retained_new.extend((fid,point) for point in uv)
                    continue
                for run in runs:
                    current=deepcopy(segment)
                    for key in ['uv','sigma_px','source_evidence','point_ids','point_bindings']:
                        if key in current:
                            require(len(current[key])==len(uv), 'Per-point evidence array length mismatch')
                            current[key]=[current[key][vi] for vi in run]
                    current.update(vertex_count=len(run),closed=False)
                    segments.append(current)
                    retained_new.extend((fid,point) for point in current['uv'])
            feature['segments']=segments
            feature['point_count']=sum(len(s['uv']) for s in segments)
            if changed:
                if 'geometric_interpretation' in feature:
                    # The old fit used deleted points. Preserve it only in before/.
                    feature.pop('geometric_interpretation')
                feature['user_review_deletion_revision']=revision
                if not feature['point_count']:
                    feature['note_zh']='本帧该层仅有的候选点已由用户审阅否决；不留拟合点。'
        require(retained_old==retained_new, 'Retained points changed or reordered')
        packet['point_count']=sum(f['point_count'] for f in packet['features'])
        require(before['point_count']-packet['point_count']==len(deleted), 'Deletion count mismatch')
        packet['user_review_deletions']=dict(revision_id=revision,receipt_path=str(receipt),
            deleted_point_count=len(deleted),before_candidate_sha256=item['before_sha256'])
        counts={f['target_id']:f['point_count'] for f in packet['features']}
        for row in packet.get('curve_readout_summary',[]):
            if row['target_id'] in counts: row['final_points']=counts[row['target_id']]
        packet['unresolved_source_regions']=[r for r in packet.get('curve_readout_summary',[]) if r['final_points']==0]
        write(path,packet)
        report['frames'].append(dict(index=index,removed_count=len(deleted),
            before_ice_point_count=before['point_count'],after_ice_point_count=packet['point_count'],
            before=reference(archive/'ice_candidates'/path.name),after=reference(path),
            remove=item['remove'],deleted_source_evidence=history))
    report['removed_point_count']=sum(row['removed_count'] for row in report['frames'])
    write(receipt,report)
    write(DATA/revision/'before_batch_addresses.json',immutable_before)
    manifest=read(DATA/'ice_extraction_manifest.json')
    from .extract_continuation_ice_source import draw_overlay, contact_sheets
    for row in manifest['rows']:
        if row['index'] not in {item['index'] for item in changes}: continue
        path=Path(row['packet_path']);packet=read(path)
        row.update(point_count=packet['point_count'],by_feature={f['id']:f['point_count'] for f in packet['features']},
                   missing=[f['id'] for f in packet['features'] if not f['point_count']],packet_sha256=sha(path))
        source=cv2.imread(str(DATA/f'frames/{row["index"]:06d}.png'))
        cv2.imwrite(str(DATA/f'ice_overlays/{row["index"]:06d}.png'),draw_overlay(source,packet))
    counts=[row['point_count'] for row in manifest['rows']]
    manifest.update(point_count_total=sum(counts),point_count_min=min(counts),point_count_max=max(counts),
                    latest_user_review_correction=reference(receipt))
    manifest['visual_review']['pre_correction_review']=reference(archive/'ice_extraction_manifest.json')
    manifest['visual_review']['status']='corrected_overlays_pending_final_review'
    contact_sheets(DATA)
    for entry in manifest['visual_review']['contact_sheets']:
        entry['sha256']=sha(entry['path'])
        if any(int(Path(entry['path']).stem.split('-')[0])<=item['index']<=int(Path(entry['path']).stem.split('-')[1]) for item in changes):
            entry['viewed_with_view_image']=False
    write(DATA/'ice_extraction_manifest.json',manifest)
    print(json.dumps(dict(receipt=str(receipt),changed_frames=len(changes),removed=report['removed_point_count']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('specification')
    apply(parser.parse_args().specification)
