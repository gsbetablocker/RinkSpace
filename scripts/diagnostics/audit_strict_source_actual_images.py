"""Read-only strict consumer and own-native-image QA for a small source delta.

This checks source rows and selected actual/alias images; it does not admit FIT
or replace complete source-window visual validation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.render_result import SceneRenderer
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from rinkspace.programme_source_stream import FrozenProgrammeSource


def reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_sha(path))


def gather_actual_rows(pack):
    """Join every row of its own exposure, retaining original candidate IDs."""
    records={};duplicates=0;seen=set()
    for ri,h in enumerate(pack.rows):
        lo,hi=map(int,pack.values['row_offsets'][ri:ri+2])
        actual=int(h['actual_pts']);native=int(h['native_pts'])
        if actual!=native:raise ValueError('Canonical source observations must retain their own actual/native clock')
        ids=np.asarray(h['original_candidate_indices'],np.int64)
        if not np.array_equal(ids,pack.values['original_point_indices'][lo:hi]):
            raise ValueError('Header and numeric original candidate indices differ')
        r=records.setdefault(actual,dict(native_pts=native,BGR_sha256=h['decoded_BGR_sha256'],
            rows=[],uv=[],sides=[],original_candidate_indices=[]))
        if r['BGR_sha256']!=h['decoded_BGR_sha256']:raise ValueError('One actual exposure has conflicting RGB bindings')
        r['rows'].append(ri);r['uv'].append(pack.values['uv'][lo:hi]);r['sides'].append(pack.values['source_sides_uv'][lo:hi])
        r['original_candidate_indices'].extend(ids.tolist())
        context=(actual,h['scope'],h['physical_material_id'],
            json.dumps(h['model_target'],sort_keys=True,separators=(',',':')),h['observation_semantics'],h['split'])
        for point in pack.values['uv'][lo:hi]:
            key=context+(point.tobytes(),)
            if key in seen:duplicates+=1
            seen.add(key)
    for r in records.values():
        r['uv']=np.concatenate(r['uv']);r['sides']=np.concatenate(r['sides'])
        r['original_candidate_indices']=np.asarray(r['original_candidate_indices'],np.int64)
    return records,duplicates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--model-identity',type=Path,required=True)
    parser.add_argument('--model-identity-sha256',required=True)
    parser.add_argument('--programme-manifest',type=Path,required=True)
    parser.add_argument('--material-id',required=True)
    parser.add_argument('--old-pack',type=Path);parser.add_argument('--old-pack-sha256')
    parser.add_argument('--actual-pts',type=int,nargs='+')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
    receipt_path=dest/'consumer_and_actual_source_image_QA.json'
    if receipt_path.exists():raise FileExistsError('Preserve the previous source-image QA')
    source_dir=args.source_dir.resolve()
    terminal=json.loads((source_dir/'role_only_source_terminal_receipt.json').read_text(encoding='utf8'))
    pack_ref=dict(path=terminal['path'],sha256=terminal['sha256'])
    pack=StrictAbsoluteSourcePack(pack_ref['path'],expected_sha256=pack_ref['sha256'])
    if any(h['scope']!='ice' or h['physical_material_id']!=args.material_id for h in pack.rows):
        raise ValueError('The delta contains an unrequested physical source material')
    model_ref=dict(path=str(args.model_identity.resolve()),sha256=args.model_identity_sha256)
    authenticate_source_reference(model_ref)
    identity=json.loads(args.model_identity.read_text(encoding='utf8'))
    coverage_path=source_dir/'gray_current_actual_source_coverage_and_pending.json'
    coverage=json.loads(coverage_path.read_text(encoding='utf8'))
    manifest_path=source_dir/'gray_role_only_FROZEN.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if (pack.metadata['expected_actual_pts']!=sorted(set(pack.metadata['expected_actual_pts']))
            or pack.metadata['processed_actual_pts']!=manifest['exact_actual_pts']
            or [r['actual_pts'] for r in coverage['frame_records']]!=manifest['exact_actual_pts']):
        raise ValueError('Saved source clock differs from its exact requested readout window')
    original,duplicates=gather_actual_rows(pack)
    old_overlap=None;old_named_rows=None
    old_ref=None
    if args.old_pack or args.old_pack_sha256:
        if args.old_pack is None or args.old_pack_sha256 is None:raise ValueError('Exact original pack path and SHA required together')
        old_ref=dict(path=str(args.old_pack.resolve()),sha256=args.old_pack_sha256)
        authenticate_source_reference(old_ref)
        headers,_=read_minimal_compact_source_headers(args.old_pack)
        old_keys={(h['scope'],h['physical_material_id'],h['actual_pts']) for h in headers}
        new_keys={(h['scope'],h['physical_material_id'],h['actual_pts']) for h in pack.rows}
        old_overlap=len(old_keys&new_keys)
        old_named_rows=sum(h['physical_material_id']==args.material_id for h in headers)
        if file_sha(args.old_pack)!=old_ref['sha256']:raise ValueError('Original pack changed during read-only audit')
    alias_path=source_dir/'gray_native_alias_own_source_readouts.npz'
    with np.load(alias_path,allow_pickle=False)as saved:
        alias={k:saved[k].copy() for k in saved.files}
    alias_records=json.loads(str(alias['metadata_json'].item()))['own_RGB_alias_records']
    if alias_records!=coverage['alias_records']:raise ValueError('Alias coverage and saved own-readout clock differ')
    if args.actual_pts:
        chosen=sorted(set(args.actual_pts))
        if not set(chosen).issubset(pack.metadata['processed_actual_pts']):raise ValueError('Image QA is outside the requested actual source window')
    else:
        available=sorted(original) or manifest['exact_actual_pts']
        with_alias=sorted(set(r['actual_pts'] for r in alias_records)&set(available))
        first=with_alias[0] if with_alias else available[0]
        following=next((p for p in available if p>first),None)
        if following is None:following=next((p for p in reversed(available) if p<first),None)
        chosen=sorted([first]+([] if following is None else [following]))
    images={};alias_pending=Counter()
    for pts in chosen:
        if pts in original:
            r=original[pts]
            images[pts]=dict(actual_pts=pts,native_pts=pts,uv=r['uv'],sides=r['sides'],
                BGR_sha256=r['BGR_sha256'],source_row_indices=r['rows'],kind='canonical_actual')
        else:
            binding=next(r['source_binding'] for r in coverage['frame_records'] if r['actual_pts']==pts)
            images[pts]=dict(actual_pts=pts,native_pts=pts,uv=np.empty((0,2)),sides=np.empty((0,2,2)),
                BGR_sha256=binding['decoded_native_BGR_sha256'],source_row_indices=[],kind='canonical_actual_without_admitted_points')
    for i,r in enumerate(alias_records):
        pts=int(r['actual_pts'])
        if pts not in chosen:continue
        lo,hi=map(int,alias['offsets'][i:i+2])
        ids=original[pts]['original_candidate_indices'] if pts in original else np.empty(0,np.int64)
        status=alias['canonical_candidate_status'][lo:hi]
        if np.any(ids<0) or np.any(ids>=len(status)):raise ValueError('Alias candidate index belongs to another actual exposure')
        accepted=ids[status[ids]==1]
        alias_pending.update(map(int,status[ids][status[ids]!=1]))
        uv=alias['measured_native_uv'][lo:hi][accepted];sides=alias['measured_native_sides_uv'][lo:hi][accepted]
        if not np.isfinite(uv).all() or not np.isfinite(sides).all():raise ValueError('Accepted own alias contains unavailable measured pixels')
        images[int(r['native_pts'])]=dict(actual_pts=pts,native_pts=int(r['native_pts']),uv=uv,sides=sides,
            BGR_sha256=r['decoded_native_BGR_sha256'],source_row_indices=original.get(pts,{}).get('rows',[]),
            kind='own_native_alias',canonical_candidate_ids=accepted.tolist(),unavailable_canonical_candidate_ids=ids[status[ids]!=1].tolist())
    argv=json.loads(args.programme_manifest.read_text(encoding='utf8'))['common_arguments']
    option=lambda name:argv[argv.index(name)+1]
    if Path(option('--identity')).resolve()!=args.model_identity.resolve():
        other=json.loads(Path(option('--identity')).read_text(encoding='utf8'))
        if any(other[k]!=identity[k] for k in ('q_sha256','camera_sha256','scene_sha256')):
            raise ValueError('Image QA and source producer use different model/optical state')
    programme=FrozenProgrammeSource(args.model_identity,option('--source'),ROOT/'calibration/program_tracking/source_shots.json')
    if programme.actual_inventory.tolist()!=pack.metadata['expected_actual_pts']:
        raise ValueError('The source delta shortened the complete programme clock')
    renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
    image_records=[]
    for frame in programme.frames(start_pts=min(images),stop_pts=max(images)+1,cpu_threads=1):
        pts=int(frame['native_pts'])
        if pts not in images:continue
        r=images[pts];raw=frame['raw_bgr']
        if int(frame['actual_pts'])!=r['actual_pts'] or hashlib.sha256(raw.tobytes()).hexdigest()!=r['BGR_sha256']:
            raise ValueError('Own original native image differs from its source-pixel proof')
        raw_path=dest/f'{pts}_EXACT_RAW_NATIVE.png';grid_path=dest/f'{pts}_same_model_ALL_GRID_own_source_q50.png'
        if raw_path.exists()or grid_path.exists():raise FileExistsError('Preserve original source-image QA')
        if not cv2.imwrite(str(raw_path),raw):raise IOError('Unable to save the original source image')
        grid,_=renderer.draw(raw,frame['camera25'],title=f'{identity["q_sha256"][:4]} ALL GRID NAV; OWN {r["kind"]} {pts}; {len(r["uv"])} q50')
        for uv,sides in zip(r['uv'],r['sides']):
            for side in sides:cv2.circle(grid,tuple(np.rint(side).astype(int)),2,(0,255,255),1,cv2.LINE_AA)
            cv2.circle(grid,tuple(np.rint(uv).astype(int)),4,(0,0,255),1,cv2.LINE_AA)
        if not cv2.imwrite(str(grid_path),grid):raise IOError('Unable to save the complete same-model grid QA')
        image_records.append({**{k:v for k,v in r.items() if k not in ('uv','sides')},
            'source_points':len(r['uv']),'raw_reference':reference(raw_path),'same_model_full_grid_reference':reference(grid_path)})
    if len(image_records)!=len(images):raise ValueError('Selected actual/own-alias source images were omitted')
    if file_sha(args.model_identity)!=model_ref['sha256'] or file_sha(pack_ref['path'])!=pack_ref['sha256']:
        raise ValueError('Immutable source/model changed during read-only QA')
    result=dict(schema='rinkspace.strict-source-consumer-own-native-image-QA.v1',source_pack_reference=pack_ref,
        source_manifest_reference=reference(manifest_path),source_coverage_reference=reference(coverage_path),
        model_identity_reference=model_ref,q_sha256=identity['q_sha256'],strict_public_consumer_passed=True,
        points=len(pack.values['uv']),rows=len(pack.rows),material_point_counts=dict(Counter({args.material_id:len(pack.values['uv'])})),
        split_point_counts=dict(Counter({s:sum(r['point_count'] for r in pack.rows if r['split']==s) for s in ('fit','held')})),
        expected_programme_actual_exposures=len(programme.actual_inventory),requested_processed_actual_exposures=len(pack.metadata['processed_actual_pts']),
        original_pack_reference=old_ref,old_named_material_rows=old_named_rows,
        old_material_actual_frame_overlap_groups=old_overlap,internal_exact_UV_target_semantics_split_duplicate_points=duplicates,
        selected_actual_image_QA_pts=chosen,own_alias_pending_status_counts=dict(alias_pending),images=image_records,
        same_model_all_ice_lines_advertisement_boundaries_and_numbers_rendered=True,
        selected_image_visual_review_completed_by_this_program=False,complete_window_visual_accuracy_claimed=False,
        aliases_create_new_FIT_points=False,original_source_or_materials_modified=False,fit_started=False,
        implementation_reference=reference(__file__))
    with receipt_path.open('x',encoding='utf8')as output:json.dump(result,output,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('source_pack_reference','points','rows','split_point_counts',
        'expected_programme_actual_exposures','requested_processed_actual_exposures','old_material_actual_frame_overlap_groups',
        'internal_exact_UV_target_semantics_split_duplicate_points','selected_actual_image_QA_pts')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
