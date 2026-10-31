"""Read new/previously hidden contours from authenticated photographic charts.

Reuse only an existing same-frame image transform. Every returned coordinate is
read afresh from source RGB with current structure, geometry and mask checks.
No original observations or fitted pixel coordinates are copied into the output.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import json
import time

import cv2
import numpy as np

from capture_native_print_patterns import (
    read, write, sha, file_ref, plain, choose_readouts, iter_retry_records,
    validate_retry_frame,
)
from programme_source_frames import iter_registration_source_frames
from rinkspace.path_policy import guard_project_write
from rinkspace.session.print_patterns import PrintPattern, project, registration_geometry_check


def validate_chart(atlas, original, tolerance=1e-4):
    if atlas['image_sha256'] != original['image_sha256']:
        raise ValueError('Cached image transform has a different photographic chart')
    old = {p['path_id']: p for p in original['paths']}
    differences = []
    pixels = []
    for path in atlas['paths']:
        if path['path_id'] not in old:
            continue
        a, b = np.asarray(path['pixels']), np.asarray(old[path['path_id']]['pixels'])
        if a.shape != b.shape:
            raise ValueError('Common photographic contour shape differs')
        differences.extend(np.linalg.norm(a-b, axis=1))
        pixels.extend(a)
    if len(pixels) < 3 or np.linalg.matrix_rank(np.c_[pixels, np.ones(len(pixels))]) != 3:
        raise ValueError('Common photographic chart lacks two-dimensional support')
    maximum = float(max(differences))
    if maximum > tolerance:
        raise ValueError('Cached transform is not in the same reference pixel chart')
    return dict(common_vertices=len(pixels), maximum_common_pixel_difference=maximum,
                source_image_sha256=atlas['image_sha256'])


def run(args):
    out = guard_project_write(args.output.resolve())
    if out.exists():
        raise FileExistsError(out)
    manifest = read(args.prepared/'joint_manifest.json')
    mapping = read(args.prepared/'mapping.json')
    atlases = read(args.atlases)['atlases']
    protected = {str(p.resolve()): sha(p) for p in
        [args.atlases, args.state, args.prepared/'joint_manifest.json',
         args.prepared/'mapping.json', args.prepared/'point_observations.npz',
         Path(PrintPattern.read_paths.__code__.co_filename), Path(__file__)]}
    if protected[str((args.prepared/'point_observations.npz').resolve())] != manifest['point_input_sha256']:
        raise ValueError('Prepared input identity differs')
    with np.load(args.state, allow_pickle=False) as state:
        if str(state['point_input_sha256']) != manifest['point_input_sha256']:
            raise ValueError('Prepared state identity differs')
    patterns = []
    for i, atlas in enumerate(atlases):
        if sha(atlas['image']) != atlas['image_sha256']:
            raise ValueError('Reference photograph changed')
        instance = atlas['instance']
        ids = ['print_instance/'+instance+'::'+p['path_id'] for p in atlas['paths']]
        if set(ids)-set(mapping['path_ids']):
            raise ValueError('New contour is absent from current native scene')
        protected[str(Path(atlas['image']).resolve())] = atlas['image_sha256']
        patterns.append(dict(atlas_index=i, instance=instance, atlas=atlas,
            pattern=PrintPattern(atlas, cv2.imread(atlas['image'])), path_ids=ids))
    binding = dict(schema='rinkspace.current_native_whole_print_RGB_capture.v1',
        prepared=str(args.prepared), state=str(args.state), state_sha256=sha(args.state),
        native_library=manifest['native_library'], source_atlases=str(args.atlases),
        source_atlases_sha256=sha(args.atlases), shots=args.shots,
        adapter_sha256=sha(__file__), reader_sha256=sha(PrintPattern.read_paths.__code__.co_filename),
        source_adapter_sha256=sha(iter_registration_source_frames.__code__.co_filename),
        query_spacing_px=args.query_spacing, retained_arc_spacing_px=48.,
        cached_transforms_are_search_only=True, old_UV_copied=False,
        native_coordinates_used_as_measurements=False, fit_eligible=False, fit_executed=False,
        patterns=[dict(atlas_index=p['atlas_index'], instance=p['instance'], path_ids=p['path_ids'],
            reference_image=p['atlas']['image'], reference_image_sha256=p['atlas']['image_sha256'])
            for p in patterns], cache_dependencies={})
    for sid in args.shots:
        cache = args.capture_root/sid
        previous = read(cache/'binding.json')
        old_atlas_path = Path(previous['source_atlases'])
        if sha(old_atlas_path) != previous['source_atlases_sha256']:
            raise ValueError('Archived photographic reference changed')
        old_atlases = read(old_atlas_path)['atlases']
        charts = []
        for item in patterns:
            atlas = item['atlas']; old_id = atlas['cached_atlas_index']
            match = next((p for p in previous['patterns'] if
                p['atlas_index']==old_id and p['instance']==item['instance']), None)
            if match is None:
                raise ValueError('Physical print instance absent from cached source run')
            charts.append(dict(atlas_index=item['atlas_index'], cached_atlas_index=old_id,
                instance=item['instance'], **validate_chart(atlas, old_atlases[old_id])))
        refs = [file_ref(cache/'binding.json'), file_ref(old_atlas_path),
                file_ref(cache/sid/'source_rows.jsonl'), file_ref(cache/sid/'registration.jsonl')]
        protected.update({r['path']:r['sha256'] for r in refs})
        binding['cache_dependencies'][sid] = dict(charts=charts, files=refs)
    out.mkdir(parents=True)
    write(out/'binding.json', binding)
    started = time.monotonic(); summaries = {}; cv2.setNumThreads(1)
    for sid in args.shots:
        folder=out/sid; folder.mkdir(); counts=Counter(); coverage=[]
        shot=next(s for s in manifest['shots'] if s['id']==sid)
        cache=args.capture_root/sid
        records=iter(iter_retry_records(cache,sid,shot))
        with (folder/'source_rows.jsonl').open('w',encoding='utf8') as sources,\
             (folder/'registration.jsonl').open('w',encoding='utf8') as registrations:
            for fi,frame in enumerate(iter_registration_source_frames(sid,bootstrap_root=args.bootstrap)):
                original, prior, source_line=next(records)
                validate_retry_frame(frame,original)
                image=frame['image']; mask=frame['source_exclusion_mask'].astype(np.uint8)*255
                pt=int(frame['pts']); notes=[]; hypotheses=[]
                source_binding=dict(source_sha256=frame['source_video_sha256'], source_pts=pt,
                    time_base='1/12800', shot_id=sid, event_id='ankara_2026',
                    rig_id=str(shot['camera_index']), installation_id=str(shot['camera_index']),
                    pixel_domain='native_distorted', width=1920, height=1080)
                for item in patterns:
                    atlas=item['atlas']; key=(atlas['cached_atlas_index'],item['instance'])
                    note=next((n for n in prior['patterns'] if (n['atlas_index'],n['instance'])==key),None)
                    if note is None or note['registration']['status']!='source_pattern_registered':
                        counts['no_registered_image_transform']+=1; continue
                    registration=copy.deepcopy(note['registration']); H=np.asarray(registration['H'])
                    initial=np.asarray(note['initial_H']); pattern=item['pattern']
                    corners=project(initial,pattern.points)
                    lo=np.maximum(np.floor(corners.min(0)-20),[0,0])
                    hi=np.minimum(np.ceil(corners.max(0)+21),[1920,1080])
                    geometry=registration_geometry_check(H,initial,pattern.points,warp_roi=(lo,hi))
                    registration['registration_geometry']=geometry
                    if not geometry['valid']:
                        registration['status']='cached_geometry_invalid'; rows=[]
                        detail=dict(reason='cached_geometry_invalid',geometry=geometry)
                    else:
                        rows,detail=pattern.read_paths(image,mask,registration,source_binding,spacing=args.query_spacing)
                    label=f"{item['atlas_index']}/{item['instance']}"
                    for row in rows:
                        suffix=row['path_id'].split('::',1)[1]
                        row.update(path_id='print_instance/'+item['instance']+'::'+suffix,
                            entity=item['instance'], capture_family='whole_print_original_RGB_glyph_boundary',
                            source_atlas_index=item['atlas_index'], source_atlas=str(args.atlases),
                            source_registration_label=label,
                            source_identity_basis='Same physical instance and photographic chart; current RGB structure and signed profiles remeasured')
                    notes.append(dict(atlas_index=item['atlas_index'],instance=item['instance'],
                        initial_H=initial.tolist(),registration=registration,source_readout=detail,
                        readout_origin='current_RGB_cached_image_search',
                        cached_capture=str(cache),cached_registration_line=source_line,
                        cached_atlas_index=atlas['cached_atlas_index']))
                    hypotheses.append(dict(label=label,registration=registration,rows=rows,
                        origin='current_RGB_cached_image_search'))
                    counts['transforms_rechecked']+=1; counts['patterns_with_actual_rows']+=bool(rows)
                retained,unthinned,choices=choose_readouts(hypotheses)
                record=dict(pts=pt,source_BGR_sha256=frame['source_BGR_sha256'],
                    source_video_sha256=frame['source_video_sha256'],mask_path=str(frame['mask_path']),
                    mask_sha256=frame['mask_sha256'],mask_sources=frame['mask_sources'],
                    source_exclusion_sha256=frame['source_exclusion_sha256'],rows=retained,
                    fit_eligible=False,review_state='current_RGB_candidates_pending_review',
                    original_RGB_measurements=True,model_UV_synthesized=False)
                sources.write(json.dumps(plain(record),ensure_ascii=False,allow_nan=False)+'\n')
                registrations.write(json.dumps(plain(dict(pts=pt,patterns=notes,
                    all_actual_readouts=unthinned,path_choices=choices)),ensure_ascii=False,allow_nan=False)+'\n')
                counts['frames']+=1; counts['source_rows']+=len(retained); counts['frames_with_rows']+=bool(retained)
                coverage.append(dict(pts=pt,rows=len(retained),paths=sorted(choices)))
                if retained and counts['frames_with_rows']%150==1:
                    marked=image.copy()
                    for row in retained: cv2.circle(marked,tuple(np.rint(row['uv']).astype(int)),2,(0,0,255),-1)
                    cv2.imwrite(str(folder/f'{pt}_source.png'),image)
                    cv2.imwrite(str(folder/f'{pt}_glyphs.png'),marked)
                if fi%100==0:
                    print(json.dumps(dict(shot=sid,frames=fi+1,pts=pt,current_rows=len(retained),
                        total_rows=counts['source_rows'],seconds=time.monotonic()-started)),flush=True)
        if next(records,None) is not None: raise ValueError('Source cache extends beyond current shot')
        summaries[sid]=dict(counts);write(folder/'coverage.json',dict(counts=dict(counts),frames=coverage))
    changed=[p for p,s in protected.items() if sha(p)!=s]
    write(out/'summary.json',dict(shots=summaries,seconds=time.monotonic()-started,
        changed_dependencies=changed,fit_executed=False,fit_eligible=False,old_UV_copied=False))
    if changed: raise RuntimeError('A current source dependency changed during readout')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','bootstrap','atlases','capture-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shots',nargs='+',required=True)
    parser.add_argument('--query-spacing',type=float,default=8.)
    run(parser.parse_args())
