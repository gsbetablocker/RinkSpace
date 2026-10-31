"""Package current public-reader board material measurements without a fit.

The caller supplies a source-role-reviewed capture. Existing RGB reader accepts,
source exclusion, operators and actual UV remain unchanged. Ambiguous front-face
navigation stays pending; failed/empty source groups cannot delete old points.
"""
from pathlib import Path
import argparse,json
from collections import Counter
import cv2,numpy as np

from capture_source_board_pilot import sha,write,plain,HERE
from build_independent_shot_packets import capture_rows,json_line
from rinkspace.path_policy import guard_project_write
from rinkspace.vision_source_roles import organize_source_line_material_branches

OPERATORS={'board/continuous/bottom':'public_actual_RGB_first_iceward_yellow_material_fall_q50',
           'board/continuous/top':'public_actual_RGB_blue_cap_upper_entry_signed_peak'}


def top_source_support(top):
    """Organize actual entries using their measured optical support intervals.

    The current far-board cap is a source line family. Neither its organizing
    line nor its coefficients supply observations. This withholds colored
    spectators connected to the cap without using any camera/model residual.
    """
    with np.load(top['actual_original_RGB_profiles']['path'],allow_pickle=False)as profile:
        anchors=profile['anchors_native_uv'];normals=profile['normals_unit']
    rows=[dict(source_search_point_uv=p.tolist(),all_actual_source_material_candidates=[])for p in anchors]
    for i,(accepted,evidence)in enumerate(zip(top['primary_role_samples'],top['source_material_role_evidence'])):
        if not accepted:continue
        qi=evidence['source_query_index'];uv=np.asarray(top['actual_source_blue_rail_entry_uv'][i]);normal=normals[qi]
        interval=np.asarray(evidence['entry_observed_same_signed_material_transition']['actual_native_normal_interval_px'])
        interval=interval-(uv-anchors[qi])@normal
        rows[qi]['all_actual_source_material_candidates'].append(dict(actual_source_uv=uv.tolist(),
            disposition='measured_material_transition',observed_normal_interval_px=interval.tolist(),
            source_search_normal=normal.tolist()))
    result=organize_source_line_material_branches(rows,
        navigation_tangent_uv=np.tile([1.,0.],(len(rows),1)),source_visible_paths=[list(range(len(rows)))])
    return set(map(int,result['source_query_indices'])),result


def qualify(capture,output):
    capture=Path(capture).resolve(strict=True);output=guard_project_write(output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads((capture/'summary.json').read_text('utf8'))
    requested=summary['only_requested_pts'];raw_path=capture/'pending_source_rows.jsonl'
    records=list(map(json.loads,raw_path.read_text('utf8').splitlines()))
    if [r['pts']for r in records]!=requested:raise ValueError('Capture does not cover every requested native exposure')
    source_sha=summary['source_video_sha256'];authority=summary['authority']
    if sha(authority['reference'])!=authority['contract_sha256']:raise ValueError('Board role authority changed')
    protected={str(raw_path):sha(raw_path),str(capture/'summary.json'):sha(capture/'summary.json')}
    output.mkdir(parents=True);folder=output/'qualified'/'s07';folder.mkdir(parents=True)
    line_proofs=output/'source_line_support';line_proofs.mkdir()
    qualified=folder/'source_rows.jsonl';coverage=[];counts=Counter();by_path=Counter();hashes={}
    def verify(path,expected):
        path=str(Path(path).resolve())
        if path not in hashes:hashes[path]=sha(path)
        if hashes[path]!=expected:raise ValueError('Current source dependency SHA differs: '+path)
    with qualified.open('w',encoding='utf8')as stream:
        for record in records:
            pts=record['pts'];verify(record['source_image'],record['source_image_sha256'])
            image=cv2.imread(record['source_image'])
            import hashlib
            if image is None or image.shape!=(1080,1920,3)or hashlib.sha256(image.tobytes()).hexdigest()!=record['source_BGR_sha256']:
                raise ValueError('Decoded original RGB identity differs')
            verify(record['mask_path'],record['mask_sha256'])
            evidence_root=Path(record.get('source_readout_root',capture)).resolve(strict=True)
            nav_path=evidence_root/f'{pts}_navigation.json';navigation=json.loads(nav_path.read_text('utf8'))
            protected[str(nav_path)]=sha(nav_path)
            front_source=(navigation['kickplate'].get('status')=='long_source_kick_plate_supported')
            bottom_path=evidence_root/f'{pts}_bottom_readout.json';top_path=evidence_root/f'{pts}_top_readout.json'
            bottom=json.loads(bottom_path.read_text('utf8'))if bottom_path.exists()else None
            top=json.loads(top_path.read_text('utf8'))if top_path.exists()else None
            top_queries,top_organization=top_source_support(top)if top is not None else(set(),None)
            if top_organization is not None:write(line_proofs/f'{pts}.json',top_organization)
            for path in [bottom_path,top_path]:
                if path.exists():protected[str(path)]=sha(path)
            good=[];pending=[]
            for row in record['rows']:
                path=row['path_id'];operator=row['measurement_operator'];uv=np.asarray(row['uv']);qi=row['source_query_index']
                if path not in OPERATORS or operator!=OPERATORS[path]:raise ValueError('Board material operator/path identity differs')
                if (row.get('model_UV_synthesized')is not False or uv.shape!=(2,)or not np.isfinite(uv).all()
                        or abs(uv[0]/48-round(uv[0]/48))>1e-6):raise ValueError('Only unchanged actual 48px RGB readouts may be packaged')
                proof=row['source_profile_proof'];verify(proof['path'],proof['sha256'])
                reasons=[]
                if path.endswith('/bottom'):
                    if not front_source:reasons.append('current_front_face_yellow_material_role_navigation_unresolved')
                    cap=[r['uv']for r in record['rows']if r['path_id']=='board/continuous/top']
                    if not any(point[1]<uv[1]for point in cap):
                        reasons.append('no_current_source_blue_cap_above_this_yellow_foot_hypothesis')
                    if bottom is None:raise ValueError('Missing actual contact readout')
                    old_ids=bottom['old_query_indices'];matches=[i for i,j in enumerate(old_ids)if j==qi]
                    exact=(len(matches)==1 and bottom['old_query_supported'][matches[0]]is True
                        and bottom['source_material_accepted'][qi]is True and bottom['candidate_source_uv'][qi]==row['uv']
                        and not bottom['source_mask_conflict'][qi]and bottom['competing_same_band_pairs'][qi]==0)
                    if not exact:raise ValueError('Contact UV differs from the public accepted q50/support')
                else:
                    if top is None:raise ValueError('Missing actual blue-entry readout')
                    matches=[i for i,e in enumerate(top['source_material_role_evidence'])if e['source_query_index']==qi
                        and top['primary_role_samples'][i]is True and top['actual_source_blue_rail_entry_uv'][i]==row['uv']]
                    if len(matches)!=1:raise ValueError('Top UV differs from the public accepted signed entry peak')
                    evidence=top['source_material_role_evidence'][matches[0]]
                    if evidence['entry_role_hypothesis']!='board_upper_rim_blue_material_entry':raise ValueError('Internal cap exit cannot become board top')
                    if qi not in top_queries:
                        reasons.append('actual_blue_entry_not_in_unique_distributed_source_cap_material_family')
                    if not front_source:
                        # A missing yellow proposal does not erase a visible
                        # cap. Separate lower blue ice/near-board components
                        # cannot inherit the reviewed upper front-face role.
                        components=navigation['blue_components']
                        cap=max(components,key=lambda c:c['bbox'][2]-c['bbox'][0])if components else None
                        seed=evidence['original_source_navigation_seed_uv']
                        if cap is None or not(cap['bbox'][0]<=seed[0]<cap['bbox'][2]and cap['bbox'][1]<=seed[1]<cap['bbox'][3]):
                            reasons.append('separate_blue_material_branch_has_no_reviewed_front_face_cap_identity')
                if reasons:pending.append(dict(row,qualification_reasons=reasons))
                else:good.append(dict(row,fit_eligible=True,review_state='current_RGB_material_source_supported',
                    source_UV_unchanged=True,human_acceptance=False))
            record.update(rows=good,pending_rows=pending,fit_eligible=bool(good),
                review_state='machine_source_supported_board_materials'if good else'pending_no_supported_board_materials',
                qualification=dict(source_capture=str(raw_path),source_capture_sha256=protected[str(raw_path)],
                    source_role_authority=authority,actual_UV_and_operators_unchanged=True,native_residual_used=False,
                    native_or_H_observations_generated=False,current_source_profile_and_mask_hashes_verified=True,
                    human_acceptance=False,accuracy_claim=False))
            stream.write(json_line(record));by_path.update(r['path_id']for r in good)
            counts['source_exposures']+=1;counts['qualified_exposures']+=bool(good);counts['rows']+=len(good)
            counts['pending_rows']+=len(pending);counts['empty_exposures_keep_old_groups']+=not bool(good)
            coverage.append(dict(pts=pts,rows_by_path=dict(Counter(r['path_id']for r in good)),pending_rows=len(pending),
                bottom_status='current_profile_read'if bottom is not None else'insufficient_yellow_navigation',
                top_status='current_profile_read'if top is not None else'insufficient_blue_navigation'))
    packet=output/'packets'/'s07';packet.mkdir(parents=True)
    captured=capture_rows(qualified,'s07',np.asarray(requested,np.int64));points=captured['points']
    paths=list(OPERATORS);lookup={path:i for i,path in enumerate(paths)}
    z=dict(pts=np.array([r['pts']for r in points],np.int64),
        observation_ids=np.array([r['observation_id']for r in points],dtype='U96'),
        source_rows=np.array([r['source_row']for r in points],np.int64),
        path_indices=np.array([lookup[r['path_id']]for r in points],np.int32),
        uv=np.array([r['uv']for r in points],np.float64),kind=np.ones(len(points),np.int64),stations=np.full(len(points),.5))
    np.savez(packet/'point_observations.npz',**z)
    (packet/'source_row_manifest.jsonl').write_text(''.join(map(json_line,captured['locations'])),encoding='utf8')
    mapping=dict(shot_id='s07',path_ids=paths,source_video_sha256=source_sha,independent_source_capture=True,fit_eligible=True,
        pixel_domain='native_distorted',image_size=[1920,1080],point_input_sha256=sha(packet/'point_observations.npz'),
        source_capture_jsonl=str(qualified),source_capture_sha256=captured['source_sha256'],
        source_row_manifest='source_row_manifest.jsonl',source_row_manifest_sha256=sha(packet/'source_row_manifest.jsonl'),
        measurement_operators=OPERATORS,final_source_spacing_px=48.,dense_search_queries_are_observations=False,
        source_UV_synthesized=False,source_measurements_and_pts_generated=False,empty_groups_delete_old_measurements=False)
    write(packet/'mapping.json',mapping)
    for path,expected in {**protected,**hashes}.items():
        if sha(path)!=expected:raise ValueError('Current source qualification dependency changed: '+path)
    result=dict(schema='rinkspace.source_board_capture_qualification.v1',capture=str(capture),requested_pts=requested,
        counts=dict(counts),rows_by_path=dict(by_path),coverage=coverage,packet=str(packet),
        point_input_sha256=mapping['point_input_sha256'],source_video_sha256=source_sha,measurement_operators=OPERATORS,
        source_UV_unchanged=True,public_reader_thresholds_changed=False,uniform_source_query_spacing_px=48.,
        old_observations_loaded=False,native_state_loaded=False,H_or_native_UV_generated=False,fit_executed=False,
        original_pending_capture_preserved=True,empty_groups_delete_old_measurements=False)
    write(output/'coverage_summary.json',result)
    print(json.dumps(dict(output=str(output),counts=dict(counts),rows_by_path=dict(by_path),packet=str(packet))),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args.capture,args.output)
