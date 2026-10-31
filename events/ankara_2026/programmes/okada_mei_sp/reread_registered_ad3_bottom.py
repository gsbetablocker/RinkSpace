"""Reread current source-supported Ad3 bottoms with complete RGB context.

Source registrations and current SAM2 masks are reused. The public panel reader
and print qualifier are unchanged. Packets contain only measured Ad3.bottom_edge;
no camera/model state, residual selection, pose fitting or deletion is involved.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
from collections import Counter
import cv2,numpy as np

from rinkspace.geometry import project
from rinkspace.path_policy import guard_project_write
from rinkspace.session.registered_panel_edges import read_registered_panel_edges
from prepare_independent_navigation import frames
from capture_independent_prints import material_guard
from capture_independent_materials import plain,source_exclusion,spaced_indices
from build_independent_shot_packets import capture_rows,json_line,native_pts
import qualify_independent_prints as qualifier

HERE=Path(__file__).resolve().parent
PATH_ID='print_instance/Ad3::Ad3.bottom_edge'


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def capture(root,out,index,shots):
    parent=root/'ad3_repair_full'
    bank_path=root/'print_references/references.json'
    bank=json.loads(bank_path.read_text('utf8'))
    references={}
    for meta in bank['references']:
        if meta['ad_id']!='Ad3':continue
        if meta['native_edge_path_ids'][2]!=PATH_ID:raise ValueError('Current Ad3 bottom path identity differs')
        with np.load(meta['descriptor_npz'],allow_pickle=False)as z:
            if sha(meta['descriptor_npz'])!=meta['descriptor_npz_sha256']:raise ValueError('Source reference descriptor changed')
            offsets=z['edge_offsets'];uv=z['projected_uv']
            references[meta['pts']]=(meta,[uv[offsets[i]:offsets[i+1]]for i in range(4)])
    source=Path(index['source']['resolved_path']);source_sha=sha(source)
    if source_sha!=index['source']['actual_sha256']:raise ValueError('Registered original hash differs')
    implementation={str(p):sha(p)for p in [Path(__file__),HERE/'qualify_independent_prints.py',
        HERE/'capture_independent_prints.py',HERE/'capture_independent_materials.py',
        HERE.parents[3]/'src/rinkspace/session/registered_panel_edges.py',
        HERE.parents[3]/'src/rinkspace/vision_programme_ice_review.py']}
    binding=dict(schema='rinkspace.independent_material_capture.v1',source_sha256=source_sha,
        source_video=str(source),independent_source_capture=True,appearance_references=str(bank_path),
        reference_bank_sha256=sha(bank_path),source_registration=str(parent),
        selected_shot_ids=[s['shot_id']for s in shots],selected_entities=['Ad3'],selected_paths=[PATH_ID],
        native_state_loaded=False,no_native_residual_selection=True,native_camera_prediction_is_identity_evidence=False,
        source_feature_match_is_boundary_measurement=False,search_radius_px=40.,query_spacing_px=24.,
        final_source_spacing_px=48.,source_qualification_criteria_changed=False,fit_eligible=False,
        empty_groups_delete_old_measurements=False,implementation_sha256=implementation)
    out.mkdir();write(out/'binding.json',binding);summaries={};started=time.monotonic()
    for shot in shots:
        sid=shot['shot_id'];reg_path=parent/sid/'registration.jsonl';reg_sha=sha(reg_path);cached={}
        for line_number,line in enumerate(reg_path.open(encoding='utf8'),1):
            frame=json.loads(line)
            notes=[n for n in frame['prints']if n.get('ad')=='Ad3'and n.get('status')=='source_content_registered'
                   and n.get('source_pattern_check',{}).get('supported')is True]
            if len(notes)!=1:continue
            n=notes[0];chart=n.get('source_readout',{}).get('source_search_chart')or{}
            if chart.get('paths')is not None:paths=chart['paths'];basis='current_actual_RGB_component_rectangle'
            else:
                _,reference_paths=references[n['reference_pts']]
                paths=[project(np.asarray(n['H_current_from_reference']),p).tolist()for p in reference_paths]
                basis='existing_independent_source_pattern_registration_navigation'
            cached[frame['pts']]=dict(reference_pts=n['reference_pts'],paths=paths,navigation_basis=basis,
                source_pattern_check=n['source_pattern_check'],source_registration_line=line_number)
        old_path=parent/sid/'source_rows.jsonl';old_sha=sha(old_path)
        old={r['pts']:r['source_BGR_sha256']for r in map(json.loads,old_path.open(encoding='utf8'))}
        folder=out/sid;folder.mkdir();proof_dir=folder/'profiles';proof_dir.mkdir()
        counts=Counter();coverage=[]
        with (folder/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)as raw_out, \
             (folder/'registration.jsonl').open('w',encoding='utf8',buffering=1)as reg_out:
            for pts,bgr in frames(index,shot):
                counts['native_frames']+=1
                if pts not in cached:continue
                item=cached[pts];bgr_sha=hashlib.sha256(bgr.tobytes()).hexdigest()
                if old.get(pts)!=bgr_sha:raise ValueError('Current source frame differs from registered readout')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mask,mp=source_exclusion(root,sid,pts);arrays={}
                def collect(stage,data):arrays.update({stage+'__'+k:v for k,v in data.items()})
                actual,record=read_registered_panel_edges(rgb,[np.asarray(p)for p in item['paths']],mask,
                    binding=dict(source_sha256=source_sha,source_pts=pts,time_base='1/12800',shot_id=sid,
                        independent_source_pattern_match_verified=True),search_radius=40.,spacing=24.,
                    candidate_material_validator=lambda row:material_guard(rgb,row,'Ad3')[0],proof_sink=collect)
                bottom=[r for r in actual if r['curve']==2]
                proof_path=proof_dir/f'{pts}.npz';np.savez_compressed(proof_path,**arrays)
                proof=dict(path=str(proof_path),sha256=sha(proof_path),profile_arrays_prefix='actual_source_envelope_search__')
                selected={c['candidate_id']:c for c in record['candidate_records']if c['candidate_id']in record['selected_candidate_ids']and c['curve']==2}
                rows=[]
                for row in bottom:
                    exact=[c for c in selected.values()if c['uv']==row['uv']]
                    if len(exact)!=1:raise ValueError('Actual bottom readout is not unique')
                    c=exact[0]
                    if row['uv']!=row['actual_source_quantiles'][0][1]:raise ValueError('UV no longer equals source q50')
                    rows.append(dict(row,path_id=PATH_ID,entity='Ad3',source_reference_pts=item['reference_pts'],
                        source_candidate_id=c['candidate_id'],source_profile_query=c['stage_query'],
                        source_topology_probes=c['source_topology_probes'],actual_source_sides=c['actual_source_sides'],
                        source_colour=material_guard(rgb,row,'Ad3')[1],source_profile_proof=proof,
                        capture_family='registered_Ad3_bottom_complete_profile_context'))
                # Retain all actual bottom candidates/rejections, but do not mix
                # other edges into this current capture or packet.
                record['candidate_records']=[c for c in record['candidate_records']if c['curve']==2]
                record['selected_candidate_ids']=sorted(selected)
                record['reports']=[r for r in record['reports']if r['curve']==2]
                record['pending_candidates']=[c for c in record['candidate_records']if c['candidate_id']not in selected]
                record['profile_debug']=[r for r in record['profile_debug']if r['curve']==2]
                record['candidate_count']=len(record['candidate_records'])
                note=dict(ad='Ad3',reference_pts=item['reference_pts'],status='source_content_registered',
                    source_pattern_check=item['source_pattern_check'],source_readout=record,profile_proof=proof,
                    source_registration=str(reg_path),source_registration_sha256=reg_sha,
                    source_registration_line=item['source_registration_line'],source_navigation_basis=item['navigation_basis'])
                raw_out.write(json_line(dict(pts=pts,source_BGR_sha256=bgr_sha,source_video_sha256=source_sha,
                    fit_eligible=False,review_state='pending_unchanged_public_print_qualification',rows=rows,pending_rows=[],
                    mask_path=str(mp),mask_sha256=sha(mp),source_profile_proof=proof)))
                reg_out.write(json_line(plain(dict(pts=pts,prints=[note],fit_eligible=False))))
                counts['source_supported_frames']+=1;counts['organized_frames']+=bool(rows);counts['organized_rows']+=len(rows)
                coverage.append(dict(pts=pts,organized_bottom_rows=len(rows),empty_keeps_old_group=not bool(rows)))
                if counts['source_supported_frames']%100==0:
                    print(json.dumps(dict(stage='bottom_capture',shot=sid,counts=dict(counts),elapsed_s=round(time.monotonic()-started,1))),flush=True)
        shutil.copyfile(parent/sid/'poses.npz',folder/'poses.npz')
        if sha(reg_path)!=reg_sha or sha(old_path)!=old_sha:raise ValueError('Current source registration changed')
        summaries[sid]=dict(counts,rows_path=str(folder/'source_rows.jsonl'),registration=str(reg_path),registration_sha256=reg_sha)
        write(folder/'coverage.json',dict(summary=summaries[sid],frames=coverage))
    for path,expected in implementation.items():
        if sha(path)!=expected:raise ValueError('Capture implementation changed')
    return binding,summaries


def qualify(root,capture_path,output):
    # Existing qualifier and thresholds; cache only immutable file hashes during
    # this phase, and verify all cached dependencies again before returning.
    hashes={};original_sha=qualifier.sha;original_argv=sys.argv
    def cached_sha(path):
        path=str(Path(path).resolve())
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]
    try:
        qualifier.sha=cached_sha
        sys.argv=['qualify_independent_prints.py','--root',str(root),'--capture',str(capture_path),
                  '--output',str(output),'--shot-ids','s07,s09']
        qualifier.main()
    finally:qualifier.sha=original_sha;sys.argv=original_argv
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError('Qualification input changed')


def package(index,shots,qualified,out,binding):
    final=out/'qualified_spaced';packets=out/'packets';final.mkdir();packets.mkdir();summaries=[];offset=0
    for shot in shots:
        sid=shot['shot_id'];folder=final/sid;folder.mkdir();count=Counter();capture_path=folder/'source_rows.jsonl'
        with capture_path.open('w',encoding='utf8')as stream:
            for frame in map(json.loads,(qualified/sid/'source_rows.jsonl').open(encoding='utf8')):
                rows=frame['rows']
                if any(r['path_id']!=PATH_ID for r in rows):raise ValueError('Non-bottom row entered Ad3 bottom packet')
                indices=spaced_indices([r['uv']for r in rows],spacing=48.)if rows else []
                kept=[rows[int(i)]for i in indices]
                frame.update(rows=kept,fit_eligible=bool(kept),final_source_spacing_px=48.,
                    qualification_precedes_spacing=True,qualified_rows_before_spacing=len(rows))
                stream.write(json_line(frame));count['source_supported_frames']+=1
                count['qualified_frames']+=bool(kept);count['qualified_rows']+=len(kept);count['empty_frames_keep_old_group']+=not bool(kept)
        rows=capture_rows(capture_path,sid,native_pts(shot['processing']),offset);offset+=len(rows['points'])
        packet=packets/sid;packet.mkdir();points=rows['points']
        z=dict(pts=np.array([r['pts']for r in points],np.int64),
            observation_ids=np.array([r['observation_id']for r in points],dtype='U96'),
            path_indices=np.zeros(len(points),np.int32),uv=np.array([r['uv']for r in points],np.float64),
            kind=np.ones(len(points),np.int64),stations=np.full(len(points),.5),
            source_rows=np.array([r['source_row']for r in points],np.int64))
        np.savez(packet/'point_observations.npz',**z)
        (packet/'source_row_manifest.jsonl').write_text(''.join(map(json_line,rows['locations'])),encoding='utf8')
        mapping=dict(shot_id=sid,path_ids=[PATH_ID],source_video_sha256=index['source']['actual_sha256'],
            independent_source_capture=True,fit_eligible=True,pixel_domain='native_distorted',image_size=[1920,1080],
            point_input_sha256=sha(packet/'point_observations.npz'),
            source_capture_jsonl=str(capture_path),source_capture_sha256=rows['source_sha256'],
            source_row_manifest='source_row_manifest.jsonl',source_row_manifest_sha256=sha(packet/'source_row_manifest.jsonl'),
            source_UV_synthesized=False,source_measurements_and_pts_generated=False,
            final_source_spacing_px=48.,qualification_precedes_spacing=True,empty_groups_delete_old_measurements=False)
        write(packet/'mapping.json',mapping)
        summaries.append(dict(shot_id=sid,packet=str(packet),native_frames=shot['processing']['native_frame_count'],
            **count,point_input_sha256=mapping['point_input_sha256']))
    summary=dict(schema='rinkspace.Ad3_bottom_source_reread.v1',source_video_sha256=index['source']['actual_sha256'],
        selected_path=PATH_ID,search_radius_px=40.,query_spacing_px=24.,final_source_spacing_px=48.,
        source_qualification_criteria_changed=False,qualification_precedes_spacing=True,
        no_native_residual_selection=True,native_state_loaded=False,pixel_fit_executed=False,
        old_source_files_preserved=True,empty_groups_delete_old_measurements=False,shots=summaries,
        source_binding=binding,qualified_root=str(final),packets_root=str(packets))
    write(out/'coverage_summary.json',summary)
    print(json.dumps(dict(stage='complete',output=str(out),shots=summaries)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True);out=guard_project_write(args.output)
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(HERE.parents[3]):raise ValueError('Source artifacts belong outside OneDrive')
    cv2.setNumThreads(2);index=json.loads((root/'index.json').read_text('utf8'))
    shots=[s for s in index['shots']if s['shot_id']in ('s07','s09')]
    if [s['shot_id']for s in shots]!=['s07','s09']:raise ValueError('Require current S07/S09 source index')
    out.mkdir(parents=True)
    binding,_=capture(root,out/'capture',index,shots)
    qualify(root,out/'capture',out/'qualified')
    package(index,shots,out/'qualified',out,binding)


if __name__=='__main__':main()
