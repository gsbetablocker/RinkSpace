"""Admit original RGB ice midlines under explicit user-confirmed identities."""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,cv2,numpy as np
import prepare_native_multicamera as p
from prepare_spatial_refinement import collect_additions,revise_observations,_initialize_touched,_preserve_old_rows,_write_prepared
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');original=p.npz(a.prepared/'point_observations.npz');state=p.npz(a.state);model=SceneParameterModel(p.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);p.validate_snapshot(state,model.initial_state.layout,mapping,m,original)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    identity=p.read(a.identity);capture=p.read(a.capture/'binding.json');pixels=p.read(a.capture/'s08/source_pixels.json');review=p.read(a.review)
    if identity.get('user_identity_confirmed') is not True:raise ValueError('User-confirmed identities are required')
    allowed={r['native_path_id'] for r in identity['curves']}
    if {r['native_path_id'] for r in capture['inventory']}!=allowed:raise ValueError('Capture inventory does not match the user identities')
    bad_new={(int(r['pts']),r['path_id'],int(r['candidate'])):r for r in review.get('exclude_new',[])};keep_new=None if 'keep_new' not in review else {(int(r['pts']),r['path_id'],int(r['candidate'])) for r in review['keep_new']};old_by_id={str(v):i for i,v in enumerate(original['observation_ids'])};keep=np.ones(len(original['pts']),bool);removed=[]
    for decision in review.get('exclude_old',[]):
        i=old_by_id[decision['observation_id']]
        if int(original['pts'][i])!=decision['pts'] or not np.array_equal(original['uv'][i],decision['uv']):raise ValueError('Reviewed old UV/PTS changed')
        keep[i]=False;removed.append(dict(decision,parent_row=int(i),source_row=int(original['source_rows'][i])))
    points={k:v[keep] for k,v in original.items()};numeric=state['stations'][keep].copy();target=deepcopy(m);begin=0
    for shot in target['shots']:
        count=int(keep[shot['row_begin']:shot['row_stop']].sum());shot['row_begin']=begin;shot['row_stop']=begin+count;begin+=count
    parent_state={k:v.copy() for k,v in state.items()};parent_state['stations']=numeric;parent_state['source_rows']=points['source_rows'].copy();accepted=[];excluded=[];profiles={};images={}
    for row in pixels['accepted']:
        key=(row['pts'],row['path_id'],row['candidate'])
        if row['path_id'] not in allowed or row.get('model_UV_synthesized') is not False:raise ValueError('Source identity or UV origin changed')
        if key in bad_new:excluded.append(dict(pts=row['pts'],path_id=row['path_id'],candidate=row['candidate'],uv=row['uv'],review_decision=bad_new[key]));continue
        if keep_new is not None and key not in keep_new:excluded.append(dict(pts=row['pts'],path_id=row['path_id'],candidate=row['candidate'],uv=row['uv'],reason='Retained as a source candidate; not in this explicitly reviewed pixel-support selection'));continue
        ref=Path(row['source_reference'])
        if str(ref) not in profiles:profiles[str(ref)]=(p.npz(ref),p.digest(ref))
        raw,raw_sha=profiles[str(ref)];cid=row['candidate']
        if not np.array_equal(raw['candidate_uv'][cid],row['uv']) or not np.array_equal(raw['candidate_sides'][cid],row['actual_source_sides']) or not np.array_equal(raw['candidate_side_quantiles_uv'][cid],row['actual_source_quantiles']):raise ValueError('Original RGB readout changed')
        if int(raw['candidate_flags'][cid]) & 3:raise ValueError('Invalid or masked public source profile')
        evidence=next((r/(str(row['pts'])+'_original.png') for r in a.evidence_roots if (r/(str(row['pts'])+'_original.png')).is_file()),None)
        if evidence is None:raise ValueError('Missing exact original source evidence')
        if str(evidence) not in images:
            image=cv2.imread(str(evidence))
            if image is None:raise ValueError('Source PNG cannot be decoded')
            images[str(evidence)]=hashlib.sha256(image.tobytes()).hexdigest()
        if images[str(evidence)]!=row['source_BGR_sha256'] or row['source_video_sha256']!=identity['source_video_sha256']:raise ValueError('Original source image identity changed')
        new=deepcopy(row);new.update(evidence_image=str(evidence),kind=1,fit_eligible=True,source_UV_synthesized=False,source_profile_proof=dict(path=str(ref),sha256=raw_sha),user_identity_reference=str(a.identity),user_identity_reference_sha256=p.digest(a.identity),user_identity_confirmed=True,user_pixel_accuracy_accepted=False,user_visual_acceptance=False);accepted.append(new)
    if not accepted:raise ValueError('No eligible original RGB ice midlines')
    links=a.capture/'qualified_ice_source_links.json';p.write(links,dict(user_identity=dict(path=str(a.identity),sha256=p.digest(a.identity)),source_capture=dict(path=str(a.capture/'binding.json'),sha256=p.digest(a.capture/'binding.json')),source_temporal_links=dict(path=str(a.capture/'s08/source_links.json'),sha256=p.digest(a.capture/'s08/source_links.json')),source_pixel_review=dict(path=str(a.review),sha256=p.digest(a.review)),profiles=[dict(path=k,sha256=v[1]) for k,v in profiles.items()],original_source_pngs=[dict(path=k,decoded_BGR_sha256=v) for k,v in images.items()],guide_UV_never_used_as_measurements=True,pending_rows_not_promoted=True,source_UV_synthesized=False))
    packet=a.capture/'qualified_ice_packet.json';p.write(packet,dict(shot='s08',fit_eligible=True,source_UV_synthesized=False,source_packet=str(a.capture/'s08/source_pixels.json'),source_packet_sha256=p.digest(a.capture/'s08/source_pixels.json'),source_links=str(links),source_links_sha256=p.digest(links),accepted=accepted,user_identity_confirmed=True,user_pixel_accuracy_accepted=False,user_visual_acceptance=False))
    additions,aliases,packets=collect_additions([packet],points,target,mapping['path_ids']);new,numeric,shots,touched,decisions,added,origins=revise_observations(points,numeric,target,mapping['path_ids'],dict(decisions=[],added_observations=additions))
    if decisions:raise ValueError('Source ice append cannot silently revise old measurements')
    notes=_initialize_touched(new,numeric,shots,touched,parent_state,target,mapping,model);_preserve_old_rows(new,points,numeric,parent_state['stations'],origins)
    receipt=dict(schema='rinkspace.user_confirmed_ice_source_preparation.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),parent_input_sha256=m['point_input_sha256'],added_rows=len(additions),removed_old_rows=removed,excluded_new_rows=excluded,original_new_rows=len(pixels['accepted']),pending_rows_not_promoted=len(pixels['pending']),all_retained_observation_fields_byte_preserved=True,all_retained_numeric_stations_byte_preserved=True,all_numeric_globals_unchanged=True,new_camera_parameters=0,new_geometry_parameters=0,source_packets=packets,duplicate_aliases=aliases,station_initialization=notes,user_identity_reference=str(a.identity),user_identity_reference_sha256=p.digest(a.identity),fit_started=False)
    result,mapped=_write_prepared(out,new,numeric,shots,parent_state,target,mapping,model,receipt)
    if 'spatial_refinement_preparation' in m:result['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:result.pop('spatial_refinement_preparation',None)
    result['user_confirmed_ice_source_preparation']=dict(receipt_path=str(out/'source_ice_preparation.json'),added_rows=len(additions),removed_old_rows=len(removed),new_geometry_parameters=0);p.write(out/'joint_manifest.json',result);p.write(out/'source_ice_preparation.json',receipt)
    print(json.dumps(dict(output=str(out),added_rows=len(additions),removed_old_rows=len(removed),total_rows=len(new['pts']),point_input_sha256=result['point_input_sha256'],new_camera_parameters=0,new_geometry_parameters=0)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','capture','identity','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--evidence-roots',type=Path,nargs='+',required=True);main(parser.parse_args())
