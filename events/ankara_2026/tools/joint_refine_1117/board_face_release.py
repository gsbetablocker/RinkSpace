"""Replay the material-qualified UV packets at their registered camera and source clock."""
from pathlib import Path

import numpy as np

from board_face_refine import FAR, MATERIALS, OUTPUT, PARENT, PRIOR, SHOTS
from deliver import file_ref
from lane_review import pixel_errors
from review import camera_state, ji, native_for
from shared import EVENT, OUT, PREPARED, PROGRAMMES, arrays, numeric_sha, prepared_packet_path, programme_state_sha, read, save, sha, write
from side_release_1118 import numbers
from side_repair import native_table


def fit_board_station(candidate,name,a,take):
    """At a fixed camera, profile a point only along the same local physical rail."""
    b=camera_state(candidate);n=native_for(candidate,name.split('_')[0])
    s=read(PREPARED/'setup.json')['shots'][name]
    error=np.full(len(a['idx']),np.nan);station=error.copy()
    ix=np.flatnonzero(take)
    paths,pi=np.unique(a['path'][ix],return_inverse=True)
    tab=native_table(n,paths,count=4097)
    world={k:v[0] for k,v in tab.items()}
    group=a['idx'][ix]*len(paths)+pi
    order=np.argsort(group,kind='stable')
    pieces=np.split(ix[order],np.flatnonzero(np.diff(group[order]))+1)
    for lo in range(0,len(pieces),32):
        batch=pieces[lo:lo+32]
        jobs=[(a['idx'][rr[0]],a['path'][rr[0]],b[name+'_times'][a['idx'][rr[0]]]) for rr in batch]
        pred=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for rr,(_,k,_),(uv,status) in zip(batch,jobs,pred,strict=True):
            tt=tab[k][1];ds=tt[1]-tt[0]
            centre=np.searchsorted(tt,a['station'][rr])-1
            radius=int(np.ceil(.035/ds))
            # The continuous rail wraps around joint zero. A cropped interval
            # near this point still belongs to the same physical branch.
            spans=(centre[:,None]+np.arange(-radius,radius+1))%(len(tt)-1)
            left=uv[spans];vec=uv[spans+1]-left
            offset=a['uv'][rr,None]-left
            norm=(vec*vec).sum(2)
            valid=(status[spans]==0)&(status[spans+1]==0)&(norm>1.e-12)
            f=((offset*vec).sum(2)/np.maximum(1.e-12,norm)).clip(0,1)
            d=np.where(valid,np.linalg.norm(offset-f[:,:,None]*vec,axis=2),np.inf)
            j=d.argmin(axis=1);rows=np.arange(len(rr));seg=spans[rows,j]
            error[rr]=d[rows,j];station[rr]=tt[seg]+f[rows,j]*ds
    if not np.isfinite(error[take]).all():
        raise ValueError(('Unobservable material in source',name))
    return error,station


def matched(candidate,name,folder=None):
    folder=Path(folder) if folder is not None else OUTPUT/'point_review'
    s=read(PREPARED/'setup.json')['shots'][name];a=arrays(prepared_packet_path(name))
    old_info=read(PRIOR/'source_bindings.json')[name]
    old=arrays(old_info['packet']['path']);base=arrays(old_info['matched_errors']['path'])
    count=len(old['idx'])
    for key in ('idx','pts','path','uv','source_row','source_packet','station_fixed'):
        np.testing.assert_array_equal(old[key],a[key][:count])
    used=a['fold']<9;new=np.arange(len(used))>=count
    b=camera_state(candidate);prev=camera_state(PARENT)
    np.testing.assert_array_equal(b['scene_values'],prev['scene_values'])
    np.testing.assert_array_equal(b[name+'_pts'][a['idx']],a['pts'])
    same=programme_state_sha(b,[name])==programme_state_sha(prev,[name])
    before=np.full(len(used),np.nan);before[:count]=base['error']
    station=before.copy();station[:count]=base['station']
    if new.any():
        e,t=pixel_errors(native_for(PARENT,s['program']),prev,s,name,a,used&new)
        before[new]=e[new];station[new]=t[new]
    if same:
        e=before.copy()
        mask=used&np.isin(a['family'],['board.top','board.bottom'])&np.isin(a['material'],MATERIALS)&~a['station_fixed']
        if mask.any():
            new_errors,new_stations=fit_board_station(candidate,name,a,mask)
            e[mask]=new_errors[mask];station[mask]=new_stations[mask]
    else:
        mask=used
        e,station=pixel_errors(native_for(candidate,s['program']),b,s,name,a,used)
        # Compare rank-one ink measurements on their declared image axis.
        from board_face_check import score
        for run,errors in ((candidate,e),(PARENT,before)):
            _,_,v,_=score(run,FAR,'far')
            ink=mask&np.char.startswith(a['material'],'ink_extent_')
            errors[ink]=abs(v[a['source_row'][ink]])
    if not np.isfinite(e[used]).all():
        raise ValueError(('A currently enabled source point could not be projected',name))
    fit=a['fold']<5;held=(a['fold']>=5)&used
    frame=b[name+'_pts']
    save(folder/(name+'.npz'),error=e,station=station,previous=before,pts=frame,
         coverage=np.bincount(a['idx'][fit],minlength=len(frame)))
    info=dict(candidate=candidate,state=programme_state_sha(b,[name]),packet_sha256=sha(prepared_packet_path(name)),
              source=s['source'],source_sha256=s['source_sha256'],frames=len(frame),old_prefix_rows=count,
              exact_camera_match=same,fitted_or_checked=int(mask.sum()),fit_before=numbers(before[fit]),
              fit_after=numbers(e[fit]),held_after=numbers(e[held]),held_before=numbers(before[held]),
              families={f:numbers(e[fit&(a['family']==f)]) for f in np.unique(a['family'][fit])},
              previous_matched=old_info['matched_errors'])
    write(folder/(name+'.json'),info)
    print(name,info['fit_after'],flush=True)


def encoded_index(candidate, reference):
    folder=OUT/candidate
    galleries={}
    for programme in PROGRAMMES:
        local=folder/'inspection/encoded'/(programme+'.json')
        index=local if local.is_file() else OUT/reference/'inspection/encoded'/(programme+'.json')
        data=read(index)
        coverage=read(folder/'review'/programme/'coverage.json')
        if not Path(coverage['video']).samefile(data['video']):
            raise ValueError(('Contact sheet must use the actual encoding',programme))
        for item in data['windows']:
            assert (index.parent/item['page']).is_file()
            assert item['review_pts']==item['pts']-round(coverage['source_seconds'][0]*12800)
        galleries[programme]=dict(index=file_ref(index),windows=len(data['windows']),video=file_ref(data['video']))
    write(folder/'inspection/encoded_windows.json',galleries)


def register(candidate):
    from deliver import camera_export
    from motion_delivery import ledger_check,media_check
    from side_release_1118 import picture_check

    label=OUTPUT.name
    folder=OUT/candidate
    b=camera_state(candidate);old=camera_state(PARENT)
    src=read(PREPARED/'setup.json')['shots']
    parent_file=EVENT/'scene'/(PRIOR.parent.name+'.json')
    parent=read(parent_file)
    geometry=parent['geometry_binding']
    assert numeric_sha(b['scene_values'])==geometry['value_sha256']
    np.testing.assert_array_equal(b['scene_values'],old['scene_values'])
    current=file_ref(folder/'state.npz')
    checked=OUTPUT/'four_check'/(folder.name+'.json')
    four=read(checked)
    assert four['totals']['shots']==len(src) and four['totals']['changed_camera_shots']==['jin_s14']
    assert set(four['totals']['changed_source_packets'])==set(SHOTS)
    assert four['current_checkpoint']['sha256']==current['sha256']
    original=read(PRIOR/'source_bindings.json')
    bindings={};programmes={};media={}
    for programme,pid in PROGRAMMES.items():
        names=[name for name,s in src.items() if s['program']==programme]
        prior_path=Path(parent['programmes'][programme]['path'])
        record=read(prior_path)
        for name in names:
            bindings[name]=dict(original[name])
            if name in SHOTS:
                reviewed_points=folder/'pixel_review' if name=='jin_s14' else OUTPUT/'point_review'
                audit=read(reviewed_points/(name+'.json'))
                packet=file_ref(prepared_packet_path(name))
                assert packet['sha256']==audit['packet_sha256']
                assert programme_state_sha(b,[name])==audit['state']
                bindings[name].update(packet=packet,matched_errors=file_ref(reviewed_points/(name+'.npz')),
                                      prior_source_rows_preserved=audit['old_prefix_rows'])
            else:
                assert bindings[name]['packet']['sha256']==sha(prepared_packet_path(name))
                assert programme_state_sha(b,[name])==programme_state_sha(old,[name])
        view=folder/'review'/programme
        info=read(view/'binding.json');coverage=read(view/'coverage.json')
        assert info['observation_packets']=={n:bindings[n]['packet']['sha256'] for n in names}
        assert current['sha256']==info['state_sha256']
        assert info['programme_state_sha256']==programme_state_sha(b,names)
        assert coverage['frames']==sum(len(b[n+'_pts']) for n in names)
        raw=coverage['video'];full_check=four['programmes'][programme]['current_review']
        if full_check.get('pending_review'):raise ValueError(('Media verification pending',programme))
        frames=np.concatenate([b[n+'_pts'] for n in names])
        media[programme]=dict(media=media_check(raw,coverage),
            ledger=ledger_check(Path(info.get('source_ledger_directory',view)),b,names),
            pictures=picture_check(raw,src[names[0]]['source'],int(frames[0]),frames[[0,len(frames)//2,-1]]),
            continuous_source_comparison=full_check)
        link=folder/'videos'/(programme+'_'+label+'_50fps.mp4')
        if link.is_file() and not link.samefile(raw):
            raise ValueError(('Playback copy and verified encoding differ',link))
        video=file_ref(link if link.is_file() else raw)
        camera=record['camera']
        if not all(programme_state_sha(b,[n])==programme_state_sha(old,[n]) for n in names):
            path=camera_export(programme,b,geometry['value_sha256'],geometry['layout_revision'],version=label,prepared=PREPARED)
            camera=file_ref(path)
        summary_file=folder/'per_programme'/(programme+'.json')
        write(summary_file,dict(shots={n:bindings[n]['matched_errors'] for n in names},full_check=file_ref(checked)))
        record.update(version_id=label,previous=file_ref(prior_path),camera=camera,
            fit_observations={n:bindings[n]['packet'] for n in names},pixel_review=file_ref(summary_file),
            full_review=dict(video=video,coverage=file_ref(view/'coverage.json'),render_state=current,
                             programme_state_sha256=info['programme_state_sha256'],frames=coverage['frames'],
                             source_seconds=coverage['source_seconds'],native_time_base=[1,12800],fps=50,
                             source_audio=coverage['source_audio'],media_verification=media[programme]))
        published=EVENT/'programmes'/pid/(label+'.json')
        write(published,record);programmes[programme]=file_ref(published)
    write(folder/'source_bindings.json',bindings)
    write(folder/'delivery_verification.json',media)
    evidence={k:file_ref(OUTPUT/p) for k,p in (
        ('source_rail','rail/rail_audit.json'),('near_print','near_prints/validation.json'),
        ('far_print','far_prints/metadata.json'),('cross_view','cross_prints/cross_camera_prints_zh.md'),
        ('cross_material','cross_prints/ad24_vertical.json'),('source_coverage','inventory.json'),
        ('centre_circle_source','motion_check/circle_c0_source/audit.json'),
        ('centre_circle_source_roles','motion_check/circle_c0_source/held_rows.csv'),
        ('centre_circle_midpoints','motion_check/circle_c0_source/c0_paired_source.npz'))}
    evidence['sportoto_letters']=file_ref(folder/'inspection/sportoto_word.json')
    evidence['centre_circle_reprojection']=file_ref(OUTPUT/'scores'/candidate.replace('/','_')/'early_ice.json')
    evidence['same_source_comparison']=file_ref(OUTPUT/'scores'/candidate.replace('/','_')/'comparison.json')
    write(EVENT/'scene'/(label+'.json'),dict(schema=parent['schema'],event_id='ankara_2026',version_id=label,
          status='source_review_candidate',user_visual_acceptance=False,previous=file_ref(parent_file),
          geometry_binding=geometry,coordinates=parent['coordinates'],programmes=programmes,selected_run=candidate,
          checkpoint=current,fit_setup=file_ref(PREPARED/'setup.json'),source_bindings=file_ref(folder/'source_bindings.json'),
          source_admission=file_ref(PREPARED/'source_audit.json'),lane4_upper=parent['lane4_upper'],
          previous_low_end_views=parent['all_low_end_views'],all_low_end_views=file_ref(folder/'motion_check/endview_recheck_numeric.json'),
          four_programme_checks=file_ref(checked),optics_and_trajectory=file_ref(checked),
          motion=file_ref(OUTPUT/'motion_check/exposure_final/comparison.json'),specific_source_evidence=evidence,
          per_shot_camera_provenance=file_ref(folder/'inputs.json'),submitted_issues=file_ref(folder/'inspection/shots.json'),
          encoded_source_windows=file_ref(folder/'inspection/encoded_windows.json'),
          material_scores=file_ref(OUTPUT/'scores'/candidate.replace('/','_')/'source.json'),
          media_verification=file_ref(folder/'delivery_verification.json')))
