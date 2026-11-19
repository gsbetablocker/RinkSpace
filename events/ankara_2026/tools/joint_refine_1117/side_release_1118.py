"""Audit the four complete source timelines and bind the released field and videos."""
from collections import defaultdict
from pathlib import Path
import shutil

import av
import cv2
import numpy as np

from deliver import camera_export,file_ref,geometry
from feedback_source import CASES as EARLIER
from lane_review import pixel_errors
from motion_delivery import ledger_check,media_check
from review import camera_state,ji,native_for,sample,source_frames
from shared import (EVENT,LIBRARY,OUT,PREPARED,PROGRAMMES,arrays,numeric_sha,prepared_packet_path,
                    programme_state_sha,read,registration,save,sha,write)
from side_repair import BASE,WORK,CAP,SIDE,TRANSVERSE,PREVIOUS,REPORTED,relative


VERSION='lane4_side_1118'


def numbers(x):
    good=x[np.isfinite(x)]
    return dict(count=len(x),invalid=len(x)-len(good),**({k:float(v) for k,v in zip(
        ('p50','p90','p99'),np.percentile(good,[50,90,99]))} if len(good) else {}))


def recheck(candidate,program,skip=(),cache=None):
    base=camera_state(BASE);new=camera_state(candidate);src=read(PREPARED/'setup.json')['shots']
    parent=read(PREVIOUS/'release/source_bindings.json');n0=native_for(BASE,program);n=native_for(candidate,program)
    changed=[]
    for k in n.paths:
        for field in ['parameters','vertices','knots']:
            if not np.allclose(n.paths[k].get(field,[]),n0.paths[k].get(field,[]),rtol=0,atol=2.e-12):
                changed.append(k);break
    assert set(changed)=={CAP,SIDE,TRANSVERSE}
    folder=OUT/candidate/'pixel_review';report={}
    for name,s in src.items():
        if s['program']!=program or name in skip:continue
        packet=prepared_packet_path(name);stamp=programme_state_sha(new,[name]);ph=sha(packet)
        for prior_folder in (folder,cache):
            if prior_folder is None:continue
            record=prior_folder/(name+'.json');file=prior_folder/(name+'.npz')
            if not record.exists() or not file.exists():continue
            cached=read(record)
            if stamp!=cached['state'] or ph!=cached['packet_sha256']:continue
            if folder!=prior_folder:
                folder.mkdir(exist_ok=True,parents=True);dest=folder/(name+'.npz')
                if not dest.exists() or sha(dest)!=sha(file):shutil.copyfile(file,dest)
                cached['candidate']=candidate;cached['same_state_source_review']=str(record)
                write(folder/(name+'.json'),cached)
            report[name]=cached
            break
        if name in report:continue
        packet=prepared_packet_path(name);q=arrays(packet);fit=q['fold']<5;check=(q['fold']>=5)&(q['fold']<9)
        used=fit|check
        prior=parent[name];old=arrays(prior['packet']['path'])
        count=len(old['idx'])
        for k in ('pts','idx','path','uv','station_fixed','origin_row'):
            np.testing.assert_array_equal(old[k],q[k][:count])
        source=arrays(prior['matched_errors']['path'])
        assert len(source['error'])==count
        extra=np.arange(len(q['pts']))>=count
        e0,st0=pixel_errors(n0,base,s,name,q,extra&used)
        e0[:count]=source['error'];st0[:count]=source['station']
        same=all(np.array_equal(base[name+k],new[name+k]) for k in ('_rigs','_controls','_pts','_times'))
        measure=(np.isin(q['path'],changed) if same else np.ones(len(fit),bool))&used
        e1,st1=pixel_errors(n,new,s,name,q,measure)
        e1[~measure]=e0[~measure];st1[~measure]=st0[~measure]
        frame_pts=new[name+'_pts'];np.testing.assert_array_equal(frame_pts[q['idx']],q['pts'])
        assert np.all(np.diff(frame_pts)==256)
        family,fi=np.unique(q['family'],return_inverse=True)
        group=q['idx']*len(family)+fi
        rr=np.flatnonzero(fit&np.isfinite(e0)&np.isfinite(e1))
        rr=rr[np.argsort(group[rr],kind='stable')]
        items=[]
        for rows in np.split(rr,np.flatnonzero(np.diff(group[rr]))+1):
            if len(rows)<4:continue
            j=rows[0];items.append((int(q['pts'][j]),str(q['family'][j]),len(rows),
                float(np.median(e0[rows])),float(np.median(e1[rows]))))
        risk=sorted(items,key=lambda v:v[-1],reverse=True)
        regression=sorted(items,key=lambda v:v[-1]-v[-2],reverse=True)
        fstats={f:dict(before=numbers(e0[fit&(fi==j)]),after=numbers(e1[fit&(fi==j)])) for j,f in enumerate(family)}
        save(folder/(name+'.npz'),error=e1,station=st1,previous=e0,pts=frame_pts,
            coverage=np.bincount(q['idx'][fit],minlength=len(frame_pts)))
        rec=dict(source=s['source'],source_sha256=s['source_sha256'],previous_matched=prior['matched_errors'],
            candidate=candidate,state=stamp,packet_sha256=ph,frames=len(frame_pts),
            old_prefix_rows=count,exact_camera_match=same,finite_curves_recomputed=int(measure.sum()),
            unchanged_geometry_and_camera_replayed=int((used&~measure).sum()),fit_before=numbers(e0[fit]),fit_after=numbers(e1[fit]),
            held_before=numbers(e0[check]),held_after=numbers(e1[check]),families=fstats,
            largest_local=risk[:8],largest_increase=regression[:8])
        report[name]=rec;write(folder/(name+'.json'),rec);print('rechecked',name,len(frame_pts),round(rec['fit_after']['p90'],2),flush=True)
    write(folder/(program+'_summary.json'),report)


def crops(candidate,cases):
    """Original, previous and current model at specified exact source exposures."""
    states=[camera_state(c) for c in (BASE,candidate)]
    natives=[native_for(c,'jin') for c in (BASE,candidate)]
    src=read(PREPARED/'setup.json')['shots'];directory=OUT/candidate/'inspection';directory.mkdir(exist_ok=True)
    out=[]
    tables={}
    for name,sec,group,rect,label in cases:
        pts=round(sec*12800);s=src[name];i=np.searchsorted(states[0][name+'_pts'],pts)
        assert states[0][name+'_pts'][i]==pts
        _,raw=next(source_frames(s['source'],[pts]));x,y,X,Y=rect
        phases=[raw[y:Y,x:X].copy()]
        for j,(b,n) in enumerate(zip(states,natives)):
            key=(j,tuple(group))
            if key not in tables:
                keys=[k for k,v in n.paths.items() if k in group or v['label'] in group or v['label'].split(':')[0] in group]
                assert len(keys),(group,name)
                tables[key]=ji.native_paths(n,keys,count=257)
            t=tables[key];panel=raw.copy();jobs=[(i,k,b[name+'_times'][i]) for k in t]
            pred=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,{k:g[0] for k,g in t.items()})
            for k,(u,status) in zip(t,pred):
                good=np.flatnonzero((status==0)&np.isfinite(u).all(1)&(abs(u)<5000).all(1))
                color=(245,153,35) if k.startswith('board/') else (57,207,67) if n.paths[k]['label'].startswith('Ad') else (0,192,249)
                for run in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
                    if len(run)>1:cv2.polylines(panel,[np.int32(np.rint(u[run]*16))],False,color,1,cv2.LINE_AA,4)
            phases.append(panel[y:Y,x:X].copy())
        pics=[]
        for title,pic in zip(['Source','Previous fit','Revised fit'],phases):
            pic=cv2.copyMakeBorder(pic,20,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
            cv2.putText(pic,f'{title}   {name}   {sec:.3f} s',(7,15),0,.46,(235,235,235),1,cv2.LINE_AA);pics.append(pic)
        path=directory/f'{name}_{pts}_{label}.jpg';cv2.imwrite(str(path),np.vstack(pics),[cv2.IMWRITE_JPEG_QUALITY,96]);out.append(str(path))
    write(directory/'cases.json',out)


def case_definitions():
    allcases=[]
    for name,secs in REPORTED.items():
        for sec in secs:
            if name=='jin_s08':
                areas=[(['Ad1','Ad2','Ad3','board.top'],(0,240,1920,550),'plates')]
            elif name=='jin_s14':
                areas=[(['Ad1','Ad2','Ad3','Ad4','Ad5','Ad21','Ad22','Ad23','Ad24','Ad25','Ad26','board.top','board.bottom'],(0,335,1920,464),'boards')]
            elif name=='jin_s15':
                areas=[(['board.top','Ad14'],(0,290,950,478),'rail'),(['ice.B1'],(0,992,1650,1080),'blue')]
            elif name=='jin_s22':
                areas=[(['Ad1','Ad2','Ad3','Ad4','Ad25','Ad26','board.top'],(0,247,1920,355),'boards')]
            elif name=='nayeon_s03':
                areas=[(['Ad5','Ad6','Ad7','board.top'],(340,407,1600,520),'far'),
                    (['Ad8','Ad9','Ad10','Ad11','Ad12','board.top','board.bottom'],(0,392,395,850),'corner')]
            else:areas=[(['Ad5','Ad6'],(120,287,1830,440),'panels')]
            allcases.extend((name,sec,groups,box,label) for groups,box,label in areas)
    from feedback_delivery import CROPS
    for name,cases in EARLIER.items():
        for sec,fams in cases:
            keys=[key for fam in fams for key in (['board.top','board.bottom'] if fam=='board' else [fam if fam.startswith('Ad') else 'ice.'+fam])]
            for j,box in enumerate(CROPS[name,sec]):allcases.append((name,sec,keys,box,'prior'+str(j)))
    return allcases


def low_end_views(candidate):
    """Score all low-end shots on the same surviving source paint and its horizontal ends."""
    from feedback_delivery import extent_scores

    scores=PREVIOUS/'scores'
    extent_scores(candidate,'material_tips')
    old_tips=scores/BASE.replace('/','_')/'material_tips'
    new_tips=scores/candidate.replace('/','_')/'material_tips'
    setup=read(PREPARED/'setup.json')['shots']
    folder=OUT/candidate
    shots={}
    totals=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    for file in sorted((PREVIOUS/'material_sweep').glob('*.npz')):
        name=file.stem
        s=setup[name];src=arrays(file);q=arrays(prepared_packet_path(name))
        assert src['source_sha256'].item()==s['source_sha256']
        error=arrays(folder/'pixel_review'/(name+'.npz'))
        np.testing.assert_array_equal(src['pts'],error['pts'])
        admitted=q['fold']<9
        table={};shots[name]=dict(frames=len(src['pts']),families=table)
        for j,family in enumerate(src['families']):
            item=dict(visible_guide=int(np.count_nonzero(src['ledger'][:,j,0])))
            table[str(family)]=item
            for code,binding in s['source_packets'].items():
                path=Path(binding['path'])
                if path.parent.name not in ('material_sweep','material_tips') or path.stem!=name:continue
                take=admitted&(q['source_packet']==int(code))&(q['family']==family)
                if not np.any(take):continue
                rows=q['source_row'][take]
                if path.parent.name=='material_sweep':
                    np.testing.assert_array_equal(q['pts'][take],src['pts'][src['index'][rows]])
                    np.testing.assert_array_equal(q['uv'][take],src['uv'][rows])
                    before,after=error['previous'][take],error['error'][take]
                    measure='contour_normal'
                else:
                    original=arrays(path);np.testing.assert_array_equal(q['uv'][take],original['uv'][rows])
                    before=abs(arrays(old_tips/path.name)['error'][rows])
                    after=abs(arrays(new_tips/path.name)['error'][rows])
                    measure='horizontal_extremum'
                item[measure]=dict(measured_exposures=len(np.unique(q['pts'][take])),
                                   before=numbers(before),after=numbers(after))
                for pool,vals in zip(totals[str(family)][measure],[before,after]):pool.append(vals)
    aggregated={f:{m:[numbers(np.concatenate(v)) for v in pair] for m,pair in g.items()} for f,g in totals.items()}
    write(folder/'end_views.json',dict(reference=BASE,shots=shots,exposures=sum(s['frames'] for s in shots.values()),
                                       surfaces=aggregated,source=str(PREVIOUS/'material_sweep'),
                                       extremum_source=str(PREVIOUS/'material_tips')))


def picture_check(output,source,first,check_pts):
    """Read actual encoded pictures, including the last, against the same source exposures."""
    results=[]
    with av.open(str(output)) as movie:
        stream=movie.streams.video[0]
        assert stream.width==2560 and stream.height==1144 and float(stream.average_rate)==50.
        for pt in check_pts:
            target=int(pt-first)
            movie.seek(target,stream=stream,backward=True)
            frame=next(frame for frame in movie.decode(stream) if frame.pts>=target)
            if frame.pts!=target:raise ValueError(('Wrong encoded exposure',source,pt))
            _,raw=next(source_frames(source,[pt]))
            preview=frame.to_ndarray(format='bgr24')[64:1144,:1920]
            # Most pixels of this upper broadcast band have no rink overlay.
            ref=cv2.cvtColor(raw[:135,625:1575],cv2.COLOR_BGR2GRAY)
            seen=cv2.cvtColor(preview[:135,625:1575],cv2.COLOR_BGR2GRAY)
            delta=cv2.absdiff(seen,ref)
            results.append(dict(source_pts=int(pt),encoded_pts=target,
                                upper_band_luma_median=float(np.median(delta)),
                                upper_band_luma_p90=float(np.percentile(delta,90))))
    return results


def decoded_timeline_check(candidate, programme):
    """Compare a broadcast picture band at every encoded exposure to its native source."""
    folder=OUT/candidate
    coverage=read(folder/'review'/programme/'coverage.json')
    shots=read(PREPARED/'setup.json')['shots']
    names=[name for name,s in shots.items() if s['program']==programme]
    state=camera_state(candidate)
    pts=np.concatenate([state[name+'_pts'] for name in names])
    source=shots[names[0]]['source']
    def upper_band(frame,header):
        if frame.format.name not in ('yuv420p','yuvj420p','nv12'):
            raise ValueError(('Unexpected decoded luma format',frame.format.name))
        plane=frame.planes[0]
        luma=np.ndarray((frame.height,plane.line_size),dtype=np.uint8,buffer=plane)
        return luma[header+8:header+135:4,625:1575:4].astype(np.int16)
    medians=np.empty(len(pts));tails=np.empty(len(pts))
    with av.open(source) as original,av.open(coverage['video']) as output:
        src=original.streams.video[0];review=output.streams.video[0]
        for stream in (src,review):
            assert (stream.time_base.numerator,stream.time_base.denominator)==(1,12800)
            stream.thread_count=3;stream.thread_type='AUTO'
        original.seek(int(pts[0]),stream=src,backward=True)
        native=(frame for frame in original.decode(src) if frame.pts>=int(pts[0]))
        encoded=output.decode(review)
        for i,pt in enumerate(pts):
            a=next(native);b=next(encoded)
            if a.pts!=pt or b.pts!=pt-pts[0]:
                raise ValueError(('Decoded picture clock',programme,int(pt),a.pts,b.pts))
            diff=np.abs(upper_band(a,0)-upper_band(b,64))
            medians[i],tails[i]=np.percentile(diff,[50,90])
        if next(encoded,None) is not None:
            raise ValueError(('Extra encoded picture',programme))
    worst=np.argsort(tails,kind='stable')[-8:][::-1]
    evidence=folder/'decoded'/programme
    save(evidence.with_suffix('.npz'),source_pts=pts,median_luma_difference=medians,p90_luma_difference=tails)
    write(evidence.with_suffix('.json'),dict(candidate=candidate,state_sha256=sha(folder/'state.npz'),
        source=file_ref(source),video=file_ref(coverage['video']),frames=len(pts),verified_native_clock=True,
        region=dict(source_xyxy=[625,8,1575,135],stride=4,review_y_offset=64,units='8-bit decoded luma'),
        per_exposure_median=numbers(medians),per_exposure_p90=numbers(tails),
        worst_p90=[dict(pts=int(pts[i]),median=float(medians[i]),p90=float(tails[i])) for i in worst]))


def register(candidate, additional_evidence=None):
    folder=OUT/candidate;state=camera_state(candidate);n=native_for(candidate,'jin');coords=state['scene_values']
    digest=numeric_sha(coords);revision=n.parameter_state.layout['layout_revision']
    checkpoint=file_ref(folder/'state.npz')
    motion=WORK/'motion_evidence/comparison_audit'/candidate/'audit.json'
    appearance=WORK/'appearance_audit'/folder.name/'programmes.json'
    photo_index=appearance.with_name('index.json')
    motion_record=read(motion)
    photo_review=read(appearance)
    assert motion_record['candidates'][candidate]['sha256']==checkpoint['sha256']
    assert photo_review['candidate']==candidate and photo_review['state_sha256']==checkpoint['sha256']
    assert read(photo_index)['candidate']==candidate
    proof={key:file_ref(path) for key,path in (additional_evidence or {}).items()}
    version=VERSION;table=EVENT/'scene'/(version+'_coordinates.json')
    snapshot=EVENT/'scene/field'/(version+'.npz')
    save(snapshot,scene_values=coords,scene_values_sha256=digest,layout_revision=revision);write(table,geometry(n))
    field=dict(version_id=version,snapshot=file_ref(snapshot),structure=file_ref(n.scene_path),value_sha256=digest,
        layout_revision=revision,native_library=file_ref(LIBRARY),coordinate_scale='rink length = 1',
        parameter_count=coords.size,path_count=len(n.paths),advertisement_count=len(read(table)['advertisements']))
    src=read(PREPARED/'setup.json')['shots'];bindings={};records={};media={}
    expected=sample(n)
    for prog,directory in PROGRAMMES.items():
        names=[key for key,s in src.items() if s['program']==prog]
        current=sample(native_for(candidate,prog));np.testing.assert_array_equal(current['xyz'],expected['xyz'])
        assert [q['id'] for q in current['cuts']]==[q['id'] for q in expected['cuts']]
        for key in names:
            test=read(folder/'pixel_review'/(key+'.json'));assert test['state']==programme_state_sha(state,[key])
            bindings[key]=dict(packet=file_ref(prepared_packet_path(key)),matched_errors=file_ref(folder/'pixel_review'/(key+'.npz')),
                source=src[key]['source'],source_sha256=src[key]['source_sha256'],prior_source_rows_preserved=test['old_prefix_rows'])
        path=folder/'review'/prog;cover=read(path/'coverage.json');info=read(path/'binding.json')
        assert info['state_sha256']==sha(folder/'state.npz') and info['mode']=='complete'
        assert info['programme_state_sha256']==programme_state_sha(state,names)
        assert info['source_sha256']==src[names[0]]['source_sha256']
        assert info['scene_sha256']==sha(n.scene_path)
        assert info['observation_packets']=={key:bindings[key]['packet']['sha256'] for key in names}
        assert cover['frames']==cover['validation']['decoded_frames']==sum(len(state[k+'_pts']) for k in names)
        media[prog]=media_check(cover['video'],cover)
        media[prog]['ledger']=ledger_check(Path(info.get('source_ledger_directory',path)),state,names)
        continuous=folder/'decoded'/(prog+'.json')
        if continuous.is_file():
            timeline=read(continuous)
            assert timeline['state_sha256']==checkpoint['sha256'] and timeline['frames']==cover['frames']
            assert timeline['source']['sha256']==info['source_sha256'] and timeline['verified_native_clock']
            assert Path(timeline['video']['path']).samefile(cover['video'])
            media[prog]['all_decoded_source_pictures']=file_ref(continuous)
        pts=np.concatenate([state[k+'_pts'] for k in names])
        media[prog]['decoded_pictures']=picture_check(cover['video'],src[names[0]]['source'],int(pts[0]),pts[[0,len(pts)//2,-1]])
        # Collect the four verified files together while retaining their encoding ledgers.
        encoded=Path(cover['video'])
        playback=folder/'videos'/encoded.name
        playback.parent.mkdir(exist_ok=True)
        if not playback.exists():
            playback.hardlink_to(encoded)
        if not playback.samefile(encoded):
            raise ValueError(('Playback file differs from the verified encode',playback,encoded))
        camera=camera_export(prog,state,digest,revision,version=version,prepared=PREPARED)
        rec=dict(schema='rinkspace.programme_joint_refinement.v1',event_id='ankara_2026',programme_id=directory,
            version_id=version,status='source_review_candidate',user_visual_acceptance=False,
            previous=file_ref(EVENT/'programmes'/directory/'review_fixes_1118.json'),source=registration(prog)['source'],
            geometry_binding=field,camera=file_ref(camera),shots=names,fit_roles=[0,1,2,3,4],
            fit_observations={key:bindings[key]['packet'] for key in names},pixel_review=file_ref(folder/'pixel_review'/(prog+'_summary.json')),
            full_review=dict(video=file_ref(playback),coverage=file_ref(path/'coverage.json'),render_state=file_ref(folder/'state.npz'),
                programme_state_sha256=info['programme_state_sha256'],source_seconds=cover['source_seconds'],frames=cover['frames'],
                native_time_base=[1,12800],fps=50,source_audio=cover['source_audio'],media_verification=media[prog]))
        reg=EVENT/'programmes'/directory/(version+'.json');write(reg,rec);records[prog]=file_ref(reg)
    write(folder/'delivery_verification.json',media);write(folder/'source_bindings.json',bindings)
    write(EVENT/'scene'/(version+'.json'),dict(schema='rinkspace.event_joint_refinement.v1',event_id='ankara_2026',
        version_id=version,status='source_review_candidate',user_visual_acceptance=False,previous=file_ref(EVENT/'scene/review_fixes_1118.json'),
        geometry_binding=field,coordinates=file_ref(table),programmes=records,selected_run=candidate,
        checkpoint=checkpoint,fit_setup=file_ref(PREPARED/'setup.json'),source_bindings=file_ref(folder/'source_bindings.json'),
        source_admission=file_ref(PREPARED/'source_audit.json'),lane4_upper=file_ref(WORK/'common_field/inputs.json'),
        original_visibility=file_ref(WORK/'lane_source/summary.json'),cap_limit=file_ref(WORK/'lane_cap_limit/limit.json'),
        transverse_limit=file_ref(WORK/'lane_cross_limit/limit.json'),
        motion=file_ref(motion),optics=file_ref(folder/'sensor_domain.json'),
        original_photo_review=file_ref(appearance),original_photo_index=file_ref(photo_index),
        specific_source_evidence=proof,
        all_low_end_views=file_ref(folder/'end_views.json'),per_shot_camera_provenance=file_ref(folder/'inputs.json'),
        submitted_issues=file_ref(folder/'inspection/cases.json'),media_verification=file_ref(folder/'delivery_verification.json')))
