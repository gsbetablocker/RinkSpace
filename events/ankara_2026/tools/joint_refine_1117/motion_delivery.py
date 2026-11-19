"""Verify identical source clocks and shared geometry in all four full-length deliverables."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json
import os
import shutil
import subprocess
import av
import numpy as np
from deliver import camera_export,file_ref,geometry
from lane_end_evidence import BASE, ENDS, WORK
from repair_motion import stat
from review import camera_state,native_for,sample
from shared import (EVENT,LIBRARY,OUT,PREPARED,PROGRAMMES,arrays,numeric_sha,prepared_packet_path,
                    programme_state_sha,read,registration,safe_path,save,sha,write)
from solve import stats,layer
from source_bundle import relative


VERSION='lane_motion_1118'
FINAL=Path(os.environ.get('RINKSPACE_LANE_FINAL',str(WORK/'checked'))).resolve()


def review_directory(programme):
    local=FINAL/'review'/programme
    if (local/'binding.json').is_file():return local
    previous=read(FINAL/'retained_reviews.json')[programme]
    folder=Path(previous['directory'])
    assert sha(folder/'binding.json')==previous['binding_sha256']
    return folder


def media_check(video,coverage):
    from scipy.signal import correlate,correlation_lags
    with av.open(str(video)) as f:
        vs=f.streams.video[0];audio=bool(f.streams.audio)
        dimensions=[vs.width,vs.height];fps=float(vs.average_rate)
    if dimensions!=[2560,1144] or fps!=50 or audio!=bool(coverage['source_audio']):
        raise ValueError(('wrong review media',video,dimensions,fps,audio))
    windows=[]
    if audio:
        exe=shutil.which('ffmpeg');duration=coverage['frames']/50
        def samples(path,at):
            data=subprocess.check_output([exe,'-nostdin','-v','error','-ss',f'{at:.6f}','-i',str(path),
                '-t','2','-map','0:a:0','-ac','1','-ar','8000','-f','f32le','pipe:1'])
            return np.frombuffer(data,dtype='<f4').astype(float)
        for t in (.5,duration*.5,duration-2.6):
            x=samples(coverage['source_audio'],coverage['source_seconds'][0]+t);y=samples(video,t)
            length=min(len(x),len(y));x=x[:length]-np.mean(x);y=y[:length]-np.mean(y)
            energy=np.linalg.norm(x)*np.linalg.norm(y)
            if length<8000 or energy<1e-5:continue
            ac=correlate(y,x,method='fft')/energy;lag=correlation_lags(length,length);use=np.flatnonzero(abs(lag)<600)
            best=use[ac[use].argmax()];windows.append(dict(video_seconds=t,lag_ms=float(lag[best]/8),
                aligned_correlation=float(ac[best])))
        if not windows or any(abs(w['lag_ms'])>20 or w['aligned_correlation']<.80 for w in windows):
            raise ValueError(('source audio alignment',video,windows))
    return dict(dimensions=dimensions,fps=fps,source_audio_present=audio,source_audio_windows=windows)


def ledger_check(folder,state,names):
    count=0;no_points=0;masks=0
    with (folder/'source_frames.jsonl').open(encoding='utf-8') as f:
        for name in names:
            pts=state[name+'_pts']
            with np.load(prepared_packet_path(name)) as a:
                selected=np.isin(a['fold'],state[name+'_active_folds'])
                counts=np.bincount(a['idx'][selected],minlength=len(pts))
            for pt,number in zip(pts,counts,strict=True):
                rec=json.loads(next(f));assert rec['pts']==pt and rec['shot']==name
                if rec['fit_points']!=number or not rec['visible_samples']:raise ValueError((name,int(pt),'frame ledger'))
                count+=1;no_points+=int(number==0);masks+=int(rec['foreground_matte'])
        if next(f,None) is not None:raise ValueError('Unexpected excess exposure in render ledger')
    return dict(exposures=count,without_source_points=no_points,source_foreground_available=masks)


def camera_audit(native,scene,name,s):
    rig=scene[name+'_rigs'];t=scene[name+'_times'];pts=scene[name+'_pts'];ctrl=scene[name+'_controls']
    unique,inv=np.unique(rig,axis=0,return_inverse=True)
    values={key:np.empty(len(t)) for key in ('pan','tilt','log_f','fx','fy','cx','cy','k1','k2')}
    for rid,r in enumerate(unique):
        ix=inv==rid;z=native.core.evaluate_motion_camera(r,ctrl,s['first'],s['dt'],t[ix])
        if np.any(z['status']!=0):raise ValueError((name,'invalid exposure camera'))
        for key in values:values[key][ix]=z[key]
    if not all(np.isfinite(a).all() for a in values.values()):raise ValueError((name,'nonfinite camera'))
    p=rig[:,15:18];board=native.scene['board']
    u=np.maximum(0.,(np.abs(p[:,0])-(board[0]-board[2]))/board[2])
    v=np.maximum(0.,(np.abs(p[:,1])-(board[1]-board[3]))/board[3])
    margin=u**board[4]+v**board[4]-1
    valid=(margin>=-1e-10)&(p[:,2]>0)&(np.linalg.norm(rig[:,18:21],axis=1)>1e-12)
    valid&=(p[:,0]<0) if s['camera_index']==2 else (p[:,1]<=-board[1]+1e-10)
    if not valid.all():raise ValueError((name,'mount outside physical domain'))
    movement={}
    for key in ('pan','tilt','log_f'):
        g=np.unwrap(values[key]) if key=='pan' else values[key]
        for order in [1,2]:
            dx=np.diff(g,n=order);i=np.argmax(abs(dx))+order
            movement[key+'_difference_'+str(order)]=dict(absolute=float(abs(dx[i-order])),source_pts=int(pts[i]))
    return dict(frames=len(t),valid=int(valid.sum()),rig_count=len(unique),
        pivot_range_L=np.ptp(p,axis=0).tolist(),pivot=p[len(p)//2].tolist(),
        lever_length_L=float(np.linalg.norm(rig[len(rig)//2,18:21])),spacing_L=float(rig[len(rig)//2,21]),
        min_board_margin=float(np.min(margin)),readout_ms=np.quantile(rig[:,37]*1000,[0,1]).tolist(),
        exposure_parameter_ranges={k:np.quantile(x,[0,1]).tolist() for k,x in values.items()},trajectory=movement)


def paired(a,b,kind):
    f=stat if kind=='motion' else stats
    return dict(before=f(a),after=f(b))


def comparison():
    b=camera_state(relative(FINAL));a0=camera_state(BASE);native=native_for(relative(FINAL),'jin')
    setup=read(PREPARED/'setup.json')['shots'];sources=read(FINAL/'inputs.json')['shots']
    original=OUT/'repair/prepared_v7';raw_setup=read(original/'setup.json')['shots']
    program_reports={};shot_reports={};bindings={};cameras={};end_inventory={}
    for prog in PROGRAMMES:
        pools=defaultdict(lambda:[[],[]]);count=defaultdict(int)
        for name,s in setup.items():
            if s['program']!=prog:continue
            candidate=sources[name]['parent'];stage=candidate.split('/')[1]
            qdir=WORK/'evaluation'/({'release_cameras':'release'}.get(stage,stage))
            packet=prepared_packet_path(name);identity=sha(packet)
            baseline_dirs=[WORK/'evaluation'/k for k in ('blue_baseline','cap_baseline','base_delivery','base_confirmed')]
            qb=next(p for p in baseline_dirs if (p/(name+'.json')).is_file()
                    and read(p/(name+'.json'))['packet_sha']==identity)
            stamp=read(qdir/(name+'.json'));baseline=read(qb/(name+'.json'))
            assert stamp['packet_sha']==baseline['packet_sha']==identity
            assert stamp['shot_state']==programme_state_sha(b,[name])
            assert baseline['shot_state']==programme_state_sha(a0,[name])
            m0=arrays(qb/(name+'.npz'));m1=arrays(qdir/(name+'.npz'))
            with np.load(packet) as f:
                a={key:f[key] for key in ('pts','path','uv','fold','station_fixed','material','family')}
            np.testing.assert_array_equal(m0['pts'],b[name+'_pts']);np.testing.assert_array_equal(m0['pts'],m1['pts'])
            np.testing.assert_array_equal(a0[name+'_times'],b[name+'_times'])
            assert len(m0['error'])==len(m1['error'])==len(a['uv'])
            old=original/(name+'.npz')
            if not old.exists():old=Path(raw_setup[name]['observations'])
            with np.load(old) as raw:
                prefix=len(raw['pts']);frozen=np.zeros(len(a['pts']),bool);frozen[:prefix]=raw['fold']<5
                np.testing.assert_array_equal(a['uv'][:prefix],raw['uv'])
                np.testing.assert_array_equal(a['pts'][:prefix],raw['pts'])
                np.testing.assert_array_equal(a['path'][:prefix],raw['path'])
            use=a['fold']<5;lanes=a['material']=='paired_lane_end_source';ring=a['material']=='real_paired_annulus_edges'
            masks=dict(original_frozen=frozen,original_retained=use&frozen,all=use,
                fixed_anchors=use&a['station_fixed'],new_lane=use&lanes,
                lane_witness=(a['fold']==6)&lanes,new_annulus=use&ring,
                new_panel=use&(np.char.startswith(a['material'],'native_purple_panel')|np.char.startswith(a['material'],'native_blue_panel')),
                all_direct_lane=lanes,initial_lane_witness=lanes&np.isin(name,['jin_s09','okada_s04','chen_s17','nayeon_s02']))
            for key in ('ice','board','ad'):masks[key]=use&(layer(a['family'])==key)
            report={}
            for k,mask in masks.items():
                x=m0['error'][mask];y=m1['error'][mask];report[k]=paired(x,y,'pixels')
                pools[k][0].append(x);pools[k][1].append(y)
            for k in ('lag1_held','lag8_held','lag1_rebound_held','lag8_rebound_held'):
                ok=np.isfinite(m0[k])&np.isfinite(m1[k]);x=m0[k][ok];y=m1[k][ok]
                report[k]=paired(x,y,'motion');pools[k][0].append(x);pools[k][1].append(y)
                if len(y):report[k]['worst_after_source_pts']=int(m1['pts'][np.flatnonzero(ok)[y.argmax()]])
            shot_reports[name]=report
            bindings[name]=dict(packet=file_ref(packet),source_sha256=s['source_sha256'],
                candidate=candidate,shot_state=stamp['shot_state'],matched_errors=file_ref(qdir/(name+'.npz')),
                baseline_errors=file_ref(qb/(name+'.npz')))
            cameras[name]=dict(before=camera_audit(native,a0,name,s),after=camera_audit(native,b,name,s))
            for i,key in enumerate(ENDS,1):
                for label,mask in [('fit',use&lanes),('check',(a['fold']==6)&lanes)]:
                    xx=mask&(a['path']==key)
                    if not xx.any():continue
                    end_inventory.setdefault(f'{prog}/LANE{i}/{label}',[]).append(dict(shot=name,rows=int(xx.sum()),
                        distinct_exposures=len(np.unique(a['pts'][xx])),before=stats(m0['error'][xx]),after=stats(m1['error'][xx])))
            count['frames']+=len(m1['pts']);count['shots']+=1;count['supplied_rows']+=len(a['pts'])
            count['fitted_rows']+=int(use.sum());count['originally_fitted_rows']+=int(frozen.sum())
        program_reports[prog]=dict(counts=dict(count),**{k:paired(np.concatenate(x),np.concatenate(y),
            'motion' if k.startswith('lag') else 'pixels') for k,(x,y) in pools.items()})
    write(FINAL/'comparison.json',dict(previous=BASE,current=relative(FINAL),prepared=str(PREPARED),
        state_sha256=sha(FINAL/'state.npz'),programmes=program_reports,shots=shot_reports,end_inventory=end_inventory,
        metric='Native 1920x1080 pixels. Finite curve distances and exact fixed material point distances. Motion uses identical original-source exposure pairs.'))
    write(FINAL/'source_bindings.json',bindings);write(FINAL/'physical_mounts.json',cameras)


def geometry_at_fixed_cameras():
    from lane_review import pixel_errors
    state=camera_state(relative(FINAL));native=native_for(BASE,'jin')
    records=read(FINAL/'source_bindings.json');setup=read(PREPARED/'setup.json')['shots']
    groups=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    for name,s in setup.items():
        with np.load(prepared_packet_path(name)) as f:
            a={key:f[key] for key in ('family','material','fold','idx','pts','station','station_fixed','uv','path')}
        ice=(layer(a['family'])=='ice')&(a['fold']<5)
        cap=a['material']=='paired_lane_end_source';ring=a['material']=='real_paired_annulus_edges'
        old,_=pixel_errors(native,state,s,name,a,ice|cap|ring)
        with np.load(records[name]['matched_errors']['path']) as f: current=f['error']
        masks=dict(ice=ice,direct_caps=cap,ring=ring,witness=cap&(a['fold']==6))
        masks.update({f'LANE{i}':cap&(a['path']==k) for i,k in enumerate(ENDS,1)})
        for label,mask in masks.items():
            if not mask.any():continue
            for group in ('all',s['program']):
                groups[group][label][0].append(old[mask]);groups[group][label][1].append(current[mask])
    entries={group:{label:paired(np.concatenate(x),np.concatenate(y),'pixels') for label,(x,y) in bins.items()}
        for group,bins in groups.items()}
    write(FINAL/'geometry_only.json',dict(camera_state_sha256=sha(FINAL/'state.npz'),
        previous_geometry=BASE,current_geometry=relative(FINAL),unit='original pixels, identical final cameras and original image points',
        groups=entries))


def surface_and_trajectory_review(previous_run):
    from repair_motion import frame_median
    state=camera_state(relative(FINAL));prior=camera_state(relative(previous_run))
    setup=read(PREPARED/'setup.json')['shots'];physical=read(FINAL/'physical_mounts.json')
    chosen={};packets={};groups=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    for name,s in setup.items():
        folder=FINAL/'repair_motion'
        if not (folder/(name+'.npz')).exists():
            assert programme_state_sha(state,[name])==programme_state_sha(prior,[name])
            folder=previous_run/'repair_motion'
        chosen[name]=folder/(name+'.npz')
        packets[name]=read(folder/'summary.json')[name]['source_path']
        z=[arrays(OUT/BASE/'repair_motion'/(name+'.npz')),arrays(chosen[name])]
        _,left,right=np.intersect1d(z[0]['source_row'],z[1]['source_row'],assume_unique=True,return_indices=True)
        for field in ('index','lag','family'):
            np.testing.assert_array_equal(z[0][field][left],z[1][field][right])
        np.testing.assert_array_equal(z[0]['pts'],z[1]['pts'])
        index=z[0]['index'][left];pts=z[0]['pts']
        for family,code,minimum in [('ice',1,3),('board',0,4)]:
            for term,lag,vector in [('adjacent',1,'displacement'),('lag8',8,'displacement'),('jitter',1,'curvature')]:
                take=(z[0]['family'][left]==code)&(z[0]['lag'][left]==lag)
                values=[frame_median(index,np.linalg.norm(record[vector][idx],axis=1),take,len(pts),minimum)
                    for record,idx in zip(z,[left,right])]
                valid=np.isfinite(values[0])&np.isfinite(values[1])
                for pool,v in zip(groups[s['program']][family+'_'+term],values):pool.append(v[valid])
    out={p:{k:paired(np.concatenate(x),np.concatenate(y),'motion') for k,(x,y) in bins.items()} for p,bins in groups.items()}
    write(FINAL/'motion_all_surfaces.json',dict(state_sha256=sha(FINAL/'state.npz'),programmes=out,
        shot_files={n:str(path) for n,path in chosen.items()},method='matched native source rows, conditional left depth in each solution'))
    peaks={};bindings=read(FINAL/'source_bindings.json')
    for prog in PROGRAMMES:
        names=[n for n,s in setup.items() if s['program']==prog];record={}
        for axis in ('pan','tilt','log_f'):
            for order in (1,2):
                key=f'{axis}_difference_{order}'
                size,name,pt=max((physical[n]['after']['trajectory'][key]['absolute'],n,
                    physical[n]['after']['trajectory'][key]['source_pts']) for n in names)
                with np.load(chosen[name]) as measured:
                    i=np.flatnonzero(measured['pts']==pt).item()
                    match=(measured['index']==i)&(measured['lag']==1)
                    ids=measured['source_row'][match];err=measured['displacement'][match]
                with np.load(packets[name]) as source:
                    step=source['uv'][ids]-source['left_uv'][ids]
                with np.load(bindings[name]['matched_errors']['path']) as q:held=q['lag1_held'][i]
                record[key]=dict(shot=name,pts=pt,size=size,verified_current_source_pairs=len(ids),
                    original_median_displacement_px=float(np.median(np.linalg.norm(step,axis=1))) if len(ids) else None,
                    source_flow_error_px=float(np.median(np.linalg.norm(err,axis=1))) if len(ids) else None,
                    held_board_flow_error_px=float(held) if np.isfinite(held) else None)
        peaks[prog]=record
    write(FINAL/'trajectory_extrema.json',dict(state_sha256=sha(FINAL/'state.npz'),programmes=peaks,
        policy='development source tracks and reserved board tracks reported separately; absent evidence explicit'))


def export():
    candidate=relative(FINAL);state=camera_state(candidate);native=native_for(candidate,'jin')
    values=native.parameter_state.values;digest=numeric_sha(values);revision=native.parameter_state.layout['layout_revision']
    setup=read(PREPARED/'setup.json')['shots'];source=read(FINAL/'source_bindings.json');report=read(FINAL/'comparison.json')
    visual=read(FINAL/'visual_review.json');state_sha=sha(FINAL/'state.npz')
    assert visual['state_sha256']==report['state_sha256']==state_sha and set(source)==set(setup)
    snapshot=safe_path(EVENT/'scene/field'/f'{VERSION}.npz')
    save(snapshot,scene_values=values,scene_values_sha256=digest,layout_revision=revision)
    coordinates=EVENT/'scene'/f'{VERSION}_coordinates.json';write(coordinates,geometry(native))
    binding=dict(version_id=VERSION,status='source_review_candidate',snapshot=file_ref(snapshot),
        structure=file_ref(native.scene_path),native_library=file_ref(LIBRARY),value_sha256=digest,
        layout_revision=revision,scene_parameters_active=False,coordinate_scale='rink length = 1')
    reference=sample(native);programmes={}
    for k,directory in PROGRAMMES.items():
        names=[n for n,s in setup.items() if s['program']==k];folder=review_directory(k)
        r=read(folder/'binding.json');c=read(folder/'coverage.json');other=sample(native_for(candidate,k))
        np.testing.assert_array_equal(other['xyz'],reference['xyz'])
        assert [z['id'] for z in other['cuts']]==[z['id'] for z in reference['cuts']]
        render_state=Path(r['state']);assert sha(render_state)==r['state_sha256']
        render_group=programme_state_sha(camera_state(relative(render_state.parent)),names)
        assert r['programme_state_sha256']==render_group==programme_state_sha(state,names)
        assert r['source_sha256']==setup[names[0]]['source_sha256']
        assert r['scene_sha256']==sha(native.scene_path) and r['mode']=='complete'
        assert c['validation']['decoded_frames']==c['frames']==sum(len(state[n+'_pts']) for n in names)
        for n in names:assert r['observation_packets'][n]==source[n]['packet']['sha256']==sha(prepared_packet_path(n))
        cam=camera_export(k,state,digest,revision,version=VERSION,prepared=PREPARED)
        old=EVENT/'programmes'/directory/'joint_dense_1118.json';assert old.exists()
        media=media_check(c['video'],c);media['ledger']=ledger_check(folder,state,names)
        record=dict(schema='rinkspace.programme_joint_refinement.v1',event_id='ankara_2026',programme_id=directory,
            version_id=VERSION,status='source_review_candidate',user_visual_acceptance=False,previous=file_ref(old),
            source=registration(k)['source'],geometry_binding=binding,camera=file_ref(cam),shots=names,
            fit_roles=[0,1,2,3,4],fit_observations={n:source[n]['packet'] for n in names},
            full_review=dict(video=file_ref(c['video']),coverage=file_ref(folder/'coverage.json'),
                render_state=file_ref(render_state),programme_state_sha256=render_group,source_seconds=c['source_seconds'],frames=c['frames'],
                native_time_base=[1,12800],fps=50,source_audio=c['source_audio'],media_verification=media),
            comparison=report['programmes'][k],physical_mounts=file_ref(FINAL/'physical_mounts.json'),
            visual_review=file_ref(FINAL/'visual_review.json'))
        dst=EVENT/'programmes'/directory/(VERSION+'.json');write(dst,record);programmes[k]=file_ref(dst)
    write(EVENT/'scene'/(VERSION+'.json'),dict(schema='rinkspace.event_joint_refinement.v1',event_id='ankara_2026',
        version_id=VERSION,status='source_review_candidate',user_visual_acceptance=False,geometry_binding=binding,
        previous=file_ref(EVENT/'scene/joint_dense_1118.json'),coordinates=file_ref(coordinates),programmes=programmes,
        selected_run=candidate,checkpoint=file_ref(FINAL/'state.npz'),fit_setup=file_ref(PREPARED/'setup.json'),
        source_admission=file_ref(PREPARED/'source_audit.json'),source_bindings=file_ref(FINAL/'source_bindings.json'),
        comparison=file_ref(FINAL/'comparison.json'),geometry_only=file_ref(FINAL/'geometry_only.json'),physical_mounts=file_ref(FINAL/'physical_mounts.json'),
        surface_motion=file_ref(FINAL/'motion_all_surfaces.json'),trajectory_extrema=file_ref(FINAL/'trajectory_extrema.json'),
        visual_review=file_ref(FINAL/'visual_review.json'),limitations=visual['limitations']))


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('task',choices=['comparison','export']);a=ap.parse_args()
    if a.task=='comparison':comparison()
    else:export()
