"""Source-clock diagnostics using the same native geometry and camera projection."""
from argparse import ArgumentParser
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import sys
from pathlib import Path
import av
import cv2
import numpy as np
from shared import prepared_packet_path
from shared import (EVENT, FIELD, LIBRARY, OUT, PREPARED, NativeCore, NativeSceneGPU, ParameterState,
    SceneParameterModel, arrays, original_native, read, save, sha,write)
from rinkspace.review.scene import sample_native_review_scene

sys.path.insert(0,str(EVENT/'tools/joint_review_1113'))
import render_okada as draw
import joint_inputs as ji


def native_for(candidate, program):
    core=NativeCore(LIBRARY)
    if candidate=='original':return original_native(program,core=core)
    z=arrays(OUT/candidate/'state.npz') if candidate!='initial' else arrays(OUT/'initial_scene.npz')
    field=Path(str(z.get('scene_path',FIELD)))
    model=SceneParameterModel(read(field),field,optimize_scene=True)
    return NativeSceneGPU(field,core,parameter_state=ParameterState(model.initial_state.layout,z['scene_values']))


def camera_state(candidate):
    base=arrays(PREPARED/'baseline.npz')
    if candidate not in ('original','initial'):
        base.update(arrays(OUT/candidate/'state.npz'))
    return base


def source_frames(path,pts):
    pts=np.asarray(sorted(set(map(int,pts))),np.int64)
    with av.open(str(path)) as video:
        s=video.streams.video[0]
        assert s.time_base==Fraction(1,12800) and s.width==1920 and s.height==1080
        s.thread_count=3;s.thread_type='AUTO'
        video.seek(int(pts[0]),stream=s,backward=True)
        i=0
        for f in video.decode(s):
            if f.pts<pts[i]:continue
            if f.pts!=pts[i]:raise ValueError(('Missing source exposure',pts[i],f.pts))
            yield f.pts,f.to_ndarray(format='bgr24')
            i+=1
            if i==len(pts):return
        raise ValueError('Incomplete source frames')


def sample(native):
    return sample_native_review_scene(read(native.scene_path),native.core,native.scene_path,
        parameter_state=native.parameter_state,native=native)


def stills(candidate, names, seconds=None,evaluation=None):
    setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    folder=OUT/candidate
    if evaluation:folder=folder/evaluation
    folder=folder/'stills';folder.mkdir(parents=True,exist_ok=True)
    native_cache={};graphics={};manifest=[]
    for name in names:
        s=setup['shots'][name];program=s['program']
        if program not in native_cache:
            native_cache[program]=native_for(candidate,program)
            graphics[program]=draw.draw_config(sample(native_cache[program]))
            graphics[program][2][66:105]=22
            draw.text(graphics[program][2],('REGISTERED ORIGINAL' if candidate=='original' else 'ANKARA | CURRENT JOINT GEOMETRY'),(18,84),.43,(60,210,255))
        native=native_cache[program];world,paths,overview=graphics[program]
        pts=base[name+'_pts'];c=base[name+'_controls'];r=base[name+'_rigs'];t=base[name+'_times']
        requested=[int(pts[round((len(pts)-1)*v)]) for v in (0.,.25,.5,.75,1.)] if seconds is None else [int(round(v*50))*256 for v in seconds if pts[0]<=round(v*12800)<=pts[-1]]
        a=arrays(prepared_packet_path(name))
        for pt,raw in source_frames(s['source'],requested):
            idx=int(np.searchsorted(pts,pt));assert pts[idx]==pt
            projected=native.core.project_motion(r[idx],c,s['first'],s['dt'],1080,float(t[idx]),world)
            points=a['uv'][(a['pts']==pt)&np.isin(a['fold'],base.get(name+'_active_folds',[0]))]
            draw.TITLE=program.upper()+' | '+candidate+' | SOURCE REVIEW'
            pic,visible,status=draw.image(raw,projected['uv'],projected['status'],paths,overview,points,
                pt,s['shot'],idx,len(pts),None,subpixel_bits=4)
            stem=name+'_'+str(pt)
            cv2.imwrite(str(folder/(stem+'_source.jpg')),raw,[cv2.IMWRITE_JPEG_QUALITY,96])
            cv2.imwrite(str(folder/(stem+'_grid.jpg')),pic,[cv2.IMWRITE_JPEG_QUALITY,96])
            manifest.append(dict(shot=name,pts=pt,seconds=pt/12800.,source=s['source'],
                raw=str(folder/(stem+'_source.jpg')),overlay=str(folder/(stem+'_grid.jpg')),
                visible_samples=visible,status=status,fit_points=len(points)))
            print(name,pt/12800.,flush=True)
    write(folder/'manifest.json',manifest)


def localize(poly,station,pixels):
    a=poly[:-1];delta=np.diff(poly,axis=0);norm=np.sum(delta*delta,axis=1)
    valid=np.isfinite(a).all(1)&np.isfinite(delta).all(1)&(norm>1e-15)
    if not valid.any():return np.full(len(pixels),np.nan),np.full(len(pixels),np.inf)
    ai=np.flatnonzero(valid);a=a[ai];delta=delta[ai];norm=norm[ai]
    offset=pixels[:,None,:]-a[None,:,:]
    u=np.clip(np.sum(offset*delta,axis=2)/norm,0,1)
    d=np.linalg.norm(offset-u[:,:,None]*delta,axis=2)
    which=np.argmin(d,axis=1);row=np.arange(len(pixels));seg=ai[which]
    return station[seg]+u[row,which]*(station[seg+1]-station[seg]),d[row,which]


def metrics(candidate,names=None,evaluation=None):
    from solve import stats
    setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    out=OUT/candidate
    if evaluation:out=out/evaluation
    out.mkdir(parents=True,exist_ok=True)
    natives={};tables={};summary={};all_errors={}
    for name,s in setup['shots'].items():
        if names and name not in names:continue
        p=s['program']
        if p not in natives:
            natives[p]=native_for(candidate,p)
            tables[p]=ji.native_paths(natives[p],natives[p].paths,count=257)
        n=natives[p];table=tables[p];a=arrays(prepared_packet_path(name))
        # All existing sampled roles are reported separately, without promoting
        # old development/check material into independent final evidence.
        paired=np.rec.fromarrays((a['idx'],a['path']))
        groups,inv=np.unique(paired,return_inverse=True)
        order=np.argsort(inv,kind='stable')
        pieces=np.split(order,np.flatnonzero(np.diff(inv[order]))+1)
        e=np.full(len(a['pts']),np.inf);stations=np.full(len(e),np.nan)
        jobs=[(int(i),str(k),float(base[name+'_times'][i])) for i,k in groups]
        for lo in range(0,len(jobs),300):
            slab=jobs[lo:lo+300]
            xy=ji.project_groups(n,base[name+'_rigs'],base[name+'_controls'],s['first'],s['dt'],slab,{k:v[0] for k,v in table.items()})
            for j,((uv,status),(_,k,_)) in enumerate(zip(xy,slab),lo):
                uv=uv.copy();uv[status!=0]=np.nan
                ix=pieces[j];stations[ix],e[ix]=localize(uv,table[k][1],a['uv'][ix])
        fixed=np.flatnonzero(a.get('station_fixed',np.zeros(len(e),bool)))
        if len(fixed):
            rigs,cam=np.unique(base[name+'_rigs'][a['idx'][fixed]],axis=0,return_inverse=True)
            with n.resident(camera=dict(first_control_time=s['first'],control_dt=s['dt'],image_height=1080,
                    reference_pts=0,time_base_num=1,time_base_den=12800),
                    controls=base[name+'_controls'],rigs=rigs,camera_ids=cam.astype(np.int32),
                    path_ids=a['path'][fixed],stations=a['station'][fixed],pts=a['pts'][fixed],
                    observed=a['uv'][fixed],motion_times=base[name+'_times'][a['idx'][fixed]]) as problem:
                problem.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.65+1e-6))
                e[fixed]=problem.evaluate_errors()
            stations[fixed]=a['station'][fixed]
        all_errors[name+'_error']=e;all_errors[name+'_station']=stations
        groups={}
        for f in np.unique(a['family']):
            groups[f]={str(role):stats(e[(a['family']==f)&(a['fold']==role)]) for role in np.unique(a['fold'][a['family']==f])}
        summary[name]=dict(all=stats(e[a['fold']<5]),fit=stats(e[a['fold']==0]),check=stats(e[(a['fold']!=0)&(a['fold']<5)]),families=groups)
        print(name,summary[name]['fit'].get('p90'),summary[name]['check'].get('p90'),flush=True)
    save(out/'curve_errors.npz',**all_errors)
    write(out/'curve_metrics.json',summary)
    write(out/'metrics_binding.json',dict(candidate=candidate,
        packets={name:sha(prepared_packet_path(name)) for name in summary},
        state_sha256=sha(OUT/candidate/'state.npz') if candidate!='original' else None,
        prepared_directory=str(PREPARED),
        roles='inherited development roles; previous checks reused for model selection',
        path_measure='nearest point on finite native projected polyline; declared fixed material stations retain exact native projection; no source UV shifted'))


def reclassify_original(evaluation, source_evaluation):
    """Reuse original projections only after proving every measurement is identical."""
    from solve import stats
    source=OUT/'original'/source_evaluation
    binding=read(source/'metrics_binding.json')
    old_setup=read(Path(binding['prepared_directory'])/'setup.json')
    old_base=arrays(Path(binding['prepared_directory'])/'baseline.npz')
    current_base=arrays(PREPARED/'baseline.npz')
    for key in old_base:np.testing.assert_array_equal(old_base[key],current_base[key])
    errors=arrays(source/'curve_errors.npz');summary={};proof={}
    for name in binding['packets']:
        old_path=Path(binding['prepared_directory'])/(name+'.npz')
        if not old_path.exists():old_path=Path(old_setup['shots'][name]['observations'])
        assert sha(old_path)==binding['packets'][name]
        old=arrays(old_path);a=arrays(prepared_packet_path(name))
        for key in old:
            if key!='fold':np.testing.assert_array_equal(old[key],a[key],err_msg=name+'/'+key)
        e=errors[name+'_error'];assert len(e)==len(a['pts'])
        families={f:{str(role):stats(e[(a['family']==f)&(a['fold']==role)])
            for role in np.unique(a['fold'][a['family']==f])} for f in np.unique(a['family'])}
        summary[name]=dict(all=stats(e[a['fold']<5]),fit=stats(e[a['fold']==0]),
            check=stats(e[(a['fold']!=0)&(a['fold']<5)]),families=families)
        proof[name]=dict(previous=sha(old_path),current=sha(prepared_packet_path(name)),
            changed_folds=int(np.count_nonzero(old['fold']!=a['fold'])))
    out=OUT/'original'/evaluation
    write(out/'curve_metrics.json',summary)
    write(out/'metrics_binding.json',dict(candidate='original',
        packets={name:v['current'] for name,v in proof.items()},prepared_directory=str(PREPARED),
        errors_path=str(source/'curve_errors.npz'),errors_sha256=sha(source/'curve_errors.npz'),
        reuse_proof=proof,previous_binding=str(source/'metrics_binding.json'),
        identity='Every source field except admission fold and all baseline camera fields compared exactly. Only summaries reclassified.',
        roles=binding['roles'],path_measure=binding['path_measure']))


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('stills','metrics','reclassify-original'));ap.add_argument('--candidate',required=True)
    ap.add_argument('--shots',nargs='+');ap.add_argument('--seconds',nargs='+',type=float)
    ap.add_argument('--evaluation')
    ap.add_argument('--source-evaluation')
    args=ap.parse_args()
    if args.action=='stills':stills(args.candidate,args.shots,args.seconds,args.evaluation)
    elif args.action=='metrics':metrics(args.candidate,args.shots,args.evaluation)
    else:
        if args.candidate!='original' or not args.evaluation or not args.source_evaluation:ap.error('Original reclassification needs both evaluation directories')
        reclassify_original(args.evaluation,args.source_evaluation)
