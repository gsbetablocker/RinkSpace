"""Chronological source-space comparison on one native time grid."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

from compare import bind
from source_review import digest,SourceVideo, whole_skate_observations
from source_camera_rebind import write
from joint_source_objective import native_measurements,source_shoe_rows


def metrics(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0)
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))


def measurements(args,name):
    if name=='early':
        cand=args.work/'source_resume/solve/candidate.npz';source=args.work/'input/source.json'
    elif name=='before':
        cand=args.parent/'candidate.npz';source=args.parent/'source.json'
    else:
        cand=args.output/'candidate.npz';source=args.output/'source.json'
    cache=(args.shared if name!='joint' else args.output)/(name+'_source_measurements.npz')
    identity=digest(cand)
    if cache.exists():
        with np.load(cache) as z:
            if str(z['candidate'])==identity and 'reviewed_arcs' in z:
                return {k:z[k] for k in ('uv','shoe','angles','height','reviewed_arcs')}
    ref,cam,_,_,packet=bind(args.prepared,source,cand)
    from types import SimpleNamespace
    from calibration.program_tracking.body_tracking.visible_skate_union import VisibleSkateUnion
    from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
    state=SequenceState(ref,packet['seconds'])
    problem=SimpleNamespace(reference=ref,binding=ref.binding,camera=cam,body=SimpleNamespace(state=state),size=17+state.size)
    arc_path=args.work/'source_review/current_visible_arcs.json'
    arc_spec=dict(path=str(arc_path),sha256=digest(arc_path),exact_template=True,reviewed_open_arcs=True)
    all_arcs={}
    for sid in dict.fromkeys(cam.shot_ids):
        ix=np.flatnonzero(cam.shot_ids==sid)
        arcs=VisibleSkateUnion(problem,arc_spec,ix)
        if arcs.batches:
            res=arcs.evaluate(np.zeros(state.size),False)
            all_arcs[str(sid)]=res['metrics'][arcs.name]
    arc_record=np.array(json.dumps(all_arcs))
    if cache.exists():
        with np.load(cache) as z:
            result={k:z[k] for k in ('uv','shoe','angles','height')} if str(z['candidate'])==identity else None
        if result is not None:
            result['reviewed_arcs']=arc_record
            pending=cache.with_suffix('.pending.npz');np.savez_compressed(pending,candidate=identity,**result);pending.replace(cache)
            return result
    with np.load(args.shared/'parent_material_uv.npz') as z:materials=z['material']
    uv,shoe,attach=native_measurements(ref,cam,materials)
    from calibration.program_tracking.body_tracking.skate_physics import fk_state,batch_features
    heights=[];rots=[]
    for a in range(0,len(ref.times),256):
        t=ref.times[a:a+256];s=ref.at(t)
        pose=fk_state(ref,s,np.zeros((len(t),69)))
        feat=batch_features(ref.binding,pose['R'],pose['O'],pose)
        blade=np.isin(feat['semantics'],['runner','toe_pick'])
        heights.append(feat['height'][...,blade].min(-1));rots.append(s['local_R'])
    r=np.concatenate(rots);q=r[:-1].swapaxes(-1,-2)@r[1:]
    ang=np.rad2deg(np.arccos(np.clip((np.trace(q,axis1=-2,axis2=-1)-1)*.5,-1,1)))
    result=dict(uv=uv,shoe=shoe,angles=np.r_[np.full((1,21),np.nan),ang],height=np.concatenate(heights),reviewed_arcs=arc_record)
    pending=cache.with_suffix('.pending.npz');np.savez_compressed(pending,candidate=identity,**result);pending.replace(cache)
    return result


def errors(name,m,obs,rows,ext,good,dirs):
    use=obs['use'];ids=obs['model_joint_ids']
    uv=m['uv'];groups={}
    err=np.linalg.norm(uv[np.arange(len(uv))[:,None],ids]-obs['uv'],axis=-1)
    for k,js in dict(body=range(17),arms=[5,6,7,8,9,10],knees=[13,14],ankles=[15,16]).items():
        groups[k]=np.where(use & np.isin(ids,js),err,np.nan)
    ex=(m['shoe'][:,:,7:]@dirs.T).max(2)
    groups['complete_shoe_outline']=np.where(good,np.abs(ex-ext).mean(-1),np.nan)
    mats=np.full((len(uv),2),np.nan);acc=np.zeros_like(mats);count=np.zeros_like(mats)
    t,f,k,lo,hi,_=map(np.asarray,zip(*rows))
    v=m['shoe'][t,f,k];dx=np.linalg.norm(v-np.clip(v,lo,hi),axis=-1)
    np.add.at(acc,(t,f),dx);np.add.at(count,(t,f),1);np.divide(acc,count,out=mats,where=count>0)
    groups['named_shoe_material']=mats
    return groups


def crop(im,points):
    inside=points[np.isfinite(points).all(-1)&(points[:,0]>=0)&(points[:,0]<1920)&(points[:,1]>=0)&(points[:,1]<1080)]
    if not len(inside):return (0,0,1920,1080)
    lo=inside.min(0)-22;hi=inside.max(0)+22
    size=min(1920,max(160,hi[0]-lo[0],(hi[1]-lo[1])*1.12))
    h=min(1080,size/1.12)
    x=np.clip((lo[0]+hi[0]-size)*.5,0,1920-size);y=np.clip((lo[1]+hi[1]-h)*.5,0,1080-h)
    return tuple(map(int,[x,y,x+size,y+h]))


def montage(args,sid,indices,reader,states,pts,folder):
    paths=[]
    for offset in range(0,len(indices),6):
        pick=indices[offset:offset+6];W,H=224,201
        canvas=Image.new('RGB',(W*4,(H+24)*len(pick)),'white');cd=ImageDraw.Draw(canvas)
        for row,i in enumerate(pick):
            raw=Image.fromarray(reader.read(int(pts[i])))
            points=np.concatenate([states[n]['uv'][i] for n in states])
            box=crop(raw,points)
            for col,name in enumerate(['source','early','before','joint']):
                im=raw.copy()
                if col:
                    state=states[name];draw=ImageDraw.Draw(im);uv=state['uv'][i]
                    for a,b in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]:
                        color='#13e8da' if b in [5,7,9,11,13,15] else '#ff9630'
                        draw.line([tuple(u) for u in uv[[a,b]]],fill=color,width=3)
                    for ft,colr in [(0,'#19daba'),(1,'#ff8520')]:
                        hull=cv2.convexHull(state['shoe'][i,ft,7:].astype('f4')).reshape(-1,2)
                        xy=[tuple(v) for v in hull]+[tuple(hull[0])];draw.line(xy,fill=colr,width=3)
                im=im.crop(box);im.thumbnail((W,H))
                x=col*W;y=row*(H+24)
                canvas.paste(im,(x+(W-im.width)//2,y+24));cd.text((x+2,y+3),f'{pts[i]/12800:.2f}s   {name}',fill='black')
        path=folder/f'{sid}_{offset//6+1}.jpg';canvas.save(path,quality=93);paths.append(str(path))
    return paths


def run(args):
    if args.step in ('early','before','joint'):
        measurements(args,args.step);return
    states={n:measurements(args,n) for n in ('early','before','joint')}
    ledger=args.shared/'final_body_observations.npz'
    if not ledger.exists():ledger=args.shared/'body_observations.npz'
    obs=np.load(ledger);pts=obs['pts']
    with np.load(args.work/'input/shoe_occlusion_source.npz') as z:
        np.testing.assert_array_equal(pts,z['source_pts']);shot=z['shot_ids']
        good=(z['role']==1)&z['visible']&~z['partial_boundary']&(z['source_pair_overlap'][:,None]<.12)
        dirs=np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1)
        ext=(np.nan_to_num(z['region'])@dirs.T).max(2)
    doc=json.loads(whole_skate_observations(args.programme).read_text(encoding='utf8'))
    for r in doc.get('source_outline_review_exclusions',[]):
        if 'pts' in r and 'foot' in r:good[np.searchsorted(pts,r['pts']),r['foot']]=False
    rows=source_shoe_rows(doc,pts,states['before']['shoe'])
    e={n:errors(n,state,obs,rows,ext,good,dirs) for n,state in states.items()}
    folder=args.output/'source_review';folder.mkdir(exist_ok=True)
    src=json.loads((args.programme/'s14_boards_1118.json').read_text(encoding='utf8'))['source']
    rd=SourceVideo(src['path'],(1,12800))
    report=dict(native_frames=len(pts),source=src,candidate=digest(args.output/'candidate.npz'),per_shot={},chronological_windows=[],full={},examples={})
    report['body_observation_ledger_sha256']=digest(ledger)
    with np.load(args.work/'input/body_track_source.npz') as original:
        keep=original['use'];ids=original['model_joint_ids'];target=original['uv']
        report['original_body_population']={n:metrics(np.linalg.norm(
            v['uv'][np.arange(len(pts))[:,None],ids]-target,axis=-1)[keep]) for n,v in states.items()}
    for key in e['joint']:
        report['full'][key]={n:metrics(v[key]) for n,v in e.items()}
    for sid in dict.fromkeys(shot):
        ix=np.flatnonzero(shot==sid);seg=dict(frames=len(ix),pts=pts[ix[[0,-1]]].tolist(),metrics={})
        seg['reviewed_boot_and_blade_arcs']={n:json.loads(str(v['reviewed_arcs'])).get(str(sid)) for n,v in states.items()}
        for key in e['joint']:seg['metrics'][key]={n:metrics(v[key][ix]) for n,v in e.items()}
        pick=set([ix[0],ix[-1]])
        for a in range(0,len(ix),150):
            win=ix[a:a+150];summ={}
            for key in ['body','knees','arms','named_shoe_material','complete_shoe_outline']:
                summ[key]={n:metrics(v[key][win]) for n,v in e.items()}
            report['chronological_windows'].append(dict(shot=str(sid),source_pts=pts[win[[0,-1]]].tolist(),frames=len(win),metrics=summ))
            pick.add(win[len(win)//2])
        for key in ['knees','arms','named_shoe_material','complete_shoe_outline']:
            val=e['joint'][key][ix]
            if np.any(np.isfinite(val)):
                pick.add(ix[np.nanargmax(np.where(np.isfinite(val),val,-np.inf).max(-1))])
        for t in [22.38,24.32,28.2,52.26,53.50,63.5,76.8,85.26,102.,111.44,136.6,153.48,163.58,182.82,185.,187.]:
            i=np.argmin(abs(pts/12800-t))
            if i in ix:pick.add(i)
        indices=sorted(pick);seg['source_seconds_reviewed']=(pts[indices]/12800).tolist()
        seg['sheets']=montage(args,sid,indices,rd,states,pts,folder);report['per_shot'][str(sid)]=seg
    rd.close()
    for sec,ji in [(24.32,14),(63.50,7)]:
        i=np.searchsorted(pts,round(sec*12800));j=np.flatnonzero(obs['model_joint_ids'][i]==ji)[0]
        report['examples'][str(sec)]={n:float(np.linalg.norm(z['uv'][i,ji]-obs['uv'][i,j])) for n,z in states.items()}
    i=np.searchsorted(pts,684800);report['examples']['53.5_left_cuff']={n:float(np.linalg.norm(z['shoe'][i,0,6]-[1280,831])) for n,z in states.items()}
    report['rotation_steps_deg']={}
    for sec,j in [(24.32,19),(53.5,1),(153.48,6),(182.82,2)]:
        i=np.searchsorted(pts,round(sec*12800));report['rotation_steps_deg'][str(sec)]={n:float(s['angles'][i,j-1]) for n,s in states.items()}
    report['semantics']='Body distances use the same admitted source pixels and mapped COCO joint. Named shoe material distances use the same source regions and track foot assignment. Outlines compare 12 support directions of isolated complete source outlines. These are development checks on the single broadcast.'
    write(args.output/'joint_quality.json',report)
    print(json.dumps(dict(metrics=report['full'],examples=report['examples']),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('step',choices=['early','before','joint','review'])
    ap.add_argument('--shared',type=lambda x:Path(x).resolve())
    for k in ('prepared','work','programme','parent','output'):ap.add_argument('--'+k,type=lambda x:Path(x).resolve(),required=True)
    args=ap.parse_args();args.shared=args.shared or args.output
    run(args)
