"""Read visible purple-panel boundaries on every original exposure in a span."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from hashlib import sha256
import sys
import cv2
import numpy as np
from shared import EVENT,OUT,PREPARED,arrays,read,save,sha,write
from review import camera_state,native_for,source_frames,ji,localize
from source_paint import matte
sys.path.append(str(EVENT/'programmes/chen_xi_jin_sp/tools'))
from physical_reads import samples,source_profiles,boundary_candidates,coherent


def run(args):
    if args.step<=0 or args.radius<16:raise ValueError('Invalid source sampling support')
    folder=OUT/args.out;folder.mkdir(parents=True,exist_ok=False)
    setup=read(PREPARED/'setup.json');s=setup['shots'][args.shot]
    base=camera_state(args.candidate);native=native_for(args.candidate,s['program'])
    pts=base[args.shot+'_pts'];clock=base[args.shot+'_times']
    controls=base[args.shot+'_controls'];rigs=base[args.shot+'_rigs']
    selected=pts[(pts>=round(args.seconds[0]*12800))&(pts<=round(args.seconds[1]*12800))]
    keys=[key for key in native.paths if key.startswith('print_instance/'+args.ad+'::') and
          key.split('::')[-1].split('.')[-1] in ('left_vertical','right_vertical','top_edge','bottom_edge')]
    if len(keys)!=4:raise ValueError('Expected four typed panel boundaries')
    table=ji.native_paths(native,keys,count=129)
    offsets=np.r_[0,np.cumsum([len(table[k][0]) for k in keys])]
    world=np.vstack([table[k][0] for k in keys]);masks=matte(s,pts)
    if masks is None:raise ValueError('Source foreground masks required for this panel reread')
    rows=defaultdict(list);ledger=[];counts=Counter()
    check_pts={round(float(t)*12800) for t in args.check_seconds}
    for pt,raw in source_frames(s['source'],selected):
        index=int(np.searchsorted(pts,pt));assert pts[index]==pt
        frame_sha=sha256(raw.tobytes()).hexdigest()
        if not masks['valid'][index]:
            ledger.append(dict(pts=int(pt),source_BGR_sha256=frame_sha,points=0,reason='foreground_mask_unavailable'))
            continue
        mask=cv2.dilate(np.unpackbits(masks['packed'][index],axis=1,count=1920),np.ones((7,7),np.uint8))
        mask[:158,:590]=1;mask[:292,:451]=1;mask[928:,1618:]=1
        pred=native.core.project_motion(rigs[index],controls,s['first'],s['dt'],1080,float(clock[index]),world)
        xy=pred['uv'].copy();xy[pred['status']!=0]=np.nan
        frame_rows=[];per_path={}
        for k,start,stop in zip(keys,offsets[:-1],offsets[1:],strict=True):
            role=k.split('.')[-1];axis=int(role.endswith('_edge'))
            sign=1 if role in ('left_vertical','top_edge') else -1
            poly=xy[start:stop];stations=table[k][1]
            sampled=samples(poly,stations,axis=axis,step=args.step)
            if sampled is None:continue
            coords,normal,tangent,station=sampled
            use=(station>.10)&(station<.90)
            coords,normal,tangent=coords[use],normal[use],tangent[use]
            if len(coords)<4:continue
            radius=args.radius
            profiles,hidden,colours=source_profiles(raw,mask,coords,normal,tangent,radius,'purple')
            options=[boundary_candidates(v,h,c,radius,sign=sign,tone='purple')
                     for v,h,c in zip(profiles,hidden,colours,strict=True)]
            accepted=coherent(options,coords,stripe=False,minimum_span=args.minimum_span)
            if not accepted:continue
            ix=np.asarray([i for i,_ in accepted],int);values=np.asarray([v for _,v in accepted])
            measured=coords[ix]+values[:,0,None]*normal[ix]
            seed,error=localize(poly,stations,measured)
            good=np.isfinite(seed)&np.isfinite(measured).all(1)&((measured>=[4,4])&(measured<=[1915,1075])).all(1)
            measured=measured[good];seed=seed[good];ix=ix[good];values=values[good]
            per_path[k]=dict(points=len(measured),signed_source_shift_median_px=float(np.median(values[:,0])) if len(values) else None)
            for j,uv in enumerate(measured):
                item=dict(pts=int(pt),idx=index,uv=uv,path=k,station=float(seed[j]),family=args.ad,
                    normal=normal[ix[j]],contrast=float(values[j,4]),query_uv=coords[ix[j]],
                    source_BGR_sha256=frame_sha)
                for key,value in item.items():rows[key].append(value)
                frame_rows.append(uv);counts[k]+=1
        ledger.append(dict(pts=int(pt),source_BGR_sha256=frame_sha,points=len(frame_rows),paths=per_path))
        if pt in check_pts:
            marked=raw.copy()
            for uv in frame_rows:cv2.circle(marked,tuple(np.rint(uv).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
            cv2.imwrite(str(folder/(str(pt)+'_source.jpg')),raw,[cv2.IMWRITE_JPEG_QUALITY,97])
            cv2.imwrite(str(folder/(str(pt)+'_measured.jpg')),marked,[cv2.IMWRITE_JPEG_QUALITY,97])
    data={key:np.asarray(value) for key,value in rows.items()}
    if not len(data.get('pts',[])):raise ValueError('No supported source boundary measurements')
    save(folder/'observations.npz',**data,source_sha256=s['source_sha256'])
    write(folder/'source_frames.json',ledger)
    write(folder/'binding.json',dict(shot=args.shot,ad=args.ad,source=s['source'],source_sha256=s['source_sha256'],
        candidate=args.candidate,camera_state_sha256=sha(OUT/args.candidate/'state.npz'),
        packet=str(folder/'observations.npz'),packet_sha256=sha(folder/'observations.npz'),
        requested_frames=len(selected),covered_frames=len(np.unique(data['pts'])),points=len(data['pts']),paths=dict(counts),
        source_seconds=args.seconds,sampling_step_px=args.step,search_radius_px=args.radius,minimum_span_px=args.minimum_span,
        operator='existing physical_reads purple signed material step with coherent source support',
        observation='actual raw-image pigment transition; foreground/graphics excluded; one exposure at a time',
        geometry_role='typed panel identity and search corridor only; station initialization is separate from unchanged measured UV',
        evidence_role='additional development source readings; source-correlated, no independent validation claim'))
    print(len(data['pts']),'points on',len(np.unique(data['pts'])),'of',len(selected),'source frames',flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True)
    p.add_argument('--shot',required=True);p.add_argument('--ad',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seconds',nargs=2,type=float,required=True);p.add_argument('--check-seconds',nargs='*',type=float,default=[])
    p.add_argument('--step',type=float,default=20.);p.add_argument('--radius',type=int,default=91)
    p.add_argument('--minimum-span',type=float,help='Explicit source edge support for a smaller panel; defaults retain the existing reader')
    run(p.parse_args())
