"""Additional red-circle readings from both actual pigment edges at every exposure."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import sys
import cv2
import numpy as np
from shared import EVENT, OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write
from review import camera_state, native_for, source_frames, ji, localize
from source_paint import matte
sys.path.append(str(EVENT/'programmes/chen_xi_jin_sp/tools'))
from physical_reads import samples, source_profiles, stripe_candidates, coherent


def run(args):
    folder=OUT/args.out;folder.mkdir(parents=True,exist_ok=False)
    setup=read(PREPARED/'setup.json');base=camera_state(args.candidate);natives={};report={}
    cv2.setNumThreads(1)
    for name in args.shots:
        s=setup['shots'][name];p=s['program']
        if p not in natives:natives[p]=native_for(args.candidate,p)
        native=natives[p];packet=arrays(prepared_packet_path(name))
        keys=np.unique(packet['path'][packet['family']==args.family])
        table=ji.native_paths(native,keys,count=513)
        pts=base[name+'_pts'];masks=matte(s,pts)
        if masks is None:raise ValueError('Native source foreground matte required')
        rows=defaultdict(list);ledger=[];checked={round(v*12800) for v in args.check_seconds}
        for pt,raw in source_frames(s['source'],pts):
            i=int(np.searchsorted(pts,pt));frame_hash=sha256(raw.tobytes()).hexdigest()
            if not masks['valid'][i]:
                ledger.append(dict(pts=pt,points=0,reason='source_foreground_unknown'));continue
            mask=cv2.dilate(np.unpackbits(masks['packed'][i],axis=1,count=1920),np.ones((7,7),np.uint8))
            mask[:158,:590]=1;mask[:292,:451]=1;mask[928:,1618:]=1
            b,g,r=np.moveaxis(raw.astype(np.int16),-1,0)
            mask[(np.minimum(g,r)-b>15)&(abs(r-g)<40)]=1
            foot=packet['uv'][(packet['pts']==pt)&(packet['path']=='board/continuous/bottom')&(packet['fold']<5)]
            if len(foot)>=4:
                foot=foot[np.argsort(foot[:,0])]
                for x0,y0,x1,y1 in np.c_[foot[:-1],foot[1:]]:
                    if not 2<x1-x0<=100 or abs(y1-y0)>.4*(x1-x0)+3:continue
                    for x in range(max(0,int(np.ceil(x0))),min(1920,int(np.floor(x1))+1)):
                        y=int(np.floor(y0+(x-x0)/(x1-x0)*(y1-y0)+3))
                        mask[:max(0,min(1080,y)),x]=1
            measured_frame=[]
            for key,(world,stations) in table.items():
                projected=native.core.project_motion(base[name+'_rigs'][i],base[name+'_controls'],
                    s['first'],s['dt'],1080,base[name+'_times'][i],world)
                poly=projected['uv'].copy();poly[projected['status']!=0]=np.nan
                sampled=samples(poly,stations,axis=None,step=20.)
                if sampled is None:continue
                xy,normal,tangent,st=sampled
                if len(xy)<4:continue
                profile,hidden,_=source_profiles(raw,mask,xy,normal,tangent,71,'red')
                choices=[stripe_candidates(v,h,71,expected_min=3,max_width=18,min_contrast=2.5)
                         for v,h in zip(profile,hidden,strict=True)]
                # Separate quadrants prevent opposite normal signs on the two
                # visible sides of a circle from being forced onto one line.
                for quadrant in range(4):
                    loc=np.flatnonzero(np.minimum((st*4).astype(int),3)==quadrant)
                    if len(loc)<4:continue
                    accepted=coherent([choices[j] for j in loc],xy[loc],stripe=True)
                    for local,value in accepted:
                        j=loc[local];uv=xy[j]+value[0]*normal[j]
                        station,error=localize(poly,stations,uv[None])
                        if not np.isfinite(station[0]):continue
                        item=dict(pts=int(pt),idx=i,uv=uv,path=str(key),station=float(station[0]),
                            family=args.family,source_bracket=xy[j]+np.asarray(value[2:4])[:,None]*normal[j],
                            contrast=float(value[4]),normal=normal[j],query_uv=xy[j],source_BGR_sha256=frame_hash)
                        for k,v in item.items():rows[k].append(v)
                        measured_frame.append(uv)
            ledger.append(dict(pts=int(pt),points=len(measured_frame),source_BGR_sha256=frame_hash))
            if pt in checked:
                pic=raw.copy()
                for uv in measured_frame:cv2.drawMarker(pic,tuple(np.rint(uv).astype(int)),(0,0,255),cv2.MARKER_CROSS,9,1)
                cv2.imwrite(str(folder/(name+'_'+str(pt)+'_source.jpg')),raw)
                cv2.imwrite(str(folder/(name+'_'+str(pt)+'_measured.jpg')),pic)
        values={k:np.asarray(v) for k,v in rows.items()}
        save(folder/(name+'.npz'),**values,source_sha256=s['source_sha256'])
        write(folder/(name+'_frames.json'),ledger)
        report[name]=dict(source=s['source'],source_sha256=s['source_sha256'],frames=len(pts),
            points=len(rows.get('pts',[])),supported_frames=len(np.unique(rows.get('pts',[]))))
        write(folder/'report.json',dict(candidate=args.candidate,state_sha256=sha(OUT/args.candidate/'state.npz'),
            family=args.family,shots=report,evidence='raw source red pigment gradient flanks, neutral surroundings and within-arc coherence; current geometry navigates only; no source UV from projection',
            status='new source proposals pending material review; existing observations unchanged'))
        print(name,report[name],flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True)
    p.add_argument('--shots',nargs='+',required=True);p.add_argument('--family',required=True)
    p.add_argument('--out',required=True);p.add_argument('--check-seconds',nargs='*',type=float,default=[])
    run(p.parse_args())
