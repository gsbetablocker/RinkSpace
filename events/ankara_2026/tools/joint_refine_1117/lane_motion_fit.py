"""Prepare complete source packets with the additional measured Lane endings."""
from collections import defaultdict
import shutil

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from lane_end_evidence import BASE, ENDS, ST, WORK, foreground
from review import camera_state, native_for, source_frames
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write
from source_paint import matte


HELD_SHOTS={'jin_s09','okada_s04','chen_s17','nayeon_s02'}


def admitted_measurements(z,name):
    accepted=np.zeros(len(z.get('lane',[])),bool)
    checked={}
    for lane in np.unique(z.get('lane',[])):
        rows=np.flatnonzero(z['lane']==lane)
        pts,first,count=np.unique(z['pts'][rows],return_index=True,return_counts=True)
        local=[];offset=[];width=[]
        for start,n in zip(first,count):
            ix=rows[start:start+n]
            st=z['station'][ix];uv=z['uv'][ix];q=z['guide'][ix];edges=z['bracket'][ix]
            fit=np.c_[np.ones(n),st-.5]
            slope=np.linalg.lstsq(fit,uv,rcond=None)[0][1]
            ref=np.linalg.lstsq(fit,q,rcond=None)[0][1]
            angle=np.degrees(np.arccos(np.clip(np.dot(slope,ref)/(np.linalg.norm(slope)*np.linalg.norm(ref)),-1,1)))
            unit=z['unit'][ix]
            w=np.linalg.norm(edges[:,1]-edges[:,0],axis=1)/unit
            dx=np.sum((uv-q)*z['norm'][ix],axis=1)/unit
            resid=abs(dx-fit@np.linalg.lstsq(fit,dx,rcond=None)[0])
            safe=(resid<.00058)&(abs(w-np.median(w))<.00065)
            # Both source flanks must follow the same visible stripe. A dark
            # crossing, coloured ring or skate shadow often tilts one flank.
            slopes=[np.linalg.lstsq(fit,e,rcond=None)[0][1] for e in edges.transpose(1,0,2)]
            flank_angle=np.degrees(np.arccos(np.clip(np.dot(*slopes)/np.prod([np.linalg.norm(v) for v in slopes]),-1,1)))
            local.append(ix[safe] if (angle<12 and flank_angle<14 and safe.sum()>=8 and
                np.ptp(st[safe])>.28 and np.median(z['contrast'][ix][safe])>=3.4) else np.empty(0,int))
            offset.append(np.median(dx));width.append(np.median(w))
        for j,ix in enumerate(local):
            if not len(ix):continue
            near=(abs(pts-pts[j])<=4*256)&(pts!=pts[j])
            support=[k for k in np.flatnonzero(near) if len(local[k]) and
                abs(offset[k]-offset[j])<.00065 and abs(width[k]-width[j])<.00072]
            if len(support)>=2:accepted[ix]=True
        checked[str(int(lane+1))]=dict(candidate_rows=len(rows),accepted_rows=int(accepted[rows].sum()))
    return accepted,checked


def check_old(a,errors,station,name,s,base,native):
    """Read ambiguous old pixels on the native frame, irrespective of fitted cost sign."""
    rows=np.flatnonzero(np.isin(a['path'],ENDS)&(a['fold']<5)&(errors>8.)&~a['station_fixed'])
    rejected=[];review=[]
    if not len(rows):return np.array([],int),review
    pts=base[name+'_pts'];packed=matte(s,pts)
    off=np.arange(-105.,106.)
    for pt,raw in source_frames(s['source'],np.unique(a['pts'][rows])):
        i=np.searchsorted(pts,pt);mask=foreground(s,packed,i,pt)
        frame=rows[a['pts'][rows]==pt]
        for key in np.unique(a['path'][frame]):
            ix=frame[a['path'][frame]==key];edge=np.array(native.paths[key]['parameters']).reshape(2,3)
            st=station[ix];w=edge[0]+st[:,None]*(edge[1]-edge[0])
            world=np.vstack((w,w+.012*(edge[1]-edge[0]),w+[.001,0,0]))
            pred=native.core.project_motion(base[name+'_rigs'][i],base[name+'_controls'],s['first'],
                s['dt'],1080,base[name+'_times'][i],world)
            q,q1,qx=np.split(pred['uv'],3)
            tan=q1-q;tan/=np.maximum(np.linalg.norm(tan,axis=1,keepdims=True),1e-9)
            normal=np.column_stack((-tan[:,1],tan[:,0]))
            scale=abs(np.sum((qx-q)*normal,axis=1))/.001
            cut=a['uv'][ix,None,:]+off[None,:,None]*normal[:,None,:]
            im=cv2.remap(raw,cut.astype('f4'),None,cv2.INTER_LINEAR)
            pixels=im.astype(float);brightness=pixels@np.array([.114,.587,.299])
            hidden=(pixels.max(2)-pixels.min(2)>25)|(pixels[:,:,1]<110)
            if mask is None:hidden[:]=True
            else:hidden|=cv2.remap(mask,cut.astype('f4'),None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)!=0
            for j,k in enumerate(ix):
                v=gaussian_filter1d(brightness[j],max(.95,min(2.6,.00017*scale[j])))
                g=np.gradient(v);centre=len(off)//2
                lower=find_peaks(-g,height=.10)[0];upper=find_peaks(g,height=.10)[0]
                found=False
                for lo in lower[(lower<centre)&(lower>=6)]:
                    for hi in upper[(upper>centre)&(upper<len(g)-6)]:
                        width=hi-lo;shift=abs((hi+lo)/2-centre)
                        if not max(3.,.0007*scale[j])<=width<=max(9.,.005*scale[j]) or shift>max(2.5,.18*width):continue
                        if hidden[j,lo-5:hi+6].any():continue
                        inside=np.median(v[lo+width//4:hi-width//4+1])
                        outside=min(np.median(v[lo-6:lo-1]),np.median(v[hi+2:hi+7]))
                        if outside-inside>=2.1:found=True;break
                    if found:break
                if not found:rejected.append(int(k))
                review.append(dict(row=int(k),pts=int(pt),path=key,uv=a['uv'][k].tolist(),
                    decision='retain_original_paired_paint' if found else 'unresolved_material_or_middle_exclude',
                    input_distance_px=float(errors[k])))
    return np.asarray(rejected,int),review


def prepare():
    folder=WORK/'prepared';folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json');base=camera_state(BASE);native=native_for(BASE,'jin')
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    evidence={};report={}
    with np.load(OUT/BASE/'full_qa/curve_errors.npz') as errors:
        for name,s in setup['shots'].items():
            src=prepared_packet_path(name)
            assert src.is_file()
            s['observations']=str(src)
            z=arrays(WORK/'reads'/(name+'.npz'))
            assert z['source_sha256'].item()==s['source_sha256']
            keep,stats=admitted_measurements(z,name)
            if not stats:
                with np.load(src) as archive:
                    if not np.isin(archive['path'],ENDS).any():
                        report[name]=dict(added=0,rejected_old=0,screen_frames=len(z['exposure_pts']));continue
            a=arrays(src);old_len=len(a['pts'])
            removed,old_review=check_old(a,errors[name+'_error'],errors[name+'_station'],name,s,base,native)
            a['fold'][removed]=9
            if 'restored_admission_reason' in a:
                a['restored_admission_reason'][removed]='native_source_no_identifiable_two_flank_lane_midpoint'
            if 'admission_reason' in a:
                a['admission_reason'][removed]='unresolved_source_lane_width_centre'
            rows=np.flatnonzero(keep)
            code=max(map(int,s['source_packets']))+1
            if len(rows):
                s['source_packets'][str(code)]=dict(path=str(WORK/'reads'/(name+'.npz')),source_row='NPZ row of measured source flanks',source_sha256=s['source_sha256'])
                role=6 if name in HELD_SHOTS else 0
                extra=dict(pts=z['pts'][rows],idx=z['index'][rows],uv=z['uv'][rows],
                    path=np.array(ENDS)[z['lane'][rows]],station=z['station'][rows],fold=role,source_fold=role,
                    family=np.array([f'lane{j+1}' for j in z['lane'][rows]]),material='paired_lane_end_source',
                    source_packet=code,source_row=rows,station_fixed=False,baseline_display=False,
                    origin_row=-(1+rows),admission_reason='paired_source_stripe',
                    restored_admission_reason='actual_paired_stripe_and_adjacent_source_confirmation')
                for k,v in a.items():
                    ext=np.broadcast_to(extra[k],(len(rows),)+v.shape[1:])
                    a[k]=np.concatenate((v,ext.astype(v.dtype if v.dtype.kind!='U' else str)))
            if len(removed) or len(rows):
                dest=folder/(name+'.npz');save(dest,**a);s['observations']=str(dest)
            report[name]=dict(added=int(len(rows)) if name not in HELD_SHOTS else 0,
                held_new=int(len(rows)) if name in HELD_SHOTS else 0,rejected_old=len(removed),original_rows=old_len,
                candidates=sum(v['candidate_rows'] for v in stats.values()),lane=stats,screen_frames=len(z['exposure_pts']))
            evidence[name]=old_review
            print(name,report[name]['added'],len(removed),flush=True)
    setup['parent']=str(PREPARED);setup['guide_state']=str(OUT/BASE/'state.npz')
    write(folder/'setup.json',setup);write(folder/'eligibility.json',report);write(folder/'old_lane_source_audit.json',evidence)


if __name__=='__main__':prepare()
