"""Same camera/scene mask replay and source-side depth and pose checks."""
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import digest,load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.source_support_locations import _runner_witness
from calibration.program_tracking.body_tracking.source_contact_image import polyline_nearest
from rinkspace.viewer3d.source import SourceVideo

def programme(project,data,cfg,prepared,source):
    from render_programme_body_com import Programme
    return Programme(SimpleNamespace(project=project,data_root=data,config=cfg,prepared=prepared,source_evidence=source))


def use_saved(p,parameter):
    original=p.reference
    saved=load(parameter)
    ref=absolute_reference(original,curve_from_packet(saved),root=original.root)
    ref.relative_skate_curve=shoe_curve_from_packet(saved)
    ref.binding=copy(original.binding)
    ref.binding.reference=ref
    ref.binding.x=saved['x'][:17].copy()
    ref.binding._parameter_key=ref.binding._packet=None
    p.reference,p.binding,p.mass=ref,ref.binding,ContinuousMassGeometry(ref)
    mid=ref.at(p.clock)
    C,R=p.mass.evaluate(mid,np.zeros((len(p.pts),69)))
    p.com=p.mass.com(C)
    p.com_ground=p.com/p.scale
    p.com_ground[:,2]=0.
    return mid


def body_sample(p,indices):
    base=p.reference.at(p.clock[indices])
    body=fk_state(p.reference,base,np.zeros((len(indices),69)))
    return base,body


def support(p,pts,rows):
    frames=np.searchsorted(p.pts,pts)
    clock=p.clock[frames]
    actual=clock+p.readout[frames]*(rows/1079.-.5)
    base=p.reference.at(actual)
    pose=fk_state(p.reference,base,np.zeros((len(frames),69)))
    nodes=(clock[:,None]+p.readout[frames,None]*np.linspace(-.5,.5,5)).ravel()
    base5=p.reference.at(nodes)
    pose5=fk_state(p.reference,base5,np.zeros((len(nodes),69)))
    ratio=p.binding.ratio
    xy=_runner_witness(p.binding,pose['R'][:,0],pose['O'][:,0])['world'][:,:2]*ratio
    curve=_runner_witness(p.binding,pose5['R'][:,0],pose5['O'][:,0])['world']
    uv=moving_projection(p.camera,curve.reshape(len(frames),5,1,3),frames)['uv'][:,0]
    return xy,uv


def crop_overlays(p,frames):
    from render_programme_body_com import project_batch,compose,legend
    reader=SourceVideo(str(p.source),(1,12800))
    panel=legend()
    result={}
    try:
        for a in range(0,len(frames),8):
            ids=frames[a:a+8]
            _,_,projected=project_batch(p,ids)
            for f,q in zip(ids,projected):
                stamp=int(p.pts[f])
                raw=reader.read(stamp)
                over,*_=compose(cv2.cvtColor(raw,cv2.COLOR_RGB2BGR),p,int(f),q,panel)
                result[stamp]=(raw,cv2.cvtColor(over,cv2.COLOR_BGR2RGB))
    finally:
        reader.close()
    return result


def main():
    project,data,cfg,prepared,source,parameter,annotations,footprints,audit,out=map(Path,sys.argv[1:11])
    only_legs='--legs' in sys.argv[11:]
    out.mkdir(parents=True,exist_ok=False)
    p=programme(project,data,cfg,prepared,source)
    note=json.loads(annotations.read_text('utf8'))
    ground=json.loads(footprints.read_text('utf8'))
    fit_audit=np.load(audit,allow_pickle=False)
    axes=fit_audit['camera_planar_axes'];origin=fit_audit['reference_pelvis']
    stamps=np.array([r['pts'] for r in ground])
    observed=np.array([r['polyline_uv'] for r in note['left_runner_ice_interface_bands']])
    truth=np.array([r['world_xy_model_m'] for r in ground])
    in_rows=observed.mean(1)[:,1]
    sec=p.pts/12800.
    within=(sec>=155.8)&(sec<=165.4)
    depth=(sec>=159.82)&(sec<=163.90)
    indices=np.flatnonzero(within)
    panels=[
        ('step',[150.78,150.86,150.92,150.98,151.04,151.16],(760,660),([0,13])),
        ('spin_entry',[157.06,157.16,157.24,157.30,157.36,157.48],(720,900),([0,-115])),
    ]
    if not only_legs:
        panels.append(('spin_depth',[159.92,160.24,160.40,161.48,162.74,163.12],(690,900),([0,-100])))
    frames=np.searchsorted(p.pts,np.rint(np.unique(np.concatenate([a for _,a,_,_ in panels]))*12800).astype(int))
    old_images=crop_overlays(p,frames)
    old_com=p.com.copy()
    old_P=p.reference.at(p.clock)['P']*p.binding.ratio
    old_xy,old_uv=support(p,stamps,in_rows)
    base,pose=body_sample(p,indices)
    old_left=pose['R'][:,0]
    baseline=p.reference
    mid=use_saved(p,parameter)
    new_images=crop_overlays(p,frames)
    new_com=p.com.copy()
    new_P=mid['P']*p.binding.ratio
    new_xy,new_uv=support(p,stamps,in_rows)
    _,pose=body_sample(p,indices)
    new_left=pose['R'][:,0]
    for label,times,(ww,hh),bias in panels:
        width=252;height=round(width*hh/ww);head=18
        canvas=Image.new('RGB',(len(times)*width,3*(height+head)),(22,29,34))
        draw=ImageDraw.Draw(canvas)
        for j,t in enumerate(times):
            pt=round(t*12800);f=np.searchsorted(p.pts,pt)
            hip=p.coco[f,[11,12]].mean(0)+np.array(bias)
            x,y=np.rint(hip-[ww/2,hh/2]).astype(int)
            for k,(label2,raw) in enumerate(zip(('Source','Previous','Revised'),(old_images[pt][0],old_images[pt][1],new_images[pt][1]))):
                im=Image.fromarray(raw).crop((int(x),int(y),int(x+ww),int(y+hh))).resize((width,height),Image.Resampling.LANCZOS)
                canvas.paste(im,(j*width,k*(height+head)+head))
                draw.text((j*width+5,k*(height+head)+2),f'{t:.2f}s {label2}',fill='white')
        canvas.save(out/(label+'.png'))

    fit=np.array([r['role']=='fit' for r in ground])
    reserved=np.array([r['role']=='check' for r in ground])
    def summaries(world,uv,com,P,R):
        dx=(world-truth)@axes
        source=np.asarray([polyline_nearest(q[None],band)[0][0] for q,band in zip(uv,observed)])
        errors=np.linalg.norm(source-uv,axis=1)
        h=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]))
        short=(sec[indices][:-1]>=156.60)&(sec[indices][1:]<=165.18)
        change=np.diff(h)[short]/.02
        return dict(COM_depth_range_m=float(np.ptp(com[depth,:2]@axes[:,1])),
                    pelvis_depth_range_m=float(np.ptp(P[depth,:2]@axes[:,1])),
                    supported_candidate_depth_error_median_m=float(np.median(abs(dx[fit,1]))),
                    supported_candidate_depth_error_p90_m=float(np.percentile(abs(dx[fit,1]),90)),
                    held_out_depth_error_m=float(dx[reserved,1][0]),
                    native_band_pixel_distance_median=float(np.median(errors[fit])),
                    native_band_pixel_distance_p90=float(np.percentile(errors[fit],90)),
                    held_out_band_px=float(errors[reserved][0]),
                    left_skate_reverse_20ms_count=int((change< -1.).sum()),
                    left_skate_min_heading_rate_radps=float(change.min()))
    diag=dict(old=summaries(old_xy,old_uv,old_com,old_P,old_left),
              revised=summaries(new_xy,new_uv,new_com,new_P,new_left),
              candidate_sha256=digest(parameter),old_candidate_sha256=digest(p.parameter),camera_sha256=p.state_hash)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(11.8,6.6),sharex=True,layout='constrained')
    for ax,ix,label in ((axs[0],0,'Cross-view rink position, m'),(axs[1],1,'View-depth rink position, m')):
        tr=(truth-origin)@axes
        std=np.sqrt(np.einsum('i,nij,j->n',axes[:,ix],np.array([r['covariance_model_m2'] for r in ground]),axes[:,ix]))
        tt=stamps/12800.
        ax.errorbar(tt[fit],tr[fit,ix],yerr=std[fit],fmt='o',ms=3,color='#35536b',alpha=.65,label='Source runner band')
        ax.errorbar(tt[reserved],tr[reserved,ix],yerr=std[reserved],fmt='s',ms=5,color='#482c91',label='Reserved source check')
        for name,a,c in [('Previous COM',old_com,'#a76127'),('Revised COM',new_com,'#008277')]:
            xy=(a[:,:2]-origin)@axes
            ax.plot(sec[within],xy[within,ix],label=name,color=c,lw=1.65)
        for name,a,c in [('Previous runner',old_xy,'#a76127'),('Revised runner',new_xy,'#008277')]:
            xy=(a-origin)@axes
            ax.scatter(tt[fit],xy[fit,ix],marker='x',color=c,s=15,label=name)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axs[0].legend(ncol=3,fontsize=8,loc='upper left')
    axs[1].set(xlabel='Original broadcast time, s',xlim=(159.55,164.14))
    fig.savefig(out/'com_source_comparison.png',dpi=123)
    plt.close(fig)
    with (out/'comparison.json').open('x',encoding='utf8') as handle:json.dump(diag,handle,indent=2)
    print(json.dumps(diag),flush=True)


if __name__=='__main__':
    main()
