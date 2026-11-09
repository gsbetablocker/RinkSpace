from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import argparse

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser(description='Check the saved overlay against frozen source and motion.')
for name in ('project','data','config','prepared','source','output','previous-video'):
    parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
repo,d,config,prep,ledger,out,exported=(args.project,args.data,args.config,args.prepared,args.source,args.output,args.previous_video)
from render_programme_body_com import Programme, project_batch, compose, legend, write_json
from rinkspace.viewer3d.source import SourceVideo
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.absolute_motion_curve import greville_times
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest

p=Programme(SimpleNamespace(project=repo,data_root=d,config=config,prepared=prep,source_evidence=ledger))
source=SourceVideo(str(p.source),(1,12800))
render=SourceVideo(str(exported),(1,12800))
obs=load(prep/'observations.npz')
joint_ids=body_joint_assignment(obs)
clock=p.pts/12800.
std=legend()
out.mkdir(parents=True,exist_ok=False)
proof=[]
try:
    selected=[150.78,150.88,150.98,151.08,157.16,157.24,157.34,159.82,160.48,161.48,162.74,163.90]
    indices=np.searchsorted(p.pts,np.rint(np.array(selected)*12800).astype(int))
    xyz,camera,uv=project_batch(p,indices)
    native=p.compare_native(indices,xyz,uv)
    for index, q, oracle in zip(indices,uv,native):
        pts=int(p.pts[index])
        raw=cv2.cvtColor(source.read(pts),cv2.COLOR_RGB2BGR)
        decoded=cv2.cvtColor(render.read(pts-int(p.pts[0])),cv2.COLOR_RGB2BGR)
        predicted,n,c,g=compose(raw,p,index,q,std)
        psnr=cv2.PSNR(decoded,predicted)
        proof.append(dict(pts=pts,sec=float(pts/12800),psnr_db=float(psnr),native_projection=oracle))
        if psnr<30.:
            raise ValueError(f'Existing export does not replay exactly at PTS {pts}')

    groups=[
        ('step',np.unique(np.r_[np.arange(150.56,151.46,.06),[150.78,150.98,151.10]]),4,292,270,800,740,([0,-5])),
        ('spin_entry',np.unique(np.r_[np.arange(156.94,157.67,.06),[157.24,157.38]]),4,274,330,755,910,([0,-130])),
        ('spin_ice',np.r_[np.arange(159.76,164.01,.16),[161.48]],5,236,154,380,248,([0,35])),
    ]
    notes={}
    for label, seconds, cols,w,h,ww,hh,bias in groups:
        q=np.unique(np.rint(seconds*12800).astype(int))
        ii=np.searchsorted(p.pts,q)
        np.testing.assert_array_equal(p.pts[ii],q)
        marks=[]
        for a in range(0,len(q),12 if cols==4 else 15):
            selected_ids=ii[a:a+(12 if cols==4 else 15)]
            sheet=Image.new('RGB',(cols*w,((len(selected_ids)+cols-1)//cols)*(h+21)),(24,29,33))
            draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(selected_ids):
                pt=int(p.pts[f]);raw=Image.fromarray(source.read(pt))
                anchor=p.coco[f,15] if label=='spin_ice' else p.coco[f,[11,12]].mean(0)
                x0,y0=np.rint(anchor+np.array(bias)-.5*np.array([ww,hh])).astype(int)
                crop=(int(x0),int(y0),int(x0+ww),int(y0+hh))
                image=raw.crop(crop).resize((w,h),Image.Resampling.LANCZOS)
                grid=ImageDraw.Draw(image,'RGBA')
                step=25 if label!='spin_ice' else 20
                for u in range(step,w,step):
                    grid.line((u,14,u,h),fill=(255,255,255,74));grid.text((u,1),str(u),fill='white')
                for v in range(step,h,step):
                    grid.line((17,v,w,v),fill=(255,255,255,74));grid.text((0,v),str(v),fill='white')
                x,y=j%cols*w,j//cols*(h+21)
                draw.text((x+4,y+3),f'{pt/12800:.2f} s | PTS {pt}',fill='white')
                sheet.paste(image,(x,y+21))
                marks.append(dict(pts=pt,crop=list(crop),size=[w,h]))
            sheet.save(out/f'{label}_{a//(12 if cols==4 else 15)+1}.png')
        notes[label]=marks
finally:
    render.close()
    source.close()
write_json(out/'source_marking_clock.json',notes)
with np.load(p.result_path,allow_pickle=False) as z:
    points=p.pts
    v=(clock>=149.90)&(clock<=164.80)
    t=clock[v]
    pos=z['P_world_mid'][v]; com=z['com_world_mid'][v]
    R=z['skate_R_world_mid'][v]
    feet=z['skate_reference_world_mid'][v]
    J=z['native22_world_mid'][v]
    cam=p.camera.midpoint['centre'][v]*p.scale
    p0=int(np.argmin(abs(t-161.8)))
    direction=pos[p0,:2]-cam[p0,:2]
    direction/=np.linalg.norm(direction)
    xaxis=np.array([direction[1],-direction[0]])
    A=np.stack([xaxis,direction],axis=1)
    np.savez_compressed(out/'saved_audit.npz',pts=points[v],seconds=t,P=pos,COM=com,
                        support_material=feet,R_shoe=R,joints=J,uv=p.coco[v],raw_uv=obs['uv'][v],
                        sigma=obs['sigma_px'][v],use=obs['use'][v],assignment=joint_ids[v],
                        camera_planar_axes=A,reference_pelvis=pos[p0,:2])
    drift=(t>=159.82)&(t<=163.90)
    def depth_metrics(arr):
        ray=arr[drift,:2]@direction
        return dict(range_m=float(np.ptp(ray)),end_minus_start_m=float(ray[-1]-ray[0]),
                    speed_p95_mps=float(np.percentile(abs(np.diff(ray)/.02),95)),
                    max_at_s=float(t[drift][np.argmax(ray)]),min_at_s=float(t[drift][np.argmin(ray)]))
    info={key:depth_metrics(a) for key,a in [('P',pos),('COM',com),('body_relative_to_pelvis',com-pos),('left_fixed_blade_material_point',feet[:,0])]}
    curve=curve_from_packet(load(p.parameter))
    grev=greville_times(curve.knots)+points[0]/12800.
    time_windows=[(150.42,151.44),(156.82,157.62),(159.40,164.18)]
    controls=[]
    for lo,hi in time_windows:
        cids=np.flatnonzero((grev>=lo)&(grev<=hi))
        log=curve._check_log_chart()[cids]
        rec=[(int(c),round(float(grev[c]),3),np.rint(np.degrees(np.linalg.norm(log[i,[0,1,2,4,5,7,8,10,11]],axis=-1))).astype(int).tolist()) for i,c in enumerate(cids)]
        controls.append(rec)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=(t>=156.4)&(t<=165.)
    fig,axs=plt.subplots(3,1,figsize=(11.2,7.8),layout='constrained')
    colors={'Pelvis':'#35536b','COM':'#f0a314','Left runner material':'#008677'}
    for label,a in [('Pelvis',pos),('COM',com),('Left runner material',feet[:,0])]:
        xy=(a[:,:2]-pos[p0,:2])@A
        axs[0].plot(t[s],xy[s,0],label=label,color=colors[label])
        axs[1].plot(t[s],xy[s,1],label=label,color=colors[label])
        if label!='Pelvis':axs[2].plot(xy[s,0],xy[s,1],label=label,color=colors[label])
    axs[0].set_ylabel('Cross-view position, m');axs[1].set_ylabel('View-depth position, m')
    axs[2].set(xlabel='Cross-view position, m',ylabel='View-depth position, m',aspect='equal')
    for ax in axs[:2]:
        ax.axvspan(159.82,163.90,color='#c6e8e5',alpha=.42)
        ax.set_xlabel('Original broadcast time, s')
    for ax in axs:
        ax.legend(loc='upper left')
        ax.grid(alpha=.2)
    fig.savefig(out/'saved_com_drift.png',dpi=123)
    plt.close(fig)
    write_json(out/'saved_display_and_motion.json',dict(source_video=str(p.source),video_sha256=digest(exported),
        candidate_sha256=digest(p.parameter),camera_sha256=p.state_hash,pts_offset=int(p.pts[0]),
        current_pixel_replay=proof,depth_from_camera_at_16182=direction.tolist(),depth_diagnostics=info,
        rotation_edges_deg_columns=['root','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'],
        affected_control_choices=controls))
    print(json.dumps(dict(min_decoded_psnr_db=min(a['psnr_db'] for a in proof),depth=info,controls=[len(c) for c in controls])),flush=True)
