"""Reread the faint S04 C material identified in the actual 929536 exposure.

The user trace and unchanged current native motion only navigate the source.
Every output still requires the existing RGB two-flank and straight-line reader.
"""
from pathlib import Path
import argparse
import hashlib
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from review_points_s04 import CurrentNativeGrid,OUT,ADS,FIRST,LAST,STEP,read,write
from reread_s04_lines import NativeLineReader
from ice_points_s04 import source_domain,REC
from rinkspace.vision_native import file_ref
from calibration.program_tracking.extract_first_shot_ice import NativeProfiles,Extractor
from calibration.program_tracking.native_ice_line_continuity import straight_guide
from calibration.program_tracking.repair_physical_wide_bands import PhysicalBandProfiles
from calibration.program_tracking.projection import ground_xy
from rinkspace.registration import adjacent_lk


class CurrentNativeGroundCamera:
    """Adapter for the historical physical-width reader, including native RS."""
    def __init__(self,grid,pt):
        self.grid=grid;self.pt=pt
        self.time=(pt-grid.shot['reference_pts'])/12800+grid.timeline.offset(grid.shot)
        c=grid.core.evaluate_motion_camera(grid.state['rig'],grid.state['controls'],grid.timeline.first,grid.timeline.dt,[self.time])
        if c['status'][0]!=0:raise ValueError('Current camera unavailable')
        distortion=np.zeros(14);distortion[:4]=[c[k][0]for k in ('k1','k2','p1','p2')]
        self.initial=np.r_[c['centre'][0],cv2.Rodrigues(c['rotation'][0])[0].ravel(),
            np.log(c['fx'][0]),np.log(c['fy'][0]),c['cx'][0],c['cy'][0],0.,distortion]

    def project(self,xyz):
        g=self.grid;r=g.core.project_motion(g.state['rig'],g.state['controls'],g.timeline.first,g.timeline.dt,1080,self.time,xyz)
        return r['uv'],np.where(r['status']==0,r['depth'],np.nan)

    def ground_xy(self,pixels):
        pixels=np.asarray(pixels,float);xy,valid=ground_xy(self.initial,pixels)
        if not valid.all():return xy,valid
        for _ in range(12):
            xyz=np.c_[xy,np.zeros(len(xy))];uv,depth=self.project(xyz);error=uv-pixels
            if np.linalg.norm(error,axis=1).max()<1e-6:break
            dx=xyz.copy();dy=xyz.copy();dx[:,0]+=.001;dy[:,1]+=.001
            jac=np.stack([(self.project(dx)[0]-uv)/.001,(self.project(dy)[0]-uv)/.001],axis=-1)
            xy-=np.linalg.solve(jac,error[...,None])[...,0]
        uv,depth=self.project(np.c_[xy,np.zeros(len(xy))]);valid&=np.isfinite(uv).all(1)&(depth>.05)&(np.linalg.norm(uv-pixels,axis=1)<.001)
        return xy,valid


def photographic_navigation(guides,annotation,frames):
    """Existing static-material image registration supplies search proposals."""
    anchor=annotation['source_pts'];prepared={};pairs={};scale=np.diag([.5,.5,1.])
    for pt in sorted(guides):
        v=frames[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Motion source changed')
        board=read(REC/f'dense_board_v1/qualified_frames/{pt}.json');graphics,border,distance,_=source_domain(im,mask,board)
        upper=np.asarray(board['roles']['board.top']['trace_uv']);upper=upper[np.argsort(upper[:,0])]
        top=np.interp(np.arange(1920),upper[:,0],upper[:,1]);y=np.arange(1080)[:,None]
        keep=(~graphics)&(distance>8)&(y>top[None]+8)&(y<border[None]-8)
        prepared[pt]=(cv2.resize(im,(960,540)),cv2.resize(keep.astype('uint8')*255,(960,540),interpolation=cv2.INTER_NEAREST))
        if pt-STEP in prepared:
            a,b=prepared[pt-STEP],prepared[pt];r=adjacent_lk(a[0],b[0],a[1],b[1]);r.update(source_pts=pt,previous_pts=pt-STEP)
            pairs[pt]=r
    out={anchor:guides[anchor]};receipt={anchor:dict(origin='actual_user_material_identity')}
    for direction in (-1,1):
        previous=anchor
        ordered=sorted((p for p in guides if (p-anchor)*direction>0),reverse=direction<0)
        for pt in ordered:
            r=pairs[max(pt,previous)]
            if r['status']=='candidate'and r['diagnostic_median_reference_px']<=2.:
                h=np.linalg.inv(scale)@np.asarray(r['H_reference_from_current'])@scale
                if direction==1:h=np.linalg.inv(h)
                proposal=cv2.perspectiveTransform(out[previous][None],h)[0]
                out[pt]=proposal
                receipt[pt]=dict(origin='adjacent_actual_static_board_RGB_search_only',pair=r,
                    accumulated_map_is_not_physical_validation=True,source_material_must_be_reread=True)
            else:
                out[pt]=guides[pt];receipt[pt]=dict(origin='current_native_relative_search_after_unusable_source_registration',pair=r)
            previous=pt
    return out,receipt


def project(grid,pt,xyz):
    time=(pt-grid.shot['reference_pts'])/12800+grid.timeline.offset(grid.shot)
    result=grid.core.project_motion(grid.state['rig'],grid.state['controls'],
        grid.timeline.first,grid.timeline.dt,1080,time,np.asarray(xyz,float))
    return result['uv'],result['status']==0


def source_navigation(grid,annotation):
    pt=annotation['source_pts'];targets=np.asarray(annotation['source_guide_uv']);xy=[]
    all_uv,valid=project(grid,pt,grid.samples['xyz'])
    ground=np.asarray(grid.samples['xyz']);valid&=abs(ground[:,2])<1e-9
    for target in targets:
        k=np.argmin(np.where(valid,np.linalg.norm(all_uv-target,axis=1),np.inf))
        fit=least_squares(lambda v:project(grid,pt,np.r_[v,0.][None])[0][0]-target,
            ground[k,:2],xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
        if np.linalg.norm(fit.fun)>.001:raise ValueError('Current native source navigation inverse failed')
        xy.append(fit.x)
    world=np.c_[np.linspace(*xy,257),np.zeros(257)]
    guides={};visible=[]
    for pt in range(FIRST,LAST+1,STEP):
        uv,good=project(grid,pt,world)
        near=good&(uv>=[-185,0]).all(1)&(uv<=[2104,1079]).all(1)
        if near.sum()>=2:
            guide=straight_guide(uv[good],uv[good])
            if len(guide)>=2:guides[pt]=guide;visible.append(pt)
    parts=np.split(np.asarray(visible,int),np.flatnonzero(np.diff(visible)!=STEP)+1)
    part=next(p for p in parts if annotation['source_pts']in p)
    return {int(pt):guides[int(pt)]for pt in part},dict(
        method='inverse_current_native_z0_of_source_trace_then_current_native_same_shot_motion',
        anchor_pts=annotation['source_pts'],anchor_world_search_only=np.asarray(xy).tolist(),
        native_geometry_or_camera_changed=False,projection_coordinates_emitted=False,
        source_connected_visibility_component=[int(part[0]),int(part[-1])])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(2);start=time.monotonic();grid=CurrentNativeGrid()
    annotation_path=OUT/'evidence/C_source_identity_929536.json';annotation=read(annotation_path)
    guides,nav=source_navigation(grid,annotation)
    reader=NativeLineReader(grid);path=next(p for p in reader.paths if p['feature_id']=='C')
    visibility=read(ADS/'evidence/visibility.json');frames={r['pts']:r for r in visibility['frames']}
    native_guides=guides
    guides,photo=photographic_navigation(guides,annotation,frames)
    dependencies=[Path(__file__),*[ROOT/'calibration/program_tracking'/name for name in
        ('extract_first_shot_ice.py','native_ice_material_repair.py','physical_ice_width.py','repair_physical_wide_bands.py')]]
    algorithm=hashlib.sha256(b''.join(p.read_bytes()for p in dependencies)).hexdigest()
    dest=OUT/'evidence/faint_C_RGB';dest.mkdir(exist_ok=True);results=[]
    for number,(pt,guide)in enumerate(sorted(guides.items())):
        v=frames[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Actual exposure changed')
        graphics,border,_,_=source_domain(im,mask,read(REC/f'dense_board_v1/qualified_frames/{pt}.json'))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        native=PhysicalBandProfiles(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dict(source_video_sha256=visibility['source_video_sha256'],
            native_pts=pt,time_base='1/12800'),excluded,CurrentNativeGroundCamera(grid,pt))
        policy=dict(current_identity_constrained=True,identity_source='user_faint_C_material_identity_with_same_shot_actual_image_motion',
            reference_frame=annotation['source_pts'],normal_identity_tolerance_px=28.,search_radius_px=28.)
        # Use the existing 35px material-query spacing on this short visible
        # trace. Output density is independently reduced by the common selector.
        local=dict(path,_line_identity_policy=policy,_source_query_spacing_px=35)
        if len(guide)<2:
            feature=dict(extraction_diagnostic=dict(query_count=0,reason='source_transported_material_outside_image'))
        else:feature,_,_=Extractor.line(None,local,guide,native)
        fit=feature.get('source_line_fit');rows=[]
        if fit:
            for index,r in enumerate(fit['direct_readouts']):
                rows.append(dict(r,native_path_id=path['id'],feature_id=path['label'],native_pts=pt,
                    source_row_id=f's04_faint_C_actual_RGB/{pt}/{index}'))
        record=dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],algorithm_sha256=algorithm,
            annotation_ref=file_ref(annotation_path),navigation=dict(**nav,current_search_guide=guide.tolist(),
                actual_image_motion=photo[pt],current_native_relative_guide=native_guides[pt].tolist()),
            diagnostic=feature['extraction_diagnostic'],source_line_fit={k:v for k,v in fit.items()if k!='direct_readouts'}if fit else None,
            rows=rows,algorithm_dependencies=[file_ref(p)for p in dependencies],
            projected_or_human_trace_coordinates_emitted=False,human_acceptance=False)
        result_path=dest/f'{pt}.json';write(result_path,record)
        baseline=OUT/f'evidence/native_family_line_RGB/{pt}.json';packet=read(baseline)
        if args.apply:
            backup=dest/f'{pt}.before.json'
            if not backup.exists():write(backup,dict(source_pts=pt,source_ref=file_ref(baseline),
                feature=packet['features'].get(path['label']),diagnostic=packet['diagnostics'].get(path['label'])))
            before_other={k:v for k,v in packet['features'].items()if k!=path['label']}
            packet['features'].pop(path['label'],None)
            if rows:
                packet['features'][path['label']]=dict(native_path_id=path['id'],rows=rows,
                    navigation=dict(search_origin=policy['identity_source'],source_line_fit=record['source_line_fit'],
                    current_native_finite_support=True,fitted_representatives_exported=False,
                    current_source_repair=file_ref(result_path)))
            packet['diagnostics'][path['label']]=dict(record['diagnostic'],source_repair=file_ref(result_path),
                no_actual_reading_reason=None if rows else 'two_clear_ice_side_flanks_and_coherent_separated_readouts_not_available')
            packet['C_source_material_repair']=dict(previous_C_preserved=file_ref(backup),current=file_ref(result_path))
            if {k:v for k,v in packet['features'].items()if k!=path['label']}!=before_other:raise ValueError('Unrelated entity modified')
            pending=baseline.with_suffix('.pending.json');write(pending,packet);pending.replace(baseline)
        results.append(dict(pts=pt,actual_readouts=len(rows),result=file_ref(result_path),
            previous_actual_readouts=len(read(dest/f'{pt}.before.json')['feature']['rows'])if args.apply and read(dest/f'{pt}.before.json')['feature']else 0))
        if number%10==0:print(dict(frame=number+1,total=len(guides),pts=pt,actual_reads=len(rows)),flush=True)
    grid.unchanged();write(dest/'summary.json',dict(frames=results,navigation=nav,native_authority=grid.receipt,
        applied=args.apply,actual_readouts=sum(r['actual_readouts']for r in results),
        source_precision_from_measured_flank_quantiles=True,human_acceptance=False,seconds=time.monotonic()-start))
    print(dict(complete=True,frames=len(results),nonempty=sum(r['actual_readouts']>0 for r in results),
        actual_readouts=sum(r['actual_readouts']for r in results),seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':main()
