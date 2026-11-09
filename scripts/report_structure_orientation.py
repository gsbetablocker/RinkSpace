"""Compare actual pelvis/chest/shoe motion and distinct shoe diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.rebuild_prepare import digest

EPISODES={'2A':(77.80,78.14),'LSp':(156.50,165.30),'CCoSp_first':(172.08,179.04),'CCoSp_transfer':(178.80,179.40),'CCoSp_second':(179.26,184.52),'CCoSp_release':(184.52,184.80)}
def stats(x):
    x=np.asarray(x);return dict(n=int(x.size),minimum=float(x.min()),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))
def intervals(mask,pts):
    ids=np.flatnonzero(mask)
    return [dict(start_pts=int(pts[a[0]]),end_pts_inclusive=int(pts[a[-1]+1])) for a in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1) if len(a)]
def omega(R,dt):
    np.testing.assert_allclose(np.swapaxes(R,-1,-2)@R,np.broadcast_to(np.eye(3),R.shape),rtol=0,atol=2e-5)
    return Rotation.from_matrix(R[1:]@np.swapaxes(R[:-1],-1,-2)).as_rotvec()/dt[:,None]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',action='append',required=True,help='NAME=RESULT|ANATOMY|ACTION_REPORT');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);report={'schema':'rinkspace.actual-pose-orientation.v1','accuracy_accepted':False,'candidates':{},'meaning':['Original anatomy FK_frames are root-local; this report explicitly exports FK_frames_world = global_root_R @ FK_frames.','Discrete actual SO(3) increments on native dt; no exact revolution count or continuous speed certificate.','Hip-to-shoulder axis is the saved model anatomical long axis; its signed rate is not an independently measured spin axis.','World-Z, body-axis and shoe-heading rates are separate conditional diagnostics. Their negative counts alone do not establish a source-visible reversal.','Shoe origin, fixed steel-web-top material point, current lowest rendered shoe-surface vertex, allowed-clearance feature witness, and sparse observed source contours remain distinct.','Lowest rendered mesh point is exact for the saved triangulated shoe surface against horizontal z; it is not a measured ice contact.']};clock=None
    for arg in a.candidate:
        label,spec=arg.split('=',1);rp,ap,fp=map(Path,spec.split('|'))
        if label in report['candidates'] or not label.replace('_','').isalnum():raise ValueError('Unique simple candidate labels required')
        with np.load(rp,allow_pickle=False) as r,np.load(ap,allow_pickle=False) as f:
            pts=r['source_pts'];tb=r['time_base'];seconds=pts*float(tb[0])/float(tb[1]);dt=np.diff(seconds)
            np.testing.assert_array_equal(pts,f['source_pts']);np.testing.assert_allclose(r['Q_world_mid'],f['global_root_R'],rtol=0,atol=1e-10)
            world_frames=r['Q_world_mid'][:,None]@f['FK_frames'];np.testing.assert_allclose(world_frames[:,0],r['Q_world_mid'],rtol=0,atol=1e-10)
            if clock is None:clock=pts.copy()
            else:np.testing.assert_array_equal(pts,clock)
            J=r['native22_world_mid'];axis=(J[:,16]+J[:,17]-J[:,1]-J[:,2])/2;axis/=np.linalg.norm(axis,axis=1)[:,None];midaxis=axis[:-1]+axis[1:];midaxis/=np.linalg.norm(midaxis,axis=1)[:,None]
            matrices={'pelvis':r['Q_world_mid'],'chest':world_frames[:,9],'left_shoe':r['skate_R_world_mid'][:,0],'right_shoe':r['skate_R_world_mid'][:,1]}
            arrays={'source_pts':pts,'time_base':tb,'FK_frames_world':world_frames,'hip_to_shoulder_axis_world':axis,'skate_origin_world_m':r['skate_origin_world_mid'],'fixed_web_top_material_world_m':r['skate_reference_world_mid'],'clearance_feature_witness_world_m':r['skate_witness_world_mid']}
            lows=[];indices=[]
            for key in ('skate_lower_world_mid','skate_upper_world_mid'):
                V=r[key];ix=V[...,2].argmin(axis=2);lows.append(np.take_along_axis(V,ix[...,None,None],axis=2)[:,:,0]);indices.append(ix)
            lower=lows[0][...,2]<=lows[1][...,2];arrays['lowest_surface_world_m']=np.where(lower[...,None],lows[0],lows[1]);arrays['lowest_surface_is_lower']=lower;arrays['lowest_surface_vertex']=np.where(lower,indices[0],indices[1])
            for name,R in matrices.items():
                w=omega(R,dt);arrays[name+'_omega_world_radps']=w;arrays[name+'_rate_hip_shoulder_axis_radps']=np.einsum('ni,ni->n',w,midaxis)
            for foot,name in enumerate(('left','right')):
                forward=r['skate_R_world_mid'][:,foot,:,0];xy=forward[:,:2];valid=np.linalg.norm(xy,axis=1)>1e-8
                rates=np.full(len(dt),np.nan);use=valid[:-1]&valid[1:];angles=np.arctan2(xy[:,1],xy[:,0]);rates[use]=np.angle(np.exp(1j*np.diff(angles[None,:],axis=1)[0][use]))/dt[use]
                arrays[name+'_heel_to_toe_world']=forward;arrays[name+'_heading_rate_world_xy_radps']=rates
            np.savez_compressed(a.output/(label+'.npz'),**arrays)
            action=json.loads(fp.read_text(encoding='utf8'));physics=action['source_and_physics'];episodes={}
            for name,(lo,hi) in EPISODES.items():
                selected=(seconds[:-1]>=lo-1e-8)&(seconds[1:]<=hi+1e-8);sample=(seconds>=lo-1e-8)&(seconds<=hi+1e-8);motion={}
                for part in matrices:
                    w=arrays[part+'_omega_world_radps'];rate=arrays[part+'_rate_hip_shoulder_axis_radps'];motion[part]=dict(world_Z=stats(w[selected,2]),along_hip_shoulder_axis=stats(rate[selected]),angular_speed=stats(np.linalg.norm(w[selected],axis=1)),negative_world_Z_intervals=intervals(selected&(w[:,2]<0),pts),negative_body_axis_intervals=intervals(selected&(rate<0),pts),largest_native_rotation_step_rad=float((np.linalg.norm(w[selected],axis=1)*dt[selected]).max()))
                feet={}
                for fi,foot in enumerate(('left','right')):
                    feet[foot]={key:dict(xyz_span_m=np.ptp(arrays[key][sample,fi],axis=0).tolist(),z_m=stats(arrays[key][sample,fi,2])) for key in ('skate_origin_world_m','fixed_web_top_material_world_m','lowest_surface_world_m','clearance_feature_witness_world_m')}
                    h=arrays[foot+'_heading_rate_world_xy_radps'][selected];feet[foot]['heel_to_toe_heading_rate_world_xy_radps']=stats(h[np.isfinite(h)]) if np.isfinite(h).any() else None
                episodes[name]=dict(source_seconds=[lo,hi],native_samples=int(sample.sum()),motion=motion,feet=feet)
            facing={key:{k:v[k] for k in ('selected_direction_sign','direction_sign_provenance','reverse_intervals','integrated_signed_yaw_proxy_turns','exact_revolutions_imposed','facing_rows') if k in v} for key,v in physics.items() if isinstance(v,dict) and 'facing_rows' in v}
            report['candidates'][label]=dict(result=dict(path=str(rp),sha256=digest(rp)),anatomical_frames=dict(path=str(ap),sha256=digest(ap)),action_report=dict(path=str(fp),sha256=digest(fp)),arrays=dict(path=str(a.output/(label+'.npz')),sha256=digest(a.output/(label+'.npz'))),episodes=episodes,source_facing_diagnostics=facing,source_visible_contour_rows=[v for v in physics.get('new_source_skate_regions',[]) if v['pts']>=round(156.5*12800)])
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(a.output/'report.json',flush=True)
if __name__=='__main__':main()
