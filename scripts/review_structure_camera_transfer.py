"""Conditional two-exposure source transfer on the frozen native board.

Both UV endpoints are source measurements. Intersections are recomputed under
EACH candidate and never become independent world observations or fit inputs.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'calibration/rinkfit/src'),str(ROOT)]
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkfit_joint.dense_source_motion import _continuous_board_exit,_inward_front
from calibration.program_tracking.body_tracking.camera_bridge import rays
from calibration.program_tracking.body_tracking.action_clip_input import camera25_from_snapshot
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=digest(Path(p)))
def stats(x):
    x=np.asarray(x,float);v=x[np.isfinite(x)]
    r=dict(rows=len(x),valid=len(v),invalid=len(x)-len(v))
    if len(v):r.update(rms=float(np.sqrt(np.mean(v*v))),median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(v.max()))
    return r

class NativeBoardTransfer:
    def __init__(self,record_path,*,controls=None):
        self.record=read(record_path);self.compact=load(self.record['state']['compact_path'])
        expected=self.record['state'].get('compact_sha256')
        if expected and digest(Path(self.record['state']['compact_path']))!=expected:raise ValueError('Camera bytes changed')
        manifest=read(Path(self.record['prepared'])/'joint_manifest.json')
        self.timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
        self.shots={s['id']:s for s in manifest['shots']}
        self.core=NativeCore(self.record['native_library']['path'])
        self.controls=self.compact['controls'] if controls is None else np.asarray(controls,float)
        if self.controls.shape!=self.compact['controls'].shape:raise ValueError('Camera layout changed')
        board=self.compact['board'];a,b,rx,ry,p,height=map(float,board)
        if a!=.5:raise ValueError('Existing board-exit primitive requires normalized half-length .5')
        def containment(xy):
            xy=np.asarray(xy,float)
            u=np.maximum(np.abs(xy[:,0])-(a-rx),0.)/rx
            v=np.maximum(np.abs(xy[:,1])-(b-ry),0.)/ry
            return u**p+v**p-1.
        self.scene=SimpleNamespace(width=2*b,rx=rx,ry=ry,p=p,height=height,containment=containment)

    def direction(self,shot_id,pts0,pts1,uv0,uv1):
        shot=self.shots[shot_id];pts0=np.asarray(pts0,np.int64);pts1=np.asarray(pts1,np.int64)
        uv0=np.asarray(uv0,float);uv1=np.asarray(uv1,float);n=len(pts0)
        if pts1.shape!=(n,) or uv0.shape!=(n,2) or uv1.shape!=(n,2):raise ValueError('Incomplete source pair layout')
        if any(np.any((p<shot['first_pts'])|(p>shot['last_pts']))for p in(pts0,pts1)):
            raise ValueError('A static source pair may not cross a cut')
        rig=self.compact['rig'][shot['camera_index']];t0=self.timeline.exposure_time(shot,pts0);t1=self.timeline.exposure_time(shot,pts1)
        H=1080;rowtime=t0+rig[37]*(uv0[:,1]/(H-1)-.5)
        pose=self.core.evaluate_motion_camera(rig,self.controls,self.timeline.first,self.timeline.dt,rowtime)
        status=np.zeros(n,np.uint8);status[pose['status']!=0]=1
        direction=np.zeros((n,3));C=pose['centre'];valid=np.flatnonzero(status==0)
        if len(valid):
            cameras=camera25_from_snapshot({k:np.asarray(v)[valid]for k,v in pose.items()})
            for i,camera in zip(valid,cameras):
                ray,ok=rays(uv0[i:i+1],camera,inverse_tolerance_px=1e-5)
                if ok[0]:direction[i]=ray[0]
                else:status[i]=1
        X,hit,lam=_continuous_board_exit(self.scene,np.nan_to_num(C),direction)
        status[(status==0)&~hit]=2
        inside=(X[:,2]>=0)&(X[:,2]<=self.scene.height)&_inward_front(self.scene,X,np.nan_to_num(C))
        status[(status==0)&~inside]=3
        inverse=np.full(n,np.nan);delta=np.full((n,2),np.nan);target_root=np.full(n,np.nan)
        ii=np.flatnonzero(status==0)
        if len(ii):
            anchor=self.core.project_motion(rig,self.controls,self.timeline.first,self.timeline.dt,H,t0[ii],X[ii])
            inverse[ii]=np.linalg.norm(anchor['uv']-uv0[ii],axis=1)
            ok=(anchor['status']==0)&(inverse[ii]<=1e-5)
            status[ii[~ok]]=4
        ii=np.flatnonzero(status==0)
        if len(ii):
            target=self.core.project_motion(rig,self.controls,self.timeline.first,self.timeline.dt,H,t1[ii],X[ii])
            target_root[ii]=target['root_equality_px'];ok=target['status']==0;status[ii[~ok]]=5
            target_time=t1[ii]+rig[37]*(target['row']/(H-1)-.5)
            target_pose=self.core.evaluate_motion_camera(rig,self.controls,self.timeline.first,self.timeline.dt,target_time)
            front=(target_pose['status']==0)&_inward_front(self.scene,X[ii],target_pose['centre'])
            status[ii[ok&~front]]=6;ok&=front
            finite=np.isfinite(target['uv']).all(1);status[ii[ok&~finite]]=7;ok&=finite
            delta[ii[ok]]=target['uv'][ok]-uv1[ii[ok]]
        return dict(status=status,delta=delta,error=np.linalg.norm(delta,axis=1),anchor_closure_px=inverse,
                    target_root_equality_px=target_root,conditional_xyz=X)

    def validate_numerics(self):
        results=[]
        # Synthetic finite-board samples check units/branches/RS only, never
        # provide source evidence or contribute to camera selection.
        perimeter=self.core.board_perimeter(self.compact['board'])
        X=np.asarray(self.core.board_points(self.compact['board'],np.linspace(0,perimeter,513)[:-1],np.full(512,self.scene.height*.5)))
        for sid,seconds in [('s05',[101.18,101.40]),('s07',[111.38,111.58]),('s08',[136.16,136.38]),('s08',[143.14,143.36])]:
            shot=self.shots[sid];rig=self.compact['rig'][shot['camera_index']];pts=np.rint(np.asarray(seconds)*12800).astype(np.int64)
            tt=self.timeline.exposure_time(shot,pts);pr=[self.core.project_motion(rig,self.controls,self.timeline.first,self.timeline.dt,1080,t,X)for t in tt]
            ok=np.logical_and.reduce([(p['status']==0)&np.isfinite(p['uv']).all(1)&(p['uv'][:,0]>5)&(p['uv'][:,0]<1915)&(p['uv'][:,1]>5)&(p['uv'][:,1]<1075)for p in pr])
            mid=self.core.evaluate_motion_camera(rig,self.controls,self.timeline.first,self.timeline.dt,[tt[0]])
            ok&=_inward_front(self.scene,X,np.repeat(mid['centre'],len(X),axis=0))
            idx=np.flatnonzero(ok)
            if not len(idx):raise ValueError('No synthetic front-board numerical check samples')
            v=self.direction(sid,np.full(len(idx),pts[0]),np.full(len(idx),pts[1]),pr[0]['uv'][idx],pr[1]['uv'][idx])
            maximum=float(np.max(np.linalg.norm(v['conditional_xyz']-X[idx],axis=1)))
            if np.any(v['status'])or maximum>1e-8 or v['error'].max()>1e-4:raise ValueError('Native board transfer numerical closure failed')
            results.append(dict(shot=sid,source_pts=pts.tolist(),samples=len(idx),max_XYZ_closure_scene_units=maximum,error_px=stats(v['error'])))
        return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera-record',type=Path,required=True)
    p.add_argument('--pairs',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--numerical-check',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);model=NativeBoardTransfer(a.camera_record)
    if a.numerical_check:
        report=dict(schema='rinkspace.native-board-transfer-numerical-check.v1',camera=ref(a.camera_record),results=model.validate_numerics(),source_pixels_used=False)
    else:
        if a.pairs is None:raise ValueError('Frozen source pair manifest required')
        pairs=read(a.pairs)
        if not pairs.get('source_only_review_complete')or pairs.get('fit_role')!='independent_final_validation':raise ValueError('Source pairing not frozen for independent validation')
        rows=pairs['rows'];records=[];saved={}
        for group in sorted({r['group']for r in rows}):
            chosen=[r for r in rows if r['group']==group];sid=chosen[0]['shot_id']
            if any(r['shot_id']!=sid for r in chosen):raise ValueError('Cross-cut group')
            pa=np.array([r['source_pts_a']for r in chosen]);pb=np.array([r['source_pts_b']for r in chosen])
            ua=np.array([r['uv_a']for r in chosen]);ub=np.array([r['uv_b']for r in chosen])
            f=model.direction(sid,pa,pb,ua,ub);b=model.direction(sid,pb,pa,ub,ua)
            for name,result in [('forward',f),('reverse',b)]:
                # Deliberately do not export inferred XYZ as a measurement file.
                for key,value in result.items():
                    if key!='conditional_xyz':saved[group+'_'+name+'_'+key]=value
            records.append(dict(group=group,rows=len(chosen),forward=stats(f['error']),reverse=stats(b['error']),
                source_pts=sorted(set(pa.tolist()+pb.tolist())),all_native_rows_valid=bool(np.all(f['status']==0)&np.all(b['status']==0))))
        np.savez_compressed(a.output/'source_transfer.npz',**saved)
        report=dict(schema='rinkspace.native-board-source-transfer.v1',camera=ref(a.camera_record),camera_compact=ref(model.record['state']['compact_path']),pairs=ref(a.pairs),groups=records,
            interpretation='Conditional relative camera motion on the frozen board; both endpoints are original image measurements; each candidate rebuilds its own latent board intersection.',
            world_measurements_created=False,cross_cut_correspondence=False,accuracy_accepted=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
