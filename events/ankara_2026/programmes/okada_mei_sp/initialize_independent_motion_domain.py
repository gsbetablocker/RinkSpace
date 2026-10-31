"""Finite native-projection navigation seeds; no observations or rig edits.

Scan alternate new-shot pan/zoom seeds only when a named observed curve has no
valid projected support. Source UV and physical identities remain unchanged;
production pixel optimization is left to the shared native solver.
"""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from rinkspace.core.native import NativeCore
from capture_independent_materials import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);a=p.parse_args()
    base=json.loads((a.root.parent/'baseline.json').read_text('utf8'));core=NativeCore(base['inputs']['library']['path']);world=dict(np.load(a.root/'native_world.npz'));cuts={c['id']:c for c in json.loads((a.root/'native_world.json').read_text('utf8'))['cuts']};summary={}
    for sid in ['s07','s08','s09']:
        folder=a.capture/sid;original=folder/'poses.npz';backup=folder/'poses_before_domain_seed.npz'
        if backup.exists():raise FileExistsError(backup)
        z=dict(np.load(original));navpts=z['pts'];rows=[json.loads(x)for x in(folder/'source_rows.jsonl').read_text('utf8').splitlines()];nav=z['poses'];cam=2 if sid=='s08'else 0;rig=world['rig'][cam];samples=[];adjustments=[]
        for frame in rows:
            if not frame['rows']:continue
            pts=frame['pts'];theta=nav[np.flatnonzero(navpts==pts)[0]].copy();groups={}
            for row in frame['rows']:groups.setdefault(row['path_id'],[]).append(row['uv'])
            xx=[];ends=[];observed=[]
            for path,uv in groups.items():
                c=cuts[path];xyz=world['xyz'][c['begin']:c['end']];take=np.linspace(0,len(xyz)-1,min(40,len(xyz))).round().astype(int);xx.extend(xyz[take]);ends.append(len(xx));observed.append(np.asarray(uv))
            xyz=np.asarray(xx)
            def score(q):
                result=core.project_motion(rig,np.tile(q,(6,1)),-2.,1.,1080,0.,xyz);err=[];start=0
                for end,uv in zip(ends,observed):
                    projected=result['uv'][start:end];valid=result['status'][start:end]==0;begin=projected[:-1];delta=np.diff(projected,axis=0);ok=valid[:-1]&valid[1:]
                    if not ok.any():return float('inf')
                    begin=begin[ok];delta=delta[ok];norm=(delta*delta).sum(1);difference=uv[:,None]-begin;station=np.clip((difference*delta).sum(2)/np.maximum(norm,1e-20),0,1);distance=((difference-station[:,:,None]*delta)**2).sum(2);err.extend(np.min(distance,axis=1));start=end
                return float(np.mean(err))
            initial=score(theta);best=(initial,theta)
            if not np.isfinite(initial):
                # These are navigation trials, not physical constraints. The
                # actual native joint solve can freely change every active value.
                pans=np.r_[theta[0],theta[0]+np.linspace(-.5,.5,11)];zooms=np.unique(np.r_[theta[2],np.log([3500.,4500.,5500.,7000.,10000.])])
                for pan in pans:
                    for zoom in zooms:
                        q=np.array([pan,theta[1],zoom]);s=score(q)
                        if s<best[0]:best=(s,q)
                if not np.isfinite(best[0]):raise ValueError(f'No native-valid source-curve navigation seed: {sid}/{pts}')
                adjustments.append(dict(pts=pts,before=theta.tolist(),after=best[1].tolist(),source_curve_navigation_rms_px=float(np.sqrt(best[0])),reason='invalid_projected_support_for_named_observed_curve',source_rows_excluded=0,physical_model_changed=False))
            samples.append((pts,best[1]))
        xp=np.array([s[0]for s in samples]);values=np.array([s[1]for s in samples]);result=np.column_stack([np.interp(navpts,xp,values[:,k])for k in range(3)])
        shutil.copyfile(original,backup);np.savez_compressed(original,pts=navpts,poses=result)
        summary[sid]=dict(initialization_only=True,source_UV_unchanged=True,rig_unchanged=True,native_solver_started=False,unsupported_source_intervals_have_only_interpolated_navigation=True,adjustments=adjustments)
        print(sid,len(adjustments),'domain navigation changes',flush=True)
    write(a.capture/'navigation_domain_seed.json',summary)
if __name__=='__main__':main()
