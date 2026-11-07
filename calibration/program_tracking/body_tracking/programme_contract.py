"""Coverage/clock and boundary receipts, never spatial stitching or fitting.

Use the project's rebuild_scope.json as authority. A valid contract is not an
accuracy pass. Adjacent cut frames are different physical times: position
change is reported, NOT forced to zero. G is pelvis projection, not COM/foot.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .trajectory_metrics import summary


def required_clock(scope):
    clock=scope['source_clock']; step=int(clock['native_pts_step'])
    tb=(int(clock['time_base_num']),int(clock['time_base_den']))
    rows=[]; labels=[]
    for shot in scope['shots']:
        p=shot['processing']; first,last=int(p['first_pts']),int(p['last_pts'])
        if step<=0 or last<first or (last-first)%step:
            raise ValueError('Invalid registered shot range')
        t=np.arange(first,last+step,step,dtype=np.int64)
        if len(t)!=p['native_frame_count']:raise ValueError('Shot count mismatch')
        rows.append(t);labels.extend([shot['id']]*len(t))
    pts=np.concatenate(rows)
    if len(np.unique(pts))!=len(pts) or np.any(np.diff(pts)!=step):
        raise ValueError('Registered programme is not a unique contiguous source clock')
    target=scope['processing_scope']
    if (len(pts)!=target['native_frame_count'] or pts[0]!=target['first_pts']
            or pts[-1]!=target['last_pts'] or pts[-1]+step!=target['stop_pts_exclusive']):
        raise ValueError('Programme range/count mismatch')
    return pts,np.asarray(labels),tb


def coverage_receipt(scope, pts):
    expected,labels,tb=required_clock(scope); pts=np.asarray(pts)
    if pts.ndim!=1 or pts.dtype.kind not in 'iu' or np.any(np.diff(pts)<=0):
        raise ValueError('Ordered unique integer source PTS required')
    foreign=np.setdiff1d(pts,expected); present=np.isin(expected,pts)
    return dict(expected_frames=len(expected),present_frames=int(present.sum()),
                foreign_pts=foreign.tolist(),missing_pts=expected[~present].tolist(),
                complete=bool(present.all() and not len(foreign)),time_base=list(tb),
                by_shot={str(s):dict(expected=int((labels==s).sum()),
                     present=int(present[labels==s].sum())) for s in np.unique(labels)},
                accuracy_accepted=False)


def boundary_receipt(scope,pts,P_world,*,velocity_world=None):
    expected,labels,tb=required_clock(scope); report=coverage_receipt(scope,pts)
    pts=np.asarray(pts); p=np.asarray(P_world,float)
    if p.shape!=(len(pts),3) or not np.isfinite(p).all():
        raise ValueError('Finite world pelvis on source clock required')
    v=None if velocity_world is None else np.asarray(velocity_world,float)
    if v is not None and (v.shape!=p.shape or not np.isfinite(v).all()):
        raise ValueError('Velocity must be finite and use the same SI clock')
    index={int(t):i for i,t in enumerate(pts)}; cuts=[]
    for j in np.flatnonzero(labels[1:]!=labels[:-1]):
        a,b=int(expected[j]),int(expected[j+1]);dt=(b-a)*tb[0]/tb[1]
        row=dict(left_shot=str(labels[j]),right_shot=str(labels[j+1]),left_pts=a,right_pts=b,dt_s=dt)
        if a not in index or b not in index:
            row['status']='boundary_not_covered'
        else:
            i,k=index[a],index[b]; change=p[k]-p[i]
            row.update(status='covered_not_validated',position_change_m=change.tolist(),
                       adjacent_mean_speed_m_s=float(np.linalg.norm(change)/dt))
            if v is not None:
                innovation=change-.5*(v[i]+v[k])*dt
                row['kinematic_innovation_m']=innovation.tolist()
                row['velocity_change_m_s']=(v[k]-v[i]).tolist()
        cuts.append(row)
    report.update(cuts=cuts,no_per_shot_alignment_applied=True,
        interpretation='Cut changes camera, not athlete identity. Physical impacts may change velocity; no automatic equal-position or equal-velocity acceptance test.')
    return report


def protect_primary_arrays(parent,trial,keys,*,numerical_atol=1e-10):
    """Only for SECONDARY decoration stages that declare root/skates frozen.

    Equality tolerance is floating-point replay tolerance, NEVER ice accuracy.
    A primary-stage redesign may legitimately change these arrays and should
    be compared as a new primary candidate, not passed through this function.
    """
    if numerical_atol<0 or not np.isfinite(numerical_atol):raise ValueError('Invalid numeric tolerance')
    changed=[]
    for key in keys:
        if key not in parent or key not in trial:raise ValueError('Missing protected field '+key)
        a,b=np.asarray(parent[key]),np.asarray(trial[key])
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid protected field '+key)
        if not np.allclose(a,b,rtol=0,atol=numerical_atol):changed.append(key)
    return dict(secondary_stage_can_publish=not changed,changed_primary_fields=changed,
                numerical_atol=numerical_atol,physical_accuracy_threshold=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    scope=json.loads(a.scope.read_text(encoding='utf-8-sig'))
    with np.load(a.result,allow_pickle=False) as z:
        pts=z['pts'] if 'pts' in z else z['source_pts']
        report=boundary_receipt(scope,pts,z['P_world_mid'],velocity_world=z['P_velocity_world_mps'] if 'P_velocity_world_mps' in z else None)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
