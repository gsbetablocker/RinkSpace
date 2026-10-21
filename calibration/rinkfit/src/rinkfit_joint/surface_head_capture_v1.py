"""Fit real shared head/optics to a 3D numerical transfer reference.

Reference UVs are old forward evaluations, never source observations.  This
small initializer consumes each target's source-row time and every retained
board height.  It does not copy historical K/R/C or modify source packets.
"""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,file_sha256,object_sha256
from .joint_model import as_event_view
from .native_baseline_init import _physical,_free
from rinkfit_start.native_scene import camera_from_event
from rinkfit_start.native_camera import rotation


def _summary(residual):
    a=np.linalg.norm(residual,axis=-1)
    return dict(denominator=len(a),rms=float(np.sqrt(np.mean(a*a))),p95=float(np.percentile(a,95)),max=float(np.max(a)))


def build_candidate(model_file, reference_file, outdir, *, rigs=('secondary','end')):
    out=Path(outdir).resolve()
    if out.exists(): raise FileExistsError('Use a new immutable head capture generation')
    out.mkdir(parents=True)
    model=read_json(model_file);parent=copy.deepcopy(model)
    ref=read_json(reference_file)
    path=Path(ref['arrays']['path'])
    if file_sha256(path)!=ref['arrays']['sha256']:raise ValueError('Reference arrays changed')
    if ref.get('source_observations_created')!=0 or ref['baseline_camera_stage']!='demo_source; V3 patch excluded':
        raise ValueError('Explicit original numerical transfer reference required')
    if set(model['rigs'])!={'main','secondary','end'} or len(model['shots'])!=9:raise ValueError('Shared model structure changed')
    transform=model.get('native_old_xyz_transform')
    if not transform:raise ValueError('An explicit old XYZ/native L1 chart transform is required')
    data=np.load(path,allow_pickle=False)
    fi=data['row_frame_index'];wi=data['row_world_index']
    all_pts=data['native_pts'];all_numbers=data['shot_number']
    pts=all_pts[fi];numbers=all_numbers[fi]
    world=data['world_xyz'][wi];target=data['target_uv']
    entity=data['row_entity'];allowed=data['row_navigation_allowed']
    reports={};changed=set()
    for rid in rigs:
        if rid=='main':raise ValueError('Main unlock requires simultaneous full S01 source factors; do not omit them')
        rig=model['rigs'][rid];hb=model['blocks'][rig['head_block']];ob=model['blocks'][rig['optics_block']]
        sids=[s for s,v in model['shots'].items() if v['rig_id']==rid]
        select=np.isin(numbers,[int(s[1:]) for s in sids])&allowed
        ids=np.flatnonzero(select);X=world[ids];uv=target[ids];pp=pts[ids];sn=numbers[ids]
        if not len(ids):raise ValueError('No permitted numerical navigation for '+rid)
        h0=np.asarray(hb['values'],float);o0=np.asarray(ob['values'],float)
        q0=np.empty((len(ids),3))
        for sid in sids:
            mask=sn==int(sid[1:]);shot=model['shots'][sid]
            cam=camera_from_event(as_event_view(model,sid))
            rowtime=(pp[mask]-shot['time_origin_pts'])/12800.+o0[11]*(uv[mask,1]/1079.-.5)
            q0[mask]=cam.spline(rowtime)
            q0[mask,:2]+=np.asarray(model['blocks'][shot['reference_angles_block']]['values'])
        refs=[];lo=[];hi=[];sc=[]
        for j in range(15):
            if _free(hb,j):refs.append(('h',None,j));lo.append(hb['lower'][j]-h0[j]);hi.append(hb['upper'][j]-h0[j]);sc.append(hb['scales'][j])
        # This remains provisional numeric calibration.  Readout is retained;
        # it is not estimated from a historical frame-center reference.
        for j in (0,1,5,7,9,10):
            if _free(ob,j):refs.append(('o',None,j));lo.append(ob['lower'][j]-o0[j]);hi.append(ob['upper'][j]-o0[j]);sc.append(ob['scales'][j])
        for sid in sids:
            shot=model['shots'][sid];ab=model['blocks'][shot['reference_angles_block']]
            for j in (0,1):
                if _free(ab,j):refs.append(('r',sid,j));lo.append(ab['lower'][j]-ab['values'][j]);hi.append(ab['upper'][j]-ab['values'][j]);sc.append(ab['scales'][j])
            blocks=[model['blocks'][i] for i in shot['control_blocks']]
            if all(_free(b,2) for b in blocks):
                refs.append(('z',sid,2));lo.append(max(b['lower'][2]-b['values'][2] for b in blocks));hi.append(min(b['upper'][2]-b['values'][2] for b in blocks));sc.append(blocks[0]['scales'][2])
        lo,hi,scale=map(np.asarray,(lo,hi,sc));N=len(refs)
        # Each original entity has one total weight per exposure.  This is a
        # declared numerical transfer metric, not a covariance or source W.
        labels=fi[ids].astype(np.int64)*len(ref['entity_names'])+entity[ids]
        _,inverse,counts=np.unique(labels,return_inverse=True,return_counts=True)
        weight=1/np.sqrt(counts[inverse])
        def evaluate(delta):
            h=h0.copy();o=o0.copy();q=q0.copy()
            for (kind,sid,j),v in zip(refs,delta):
                if kind=='h':h[j]+=v
                elif kind=='o':o[j]+=v
                else:q[sn==int(sid[1:]),j]+=v
            val,depth,radius,*_=_physical(q,h,o,rig['camera'],X[:,None,:])
            val=val[:,0];valid=np.isfinite(val).all(-1)&(depth[:,0]>0)&(radius[:,0]<=rig['camera']['radius_max'])
            return val,valid
        delta=np.zeros(N);pred,valid=evaluate(delta)
        if not valid.all():
            reports[rid]=dict(started=False,invalid=int((~valid).sum()),denominator=len(valid),reason='initial_numeric_reference_domain_invalid',dropped_rows=0)
            continue
        error=pred-uv;before=error.copy();res=(error*weight[:,None]).ravel();cost=float(res@res/2)
        initial=cost;damping=1e-3;steps=0;rejected=0;stop='';started=time.perf_counter()
        while not stop:
            # Tiny head/offset problem: stream finite directional derivatives
            # into its normal system, retaining all original 3D rows.
            J=np.empty((len(res),N));eps=2e-7
            for j in range(N):
                trial=delta.copy();trial[j]+=eps*scale[j]
                vv,ok=evaluate(trial)
                if not ok.all():
                    trial[j]-=2*eps*scale[j];vv,ok=evaluate(trial);sign=-1.
                else:sign=1.
                if not ok.all():raise ValueError('No legal numerical derivative at shared head domain')
                J[:,j]=(((vv-pred)*weight[:,None]).ravel()/(sign*eps))
            normal=J.T@J;gradient=J.T@res;diag=np.maximum(np.diag(normal),1e-20)
            pg=gradient.copy();pg[(delta<=lo+1e-12)&(pg>0)]=0.;pg[(delta>=hi-1e-12)&(pg<0)]=0.
            normalized=float(np.max(abs(pg)/np.sqrt(diag))/(1+np.linalg.norm(res)))
            if normalized<1e-8:stop='projected_scaled_gradient';break
            while True:
                try:step=np.linalg.solve(normal+np.diag(damping*diag),-gradient)
                except np.linalg.LinAlgError:damping*=10.;continue
                trial=np.clip(delta+step*scale,lo,hi);actual=(trial-delta)/scale
                if np.max(abs(actual))<=1e-10*(1+np.max(abs(delta/scale))):stop='no_representable_improving_step';break
                vv,ok=evaluate(trial)
                if not ok.all():damping*=10.;rejected+=1;continue
                rr=((vv-uv)*weight[:,None]).ravel();cc=float(rr@rr/2)
                expected=float(-gradient@actual-.5*actual@(normal@actual))
                ratio=(cost-cc)/expected if expected>0 else -np.inf
                if np.isfinite(cc) and cc<cost and ratio>0:
                    improvement=cost-cc;old=cost
                    delta,pred,res,cost=trial,vv,rr,cc;steps+=1;damping=max(damping*.3,1e-12)
                    print(json.dumps(dict(rig=rid,accepted_step=steps,cost=cost,improvement=improvement)),flush=True)
                    if improvement<=1e-8*(1+old) and ratio>.25:stop='relative_numeric_transfer_cost';
                    break
                damping*=10.;rejected+=1
        for (kind,sid,j),v in zip(refs,delta):
            if kind=='h':hb['values'][j]+=float(v);changed.add(rig['head_block'])
            elif kind=='o':ob['values'][j]+=float(v);changed.add(rig['optics_block'])
            elif kind=='r':bi=model['shots'][sid]['reference_angles_block'];model['blocks'][bi]['values'][j]+=float(v);changed.add(bi)
            else:
                for bi in model['shots'][sid]['control_blocks']:model['blocks'][bi]['values'][2]+=float(v);changed.add(bi)
        # Replay the serialized production consumer at the target's actual row
        # time and then solve its genuine RS root.  Neither domain failure is
        # silently dropped from the report.
        write_json(out/(rid+'_checkpoint.json'),model)
        forward_delta=0.;rs_invalid=0;rs_error=[]
        for sid in sids:
            mm=sn==int(sid[1:]);shot=model['shots'][sid];cam=camera_from_event(as_event_view(model,sid));t=(pp[mm]-shot['time_origin_pts'])/12800.
            ab=model['blocks'][shot['reference_angles_block']]['values']
            cam.R0=rotation(cam.b,ab[1])@cam.R0@rotation(cam.n,ab[0])
            replay=cam.at(X[mm],t+cam.tau*(uv[mm,1]/1079.-.5));forward_delta=max(forward_delta,float(np.max(abs(replay['uv']-pred[mm]))))
            actual=cam.project_rs(X[mm],t);rs_invalid+=int((~actual['valid']).sum());rs_error.extend(np.linalg.norm(actual['uv']-uv[mm],axis=-1).tolist())
        if forward_delta>2e-6:raise ValueError('Serialized production row-time forward differs')
        reports[rid]=dict(started=True,denominator=len(ids),numeric_rowtime_before=_summary(before),numeric_rowtime_after=_summary(pred-uv),cost_before=initial,cost_after=cost,accepted_steps=steps,rejected_trials=rejected,stop_reason=stop,parameters=N,seconds=time.perf_counter()-started,production_rowtime_max_delta_px=forward_delta,true_RS_invalid=rs_invalid,true_RS_full_RMS=None if rs_invalid else float(np.sqrt(np.mean(np.square(rs_error)))),source_accuracy_claimed=False)
    for i,b in enumerate(model['blocks']):
        if i not in changed and b!=parent['blocks'][i]:raise ValueError('An unrelated block changed')
        a=np.asarray(b['values']);
        if not np.isfinite(a).all() or np.any(a<b['lower']) or np.any(a>b['upper']):raise ValueError('Physical bounds violated')
    model['surface_head_capture_v1']=dict(parent_model_sha256=file_sha256(model_file),reference_sha256=file_sha256(reference_file),accepted=False,source_observations_created=0,main_and_S01_unchanged=True,provisional_shared_optics=True)
    write_json(out/'model.json',model)
    report=dict(schema='rinkfit.surface_head_capture.v1',model=dict(path=str(out/'model.json'),sha256=file_sha256(out/'model.json')),reports=reports,scene_count=1,rig_count=3,shot_count=9,source_accuracy_claimed=False,accepted=False,reference_sha256=file_sha256(reference_file),code_sha256=file_sha256(__file__))
    write_json(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(build_candidate(a.model,a.reference,a.out),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
