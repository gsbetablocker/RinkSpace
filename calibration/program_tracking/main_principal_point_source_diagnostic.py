"""Diagnose a shared two-parameter principal point on actual main source rays.

One C and one principal point belong to the same fixed camera installation.
Only R and one focal length vary per actual exposure. No scene or ad geometry
changes, per-frame distortion, or additional source fitting rows are introduced.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.sparse import csr_matrix
from main_installation_source_bundle_candidate import prepare
from main7_source_bundle_candidate import Bundle,_relative_kernel,solve
from main7_source_camera_candidate import arrays,pinhole,sha,errors,value_only
from projection import project_fast
from ice_refine import _curve_vector_error


def absolute_value_jac(camera,rows):
    values=[];jac=[]
    for row in rows:
        if row.get('point_correspondence'):
            projected,depth,j=project_fast(camera,row['world'],with_jacobian=True);r=(projected-row['uv']).ravel()
        else:r,j=_curve_vector_error(camera,row['world'],row['uv'],with_jacobian=True)
        if 'normal_interval' in row:
            from source_edge_intervals import SourceNormalIntervals
            support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(row['uv']),bool))
            r,matrices=support.apply(r,[j]);j=matrices[0]
        scale=np.repeat(row['sigma'],2);values.append(r/scale)
        jac.append(np.c_[j[:,:3],j[:,8:10],j[:,3:6],j[:,6]+j[:,7]]/scale[:,None])
    return np.concatenate(values),np.vstack(jac)


class PrincipalBundle(Bundle):
    def __init__(self,base):
        self.__dict__.update(base.__dict__);self.view_offset=5
        blocks=[];counts=[]
        motion=base.indices[:base.motion_rows*8].reshape(-1,8)+2
        blocks.append(np.c_[np.broadcast_to([3,4],(base.motion_rows,2)),motion].astype(np.int32).ravel())
        counts.append(np.full(base.motion_rows,10,dtype=np.int32))
        for p,rows in base.abs_groups:
            count=2*sum(len(r['uv']) for r in rows);cols=np.r_[np.arange(5),5+4*self.index[p]+np.arange(4)]
            blocks.append(np.broadcast_to(cols,(count,9)).astype(np.int32).ravel());counts.append(np.full(count,9,dtype=np.int32))
        self.indices=np.concatenate(blocks);self.indptr=np.r_[0,np.cumsum(np.concatenate(counts),dtype=np.int64)].astype(np.int32)

    def camera(self,q,p):
        c=pinhole(np.r_[q[:3],q[5+4*self.index[p]:9+4*self.index[p]]]);c[8:10]=q[3:5]*1000.;return c

    def evaluate(self,q,jacobian=True):
        views=q[5:].reshape(-1,4);principal=q[3:5]*1000.
        rotations=[cv2.Rodrigues(x[:3]) for x in views]
        rot=np.asarray([v[0] for v in rotations]);drot=np.asarray([v[1].reshape(3,3,3) for v in rotations]) if jacobian else None
        focal=np.exp(views[:,3]);values=[];data=[]
        for lo in range(0,len(self.a),65536):
            hi=min(lo+65536,len(self.a));ia=self.ia[lo:hi];ib=self.ib[lo:hi];scale=self.motion_sigma[2*lo:2*hi]
            result=_relative_kernel(focal[ia],focal[ib],rot[ia],rot[ib],None if drot is None else drot[ia],
                None if drot is None else drot[ib],self.a[lo:hi],self.b[lo:hi],jacobian=jacobian,
                principal=principal,principal_jacobian=jacobian)
            if jacobian:
                r,ja,jb,jp=result;block=np.c_[jp,ja,jb];swap=self.swap_rows[2*lo:2*hi]
                block[swap]=np.c_[jp[swap],jb[swap],ja[swap]];data.append((block/scale[:,None]).ravel())
            else:r=result
            values.append(r/scale)
        for p,rows in self.abs_groups:
            if jacobian:v,j=absolute_value_jac(self.camera(q,p),rows);data.append(j.ravel())
            else:v=value_only(self.camera(q,p),rows)
            values.append(v)
        r=np.concatenate(values)
        if not jacobian:return r
        return r,csr_matrix((np.concatenate(data),self.indices,self.indptr),shape=(self.residual_rows,len(q)))


def check_real_derivative(base,q0,output):
    # Both directions and shared PP derivatives use genuine source matches;
    # take a tiny disconnected subgraph so finite differences remain cheap.
    pairs=[]
    for lo in (0,min(len(base.a)-20,10000)):
        hi=lo+20;pa=base.stamps[int(base.ia[lo])];pb=base.stamps[int(base.ib[lo])]
        take=(base.ia[lo:hi]==base.ia[lo])&(base.ib[lo:hi]==base.ib[lo])
        pairs.append((pa,pb,base.a[lo:hi][take],base.b[lo:hi][take]))
    pa,pb,a,b=pairs[-1];pairs.append((pb,pa,b,a));stamps=sorted(set(p for pair in pairs for p in pair[:2]))
    absolute={p:base.observations.get(p,[])[:1] for p in stamps};small=PrincipalBundle(Bundle(stamps,absolute,pairs))
    q=np.r_[q0[:5],np.concatenate([q0[5+4*base.index[p]:9+4*base.index[p]] for p in stamps])]
    r,j=small.evaluate(q);h=1e-7;fd=np.array([(small.evaluate(q+np.eye(len(q))[k]*h,False)-small.evaluate(q-np.eye(len(q))[k]*h,False))/(2*h) for k in range(len(q))]).T
    report=dict(real_source_pairs=[[int(v[0]),int(v[1])] for v in pairs],residual_rows=len(r),
        maximum_difference=float(np.max(np.abs(j.toarray()-fd))),relative_difference=float(np.linalg.norm(j.toarray()-fd)/np.linalg.norm(fd)))
    (output/'shared_principal_real_source_derivative_check.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    if report['relative_difference']>1e-6:raise ValueError('Shared principal source derivative failed numerical check')
    return report


def run(base,previous,centred,output):
    output.mkdir(parents=True,exist_ok=True);scene,streams,canonical,cuts,rows,dense,bindings,suppressed,old,q0=prepare(base,previous)
    seed=arrays(centred/'all_main_candidate_cameras.npz');initial={canonical[int(p)]:c for p,c in zip(seed['pts'],seed['camera25'])}
    q=np.r_[seed['camera25'][0,:3]/100.,np.array([960.,540.])/1000.,np.concatenate([np.r_[initial[p][3:6],initial[p][6]] for p in old.stamps])]
    bundle=PrincipalBundle(old);del old;check=check_real_derivative(bundle,q,output)
    print(json.dumps(dict(stage='principal_derivative_checked',**check)),flush=True)
    q,history,seconds=solve(bundle,q,output);cameras={p:bundle.camera(q,p) for p in bundle.stamps}
    for n,s in streams.items():
        np.savez_compressed(output/f'main{n}_candidate_cameras.npz',pts=s['pts'],camera25=np.array([cameras[canonical[int(p)]] for p in s['pts']]),
            exposure_representative_pts=s['exposure_representative_pts'],status=np.full(len(s['pts']),'shared_principal_candidate_unvalidated'))
    pts=np.sort(np.concatenate([s['pts'] for s in streams.values()]));np.savez_compressed(output/'all_main_candidate_cameras.npz',pts=pts,
        camera25=np.array([cameras[canonical[int(p)]] for p in pts]),exposure_representative_pts=np.array([canonical[int(p)] for p in pts]))
    report=dict(schema='rinkspace.shared-principal-source-diagnostic.v1',accuracy_accepted=False,shared_C=(q[:3]*100).tolist(),
        shared_principal=(q[3:5]*1000).tolist(),seconds=seconds,history=history,derivative_check=check,
        source_bindings=bindings,centred_initial_report=str(centred/'report.json'),centred_initial_report_sha256=sha(centred/'report.json'),
        physical_parameters='one shared C and principal point; one R/f per actual exposure; no Brown/skew; square pixels',
        source_absolute_errors=[dict(native_pts=r['pts'],cut=cuts[canonical[r['pts']]],id=r['id'],split=r['split'],origin=r['origin'],
        errors=errors(cameras[canonical[r['pts']]],[r])) for r in rows])
    holdout=base/'pelvis/main4_geometry/main4_native_lane_holdout_rows.json';held=json.loads(holdout.read_text(encoding='utf8'))
    for r in held:
        for k in ('world','uv','sigma'):r[k]=np.asarray(r[k],float)
    report['independent_main4_lane_holdout']=dict(path=str(holdout),sha256=sha(holdout),errors=[dict(pts=r['pts'],id=r['id'],errors=errors(cameras[canonical[r['pts']]],[r])) for r in held])
    file=output/'report.json';file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'main_camera_centre.json').write_text(json.dumps(dict(C=report['shared_C'],principal=report['shared_principal'],conditional_scene_units=True,
        accuracy_accepted=False,report=str(file),report_sha256=sha(file)),indent=2),encoding='utf8')
    print(json.dumps(dict(stage='complete',shared_C=report['shared_C'],principal=report['shared_principal'],seconds=seconds,accuracy_accepted=False)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('base','previous','centred','output'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();run(a.base,a.previous,a.centred,a.output)
