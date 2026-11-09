"""Compare real native outline sensitivity in attached and free diagnostic subspaces."""
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
from scipy.optimize import lsq_linear

from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.fullbody_fit import skew
from calibration.program_tracking.body_tracking.ordered_native_skate import (
    OrderedNativeSkateFactors, SemanticSkateMesh, edge_terms, measured_chord_fraction)
from calibration.program_tracking.body_tracking.programme_skate_replay import write
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_binding import SHAPE_NAMES
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


def raw_outline(f,values):
    packed=[]
    for i,(r,match) in enumerate(zip(f.rows,f.matches,strict=True)):
        source=np.asarray(r['polyline_uv'])
        chord=measured_chord_fraction(r,source)
        _,_,raw,D=edge_terms(values[i,:len(match.vertices)],source,match,r['radius_px'],1.,
                              r['complete_visible'],r.get('structural_endpoints',[]),chord_fraction=chord)
        packed.append((raw[:len(source)].ravel(),D[:2*len(source)]))
    return packed


def project(camera,f,nodes,*,derivative=False):
    return moving_projection(camera,nodes,f.frames,jacobian=derivative)


def get_depth(f,s):
    feet=np.repeat(f.feet,5)
    O=s['O'][np.arange(len(feet)),feet]
    R=s['R'][np.arange(len(feet)),feet]
    return O,R,feet


def shape_nodes(binding,f,s,ids):
    mesh=SemanticSkateMesh(binding)
    O,R,feet=get_depth(f,s)
    nominal=f.mesh
    map_id={}
    if nominal.upper_end-nominal.upper_offset!=mesh.upper_end-mesh.upper_offset:
        raise ValueError('Shared geometry changed upper material station identities')
    map_id.update(zip(range(nominal.upper_offset,nominal.upper_end),
                      range(mesh.upper_offset,mesh.upper_end),strict=True))
    for k in nominal.runner:
        for source,target in zip(nominal.runner[k],mesh.runner[k],strict=True):
            map_id.update(zip(source,target,strict=True))
    physical_ids=np.vectorize(map_id.__getitem__,otypes=[int])(ids)
    verts=mesh.local(s,feet,np.repeat(physical_ids,5,axis=0))
    return (O[:,None]+np.einsum('nij,nkj->nki',R,verts)).reshape(len(f.rows),5,-1,3)


def pixels_error(v):
    d=np.linalg.norm(v.reshape(-1,2),axis=-1)
    return dict(median_px=float(np.median(d)),p95_px=float(np.percentile(d,95)),
                rms_px=float(np.sqrt(np.mean(d*d))))


def linear_step(A,b,maxima,damping=.20):
    W=A*maxima
    d=lsq_linear(np.vstack((W,damping*np.eye(len(maxima)))),
                 np.r_[-b,np.zeros(len(maxima))],bounds=(-1.,1.),lsmr_tol='auto').x
    return d*maxima


def diagnose(f,x):
    P=f.problem
    d=P.body.state.increments(x,f.times)
    s=fk_state(P.reference,f.base,d)
    widths=[len(m.vertices) for m in f.matches]
    ids=np.array([np.pad(m.vertices,(0,max(widths)-n),mode='edge') for n,m in zip(widths,f.matches)],int)
    W=f._world(s,ids)
    out=project(P.camera,f,W,derivative=True)
    J=out['pixel_jac_nodes']
    np.testing.assert_allclose(W,shape_nodes(P.binding,f,s,ids),rtol=0,atol=1.e-10)
    R=[];B=[];S=[]
    O,_,feet=get_depth(f,s)
    o=O.reshape(len(f.rows),5,1,3)
    T=J.sum(axis=3)/P.binding.ratio
    U=np.einsum('nkpac,nakcd->nkpd',J,-skew(W-o))
    for axis in (6,7,8,15,16,17,24,25,26):
        plus=d.copy();minus=d.copy()
        for k,q in zip((1,-1),(plus,minus)):
            q[:,axis+3*feet]+=k*1e-5
        dv=(f._world(fk_state(P.reference,f.base,plus),ids)-
            f._world(fk_state(P.reference,f.base,minus),ids))/(2e-5)
        B.append(np.einsum('nkpac,nakc->nkp',J,dv))
    for a in range(12,17):
        variants=[]
        for sign in (1,-1):
            binding=copy(P.binding);binding.x=binding.x.copy();binding.x[a]+=sign*1e-4
            variants.append(shape_nodes(binding,f,s,ids))
        S.append(np.einsum('nkpac,nakc->nkp',J,(variants[0]-variants[1])/(2e-4)))
    n=len(f.rows)
    R=np.concatenate((T,U),axis=-1)
    B=np.stack(B,axis=-1)
    S=np.stack(S,axis=-1)
    groups=defaultdict(list)
    raw=raw_outline(f,out['uv'])
    for i,(r,(b,D),k) in enumerate(zip(f.rows,raw,widths,strict=True)):
        mats=[D@Z[i,:k].reshape(-1,Z.shape[-1]) for Z in (R,B,S)]
        groups[r['pts'],r['foot']].append((i,b,*mats))
    xshape=linear_step(np.vstack([z[-1] for g in groups.values() for z in g]),
                       np.concatenate([z[1] for g in groups.values() for z in g]),np.full(5,.10),1.)
    shape_results=[]
    for fraction in (0.,.25,.50,1.):
        binding=copy(P.binding);binding.x=binding.x.copy();binding.x[12:]+=fraction*xshape
        val=raw_outline(f,project(P.camera,f,shape_nodes(binding,f,s,ids))['uv'])
        shape_results.append((sum(r@r for r,_ in val),fraction,val))
    _,shape_fraction,changed_shape=min(shape_results,key=lambda z:z[0])
    xshape*=shape_fraction
    steps={k:np.zeros((n,width)) for k,width in [('ankle_only',3),('linked_leg',9),('free_rigid',6)]}
    result=[]
    for (pts,foot),g in groups.items():
        ix,b,R,B,S=zip(*g,strict=True)
        b=np.concatenate(b);R=np.vstack(R);B=np.vstack(B);S=np.vstack(S)
        a=linear_step(B[:,6:],b,np.full(3,.45),1.0)
        v=linear_step(B,b,np.full(9,.36),1.5)
        u=linear_step(R,b,np.r_[np.full(3,.10),np.full(3,.50)],1.5)
        steps['ankle_only'][list(ix)]=a
        steps['linked_leg'][list(ix)]=v
        steps['free_rigid'][list(ix)]=u
        result.append(dict(pts=int(pts),foot=int(foot),semantics=[f.rows[i]['semantic'] for i in ix],
                           row_indices=list(ix),raw=pixels_error(b),
                           linear_ankle_only=pixels_error(b+B[:,6:]@a),
                           linear_linked_leg=pixels_error(b+B@v),linear_free_rigid=pixels_error(b+R@u),
                           linked_leg_deg=np.rad2deg(v.reshape(3,3)).tolist(),
                           free_rigid_shift_m=u[:3].tolist(),free_rigid_rotation_deg=np.rad2deg(u[3:]).tolist(),
                           normalized_rigid_singular=np.linalg.svd(R*np.r_[np.full(3,.12),np.full(3,.65)],compute_uv=False).tolist(),
                           shape=pixels_error(np.concatenate([changed_shape[i][0] for i in ix]))))
    changed={}
    for name,initial in steps.items():
        trials=[]
        for fraction in (0.,.125,.25,.5,.75,1.):
            step=initial*fraction
            if name=='free_rigid':
                v=np.repeat(step[:,3:],5,axis=0).reshape(n,5,3)
                trans=np.repeat(step[:,:3],5,axis=0).reshape(n,5,1,3)/P.binding.ratio
                nodes=o+np.einsum('naij,nakj->naki',_exp(v),W-o)+trans
            else:
                update=d.copy()
                axes=np.array([24,25,26] if name=='ankle_only' else [6,7,8,15,16,17,24,25,26])
                select=axes[None,:]+3*feet[:,None]
                update[np.arange(len(feet))[:,None],select]+=np.repeat(step,5,axis=0)
                nodes=f._world(fk_state(P.reference,f.base,update),ids)
            trials.append((fraction,raw_outline(f,project(P.camera,f,nodes)['uv'])))
        changed[name]=trials
    for g in result:
        ix=g.pop('row_indices')
        for k,trials in changed.items():
            values=[(f,pixels_error(np.concatenate([rows[i][0] for i in ix]))) for f,rows in trials]
            fraction,z=min(values,key=lambda q:q[1]['rms_px'])
            g[k]=dict(z,chosen_step_fraction=fraction)
    return dict(static_shape_log_step=dict(zip(SHAPE_NAMES,xshape.tolist(),strict=True)),groups=result,
                role=f.rows[0]['role'],fit_or_replay_accepted=False,
                diagnostic='one bounded linear step followed by the exact projection on the same visible ordered path')


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','parameter','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    r,c,info,e,_=load_reference(a.project,a.prepared,a.source)
    state=load(a.parameter)
    r.absolute_curve=curve_from_packet(state,seed=True)
    r.relative_skate_curve=shoe_curve_from_packet(state,seed=True)
    ss=SequenceState(r,tuple(r.times[[0,-1]]))
    problem=SimpleNamespace(body=SimpleNamespace(state=ss),reference=r,binding=r.binding,camera=c,size=len(state['x']))
    archive={q['id']:q for q in __import__('json').loads(str(state['ordered_source_association_json']))}
    direct=[q for q in e['programme_skate_evidence']['ordered_contours'] if q['role']=='fit'
            and q.get('provenance',{}).get('kind')=='direct source visual read' and 79.02<q['pts']/12800<=95.0]
    f=OrderedNativeSkateFactors(problem,direct)
    f.restore([archive[q['raw_group_id']] for q in direct])
    write(a.output,diagnose(f,state['x'][17:]))


if __name__=='__main__':
    main()
