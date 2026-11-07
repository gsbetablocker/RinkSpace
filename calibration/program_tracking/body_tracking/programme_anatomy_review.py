"""Mesh-level checks at actual whole-curve collision candidates.

The fit uses inexpensive envelopes. This review checks where the posed
skin and deformed attached equipment surfaces actually cross. Open part
boundaries and coplanar contact do not supply a penetration depth.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from .absolute_motion_state import absolute_reference, curve_from_packet
from .fullbody_round import write_json
from .programme_anatomy import anatomy_geometry
from .rebuild_prepare import digest, load
from .sequence_skate_state import shoe_curve_from_packet
from .skate_physics import fk_state
from .trajectory_backend import load_reference


def segment_triangle_cross(a,b,tri):
    """Interior segment crossing, excluding a shared end and coplanar touch."""
    e1=tri[:,1]-tri[:,0]
    e2=tri[:,2]-tri[:,0]
    ray=b-a
    h=np.cross(ray,e2)
    den=np.sum(e1*h,axis=1)
    valid=np.abs(den)>1e-14
    inverse=np.divide(1.,den,out=np.zeros_like(den),where=valid)
    s=a-tri[:,0]
    u=np.sum(s*h,axis=1)*inverse
    q=np.cross(s,e1)
    v=np.sum(ray*q,axis=1)*inverse
    t=np.sum(e2*q,axis=1)*inverse
    return valid&(u>=0.)&(v>=0.)&(u+v<=1.)&(t>1e-9)&(t<1.-1e-9)


def triangle_crossings(A,B,block_size=64):
    if not len(A) or not len(B):
        return 0
    loB,hiB=B.min(axis=1),B.max(axis=1)
    count=0
    for first in range(0,len(A),block_size):
        X=A[first:first+block_size]
        loA,hiA=X.min(axis=1),X.max(axis=1)
        left,right=np.nonzero(np.all((loA[:,None]<=hiB)&(loB<=hiA[:,None]),axis=-1))
        if not len(left):
            continue
        a,b=X[left],B[right]
        hit=np.zeros(len(a),bool)
        for j in range(3):
            hit |= segment_triangle_cross(a[:,j],a[:,(j+1)%3],b)
            hit |= segment_triangle_cross(b[:,j],b[:,(j+1)%3],a)
        count+=int(hit.sum())
    return count


def parts_for_sample(ref,state,capsules,boxes,labels,wanted):
    ratio=ref.binding.ratio
    skin={}
    if any(name in capsules for name in wanted):
        surface=ref.binding.surface
        sample=surface.sample_vertices(state['native'],state['frames'],
                                       hand_local_R=state.get('hand_local_R'))[0]
        xyz=(sample@state['Q'][0].T+state['P'][0])*ratio
        for name in wanted:
            if name in capsules:
                selected=np.isin(labels,capsules[name].skin_group)
                faces=surface.faces[np.all(selected[surface.faces],axis=1)]
                skin[name]=xyz[faces]
    gear={}
    if any(name in boxes for name in wanted):
        alpha=ref.binding.deformation(state['native'],state['frames'],
                                      skate_relative_R=state['skate_relative_R'])
        packet=ref.binding.packet()
        uppers=ref.binding.deform(packet['upper'],alpha)[0]
        n_upper=len(ref.binding.faces)
        for name in wanted:
            if name not in boxes:
                continue
            model=boxes[name]
            triangles=model.triangles
            if name.endswith('_boot'):
                triangles=np.concatenate((triangles[:-n_upper],
                    uppers[model.foot][ref.binding.faces]*ratio))
            gear[name]=triangles@state['R'][0,model.foot].T+state['O'][0,model.foot]*ratio
    return dict(skin,**gear)


def review(project,prepared,parameters,physics_report,output):
    doc=json.loads(Path(physics_report).read_text(encoding='utf8'))
    suspects=doc['source_and_physics']['self_collision']['pairs']
    root,camera,info,source,seed=load_reference(project,prepared)
    origin=float(root.p['pts'][0]/12800.-root.times[0])
    selected={}
    for entry in suspects:
        if entry['positive_samples']:
            t=entry['sample_seconds']
            selected.setdefault(t,set()).add(tuple(entry['pair']))
    # Explicit original centreline failures get same-clock before/after checks
    # even where the fitted envelope has already cleared.
    for t in (49.96,97.82,104.56,117.74):
        selected.setdefault(t,set()).add(('left_shank','right_shank'))
    reports={}
    for name,param in parameters:
        saved=load(param)
        ref=absolute_reference(root,curve_from_packet(saved),root=Path(prepared))
        ref.relative_skate_curve=shoe_curve_from_packet(saved)
        import copy
        ref.binding=copy.copy(root.binding)
        ref.binding.reference=ref
        ref.binding.x=saved['x'][:17].copy()
        ref.binding._parameter_key=ref.binding._packet=None
        capsules,boxes,labels=anatomy_geometry(ref)
        rows=[]
        for seconds,keys in sorted(selected.items()):
            time=np.array([seconds-origin])
            state=fk_state(ref,ref.at(time),np.zeros((1,69)))
            parts=parts_for_sample(ref,state,capsules,boxes,labels,
                                   set(p for pair in keys for p in pair))
            near=int(np.argmin(abs(ref.times-time[0])))
            for a,b in sorted(keys):
                n=triangle_crossings(parts[a],parts[b])
                rows.append(dict(seconds=seconds,nearest_native_pts=int(ref.p['pts'][near]),
                    native_exposure=bool(abs(ref.times[near]-time[0])<1e-10),pair=[a,b],
                    intersecting_triangle_pairs=n,triangle_counts=[len(parts[a]),len(parts[b])]))
        reports[name]=dict(parameter=str(param),parameter_sha256=digest(param),samples=rows,
                           confirmed_model_crossing_checks=sum(r['intersecting_triangle_pairs']>0 for r in rows))
    write_json(output,dict(evaluated=reports,source_alignment_performed=False,
        scope='Worst positive fitted envelope instant per pair and the four original shank candidates, same instants for all variants',
        semantics='Noncoplanar surface crossings in the saved SMPL-X and bound skate mesh; does not measure skin penetration depth or guarantee absence of other intersections'))
    return reports


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('project','prepared','physics-report','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--candidate',action='append',required=True,help='ID=parameter.npz')
    a=ap.parse_args()
    rows=[(k,Path(p)) for k,p in (item.split('=',1) for item in a.candidate)]
    results=review(a.project,a.prepared,rows,a.physics_report,a.output)
    print({k:dict(tested=len(v['samples']),crossings=v['confirmed_model_crossing_checks']) for k,v in results.items()})


if __name__=='__main__':
    main()
