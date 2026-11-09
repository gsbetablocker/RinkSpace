"""Archive the real restricted programme Jacobian for controlled source-pressure trials."""
from argparse import ArgumentParser
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import vstack
from scipy.sparse.linalg import lsmr

from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay, save, write
from .solve_tracked_programme import protected, scopes, IMAGE

LO, HI = 79.02, 95.0


def local_columns(r):
    n=r.problem.body.state.body_size
    cols=np.r_[np.arange(17,17+n).reshape(-1,69)[:,np.r_[0:3,6:12,15:21,24:30]].ravel(),
               np.arange(17+n,len(r.x))]
    lo,hi,_=scopes(r,cols)
    take=(lo<HI)&(hi>LO)
    for a,b in protected(r):
        take&=(lo>b)|(hi<a)
    return cols[take]


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.previous=a.latest
    a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(a,log)
        before=r.full(r.x,'parent_complete_native_jacobian',jacobian=True)
        blocks=before['value']['weighted_blocks']
        cols=local_columns(r)
        pieces=[]
        values=[]
        labels=[]
        source_role=[]
        meta=[]
        for b in blocks:
            j=b.jacobian[:,cols].tocsr()
            count=np.diff(j.indptr)
            select=count>0
            if not select.any():
                continue
            # Record the exact owner of native contour rows independently of
            # whether a residual lies in its band and has zero gradient.
            source=np.zeros(len(b.residual),bool)
            if b.name=='ordered_native_skate':
                row_names=[]
                pos=0
                for row,J,res in r.ordered.task_jacobians:
                    stop=pos+len(res)
                    use=(LO<row['pts']/12800<=HI and row['semantic'] in
                         ('runner_lower_edge','runner_upper_edge','outsole','front_mount','rear_mount','boot_shell'))
                    source[pos:stop]=use
                    row_names.append(dict(id=row['raw_group_id'],start=pos,stop=stop,foot=row['foot'],
                                          pts=row['pts'],semantic=row['semantic'],active=bool(use)))
                    pos=stop
                if pos!=len(b.residual):
                    raise ValueError('Contour residual ownership changed')
                write(a.output/'contour_rows.json',row_names)
            meta.append(dict(name=b.name,index=len(meta),rows=int(select.sum()),
                cost=float(.5*np.dot(b.residual[select],b.residual[select])),
                unweighted_count=len(b.residual),is_image=b.name in IMAGE))
            pieces.append(j[select])
            values.append(b.residual[select])
            labels.append(np.full(select.sum(),len(meta)-1,np.int32))
            source_role.append(source[select])
        J=vstack(pieces,format='csr')
        save(a.output/'restricted_jacobian.npz',x=r.x,columns=cols,data=J.data,indices=J.indices,indptr=J.indptr,
             shape=np.array(J.shape),residual=np.concatenate(values),labels=np.concatenate(labels),
             target=np.concatenate(source_role),body_size=np.array(r.problem.body.state.body_size))
        write(a.output/'block_index.json',meta)
        save(a.output/'parent_anatomy_and_cavity.npz',anatomy=before['anatomy'],cavity=before['cavity'])
        write(a.output/'cost.json',dict(total=before['cost'],weighted=before['weighted'],coverage=before['coverage']))
        r.emit('source_experiment_ready',columns=len(cols),rows=J.shape[0],nonzeros=J.nnz,
               body_size=r.problem.body.state.body_size,log=str(a.output/'progress.jsonl'))


if __name__=='__main__':
    main()
