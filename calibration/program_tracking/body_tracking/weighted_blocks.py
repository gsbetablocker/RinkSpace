"""Named, actual weighted residual/Jacobian blocks recorded at assembly sites."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix,vstack,hstack


@dataclass
class WeightedBlock:
    name: str
    residual: np.ndarray
    jacobian: object = None
    provenance: object = None
    association: str = ''


def block(name,r,J=None,provenance=None,association=''):
    return WeightedBlock(name,np.asarray(r).reshape(-1).copy(),J,provenance,association)


def full_columns(b,shared=17,scale=1.):
    J=None if b.jacobian is None else hstack((csr_matrix((len(b.residual),shared)),b.jacobian),format='csr')
    return block(b.name,b.residual*scale,None if J is None else J*scale,b.provenance,b.association)


def signature(*values):
    h=hashlib.sha256()
    for value in values:
        if value is None:continue
        if isinstance(value,str):h.update(value.encode('utf-8'))
        else:
            try:a=np.asarray(value)
            except ValueError:h.update(json.dumps(value,sort_keys=True).encode());continue
            if a.dtype.hasobject:h.update(json.dumps(value,sort_keys=True,default=str).encode())
            else:h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def assemble(blocks,jacobian=False):
    if len({b.name for b in blocks})!=len(blocks):raise ValueError('Duplicate weighted block name')
    r=np.concatenate([b.residual for b in blocks]);J=vstack([b.jacobian for b in blocks],format='csr')if jacobian else None
    return r,J


def verify(blocks,r,J=None):
    start=0;cost=0.;rows=[]
    for b in blocks:
        stop=start+len(b.residual)
        if not np.array_equal(b.residual,r[start:stop]):raise ValueError('Residual block mismatch: '+b.name)
        if J is not None:
            if b.jacobian is None:raise ValueError('Missing actual block Jacobian: '+b.name)
            diff=(J[start:stop]-b.jacobian).tocsr()
            if diff.nnz and np.max(abs(diff.data))>1e-13:raise ValueError('Jacobian block mismatch: '+b.name)
        c=float(b.residual@b.residual/2);cost+=c;rows.append(dict(name=b.name,start=start,stop=stop,rows=stop-start,cost=c,association=b.association))
        start=stop
    if start!=len(r)or not np.isclose(cost,float(r@r/2),rtol=2e-13,atol=1e-10):raise ValueError('Incomplete weighted block registry')
    return dict(rows=len(r),columns=None if J is None else J.shape[1],cost=cost,blocks=rows,exact_residual=True,actual_jacobian_checked=J is not None)
