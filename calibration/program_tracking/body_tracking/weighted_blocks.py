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


def image_cost(parts):
    """The actual weighted image/source blocks used by the quality monitor.

    Scene-derived support remains included for historical monitor continuity;
    reporting separates it from pure source pixels. Episode additions cannot
    silently disappear from this monitor just because they have unique IDs.
    """
    names={'body_pixels','old_gear_pixels','full_source_skate_structure',
        'source_pixel_edge_profile','complete_runner_extent','complete_visible_structure_extent',
        'source_support_location_xy','native_source_support_contact','unassigned_source_skate_sets',
        'old_shoe_support_regions','body_source_regions'}
    return sum(float(value) for name,value in parts.items()
        if name in names or name.startswith('additional_unassigned_source_'))


def block(name,r,J=None,provenance=None,association=''):
    return WeightedBlock(name,np.asarray(r).reshape(-1).copy(),J,provenance,association)


def full_columns(b,shared=17,scale=1.):
    J=None
    if b.jacobian is not None:
        source=b.jacobian.tocsr()
        J=csr_matrix((source.data if scale==1. else source.data*scale,source.indices+shared,source.indptr),shape=(source.shape[0],source.shape[1]+shared),copy=False)
    return block(b.name,b.residual*scale,J,b.provenance,b.association)


def row_view(J,start,stop):
    """A CSR row view sharing the final data/indices, not another matrix copy."""
    lo,hi=map(int,J.indptr[[start,stop]])
    result=csr_matrix((stop-start,J.shape[1]),dtype=J.dtype)
    result.data=J.data[lo:hi];result.indices=J.indices[lo:hi];result.indptr=J.indptr[start:stop+1]-lo
    return result


def pack_weighted(blocks,weights,jacobian=False):
    """Write the final weighted CSR once; named blocks share its storage."""
    counts=[len(b.residual) for b in blocks];offsets=np.r_[0,np.cumsum(counts)]
    r=np.empty(int(offsets[-1]),float);J=None
    if jacobian:
        columns=blocks[0].jacobian.shape[1];nnz=sum(b.jacobian.nnz for b in blocks)
        dtype=np.int32 if max(columns,nnz,int(offsets[-1]))<2**31 else np.int64
        data=np.empty(nnz,float);indices=np.empty(nnz,dtype);ptr=np.empty(len(r)+1,dtype);ptr[0]=0;at=0
    for b,a,z in zip(blocks,offsets[:-1],offsets[1:]):
        w=np.broadcast_to(np.asarray(weights.get(b.name,1.),float),b.residual.shape)
        if not np.isfinite(w).all()or np.any(w<=0):raise ValueError('Objective requires finite positive declared weights')
        factor=np.sqrt(w);r[a:z]=b.residual*factor
        if jacobian:
            source=b.jacobian.tocsr();stop=at+source.nnz;indices[at:stop]=source.indices
            ptr[a:z+1]=source.indptr+at
            for first in range(0,len(factor),4096):
                last=min(first+4096,len(factor));lo,hi=map(int,source.indptr[[first,last]])
                data[at+lo:at+hi]=source.data[lo:hi]*np.repeat(factor[first:last],np.diff(source.indptr[first:last+1]))
            at=stop
    if jacobian:J=csr_matrix((data,indices,ptr),shape=(len(r),columns),copy=False)
    named=[WeightedBlock(b.name,r[a:z],None if J is None else row_view(J,int(a),int(z)),b.provenance,b.association)for b,a,z in zip(blocks,offsets[:-1],offsets[1:])]
    return r,J,named


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
        if not np.isfinite(b.residual).all():raise ArithmeticError('Nonfinite weighted residual: '+b.name)
        if not np.array_equal(b.residual,r[start:stop]):raise ValueError('Residual block mismatch: '+b.name)
        if J is not None:
            if b.jacobian is None:raise ValueError('Missing actual block Jacobian: '+b.name)
            if not np.isfinite(b.jacobian.data).all():raise ArithmeticError('Nonfinite actual weighted Jacobian: '+b.name)
            for a in range(start,stop,4096):
                z=min(a+4096,stop);diff=(J[a:z]-b.jacobian[a-start:z-start]).tocsr()
                if diff.nnz and np.max(abs(diff.data))>1e-13:raise ValueError('Jacobian block mismatch: '+b.name)
        c=float(b.residual@b.residual/2);cost+=c;rows.append(dict(name=b.name,start=start,stop=stop,rows=stop-start,cost=c,association=b.association))
        start=stop
    if start!=len(r)or not np.isclose(cost,float(r@r/2),rtol=2e-13,atol=1e-10):raise ValueError('Incomplete weighted block registry')
    return dict(rows=len(r),columns=None if J is None else J.shape[1],cost=cost,blocks=rows,exact_residual=True,actual_jacobian_checked=J is not None)
