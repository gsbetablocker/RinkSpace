"""Opt-in complete source block for one fixed XYZ installation.

All original observations and exposure states remain in the source bundle.
Rink numeric parameters are common; three global pivot coordinates per
installation replace an approximate XY that used to be frozen. There is no
per-exposure camera translation or new lens-offset parameter.
"""
from __future__ import annotations
from types import SimpleNamespace
import os
import numpy as np
from scipy.sparse import csr_matrix
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
from z190_joint_rink_dimensions_source import RinkBlock


def scatter_fixed_xyz_source_csr(localj,source_rows,geometry,geometry_row_start,
                                 geometry_indices,xyz_slots,view_start,total_columns):
    """Map the original CSR straight into one final global matrix.

    Local shared column3 is the old zero padding and is omitted exactly as
    before. Original source rows have one such entry; roll priors do not.
    Geometry is the small absolute-source span, with its explicit shared
    parameter indices. No full local column slices or global hstack exist.
    """
    localj=localj.tocsr(copy=False);rows=localj.shape[0]
    source_rows=int(source_rows);geometry_row_start=int(geometry_row_start)
    geometry_indices=np.asarray(geometry_indices,int);xyz_slots=np.asarray(xyz_slots,int)
    geom=csr_matrix(geometry);geo_rows=geom.shape[0]
    if not(0<=source_rows<=rows and 0<=geometry_row_start<=geometry_row_start+geo_rows<=source_rows):
        raise ValueError('Original source/geometry residual row domain changed')
    if geom.shape[1]!=len(geometry_indices)or xyz_slots.shape!=(3,):raise ValueError('Explicit global source columns required')
    if (not np.array_equal(geometry_indices[:5],np.arange(5))or
            any(int(c)in xyz_slots or view_start<=int(c)<view_start+localj.shape[1]-4 for c in geometry_indices[5:])):
        raise ValueError('Geometry columns overlap installation/view coordinates')
    # In this source layout columns 0..3 lead each absolute/motion row.
    source_starts=localj.indptr[:source_rows]
    if (np.any(np.diff(localj.indptr[:source_rows+1])<4)or
            np.any(localj.indices[source_starts+3]!=3)or
            np.any(localj.indices[localj.indptr[source_rows]:]==3)):
        raise ValueError('Original shared-padding/roll-prior CSR layout changed')
    mode=os.environ.get('RINKSPACE_SCATTER_BACKEND',os.environ.get('RINKSPACE_MOTION_BACKEND','auto'))
    if mode not in ('auto','native','numpy'):raise ValueError('Unknown RINKSPACE_SCATTER_BACKEND: '+mode)
    if mode!='numpy':
        from z190_csr_scatter_native import metadata,native_scatter
        compatible=(localj.data.dtype==np.dtype(np.float64)and geom.data.dtype==np.dtype(np.float64)
                    and localj.indices.dtype in (np.dtype(np.int32),np.dtype(np.int64))
                    and localj.indptr.dtype==localj.indices.dtype)
        if (metadata()['available']and compatible)or mode=='native':
            return native_scatter(localj,source_rows,geom,geometry_row_start,geometry_indices,
                xyz_slots,view_start,total_columns,1./(100.*SCENE_UNIT_M))
    pointer=localj.indptr.copy()
    pointer-=np.minimum(np.arange(rows+1,dtype=pointer.dtype),source_rows)
    pointer[geometry_row_start+1:geometry_row_start+geo_rows+1]+=geom.indptr[1:]
    pointer[geometry_row_start+geo_rows+1:]+=geom.nnz
    data=np.empty(int(pointer[-1]),localj.data.dtype)
    columns=np.empty(len(data),localj.indices.dtype)
    head_counts=np.diff(csr_matrix(np.asarray(geometry)[:,:5]).indptr)
    scale=1./(100.*SCENE_UNIT_M)
    for lo in range(0,rows,65536):
        hi=min(lo+65536,rows);left,right=map(int,localj.indptr[[lo,hi]])
        counts=np.diff(localj.indptr[lo:hi+1])
        row_ids=np.repeat(np.arange(lo,hi,dtype=pointer.dtype),counts)
        inner=np.arange(right-left,dtype=pointer.dtype)-np.repeat(localj.indptr[lo:hi]-left,counts)
        original_cols=localj.indices[left:right];keep=original_cols!=3
        row_ids=row_ids[keep];inner=inner[keep];original_cols=original_cols[keep]
        inner-=((row_ids<source_rows)&(original_cols>3))
        positions=pointer[row_ids]+inner
        within=(row_ids>=geometry_row_start)&(row_ids<geometry_row_start+geo_rows)
        positions[within]+=head_counts[row_ids[within]-geometry_row_start]
        values=localj.data[left:right][keep]
        mapped=original_cols.copy();shared=mapped<3
        values[shared]*=scale;mapped[shared]=xyz_slots[mapped[shared]]
        mapped[~shared]+=view_start-4
        data[positions]=values;columns[positions]=mapped
    if geom.nnz:
        geo_row_ids=np.repeat(np.arange(geo_rows,dtype=pointer.dtype),np.diff(geom.indptr))
        global_rows=geometry_row_start+geo_row_ids
        inner=np.arange(geom.nnz,dtype=pointer.dtype)-np.repeat(geom.indptr[:-1],np.diff(geom.indptr))
        positions=pointer[global_rows]+inner
        tail=geom.indices>=5
        local_counts=np.diff(localj.indptr)[global_rows]-1
        positions[tail]+=local_counts[tail]
        data[positions]=geom.data;columns[positions]=geometry_indices[geom.indices]
    matrix=csr_matrix((data,columns,pointer),shape=(rows,total_columns))
    del data,columns,pointer
    if len(geometry_indices)>5:
        # The old '+=' tail merge removed explicit zero entries throughout
        # the matrix. Preserve that storage behaviour for q7/q11 snapshots.
        matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


class FixedXYZRinkBlock:
    def __init__(self,name,prepared,geometry,xyz_slots,start,*,geometry_parameter_count=5,absolute=None):
        self.name,self.geometry=name,geometry
        self.geometry_parameter_indices=np.arange(5,dtype=int)
        self.geometry_parameter_count=int(geometry_parameter_count)
        self.global_parameter_count=self.geometry_parameter_count+9
        self.xyz_slots=np.asarray(xyz_slots,int)
        if self.geometry_parameter_count!=5 or self.xyz_slots.shape!=(3,):
            raise ValueError('Explicit common five-parameter rink and one XYZ slot triplet required')
        if np.any(self.xyz_slots<5)or np.any(self.xyz_slots>=14)or len(set(self.xyz_slots))!=3:
            raise ValueError('Installation XYZ slots must be distinct global coordinates')
        self.start=int(start)
        if self.start<self.global_parameter_count:raise ValueError('Complete views must follow all global parameters')
        oldq=prepared.q.copy();old=prepared.bundle
        self.bundle=FixedLeverXYZSourceBundle.from_bundle(old)
        if absolute is not None:self.bundle.absolute_evaluator=absolute
        self.bundle.calibration=old.current_calibration(oldq)
        self.q0=np.r_[self.bundle.initial_shared(),oldq[4:]]
        self.prepared=SimpleNamespace(**vars(prepared));self.prepared.bundle=self.bundle;self.prepared.q=self.q0
        self.groups=self.bundle.abs_groups;self.absolute=self.bundle.absolute_evaluator
        self.bundle.priors=lambda q,cache:tuple(x[1:]for x in MovingPivotSourceBundle.priors(self.bundle,q,cache))
        self.quadratic=np.r_[np.zeros(self.bundle.motion_rows,bool),np.concatenate([
            np.full(2*len(r['uv']),r.get('curve_observation_kind')in('horizontal_extent','clipped_horizontal_extent')
                or r.get('absolute_support_volume_quadratic')is True,bool)
            for _,rows in self.groups for r in rows]),np.ones(len(self.bundle.stamps),bool)]

    def local(self,q):
        result=self.q0.copy();result[:3]=q[self.xyz_slots]/(100.*SCENE_UNIT_M);result[3]=0.
        result[4:]=q[self.start:self.start+len(result)-4]
        return result

    def set_shared_geometry(self,geometry,parameter_indices):
        indices=np.asarray(parameter_indices,dtype=int).reshape(-1)
        if (len(indices)not in(5,7,11)or not np.array_equal(indices[:5],np.arange(5))or
                len(set(indices.tolist()))!=len(indices)or np.any(indices[5:]<14)):
            raise ValueError('Shared geometry keeps the original five leading columns and explicit new material columns')
        self.geometry=geometry;self.geometry_parameter_indices=indices.copy()
        self.geometry_parameter_count=len(indices)

    def geometry_values(self,q):return q[self.geometry_parameter_indices]

    def evaluate(self,q,jacobian=True):
        local=self.local(q);ng=self.geometry_parameter_count
        geometry_values=self.geometry_values(q)
        self.bundle.abs_groups=[(p,[self.geometry.rebind(r,geometry_values)for r in rows])for p,rows in self.groups]
        try:
            if not jacobian:return self.bundle.evaluate(local,False)
            residual,localj=self.bundle.evaluate(local,True)
            count=sum(2*len(r['uv'])for _,rows in self.groups for r in rows)
            geo=np.zeros((count,ng));cursor=0
            step=float(os.environ.get('RINKSPACE_GEOMETRY_FD_STEP_M','1e-5'))
            if not np.isfinite(step)or step<=0:raise ValueError('Positive finite source geometry FD step required')
            for p,rows in self.groups:
                camera=self.bundle.camera(local,p)
                for row in rows:
                    size=2*len(row['uv'])
                    if self.geometry.feature_id(row)is not None:
                        columns=range(ng)
                        if os.environ.get('RINKSPACE_GEOMETRY_SPARSE_FD','0')=='1':
                            # A declared zero world derivative identifies
                            # parameters absent from this material row. Verify
                            # the actual two FD worlds before omitting calls.
                            dw=self.geometry.world_jacobian(row,geometry_values)
                            candidates=np.flatnonzero(np.any(dw!=0.,axis=(0,1)))
                            inactive=set(range(ng))-set(candidates.tolist())
                            for c in inactive:
                                plus,minus=geometry_values.copy(),geometry_values.copy();plus[c]+=step;minus[c]-=step
                                if not np.array_equal(self.geometry.rebind(row,plus)['world'],self.geometry.rebind(row,minus)['world']):
                                    raise ValueError('Declared absent material column changed the actual FD world')
                            columns=candidates
                        for c in columns:
                            plus,minus=geometry_values.copy(),geometry_values.copy();plus[c]+=step;minus[c]-=step
                            geo[cursor:cursor+size,c]=(self.absolute(camera,[self.geometry.rebind(row,plus)],False)-self.absolute(camera,[self.geometry.rebind(row,minus)],False))/(2*step)
                    cursor+=size
            matrix=scatter_fixed_xyz_source_csr(localj,self.bundle.source_residual_rows,geo,
                self.bundle.motion_rows,self.geometry_parameter_indices,self.xyz_slots,self.start,len(q))
            return residual,matrix
        finally:self.bundle.abs_groups=self.groups

    absolute_records=RinkBlock.absolute_records
