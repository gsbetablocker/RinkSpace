"""Opt-in exact complete-physical-record derivative patches.

The supplied complete current residual and original row inventory remain the
objective. This helper changes only requested zero-J rows in a declared
modified-Newton matrix. Motion rows retain complete-family fallback: native
Brown inversion has an original batch-wide stopping condition.
"""
from dataclasses import dataclass
import copy
import numpy as np
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock, _checked_block

@dataclass(frozen=True)
class SourceJacobianRowPatch:
    row_indices: np.ndarray
    residual: np.ndarray
    matrix: csr_matrix
    inventory_sha256: str
    source_row_count: int
    detail: dict

def _physical_record_selection(groups, motion_rows, requested, source_rows):
    """Keep every UV of each touched original record, in original order."""
    requested=np.asarray(requested,dtype=np.int64)
    if requested.ndim!=1 or len(requested)==0 or np.any(np.diff(requested)<=0):
        raise ValueError('Strictly ordered unique activated source scalar rows required')
    if np.any(requested<motion_rows) or np.any(requested>=source_rows):return None
    output=[];indices=[];cursor=int(motion_rows);count=0
    for pts,rows in groups:
        chosen=[]
        for record in rows:
            stop=cursor+2*len(record['uv'])
            if np.any((requested>=cursor)&(requested<stop)):
                chosen.append(record);indices.extend(range(cursor,stop));count+=1
            cursor=stop
        if chosen:output.append((pts,chosen))
    if cursor!=source_rows:raise ValueError('Original complete physical absolute source spans changed')
    if not np.isin(requested,indices).all():raise ValueError('Activated row has no original physical absolute record')
    return output,np.asarray(indices,dtype=np.int64),count

def programme_absolute_rows(owner,q,requested):
    """Run the original StructuredProgrammeBlock._evaluate on original records.

    All absolute camera, mount, optics and logical world derivatives use the
    existing public code and original FD widths/stencils. No motion/profile
    rows or priors enter this derivative-only view; the complete supplied
    objective is separately retained by the caller.
    """
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from z190_moving_pivot_source_bundle import motion_source_csr_layout
    if type(owner)is not StructuredProgrammeBlock:return None
    b=owner.bundle
    choice=_physical_record_selection(owner.groups,b.motion_rows,requested,b.source_residual_rows)
    if choice is None:return None
    groups,original_rows,physical_count=choice
    view=copy.copy(owner);view.bundle=copy.copy(b);small=view.bundle
    view.groups=groups;small.abs_groups=groups
    small.a=b.a[:0];small.b=b.b[:0];small.ia=b.ia[:0];small.ib=b.ib[:0]
    small.motion_sigma=b.motion_sigma[:0];small.motion_rows=0
    small.profile_weights=None if b.profile_weights is None else b.profile_weights[:0]
    small.finite_planes=b.finite_planes[:0];small.finite_mask=b.finite_mask[:0];small.finite_records=[]
    small.source_residual_rows=len(original_rows)
    small.native_residual_sigma=b.native_residual_sigma[original_rows]
    small.indices,small.pointer,small.swap_rows=motion_source_csr_layout(small.ia,small.ib,small.index,groups)
    small.priors=lambda q,cache:(np.empty(0),None)
    small.scene=dict(b.scene,ice_boundary=dict(b.scene['ice_boundary']),board_surface=dict(b.scene['board_surface']))
    view.course_board_indices=np.empty(0,dtype=int);view.course_board_record_positions=np.empty(0,dtype=int)
    view.course_original_board_records=[];view.course_original_board_planes=np.empty((0,4))
    view.course_current_plane_values=np.empty((0,4));view.course_current_plane_jacobian=np.empty((0,4,6))
    view.course_current_motion_geometry_logical_columns=np.arange(44,50)
    view.course_current_motion_geometry_parameter_indices=view.geometry_parameter_indices[44:50]
    view._course_current_geometry_key=None;view._course_camera_cache_key=None;view._course_camera_cache=None
    view.motion_columns_provider=lambda *args,**kwargs:np.empty((0,3))
    residual,matrix=view._evaluate(q,True)
    if len(residual)!=len(original_rows) or matrix.shape!=(len(original_rows),len(q)):
        raise ValueError('Public absolute derivative subset changed source dimensions')
    return original_rows,np.asarray(residual),matrix,dict(
        path='public StructuredProgrammeBlock._evaluate original complete absolute records',
        activated_row_count=len(requested),refreshed_scalar_rows=len(original_rows),
        original_physical_records=physical_count,full_camera_mount_optics_world_columns_retained=True,
        original_geometry_FD_width_stencils_and_record_UV_sigma_retained=True,
        derivative_only_motion_and_prior_rows_not_evaluated=True)

def sparse_full_row_view(indices,matrix,complete_row_count):
    matrix=matrix.tocsr(copy=False);indices=np.asarray(indices,dtype=np.int64)
    sizes=np.zeros(complete_row_count,dtype=matrix.indptr.dtype);sizes[indices]=np.diff(matrix.indptr)
    pointer=np.empty(complete_row_count+1,dtype=matrix.indptr.dtype);pointer[0]=0
    np.cumsum(sizes,out=pointer[1:])
    return csr_matrix((matrix.data,matrix.indices,pointer),shape=(complete_row_count,matrix.shape[1]),copy=False)

def patch_result(supplied,indices,residual,matrix,detail,parameter_count):
    r,_,_,_,signature=_checked_block(supplied,parameter_count,False)
    indices=np.asarray(indices,dtype=np.int64);matrix=matrix.tocsr(copy=False)
    if (matrix.shape!=(len(indices),parameter_count) or not np.isfinite(matrix.data).all()
            or not np.array_equal(np.asarray(residual),r[indices])):
        raise ValueError('Exact current-row derivative patch changed current original residual/scales/normalization')
    if not matrix.has_sorted_indices:matrix=matrix.copy();matrix.sort_indices()
    return SourceJacobianRowPatch(indices,np.asarray(residual),matrix,signature.inventory_sha256,len(r),detail)

def replace_csr_rows(original,patch,requested,current_residual,inventory_sha256):
    """Replace only exact patch rows; every untouched CSR row keeps its bytes."""
    m=original.tocsr(copy=False);rows=np.asarray(patch.row_indices,dtype=np.int64);j=patch.matrix.tocsr(copy=False)
    if (patch.inventory_sha256!=inventory_sha256 or patch.source_row_count!=m.shape[0]
            or rows.ndim!=1 or len(rows)==0 or np.any(np.diff(rows)<=0)
            or rows[0]<0 or rows[-1]>=m.shape[0] or j.shape!=(len(rows),m.shape[1])
            or not np.isin(requested,rows).all() or not np.isfinite(j.data).all()
            or not np.array_equal(patch.residual,np.asarray(current_residual)[rows])):
        raise ValueError('Exact activated derivative row patch failed original all-row identity gate')
    sizes=np.diff(m.indptr).astype(np.int64);sizes[rows]=np.diff(j.indptr)
    pointer=np.empty(m.shape[0]+1,dtype=m.indptr.dtype);pointer[0]=0
    np.cumsum(sizes,out=pointer[1:])
    data=np.empty(int(pointer[-1]),dtype=m.data.dtype);columns=np.empty(int(pointer[-1]),dtype=m.indices.dtype)
    previous=0
    for k,row in enumerate(rows):
        source_first,source_last=m.indptr[[previous,row]]
        target_first,target_last=pointer[[previous,row]]
        data[target_first:target_last]=m.data[source_first:source_last];columns[target_first:target_last]=m.indices[source_first:source_last]
        first,last=j.indptr[k:k+2];start,stop=pointer[row:row+2]
        data[start:stop]=j.data[first:last];columns[start:stop]=j.indices[first:last];previous=int(row)+1
    first,last=m.indptr[[previous,m.shape[0]]];start,stop=pointer[[previous,m.shape[0]]]
    data[start:stop]=m.data[first:last];columns[start:stop]=m.indices[first:last]
    result=csr_matrix((data,columns,pointer),shape=m.shape,copy=False)
    if not result.has_sorted_indices:raise ValueError('Exact row replacement changed public sorted CSR order')
    return result
