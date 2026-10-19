"""Explicit common hockey world on the original six owned source blocks.

Install after certified advertisement bindings and before statistical/high
wrappers or actors. This module changes no source admission, material plane,
profile, loss, sigma, exposure, camera index or physical lens function. It
does not install the parent's external/wide driver or nominal reference rows.
"""
from __future__ import annotations

import copy
import cv2
import numpy as np
from scipy.sparse import csr_matrix

from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock
from z190_continuous_source_bundle import SCENE_UNIT_M
from installation_mount_candidate.mount_graph import (
    motion_mount_columns, insert_mount_csr)
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES
from shared_advertisement_pairs_candidate.paired_ad_graph import PairedAdMountOpticsBlock
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import (
    CorrectedSharedRinkNumericGeometryLanesAdPairs)
from .shared_ice_hockey_numeric_geometry import SharedIceHockeyNumericGeometry
from .hockey_source_chart import HockeyCoordinateChart, MountedHockeyOpticsChart, OLD_WORLD_INDICES


def append_hockey_chart_optics_csr(matrix, optics, indices,
                                 original_parameter_count=34204):
    """Keep all old CSR storage and insert lens3 before pair/hockey columns.

    The whole physical shape is explicit. No hidden tail truncation or old
    duplicate coalescing is allowed. Original roll rows receive no new lens
    entries. This also serves original external families with zero hockey J.
    """
    m=matrix.tocsr(copy=False);v=np.asarray(optics,float)
    if m.shape[1] not in (34245,34248):
        raise ValueError('Declared complete physical hockey20 or hockey23 CSR required')
    slots=np.asarray(indices,dtype=m.indices.dtype)
    if (original_parameter_count!=34204 or slots.shape!=(3,) or
        not np.array_equal(slots,np.arange(slots[0],slots[0]+3)) or
        slots[0] not in (34204,34207,34210) or np.any(m.indices<0) or
        np.any((m.indices>=34195)&(m.indices<34213)) or np.any(m.indices>=m.shape[1]) or
        not m.has_sorted_indices or v.ndim!=2 or v.shape[1]!=3 or len(v)>m.shape[0] or
        not np.isfinite(v).all()):
        raise ValueError('Original sorted source CSR, explicit pair/hockey world, and current lens3 required')
    active=v!=0.;counts=np.zeros(m.shape[0],dtype=m.indptr.dtype)
    counts[:len(v)]=active.sum(axis=1)
    pointer=m.indptr+np.r_[0,np.cumsum(counts,dtype=m.indptr.dtype)]
    data=np.empty(int(pointer[-1]),dtype=m.data.dtype)
    columns=np.empty(len(data),dtype=m.indices.dtype)
    for lo in range(0,m.shape[0],65536):
        hi=min(lo+65536,m.shape[0]);left,right=map(int,m.indptr[[lo,hi]])
        sizes=np.diff(m.indptr[lo:hi+1])
        rows=np.repeat(np.arange(lo,hi,dtype=m.indptr.dtype),sizes)
        inner=np.arange(right-left,dtype=m.indptr.dtype)-np.repeat(m.indptr[lo:hi]-left,sizes)
        old_columns=m.indices[left:right];tail=old_columns>=34213
        dest=pointer[rows]+inner+counts[rows]*tail
        data[dest],columns[dest]=m.data[left:right],old_columns
        stop=min(hi,len(v))
        if stop>lo:
            tails=np.bincount(rows[tail]-lo,minlength=hi-lo)
            rr,cc=np.nonzero(active[lo:stop]);rank=np.cumsum(active[lo:stop],axis=1)-1
            old_prefix=sizes[:stop-lo]-tails[:stop-lo]
            dest=pointer[lo+rr]+old_prefix[rr]+rank[rr,cc]
            data[dest],columns[dest]=v[lo+rr,cc],slots[cc]
    result=csr_matrix((data,columns,pointer),shape=m.shape,copy=False)
    result.has_sorted_indices=m.has_sorted_indices
    result.has_canonical_format=m.has_canonical_format
    return result


class HockeyMountOpticsBlock(PairedAdMountOpticsBlock):
    """Original source operations on one exact complete expanded chart."""
    def set_shared_geometry(self,geometry,parameter_indices):
        if type(geometry) is not SharedIceHockeyNumericGeometry:
            raise ValueError('Declared common hockey world owner required')
        chart=HockeyCoordinateChart(geometry.new_parameter_count)
        slots=np.asarray(parameter_indices,int)
        if not np.array_equal(slots,chart.world_indices):
            raise ValueError('Old q23 indices and contiguous new shared hockey indices required')
        self.geometry=geometry
        self.geometry_parameter_indices=slots.copy()
        self.geometry_parameter_count=geometry.parameter_count

    def evaluate(self,q,jacobian=True):
        self.optics_chart.coordinate_chart.core_q(q)
        # The original fixed XYZ scatter already accepts explicit geometry
        # slots/len(q). Its source operators, FD widths and source loss are
        # replayed unchanged, including all typed/custom factory fallbacks.
        result=FixedXYZRinkBlock.evaluate(self,q,jacobian)
        if not jacobian:return result
        residual,matrix=result;local=self.local(q)
        cal=self.bundle.current_calibration(local)
        rows=int(self.bundle.source_residual_rows)
        optics=np.zeros((rows,3));mount=np.zeros((rows,3))
        value=np.asarray(self.motion_columns_provider(self.bundle,local,cal,
            self.bundle.profile_weights,steps=self.optics_chart.physical_steps,
            physical_column_scales=PHYSICAL_COLUMN_SCALES),float)
        if value.shape!=(self.bundle.motion_rows,3):
            raise ValueError('Original full motion lens columns required')
        np.divide(value,self.bundle.motion_sigma[:,None],out=optics[:self.bundle.motion_rows])
        np.divide(motion_mount_columns(self.bundle,local,cal),self.bundle.motion_sigma[:,None],
                  out=mount[:self.bundle.motion_rows])
        cursor=self.bundle.motion_rows;geometry=self.geometry_values(q)
        for p,old_rows in self.groups:
            camera=self.bundle.camera(local,p)
            moved=[self.geometry.rebind(r,geometry) for r in old_rows]
            _,raw=self.absolute(camera,moved,True);raw=np.asarray(raw,float)
            count=sum(2*len(r['uv']) for r in old_rows)
            if raw.shape!=(count,15):
                raise ValueError('Original complete sigma-normalized camera15 absolute chain required')
            optics[cursor:cursor+count]=raw[:,[8,9,11]]*.001
            rotation=cv2.Rodrigues(np.asarray(camera[3:6]))[0]
            mount[cursor:cursor+count]=raw[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
            cursor+=count
        if cursor!=rows:raise ValueError('Original source row inventory changed')
        matrix=append_hockey_chart_optics_csr(matrix,optics,self.optics_chart.indices)
        return residual,insert_mount_csr(matrix,mount,self.optics_chart.mount_indices,34195)


def install_shared_ice_hockey_geometry(state,*,include_fr_ratio_deltas=False):
    """Install physical core/world only, atomically, before owned actors.

    Input is actual corrected physical34225/world23 (no bias/high tail).
    Output K20 is physical34245/world43. The caller then reconstructs its
    original three statistical bias rows and high-camera25 wrapper on the
    new shape; HockeyCoordinateChart maps the frozen original whole state.
    """
    if getattr(state,'source_actors',None) is not None:
        raise ValueError('Install common geometry before owned source actors')
    if hasattr(state,'shared_ice_hockey_contract'):
        raise ValueError('Common hockey geometry is installed once on a fresh physical graph')
    old=state.geometry;oldq=np.asarray(state.q,float)
    if (type(old) is not CorrectedSharedRinkNumericGeometryLanesAdPairs or
        old.parameter_count!=23 or oldq.shape!=(34225,) or not np.isfinite(oldq).all() or
        len(state.blocks)!=6 or not hasattr(state,'corrected_advertisement_source_binding_contract') or
        not np.array_equal(state.geometry_parameter_indices,OLD_WORLD_INDICES)):
        raise ValueError('Actual corrected advertisement physical34225/world23 graph required')
    if any(b.geometry is not old for b in state.blocks):
        raise ValueError('All original blocks must share the actual current world owner')
    # Reject pending old process jobs instead of touching somebody else's pool.
    from z190_parallel_pattern_evaluation import _pending
    if state.patterns is not None and id(state.patterns) in _pending:
        raise ValueError('Install before original pattern prefetch jobs are submitted')
    geometry=SharedIceHockeyNumericGeometry(old,include_fr_ratio_deltas=include_fr_ratio_deltas)
    chart=HockeyCoordinateChart(geometry.new_parameter_count)
    q=np.r_[oldq,np.zeros(geometry.new_parameter_count)]
    geometry.validate_parameters(q[np.asarray(chart.world_indices)])
    charts={name:MountedHockeyOpticsChart.from_original(c,geometry.new_parameter_count)
            for name,c in state.installation_optics.items()}
    replacements={};source_inventory=[]
    for block in state.blocks:
        name=block.bundle.calibration.installation_id
        if name not in charts:raise ValueError('Original installation chart identity missing')
        new=HockeyMountOpticsBlock.__new__(HockeyMountOpticsBlock)
        new.__dict__.update(vars(block));new.bundle=copy.copy(block.bundle)
        if getattr(block,'prepared',None) is not None:
            new.prepared=copy.copy(block.prepared);new.prepared.bundle=new.bundle
        new.optics_chart=charts[name]
        new.set_shared_geometry(geometry,chart.world_indices)
        for owner in (new,new.bundle,getattr(new,'prepared',None)):
            if owner is None:continue
            for key,value in list(vars(owner).items()):
                if value is old:setattr(owner,key,geometry)
        absolute_rows=int(block.bundle.source_residual_rows-block.bundle.motion_rows)
        capacity=dict(block.actor_output_capacity)
        capacity['nnz']+=geometry.new_parameter_count*absolute_rows
        new.actor_output_capacity=capacity
        replacements[id(block)]=new
        source_inventory.append(dict(name=block.name,installation_id=name,
            source_residual_rows=int(block.bundle.source_residual_rows),motion_rows=int(block.bundle.motion_rows),
            absolute_component_rows=absolute_rows,original_nnz_capacity=int(block.actor_output_capacity['nnz']),
            new_nnz_capacity=int(capacity['nnz']),row_capacity=int(capacity['rows']),
            original_motion_kernel_and_profiles_preserved=new.bundle.motion_kernel is block.bundle.motion_kernel and
                new.bundle.profile_weights is block.bundle.profile_weights,
            original_custom_absolute_callable_preserved=new.absolute is block.absolute))
    blocks=[replacements[id(b)] for b in state.blocks]
    bypts={p:replacements[id(b)] for p,b in state.bypts.items()}
    patterns=None
    if state.patterns is not None:
        patterns=type(state.patterns).__new__(type(state.patterns))
        patterns.__dict__.update(vars(state.patterns))
        patterns.installation_optics=charts
        if getattr(patterns,'state',state) is not state:
            raise ValueError('Pattern observer owner must be this actual shared state')
    contract=dict(schema='rinkspace.shared-ice-hockey-physical-world-graph-candidate.v1',
        coordinate_chart=chart.manifest(),world_geometry=geometry.manifest(q[np.asarray(chart.world_indices)]),
        original_physical_parameter_count=34225,physical_parameter_count=chart.core_count,
        original_world_parameter_count=23,world_parameter_count=geometry.parameter_count,
        new_shared_geometry_indices=list(chart.hockey_indices),geometry_parameter_indices=list(chart.world_indices),
        six_original_blocks=source_inventory,programme_source_blocks_installed=True,
        current_state_bypts_and_pattern_observers_follow_expanded_camera_chart=True,
        original_motion_ice_and_board_finite_material_planes_and_guards_unchanged=True,
        original_source_UV_sigma_role_support_order_count_and_loss_unchanged=True,
        no_new_source_admission=True,initial_new_geometry_preserves_original_world=True,
        old_core_coordinates_unchanged=True,additional_prior_rows=0,
        original_FR_ratio_interface_installed=bool(include_fr_ratio_deltas),
        rink_design_target_metres=[60.,30.],rink_design_target_is_engineering_not_survey=True,
        rink_60x30_boundary_transport_or_nominal_prior_installed=False,
        boundary_advertisements_and_finite_board_motion_geometry_unchanged=True,
        external_and_wide_driver_installed=False,complete_source_graph_installed=False,
        must_pass_fresh_complete_source_gate_before_solver=True,accuracy_accepted=False)
    optics_contract=dict(state.optics_contract,
        installations={name:c.manifest(next(b for b in blocks if b.bundle.calibration.installation_id==name).optics_base_calibration)
                       for name,c in charts.items()},
        whole_physical_parameter_count=chart.core_count,shared_ice_hockey_contract=contract)
    # Commit only after all source owners, capacity and chart checks succeed.
    old_blocks={id(b) for b in state.blocks}
    for key,value in list(vars(state).items()):
        if value is old:setattr(state,key,geometry)
        elif id(value) in old_blocks:setattr(state,key,replacements[id(value)])
    state.blocks=blocks;state.bypts=bypts;state.patterns=patterns
    state.geometry=geometry;state.geometry_parameter_indices=np.asarray(chart.world_indices,int)
    state.installation_optics=charts;state.q=q
    state.shared_ice_hockey_coordinate_chart=chart
    state.shared_ice_hockey_contract=contract
    state.hockey_parameter_indices=np.asarray(chart.hockey_indices,int)
    state.optics_contract=optics_contract
    return contract
