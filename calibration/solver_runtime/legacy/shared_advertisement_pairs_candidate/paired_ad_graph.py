"""Explicit q23 advertisement world chart on the complete mount+optics graph.

This installs no source admission or solver. The three qualified ice rows
must already be bound in the authenticated physical34213 graph. Only twelve
shared physical print coordinates are appended, before any owned actors.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import cv2,numpy as np
from scipy.sparse import csr_matrix
from installation_mount_candidate.mount_graph import (MountOpticsChart,MountOpticsBlock,MountOpticsPatterns,
    motion_mount_columns,insert_mount_csr)
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES
from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock
from z190_continuous_source_bundle import SCENE_UNIT_M
from .shared_rink_numeric_geometry_lanes_ad_pairs import SharedRinkNumericGeometryLanesAdPairs
from .shared_advertisement_pairs_numeric_geometry import DEFAULT_PAIRS

OLD_COUNT=34195
MOUNT_INDICES=tuple(range(34195,34204))
OPTICS_INDICES=tuple(range(34204,34213))
PAIRED_INDICES=tuple(range(34213,34225))
PHYSICAL_COUNT=34225


@dataclass(frozen=True)
class MountedPairedAdOpticsChart(MountOpticsChart):
    """Current original mount/lens function with a declared wholeq34225."""
    full_parameter_count:int=PHYSICAL_COUNT
    paired_ad_parameter_indices:tuple[int,...]=PAIRED_INDICES

    def delta(self,q):
        q=np.asarray(q,float)
        if (q.shape!=(PHYSICAL_COUNT,) or not np.isfinite(q).all() or
            self.full_parameter_count!=PHYSICAL_COUNT or self.original_parameter_count!=34204 or
            self.paired_ad_parameter_indices!=PAIRED_INDICES or
            len(self.indices)!=3 or tuple(self.indices)!=tuple(range(self.indices[0],self.indices[0]+3)) or
            self.indices[0] not in (34204,34207,34210) or
            tuple(self.mount_indices)!=tuple(range(34195+self.indices[0]-34204,34198+self.indices[0]-34204))):
            raise ValueError('Explicit complete34225 paired-ad/mount/lens chart required; no tail truncation')
        return q[np.asarray(self.indices)]*np.asarray(PHYSICAL_COLUMN_SCALES)

    def manifest(self,base):
        result=super().manifest(base)
        result.update(whole_physical_parameter_count=PHYSICAL_COUNT,
            current_camera_chart='explicit_mount9_optics9_paired_ad12_whole_state',
            mount_parameter_indices=list(self.mount_indices),paired_ad_parameter_indices=list(PAIRED_INDICES),
            extra_print_geometry_does_not_change_camera_function=True)
        return result


def append_paired_chart_optics_csr(matrix,optics,indices,original_parameter_count=34204):
    """Insert lens3 between preserved original CSR entries and pair12 entries.

    Only original source columns<34195 or explicitly declared new pair slots
    may already exist. Old repeated entries keep their values and row order.
    This helper also supports the external family with no pair NNZ at all.
    """
    m=matrix.tocsr(copy=False);v=np.asarray(optics,float)
    slots=np.asarray(indices,dtype=m.indices.dtype)
    if (original_parameter_count!=34204 or m.shape[1]!=PHYSICAL_COUNT or slots.shape!=(3,) or
        not np.array_equal(slots,np.arange(slots[0],slots[0]+3)) or slots[0] not in (34204,34207,34210) or
        np.any(m.indices<0) or np.any((m.indices>=OLD_COUNT)&(m.indices<PAIRED_INDICES[0])) or np.any(m.indices>=PHYSICAL_COUNT) or
        not m.has_sorted_indices or v.ndim!=2 or v.shape[1]!=3 or len(v)>m.shape[0] or not np.isfinite(v).all()):
        raise ValueError('Original CSR plus explicit pair12 columns, current lens3, and whole34225 required')
    active=v!=0.;counts=np.zeros(m.shape[0],dtype=m.indptr.dtype);counts[:len(v)]=active.sum(axis=1)
    pointer=m.indptr+np.r_[0,np.cumsum(counts,dtype=m.indptr.dtype)]
    data=np.empty(int(pointer[-1]),dtype=m.data.dtype);columns=np.empty(len(data),dtype=m.indices.dtype)
    for lo in range(0,m.shape[0],65536):
        hi=min(lo+65536,m.shape[0]);left,right=map(int,m.indptr[[lo,hi]])
        sizes=np.diff(m.indptr[lo:hi+1]);rows=np.repeat(np.arange(lo,hi,dtype=m.indptr.dtype),sizes)
        inner=np.arange(right-left,dtype=m.indptr.dtype)-np.repeat(m.indptr[lo:hi]-left,sizes)
        old_columns=m.indices[left:right];tail=old_columns>=PAIRED_INDICES[0]
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
    result.has_sorted_indices=m.has_sorted_indices;result.has_canonical_format=m.has_canonical_format
    return result


class PairedAdMountOpticsBlock(MountOpticsBlock):
    """Original native motion/camera chain; explicit q23 absolute geometry."""
    def set_shared_geometry(self,geometry,parameter_indices):
        slots=np.asarray(parameter_indices,int)
        expected=np.r_[np.array([0,1,2,3,4,34189,34190,34191,34192,34193,34194]),PAIRED_INDICES]
        if type(geometry) is not SharedRinkNumericGeometryLanesAdPairs or not np.array_equal(slots,expected):
            raise ValueError('Known original q11 plus declared pair12 geometry indices required')
        self.geometry=geometry;self.geometry_parameter_indices=slots.copy();self.geometry_parameter_count=23

    def evaluate(self,q,jacobian=True):
        # Replay the same base source operator. Only the previous tail-only
        # CSR inserter is inapplicable once new geometry columns follow lens.
        result=FixedXYZRinkBlock.evaluate(self,q,jacobian)
        if not jacobian:return result
        residual,matrix=result;local=self.local(q);cal=self.bundle.current_calibration(local)
        rows=self.bundle.source_residual_rows;optics=np.zeros((rows,3));mount=np.zeros((rows,3))
        value=np.asarray(self.motion_columns_provider(self.bundle,local,cal,self.bundle.profile_weights,
            steps=self.optics_chart.physical_steps,physical_column_scales=PHYSICAL_COLUMN_SCALES),float)
        if value.shape!=(self.bundle.motion_rows,3):raise ValueError('Original full motion lens columns required')
        np.divide(value,self.bundle.motion_sigma[:,None],out=optics[:self.bundle.motion_rows])
        np.divide(motion_mount_columns(self.bundle,local,cal),self.bundle.motion_sigma[:,None],out=mount[:self.bundle.motion_rows])
        cursor=self.bundle.motion_rows;geometry=self.geometry_values(q)
        for p,old_rows in self.groups:
            camera=self.bundle.camera(local,p);moved=[self.geometry.rebind(r,geometry) for r in old_rows]
            _,raw=self.absolute(camera,moved,True);raw=np.asarray(raw,float)
            count=sum(2*len(r['uv']) for r in old_rows)
            if raw.shape!=(count,15):raise ValueError('Complete original normalized camera15 absolute chain required')
            optics[cursor:cursor+count]=raw[:,[8,9,11]]*.001
            rotation=cv2.Rodrigues(np.asarray(camera[3:6]))[0]
            mount[cursor:cursor+count]=raw[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
            cursor+=count
        if cursor!=rows:raise ValueError('Original absolute source inventory changed')
        matrix=append_paired_chart_optics_csr(matrix,optics,self.optics_chart.indices)
        return residual,insert_mount_csr(matrix,mount,self.optics_chart.mount_indices,OLD_COUNT)


def install_paired_ad_geometry(state,geometry23=None):
    """Install only after reviewed ice on physical34213 and before actors."""
    from source_material_197_review.finite_ice_candidate import authenticated_physical_chart_scope
    if getattr(state,'source_actors',None) is not None or hasattr(state,'paired_ad_contract'):
        raise ValueError('Explicit paired-ad chart is installed once before source actors')
    scope=authenticated_physical_chart_scope(state,state.q)
    if scope['physical_core_parameter_count']!=34213:raise ValueError('Authenticated mount+optics physical graph required')
    material=getattr(state,'qualified_original_motion_ice_contract',None)
    if (not isinstance(material,dict) or material.get('schema')!='rinkspace.original-motion-three-reviewed-ice-plane-candidate.v1' or
        material.get('candidate_finite_count')!=471 or material.get('global_bundle_indices')!=[113623,113625,113636] or
        material.get('physical_core_parameter_count')!=34213):
        raise ValueError('Exactly the three reviewed ice rows must already be bound in the physical34213 graph')
    geometry=SharedRinkNumericGeometryLanesAdPairs(state.geometry.base) if geometry23 is None else geometry23
    if (type(geometry) is not SharedRinkNumericGeometryLanesAdPairs or geometry.parameter_count!=23 or
        geometry.pair_panels.pairs!=DEFAULT_PAIRS or geometry.base!=state.geometry.base):
        raise ValueError('Known q23 combined geometry on the unchanged shared board course required')
    # Keep the already authenticated old q11 world owner and its Ad1/14
    # panel available to the untouched external source binder. Worker
    # pickle reconstructs the same original classes and numeric authority.
    geometry.original=state.geometry
    geometry.rink=state.geometry.rink;geometry.panel=state.geometry.panel
    geometry.base=state.geometry.base
    # Correct only the source-only four HELD edges whose exact endpoint/t
    # chord generator is independently authenticated. FIT source groups and
    # all masks/profile weights remain untouched. This is a declared world
    # representation correction, never a nearest-course snap of an unknown row.
    held=dict(state.held);secondary_held=[];held_representation_ledger=[]
    selected=set()
    for pts,rows in held['secondary']:
        replacements=[]
        for row in rows:
            if int(pts)==229534976 and row.get('id') in('ad2:left','ad2:right','ad2:top','ad2:bottom'):
                if row['id'] in selected:raise ValueError('Original qualified HELD Ad2 edge duplicated')
                selected.add(row['id'])
                current,ledger=geometry.pair_panels.correct_certified_legacy_held_ad2_chord(row,pts,state.original.aliases)
                replacements.append(current);held_representation_ledger.append(ledger)
            else:replacements.append(row)
        secondary_held.append((pts,replacements))
    if selected!={'ad2:left','ad2:right','ad2:top','ad2:bottom'}:
        raise ValueError('All four actual packet-qualified HELD Ad2 edges must retain their original inventory')
    held['secondary']=secondary_held
    original_q=np.asarray(state.q);original_geometry_indices=state.geometry_parameter_indices.copy()
    q=np.r_[original_q,geometry.pair_panels.initial]
    geometry_indices=np.r_[original_geometry_indices,PAIRED_INDICES]
    geometry.validate_parameters(q[geometry_indices])
    charts={name:MountedPairedAdOpticsChart(c.installation_id,c.indices,c.original_parameter_count,c.physical_steps,
        c.mount_indices) for name,c in state.installation_optics.items()}
    replacements={}
    for old in state.blocks:
        new=PairedAdMountOpticsBlock.__new__(PairedAdMountOpticsBlock);new.__dict__.update(vars(old))
        new.bundle=copy.copy(old.bundle)
        if old.prepared is not None:
            new.prepared=copy.copy(old.prepared);new.prepared.bundle=new.bundle
        new.optics_chart=charts[old.bundle.calibration.installation_id]
        new.set_shared_geometry(geometry,geometry_indices)
        absolute_rows=old.bundle.source_residual_rows-old.bundle.motion_rows
        new.actor_output_capacity=dict(old.actor_output_capacity,nnz=old.actor_output_capacity['nnz']+12*absolute_rows)
        replacements[id(old)]=new
    state.blocks=[replacements[id(b)] for b in state.blocks]
    state.bypts={p:replacements[id(b)] for p,b in state.bypts.items()}
    patterns=MountOpticsPatterns.__new__(MountOpticsPatterns);patterns.__dict__.update(vars(state.patterns))
    patterns.installation_optics=charts;state.patterns=patterns
    state.geometry=geometry;state.geometry_parameter_indices=geometry_indices
    state.installation_optics=charts;state.q=q;state.held=held
    state.paired_ad_parameter_indices=np.asarray(PAIRED_INDICES,int)
    contract=dict(schema='rinkspace.shared-three-advertisement-pairs-physical-world-candidate.v1',
        original_physical_parameter_count=34213,physical_parameter_count=PHYSICAL_COUNT,
        original_q11_geometry_parameter_indices=original_geometry_indices.tolist(),
        geometry_parameter_indices=geometry_indices.tolist(),paired_ad_parameter_indices=list(PAIRED_INDICES),
        paired_ad_parameterization='absolute metre station_center_minor_arc_width_z_center_height_per_pair',
        pairs=[list(p) for p in DEFAULT_PAIRS],initial_paired_ad_parameters_m=geometry.pair_panels.initial.tolist(),
        initial_pair_tail_preserves_original_world_bitexact=True,original_core_indices_preserved=True,
        original_mount_parameter_indices=list(MOUNT_INDICES),original_optics_parameter_indices=list(OPTICS_INDICES),
        inherited_material_qualification_not_reinstalled=True,three_reviewed_ice_current_physical_chart_binding=material,
        geometry=geometry.manifest(q[geometry_indices]),
        one_common_physical_pair_per_event=True,source_pixels_sigmas_roles_order_held_unchanged=True,
        original_ad12_13_print_material_world_unchanged=True,original_ad1_14_external_panel_object_preserved=True,
        actual_board_course_and_existing_finite_planes_extents_unchanged=True,
        qualified_legacy_HELD_world_representation_corrections=held_representation_ledger,
        original_HELD_pixels_roles_support_and_weights_unchanged=True,
        historical_HELD_world_chord_is_retained_in_representation_ledger=True,
        corrected_HELD_world_bitexact_claim=False,
        added_physical_shared_parameters=12,physical_shared_parameter_count=59,additional_prior_rows=0,
        statistical_nuisance_shared_parameters=3,required_complete_fresh_shared_gate_parameter_count=62,
        complete_gate_passed=False,accuracy_accepted=False,production_modified=False)
    state.paired_ad_contract=contract
    state.optics_contract=dict(state.optics_contract,
        installations={name:c.manifest(next(b for b in state.blocks if b.bundle.calibration.installation_id==name).optics_base_calibration) for name,c in charts.items()},
        whole_physical_parameter_count=PHYSICAL_COUNT,paired_ad_contract=contract)
    return contract
