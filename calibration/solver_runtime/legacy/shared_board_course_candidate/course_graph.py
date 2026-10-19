"""Isolated programme source blocks on oldworld23 plus explicit course6.

Only six programme blocks and their finite plane/extent guards are installed.
Pattern, external observer, complete source adapter and full gate are mandatory
caller work; this installer deliberately makes no completegraph claim.
"""
from __future__ import annotations
import copy,os
import cv2,numpy as np
from scipy.sparse import csr_matrix
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock,scatter_fixed_xyz_source_csr
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_motion_native import _inverse_brown_native
from installation_mount_candidate.mount_graph import motion_mount_columns,insert_mount_csr
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES
from shared_advertisement_pairs_candidate.paired_ad_graph import PairedAdMountOpticsBlock
from .course_source_chart import (MountedCourseOpticsChart,CAMERA_COUNT,OLD_CAMERA_COUNT,
    WORLD_INDICES,COURSE_INDICES,OLD_WORLD_INDICES,COORDINATE_CONTRACT,
    append_course_chart_optics_csr)
from .shared_rink_course_numeric_geometry import SharedRinkCourseNumericGeometry
from .finite_plane_motion_columns import prepare_finite_plane_motion_cache,finite_plane_motion_columns
from .course_absolute_material import register_declared_absolute_course_rows

BOARD_COUNTS={'main1':2,'main7':0,'secondary':0,'low':15,'main9':0,'main4':0}
FINITE_COUNTS={'main1':13,'main7':0,'secondary':6,'low':471,'main9':0,'main4':0}


def _require(value,message):
    if not value:raise ValueError(message)


def _exact_source_data(left,right):
    if isinstance(left,dict)and isinstance(right,dict):
        return left.keys()==right.keys()and all(_exact_source_data(left[k],right[k])for k in left)
    if isinstance(left,(list,tuple))and isinstance(right,(list,tuple)):
        return len(left)==len(right)and all(_exact_source_data(a,b)for a,b in zip(left,right))
    if isinstance(left,np.ndarray)or isinstance(right,np.ndarray):
        return np.array_equal(left,right)
    return left==right


def _camera_q(q):
    value=np.asarray(q,float)
    _require(value.shape==(CAMERA_COUNT,)and np.isfinite(value).all(),
        'Explicit original34225 plus physicalcourse6 completecamera34231 state required')
    return value


def insert_selected_course_motion_csr(matrix,residual_rows,values,*,motion_rows):
    """Append declared motioncourse6 entries without canonicalizing old storage."""
    m=matrix;rr=np.asarray(residual_rows,int);v=np.asarray(values,float)
    _require(getattr(m,'format',None)=='csr'and m.shape[1]==CAMERA_COUNT and m.has_sorted_indices and
        rr.ndim==1 and v.shape==(len(rr),6)and np.isfinite(v).all()and
        np.all(rr>=0)and np.all(rr<motion_rows)and (len(rr)<2 or np.all(np.diff(rr)>0)),
        'Sorted original34231 CSR and unique sorted declared motion residual rows required')
    for row in rr:
        lo,hi=map(int,m.indptr[[row,row+1]])
        _require(not hi>lo or m.indices[hi-1]<COURSE_INDICES[0],
            'Motion course entries must be inserted once, after old camera/lever/lens columns')
    active=v!=0.;counts=np.zeros(m.shape[0],dtype=m.indptr.dtype);counts[rr]=active.sum(axis=1)
    pointer=m.indptr+np.r_[0,np.cumsum(counts,dtype=m.indptr.dtype)]
    data=np.empty(int(pointer[-1]),dtype=m.data.dtype);indices=np.empty(len(data),dtype=m.indices.dtype)
    source_cursor=destination=0
    for ordinal,row in enumerate(rr):
        stop=int(m.indptr[row+1]);size=stop-source_cursor
        data[destination:destination+size]=m.data[source_cursor:stop]
        indices[destination:destination+size]=m.indices[source_cursor:stop]
        destination+=size;source_cursor=stop
        columns=np.flatnonzero(active[ordinal]);n=len(columns)
        data[destination:destination+n]=v[ordinal,columns]
        indices[destination:destination+n]=np.asarray(COURSE_INDICES)[columns]
        destination+=n
    size=m.nnz-source_cursor
    data[destination:destination+size]=m.data[source_cursor:]
    indices[destination:destination+size]=m.indices[source_cursor:]
    result=csr_matrix((data,indices,pointer),shape=m.shape,copy=False)
    result.has_sorted_indices=m.has_sorted_indices;result.has_canonical_format=m.has_canonical_format
    return result


class CourseMountOpticsBlock(PairedAdMountOpticsBlock):
    def set_shared_geometry(self,geometry,parameter_indices):
        slots=np.asarray(parameter_indices,int)
        _require(type(geometry)is SharedRinkCourseNumericGeometry and geometry.parameter_count==29 and
            np.array_equal(slots,WORLD_INDICES),'Authenticated explicitold23 + course6 world29 indices required')
        self.geometry=geometry;self.geometry_parameter_indices=slots.copy();self.geometry_parameter_count=29

    def synchronize_course(self,q):
        value=_camera_q(q);q29=value[self.geometry_parameter_indices]
        key=q29.tobytes()
        if self._course_current_geometry_key==key:return
        self.geometry.validate_parameters(q29)
        bottom,top,_=self.geometry.course.sample_perimeter(q29[23:])
        # These are precisely the two scene fields consumed by the unchanged
        # finite source guard. Other source geometry/provenance stays intact.
        self.bundle.scene['ice_boundary']['points_xyz']=bottom.tolist()
        self.bundle.scene['board_surface']['top_edge_xyz']=top.tolist()
        if len(self.course_board_indices):
            bound=self.geometry.finite_board_records(self.course_original_board_records,q29,
                original_planes=self.course_original_board_planes)
            self.bundle.finite_planes[self.course_board_indices]=bound['planes']
            records=(self.course_original_board_records if np.array_equal(q29[23:],self.geometry.course.initial)
                else bound['records'])
            for position,record in zip(self.course_board_record_positions,records):
                self.bundle.finite_records[int(position)]=copy.deepcopy(record)
            self.course_current_plane_values=bound['planes']
            self.course_current_plane_jacobian=bound['plane_jacobian']
        self._course_current_geometry_key=key

    def local(self,q):
        # Crucial for actor profile_motion_residual(block.local(q)) as well
        # as normal/replay/trial guards: geometry updates before cache/rays.
        self.synchronize_course(q)
        return super().local(q)

    def course_motion_columns(self,local,cal):
        selected=self.course_board_indices;m=len(selected)
        if not m:return np.empty(0,int),np.empty((0,6))
        pp=np.array([1920.*cal.principal_x_fraction[0],1080.*cal.principal_y_fraction[0]])
        key=(np.ascontiguousarray(local).tobytes(),pp.tobytes(),float(cal.k1[0]),tuple(cal.lever_xyz))
        if self._course_camera_cache_key!=key:
            vv,rr,_,cc,_,_=self.bundle.cache(local,False);focal=np.exp(vv[:,3])
            ia,ib=self.bundle.ia[selected],self.bundle.ib[selected]
            ra,rb=rr[ia],rr[ib]
            normalized=np.empty((m,2));threads=int(os.environ.get('RINKSPACE_MOTION_THREADS','1'))
            _require(threads>0,'Original positive native motion thread count required')
            # Preserve the original native batch-wide Brown inversion stop.
            # Invert only the original65536 chunks containing admittedboard
            # rows; never run six fullsource finite differences.
            for chunk in np.unique(selected//65536):
                lo=int(chunk)*65536;hi=min(lo+65536,len(self.bundle.a))
                group=np.flatnonzero((selected>=lo)&(selected<hi))
                ray,_=_inverse_brown_native(np.ascontiguousarray(self.bundle.a[lo:hi]),
                    np.ascontiguousarray(focal[self.bundle.ia[lo:hi]]),pp,float(cal.k1[0]),threads)
                normalized[group]=ray[selected[group]-lo]
            direction=np.einsum('ni,nij->nj',np.c_[normalized,np.ones(m)],ra)
            sigma=self.bundle.motion_sigma.reshape(-1,2)[selected]
            weights=None if self.bundle.profile_weights is None else self.bundle.profile_weights[selected]
            k=dict(a=self.bundle.a[selected],b=self.bundle.b[selected],fa=focal[ia],fb=focal[ib],
                ra=ra,rb=rb,ca=cc[ia],cb=cc[ib],pp=pp,k1=cal.k1[0],
                planes=self.bundle.finite_planes[selected],sigma=sigma,profile_weight=weights)
            self._course_camera_cache=prepare_finite_plane_motion_cache(k,np.ones(m,dtype=bool),
                source_direction_selected=direction)
            self._course_camera_cache_key=key
        _,columns,audit=finite_plane_motion_columns(self._course_camera_cache,
            self.course_current_plane_values,self.course_current_plane_jacobian,sigma_normalized=True)
        self.course_latest_motion_column_audit=audit
        rows=(2*selected[:,None]+np.arange(2)).reshape(-1)
        return rows,columns

    def evaluate(self,q,jacobian=True):
        result=self.evaluate_camera_geometry(q,jacobian)
        if not jacobian:return result
        residual,matrix=result;local=self.local(q);cal=self.bundle.current_calibration(local)
        rows=self.bundle.source_residual_rows;optics=np.zeros((rows,3));mount=np.zeros((rows,3))
        value=np.asarray(self.motion_columns_provider(self.bundle,local,cal,self.bundle.profile_weights,
            steps=self.optics_chart.physical_steps,physical_column_scales=PHYSICAL_COLUMN_SCALES),float)
        _require(value.shape==(self.bundle.motion_rows,3),'All original native motion optics rows required')
        np.divide(value,self.bundle.motion_sigma[:,None],out=optics[:self.bundle.motion_rows])
        np.divide(motion_mount_columns(self.bundle,local,cal),self.bundle.motion_sigma[:,None],out=mount[:self.bundle.motion_rows])
        cursor=self.bundle.motion_rows;geometry=self.geometry_values(q)
        for p,old_rows in self.groups:
            camera=self.bundle.camera(local,p);moved=[self.geometry.rebind(r,geometry)for r in old_rows]
            _,raw=self.absolute(camera,moved,True);raw=np.asarray(raw,float)
            count=sum(2*len(r['uv'])for r in old_rows)
            _require(raw.shape==(count,15),'Complete originalnormalized camera15 absolute chain required')
            optics[cursor:cursor+count]=raw[:,[8,9,11]]*.001
            rotation=cv2.Rodrigues(np.asarray(camera[3:6]))[0]
            mount[cursor:cursor+count]=raw[:,:3]@rotation.T/(100.*SCENE_UNIT_M);cursor+=count
        _require(cursor==rows,'Original absolute source inventory changed')
        matrix=append_course_chart_optics_csr(matrix,optics,self.optics_chart.indices)
        matrix=insert_mount_csr(matrix,mount,self.optics_chart.mount_indices,34195)
        rcourse,jcourse=self.course_motion_columns(local,cal)
        matrix=insert_selected_course_motion_csr(matrix,rcourse,jcourse,motion_rows=self.bundle.motion_rows)
        return residual,matrix

    def evaluate_camera_geometry(self,q,jacobian=True):
        """Original FixedXYZ evaluation with certified companion-aware activity.

        Only the sparse-FD activity declaration and its actual inactive-field
        verification differ. Source operators, FD step, original residual and
        camera Jacobian, sigma handling and scatter remain identical.
        """
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
                            active=np.asarray(self.geometry.source_parameter_activity(row,geometry_values))
                            _require(active.dtype==np.dtype(bool)and active.shape==(ng,),
                                'Actual primary-plus-companion activity must declare all29 geometry columns')
                            candidates=np.flatnonzero(active)
                            fields=tuple(self.geometry.source_parameter_fields(row))
                            _require(fields and fields[0]=='world'and len(set(fields))==len(fields)and
                                all(field in row for field in fields),'Explicit original source world fields required')
                            for c in np.flatnonzero(~active):
                                plus,minus=geometry_values.copy(),geometry_values.copy();plus[c]+=step;minus[c]-=step
                                rp,rm=self.geometry.rebind(row,plus),self.geometry.rebind(row,minus)
                                if any(not np.array_equal(rp[field],rm[field])for field in fields):
                                    raise ValueError('Declared absent material column changed an actual primary or companion FD world')
                            columns=candidates
                        for c in columns:
                            plus,minus=geometry_values.copy(),geometry_values.copy();plus[c]+=step;minus[c]-=step
                            geo[cursor:cursor+size,c]=(self.absolute(camera,[self.geometry.rebind(row,plus)],False)-
                                self.absolute(camera,[self.geometry.rebind(row,minus)],False))/(2*step)
                    cursor+=size
            matrix=scatter_fixed_xyz_source_csr(localj,self.bundle.source_residual_rows,geo,
                self.bundle.motion_rows,self.geometry_parameter_indices,self.xyz_slots,self.start,len(q))
            return residual,matrix
        finally:self.bundle.abs_groups=self.groups


def activate_course_state(state,q_camera):
    """Synchronize owned parentblocks before caller's external/profile route."""
    q=_camera_q(q_camera)
    _require(getattr(state,'course_contract',{}).get('schema')=='rinkspace.programme-block-course6-physical-graph-candidate.v1',
        'Explicit once-installed programme course graph required')
    q29=q[np.asarray(WORLD_INDICES)];state.geometry.current_course_q=q29[23:].copy()
    state.geometry.current_world_q29=q29.copy()
    for block in state.blocks:block.synchronize_course(q)
    # Root-owned adapters may attach explicit callbacks/panelproxy afterthis
    # block installer; no stale external/pattern route is silently certified.
    proxy=getattr(state.geometry,'panel',None)
    if hasattr(proxy,'bind_course'):proxy.bind_course(q29[23:].copy())
    for callback in getattr(state,'course_activation_callbacks',()):callback(q)


def install_course_geometry(state):
    """Install exactoldprogramme prefix and course6 before any sourceactor."""
    _require(getattr(state,'source_actors',None)is None and not hasattr(state,'course_contract')and
        np.shape(state.q)==(OLD_CAMERA_COUNT,)and
        state.paired_ad_contract['physical_parameter_count']==OLD_CAMERA_COUNT and
        state.optics_contract['whole_physical_parameter_count']==OLD_CAMERA_COUNT and
        np.array_equal(state.geometry_parameter_indices,OLD_WORLD_INDICES),
        'Authenticated physical34225 paired/mount/lens graph before source actors required')
    _require(state.paired_ad_contract['three_reviewed_ice_current_physical_chart_binding']['candidate_finite_count']==471,
        'Reviewed three material qualification must already be installed, never reapplied')
    geometry=SharedRinkCourseNumericGeometry(state.geometry.base,SCENE_UNIT_M)
    # Keep the exact old numeric owner private; never mutate the previous
    # sourcegraph when currentcourse guards temporarily alter oldguard fields.
    geometry.original=copy.deepcopy(state.geometry)
    geometry.initial[:23]=geometry.original.initial
    absolute_rows=[row for block in state.blocks for _,rows in block.groups for row in rows]
    absolute_rows.extend(row for groups in state.held.values()for _,rows in groups for row in rows)
    absolute_material=register_declared_absolute_course_rows(geometry,absolute_rows)
    q=np.r_[np.asarray(state.q),geometry.course.initial]
    geometry.validate_parameters(q[np.asarray(WORLD_INDICES)])
    charts={name:MountedCourseOpticsChart.from_paired(c)for name,c in state.installation_optics.items()}
    replacements={};inventory=[]
    for old in state.blocks:
        _require(type(old)is PairedAdMountOpticsBlock and old.name in BOARD_COUNTS,
            'All six actual original programme PairedAdMountOpticsBlock objects required')
        new=CourseMountOpticsBlock.__new__(CourseMountOpticsBlock);new.__dict__.update(vars(old))
        new.bundle=copy.copy(old.bundle);new.bundle.finite_planes=old.bundle.finite_planes.copy()
        new.bundle.finite_mask=old.bundle.finite_mask.copy()
        new.bundle.finite_records=copy.deepcopy(old.bundle.finite_records)
        guard_scene=geometry.base if old.bundle.scene is None else old.bundle.scene
        new.bundle.scene=dict(guard_scene,
            ice_boundary=dict(guard_scene['ice_boundary']),board_surface=dict(guard_scene['board_surface']))
        new.bundle.priors=lambda local,cache,b=new.bundle:tuple(x[1:]for x in MovingPivotSourceBundle.priors(b,local,cache))
        if old.prepared is not None:
            new.prepared=copy.copy(old.prepared);new.prepared.bundle=new.bundle
        new.optics_chart=charts[old.bundle.calibration.installation_id]
        new.set_shared_geometry(geometry,WORLD_INDICES)
        positions=[i for i,r in enumerate(new.bundle.finite_records)if r['surface_kind']=='finite_board_face']
        positions.sort(key=lambda i:int(new.bundle.finite_records[i]['global_bundle_index']))
        records=[copy.deepcopy(new.bundle.finite_records[i])for i in positions]
        indices=np.array([r['global_bundle_index']for r in records],dtype=int)
        _require(len(indices)==BOARD_COUNTS[old.name]and len(set(indices.tolist()))==len(indices)and
            np.all(indices>=0)and np.all(indices<len(new.bundle.a))and
            int(np.sum(new.bundle.finite_mask))==FINITE_COUNTS[old.name],
            'Actual declared board2+15 and finite490 source inventory must remain exact')
        new.course_board_indices=indices;new.course_board_record_positions=np.asarray(positions,int)
        new.course_original_board_records=records;new.course_original_board_planes=new.bundle.finite_planes[indices].copy()
        new.course_current_plane_values=new.course_original_board_planes.copy()
        new.course_current_plane_jacobian=np.zeros((len(indices),4,6))
        new._course_current_geometry_key=None;new._course_camera_cache_key=None;new._course_camera_cache=None
        absolute_rows=old.bundle.source_residual_rows-old.bundle.motion_rows
        new.actor_output_capacity=dict(old.actor_output_capacity,
            nnz=old.actor_output_capacity['nnz']+6*absolute_rows+6*2*len(indices))
        new.synchronize_course(q)
        _require(np.array_equal(new.bundle.finite_planes,old.bundle.finite_planes)and
            _exact_source_data(new.bundle.finite_records,old.bundle.finite_records),
            'Original finite plane coefficients or source records changed at course initialization')
        replacements[id(old)]=new
        inventory.append(dict(block=old.name,all_finite_pairs=int(np.sum(new.bundle.finite_mask)),
            finite_board_pairs=len(indices),finite_board_bundle_indices=indices.tolist(),
            source_rows=old.bundle.source_residual_rows,motion_rows=old.bundle.motion_rows,
            original_source_material_row_roles_preserved=True,actor_capacity=new.actor_output_capacity))
    _require({r['block']for r in inventory}==set(BOARD_COUNTS), 'All actualsix programme sourceblocks required')
    state.blocks=[replacements[id(b)]for b in state.blocks]
    state.bypts={p:replacements[id(b)]for p,b in state.bypts.items()}
    state.geometry=geometry;state.geometry_parameter_indices=np.asarray(WORLD_INDICES,int)
    state.installation_optics=charts;state.q=q;state.camera_count=CAMERA_COUNT
    state.course_parameter_indices=np.asarray(COURSE_INDICES,int)
    contract=dict(schema='rinkspace.programme-block-course6-physical-graph-candidate.v1',
        original_camera_parameter_count=OLD_CAMERA_COUNT,camera_parameter_count=CAMERA_COUNT,
        course_parameter_indices=list(COURSE_INDICES),world_parameter_indices=list(WORLD_INDICES),
        initial_course_parameters=geometry.course.initial.tolist(),course_parameter_names=list(geometry.course.parameter_names),
        coordinate_contract=COORDINATE_CONTRACT,geometry=geometry.manifest(q[np.asarray(WORLD_INDICES)]),
        actual_source_block_inventory=inventory,actual_finite_board_pairs=17,actual_all_finite_pairs=490,
        previous_low_reviewed_finite471_qualification_preserved=True,
        planes_convention='n.X+d=0 in original scene units',original1m_unknown_depth_domain_preserved=True,
        absolute_course_material_binding=absolute_material,
        absolute_geometry_FD_activity_includes_certified_companion_fields=True,
        unknown_depth_rows_not_changed_or_materialized=True,finite_plane_J_has_no_depth_profile_or_Schur=True,
        finite_plane_geometry_J_sigma_normalized_once=True,finite_plane_J_robust_weight_multiplications=0,
        private_guard_ice_footprint_board_top_and_named_extent_synchronized_before_local_cache=True,
        current_planes_synchronized_before_ownedactor_profile_update=True,
        initial_old_r_and_old_J_full_graph_verified=False,old_source_counts_rows_pixels_sigmas_roles_preserved=True,
        programme_block_route_installed=True,pattern_course_route_installed=False,
        external_current_panel_route_installed=False,complete_source_graph_installed=False,
        expected_physical_shared_parameters=65,expected_full_shared_parameters_with_stat3=68,
        required_complete_fresh_gate_directions=72,required_relative_tolerance=.005,
        new_source_rows_or_prior_rows=0,new_physical_world_parameters=6,new_camera_parameters=0,
        hardware_survey_measurement_claim=False,accuracy_accepted=False,production_modified=False)
    state.course_contract=contract
    state.optics_contract=dict(state.optics_contract,
        installations={name:c.manifest(next(b for b in state.blocks if b.bundle.calibration.installation_id==name).optics_base_calibration)
            for name,c in charts.items()},whole_physical_parameter_count=CAMERA_COUNT,course_contract=contract)
    activate_course_state(state,q)
    return contract
