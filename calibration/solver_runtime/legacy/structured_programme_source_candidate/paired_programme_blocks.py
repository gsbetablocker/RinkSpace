"""Owned six-programme component route; complete source factory stays caller-owned.

No old kernel, row admission, robust/profile rule, FD width or source operator
is replaced. Named geometry changes intentionally; no zero-r/J claim follows.
"""
from __future__ import annotations
import copy,os,time,json,cProfile,pstats,hashlib,numpy as np
from pathlib import Path
import cv2
from scipy.sparse import csr_matrix
from shared_board_course_candidate.course_graph import CourseMountOpticsBlock,BOARD_COUNTS,FINITE_COUNTS
from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock,PHYSICAL_COLUMN_SCALES
from installation_mount_candidate.mount_graph import insert_mount_csr,motion_mount_columns
from shared_ice_hockey_geometry_candidate.hockey_graph import HockeyMountOpticsBlock
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_joint_fixed_xyz_rink_block import scatter_fixed_xyz_source_csr
from .paired_programme_chart import PairedProgrammeCoordinateChart as StructuredProgrammeCoordinateChart,PairedProgrammeMountOpticsChart as StructuredProgrammeMountOpticsChart,WORLD_INDICES,CORE_COUNT,OLD
from .paired_location_programme_geometry import PairedLocationProgrammeGeometry as StructuredProgrammeGeometry

SCHEMA='rinkspace.owned-six-programme-structured66-component-source-graph.v1'
COURSE_GLOBAL=WORLD_INDICES[44:50]

def insert_selected_motion_geometry_csr(matrix,rows,values,*,motion_rows,geometry_columns=None):
    """Keep original CSR entries; add only the current finite source columns."""
    m=matrix;rr=np.asarray(rows,int);v=np.asarray(values,float)
    columns=np.asarray(COURSE_GLOBAL if geometry_columns is None else geometry_columns,int)
    if (getattr(m,'format',None)!='csr' or m.shape[1]<CORE_COUNT or not m.has_sorted_indices or
        columns.ndim!=1 or not len(columns)or len(np.unique(columns))!=len(columns)or
        np.any(columns<0)or np.any(columns>=m.shape[1])or
        rr.ndim!=1 or v.shape!=(len(rr),len(columns)) or not np.isfinite(v).all() or
        np.any(rr<0) or np.any(rr>=motion_rows) or len(rr)>1 and np.any(np.diff(rr)<=0)):
        raise ValueError('Declared physical34268 sorted CSR and actual selected motion rows required')
    order=np.argsort(columns,kind='stable');columns=columns[order];v=v[:,order]
    for row in rr:
        lo,hi=map(int,m.indptr[[row,row+1]])
        if np.intersect1d(m.indices[lo:hi],columns).size:raise ValueError('Selected finite geometry entries must be inserted exactly once')
    active=v!=0;counts=np.zeros(m.shape[0],dtype=m.indptr.dtype);counts[rr]=active.sum(axis=1)
    ptr=m.indptr+np.r_[0,np.cumsum(counts,dtype=m.indptr.dtype)]
    data=np.empty(int(ptr[-1]),m.data.dtype);cols=np.empty(len(data),m.indices.dtype);src=dest=0
    for ordinal,row in enumerate(rr):
        begin=int(m.indptr[row]);stop=int(m.indptr[row+1]);n=stop-src
        data[dest:dest+n]=m.data[src:stop];cols[dest:dest+n]=m.indices[src:stop];dest+=n;src=stop
        selected=np.flatnonzero(active[ordinal]);n=len(selected)
        data[dest:dest+n]=v[ordinal,selected];cols[dest:dest+n]=columns[selected];dest+=n
        # Current course columns normally follow every camera column. Retain
        # the same fast append; a generic layout still gets a sorted row.
        if n and stop>begin and columns[selected[0]]<m.indices[stop-1]:
            first=int(ptr[row]);last=int(ptr[row+1]);sort=np.argsort(cols[first:last],kind='stable')
            data[first:last]=data[first:last][sort];cols[first:last]=cols[first:last][sort]
    data[dest:]=m.data[src:];cols[dest:]=m.indices[src:];out=csr_matrix((data,cols,ptr),shape=m.shape,copy=False)
    out.has_sorted_indices=m.has_sorted_indices;out.has_canonical_format=m.has_canonical_format;return out


def check_existing_finite_perimeter_sources(state,q,*,output_path=None):
    """Check the seventeen real retained pixels on the already prepared graph.

    Only their native finite-plane values and compact derivatives are produced;
    the complete graph, source cameras, profiles and rows remain caller-owned.
    """
    import cv2
    from z190_motion_native import native_motion
    from shared_board_course_candidate.finite_plane_motion_columns import prepare_finite_plane_motion_cache,finite_plane_motion_columns
    from .fixed_pivot_positive_domain_fd import print_domain_directional_stencil
    binder=state.geometry;existing=binder._existing_finite_perimeter_binding
    core=np.asarray(q,float)[:state.structured_programme_coordinate_chart.core_count].copy()
    world=core[state.geometry_parameter_indices].copy();ng=binder.parameter_count
    if len(existing.records)!=17:raise ValueError('All original seventeen finite board pixels are required')
    checks=[];bound=existing._build_current_binding(binder,world)
    for name in ('main1','low'):
        block=next(b for b in state.blocks if b.name==name);bundle=block.bundle
        positions=np.array([i for i,r in enumerate(existing.records)if r['source_role_block']==name],int)
        rows=[existing.records[i]for i in positions]
        current={tuple(r['existing_finite_perimeter_key']):int(r['global_bundle_index'])
                 for r in block.course_original_board_records if 'existing_finite_perimeter_key'in r}
        indices=np.array([current[(name,int(r['original_global_bundle_index']))]for r in rows],int)
        a=bundle.a[indices];b=bundle.b[indices];sigma=bundle.motion_sigma.reshape(-1,2)[indices]
        stamps=np.asarray(bundle.stamps,int)
        stamps_a=stamps[bundle.ia[indices]];stamps_b=stamps[bundle.ib[indices]]
        if not (bundle.finite_mask[indices].all()and np.array_equal(a,[r['source_uv']for r in rows])
                and np.array_equal(b,[r['target_uv']for r in rows])
                and np.array_equal(sigma,[r['source_sigma_px']for r in rows])
                and np.array_equal(stamps_a,[r['source_pts']for r in rows])
                and np.array_equal(stamps_b,[r['target_pts']for r in rows])):
            raise ValueError('Original seventeen UV/sigma/actual clock/finite roles changed')
        local=block.camera_local(core);ca=np.array([bundle.camera(local,int(p))for p in stamps_a]);cb=np.array([bundle.camera(local,int(p))for p in stamps_b])
        ra=np.array([cv2.Rodrigues(c[3:6])[0]for c in ca]);rb=np.array([cv2.Rodrigues(c[3:6])[0]for c in cb])
        cal=bundle.current_calibration(local)
        pp=np.array([1920.*cal.principal_x_fraction[0],1080.*cal.principal_y_fraction[0]])
        weights=None if bundle.profile_weights is None else bundle.profile_weights[indices]
        k=dict(a=a,b=b,fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),ra=ra,rb=rb,ca=ca[:,:3],cb=cb[:,:3],
            pp=pp,k1=float(cal.k1[0]),sigma=sigma,profile_weight=weights,da=None,db=None,
            dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,jacobian=False)
        minimum=float(bundle.motion_kernel.minimum_source_depth_scene_units)
        def values(w):
            current_bound=existing._build_current_binding(binder,w)
            r,_=native_motion(**dict(k,planes=current_bound['planes'][positions]),minimum_source_depth_scene_units=minimum,schur=True)
            return r/sigma.reshape(-1),current_bound
        residual,_=values(world);planes=bound['planes'][positions];jp=bound['plane_jacobian'][positions]
        cache=prepare_finite_plane_motion_cache(dict(k,planes=planes),np.ones(len(indices),bool))
        _,columns,audit=finite_plane_motion_columns(cache,planes,jp,sigma_normalized=True)
        logical=np.unique(np.r_[np.arange(44,50),np.flatnonzero(np.any(jp!=0.,axis=(0,1)))]).astype(int)
        actual=block.course_current_plane_jacobian[np.searchsorted(block.course_board_indices,indices)]
        actual_logical=block.course_current_motion_geometry_logical_columns
        if not np.array_equal(actual,jp[:,:,actual_logical]):raise ValueError('Actual programme finite-plane derivative chain differs')
        fd_checks=[]
        for col in logical:
            direction=np.zeros(ng);direction[col]=1.
            stencil=print_domain_directional_stencil(world,direction,1e-6,binder._q)
            while True:
                samples=[(residual,bound)if offset==0. else values(world+offset*direction)for offset in stencil['offsets']]
                if all(np.array_equal(x[1]['selected_course_facets'][positions],bound['selected_course_facets'][positions])for x in samples):break
                if not np.any(world+stencil['step']*direction!=world):raise ValueError('Actual finite facet derivative has no representable branch step')
                stencil=print_domain_directional_stencil(world,direction,stencil['step']*.1,binder._q)
            if stencil['method']=='original_symmetric':observed=(samples[1][0]-samples[0][0])/(2*stencil['step'])
            else:observed=stencil['side']*(4*(samples[1][0]-samples[0][0])-(samples[2][0]-samples[0][0]))/(2*stencil['step'])
            error=float(abs(observed-columns[:,col]).max());allowed=2e-3+2e-5*float(abs(observed).max())
            if error>allowed:raise ValueError('Actual original finite-perimeter native FD failed: '+str((name,col,error,allowed)))
            fd_checks.append(dict(logical_column=int(col),canonical_physical_column=int(state.geometry_parameter_indices[col]),
                FD_error=error,allowed=allowed,stencil=stencil))
        old6=finite_plane_motion_columns(cache,planes,jp[:,:,44:50],sigma_normalized=True)[1]
        if not np.array_equal(old6,columns[:,44:50]):raise ValueError('Generalized plane columns changed original six-column arithmetic')
        da=np.array([cv2.Rodrigues(c[3:6])[1].reshape(3,3,3)for c in ca]);db=np.array([cv2.Rodrigues(c[3:6])[1].reshape(3,3,3)for c in cb])
        lever=np.asarray(cal.lever_xyz);dva=np.zeros((len(indices),3,4));dvb=np.zeros_like(dva)
        dva[:,:,:3]=np.einsum('naij,i->nja',da,lever);dvb[:,:,:3]=np.einsum('naij,i->nja',db,lever)
        raw,_=native_motion(**dict(k,planes=planes,jacobian=True,da=da,db=db,dca_view=dva,dcb_view=dvb,
            dca_shared=np.zeros((len(indices),3,4)),dcb_shared=np.zeros((len(indices),3,4))),
            minimum_source_depth_scene_units=minimum,schur=True)
        camera_rows=np.repeat(np.arange(2*len(indices)),4)
        parts=[];camera_columns=[]
        for view,jcam in zip((bundle.ia[indices],bundle.ib[indices]),raw[1:3]):
            parts.append(np.asarray(jcam)/sigma.reshape(-1,1))
            camera_columns.append(np.repeat(block.start+4*view,2)[:,None]+np.arange(4))
        original_csr=csr_matrix((np.r_[parts[0].ravel(),parts[1].ravel()],
            (np.r_[camera_rows,camera_rows],np.r_[camera_columns[0].ravel(),camera_columns[1].ravel()])),shape=(2*len(indices),len(core)))
        inserted=insert_selected_motion_geometry_csr(original_csr,np.arange(2*len(indices)),
            columns[:,logical],motion_rows=2*len(indices),geometry_columns=state.geometry_parameter_indices[logical])
        if not inserted.has_sorted_indices or not np.array_equal(inserted[:,state.geometry_parameter_indices].toarray(),columns):
            raise ValueError('Actual current full physical columns are not mapped once into sorted CSR')
        old_camera_columns=np.unique(original_csr.indices)
        if not np.array_equal(inserted[:,old_camera_columns].toarray(),original_csr[:,old_camera_columns].toarray()):
            raise ValueError('Existing native camera CSR changed during course-column insertion')
        legacy_check={}
        try:
            historical=[r['original_finite_record']for r in rows]
            legacy=binder._qualification_prefix.finite_board_records(historical,world[:62],original_planes=planes)
            legacy_r,_=native_motion(**dict(k,planes=legacy['planes']),minimum_source_depth_scene_units=minimum,schur=True)
            legacy_delta=float(abs(residual-legacy_r/sigma.reshape(-1)).max())
            # Equality is claimed only where the original straight plane is
            # actually the same current facet; a different physical operator
            # remains explicit rather than being forced into an old face.
            norm=lambda value:value/np.linalg.norm(value[:,:3],axis=1)[:,None]
            lp,cp=norm(legacy['planes']),norm(planes)
            same=np.minimum(np.max(abs(lp-cp),axis=1),np.max(abs(lp+cp),axis=1))<1e-12
            if same.all()and legacy_delta>1e-10:raise ValueError('Same actual straight planes changed retained native residuals')
            legacy_check=dict(old_straight_domain_legal=True,same_current_physical_plane=bool(same.all()),
                legacy_native_r_max_difference=legacy_delta,retained_r_equal_claimed=bool(same.all()))
        except ValueError as error:
            if 'current qualified print exceeds current board Z domain'in str(error):raise
            if 'Same actual straight planes changed'in str(error):raise
            legacy_check=dict(old_straight_domain_legal=False,old_rejection=str(error),
                current_full_course_bound_valid=True,retained_r_equal_claimed=False)
        checks.append(dict(block=name,original_points=len(indices),active_point_indices=indices.tolist(),
            exact_original_UV_sigma_actual_clock_and_finite_mask=True,complete_logical_columns=logical.tolist(),
            maximum_nonzero_columns_per_scalar_row=int(np.count_nonzero(columns,axis=1).max()),
            dynamic_derivatives_match_actual_programme_chain=True,original_six_columns_bitexact=True,
            sorted_CSR_full_physical_column_mapping_exact=True,actual_native_camera_CSR_preserved=True,
            old_straight_operator_check=legacy_check,FD_checks=fd_checks,
            current_facets=bound['selected_course_facets'][positions].tolist(),native_residual_max=float(abs(residual).max()),audit=audit))
    # Move the existing printed material across both former straight boundaries.
    # Source-ray/facet geometry has no dependency on a print's station label.
    report=dict(schema='rinkspace.actual-existing-seventeen-continuous-perimeter-check.v1',
        original_points=17,source_scalar_rows_added=0,actual_complete_world_count=ng,block_checks=checks,
        actual_print_station_cross_region_checks=[],cross_region_check_complete=False,
        original_full_graph_and_profile_buffers_reused=True,
        complete_motion_J_or_normal_not_produced_by_this_check=True,accuracy_accepted=False)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    station=int(binder.current.parameter_names.index('Ad4_independent_station_center_m'))
    course,cq,_=binder.current.print_course_binding(world)
    perimeter=np.asarray(course.sample_perimeter(cq,with_jacobian=False)[0]);period=float(np.linalg.norm(np.roll(perimeter,-1,axis=0)-perimeter,axis=1).sum())*binder.scene_unit_m
    crossings=[]
    for fraction in (-.125,.125,.375,.625,.875,1.125):
        trial=world.copy();trial[station]=period*fraction;binder._q(trial)
        current_bound=existing._build_current_binding(binder,trial)
        if not np.array_equal(bound['planes'],current_bound['planes']):raise ValueError('Print station still restricts original current perimeter rays')
        crossings.append(dict(Ad4_station_m=float(trial[station]),fraction_of_full_period=fraction,current_source_planes_bitexact=True))
    report.update(actual_print_station_cross_region_checks=crossings,cross_region_check_complete=True)
    return report

class StructuredProgrammeBlock(CourseMountOpticsBlock):
    def set_shared_geometry(self,geometry,indices):
        indices=np.asarray(indices,int);chart=self.optics_chart.coordinate_chart
        if type(geometry)is not StructuredProgrammeGeometry or not np.array_equal(indices,chart.world_indices)or len(indices)!=geometry.parameter_count:
            raise ValueError('Explicit qualified physical programme material owner and complete declared slots required')
        self.geometry=geometry;self.geometry_parameter_indices=indices.copy();self.geometry_parameter_count=geometry.parameter_count
    def synchronize_course(self,q):
        q=self.optics_chart.coordinate_chart.core_q(q);world=q[self.geometry_parameter_indices];key=world.tobytes()
        if self._course_current_geometry_key==key:return
        self.geometry.validate_parameters(world)
        course,cq,_=self.geometry.print_course_binding(world);bottom,top,*_=course.sample_perimeter(cq,with_jacobian=False)
        self.bundle.scene['ice_boundary']['points_xyz']=bottom.tolist();self.bundle.scene['board_surface']['top_edge_xyz']=top.tolist()
        if len(self.course_board_indices):
            bound=self.geometry.finite_board_records(self.course_original_board_records,world,original_planes=self.course_original_board_planes)
            full_plane_j=np.asarray(bound['plane_jacobian'],float)
            if full_plane_j.shape!=(len(self.course_board_indices),4,self.geometry_parameter_count):
                raise ValueError('Every current finite plane needs the complete logical-world derivative chain')
            active=np.flatnonzero(np.any(full_plane_j!=0.,axis=(0,1)))
            logical=np.unique(np.r_[np.arange(44,50),active]).astype(int)
            self.course_current_motion_geometry_logical_columns=logical
            self.course_current_motion_geometry_parameter_indices=self.geometry_parameter_indices[logical]
            self.bundle.finite_planes[self.course_board_indices]=bound['planes']
            for position,record in zip(self.course_board_record_positions,bound['records']):self.bundle.finite_records[int(position)]=copy.deepcopy(record)
            self.course_current_plane_values=bound['planes'];self.course_current_plane_jacobian=full_plane_j[:,:,logical]
            self.structured_current_extent_jacobian=bound['extent_world_jacobian']
        self._course_current_geometry_key=key
    def local(self,q):
        # Actor profile, cache, finite extents and native rows share current geometry.
        self.synchronize_course(q)
        return SecondaryOpticsFixedXYZRinkBlock.local(self,q)
    def camera_local(self,q):
        """The same physical camera chart without updating unused motion rows.

        Native append factors use their own current geometry producers. Keep
        the full physical-world domain check, while leaving programme motion
        synchronization on local() for its original consumers.
        """
        core=self.optics_chart.coordinate_chart.core_q(q)
        world=core[self.geometry_parameter_indices];key=world.tobytes()
        checked=getattr(self,'_native_camera_valid_worlds',None)
        if checked is None:
            from collections import OrderedDict
            checked=self._native_camera_valid_worlds=OrderedDict()
        if key not in checked:
            self.geometry.validate_parameters(world)
            checked[key]=True
            while len(checked)>3:checked.popitem(last=False)
        return SecondaryOpticsFixedXYZRinkBlock.local(self,q)
    def evaluate_camera_geometry(self,q,jacobian=True):
        """Same native camera scatter; inactive fields certified per material."""
        local=self.local(q);world=self.geometry_values(q);ng=self.geometry_parameter_count
        self.bundle.abs_groups=[(p,[self.geometry.rebind(r,world)for r in rows])for p,rows in self.groups]
        try:
            if not jacobian:return self.bundle.evaluate(local,False)
            residual,localj=self.bundle.evaluate(local,True)
            count=sum(2*len(r['uv'])for _,rows in self.groups for r in rows);geo=np.zeros((count,ng));cursor=0
            h=float(os.environ.get('RINKSPACE_GEOMETRY_FD_STEP_M','1e-7'))
            checked=getattr(self,'_inactive_material_certificates',None)
            if checked is None:checked=self._inactive_material_certificates=set()
            pool=getattr(self,'_absolute_geometry_fd_pool',None)
            if pool is not None:
                groups=((p,rows,self.bundle.camera(local,p))for p,rows in self.groups)
                for result in pool.ordered_groups(groups,world,self.absolute,h,batched_projection=True):
                    span=result['geo'];size=len(span);geo[cursor:cursor+size]=span;cursor+=size
                if cursor!=count:raise ValueError('Parallel original absolute geometry span omitted source rows')
            else:
                from .batched_absolute_geometry_fd import evaluate_geometry_group
                for p,rows in self.groups:
                    camera=self.bundle.camera(local,p)
                    span=evaluate_geometry_group(self.geometry,world,camera,rows,self.absolute,h,
                        certificates=checked,batched_projection=False)
                    size=len(span);geo[cursor:cursor+size]=span;cursor+=size
            matrix=scatter_fixed_xyz_source_csr(localj,self.bundle.source_residual_rows,geo,
                self.bundle.motion_rows,self.geometry_parameter_indices,self.xyz_slots,self.start,len(q))
            return residual,matrix
        finally:self.bundle.abs_groups=self.groups
    def evaluate(self,q,jacobian=True):
        profiling=jacobian and not getattr(self,'_actual_first_source_profiled',False)and bool(os.environ.get('RINKSPACE_SOURCE_RUNTIME_DIR'))
        if not profiling:return self._evaluate(q,jacobian)
        profile=cProfile.Profile();begin=time.perf_counter();profile.enable()
        try:result=self._evaluate(q,jacobian)
        finally:profile.disable()
        self._actual_first_source_profiled=True
        hot=[]
        for(file,line,name),(primitive,calls,total,cumulative,_)in pstats.Stats(profile).stats.items():
            if any(token in(file+' '+name).lower()for token in('native','scatter','station','rebind','material','validate','projection','source_scene')):
                hot.append(dict(file=file,line=line,function=name,calls=calls,self_seconds=total,cumulative_seconds=cumulative))
        from z190_motion_native import metadata as motion_metadata
        from z190_csr_scatter_native import metadata as scatter_metadata
        owners=(self.geometry.current.print_course_binding(np.asarray(q)[self.geometry_parameter_indices])[0],self.geometry.source_course.course)
        caches=[owner.station.statistics()for owner in owners if callable(getattr(owner.station,'statistics',None))]
        report=dict(block=self.name,pid=os.getpid(),jacobian=True,actual_rows=len(result[0]),actual_nnz=result[1].nnz,
            wall_seconds=time.perf_counter()-begin,q_sha256=hashlib.sha256(np.asarray(q).tobytes()).hexdigest(),
            native_motion=motion_metadata(),native_scatter=scatter_metadata(),actual_course_station_caches=caches,
            actual_inactive_material_certificates=len(getattr(self,'_inactive_material_certificates',())),
            actual_geometry_validation_cache_entries=len(getattr(self.geometry,'_successful_world_validation',{})),
            source_ice_material_view=None if self.geometry.__dict__.get('_source_ice_material_view')is None else dict(
                cache_bytes=self.geometry._source_ice_material_view.cache_bytes,
                maximum_bytes=self.geometry._source_ice_material_view.maximum,
                entries=len(self.geometry._source_ice_material_view.cache),
                hits=self.geometry._source_ice_material_view.hits,misses=self.geometry._source_ice_material_view.misses),
            actual_hot_calls=sorted(hot,key=lambda x:x['cumulative_seconds'],reverse=True),accuracy_accepted=False)
        Path(os.environ['RINKSPACE_SOURCE_RUNTIME_DIR'],self.name+'_first_actual_normal.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        return result
    def _evaluate(self,q,jacobian=True):
        result=self.evaluate_camera_geometry(q,jacobian)
        if not jacobian:return result
        residual,matrix=result;local=self.local(q);cal=self.bundle.current_calibration(local);n=self.bundle.source_residual_rows
        lens=np.zeros((n,3));mount=np.zeros((n,3));raw=np.asarray(self.motion_columns_provider(self.bundle,local,cal,self.bundle.profile_weights,steps=self.optics_chart.physical_steps,physical_column_scales=PHYSICAL_COLUMN_SCALES),float)
        if raw.shape!=(self.bundle.motion_rows,3):raise ValueError('Original complete native motion optics source rows required')
        np.divide(raw,self.bundle.motion_sigma[:,None],out=lens[:self.bundle.motion_rows]);np.divide(motion_mount_columns(self.bundle,local,cal),self.bundle.motion_sigma[:,None],out=mount[:self.bundle.motion_rows])
        cursor=self.bundle.motion_rows;world=self.geometry_values(q)
        for p,rows in self.groups:
            cam=self.bundle.camera(local,p);moved=[self.geometry.rebind(r,world)for r in rows];_,raw=self.absolute(cam,moved,True);raw=np.asarray(raw,float);size=sum(2*len(r['uv'])for r in rows)
            if raw.shape!=(size,15):raise ValueError('Original sigma-normalized complete camera15 absolute chain required')
            lens[cursor:cursor+size]=raw[:,[8,9,11]]*.001;mount[cursor:cursor+size]=raw[:,:3]@cv2.Rodrigues(np.asarray(cam[3:6]))[0].T/(100.*SCENE_UNIT_M);cursor+=size
        if cursor!=n:raise ValueError('Original absolute inventory changed')
        # The generic insertion is identical storage logic for old lens/mount spans.
        matrix=insert_mount_csr(matrix,lens,self.optics_chart.indices,34204)
        matrix=insert_mount_csr(matrix,mount,self.optics_chart.mount_indices,34195)
        rows,values=self.course_motion_columns(local,cal)
        columns=getattr(self,'course_current_motion_geometry_parameter_indices',COURSE_GLOBAL)
        return residual,insert_selected_motion_geometry_csr(matrix,rows,values,motion_rows=self.bundle.motion_rows,geometry_columns=columns)

def clone_structured_programme_block(old,binder):
    """Prepare one fresh owner before actors; no source state commit by this function."""
    if type(old)is not HockeyMountOpticsBlock or old.optics_chart.hockey_parameter_count!=20 or old.name not in BOARD_COUNTS:
        raise TypeError('Actual original hockey43 programme block required')
    new=StructuredProgrammeBlock.__new__(StructuredProgrammeBlock);new.__dict__.update(vars(old));new.bundle=copy.copy(old.bundle)
    new.bundle.finite_planes=old.bundle.finite_planes.copy();new.bundle.finite_mask=old.bundle.finite_mask.copy();new.bundle.finite_records=copy.deepcopy(old.bundle.finite_records)
    scene=binder.base if old.bundle.scene is None else old.bundle.scene;new.bundle.scene=dict(scene,ice_boundary=dict(scene['ice_boundary']),board_surface=dict(scene['board_surface']))
    new.bundle.priors=lambda local,cache,b=new.bundle:tuple(x[1:]for x in MovingPivotSourceBundle.priors(b,local,cache))
    if old.prepared is not None:new.prepared=copy.copy(old.prepared);new.prepared.bundle=new.bundle
    new.optics_chart=StructuredProgrammeMountOpticsChart.from_original(old.optics_chart,additional_world_parameter_count=binder.parameter_count-66)
    new.set_shared_geometry(binder,new.optics_chart.coordinate_chart.world_indices)
    positions=sorted((i for i,r in enumerate(new.bundle.finite_records)if r['surface_kind']=='finite_board_face'),key=lambda i:int(new.bundle.finite_records[i]['global_bundle_index']))
    records=[copy.deepcopy(new.bundle.finite_records[i])for i in positions];indices=np.array([r['global_bundle_index']for r in records],int)
    if len(indices)!=BOARD_COUNTS[old.name] or int(new.bundle.finite_mask.sum())!=FINITE_COUNTS[old.name] or len(set(indices))!=len(indices):raise ValueError('Actual original finite490/board17 source inventory changed')
    new.course_board_indices=indices;new.course_board_record_positions=np.asarray(positions,int);new.course_original_board_records=records;new.course_original_board_planes=new.bundle.finite_planes[indices].copy();new.course_current_plane_values=new.course_original_board_planes.copy();new.course_current_plane_jacobian=np.zeros((len(indices),4,6))
    new._course_current_geometry_key=new._course_camera_cache_key=None;new._course_camera_cache=None
    absolute=old.bundle.source_residual_rows-old.bundle.motion_rows;new.actor_output_capacity=dict(old.actor_output_capacity,nnz=old.actor_output_capacity['nnz']+(binder.parameter_count-43)*absolute+12*len(indices))
    return new

def install_structured_programme_blocks(state,geometry66,old_full_q,*,inherited_profile_token):
    """Install only six programme blocks/held geometry on a fresh old43 graph.

    Caller must rebuild patterns, external, cross, wide/high and reference
    adapters on the returned full34296 chart before declaring a full source.
    """
    if getattr(state,'source_actors',None)is not None or hasattr(state,'structured_programme_contract'):
        raise ValueError('Fresh original graph before owned actors required')
    if np.shape(state.q)!=(OLD.core_count,) or not np.array_equal(state.geometry_parameter_indices,OLD.world_indices) or not isinstance(inherited_profile_token,str) or len(inherited_profile_token)!=64:
        raise ValueError('Authenticated current programme core34245/world43 and original profile token required')
    chart=StructuredProgrammeCoordinateChart(geometry66.parameter_count-66);old_full=OLD.full_q(old_full_q)
    if not np.array_equal(state.q,old_full[:OLD.core_count]):raise ValueError('Actual initial graph and full parent core differ')
    binder=StructuredProgrammeGeometry(geometry66,state.geometry)
    for old in state.blocks:
        for p,rows in old.groups:
            for row in rows:binder.register_row(row,actual_pts=p)
    for groups in state.held.values():
        for p,rows in groups:
            for row in rows:binder.register_row(row,actual_pts=p,allow_previously_qualified_held_ad2_representation=True)
    full=chart.initialize_full_q(old_full,binder.initial);core=full[:chart.core_count];newblocks=[clone_structured_programme_block(old,binder)for old in state.blocks]
    for b in newblocks:b.synchronize_course(core)
    replacements={id(old):new for old,new in zip(state.blocks,newblocks)};bypts={p:replacements[id(b)]for p,b in state.bypts.items()}
    contract=dict(schema=SCHEMA,coordinate_chart=chart.manifest(),world_geometry=binder.manifest(binder.initial),programme_blocks=6,finite_source_counts=[FINITE_COUNTS[b.name]for b in newblocks],finite_board_counts={b.name:len(b.course_board_indices)for b in newblocks},profile_source_token=inherited_profile_token,original_profile_buffers_restored_not_reweighted=True,new_geometry_requires_profile_update_before_normal=True,
        original_programme_UV_sigma_roles_quad_scales_and_operators_preserved=True,complete_source_installed=False,
        caller_source_routes=['all_six_original_programme_blocks','patterns_current_course_and_currentAd12_13_25_26',
            'external_inverse_anchor_and_forward_currentAd14','direct_cross_current_world_camera',
            'original_wide45_26_and_two_correlated_user_top_strokes_same_world'],
        additional_standard_geometry_reference_rows=0,full_source_normal_validation_pending=True,accuracy_accepted=False)
    state.blocks=newblocks;state.bypts=bypts;state.geometry=binder;state.geometry_parameter_indices=np.asarray(chart.world_indices,int);state.q=core.copy();state.installation_optics={b.bundle.calibration.installation_id:b.optics_chart for b in newblocks};state.structured_programme_contract=contract;state.structured_programme_coordinate_chart=chart
    for key,value in list(vars(state).items()):
        if id(value)in replacements:setattr(state,key,replacements[id(value)])
    return contract,full
