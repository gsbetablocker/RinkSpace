"""Current S01 CPU/GPU block assembly, with one physical model/operator.

No old solver is imported. Whole W factors stay intact. Static world loci,
shared material landmarks, source Query alpha and formal RS rows have separate
owners and derivative chains. This module assembles; run_solver owns logging,
visible publication and explicitly bounded numerical chunks.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib,json,time
import numpy as np
from .parameterization import S01Parameterization
from .native_backend import BatchLayout,RowCombinationPlan
from .constrained import ConstraintSet,ConstraintFamily,ConstrainedProblem,AugmentedLagrangian
from .gpu_projector import CudaProjectionBatch
from .gpu_board import CudaBoardBatch
from .whitening import WhiteningJacobianPlan


IMPLEMENTED_CONSTRAINTS={
 'sequence.single_shot':'S01 alone; no next-shot solve or visual acceptance claim',
 'clock.native':'native PTS/timebase; row-auxiliary rolling-shutter equality',
 'source.material_only':'current admitted static material observations only',
 'source.line_midline':'current prepared source definitions retain material midline semantics',
 'world.common_scene':'one shared scene state across this shot and all observations',
 'board.vertical':'current continuous board generator has vertical material coordinate',
 'curling.centres':'current shared structural-family generator',
 'curling.radii':'current shared structural-family radius parameterization',
 'lane.width':'current shared structural-family common width parameterization',
 'hockey.centre_blue':'current shared structural-family generator',
 'rig.installations':'current fixed-pivot, constant camera-local lever family',
 'rig.nonzero_offset':'free camera-local lever block; no zero-offset assumption',
 'rig.independent_heights':'independent camera pivot height in current shared state',
 'rig.zoom_offset':'constant-lever target family has no extended pupil/axis-gap blocks',
 'numerical.gauge':'exact reference spline angle dependency; native minimal axis charts',
 'compute.resources':'resident CUDA projection/Jacobian plus explicit CPU-thread policy',
 'observation.fixed_vs_locus':'fixed points, shared material-point positions, local loci and Query alpha distinct',
 'observation.intersections':'five shared C stations, eight exposure rows; no per-exposure point station',
 'ad.object_model':'one static similarity block per current artwork object; sampled vertices are not free',
 'numerical.intersection_budget':'explicit current radius engineering inequality; not sigma',
 'numerical.shared_budget_schedule':'one global b: feasibility minimize b, then b=1 statistical phase',
 'numerical.ad7_fixed_corner_budget':'24 static fixed-template-corner source inequalities; no per-exposure station',
 'numerical.s01_block_warm_start':'explicit ice/board geometry, fixed-camera artwork, then full joint numerical stage; original admission and W unchanged',
}


class ForwardDomainRejection(ValueError):pass


class S01Optimizer:
    def __init__(self,model,prepared,backend,*,constraints_path,whitening_manifest,num_threads,geometry_difference_step,axis_difference_step,row_equality_scale_px,query_equality_scale_px=1.,phase="feasibility",initial_vector=None,initial_values_by_block=None,initialize_loci=True,free_mode="all",domain_event_callback=None,level_set_budget_sq=None,metric_event_callback=None):
        if type(num_threads)is not int or num_threads<=0:raise ValueError('explicit positive CPU thread count required')
        if min(geometry_difference_step,axis_difference_step,row_equality_scale_px)<=0:raise ValueError('explicit positive numerical scales required')
        if (prepared.manifest['model_id'],prepared.manifest['prepared_against_model_revision'])!=(model.id,model.revision):raise ValueError('current prepared geometry binding differs from current model structure; rebind affected geometry only')
        if prepared.manifest.get('noise_contract_revision')!=2:raise ValueError('current input must deliver corrected raw-UV W, all-source engineering E and separate Query equalities')
        self.backend,self.prepared=backend,prepared;self.num_threads=num_threads
        self.domain_event_callback=domain_event_callback;self.domain_rejection_index=0
        self.geometry_step=float(geometry_difference_step);self.axis_step=float(axis_difference_step)
        self.phase=phase;self.free_mode=free_mode
        self.level_set_budget_sq=level_set_budget_sq
        self.metric_event_callback=metric_event_callback;self.last_inner_metric=None
        if free_mode not in ("all","camera_scene","ad_layouts"):raise ValueError("unknown explicit variable-stage selection")
        if phase not in ("feasibility","statistics","statistical_level_set"):raise ValueError("unknown explicit solve phase")
        self.constraints_path=Path(constraints_path).resolve()
        self.constraint_snapshot=prepared.assert_current_constraint_snapshot(self.constraints_path)
        if not self.constraint_snapshot.get('bound'):raise ValueError('current prepared engineering target needs an explicit current binding')
        ledger=json.loads(self.constraints_path.read_text(encoding='utf-8-sig'));self.ledger_revision=ledger['revision']
        self.ledger_by_id={v['id']:v for v in ledger['constraints']}
        self.constraint_bindings=[dict({k:v.get(k) for k in ('id','enabled','approval','expression','scope')},implementation=getattr(self,'constraint_implementations',IMPLEMENTED_CONSTRAINTS)[v['id']])
            for v in ledger['constraints'] if v['id'] in getattr(self,'constraint_implementations',IMPLEMENTED_CONSTRAINTS) and not (prepared.manifest.get('numerical_data_scope')=='ice_board' and v['id'] in ('ad.object_model','numerical.ad7_fixed_corner_budget'))]
        self.constraint_semantic_hash=hashlib.sha256(json.dumps(self.constraint_bindings,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()
        self.intersection_policy=self.ledger_by_id['numerical.intersection_budget']
        self.ad7_corner_policy=self.ledger_by_id.get('numerical.ad7_fixed_corner_budget',{'enabled':False})
        schedule=self.ledger_by_id['numerical.shared_budget_schedule']
        if not schedule['enabled']:raise ValueError('shared-budget schedule is disabled; an explicit alternative numerical policy is required')
        expression=schedule['expression']
        if not expression.get('one_shared_budget_for_all_active_sources') or expression['phase_I']['objective']!='min b':raise ValueError('unsupported changed budget schedule; do not silently use old semantics')
        self.target_budget_sq=float(expression['phase_II']['fixed_budget_squared'])
        if not np.isfinite(self.target_budget_sq) or self.target_budget_sq<=0:raise ValueError('explicit fixed budget squared must be positive')
        if phase=='statistical_level_set':
            level=self.ledger_by_id['numerical.s01_block_warm_start']['expression'].get('statistical_level_set',{})
            value=float(level_set_budget_sq) if level_set_budget_sq is not None else np.nan
            if not level.get('enabled') or not np.isfinite(value) or value<self.target_budget_sq or float(level.get('fixed_budget_squared',np.nan))!=value:
                raise ValueError('statistical level-set stage requires the explicitly registered current fixed b and scope')
            self.level_set_budget_sq=value
        self.params=getattr(self,'parameterization_class',S01Parameterization)(model,prepared,intersection_enabled=self.intersection_policy['enabled'],ad7_corners_enabled=self.ad7_corner_policy['enabled'])
        p=self.params;self.model=model
        if initial_vector is not None:
            value=np.asarray(initial_vector,dtype=np.float64)
            if value.shape!=p.initial.shape or not np.isfinite(value).all():raise ValueError('restored native initialization differs from current complete layout')
            p.initial[:]=value
        if initial_values_by_block:
            for b,identifier in enumerate(p.block_ids):
                value=initial_values_by_block.get(identifier)
                if value is not None and p.block_owners[b]['kind']=='current_source_nuisance_group':
                    value=np.asarray(value,dtype=np.float64)
                    if value.shape==(p.block_sizes[b],):p.initial[p.parameter_offsets[b]:p.parameter_offsets[b+1]]=value
        # Current preparation improved only the five shared station seeds. Keep
        # them as visible initialization, without creating duplicate variables.
        for record in prepared.manifest.get('new_intersection_landmarks',[]):
            key=record['station_variable_block_id']
            if initial_vector is None and initial_values_by_block is None and model.state.digest()==prepared.manifest.get('current_model_state_initialization_sha256') and key in p.semantic_to_native:
                b=p.semantic_to_native[key];p.initial[p.parameter_offsets[b]]=record['station_seed']
        self._last_x=None;self._jacobian_ready=False;self._cache=None;self.last_domain_failure=None
        self.counters={'physical_points':p.count,'forward_calls':0,'full_linearizations':0,'domain_rejections':0}
        self.timings={};self.pool=ThreadPoolExecutor(max_workers=num_threads,thread_name_prefix='rinksolve_geometry')
        self.Jprojection=np.zeros(p.raw_layout.jacobian_value_count,dtype=np.float64)
        self.Jerror=np.empty_like(self.Jprojection)
        self._local_offsets=np.empty(p.count,dtype=np.int64)
        for b in np.unique(p.local_block):
            locations=p.block_row_locations[int(b)];self._local_offsets[locations[:,0]]=locations[:,1]
        self._world_plans={}
        for key in [p.board_key,*p.scene_keys]:
            if key not in p.semantic_to_native:continue
            b=p.semantic_to_native[key];locations=p.block_row_locations.get(b)
            if locations is None:continue
            rr=locations[:,0];address=np.full(p.count,-1,dtype=np.int64);address[rr]=np.arange(len(rr))
            plan=[]
            for curve,all_rows in p.world_groups.items():
                sub=all_rows[address[all_rows]>=0]
                if len(sub):plan.append((curve,sub,address[sub]))
            self._world_plans[key]=(b,rr,plan)
        factors=prepared.manifest['statistical_factors'];stat_blocks=[]
        for f in factors:
            active=p.source_to_active[np.asarray(f['rows'],dtype=np.int64)]
            if np.any(active<0):raise ValueError('original complete W factor lost a current source member')
            blocks=set()
            for r in active:blocks.update(p.raw_layout.factor_blocks[r])
            stat_blocks.append(tuple(sorted(blocks)))
        self.stat_layout=BatchLayout(p.block_ids,p.block_sizes,tuple(f['id'] for f in factors),tuple(f['residual_count'] for f in factors),tuple(stat_blocks))
        self.whitening=WhiteningJacobianPlan(prepared,p.raw_layout,self.stat_layout,p.raw_observation_rows,manifest=whitening_manifest)
        ready=np.asarray(prepared.arrays['budget_ready_rows'],dtype=np.int64)
        self.budget_rows=p.source_to_active[ready]
        if np.any(self.budget_rows<0):raise ValueError('engineering budget support not present in current batch')
        self.budgets=np.array(prepared.arrays['engineering_budget_px'][ready],dtype=np.float64)
        self.original_budget_count=len(self.budget_rows)
        self.intersection_budget_count=0;self.ad7_corner_budget_count=0
        extra=np.arange(p.original_count,p.count,dtype=np.int64)
        if len(extra):
            radii=[]
            for row in extra:
                policy=self.intersection_policy if p.kinds[row]==3 else self.ad7_corner_policy
                radii.append(float(policy['expression']['radius_px']))
            self.budget_rows=np.r_[self.budget_rows,extra];self.budgets=np.r_[self.budgets,radii]
            self.intersection_budget_count=int(np.sum(p.kinds[extra]==3));self.ad7_corner_budget_count=int(np.sum(p.kinds[extra]==4))
        if not np.isfinite(self.budgets).all() or np.any(self.budgets<=0):raise ValueError('current pixel budget invalid')
        self.profile_rows=np.flatnonzero(p.kinds==2)
        self.source_normals=np.asarray(prepared.arrays['profile_normal'],dtype=np.float64)
        if self.source_normals.shape!=(prepared.row_count,2):raise ValueError('current original profile normals missing')
        budget_blocks=tuple(p.raw_layout.factor_blocks[i]+(p.budget_block,) for i in self.budget_rows)
        self.budget_layout=BatchLayout(p.block_ids,p.block_sizes,tuple('budget/'+p.observation_ids[i] for i in self.budget_rows),(1,)*len(self.budget_rows),budget_blocks)
        self.roweq_layout=BatchLayout(p.block_ids,p.block_sizes,tuple('roweq/'+v for v in p.observation_ids),(1,)*p.count,p.raw_layout.factor_blocks)
        self.query_layout=BatchLayout(p.block_ids,p.block_sizes,tuple('query_eq/'+p.observation_ids[i] for i in self.profile_rows),(2,)*len(self.profile_rows),tuple(p.raw_layout.factor_blocks[i] for i in self.profile_rows)) if len(self.profile_rows) else None
        self.budget_plan=RowCombinationPlan(backend,p.raw_layout,self.budget_rows,appended_blocks=(p.budget_block,));self.budget_plan.check_output_layout(self.budget_layout)
        self.budget_direct=self.budget_plan.out_offsets[1:]-1
        self.budget_direct_values=-np.ones(len(self.budget_rows),dtype=np.float64)
        self.roweq_plan=RowCombinationPlan(backend,p.raw_layout,np.arange(p.count,dtype=np.int64));self.roweq_plan.check_output_layout(self.roweq_layout)
        self.roweq_direct=np.empty(p.count,dtype=np.int64)
        for i,blocks in enumerate(self.roweq_layout.factor_blocks):
            local=int(p.local_block[i]);edge=int(self.roweq_layout.factor_edge_offsets[i])+blocks.index(local)
            self.roweq_direct[i]=self.roweq_layout.jacobian_offsets[edge]+p.row_column[i]
        self._ones=np.ones(p.count,dtype=np.float64)
        query_indices=[]
        for row in self.profile_rows:
            lo=p.raw_layout.jacobian_offsets[p.raw_layout.factor_edge_offsets[row]]
            hi=p.raw_layout.jacobian_offsets[p.raw_layout.factor_edge_offsets[row+1]]
            query_indices.append(np.arange(lo,hi,dtype=np.int64))
        self.query_jacobian_indices=np.concatenate(query_indices) if query_indices else np.empty(0,dtype=np.int64)
        constraint_layout=BatchLayout(p.block_ids,p.block_sizes,self.budget_layout.factor_ids+self.roweq_layout.factor_ids+(self.query_layout.factor_ids if self.query_layout else ()),
            self.budget_layout.residual_sizes+self.roweq_layout.residual_sizes+(self.query_layout.residual_sizes if self.query_layout else ()),
            self.budget_layout.factor_blocks+self.roweq_layout.factor_blocks+(self.query_layout.factor_blocks if self.query_layout else ()))
        nb=len(self.budget_rows);families=[]
        if self.original_budget_count:
            families.append(ConstraintFamily('all_current_source_engineering_E','less_equal',tuple(range(self.original_budget_count)),'dimensionless',1.,
                'all active original sources; ordinary squared raw-pixel norm or exact source-profile-normal component / E^2; one shared b; not sigma',('source.current_engineering_budget','numerical.shared_budget_schedule')))
        if self.intersection_budget_count:
            families.append(ConstraintFamily('new_fixed_intersection_engineering_budget','less_equal',tuple(range(self.original_budget_count,self.original_budget_count+self.intersection_budget_count)),'dimensionless',1.,
                'current editable programme engineering target radius_px with one shared b; not measurement sigma',('numerical.intersection_budget','numerical.shared_budget_schedule')))
        if self.ad7_corner_budget_count:
            families.append(ConstraintFamily('ad7_fixed_template_corner_engineering_budget','less_equal',tuple(range(self.original_budget_count+self.intersection_budget_count,nb)),'dimensionless',1.,
                'current editable static template-corner engineering target radius_px; one shared b; sigma unknown',('numerical.ad7_fixed_corner_budget','numerical.shared_budget_schedule')))
        families.append(ConstraintFamily('formal_predicted_row_equality','equality',tuple(range(nb,nb+p.count)),'native_pixel',float(row_equality_scale_px),
            'explicit equality normalization, not measurement tolerance',('clock.native','numerical.gauge')))
        if len(self.profile_rows):families.append(ConstraintFamily('source_query_exact_pixel_equality','equality',tuple(range(nb+p.count,nb+p.count+len(self.profile_rows))),'native_pixel',float(query_equality_scale_px),
            'delivered rational Q(alpha); two exact equalities; alpha never enters raw statistical error',('source.current_query','numerical.shared_budget_schedule')))
        self.constraint_set=ConstraintSet(constraint_layout,tuple(families))
        self.base_identity={'model_id':model.id,'model_revision':model.revision,'initial_parameter_state_id':model.state.id,'initial_parameter_sha256':model.state.digest(),
            'coordinate_system_id':model.scene.coordinate_system_id,'prepared_directory':str(prepared.directory),'prepared_schema':prepared.manifest['schema'],
            'constraint_semantic_sha256':self.constraint_semantic_hash,'selection_revision':prepared.manifest.get('selection_revision',0),'prepared_target_sha256':prepared.manifest['constraints_binding']['relevant_sha256'],
            'physical_operator':'rinksolve.physical_projector.v1','native_parameter_layout':'rinksolve.s01_native_parameters.v1','numerical_data_scope':prepared.manifest.get('numerical_data_scope','all'),'free_mode':free_mode}
        self._configure_phase(phase)
        stations,rows,alpha=p.auxiliaries(p.initial,p.model_for(p.initial));initial_model=p.model_for(p.initial)
        xyz,_=self._world(initial_model,stations,need_station=False)
        op=initial_model.operator('S01');times=p.frame_times+op.camera_definition['signed_readout_seconds']*(rows/(p.rig.image_height-1)-.5)
        self.gpu=CudaProjectionBatch(op,np.ascontiguousarray(xyz),np.ascontiguousarray(times))
        self.initialization_repairs=[]
        if initialize_loci:
            self._repair_initial_locus_domains()
            self._reseed_current_loci()
        initial_cache=self._compute(p.initial,False)
        if phase=="feasibility":p.initial[p.budget_offset]=max(self.target_budget_sq,float(self._budget_quantities(initial_cache)[0].max()))
        elif phase=='statistical_level_set':p.initial[p.budget_offset]=self.level_set_budget_sq
        else:p.initial[p.budget_offset]=self.target_budget_sq



    def _configure_phase(self,phase):
        p=self.params;self.phase=phase;constant=np.zeros(len(p.block_ids),dtype=np.uint8)
        if self.free_mode=='ad_layouts':
            constant.fill(1)
            for b,owner in enumerate(p.block_owners):
                if owner.get('semantic_block_id','').startswith('artwork_group/'):constant[b]=0
            ad_rows=np.array([bool(p.curve_group_keys[str(curve)]) for curve in p.curve_ids])
            constant[np.unique(p.local_block[ad_rows])]=0
            constant[p.budget_block]=0
        elif self.free_mode=='camera_scene':
            for b,owner in enumerate(p.block_owners):
                if owner.get('semantic_block_id','').startswith('artwork_group/'):constant[b]=1
        if phase=='feasibility':
            objective=BatchLayout(p.block_ids,p.block_sizes,('numerical/minimize_shared_budget_sq',),(1,),((p.budget_block,),))
        elif phase in ('statistics','statistical_level_set'):objective=self.stat_layout;constant[p.budget_block]=1
        else:raise ValueError('unknown solve phase')
        self.objective_layout=objective
        identity={**self.base_identity,'stage':phase,'statistical_error':'predUV-immutable_rawUV; no Query alpha'}
        if phase=='statistical_level_set':identity['fixed_budget_level_set_squared']=self.level_set_budget_sq
        self.problem=ConstrainedProblem(identity,
            p.block_ids,p.block_sizes,p.scales,objective,self.constraint_set,constant_blocks=constant,
            lower_bounds=p.lower,upper_bounds=p.upper,ordering_groups=p.ordering_groups)
        self.al=AugmentedLagrangian(self.backend,self.problem,self.evaluate_objective,self.evaluate_constraints,inner_scale_callback=self.full_uv_metric_scales)

    def full_uv_metric_scales(self,x):
        """Numerical support-RMS chart from all current 2D projection directions.

        No W, sigma, engineering E, active-set mask, or multiplier enters this
        metric. The existing Query-alpha tangent is included. Zero columns
        retain their declared units, and KKT diagnostics keep original units.
        """
        cache=self._compute(x,True);layout=self.params.raw_layout;j=self.Jerror
        squares=np.ascontiguousarray(j*j);nonzero=np.ascontiguousarray((j!=0).astype(np.float64));ones=np.ones(layout.residual_count)
        summed=self.backend.transpose_product(layout,squares,ones);counts=self.backend.transpose_product(layout,nonzero,ones)
        rms=np.sqrt(np.divide(summed,counts,out=np.zeros_like(summed),where=counts>0));declared=self.problem.parameter_scales
        effective=declared.copy();nonzero_rms=rms>0;effective[nonzero_rms]=np.minimum(declared[nonzero_rms],1/rms[nonzero_rms])
        if not np.isfinite(effective).all() or np.any(effective<=0) or np.any(effective>declared):raise ValueError('invalid full-UV support-RMS numerical chart')
        effective.setflags(write=False)
        record={'schema':'rinksolve.inner_numerical_metric.v1','implementation':'full_uv_unit_tangent_identity_lm','formula':'S_declared/max(1,RMS_nonzero_support(J_full_2D_UV)*S_declared)','source_native_parameters_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'source_physical_state_sha256':cache['model'].state.digest(),'declared_scales_sha256':hashlib.sha256(declared.tobytes()).hexdigest(),'parameter_scales_sha256':hashlib.sha256(effective.tobytes()).hexdigest(),'parameter_count':len(effective),'effective_scale_min':float(effective.min()),'effective_scale_max':float(effective.max()),'zero_derivative_columns_keep_declared_units':int(np.sum(counts==0)),'nonzero_support_components_min':int(counts[counts>0].min()) if np.any(counts>0) else 0,'nonzero_support_components_max':int(counts.max()),'raw_UV_components':layout.residual_count,'W_sigma_E_or_active_set_weighting':False,'query_alpha_existing_tangent_included':True,'all_DOF_preserved':True,'ambient_bounds_unchanged':True,'jacobi_scaling':False,'min_lm_diagonal':1.,'max_lm_diagonal':1.,'fixed_during_one_inner_solve':True,'KKT_units':'original_declared_parameter_scales'}
        if self.metric_event_callback is not None:
            extra=self.metric_event_callback(record,effective)
            if extra is not None:record.update(extra)
        self.last_inner_metric=record
        return effective

    def set_phase(self,phase,vector):
        x=np.array(vector,dtype=np.float64,copy=True)
        if phase=='statistics':x[self.params.budget_offset]=self.target_budget_sq
        if phase=='statistical_level_set':x[self.params.budget_offset]=self.level_set_budget_sq
        self._configure_phase(phase)
        return x

    def _budget_quantities(self,cache):
        error=cache['error'][self.budget_rows];q=np.sum(error*error,axis=1)/(self.budgets*self.budgets)
        coefficients=2*error/(self.budgets*self.budgets)[:,None]
        profile=self.params.kinds[self.budget_rows]==2
        if np.any(profile):
            original=self.params.raw_observation_rows[self.budget_rows[profile]];normal=self.source_normals[original]
            component=np.sum(normal*error[profile],axis=1)
            q[profile]=component*component/(self.budgets[profile]**2)
            coefficients[profile]=2*component[:,None]*normal/(self.budgets[profile]**2)[:,None]
        return q,np.ascontiguousarray(coefficients)

    def _repair_initial_locus_domains(self):
        """Repair a numerical seed inside its already-confirmed finite branch.

        No entity, UV, source clock, source domain or fixed landmark is changed.
        Reinitializing a variable locus is not a new physical observation.
        """
        from scipy.optimize import minimize_scalar
        p=self.params;trial=self.gpu.evaluate_uv();bad=np.flatnonzero(~trial['valid'])
        if not len(bad):return
        model=p.model_for(p.initial);op=model.operator('S01');stations,rows,alpha=p.auxiliaries(p.initial,model)
        tau=op.camera_definition['signed_readout_seconds'];h=p.rig.image_height-1
        for i in bad:
            if p.station_column[i]<0:continue
            curve=str(p.curve_ids[i]);time_value=p.frame_times[i]+tau*(rows[i]/h-.5);target=p.target_uv[i]
            if p.kinds[i]==2:target=p.profiles[int(p.raw_observation_rows[i])].evaluate(alpha[i])
            def score(s):
                xyz=model.scene.world_path(curve,np.array([s]),model.state)
                out=op.project_at(xyz,np.array([time_value]));uv=out['uv'][0]
                return float(np.sum((uv-target)**2)) if np.isfinite(uv).all() else np.inf
            result=minimize_scalar(score,bounds=(0.,1.),method='bounded',options={'xatol':np.finfo(np.float64).eps**.5})
            candidates=[(float(result.x),score(result.x)),(0.,score(0.)),(1.,score(1.))]
            station,value=min(candidates,key=lambda v:v[1])
            xyz=model.scene.world_path(curve,np.array([station]),model.state)
            check=op.project_at(xyz,np.array([time_value]))
            if not check['valid'][0]:continue
            offset=p.parameter_offsets[p.local_block[i]]+p.station_column[i]
            before=float(p.initial[offset]);p.initial[offset]=station
            self.initialization_repairs.append({'observation_id':p.observation_ids[i],'curve_id':curve,'parameter_kind':'observation_local_curve_locus_station',
                'before':before,'after':station,'after_squared_pixel_error':value,'reason':'current confirmed finite branch changed chart while inherited numerical seed left the current optical domain',
                'source_pixels_modified':False,'source_entity_binding_modified':False,'physical_landmark_moved_per_exposure':False,'initialization_is_measurement_or_fact':False})
        if self.initialization_repairs:
            model=p.model_for(p.initial);op=model.operator('S01');stations,rows,alpha=p.auxiliaries(p.initial,model)
            xyz,_=self._world(model,stations,need_station=False)
            times=p.frame_times+op.camera_definition['signed_readout_seconds']*(rows/h-.5)
            self.gpu.set_state(op);self.gpu.set_rows(xyz=np.ascontiguousarray(xyz),time=np.ascontiguousarray(times))

    def _seed_project(self,model,op,curve,stations,times,board,*,need_station=False):
        """GPU numerical initialization using the same two public operators."""
        n=len(stations);capacity=1<<(max(1,n)-1).bit_length()
        pool=getattr(self,'_seed_gpu_pool',None)
        if pool is None:self._seed_gpu_pool=pool={}
        if capacity not in pool:
            camera=CudaProjectionBatch(op,np.zeros((capacity,3),dtype=np.float64),np.zeros(capacity,dtype=np.float64),device=self.gpu.device)
            pool[capacity]=(camera,CudaBoardBatch(capacity,device=self.gpu.device))
        camera,geometry=pool[capacity]
        is_board=self.params.board_key in self.params.world_dependencies[curve]
        if is_board:
            if need_station:
                world=model.scene.unwrapped_path_jacobian(curve,stations,model.state,board=board);q=world['unwrapped'];dq=world['jacobian_station']
            else:q=model.scene.unwrapped_path(curve,stations,model.state,board=board)
            padded=np.empty((capacity,2),dtype=np.float64);padded[:n]=q;padded[n:]=q[-1]
            result=geometry.evaluate(board,padded);xyz=result['xyz'][0]
            if need_station:
                ds=result['jacobian_station'][0,:n]*dq[:,0,None];ds[:,2]+=dq[:,1]
        else:
            if need_station:
                world=model.scene.world_path_jacobian(curve,stations,model.state,board=board);value=world['xyz'];ds=world['jacobian_station']
            else:value=model.scene.world_path(curve,stations,model.state,board=board)
            xyz=np.empty((capacity,3),dtype=np.float64);xyz[:n]=value;xyz[n:]=value[-1]
        clock=np.empty(capacity,dtype=np.float64);clock[:n]=times;clock[n:]=times[-1]
        camera.set_state(op);camera.set_rows(xyz=np.ascontiguousarray(xyz),time=clock)
        value=camera.evaluate() if need_station else camera.evaluate_uv()
        out={key:array[:n] for key,array in value.items() if isinstance(array,np.ndarray) and array.shape[:1]==(capacity,)}
        if need_station:out['world_station_jacobian']=ds
        return out

    def _reseed_current_loci(self,vector=None):
        """Reinitialize only local coordinates on the current bound HD branches.

        Coarse global search followed by scalar projection Newton is an initial
        numerical guess, never an entity association or a fixed source fact.
        """
        p=self.params;working=p.initial if vector is None else np.array(vector,dtype=np.float64,copy=True)
        model=p.model_for(working);op=model.operator('S01')
        stations,rows,alpha=p.auxiliaries(working,model)
        board=model.scene.board(model.state)
        changed=[];tau=op.camera_definition['signed_readout_seconds'];height=p.rig.image_height-1
        for curve,rr0 in p.world_groups.items():
            c=p.curve_lookup[curve]
            rr=rr0[p.station_column[rr0]>=0]
            if self.free_mode=='ad_layouts' and not p.curve_group_keys[curve]:continue
            rr=rr[self.problem.constant_blocks[p.local_block[rr]]==0]
            if not len(rr):continue
            definition=model.scene.resolved_definition(c,model.state)
            vertices=np.asarray(definition.get('unwrapped_station_height',[]),dtype=np.float64)
            # Full finite-branch candidates must resolve both visible basins
            # of a grazing projected circle. Retain every polyline vertex.
            minimum=513 if definition['kind'] in ('ice_circle','board_boundary') else 33
            grid=np.linspace(0.,1.,minimum)
            if len(vertices)>1:grid=np.unique(np.r_[grid,np.linspace(0.,1.,len(vertices))])
            count=len(grid)
            times=p.frame_times[rr]+tau*(rows[rr]/height-.5)
            target=p.target_uv[rr].copy()
            for local,i in enumerate(rr):
                if p.kinds[i]==2:target[local]=p.profiles[int(p.raw_observation_rows[i])].evaluate(alpha[i])
            before=stations[rr].copy();best=before.copy();score=np.full(len(rr),np.inf)
            for begin in range(0,len(rr),max(1,250000//count)):
                end=min(len(rr),begin+max(1,250000//count));take=end-begin
                s=np.tile(grid,take)
                forward=self._seed_project(model,op,curve,s,np.repeat(times[begin:end],count),board)
                errors=np.sum((forward['uv'].reshape(take,count,2)-target[begin:end,None,:])**2,axis=2)
                errors[~forward['valid'].reshape(take,count)]=np.inf
                index=np.argmin(errors,axis=1);best[begin:end]=grid[index];score[begin:end]=errors[np.arange(take),index]
            refinements=0
            mantissa_iterations=np.finfo(np.float64).nmant+1
            for _ in range(mantissa_iterations):
                forward=self._seed_project(model,op,curve,best,times,board,need_station=True)
                derivative=np.einsum('nij,nj->ni',forward['jacobian_world'],forward['world_station_jacobian'])
                error=forward['uv']-target;den=np.sum(derivative*derivative,axis=1)
                step=np.divide(np.sum(error*derivative,axis=1),den,out=np.zeros_like(den),where=den>0)
                pending=np.flatnonzero(np.isfinite(step)&(np.abs(step)>16*np.finfo(float).eps));accepted_any=False
                for _ in range(mantissa_iterations):
                    if not len(pending):break
                    proposal=np.clip(best[pending]-step[pending],0.,1.)
                    trial=self._seed_project(model,op,curve,proposal,times[pending],board)
                    trial_score=np.sum((trial['uv']-target[pending])**2,axis=1)
                    improvement=score[pending]-trial_score
                    accept=trial['valid']&(improvement>64*np.finfo(float).eps*np.maximum(1.,score[pending]))
                    chosen=pending[accept];best[chosen]=proposal[accept];score[chosen]=trial_score[accept]
                    refinements+=len(chosen);accepted_any|=bool(len(chosen))
                    still=(~accept)&(proposal!=best[pending]);pending=pending[still];step[pending]*=.5
                if not accepted_any:break
            old=self._seed_project(model,op,curve,before,times,board)
            old_score=np.sum((old['uv']-target)**2,axis=1)
            use=score<old_score
            offsets=p.parameter_offsets[p.local_block[rr]]+p.station_column[rr]
            working[offsets[use]]=best[use]
            changed.append({'curve_id':curve,'observations':len(rr),'reseeded':int(np.sum(use)),
                'global_finite_branch_grid_count':count,'backtracked_refinement_accepts':refinements,'initial_squared_pixel_error_sum':float(old_score.sum()),'reseeded_squared_pixel_error_sum':float(np.where(use,score,old_score).sum()),
                'reason':'closest numerical seed on the current already-bound finite branch; inherited station chart is not a fixed observation',
                'source_pixels_modified':False,'source_entity_binding_modified':False,'initialization_is_measurement_or_fact':False})
        for camera,geometry in self._seed_gpu_pool.values():camera.close();geometry.close()
        self._seed_gpu_pool.clear()
        if changed:
            self.initialization_repairs.extend(changed);self._last_x=None
        return working,changed

    def _world(self,model,stations,*,need_station):
        p=self.params;xyz=np.empty((p.count,3));ds=np.empty_like(xyz) if need_station else None
        board=model.scene.board(model.state)
        accelerated=hasattr(self,'gpu') and hasattr(model.scene,'unwrapped_path_jacobian')
        if accelerated:
            native=p.semantic_to_native[p.board_key];board_rows=p.block_row_locations[native][:,0]
            if not hasattr(self,'board_gpu'):self.board_gpu=CudaBoardBatch(len(board_rows),device=self.gpu.device)
            address=np.full(p.count,-1,dtype=np.int64);address[board_rows]=np.arange(len(board_rows))
            q=np.empty((len(board_rows),2));dq=np.empty_like(q) if need_station else None
        for curve,rr in p.world_groups.items():
            if accelerated and p.board_key in p.world_dependencies[curve]:
                positions=address[rr]
                if need_station:
                    value=model.scene.unwrapped_path_jacobian(curve,stations[rr],model.state,board=board)
                    q[positions]=value['unwrapped'];dq[positions]=value['jacobian_station']
                else:q[positions]=model.scene.unwrapped_path(curve,stations[rr],model.state,board=board)
            elif need_station:
                value=model.scene.world_path_jacobian(curve,stations[rr],model.state,board=board)
                xyz[rr]=value['xyz'];ds[rr]=value['jacobian_station']
            else:xyz[rr]=model.scene.world_path(curve,stations[rr],model.state,board=board)
        if accelerated:
            result=self.board_gpu.evaluate(board,q)
            if not result['valid'].all():raise ForwardDomainRejection('current continuous-board GPU domain rejected rows')
            tangent=result['jacobian_station'][0];xyz[board_rows]=result['xyz'][0]
            if need_station:
                derivative=tangent*dq[:,0,None];derivative[:,2]+=dq[:,1];ds[board_rows]=derivative
            self._current_board_rows=board_rows;self._current_board_q=q;self._current_board_tangent=tangent;self._current_board_address=address
        for row,feature in p.fixed_template_landmarks.items():
            xyz[row]=model.scene.landmark_world(feature,model.state,board=board)
            if need_station:ds[row]=0.
        return xyz,ds

    def _world_subset(self,model,stations,plan,count):
        result=np.empty((count,3));board=model.scene.board(model.state)
        for curve,rows,positions in plan:result[positions]=model.scene.world_path(curve,stations[rows],model.state,board=board)
        return result

    def _write_block(self,b,jacobian):
        locations=self.params.block_row_locations[b];width=self.params.block_sizes[b]
        if jacobian.shape!=(len(locations),2,width):raise ValueError('physical block Jacobian shape mismatch')
        indices=locations[:,1,None]+np.arange(2*width,dtype=np.int64)
        self.Jprojection[indices.reshape(-1)]=np.ascontiguousarray(jacobian).reshape(-1)

    def _geometry_column(self,model,stations,xyz,key,column):
        b,rows,plan=self._world_plans[key];q=model.state.get(key);step=self.geometry_step*max(1.,abs(float(q[column])))
        base=xyz[rows];plus=minus=None;method='central_small_geometry_definition'
        def evaluate(values):
            state=model.state.updated({key:values})
            if key==self.params.board_key:
                board=model.scene.board(state);unwrapped=self._board_unwrapped.copy()
                for curve,rr,positions in plan:
                    definition=model.scene.resolved_definition(self.params.curve_lookup[curve],state)
                    if definition['kind']=='board_boundary':unwrapped[positions]=model.scene.unwrapped_path(curve,stations[rr],state,board=board)
                return board.xyz(unwrapped[:,0],unwrapped[:,1])
            definitions=model.scene.structural_family.curves(state);result=base.copy()
            for curve,rr,positions in plan:
                definition=definitions.get(self.params.curve_lookup[curve].label)
                if definition is None:continue
                s=stations[rr]
                if definition['kind']=='ice_segment':
                    ends=np.asarray(definition['endpoints_xy']).reshape(2,2)
                    result[positions,:2]=(1-s[:,None])*ends[0]+s[:,None]*ends[1];result[positions,2]=0
                elif definition['kind']=='ice_circle':
                    a,z=definition['angle_interval'];angle=a+s*(z-a)
                    result[positions,:2]=definition['center_xy']+definition['radius']*np.stack((np.cos(angle),np.sin(angle)),axis=1);result[positions,2]=0
                else:raise ValueError('shared geometry derivative requires a registered line/circle generator')
            return result
        while step>abs(float(np.spacing(q[column]))):
            plus=minus=None;qp=q.copy();qm=q.copy();qp[column]+=step;qm[column]-=step
            try:plus=evaluate(qp)
            except (ValueError,FloatingPointError):pass
            try:minus=evaluate(qm)
            except (ValueError,FloatingPointError):pass
            if plus is not None or minus is not None:break
            step*=.5
        if plus is not None and minus is not None:derivative=(plus-minus)/(2*step)
        elif plus is not None:derivative=(plus-base)/step;method='one_sided_positive'
        elif minus is not None:derivative=(base-minus)/step;method='one_sided_negative'
        else:raise ForwardDomainRejection('no representable finite difference within current geometry domain: '+key)
        return b,column,rows,derivative,method,step

    def _board_geometry_derivatives(self,model,stations,xyz,forward):
        p=self.params;key=p.board_key;b,rows,plan=self._world_plans[key];base=model.state.get(key)
        count=len(base);boards=[];coordinates=[];steps=[];methods=[];current_q=self._current_board_q
        for column in range(count):
            step=self.geometry_step*max(1.,abs(float(base[column])));positive=negative=None
            while step>abs(float(np.spacing(base[column]))):
                positive=negative=None
                for sign in (1,-1):
                    value=base.copy();value[column]+=sign*step
                    try:
                        state=model.state.updated({key:value});board=model.scene.board(state)
                        q=current_q.copy()
                        for curve,rr,positions in plan:
                            definition=model.scene.resolved_definition(p.curve_lookup[curve],state)
                            if definition['kind']=='board_boundary':q[positions]=model.scene.unwrapped_path(curve,stations[rr],state,board=board)
                        if sign>0:positive=(board,q)
                        else:negative=(board,q)
                    except (ValueError,FloatingPointError):pass
                if positive is not None or negative is not None:break
                step*=.5
            if positive is None and negative is None:raise ForwardDomainRejection('no finite board geometry derivative in current exact domain')
            method='central_gpu_continuous_board'
            if positive is None:positive=(model.scene.board(model.state),current_q);method='one_sided_negative_gpu_continuous_board'
            if negative is None:negative=(model.scene.board(model.state),current_q);method='one_sided_positive_gpu_continuous_board'
            boards.extend((positive[0],negative[0]));coordinates.extend((positive[1],negative[1]));steps.append(step);methods.append(method)
        if not hasattr(self,'board_fd_gpu'):self.board_fd_gpu=CudaBoardBatch(len(rows),variants=2*count,device=self.gpu.device)
        result=self.board_fd_gpu.evaluate(boards,np.stack(coordinates))
        if not result['valid'].all():raise ForwardDomainRejection('continuous-board finite-difference variant domain rejected rows')
        jac=np.empty((len(rows),2,count));records=[]
        for column in range(count):
            den=2*steps[column] if methods[column].startswith('central') else steps[column]
            dx=(result['xyz'][2*column]-result['xyz'][2*column+1])/den
            jac[:,:,column]=np.einsum('nij,nj->ni',forward['jacobian_world'][rows],dx)
            records.append({'block_id':p.block_ids[b],'component':column,'method':methods[column],'step':steps[column]})
        return b,jac,records

    def _compute(self,x,need_jacobian):
        p=self.params
        if self._last_x is not None and np.array_equal(x,self._last_x) and (not need_jacobian or self._jacobian_ready):return self._cache
        started=time.perf_counter();same=self._last_x is not None and np.array_equal(x,self._last_x)
        self._evaluation_context={'family':'physical_state_or_world_geometry'}
        try:
            if same:
                model,op,stations,rows,alpha,xyz=(self._cache[k] for k in ('model','operator','stations','rows','alpha','xyz'))
            else:
                model=p.model_for(x);op=model.operator('S01');stations,rows,alpha=p.auxiliaries(x,model)
                xyz,_=self._world(model,stations,need_station=False)
            tau=op.camera_definition['signed_readout_seconds'];h=p.rig.image_height-1
            times=p.frame_times+tau*(rows/h-.5)
            self.gpu.set_state(op);self.gpu.set_rows(xyz=np.ascontiguousarray(xyz),time=np.ascontiguousarray(times))
            self._evaluation_context={'family':'physical_projection','jacobian_requested':bool(need_jacobian)}
            forward=self.gpu.evaluate() if need_jacobian else self.gpu.evaluate_uv()
            if not np.all(forward['valid']):
                bad=np.flatnonzero(~forward['valid'])
                self.last_rejected_rows=bad;self.last_rejected_xyz=xyz.copy();self.last_rejected_forward=forward
                self._evaluation_context.update(first_active_row=int(bad[0]),status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(forward['status'][bad],return_counts=True))})
                raise ForwardDomainRejection('current physical projection domain rejected '+str(len(bad))+' rows; first source '+p.observation_ids[int(bad[0])])
            target=p.target_uv.copy();query_derivative=np.zeros((p.count,2))
            for row in np.flatnonzero(p.kinds==2):
                self._evaluation_context={'family':'source_query_domain','first_active_row':int(row)}
                profile=p.profiles[int(p.raw_observation_rows[row])]
                target[row],query_derivative[row]=profile.evaluate(alpha[row],derivative=True)
            error=np.ascontiguousarray(forward['uv']-p.target_uv);query_error=np.ascontiguousarray(forward['uv']-target);roweq=rows-forward['uv'][:,1]
            cache={'model':model,'operator':op,'stations':stations,'rows':rows,'alpha':alpha,'xyz':xyz,'times':times,
                'forward':forward,'error':error,'query_error':query_error,'roweq':roweq,'query_derivative':query_derivative,'target_uv_current_query':target}
            if need_jacobian:
                self.Jprojection.fill(0)
                _,dstation=self._world(model,stations,need_station=True)
                # All shared camera and optical derivatives come from the same
                # resident GPU operator; time/readout chain is explicit below.
                for suffix,name in (('pivot','jacobian_pivot'),('lever','jacobian_lever'),('reference_rotation_vector','jacobian_reference_rotation')):
                    b=p.semantic_to_native[p.rig.key(suffix)];rr=p.block_row_locations[b][:,0]
                    self._write_block(b,forward[name][rr])
                b=p.semantic_to_native[p.rig.key('optics')];rr=p.block_row_locations[b][:,0]
                optics=forward['jacobian_optics_at_fixed_time'][rr].copy()
                optics[:,:,11]+=forward['jacobian_time'][rr]*(rows[rr,None]/h-.5)
                self._write_block(b,optics)
                self._evaluation_context={'family':'axis_chart_jacobian'}
                for suffix in ('pan_axis_chart','tilt_axis_chart'):
                    key=p.rig.key(suffix);b=p.semantic_to_native[key];rr=p.block_row_locations[b][:,0];q=model.state.get(key);columns=[]
                    for column in range(len(q)):
                        step=self.axis_step;qp=q.copy();qm=q.copy();qp[column]+=step;qm[column]-=step
                        self.gpu.set_state(model.with_state(model.state.updated({key:qp})).operator('S01'));positive=self.gpu.evaluate_uv()
                        self.gpu.set_state(model.with_state(model.state.updated({key:qm})).operator('S01'));negative=self.gpu.evaluate_uv()
                        if not positive['valid'].all() or not negative['valid'].all():raise ForwardDomainRejection('axis chart derivative left current optical domain')
                        columns.append((positive['uv'][rr]-negative['uv'][rr])/(2*step))
                    self._write_block(b,np.stack(columns,axis=-1))
                self.gpu.set_state(op)
                # Sparse cardinal controls. Each projection uses four supports;
                # exact reference-gauge dependency may add another native view.
                indices=forward['control_indices'];weights=forward['cubic_weights'];motion=forward['jacobian_motion']
                for control,b in p.control_native.items():
                    locations=p.block_row_locations.get(b)
                    if locations is None:continue
                    rr=locations[:,0];w=np.sum(weights[rr]*(indices[rr]==control),axis=1)
                    if control==2:jac=motion[rr,:,2,None]*w[:,None,None]
                    else:
                        jac=motion[rr]*w[:,None,None]
                        if control in (1,3):
                            ref=np.sum(weights[rr]*(indices[rr]==2),axis=1)
                            jac[:,:,:2]-=.25*motion[rr,:,:2]*ref[:,None,None]
                    self._write_block(b,jac)
                # Shared scene dimensions use explicitly reported numerical
                # world derivatives, then the same analytic GPU dUV/dXYZ chain.
                futures=[];world_jac={};methods=[]
                if p.board_key in self._world_plans:
                    _,board_rows,board_plan=self._world_plans[p.board_key]
                    self._board_unwrapped=np.empty((len(board_rows),2));current_board=model.scene.board(model.state)
                    for curve,rr,positions in board_plan:self._board_unwrapped[positions]=model.scene.unwrapped_path(curve,stations[rr],model.state,board=current_board)
                for key,(b,rr,plan) in self._world_plans.items():
                    if key==p.board_key and hasattr(self,'board_gpu'):
                        board_block,board_jacobian,board_methods=self._board_geometry_derivatives(model,stations,xyz,forward)
                        world_jac[board_block]=board_jacobian;methods.extend(board_methods);continue
                    world_jac[b]=np.zeros((len(rr),2,p.block_sizes[b]))
                    for column in range(p.block_sizes[b]):futures.append(self.pool.submit(self._geometry_column,model,stations,xyz,key,column))
                for task in futures:
                    b,column,rr,dx,method,step=task.result()
                    world_jac[b][:,:,column]=np.einsum('nij,nj->ni',forward['jacobian_world'][rr],dx)
                    methods.append({'block_id':p.block_ids[b],'component':column,'method':method,'step':step})
                for b,jac in world_jac.items():self._write_block(b,jac)
                # Every Ad uses one shared static layout block. Canonical sampled
                # vertices never enter the free vector and never move per frame.
                layout_jac={}
                for curve,rr in p.world_groups.items():
                    key=p.curve_group_keys[curve]
                    if not key:continue
                    b=p.semantic_to_native[key];loc=p.block_row_locations[b];target_rows=loc[:,0]
                    if b not in layout_jac:layout_jac[b]=np.zeros((len(loc),2,p.block_sizes[b]))
                    address=np.searchsorted(target_rows,rr)
                    if hasattr(self,'board_gpu') and hasattr(model.scene,'unwrapped_layout_jacobian'):
                        derivative=model.scene.unwrapped_layout_jacobian(curve,stations[rr],model.state,board=current_board)['jacobian_layout']
                        tangent=self._current_board_tangent[self._current_board_address[rr]]
                        world=tangent[:,:,None]*derivative[:,0,None,:];world[:,2,:]+=derivative[:,1,:]
                    else:world=model.scene.world_layout_jacobian(curve,stations[rr],model.state,board=current_board)['jacobian_layout']
                    layout_jac[b][address]=np.einsum('nij,njk->nik',forward['jacobian_world'][rr],world)
                for b,jac in layout_jac.items():self._write_block(b,jac)
                station_pixel=np.einsum('nij,nj->ni',forward['jacobian_world'],dstation)
                # Five shared physical point stations are model-owned, whereas
                # the eight exposure rows are observation-owned auxiliaries.
                for key in set(p.shared_landmark_blocks.values()):
                    b=p.semantic_to_native[key];rr=p.block_row_locations[b][:,0]
                    self._write_block(b,station_pixel[rr,:,None])
                local_width=np.asarray(p.block_sizes,dtype=np.int64)[p.local_block]
                uvrow=forward['jacobian_time']*(tau/h)
                for component in range(2):
                    base=self._local_offsets+component*local_width
                    selected=p.station_column>=0
                    self.Jprojection[base[selected]+p.station_column[selected]]=station_pixel[selected,component]
                    self.Jprojection[base+p.row_column]=uvrow[:,component]
                np.copyto(self.Jerror,self.Jprojection)
                selected=p.alpha_column>=0
                for component in range(2):
                    base=self._local_offsets+component*local_width
                    self.Jerror[base[selected]+p.alpha_column[selected]]=-query_derivative[selected,component]
                cache['geometry_derivative_methods']=methods
                self.counters['full_linearizations']+=1
            self.counters['forward_calls']+=1
            self.timings['last_full_linearization_seconds' if need_jacobian else 'last_residual_evaluation_seconds']=time.perf_counter()-started
            self._last_x=np.array(x,copy=True);self._jacobian_ready=need_jacobian;self._cache=cache
            return cache
        except (ValueError,FloatingPointError) as exc:
            self.counters['domain_rejections']+=1;self.domain_rejection_index+=1
            event={'schema':'rinksolve.domain_rejection.v1','trial_index':self.domain_rejection_index,'error':str(exc),'parameter_point_changed':not same,
                'stage':self.phase,'free_mode':self.free_mode,'parameter_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),**self._evaluation_context}
            first=event.get('first_active_row')
            if first is not None:
                i=int(first);event.update(observation_id=p.observation_ids[i],native_pts=int(p.native_pts[i]),curve_id=str(p.curve_ids[i]))
                try:
                    event.update(formal_row=float(rows[i]),world_station=float(stations[i]),source_uv=p.target_uv[i].tolist(),source_time_seconds=float(times[i]),world_xyz=xyz[i].tolist())
                    detail=op.project_at(xyz[i:i+1],times[i:i+1])
                    for name in ('uv','depth','ray_radius','status'):
                        if name in detail:event['projection_'+name]=np.asarray(detail[name][0]).tolist()
                except BaseException as detail_error:event['detail_error']=str(detail_error)
            def clean(value):
                if isinstance(value,float) and not np.isfinite(value):return None
                if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
                if isinstance(value,list):return [clean(v) for v in value]
                return value
            event=clean(event);self.last_domain_failure=event
            if self.domain_event_callback is not None:self.domain_event_callback(event,x)
            self._last_x=None;self._jacobian_ready=False
            raise ForwardDomainRejection(str(exc)) from exc

    def evaluate_statistics(self,x,residual_out,jacobian_out,new_point):
        try:cache=self._compute(x,jacobian_out is not None)
        except ForwardDomainRejection:return False
        self.whitening.apply(cache['error'],self.Jprojection if jacobian_out is not None else None,residual_out,jacobian_out,num_threads=self.num_threads)

    def evaluate_objective(self,x,residual_out,jacobian_out,new_point):
        if self.phase in ('statistics','statistical_level_set'):return self.evaluate_statistics(x,residual_out,jacobian_out,new_point)
        b=float(x[self.params.budget_offset])
        if b<0:return False
        residual_out[0]=np.sqrt(2*b+1)
        if jacobian_out is not None:jacobian_out[0]=1/residual_out[0]

    def evaluate_constraints(self,x,constraint_out,jacobian_out,new_point):
        try:cache=self._compute(x,jacobian_out is not None)
        except ForwardDomainRejection:return False
        nb=len(self.budget_rows);nr=self.params.count;q,coefficients=self._budget_quantities(cache)
        constraint_out[:nb]=q-x[self.params.budget_offset]
        constraint_out[nb:nb+nr]=cache['roweq']
        constraint_out[nb+nr:]=cache['query_error'][self.profile_rows].reshape(-1)
        if jacobian_out is not None:
            first=self.budget_layout.jacobian_value_count;second=first+self.roweq_layout.jacobian_value_count
            self.budget_plan.combine(self.Jprojection,coefficients,out=jacobian_out[:first],direct_positions=self.budget_direct,direct_values=self.budget_direct_values,num_threads=self.num_threads)
            coefficient=np.zeros((nr,2),dtype=np.float64);coefficient[:,1]=-1
            self.roweq_plan.combine(self.Jprojection,coefficient,out=jacobian_out[first:second],direct_positions=self.roweq_direct,direct_values=self._ones,num_threads=self.num_threads)
            jacobian_out[second:]=self.Jerror[self.query_jacobian_indices]

    def precondition_formal_rows(self,vector,*,tolerance_px):
        """Solve existing RS row equalities at fixed physics and fixed loci.

        Only a bracketed root inside the native sensor may replace a row. An
        unbracketed root leaves its original row and full equality untouched.
        """
        if not np.isfinite(tolerance_px) or tolerance_px<=0:raise ValueError('explicit positive row-root equality tolerance required')
        p=self.params;x=np.array(vector,dtype=np.float64,copy=True);before=x.copy()
        cache=self._compute(x,False);model=cache['model'];physics_sha=model.state.digest();stations=cache['stations'].copy();rows=cache['rows'].copy()
        op=cache['operator'];tau=op.camera_definition['signed_readout_seconds'];height=p.rig.image_height-1
        xyz=cache['xyz'];eligible=self.problem.constant_blocks[p.local_block]==0
        self.gpu.set_state(op)
        def project(row,full=False):
            times=p.frame_times+tau*(row/height-.5)
            self.gpu.set_rows(xyz=np.ascontiguousarray(xyz),time=np.ascontiguousarray(times))
            return self.gpu.evaluate() if full else self.gpu.evaluate_uv()
        left=np.zeros(p.count);right=np.full(p.count,float(height));fl=project(left);fr=project(right)
        fleft=left-fl['uv'][:,1];fright=right-fr['uv'][:,1]
        bracket=eligible&fl['valid']&fr['valid']&(fleft*fright<=0)
        exact_left=bracket&(np.abs(fleft)<=tolerance_px);exact_right=bracket&(np.abs(fright)<=tolerance_px)
        work=rows.copy();work[exact_left]=0.;work[exact_right]=height
        converged=exact_left|exact_right;failed=eligible&~bracket
        failure_reason={int(i):'no_in_sensor_root_at_current_world_point' for i in np.flatnonzero(failed)}
        bound=int(np.ceil(np.log(max(float(height)/tolerance_px,1.))/(-np.log(.75))))+4
        for iteration in range(bound):
            active=bracket&~converged&~failed
            if not np.any(active):break
            forward=project(work,True);value=work-forward['uv'][:,1]
            converged|=active&forward['valid']&(np.abs(value)<=tolerance_px)
            invalid=active&~forward['valid'];failed|=invalid
            for i in np.flatnonzero(invalid):failure_reason[int(i)]='projection_domain_during_row_root'
            active&=~converged&~failed
            same_sign=np.signbit(value)==np.signbit(fleft)
            choose=active&same_sign;left[choose]=work[choose];fleft[choose]=value[choose]
            choose=active&~same_sign;right[choose]=work[choose];fright[choose]=value[choose]
            derivative=1-forward['jacobian_time'][:,1]*tau/height
            proposal=work-np.divide(value,derivative,out=np.zeros_like(value),where=np.abs(derivative)>np.finfo(float).eps)
            width=right-left;safe=np.isfinite(proposal)&(proposal>left+.25*width)&(proposal<right-.25*width)
            proposal=np.where(safe,proposal,(left+right)/2)
            stalled=active&(proposal==work);failed|=stalled
            for i in np.flatnonzero(stalled):failure_reason[int(i)]='row_root_floating_point_stagnation'
            active&=~failed;work[active]=proposal[active]
        remaining=bracket&~converged&~failed
        for i in np.flatnonzero(remaining):failure_reason[int(i)]='derived_row_bracket_iteration_bound_not_converged'
        failed|=remaining
        # No clipping: only actually converged in-sensor equalities are used.
        offsets=p.parameter_offsets[p.local_block]+p.row_column
        x[offsets[converged]]=work[converged]
        self._last_x=None;after=self._compute(x,False)
        if after['model'].state.digest()!=physics_sha or not np.array_equal(after['stations'],stations):raise RuntimeError('row preconditioner changed a physical parameter or station')
        if not np.array_equal(x[:p.model_parameter_count],before[:p.model_parameter_count]):raise RuntimeError('row preconditioner changed physical native views')
        source=[]
        for i in np.flatnonzero(failed):
            source.append({'observation_id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),'curve_id':str(p.curve_ids[i]),'reason':failure_reason[int(i)],
                'row_before':float(rows[i]),'row_after':float(after['rows'][i]),'row_equality_before_px':float(cache['roweq'][i]),'row_equality_after_px':float(after['roweq'][i]),
                'row_equality_at_sensor_zero_px':float(-fl['uv'][i,1]) if np.isfinite(fl['uv'][i,1]) else None,'row_equality_at_sensor_last_px':float(height-fr['uv'][i,1]) if np.isfinite(fr['uv'][i,1]) else None,
                'source_uv':p.target_uv[i].tolist(),'station_unchanged':float(stations[i]),'equation_still_active':True,'source_identity_or_pixels_changed':False})
        report={'schema':'rinksolve.formal_row_preconditioner.v1','method':'safeguarded Newton with a genuine in-sensor bracket; no clipped roots',
            'tolerance_px':float(tolerance_px),'derived_iteration_bound':bound,'eligible':int(np.sum(eligible)),'converged':int(np.sum(converged)),'unresolved':len(source),
            'physical_state_sha256_before':physics_sha,'physical_state_sha256_after':after['model'].state.digest(),'all_stations_identical':bool(np.array_equal(stations,after['stations'])),
            'source_ids_UV_PTS_unchanged':True,'counted_as_Ceres_accepted_step':False,
            'row_equality_cost_before':.5*float(cache['roweq']@cache['roweq']),'row_equality_cost_after':.5*float(after['roweq']@after['roweq']),
            'row_equality_max_before':float(np.abs(cache['roweq']).max()),'row_equality_max_after':float(np.abs(after['roweq']).max()),
            'budget_sq_unchanged':bool(x[p.budget_offset]==before[p.budget_offset]),'budget_q_max_before':float(self._budget_quantities(cache)[0].max()),'budget_q_max_after':float(self._budget_quantities(after)[0].max()),
            'unresolved_sources':source}
        return x,report

    def precondition_joint_rows(self,vector,multipliers,penalties,*,tolerance_px):
        from scipy.optimize import brentq,minimize_scalar
        if not np.isfinite(tolerance_px) or tolerance_px<=0:raise ValueError('explicit positive joint row tolerance required')
        p=self.params;x=np.array(vector,dtype=np.float64,copy=True);before=x.copy();cache=self._compute(x,False)
        model=cache['model'];physics_sha=model.state.digest();op=cache['operator'];tau=op.camera_definition['signed_readout_seconds'];height=p.rig.image_height-1
        c=self.problem.constraints;lam=np.asarray(multipliers,dtype=np.float64);rho=self.al._rho_rows(np.asarray(penalties,dtype=np.float64))
        if lam.shape!=(c.layout.residual_count,) or rho.shape!=lam.shape or not np.isfinite(lam).all() or not np.isfinite(rho).all() or np.any(rho<=0):raise ValueError('explicit compatible PHR multiplier/penalty state required')
        def merit(vector):
            residual,_,raw,_=self.al._evaluate(vector,False);q=raw/c.scales;shift=q+lam/rho
            return .5*float(residual@residual)+.5*float(np.sum(rho*np.where(c.equality,shift,np.maximum(shift,0))**2)),q
        current_merit,current_q=merit(x);start_merit=current_merit
        unresolved=np.flatnonzero(np.abs(cache['roweq'])>tolerance_px);records=[];count=0
        for i in unresolved:
            i=int(i);basecache=self._compute(x,False);kind=int(p.kinds[i]);record={'observation_id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),'curve_id':str(p.curve_ids[i]),'semantic_kind':kind,'row_equality_before_px':float(basecache['roweq'][i]),'accepted':False}
            if p.station_column[i]<0 or self.problem.constant_blocks[p.local_block[i]]:
                record['reason']='no_free_observation_station_on_this_typed_feature';records.append(record);continue
            curve=str(p.curve_ids[i]);time_value=float(p.frame_times[i]);rawuv=p.target_uv[i];s0=float(basecache['stations'][i])
            def fixed(station,row):
                stations=np.atleast_1d(station);world=model.scene.world_path(curve,stations,model.state)
                return op.project_at(world,time_value+tau*(row/height-.5))
            def roots(station):
                stations=np.atleast_1d(station);world=model.scene.world_path(curve,stations,model.state)
                return op.project_rs(world,np.full(len(stations),p.native_pts[i],dtype=np.int64),root_iterations=64,root_tolerance_px=tolerance_px)
            grid=np.unique(np.r_[np.linspace(0,1,513),s0]);left=fixed(grid,0);right=fixed(grid,height);fa=-left['uv'][:,1];fb=height-right['uv'][:,1]
            valid=left['valid']&right['valid']&(fa<=0)&(fb>=0);candidates=[]
            for row,values in ((0.,fa),(float(height),fb)):
                crossing=np.flatnonzero(np.isfinite(values[:-1])&np.isfinite(values[1:])&(np.signbit(values[:-1])!=np.signbit(values[1:])))
                for j in crossing:
                    try:
                        station=brentq(lambda s:row-float(fixed(s,row)['uv'][0,1]),float(grid[j]),float(grid[j+1]),xtol=1e-14)
                        projected=fixed(station,row)
                        if projected['valid'][0] and abs(row-projected['uv'][0,1])<=tolerance_px:
                            uv=projected['uv'][0];candidates.append((float(np.sum((uv-rawuv)**2)),station,row,uv))
                    except ValueError:pass
            if np.any(valid):
                f=roots(grid[valid]);error=np.sum((f['uv']-rawuv)**2,axis=1);error[~f['valid']]=np.inf
                if np.isfinite(error).any():
                    j=int(np.argmin(error));candidates.append((float(error[j]),float(grid[valid][j]),float(f['row'][j]),f['uv'][j]))
                cuts=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(np.int8)))
                for lo,hi in zip(cuts[::2],cuts[1::2]):
                    if hi-lo<2:continue
                    def objective(station):
                        f=roots(station);return float(np.sum((f['uv'][0]-rawuv)**2)) if f['valid'][0] else 1e100
                    fit=minimize_scalar(objective,bounds=(float(grid[lo]),float(grid[hi-1])),method='bounded',options={'xatol':1e-13})
                    if fit.fun<1e99:
                        f=roots(fit.x);candidates.append((float(fit.fun),float(fit.x),float(f['row'][0]),f['uv'][0]))
            if not candidates:
                record['reason']='no_in_sensor_root_found_on_same_finite_branch';records.append(record);continue
            best=min(candidates,key=lambda v:v[0]);error2,station,row,uv=best;trial=x.copy()
            station_offset=int(p.parameter_offsets[p.local_block[i]]+p.station_column[i]);row_offset=int(p.parameter_offsets[p.local_block[i]]+p.row_column[i]);trial[station_offset]=station;trial[row_offset]=row
            proposed_merit,proposed_q=merit(trial);original_budget=np.flatnonzero(self.budget_rows==i)
            budget_good=not len(original_budget) or bool(np.all(np.maximum(proposed_q[original_budget],0)<=np.maximum(current_q[original_budget],0)+tolerance_px))
            merit_tolerance=64*np.finfo(float).eps*max(1,abs(current_merit));merit_good=proposed_merit<=current_merit+merit_tolerance
            record.update(station_before=s0,station_candidate=station,row_candidate=row,predicted_uv_candidate=np.asarray(uv).tolist(),pixel_error_candidate=error2**.5,row_equality_candidate_px=float(row-uv[1]),PHR_merit_before=current_merit,PHR_merit_candidate=proposed_merit,engineering_E_violation_not_increased=budget_good)
            if len(original_budget):record['candidate_q_minus_b']=float(proposed_q[original_budget[0]])
            if budget_good and merit_good:
                x=trial;current_merit=proposed_merit;current_q=proposed_q;record['accepted']=True;record['reason']='genuine_same_branch_sensor_row_root_with_nonincreasing_E_violation_and_PHR_merit';count+=1
            else:record['reason']='same_branch_sensor_root_exists_but_worsens_engineering_E_or_current_PHR_merit'
            records.append(record)
        after=self._compute(x,False)
        if after['model'].state.digest()!=physics_sha or not np.array_equal(x[:p.model_parameter_count],before[:p.model_parameter_count]) or x[p.budget_offset]!=before[p.budget_offset]:raise RuntimeError('joint auxiliary initializer changed physical parameters or shared budget')
        return x,{'schema':'rinksolve.joint_auxiliary_preconditioner.v1','physical_state_sha256_before':physics_sha,'physical_state_sha256_after':after['model'].state.digest(),'Ceres_accepted_step':False,'source_pixels_ids_entity_bindings_unchanged':True,'multipliers_penalties_budget_unchanged':True,'accepted':count,'unresolved':sum(not v['accepted'] for v in records),'PHR_merit_before':start_merit,'PHR_merit_after':current_merit,'row_equality_max_after_px':float(np.abs(after['roweq']).max()),'records':records}

    def current_diagnostics(self,x):
        cache=self._compute(x,False);p=self.params;error=cache['error'];pixel=np.linalg.norm(error,axis=1)
        entities={}
        for curve,rr in p.world_groups.items():
            entity=p.curve_lookup[curve].entity_id;d=entities.setdefault(entity,{'rows':0,'pixel_max':0.,'squared_error':0.})
            d['rows']+=len(rr);d['pixel_max']=max(d['pixel_max'],float(pixel[rr].max()));d['squared_error']+=float(pixel[rr]@pixel[rr])
        for value in entities.values():value['pixel_rms']=(value.pop('squared_error')/value['rows'])**.5
        budget_pixel=pixel[self.budget_rows]
        return {'same_model_id':cache['model'].id,'model_revision':cache['model'].revision,'state_sha256':cache['model'].state.digest(),
            'native_exposure_count':len(np.unique(p.native_pts)),'active_observations':p.count,'original_source_evidence_modified':False,
            'pixel_error':{'rms':float(np.sqrt(np.mean(pixel*pixel))),'maximum':float(pixel.max())},
            'stage':self.phase,'budget_sq':float(x[p.budget_offset]),
            'engineering_budget':{'count':len(budget_pixel),'target_budget_squared':self.target_budget_sq,'violating_at_current_target':int(np.sum(self._budget_quantities(cache)[0]>self.target_budget_sq)),
                'maximum_ratio':float(np.sqrt(self._budget_quantities(cache)[0].max())),'is_sigma':False},
            'query_equality_max_px':float(np.max(np.abs(cache['query_error'][self.profile_rows]),initial=0.)),
            'formal_row_equality_max_px':float(np.max(np.abs(cache['roweq']))),'entities':entities,'counters':dict(self.counters),'timings':dict(self.timings),
            'initialization_repairs':self.initialization_repairs,'inner_numerical_metric':self.last_inner_metric,'last_domain_failure':self.last_domain_failure,'canonical_shapes_are_user_fixed_facts':False,'visual_acceptance':False,'next_shot_allowed':False}

    def write_conflict_report(self,x,path,*,worst_count=100):
        from .store import atomic_json
        p=self.params;cache=self._compute(x,False);q,_=self._budget_quantities(cache)
        source_q=np.full(p.count,np.nan);source_q[self.budget_rows]=q
        groups={}
        for curve,rr in p.world_groups.items():
            c=p.curve_lookup[curve];finite=source_q[rr][np.isfinite(source_q[rr])]
            groups[curve]={'entity_id':c.entity_id,'observations':len(rr),'q_max':float(finite.max()) if len(finite) else None,
                'q_p95':float(np.quantile(finite,.95)) if len(finite) else None,
                'outside_target_one':int(np.sum(finite>1)),
                'raw_pixel_max':float(np.linalg.norm(cache['error'][rr],axis=1).max()),
                'row_equality_max_px':float(np.abs(cache['roweq'][rr]).max())}
        ranked=self.budget_rows[np.argsort(q)[::-1][:worst_count]]
        worst=[]
        for i in ranked:
            curve=str(p.curve_ids[i]);c=p.curve_lookup[curve]
            worst.append({'observation_id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),'curve_id':curve,'entity_id':c.entity_id,
                'semantic_kind':int(p.kinds[i]),'raw_uv':p.target_uv[i].tolist(),'predicted_uv':cache['forward']['uv'][i].tolist(),
                'raw_error_px':cache['error'][i].tolist(),'q':float(source_q[i]),'row_equality_px':float(cache['roweq'][i]),
                'query_equality_px':cache['query_error'][i].tolist() if p.kinds[i]==2 else None,
                'locus_station':float(cache['stations'][i]),'fixed_physical_landmark':bool(p.station_column[i]<0),
                'raw_source_row':int(p.raw_observation_rows[i])})
        report={'schema':'rinksolve.entity_conflicts.v1','model_id':cache['model'].id,'model_revision':cache['model'].revision,
            'stage':self.phase,'budget_sq':float(x[p.budget_offset]),'target_budget_sq':self.target_budget_sq,'same_source_ids_and_pixels':True,
            'groups':dict(sorted(groups.items(),key=lambda v:v[1]['q_max'] or 0.,reverse=True)),'worst_observations':worst,
            'source_error_semantics':'statistical pred-raw; ordinary budget squared norm; profile budget source-normal component; exact Query equality separate',
            'policy':'diagnose material/entity conflicts; never increase E or introduce per-observation b to hide conflicts'}
        atomic_json(path,report);return report

    def assert_current_inputs(self):
        ledger=json.loads(self.constraints_path.read_text(encoding='utf-8-sig'))
        current=[dict({k:v.get(k) for k in ('id','enabled','approval','expression','scope')},implementation=getattr(self,'constraint_implementations',IMPLEMENTED_CONSTRAINTS)[v['id']])
            for v in ledger['constraints'] if v['id'] in getattr(self,'constraint_implementations',IMPLEMENTED_CONSTRAINTS) and not (self.prepared.manifest.get('numerical_data_scope')=='ice_board' and v['id'] in ('ad.object_model','numerical.ad7_fixed_corner_budget'))]
        if current!=self.constraint_bindings:raise ValueError('current bound constraint semantics changed; rebuild only affected model/factor families')
        return self.prepared.assert_current_constraint_snapshot(self.constraints_path)

    def close(self):
        for camera,geometry in getattr(self,'_seed_gpu_pool',{}).values():camera.close();geometry.close()
        if hasattr(self,'board_fd_gpu'):self.board_fd_gpu.close()
        if hasattr(self,'board_gpu'):self.board_gpu.close()
        if hasattr(self,'gpu'):self.gpu.close()
        if hasattr(self,'whitening'):self.whitening.close()
        self.pool.shutdown(wait=True,cancel_futures=True)
