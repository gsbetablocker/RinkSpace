"""Fit static advertisement objects while the recovered first shot stays fixed.

Only ARTWORK-owned similarity blocks and observation-owned finite-locus,
formal-row and Query-alpha auxiliaries enter the native vector. The recovered
camera, motion, ice and board blocks are absent from it. This is a new adapter
for the public block Ceres/PHR math, not an import of a historical optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
from scipy.sparse import save_npz
from .parameterization import S01Parameterization
from .optimizer import S01Optimizer,ForwardDomainRejection
from .native_backend import BatchLayout
from .gpu_board import CudaBoardBatch
from .preparation import load_prepared,_incremental_ref
from .observation_edits import SelectionSnapshot,apply_selection
from . import schema as S


ORIGINAL_AD_CONSTRAINTS={
 'sequence.single_shot':'S01 only; later shots remain locked',
 'clock.native':'unchanged recovered official native PTS and exact formal RS row equation',
 'source.material_only':'current admitted static advertisement material only; existing withdrawals preserved',
 'world.common_scene':'recovered camera, trajectory, ice and continuous board remain bitwise fixed',
 'board.vertical':'unchanged recovered continuous-board coordinates, without a flat-banner substitution',
 'compute.resources':'new resident CUDA projection/Jacobian and explicit CPU-thread policy',
 'observation.fixed_vs_locus':'fixed print corners, per-observation curve loci, formal rows and Query alpha have separate owners',
 'ad.object_model':'one static ARTWORK-owned similarity block per advertisement, shared across all exposures',
 'numerical.shared_budget_schedule':'original per-source E and one global b; feasibility min b then original-W statistics at b=1',
 'numerical.ad7_fixed_corner_budget':'existing fixed artwork-corner engineering E, sigma remains unknown',
}


@dataclass(frozen=True)
class AdView:
    id:str
    semantic_block_id:str
    size:int
    vector_offset:int
    component_offset:int
    owner:object


class OriginalAdsParameterization(S01Parameterization):
    """The native physical prefix contains static ads and no camera/ice state."""
    def __init__(self,model,prepared,*,intersection_enabled=False,ad7_corners_enabled=True):
        self.model,self.prepared=model,prepared
        self.shot=next(s for s in model.shots if s.id=='S01')
        self.rig=next(r for r in model.rigs if r.id==self.shot.rig_id)
        self.original_rows=prepared.selection_current_active_rows();rows=self.original_rows
        self.original_count=len(rows);a=prepared.arrays
        ids=list(prepared.observation_ids(rows));self.curve_ids=list(prepared.bound_curve_ids(rows))
        self.native_pts=np.array(a['native_pts'][rows],dtype=np.int64)
        self.target_uv=np.array(a['immutable_uv'][rows],dtype=np.float64)
        self.kinds=np.array(a['semantic_kind'][rows],dtype=np.uint8)
        self.station_seed=np.array(a['station_seed'][rows],dtype=np.float64)
        self.row_seed=np.array(a['row_auxiliary_seed'][rows],dtype=np.float64)
        self.alpha_seed=np.array(a['profile_alpha_seed'][rows],dtype=np.float64)
        self.profiles={p.observation_row:p for p in prepared.profiles}
        self.shared_landmark_blocks={};self.intersection_enabled=False
        self.fixed_template_landmarks={}
        if ad7_corners_enabled:
            features={v.id:v for v in model.landmarks}
            for observation in prepared.manifest.get('ad7_fixed_observations',[]):
                feature=features[observation['fixed_feature_id']]
                curve=feature.position.geometry_feature_ids[0]
                definition=model.scene.resolved_definition(next(c for c in model.scene.curves if c.id==curve),model.state)
                station=float(definition['fixed_artwork_landmarks'][feature.id]['station'])
                self.fixed_template_landmarks[len(ids)]=feature
                ids.append(observation['id']);self.curve_ids.append(curve)
                self.native_pts=np.r_[self.native_pts,int(observation['native_pts'])]
                self.target_uv=np.concatenate((self.target_uv,np.array([observation['uv_native_pixels']],dtype=np.float64)))
                self.kinds=np.r_[self.kinds,np.uint8(4)];self.station_seed=np.r_[self.station_seed,station]
                self.row_seed=np.r_[self.row_seed,float(observation['uv_native_pixels'][1])];self.alpha_seed=np.r_[self.alpha_seed,0.]
        self.observation_ids=tuple(v.decode('utf8') if isinstance(v,(bytes,np.bytes_)) else str(v) for v in ids)
        if not ids or len(set(self.observation_ids))!=len(ids):raise ValueError('empty or duplicate advertisement source IDs')
        self.raw_observation_rows=np.r_[rows,np.full(len(ids)-len(rows),-1,dtype=np.int64)]
        self.count=len(ids);self.curve_ids=np.asarray(self.curve_ids)
        self.source_to_active=np.full(prepared.row_count,-1,dtype=np.int64);self.source_to_active[rows]=np.arange(len(rows))
        self.curve_lookup={c.id:c for c in model.scene.curves}
        if any(k not in self.curve_lookup for k in self.curve_ids):raise ValueError('current ad source curve missing from recovered candidate')
        self.scene_keys=[];self.board_key=model.scene.board_block_id
        self.curve_group_keys={c.id:model.scene.resolved_definition(c,model.state).get('shared_layout_block_id') for c in model.scene.curves}
        used={self.curve_group_keys[str(k)] for k in self.curve_ids}
        if None in used:raise ValueError('advertisement scope includes a curve without one static object owner')
        self.selected_blocks=tuple(b.id for b in model.state.blocks if b.id in used)
        if set(self.selected_blocks)!=used:raise ValueError('current static advertisement block missing')
        specs={b.id:b for b in model.state.blocks}
        for key in self.selected_blocks:
            if specs[key].owner.kind!=S.OwnerKind.ARTWORK or len(model.state.get(key))!=4:
                raise ValueError('only four-component ARTWORK-owned static similarities may vary: '+key)
        initial=[];self.views=[]
        for key in self.selected_blocks:
            value=model.state.get(key);self.views.append(AdView(key,key,len(value),len(initial),0,specs[key].owner));initial.extend(value)
        self.views=tuple(self.views);self.model_parameter_count=len(initial)
        self.block_ids=list(self.selected_blocks);self.block_sizes=[4]*len(self.selected_blocks)
        self.semantic_to_native={v:i for i,v in enumerate(self.selected_blocks)};self.control_native={};self.rig_native=()
        self.block_owners=[{'kind':'static_artwork_object','id':k,'semantic_block_id':k,'component_offset':0} for k in self.selected_blocks]
        lower=[-np.inf]*len(initial);upper=[np.inf]*len(initial);scales=[1.]*len(initial)
        self.local_block=np.full(self.count,-1,dtype=np.int32)
        self.station_column=np.full(self.count,-1,dtype=np.int32);self.row_column=self.station_column.copy();self.alpha_column=self.station_column.copy()
        groups=list(prepared.nuisance_groups())
        groups.extend({'id':'aux/'+self.observation_ids[i],'rows':np.array([],dtype=np.int64),'extra_active_row':i,'kind':'fixed_artwork_corner_row'} for i in range(self.original_count,self.count))
        for group in groups:
            rr=[group['extra_active_row']] if 'extra_active_row' in group else [int(self.source_to_active[r]) for r in group['rows']]
            if not rr or any(r<0 for r in rr):raise ValueError('complete original W owner lost an advertisement member')
            values=[];lo=[];hi=[];scale=[]
            for i in rr:
                if self.local_block[i]!=-1:raise ValueError('observation has duplicate local auxiliary ownership')
                original=int(self.raw_observation_rows[i]);kind=int(self.kinds[i])
                if kind in (0,2):
                    station=float(self.station_seed[i])
                    if not np.isfinite(station) or not 0<=station<=1:raise ValueError('finite branch initialization outside [0,1]')
                    self.station_column[i]=len(values);values.append(station);lo.append(0.);hi.append(1.);scale.append(1.)
                row=float(self.row_seed[i])
                if not np.isfinite(row) or not 0<=row<=self.rig.image_height-1:raise ValueError('formal row initialization outside source sensor')
                self.row_column[i]=len(values);values.append(row);lo.append(0.);hi.append(float(self.rig.image_height-1));scale.append(float(self.rig.image_height-1))
                if kind==2:
                    profile=self.profiles.get(original)
                    if profile is None:raise ValueError('genuine profile missing current independent rational Query')
                    alpha=float(self.alpha_seed[i]);amin,amax=profile.alpha_bounds
                    if not amin<=alpha<=amax:raise ValueError('Query-alpha initialization outside original bounds')
                    if amin!=amax:self.alpha_column[i]=len(values);values.append(alpha);lo.append(amin);hi.append(amax);scale.append(profile.alpha_scale)
            self.local_block[rr]=len(self.block_ids);self.block_ids.append(str(group['id']));self.block_sizes.append(len(values))
            self.block_owners.append({'kind':'current_source_nuisance_group','id':str(group['id']),'source_group_kind':group['kind'],'contains_shared_landmark_position':False})
            initial.extend(values);lower.extend(lo);upper.extend(hi);scales.extend(scale)
        if np.any(self.local_block<0):raise ValueError('advertisement observation lacks auxiliary owner')
        self.local_group_count=len(groups);self.local_component_records=[]
        self.budget_block=len(self.block_ids);self.block_ids.append('numerical/shared_budget_sq');self.block_sizes.append(1)
        self.block_owners.append({'kind':'numerical_feasibility','id':'numerical/shared_budget_sq'})
        initial.append(1.);lower.append(0.);upper.append(np.inf);scales.append(1.)
        self.block_ids=tuple(self.block_ids);self.block_sizes=tuple(self.block_sizes)
        self.parameter_offsets=np.r_[0,np.cumsum(self.block_sizes,dtype=np.int64)]
        self.initial=np.array(initial,dtype=np.float64);self.lower=np.array(lower);self.upper=np.array(upper);self.scales=np.array(scales)
        self.shared_station_bounds=[];self.frame_times=self.shot.frame_time(self.native_pts)
        op=model.operator('S01');left,right=op.domain
        self.tau_domain=2*min(float(self.frame_times.min()-left),float(right-self.frame_times.max()))
        if abs(op.camera_definition['signed_readout_seconds'])>self.tau_domain:raise ValueError('recovered camera readout leaves current complete trajectory domain')
        self.world_groups={str(k):np.flatnonzero(self.curve_ids==k) for k in np.unique(self.curve_ids)}
        self.world_dependencies={k:(self.board_key,self.curve_group_keys[k]) for k in self.world_groups}
        self.raw_layout=BatchLayout(self.block_ids,self.block_sizes,self.observation_ids,(2,)*self.count,
            tuple((self.semantic_to_native[self.curve_group_keys[str(c)]],int(self.local_block[i])) for i,c in enumerate(self.curve_ids)))
        self.block_row_locations={}
        for row,blocks in enumerate(self.raw_layout.factor_blocks):
            edge=int(self.raw_layout.factor_edge_offsets[row])
            for k,b in enumerate(blocks):self.block_row_locations.setdefault(b,[]).append((row,int(self.raw_layout.jacobian_offsets[edge+k])))
        self.block_row_locations={b:np.asarray(v,dtype=np.int64) for b,v in self.block_row_locations.items()}
        self.ordering_groups=np.ones(len(self.block_ids),dtype=np.int32);self.ordering_groups[len(self.views):-1]=0
        self.budget_offset=int(self.parameter_offsets[self.budget_block])
        self.frozen_block_sha256=self._frozen_digest(model)

    def _frozen_digest(self,model):
        h=hashlib.sha256()
        for block in model.state.blocks:
            if block.id in self.selected_blocks:continue
            h.update(block.id.encode('utf8'));h.update(model.state.get(block.id).tobytes())
        return h.hexdigest()

    def model_for(self,vector,*,revision=None,state_id=None):
        x=np.asarray(vector,dtype=np.float64)
        if x.shape!=self.initial.shape or not np.isfinite(x).all():raise ValueError('complete advertisement vector differs from bound layout')
        changes={v.semantic_block_id:x[v.vector_offset:v.vector_offset+v.size] for v in self.views}
        state=self.model.state.updated(changes,new_id=state_id)
        if revision is not None:state=replace(state,revision=int(revision))
        model=replace(self.model,state=state)
        if self._frozen_digest(model)!=self.frozen_block_sha256:raise RuntimeError('advertisement update changed recovered camera, ice or board block')
        return model

    def inventory(self):
        return {'schema':'rinksolve.original_ads_native_parameters.v1','model_id':self.model.id,'model_revision':self.model.revision,
            'model_state_id':self.model.state.id,'model_state_sha256':self.model.state.digest(),'prepared_path':str(self.prepared.directory),
            'native_blocks':len(self.block_ids),'native_parameters':len(self.initial),'physical_model_native_views':len(self.views),
            'observation_rows':self.count,'original_current_rows':self.original_count,'new_shared_landmark_observations':self.count-self.original_count,
            'selected_model_blocks':list(self.selected_blocks),'recovered_camera_ice_board_frozen_sha256':self.frozen_block_sha256,
            'camera_trajectory_ice_board_in_free_vector':False,'sampled_polyline_vertices_free':False,
            'one_static_object_per_all_exposures':True,'formal_rs_row_auxiliary':True,'query_alpha_is_world_station':False,
            'source_observation_ids_preserved':True,'local_group_count':self.local_group_count,
            'global_views':[{'id':v.id,'semantic_block_id':v.semantic_block_id,'component_offset':0,'size':v.size,'vector_offset':v.vector_offset,'owner':self.block_owners[i]} for i,v in enumerate(self.views)]}


class OriginalAdsOptimizer(S01Optimizer):
    parameterization_class=OriginalAdsParameterization
    constraint_implementations=ORIGINAL_AD_CONSTRAINTS

    def _reseed_current_loci(self,vector=None):
        # Legal delivered finite-piece coordinates are numerical initial values.
        # A full all-source dense contour search is not a startup requirement.
        # Invalid optical-domain seeds are still repaired by the public repair
        # before this hook; the original constrained solver then adjusts loci.
        working=self.params.initial if vector is None else np.array(vector,dtype=np.float64,copy=True)
        return working,[]

    def _world(self,model,stations,*,need_station):
        p=self.params;xyz=np.empty((p.count,3));ds=np.empty_like(xyz) if need_station else None
        board=model.scene.board(model.state);accelerated=hasattr(self,'gpu')
        if accelerated:
            if not hasattr(self,'board_gpu'):self.board_gpu=CudaBoardBatch(p.count,device=self.gpu.device)
            q=np.empty((p.count,2));dq=np.empty_like(q) if need_station else None
        for curve,rr in p.world_groups.items():
            if accelerated:
                value=model.scene.unwrapped_path_jacobian(curve,stations[rr],model.state,board=board) if need_station else None
                q[rr]=value['unwrapped'] if need_station else model.scene.unwrapped_path(curve,stations[rr],model.state,board=board)
                if need_station:dq[rr]=value['jacobian_station']
            elif need_station:
                value=model.scene.world_path_jacobian(curve,stations[rr],model.state,board=board);xyz[rr]=value['xyz'];ds[rr]=value['jacobian_station']
            else:xyz[rr]=model.scene.world_path(curve,stations[rr],model.state,board=board)
        if accelerated:
            result=self.board_gpu.evaluate(board,q)
            if not result['valid'].all():raise ForwardDomainRejection('recovered continuous-board chart rejected advertisement trial')
            xyz[:]=result['xyz'][0];self._ad_board_tangent=result['jacobian_station'][0]
            if need_station:ds[:]=self._ad_board_tangent*dq[:,0,None];ds[:,2]+=dq[:,1]
        for row,feature in p.fixed_template_landmarks.items():
            xyz[row]=model.scene.landmark_world(feature,model.state,board=board)
            if need_station:ds[row]=0.
        return xyz,ds

    def _compute(self,x,need_jacobian):
        p=self.params
        if self._last_x is not None and np.array_equal(x,self._last_x) and (not need_jacobian or self._jacobian_ready):return self._cache
        started=time.perf_counter()
        try:
            model=p.model_for(x);op=model.operator('S01');stations,rows,alpha=p.auxiliaries(x,model)
            xyz,dstation=self._world(model,stations,need_station=need_jacobian)
            tau=op.camera_definition['signed_readout_seconds'];h=p.rig.image_height-1
            times=p.frame_times+tau*(rows/h-.5)
            self.gpu.set_state(op);self.gpu.set_rows(xyz=np.ascontiguousarray(xyz),time=np.ascontiguousarray(times))
            forward=self.gpu.evaluate() if need_jacobian else self.gpu.evaluate_uv()
            if not forward['valid'].all():
                bad=np.flatnonzero(~forward['valid']);i=int(bad[0])
                raise ForwardDomainRejection('fixed original camera rejected current ad trial: '+p.observation_ids[i]+' status '+str(int(forward['status'][i])))
            target=p.target_uv.copy();query_derivative=np.zeros((p.count,2))
            for row in np.flatnonzero(p.kinds==2):target[row],query_derivative[row]=p.profiles[int(p.raw_observation_rows[row])].evaluate(alpha[row],derivative=True)
            error=np.ascontiguousarray(forward['uv']-p.target_uv);query_error=np.ascontiguousarray(forward['uv']-target)
            cache={'model':model,'operator':op,'stations':stations,'rows':rows,'alpha':alpha,'xyz':xyz,'times':times,'forward':forward,
                'error':error,'query_error':query_error,'roweq':rows-forward['uv'][:,1],'query_derivative':query_derivative,'target_uv_current_query':target}
            if need_jacobian:
                self.Jprojection.fill(0);board=model.scene.board(model.state)
                layout_jac={}
                for curve,rr in p.world_groups.items():
                    key=p.curve_group_keys[curve];b=p.semantic_to_native[key];loc=p.block_row_locations[b]
                    if b not in layout_jac:layout_jac[b]=np.zeros((len(loc),2,p.block_sizes[b]))
                    derivative=model.scene.unwrapped_layout_jacobian(curve,stations[rr],model.state,board=board)['jacobian_layout']
                    world=self._ad_board_tangent[rr,:,None]*derivative[:,0,None,:];world[:,2,:]+=derivative[:,1,:]
                    layout_jac[b][np.searchsorted(loc[:,0],rr)]=np.einsum('nij,njk->nik',forward['jacobian_world'][rr],world)
                for b,jac in layout_jac.items():self._write_block(b,jac)
                station_pixel=np.einsum('nij,nj->ni',forward['jacobian_world'],dstation)
                local_width=np.asarray(p.block_sizes,dtype=np.int64)[p.local_block]
                uvrow=forward['jacobian_time']*(tau/h)
                for component in range(2):
                    base=self._local_offsets+component*local_width;selected=p.station_column>=0
                    self.Jprojection[base[selected]+p.station_column[selected]]=station_pixel[selected,component]
                    self.Jprojection[base+p.row_column]=uvrow[:,component]
                np.copyto(self.Jerror,self.Jprojection);selected=p.alpha_column>=0
                for component in range(2):
                    base=self._local_offsets+component*local_width
                    self.Jerror[base[selected]+p.alpha_column[selected]]=-query_derivative[selected,component]
                self.counters['full_linearizations']+=1
            self.counters['forward_calls']+=1
            self.timings['last_full_linearization_seconds' if need_jacobian else 'last_residual_evaluation_seconds']=time.perf_counter()-started
            self._last_x=np.array(x,copy=True);self._jacobian_ready=need_jacobian;self._cache=cache
            return cache
        except (ValueError,FloatingPointError) as exc:
            self.counters['domain_rejections']+=1;self.domain_rejection_index+=1
            event={'schema':'rinksolve.original_ads_domain_rejection.v1','error':str(exc),'trial_index':self.domain_rejection_index,
                'camera_ice_board_parameters_cannot_change':True,'native_parameters_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
            self.last_domain_failure=event
            if self.domain_event_callback is not None:self.domain_event_callback(event,x)
            self._last_x=None;self._jacobian_ready=False
            raise ForwardDomainRejection(str(exc)) from exc


    def current_diagnostics(self,x):
        result=super().current_diagnostics(x);p=self.params;cache=self._compute(x,False)
        error_norm=np.linalg.norm(cache['error'],axis=1);worst={}
        for curve,rr in p.world_groups.items():
            label=p.curve_lookup[curve].label;ad=label.split('.')[0].split(':')[0].split('/')[0]
            i=int(rr[np.argmax(error_norm[rr])])
            if ad not in worst or error_norm[i]>worst[ad][0]:worst[ad]=(float(error_norm[i]),i,curve)
        budget_by_row=np.full(p.count,np.nan);budget_by_row[self.budget_rows]=self.budgets
        records={}
        for ad,(norm,i,curve) in sorted(worst.items()):
            definition=cache['model'].scene.resolved_definition(p.curve_lookup[curve],cache['model'].state)
            records[ad]={'observation_id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),
                'source_seconds':float(p.native_pts[i]*p.shot.time_base_num/p.shot.time_base_den),
                'raw_uv':p.target_uv[i].tolist(),'predicted_uv':cache['forward']['uv'][i].tolist(),
                'error_px':cache['error'][i].tolist(),'error_norm_px':norm,'curve_id':curve,'curve_label':p.curve_lookup[curve].label,
                'finite_piece':definition.get('finite_semantic_arc',definition.get('semantic_arc',curve)),
                'source_feature_id':definition.get('source_template_feature_id'),
                'station':float(cache['stations'][i]),'formal_row':float(cache['rows'][i]),'row_equality_px':float(cache['roweq'][i]),
                'semantic_kind':int(p.kinds[i]),'world_locus_station_variable':bool(p.station_column[i]>=0),
                'fixed_artwork_corner':i in p.fixed_template_landmarks,
                'engineering_E_px':float(budget_by_row[i]) if np.isfinite(budget_by_row[i]) else None,
                'withdrawn_or_reassociated_due_to_error':False}
        result.update(per_ad_worst_source=records,recovered_camera_motion_ice_board_frozen_sha256=p.frozen_block_sha256,
            source_statistics_scope='active advertisement sources only; no full-source b=1 claim',
            finite_legal_station_seeds_reused_without_dense_global_search=True)
        return result


def prepare_original_ad_scope(source,model,output,*,data_root):
    """Reuse current prepared evidence; only bind/scope the affected ad graph.

    Original UV, PTS, IDs, sigma, E and complete W values are never regenerated.
    A cross-scope complete W factor must retain constant rows; this adapter
    rejects such a scope instead of silently cropping or discarding it.
    """
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/'manifest.json').exists():raise FileExistsError('independent ad preparation already exists')
    active=source.selection_current_active_rows();bound=source.bound_curve_ids(active)
    lookup={c.id:c for c in model.scene.curves}
    ad_curves={key for key,c in lookup.items() if model.scene.resolved_definition(c,model.state).get('shared_layout_block_id')}
    admitted=np.array([str(key) in ad_curves for key in bound])
    excluded=active[~admitted];excluded_set=set(map(int,excluded));group_ids=[];mixed=[]
    for i,g in enumerate(source.manifest['groups']):
        if not g.get('active_statistical'):continue
        lo,hi=source.arrays['group_member_offsets'][i:i+2];members=source.arrays['group_member_rows'][lo:hi]
        if any(int(r) in excluded_set for r in members):
            group_ids.append(g['id'])
            if any(int(r) not in excluded_set for r in members):mixed.append(g['id'])
    if mixed:raise ValueError('cross-entity positive-rank W requires explicit constant-row support: '+repr(mixed))
    supplements=tuple(v['id'] for v in source.manifest.get('new_intersection_observations',[]))
    view=apply_selection(source,SelectionSnapshot(int(source.manifest.get('selection_revision',0)),excluded,excluded,supplements,tuple(group_ids),{}))
    meta=view.manifest;arrays=view.arrays
    meta.update(model_id=model.id,prepared_against_model_revision=model.revision,model_coordinate_system_id=model.scene.coordinate_system_id,
        current_model_state_initialization_sha256=model.state.digest(),numerical_data_scope='original_S01_static_ads',numerical_stage_id='original_ads',
        original_source_preparation_replayed=False,current_source_preparation=str(source.directory/'manifest.json'),generation=int(meta['generation'])+1,
        source_admission_changed_by_stage=False,original_camera_ice_board_frozen=True)
    meta['original_ad_scope']={'original_active_rows':len(active),'current_ad_source_rows':len(view.selection_current_active_rows()),
        'excluded_non_ad_rows':len(excluded),'mixed_complete_W_factors_withdrawn_as_whole':mixed,
        'new_independent_sigma_invented':False,'prior_bad_associations_restored':False,'raw_UV_PTS_ID_W_values_changed':False}
    meta['new_intersection_observations']=[];meta['new_intersection_landmarks']=[]
    current_registry={c.id:c for c in model.scene.curves}
    for entry in meta['curve_registry']:
        if entry['id'] in current_registry:
            c=current_registry[entry['id']];entry.update(label=c.label,entity_id=c.entity_id,kind=c.kind)
    out.mkdir(parents=True,exist_ok=True);refs=[]
    for prior in source.manifest['array_refs']:
        key=prior['id'].removeprefix('preparation.');value=arrays[key]
        if value is source.arrays.get(key):refs.append(prior);continue
        path=out/(key+'.npy');np.save(path,value,allow_pickle=False)
        refs.append(S.to_json_value(_incremental_ref(key,path,value,meta['generation'],source.manifest['array_refs'])))
    whitening=out/'whitening.npz';save_npz(whitening,view.whitening)
    meta['array_refs']=refs;meta['whitening_path']=str(whitening)
    meta['report'].update(prepared_against_model_revision=model.revision,status='current_original_S01_ad_only_prepared',numerical_data_scope='original_S01_static_ads')
    (out/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    (out/'report.json').write_text(json.dumps(meta['original_ad_scope'],ensure_ascii=False,indent=2),encoding='utf8')
    return load_prepared(out,data_root=data_root)
