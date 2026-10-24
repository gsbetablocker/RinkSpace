"""Native S01 parameter views: physical owners and numerical auxiliaries distinct."""
from __future__ import annotations
from dataclasses import asdict,replace
import numpy as np
from .model import pack_reference_gauge,unpack_reference_gauge,TARGET_MOUNT
from .native_backend import BatchLayout


class S01Parameterization:
    def __init__(self,model,prepared,*,intersection_enabled,ad7_corners_enabled=True):
        self.model,self.prepared=model,prepared
        self.shot=next(s for s in model.shots if s.id=='S01')
        self.rig=next(r for r in model.rigs if r.id==self.shot.rig_id)
        if self.rig.mount_family!=TARGET_MOUNT:raise ValueError('actual solver requires the explicit constant-lever target model')
        self.original_rows=prepared.selection_current_active_rows()
        self.original_count=len(self.original_rows)
        a=prepared.arrays;rows=self.original_rows
        ids=list(prepared.observation_ids(rows))
        self.curve_ids=list(prepared.bound_curve_ids(rows))
        self.native_pts=np.array(a['native_pts'][rows],dtype=np.int64)
        self.target_uv=np.array(a['immutable_uv'][rows],dtype=np.float64)
        self.kinds=np.array(a['semantic_kind'][rows],dtype=np.uint8)
        self.station_seed=np.array(a['station_seed'][rows],dtype=np.float64)
        self.row_seed=np.array(a['row_auxiliary_seed'][rows],dtype=np.float64)
        self.alpha_seed=np.array(a['profile_alpha_seed'][rows],dtype=np.float64)
        self.profiles={p.observation_row:p for p in prepared.profiles}
        self.shared_landmark_blocks={}
        self.intersection_enabled=bool(intersection_enabled)
        if intersection_enabled:
            features=prepared.manifest['new_intersection_landmarks'];obs=prepared.manifest['new_intersection_observations']
            feature_indices=np.asarray(a['intersection_landmark_index'],dtype=np.int64)
            for i,fi in enumerate(feature_indices):self.shared_landmark_blocks[len(ids)+i]=features[fi]['station_variable_block_id']
            ids.extend(o['id'] for o in obs)
            self.curve_ids.extend(features[i]['curve_id'] for i in feature_indices)
            self.native_pts=np.r_[self.native_pts,a['intersection_native_pts']]
            self.target_uv=np.concatenate((self.target_uv,a['intersection_uv']))
            self.kinds=np.r_[self.kinds,np.full(len(obs),3,dtype=np.uint8)]
            self.station_seed=np.r_[self.station_seed,a['intersection_shared_station_seed'][feature_indices]]
            self.row_seed=np.r_[self.row_seed,a['intersection_row_auxiliary_seed']]
            self.alpha_seed=np.r_[self.alpha_seed,np.zeros(len(obs))]
        self.fixed_template_landmarks={}
        if ad7_corners_enabled:
            features={v.id:v for v in model.landmarks}
            observations=prepared.manifest.get('ad7_fixed_observations',[])
            for observation in observations:
                feature=features[observation['fixed_feature_id']]
                curve=feature.position.geometry_feature_ids[0]
                definition=model.scene.resolved_definition(next(c for c in model.scene.curves if c.id==curve),model.state)
                station=float(definition['fixed_artwork_landmarks'][feature.id]['station'])
                self.fixed_template_landmarks[len(ids)]=feature
                ids.append(observation['id']);self.curve_ids.append(curve)
                self.native_pts=np.r_[self.native_pts,int(observation['native_pts'])]
                self.target_uv=np.concatenate((self.target_uv,np.array([observation['uv_native_pixels']],dtype=np.float64)))
                self.kinds=np.r_[self.kinds,np.uint8(4)]
                self.station_seed=np.r_[self.station_seed,station]
                self.row_seed=np.r_[self.row_seed,float(observation['uv_native_pixels'][1])]
                self.alpha_seed=np.r_[self.alpha_seed,0.]
        self.observation_ids=tuple(v.decode('utf8') if isinstance(v,(bytes,np.bytes_)) else str(v) for v in ids)
        if len(set(self.observation_ids))!=len(ids):raise ValueError('duplicate current observation stable IDs')
        self.raw_observation_rows=np.r_[self.original_rows,np.full(len(ids)-len(rows),-1,dtype=np.int64)]
        self.count=len(ids);self.curve_ids=np.asarray(self.curve_ids)
        self.source_to_active=np.full(prepared.row_count,-1,dtype=np.int64);self.source_to_active[rows]=np.arange(len(rows))
        lookup={c.id:c for c in model.scene.curves};self.curve_lookup=lookup
        if any(v not in lookup for v in self.curve_ids):raise ValueError('prepared curve not present in current target model')
        family=model.scene.structural_family
        self.scene_keys=[]
        if family:
            self.scene_keys=[v for v in (family.curling_block_id,family.lane_block_id,family.hockey_block_id,family.hockey_extra_block_id,family.hockey_midline_block_id,family.hockey_end_block_id) if v]
        self.board_key=model.scene.board_block_id
        self.curve_group_keys={c.id:model.scene.resolved_definition(c,model.state).get('shared_layout_block_id') for c in model.scene.curves}
        used_group_keys={self.curve_group_keys[k] for k in self.curve_ids if self.curve_group_keys[k]}
        rig_keys={self.rig.key(s) for s in ('pivot','lever','reference_rotation_vector','pan_axis_chart','tilt_axis_chart','optics')}
        selected=rig_keys|set(self.scene_keys)|{self.board_key,self.shot.control_block_id}|used_group_keys|set(self.shared_landmark_blocks.values())
        for key in selected:
            if key not in model.state._specs:raise ValueError('target is missing required native model block: '+key)
        self.selected_blocks=tuple(b.id for b in model.state.blocks if b.id in selected)
        if any(k.startswith('geometry/') for k in self.selected_blocks):raise ValueError('sampled shape vertices cannot enter the physical free vector')
        prefix,self.views=pack_reference_gauge(model.state,self.selected_blocks,control_block_id=self.shot.control_block_id)
        self.model_parameter_count=len(prefix)
        self.block_ids=[v.id for v in self.views];self.block_sizes=[v.size for v in self.views]
        self.semantic_to_native={v.semantic_block_id:i for i,v in enumerate(self.views) if v.semantic_block_id!=self.shot.control_block_id}
        self.control_native={v.component_offset//3:i for i,v in enumerate(self.views) if v.semantic_block_id==self.shot.control_block_id}
        self.block_owners=[{'kind':v.owner.kind.value,'id':v.owner.id,'semantic_block_id':v.semantic_block_id,'component_offset':v.component_offset} for v in self.views]
        initial=list(prefix);lower=[-np.inf]*len(prefix);upper=[np.inf]*len(prefix);scales=[1.]*len(prefix)
        self.local_block=np.full(self.count,-1,dtype=np.int32)
        self.station_column=np.full(self.count,-1,dtype=np.int32);self.row_column=self.station_column.copy();self.alpha_column=self.station_column.copy()
        self.local_component_records=[]
        groups=list(prepared.nuisance_groups())
        if self.count>self.original_count:
            groups.extend({'id':'aux/'+self.observation_ids[i],'rows':np.array([],dtype=np.int64),'extra_active_row':i,'kind':'new_shared_landmark_observation'} for i in range(self.original_count,self.count))
        for group in groups:
            rr=[group['extra_active_row']] if 'extra_active_row' in group else [int(self.source_to_active[r]) for r in group['rows']]
            if not rr or any(r<0 for r in rr):raise ValueError('local ownership group is outside current active rows')
            values=[];lo=[];hi=[];scale=[];records=[]
            for i in rr:
                if self.local_block[i]!=-1:raise ValueError('observation assigned to two native local blocks')
                original=int(self.raw_observation_rows[i]);kind=int(self.kinds[i])
                if kind in (0,2):
                    station=float(self.station_seed[i])
                    if not np.isfinite(station) or not 0<=station<=1:raise ValueError('world-locus station seed outside exact finite branch: '+self.observation_ids[i])
                    self.station_column[i]=len(values);values.append(station);lo.append(0.);hi.append(1.);scale.append(1.)
                row=float(self.row_seed[i])
                if not np.isfinite(row) or not 0<=row<=self.rig.image_height-1:raise ValueError('formal row seed outside native sensor: '+self.observation_ids[i])
                self.row_column[i]=len(values);values.append(row);lo.append(0.);hi.append(float(self.rig.image_height-1));scale.append(float(self.rig.image_height-1))
                if kind==2:
                    profile=self.profiles.get(original)
                    if profile is None:raise ValueError('profile observation missing exact current rational Query')
                    alpha=float(self.alpha_seed[i]);amin,amax=profile.alpha_bounds
                    if not amin<=alpha<=amax:raise ValueError('Query alpha seed outside delivered bounds')
                    if amin!=amax:
                        self.alpha_column[i]=len(values);values.append(alpha);lo.append(amin);hi.append(amax);scale.append(profile.alpha_scale)
            native=len(self.block_ids);self.local_block[rr]=native
            self.block_ids.append(str(group['id']));self.block_sizes.append(len(values))
            self.block_owners.append({'kind':'current_source_nuisance_group','id':str(group['id']),'source_group_kind':group['kind'],'contains_shared_landmark_position':False})

            initial.extend(values);lower.extend(lo);upper.extend(hi);scales.extend(scale)
        if np.any(self.local_block<0):raise ValueError('active observation missing nuisance owner')
        self.local_group_count=len(groups)
        self.budget_block=len(self.block_ids)
        self.block_ids.append('numerical/shared_budget_sq');self.block_sizes.append(1)
        self.block_owners.append({'kind':'numerical_feasibility','id':'numerical/shared_budget_sq','meaning':'one shared squared error budget ratio; not sigma or a physical scene parameter'})
        initial.append(1.);lower.append(0.);upper.append(np.inf);scales.append(1.)
        self.block_ids=tuple(self.block_ids);self.block_sizes=tuple(self.block_sizes)
        if len(set(self.block_ids))!=len(self.block_ids):raise ValueError('native owner IDs collide')
        self.parameter_offsets=np.r_[0,np.cumsum(self.block_sizes,dtype=np.int64)]
        self.initial=np.asarray(initial,dtype=np.float64);self.lower=np.asarray(lower);self.upper=np.asarray(upper);self.scales=np.asarray(scales)
        # A shared material point has the same finite chart domain as its
        # typed position. Expose that existing domain to Ceres as ambient
        # bounds instead of waiting for world_path to reject an invalid step.
        features={v.position.station_variable_block_id:v for v in model.landmarks
                  if hasattr(v.position,'station_variable_block_id')}
        self.shared_station_bounds=[]
        for key in sorted(set(self.shared_landmark_blocks.values())):
            feature=features[key];interval=feature.position.allowed_interval
            native=self.semantic_to_native[key];offset=int(self.parameter_offsets[native])
            if self.block_sizes[native]!=1:raise ValueError('shared material station must be one scalar: '+key)
            lo=-np.inf if interval.lower is None else float(interval.lower)
            hi=np.inf if interval.upper is None else float(interval.upper)
            if not interval.lower_closed and np.isfinite(lo):lo=np.nextafter(lo,np.inf)
            if not interval.upper_closed and np.isfinite(hi):hi=np.nextafter(hi,-np.inf)
            if lo>hi or not lo<=self.initial[offset]<=hi:raise ValueError('shared material station seed outside typed interval: '+key)
            self.lower[offset]=lo;self.upper[offset]=hi
            self.shared_station_bounds.append({'feature_id':feature.id,'semantic_block_id':key,
                'native_block':native,'parameter_offset':offset,'lower':lo,'upper':hi,
                'source':'SharedCurvePointPosition.allowed_interval','new_physical_prior':False})
        self.frame_times=self.shot.frame_time(self.native_pts)
        op=model.operator('S01');left,right=op.domain
        self.tau_domain=2*min(float(self.frame_times.min()-left),float(right-self.frame_times.max()))
        if not self.tau_domain>0:raise ValueError('current trajectory cannot cover a full sensor readout')
        optics_native=self.semantic_to_native[self.rig.key('optics')]
        tau_offset=self.parameter_offsets[optics_native]+11
        self.lower[tau_offset]=-self.tau_domain;self.upper[tau_offset]=self.tau_domain
        if not self.lower[tau_offset]<=self.initial[tau_offset]<=self.upper[tau_offset]:raise ValueError('initial readout exceeds current trajectory representation domain; extend current controls')
        self.scales[tau_offset]=self.shot.step_native_pts*self.shot.time_base_num/self.shot.time_base_den
        optical_offset=self.parameter_offsets[optics_native]
        self.scales[optical_offset+1:optical_offset+5]=[self.rig.image_width,self.rig.image_height,self.rig.image_width,self.rig.image_height]
        self.world_groups={str(k):np.flatnonzero(self.curve_ids==k) for k in np.unique(self.curve_ids)}
        self.rig_native=tuple(self.semantic_to_native[k] for k in rig_keys)
        generated=family.curves(model.state) if family else {}
        self.raw_blocks=[];self.world_dependencies={}
        for identifier,ii in self.world_groups.items():
            curve=lookup[identifier];d=model.scene.resolved_definition(curve,model.state)
            dependencies=[]
            if d['kind'] in ('board_material_polyline','board_boundary'):dependencies.append(self.board_key)
            if curve.label in generated:dependencies.extend(self.scene_keys)
            if self.curve_group_keys[identifier]:dependencies.append(self.curve_group_keys[identifier])
            self.world_dependencies[identifier]=tuple(dict.fromkeys(dependencies))
        dt=self.shot.control_dt_seconds;first=self.shot.first_control_time_seconds;nc=self.shot.control_count
        for i,(identifier,t) in enumerate(zip(self.curve_ids,self.frame_times)):
            dep=set(self.rig_native)
            dep.update(self.semantic_to_native[k] for k in self.world_dependencies[str(identifier)])
            low_cell=int(np.floor((t-self.tau_domain/2-first)/dt));high_cell=int(np.floor((t+self.tau_domain/2-first)/dt))
            controls=set(range(max(0,low_cell-1),min(nc,high_cell+3)))
            if 2 in controls:controls.update((1,3))
            dep.update(self.control_native[k] for k in controls)
            if i in self.shared_landmark_blocks:dep.add(self.semantic_to_native[self.shared_landmark_blocks[i]])
            dep.add(int(self.local_block[i]));self.raw_blocks.append(tuple(sorted(dep)))
        self.raw_layout=BatchLayout(self.block_ids,self.block_sizes,self.observation_ids,(2,)*self.count,tuple(self.raw_blocks))
        self.raw_blocks=None
        self.block_row_locations={}
        for row,blocks in enumerate(self.raw_layout.factor_blocks):
            edge=int(self.raw_layout.factor_edge_offsets[row])
            for k,b in enumerate(blocks):self.block_row_locations.setdefault(b,[]).append((row,int(self.raw_layout.jacobian_offsets[edge+k])))
        self.block_row_locations={b:np.asarray(v,dtype=np.int64) for b,v in self.block_row_locations.items()}
        self.ordering_groups=np.ones(len(self.block_ids),dtype=np.int32)
        self.ordering_groups[len(self.views):]=0
        self.ordering_groups[self.budget_block]=1
        self.budget_offset=int(self.parameter_offsets[self.budget_block])

    def model_for(self,vector,*,revision=None,state_id=None):
        x=np.asarray(vector,dtype=np.float64)
        if x.shape!=self.initial.shape or not np.isfinite(x).all():raise ValueError('current complete native vector differs from bound layout')
        state=unpack_reference_gauge(self.model.state,self.views,x[:self.model_parameter_count],control_block_id=self.shot.control_block_id,new_id=state_id)
        if revision is not None:state=replace(state,revision=int(revision))
        return replace(self.model,state=state)

    def auxiliaries(self,vector,model):
        x=np.asarray(vector);base=self.parameter_offsets[self.local_block]
        stations=self.station_seed.copy();rows=x[base+self.row_column];alpha=self.alpha_seed.copy()
        selected=self.station_column>=0;stations[selected]=x[base[selected]+self.station_column[selected]]
        selected=self.alpha_column>=0;alpha[selected]=x[base[selected]+self.alpha_column[selected]]
        for row,key in self.shared_landmark_blocks.items():stations[row]=model.state.get(key)[0]
        return stations,rows,alpha

    def inventory(self):
        return {'schema':'rinksolve.s01_native_parameters.v1','model_id':self.model.id,'model_revision':self.model.revision,
            'model_state_id':self.model.state.id,'model_state_sha256':self.model.state.digest(),'prepared_path':str(self.prepared.directory),
            'native_blocks':len(self.block_ids),'native_parameters':len(self.initial),'physical_model_native_views':len(self.views),
            'observation_rows':self.count,'original_current_rows':self.original_count,'new_shared_landmark_observations':self.count-self.original_count,
            'source_row_addresses_are_semantic_ids':False,'source_observation_ids_preserved':True,
            'sampled_polyline_vertices_free':False,'formal_rs_row_auxiliary':True,'query_alpha_is_world_station':False,
            'reference_angle_gauge':'control2 alpha,beta = -.25*(control1+control3); exact chain to native views',
            'tau_domain_seconds':[-self.tau_domain,self.tau_domain],'tau_domain_source':'current complete-readout trajectory representation; extend controls if boundary reached; not hardware fact',
            'parameter_scaling':'one declared unit, source Query alpha scale, native sensor dimensions, native frame duration; numerical chart only',
            'shared_feature_station_native_bounds':self.shared_station_bounds,
            'global_views':[{'id':v.id,'semantic_block_id':v.semantic_block_id,'component_offset':v.component_offset,'size':v.size,'vector_offset':v.vector_offset,'owner':self.block_owners[i]} for i,v in enumerate(self.views)],
            'local_group_count':self.local_group_count,'local_component_storage':'SoA arrays: local_block/station_column/row_column/alpha_column, keyed by stable observation IDs; no per-component objects','selected_model_blocks':list(self.selected_blocks)}


    def save_layout(self,directory):
        from pathlib import Path
        import json
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        arrays={k:getattr(self,k) for k in ('initial','lower','upper','scales','parameter_offsets','raw_observation_rows','local_block','station_column','row_column','alpha_column','native_pts')}
        arrays['block_sizes']=np.asarray(self.block_sizes,dtype=np.int32)
        for name,strings in (('native_block_ids',self.block_ids),('observation_ids',self.observation_ids)):
            encoded=[v.encode('utf8') for v in strings]
            arrays[name+'_utf8']=np.frombuffer(b''.join(encoded),dtype=np.uint8)
            arrays[name+'_offsets']=np.r_[0,np.cumsum([len(v) for v in encoded],dtype=np.int64)]
        np.savez(path/'parameter_layout.npz',**arrays)
        metadata=self.inventory();metadata['array_file']='parameter_layout.npz';metadata['immutable_source_observation_manifest']=str(self.prepared.manifest['source_observation_manifest'])
        (path/'parameter_layout.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        return path/'parameter_layout.json'
