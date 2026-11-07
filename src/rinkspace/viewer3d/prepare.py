"""Bake existing articulated results to display meshes; never fit or infer poses."""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,time,uuid
import numpy as np
from scipy.spatial import cKDTree
from calibration.program_tracking.body_tracking.rebuild_body_geometry import CompactBody
from calibration.program_tracking.body_tracking.footwear_geometry import extra_rotations,equipment_world,lowest_blade_world
from .data import ResultCatalog,CachedSequence,load_npz,digest,source_support_state,_rotations,MESH_CACHE_SCHEMA


def _body_parameters(packet):
    beta=np.asarray(packet['shared_betas'])
    scale=np.asarray(packet['body_scale'])
    if beta.ndim!=1 or not len(beta) or beta.dtype.kind not in 'fiu' or not np.isfinite(beta).all():
        raise ValueError('Invalid surface shared_betas: finite one-dimensional coefficients required')
    if scale.shape!=() or scale.dtype.kind not in 'fiu' or not np.isfinite(scale) or float(scale)<=0:
        raise ValueError('Invalid surface shared_body_scale: positive finite scalar required')
    return beta.copy(),float(scale)


def _variant_surface_packet(catalog,saved):
    """Own only this surface's shape/scale; never mutate the shared catalog."""
    packet=dict(catalog.packet)
    if 'shared_betas' in saved:
        beta=np.asarray(saved['shared_betas'])
        expected=np.asarray(catalog.packet['shared_betas'])
        if beta.shape!=expected.shape or beta.ndim!=1 or not len(beta) or beta.dtype.kind not in 'fiu' or not np.isfinite(beta).all():
            raise ValueError('Invalid saved surface shared_betas dimensions/values')
        packet['shared_betas']=beta.copy()
    if 'shared_body_scale' in saved:
        scale=np.asarray(saved['shared_body_scale'])
        if scale.shape!=() or scale.dtype.kind not in 'fiu' or not np.isfinite(scale) or float(scale)<=0:
            raise ValueError('Invalid saved surface shared_body_scale')
        packet['body_scale']=scale.copy()
    return packet


def _surface_matches_packet(surface,packet):
    actual=getattr(surface,'packet',{})
    return ('shared_betas' in actual and 'body_scale' in actual
            and np.array_equal(actual['shared_betas'],packet['shared_betas'])
            and np.array_equal(actual['body_scale'],packet['body_scale']))


class FullSurface:
    """Expand the exact existing compact model to its full SMPL-X surface."""
    def __init__(self,catalog,packet=None):
        self.packet=dict(catalog.packet if packet is None else packet)
        beta,scale=_body_parameters(self.packet)
        self.packet.update(shared_betas=beta.copy(),body_scale=np.array(scale))
        p=self.packet
        self.body=CompactBody(catalog.root/'compact_smplx.npz',p['shared_betas'],float(p['body_scale']))
        with np.load(catalog.model_path,allow_pickle=False) as z:
            template=z['v_template'].astype(np.float32)
            shape=z['shapedirs'][:,:,:len(p['shared_betas'])].astype(np.float32)
            self.posedirs=z['posedirs'].transpose(2,0,1).reshape(486,-1).copy()
            self.weights=z['weights'].astype(np.float32);self.faces=z['f'].astype(np.int32)
        self.shaped=template+np.einsum('k,vck->vc',p['shared_betas'].astype(np.float32),shape)
        _,ids=cKDTree(template).query(self.body.z['v_template'])
        np.testing.assert_allclose(template[ids],self.body.z['v_template'],rtol=0,atol=1e-7)
        np.testing.assert_allclose(self.weights[ids],self.body.W,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(shape[ids],self.body.z['shapedirs'],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(self.posedirs.reshape(486,-1,3)[:,ids],self.body.z['posedirs'].reshape(486,-1,3),rtol=1e-7,atol=1e-8)
        self.compact_vertex_ids=ids
        # One immutable selected-vertex cache per surface; the usual foot
        # subset replaces the one-off complete neutral surface without keeping
        # a second full FP64 model resident.
        self._sample_double_cache=None

    def forward(self,pose,delta,leg):
        result=self.body.forward(pose,delta,additional_rotations=extra_rotations(leg),with_frames=True)
        return self._surface_from_compact(*result)

    def forward_rotations(self,local_R,*,hand_local_R=None):
        """Use actual body/hand rotations once, without adding a hand mean."""
        kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
        result=self.body.forward_rotations(local_R,with_frames=True,**kwargs)
        return self._surface_from_compact(*result,**kwargs)

    def sample_vertices(self,joints,frames,vertex_ids=None,*,hand_local_R=None,double_precision=False):
        """Skin full or selected SMPL-X vertices with the same FK/LBS core.

        joints/frames are CompactBody's body-local native22 outputs. The return
        uses that body's local length unit and scale; no world Q/P is applied.
        This path performs no COCO check and never expands unselected vertices.
        double_precision promotes the existing stored shape/model values and
        every skinning operation to float64 for stable numerical derivatives;
        the default preserves the established float32 display calculations.
        """
        n=len(joints);rot=np.empty((n,55,3,3));position=np.empty((n,55,3));local=np.broadcast_to(np.eye(3),(n,55,3,3)).copy()
        rot[:,:22]=frames;position[:,:22]=joints/self.body.scale+self.body.J[0]
        for j in range(1,22):local[:,j]=np.swapaxes(frames[:,self.body.parents[j]],-1,-2)@frames[:,j]
        local[:,22:]=self.body.default
        if hand_local_R is not None:
            # Standard native55: face 22..24, left hand 25..39, right 40..54.
            hands=_rotations(hand_local_R,(n,2,15,3,3),'surface hand local rotations')
            local[:,25:55]=hands.reshape(n,30,3,3)
        for j in range(22,55):
            parent=self.body.parents[j];rot[:,j]=rot[:,parent]@local[:,j]
            position[:,j]=position[:,parent]+np.einsum('nij,j->ni',rot[:,parent],self.body.J[j]-self.body.J[parent])
        translation=position-np.einsum('njab,jb->nja',rot,self.body.J)
        ids=None if vertex_ids is None else np.asarray(vertex_ids)
        if ids is not None and (ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(self.shaped))):
            raise ValueError('Invalid SMPL-X vertex indices')
        if double_precision:
            # Preserve order and duplicates in the exact selection key. The
            # cache belongs to this model instance and length scale. Changing
            # the model arrays/rotation owner invalidates it; these prepared
            # model tensors are otherwise immutable during fitting.
            ids_key=None if ids is None else (ids.dtype.str,ids.tobytes())
            key=(id(self.body),self.body.scale,id(self.shaped),id(self.posedirs),id(self.weights),ids_key)
            entry=getattr(self,'_sample_double_cache',None)
            if entry is None or entry[0]!=key:
                if ids is None:
                    shaped=self.shaped.astype(np.float64)
                    posedirs=self.posedirs.astype(np.float64)
                    weights=self.weights.astype(np.float64)
                else:
                    shaped=self.shaped[ids].astype(np.float64)
                    posedirs=self.posedirs.reshape(486,-1,3)[:,ids].reshape(486,-1).astype(np.float64)
                    weights=self.weights[ids].astype(np.float64)
                # Cache only immutable model tensors; hands vary per state.
                posedirs=np.ascontiguousarray(posedirs)
                weights_T=np.ascontiguousarray(weights.T)
                for array in (shaped,posedirs,weights_T):array.setflags(write=False)
                entry=(key,(shaped,posedirs,weights_T))
                # Publish only a completed immutable entry. Concurrent first
                # calls may duplicate preparation; they share no work buffer.
                self._sample_double_cache=entry
            shaped,posedirs,weights_T=entry[1]
            if hand_local_R is None:
                # Preserve the legacy constant-tail arithmetic and geometry.
                tail=(self.body.default-np.eye(3)).reshape(297)
                default_shaped=shaped+(tail@posedirs[189:]).reshape(-1,3)
                feature=(local[:,1:22]-np.eye(3)).reshape(n,189)
                posed=default_shaped+(feature@posedirs[:189]).reshape(n,-1,3)
            else:
                feature=(local[:,1:]-np.eye(3)).reshape(n,486)
                posed=shaped+(feature@posedirs).reshape(n,-1,3)
            blend_r=(rot.reshape(n,55,9).transpose(0,2,1)@weights_T).transpose(0,2,1).reshape(n,len(shaped),3,3)
            blend_t=(translation.transpose(0,2,1)@weights_T).transpose(0,2,1)
            vertices=np.einsum('nvab,nvb->nva',blend_r,posed)+blend_t
            return (vertices-self.body.J[0])*self.body.scale
        # Keep the existing display path and float32 operation order unchanged.
        feature=(local[:,1:]-np.eye(3)).reshape(n,486).astype(np.float32)
        if ids is None:shaped,posedirs,weights=self.shaped,self.posedirs,self.weights
        else:
            shaped=self.shaped[ids];posedirs=self.posedirs.reshape(486,-1,3)[:,ids].reshape(486,-1);weights=self.weights[ids]
        posed=shaped+(feature@posedirs).reshape(n,-1,3)
        r=rot.astype(np.float32);t=translation.astype(np.float32)
        blend_r=np.einsum('vj,njab->nvab',weights,r,optimize=True)
        blend_t=np.einsum('vj,nja->nva',weights,t,optimize=True)
        vertices=np.einsum('nvab,nvb->nva',blend_r,posed)+blend_t
        return (vertices-self.body.J[0])*self.body.scale

    def _surface_from_compact(self,coco,joints,proxy,frames,*,hand_local_R=None):
        kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
        vertices=self.sample_vertices(joints,frames,**kwargs)
        model_vertices=vertices[:,self.compact_vertex_ids]/self.body.scale+self.body.J[0]
        full_coco=(np.einsum('vj,nvc->njc',self.body.reg,model_vertices)-self.body.J[0])*self.body.scale
        error=float(np.max(abs(full_coco-coco)))
        return vertices,coco,joints,proxy,frames,error


def _reconstruction_error(actual,expected,label):
    """Never turn NaN comparisons or broadcasted shapes into a zero error."""
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid display reconstruction comparison: '+label)
    error=float(np.max(abs(actual-expected)))
    if not np.isfinite(error):raise ValueError('Nonfinite display reconstruction error: '+label)
    return error


def prepare_variant(catalog,variant,surface=None):
    path,identity=catalog.cache_path(variant)
    if (path/'metadata.json').is_file():return path
    v=catalog.variant(variant) if isinstance(variant,str) else variant
    source_flags,warning_identity=catalog.source_identity()
    if warning_identity!=identity['source_identity_sha256']:
        raise ValueError('Source identity warnings changed during cache preparation')
    state=catalog.parameters(v)
    surface_packet=_variant_surface_packet(catalog,state['saved'])
    has_saved_shape=any(key in state['saved'] for key in ('shared_betas','shared_body_scale'))
    if surface is None or (has_saved_shape and not _surface_matches_packet(surface,surface_packet)):
        surface=FullSurface(catalog,packet=surface_packet)
    catalog.cache_root.mkdir(parents=True,exist_ok=True)
    staging=catalog.cache_root/('.preparing_'+v['id']+'_'+uuid.uuid4().hex[:8]);staging.mkdir()
    fullbody=v['kind']=='fullbody';ratio=state['unit_ratio']
    n=len(catalog.pts);vertices=np.lib.format.open_memmap(staging/'vertices.npy',mode='w+',dtype=np.float32,shape=(n,len(surface.shaped),3))
    joints=np.empty((n,22,3));cocos=np.empty((n,17,3));gear=np.empty((n,2,48 if fullbody or state['geometry'] is not None else 5,3));max_surface_error=0.;max_saved_error=0.;max_native_error=0.;max_blade_error=0.;start=time.perf_counter()
    with (staging/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for a in range(0,n,16):
            b=min(a+16,n)
            if fullbody:
                hand=state.get('hand_local_R')
                kwargs={} if hand is None else dict(hand_local_R=hand[a:b])
                local,coco,native,proxy,frames,error=surface.forward_rotations(state['local_R'][a:b],**kwargs)
            else:local,coco,native,proxy,frames,error=surface.forward(catalog.packet['pose_axis_angle'][a:b],state['delta'][a:b],state['leg'][a:b])
            if not np.isfinite(error) or error<0.:raise ValueError('Nonfinite or invalid full/compact surface error')
            max_surface_error=max(max_surface_error,error)
            local,coco,native=local*state['length_ratio'],coco*state['length_ratio'],native*state['length_ratio']
            P,Q=state['P'][a:b],state['Q'][a:b]
            vertices[a:b]=((np.einsum('nij,nvj->nvi',Q,local)+P[:,None])*ratio).astype(np.float32)
            bodyworld=P[:,None]+np.einsum('nij,nkj->nki',Q,coco);cocos[a:b]=bodyworld*ratio
            nativeworld=P[:,None]+np.einsum('nij,nkj->nki',Q,native);joints[a:b]=nativeworld*ratio
            saved=state['saved'].get('COCO_world_mid',state['saved'].get('COCO_world_m'))
            if saved is not None:max_saved_error=max(max_saved_error,_reconstruction_error(bodyworld,saved[a:b],'COCO world'))
            if fullbody:
                max_native_error=max(max_native_error,_reconstruction_error(nativeworld,state['saved']['native22_world_mid'][a:b],'native joints world'))
                # Unified candidates own their saved rigid equipment. The old
                # shared_geometry is provenance, never a second attachment pass.
                gear[a:b]=state['saved']['gear_world_mid'][a:b]*ratio
            elif state['geometry'] is not None:
                points,_,boot_r,boot_o=equipment_world(P,Q,native,frames,state['geometry']);gear[a:b]=points*ratio
                minimum=lowest_blade_world(boot_r,boot_o,state['geometry'])
                if 'nominal_blade_minimum_world_m' in state['saved']:max_blade_error=max(max_blade_error,_reconstruction_error(minimum,state['saved']['nominal_blade_minimum_world_m'][a:b],'blade minimum'))
            else:gear[a:b]=(P[:,None,None]+np.einsum('nij,nfkj->nfki',Q,proxy))*ratio
            # Check the actual stored float32 vertices too: a finite float64
            # transformation can overflow during the display-format cast.
            for label,values in (('vertices',vertices[a:b]),('joints',joints[a:b]),('coco',cocos[a:b]),('gear',gear[a:b])):
                if not np.isfinite(values).all():raise ValueError('Nonfinite display geometry: '+label)
            log.write(json.dumps(dict(variant=v['id'],prepared_frames=b,total_frames=n,seconds=time.perf_counter()-start))+'\n')
        vertices.flush();del vertices
        if max_saved_error>1e-7 or max_native_error>1e-7 or max_surface_error>2e-5 or max_blade_error>1e-7:raise ValueError(f'Display geometry differs from saved results: {max_saved_error}, {max_native_error}, {max_surface_error}, {max_blade_error}')
        P=state['P']*ratio;G=P.copy();G[:,2]=0
        if fullbody:
            _reconstruction_error(G,state['saved']['G_world_mid']*ratio,'saved pelvis projection')
            if not np.allclose(G,state['saved']['G_world_mid']*ratio,rtol=0,atol=1e-7):raise ValueError('Saved G differs from the pelvis vertical projection')
            G=state['saved']['G_world_mid']*ratio
        np.save(staging/'faces.npy',surface.faces)
        lower=state['saved']['source_lower_body_inferred'] if fullbody else catalog.packet['lower_body_inferred']
        cache_state=dict(pts=catalog.pts,P=P,G=G,Q=state['Q'],joints=joints,coco=cocos,gear=gear,parents=surface.body.parents[:22],source_lower_body_inferred=lower,source_identity_suspect=source_flags)
        cache_state.update(source_support_state(state['saved'],n))
        cache_state.update(state['motion_diagnostics'])
        for key,shape in (('grasp_palm_world_mid',(n,3)),('grasp_segment_world_mid',(n,2,3)),('palm_to_segment_m_mid',(n,))):
            if key in state['saved']:
                values=np.asarray(state['saved'][key])
                if values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid '+key)
                cache_state[key]=values*ratio
        for key in ('grasp_geometry_json','grasp_relations_json'):
            if key in state['saved']:
                value=np.asarray(state['saved'][key])
                if value.shape!=() or value.dtype.kind!='U':raise ValueError('Invalid '+key)
                json.loads(str(value));cache_state[key]=value.copy()
        if state.get('hand_local_R') is not None:
            cache_state['hand_local_R_mid']=state['hand_local_R']
            for key in ('hand_pose_schema_version','hand_pose_origin','hand_mean_convention','hand_prior_sha256'):
                if key in state['saved']:cache_state[key]=state['saved'][key]
        if fullbody:
            cache_state['window_mask']=state['saved']['window_mask']
            cache_state.update(state['footwear'])
        np.savez_compressed(staging/'state.npz',**cache_state)
        versioned_skate=fullbody and 'template_version' in state['footwear']
        dynamic_surface=fullbody and 'boot_surface' in state['footwear']
        equipment_description='saved fixed-topology boot surface with local dynamic upper; rigid lower-boot and blade attachment' if dynamic_surface else ('saved rigid 14-vertex shell and lower-edge strip' if fullbody or state['geometry'] is not None else 'original five-point blade proxy; no fitted boot shell in this old variant')
        if versioned_skate:equipment_description='saved connected lower-skate solids, fixed-topology articulated upper and cavity, exact dual edges and contact witnesses; legacy landmarks are observation adapters only'
        metadata=dict(schema=MESH_CACHE_SCHEMA,identity=identity,variant=v,frames=n,vertices_per_frame=len(surface.shaped),source_time_base=list(getattr(catalog,'time_base',(1,12800))),source_first_last_pts=[int(catalog.pts[0]),int(catalog.pts[-1])],world_up='Z',display_unit='metre',display_metres_per_scene_unit=catalog.scale,old_result_metres_per_scene_unit=catalog.legacy_scale,candidate_metres_per_scene_unit=state['metres_per_scene_unit'],body_length_ratio=state['length_ratio'],uniform_unit_reexpression_ratio=ratio,time_semantics='all body vertices, native joints and equipment evaluated at common source mid-exposure; rolling-shutter row samples are not mixed into a 3D pose',surface='full 10475-vertex SMPL-X model with effective local_R_mid' if fullbody else 'full 10475-vertex SMPL-X model with existing shared beta and pose corrections',equipment=equipment_description,source_result=str(catalog.result_path(v).resolve()),saved_coco_mid_maximum_reconstruction_error_m=max_saved_error,saved_native22_mid_maximum_reconstruction_error_m=max_native_error,full_vs_compact_coco_maximum_error_m=max_surface_error,saved_nominal_blade_minimum_maximum_error_m=max_blade_error,no_new_pose_or_root_fit=True,no_neural_inference=True,no_foot_lock_or_ground_snapping=True,accuracy_accepted=False,seconds=time.perf_counter()-start)
        actual_surface_packet=getattr(surface,'packet',surface_packet)
        metadata.update(shared_betas=None if 'shared_betas' not in actual_surface_packet else np.asarray(actual_surface_packet['shared_betas']).tolist(),
            shared_body_scale=None if 'body_scale' not in actual_surface_packet else float(actual_surface_packet['body_scale']),
            surface_parameter_source=dict(shared_betas='saved variant' if 'shared_betas' in state['saved'] else 'catalog default',
                shared_body_scale='saved variant' if 'shared_body_scale' in state['saved'] else 'catalog default'))
        metadata['motion_diagnostics']=dict(saved_fields=list(state['motion_diagnostics']),
            com='saved conditional-metric mass-model approximation; not measured true centre of mass',
            blade_material_order=['rear','middle','front'],foot_order='existing saved foot 0/1; anatomical identity not inferred',
            witness='saved moving minimum/contact candidate; not verified ice contact',
            estimated_by_viewer=False)
        if fullbody:
            footwear=state['footwear']
            if 'body_refinement_window_mask' in footwear:
                body_frames=int(np.count_nonzero(footwear['body_refinement_window_mask']))
                reference=v.get('body_reference_label','fullbody1102')
                intervals=v.get('body_refinement_intervals_seconds',getattr(catalog,'config',{}).get('body_refinement_intervals_seconds',[]))
                metadata.update(optimized_frames=body_frames,body_refinement_frames=body_frames,
                                parent_window_frames=int(np.count_nonzero(state['saved']['window_mask'])),
                                body_reference_label=reference,body_reference='parent '+reference+' trajectory',
                                body_refinement_intervals_seconds=intervals,
                                outside_window='body inherited from parent '+reference+' trajectory; shared footwear binding can still apply')
            else:metadata.update(optimized_frames=int(np.count_nonzero(state['saved']['window_mask'])),outside_window=catalog.config.get('display_outside_window_label','old J1 reference, reexpressed in candidate length units'))
            if 'footwear_binding_applied_mask' in footwear:
                binding_frames=int(np.count_nonzero(footwear['footwear_binding_applied_mask']))
                metadata.update(footwear_binding_applied_frames=binding_frames,
                                footwear_binding_scope=catalog.config.get('display_footwear_binding_scope','shared across all S01 exposures' if binding_frames==n else 'saved application mask'))
            metadata['candidate_provenance']={k:value.tolist() for k,value in state['saved'].items() if ('reference' in k or 'ledger' in k) and k!='skate_reference_world_mid' and value.dtype.kind in 'USbiuf'}
            metadata['footwear_display']=dict(topology='saved' if versioned_skate else v.get('shell_topology','saved' if dynamic_surface else 'open_collar'),saved_dynamic_surface=dynamic_surface or versioned_skate,retriangulated=False,coordinates_modified=False)
            if versioned_skate:
                metadata['footwear_display'].update(template_version=str(footwear['template_version']),binding_version=str(footwear['binding_version']),parameter_provenance=json.loads(str(footwear['parameter_provenance_json'])),contact_witness='moving minimum; not a material-point velocity',support_gap='eligible support-group minimum raw height; NaN means unknown',clearance='minimum height plus semantic allowed entry')
        (staging/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    assert staging.resolve().is_relative_to(catalog.cache_root.resolve()) and path.resolve().is_relative_to(catalog.cache_root.resolve())
    if path.exists():raise FileExistsError('Another cache preparation finished; preserve both paths')
    staging.rename(path)
    print(json.dumps(dict(variant=v['id'],frames=n,seconds=time.perf_counter()-start,path=str(path))),flush=True)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--data-root');p.add_argument('--config');p.add_argument('--variant');p.add_argument('--all',action='store_true');a=p.parse_args();catalog=ResultCatalog(a.project,a.data_root,a.config)
    variants=catalog.variants if a.all else [catalog.variant(a.variant or catalog.config['default_variant'])]
    for v in variants:
        path,_=catalog.cache_path(v)
        if (path/'metadata.json').is_file():continue
        prepare_variant(catalog,v)
    print(json.dumps(dict(ready=True,variants=[v['id'] for v in variants])),flush=True)

if __name__=='__main__':main()
