"""Bake existing articulated results to display meshes; never fit or infer poses."""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,time,uuid
import numpy as np
from scipy.spatial import cKDTree
from calibration.program_tracking.body_tracking.rebuild_body_geometry import CompactBody
from calibration.program_tracking.body_tracking.footwear_geometry import extra_rotations,equipment_world,lowest_blade_world
from .data import ResultCatalog,CachedSequence,load_npz,digest


class FullSurface:
    """Expand the exact existing compact model to its full SMPL-X surface."""
    def __init__(self,catalog):
        p=catalog.packet
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

    def forward(self,pose,delta,leg):
        coco,joints,proxy,frames=self.body.forward(pose,delta,additional_rotations=extra_rotations(leg),with_frames=True)
        n=len(pose);rot=np.empty((n,55,3,3));position=np.empty((n,55,3));local=np.broadcast_to(np.eye(3),(n,55,3,3)).copy()
        rot[:,:22]=frames;position[:,:22]=joints/self.body.scale+self.body.J[0]
        for j in range(1,22):local[:,j]=np.swapaxes(frames[:,self.body.parents[j]],-1,-2)@frames[:,j]
        local[:,22:]=self.body.default
        for j in range(22,55):
            parent=self.body.parents[j];rot[:,j]=rot[:,parent]@local[:,j]
            position[:,j]=position[:,parent]+np.einsum('nij,j->ni',rot[:,parent],self.body.J[j]-self.body.J[parent])
        translation=position-np.einsum('njab,jb->nja',rot,self.body.J)
        feature=(local[:,1:]-np.eye(3)).reshape(n,486).astype(np.float32)
        posed=self.shaped+(feature@self.posedirs).reshape(n,-1,3)
        r=rot.astype(np.float32);t=translation.astype(np.float32)
        blend_r=np.einsum('vj,njab->nvab',self.weights,r,optimize=True)
        blend_t=np.einsum('vj,nja->nva',self.weights,t,optimize=True)
        vertices=np.einsum('nvab,nvb->nva',blend_r,posed)+blend_t
        full_coco=(np.einsum('vj,nvc->njc',self.body.reg,vertices[:,self.compact_vertex_ids])-self.body.J[0])*self.body.scale
        error=float(np.max(abs(full_coco-coco)))
        return (vertices-self.body.J[0])*self.body.scale,coco,joints,proxy,frames,error


def prepare_variant(catalog,variant,surface=None):
    path,identity=catalog.cache_path(variant)
    if (path/'metadata.json').is_file():return path
    v=catalog.variant(variant) if isinstance(variant,str) else variant
    state=catalog.parameters(v);surface=surface or FullSurface(catalog)
    catalog.cache_root.mkdir(parents=True,exist_ok=True)
    staging=catalog.cache_root/('.preparing_'+v['id']+'_'+uuid.uuid4().hex[:8]);staging.mkdir()
    n=len(catalog.pts);vertices=np.lib.format.open_memmap(staging/'vertices.npy',mode='w+',dtype=np.float32,shape=(n,len(surface.shaped),3))
    joints=np.empty((n,22,3));cocos=np.empty((n,17,3));gear=np.empty((n,2,48 if state['geometry'] is not None else 5,3));max_surface_error=0.;max_saved_error=0.;max_blade_error=0.;start=time.perf_counter()
    with (staging/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for a in range(0,n,16):
            b=min(a+16,n);local,coco,native,proxy,frames,error=surface.forward(catalog.packet['pose_axis_angle'][a:b],state['delta'][a:b],state['leg'][a:b]);max_surface_error=max(max_surface_error,error)
            P,Q=state['P'][a:b],state['Q'][a:b]
            vertices[a:b]=((np.einsum('nij,nvj->nvi',Q,local)+P[:,None])*catalog.unit_ratio).astype(np.float32)
            bodyworld=P[:,None]+np.einsum('nij,nkj->nki',Q,coco);cocos[a:b]=bodyworld*catalog.unit_ratio
            joints[a:b]=(P[:,None]+np.einsum('nij,nkj->nki',Q,native))*catalog.unit_ratio
            saved=state['saved'].get('COCO_world_mid',state['saved'].get('COCO_world_m'))
            if saved is not None:max_saved_error=max(max_saved_error,float(np.max(abs(bodyworld-saved[a:b]))))
            if state['geometry'] is not None:
                points,_,boot_r,boot_o=equipment_world(P,Q,native,frames,state['geometry']);gear[a:b]=points*catalog.unit_ratio
                minimum=lowest_blade_world(boot_r,boot_o,state['geometry'])
                if 'nominal_blade_minimum_world_m' in state['saved']:max_blade_error=max(max_blade_error,float(np.max(abs(minimum-state['saved']['nominal_blade_minimum_world_m'][a:b]))))
            else:gear[a:b]=(P[:,None,None]+np.einsum('nij,nfkj->nfki',Q,proxy))*catalog.unit_ratio
            log.write(json.dumps(dict(variant=v['id'],prepared_frames=b,total_frames=n,seconds=time.perf_counter()-start))+'\n')
        vertices.flush();del vertices
        if max_saved_error>1e-7 or max_surface_error>2e-5 or max_blade_error>1e-7:raise ValueError(f'Display geometry differs from saved results: {max_saved_error}, {max_surface_error}, {max_blade_error}')
        P=state['P']*catalog.unit_ratio;G=P.copy();G[:,2]=0
        np.save(staging/'faces.npy',surface.faces)
        np.savez_compressed(staging/'state.npz',pts=catalog.pts,P=P,G=G,Q=state['Q'],joints=joints,coco=cocos,gear=gear,parents=surface.body.parents[:22],source_lower_body_inferred=catalog.packet['lower_body_inferred'],source_identity_suspect=load_npz(catalog.project/'events/ankara_2026/programmes/okada_mei_sp/body_s01_joint_candidate_20261029.npz')['source_identity_suspect'])
        metadata=dict(schema='rinkspace.viewer-mesh-cache.v1',identity=identity,variant=v,frames=n,vertices_per_frame=len(surface.shaped),source_time_base=[1,12800],source_first_last_pts=[int(catalog.pts[0]),int(catalog.pts[-1])],world_up='Z',display_unit='metre',display_metres_per_scene_unit=catalog.scale,old_result_metres_per_scene_unit=catalog.legacy_scale,uniform_unit_reexpression_ratio=catalog.unit_ratio,time_semantics='all body vertices, native joints and equipment evaluated at common source mid-exposure; rolling-shutter row samples are not mixed into a 3D pose',surface='full 10475-vertex SMPL-X model with existing shared beta and pose corrections',equipment='original rigid 14-vertex shell and lower-edge strip' if state['geometry'] is not None else 'original five-point blade proxy; no fitted boot shell in this old variant',source_result=str((catalog.root/v['result']).resolve()),saved_coco_mid_maximum_reconstruction_error_m=max_saved_error,full_vs_compact_coco_maximum_error_m=max_surface_error,saved_nominal_blade_minimum_maximum_error_m=max_blade_error,no_new_pose_or_root_fit=True,no_neural_inference=True,no_foot_lock_or_ground_snapping=True,accuracy_accepted=False,seconds=time.perf_counter()-start)
        (staging/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    assert staging.resolve().is_relative_to(catalog.cache_root.resolve()) and path.resolve().is_relative_to(catalog.cache_root.resolve())
    if path.exists():raise FileExistsError('Another cache preparation finished; preserve both paths')
    staging.rename(path)
    print(json.dumps(dict(variant=v['id'],frames=n,seconds=time.perf_counter()-start,path=str(path))),flush=True)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--data-root');p.add_argument('--config');p.add_argument('--variant');p.add_argument('--all',action='store_true');a=p.parse_args();catalog=ResultCatalog(a.project,a.data_root,a.config);surface=None
    variants=catalog.variants if a.all else [catalog.variant(a.variant or catalog.config['default_variant'])]
    for v in variants:
        path,_=catalog.cache_path(v)
        if (path/'metadata.json').is_file():continue
        if surface is None:surface=FullSurface(catalog)
        prepare_variant(catalog,v,surface)
    print(json.dumps(dict(ready=True,variants=[v['id'] for v in variants])),flush=True)

if __name__=='__main__':main()
