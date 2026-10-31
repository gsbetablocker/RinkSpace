"""Exact reduced SMPL-X LBS and rigid skate attachments for fixed-camera tests.

Local rotations are corrected on anatomically relevant joint axes. COCO points
use the original vertex regressor. No free XYZ joints or bone-length changes.
"""

from rinkspace.storage import resolve_asset
from pathlib import Path
import sys, json
import numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load, digest

# SMPL-X native joint index and local flexion axis. Left/right remain physical.
POSE_DOFS=((1,0),(2,0),(4,0),(5,0),(7,0),(8,0),(3,0),(6,0),(9,0))
DOF_NAMES=('left_hip_flexion','right_hip_flexion','left_knee_flexion','right_knee_flexion',
           'left_ankle_pitch','right_ankle_pitch','spine1_flexion','spine2_flexion','spine3_flexion')


def export_compact(root, project):
    """Materialize existing model tensors on CPU; never initialize CUDA."""
    project,root=Path(project),Path(root)
    sys.path.insert(0,str(resolve_asset(project, project/'environments/body/GVHMR')))
    import torch
    torch.set_num_threads(2)
    from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteCoco17
    from .pelvis_footwear_geometry import plantar_patch_indices
    model_path=resolve_asset(project, project/'environments/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz')
    full=SmplxLite(model_path=model_path,gender='neutral',num_betas=10).eval().requires_grad_(False)
    coco=SmplxLiteCoco17(model_path=model_path,gender='neutral',num_betas=10).eval().requires_grad_(False)
    inp=load(root/'input.npz');beta=inp['shared_betas']
    rest=load(root/'source_cache/neutral_shared_beta_rest_mesh.npz')
    vertices=rest['vertices_model_m'];joints=rest['native22_rest_joints_model_m']
    patches,desc=plantar_patch_indices(vertices,joints,full.lbs_weights.numpy())
    centroids=np.array([vertices[p].mean(0) for p in patches]).reshape(2,2,3)
    rest_height=centroids[:,:,1]-vertices[:,1].min()
    arrays={n:getattr(coco,n).detach().cpu().numpy() for n in ['v_template','shapedirs','posedirs','lbs_weights','parents','other_default_pose','J_template','J_shapedirs','smplx2coco17_interestd']}
    arrays.update(rest_patch_centroids=centroids,rest_patch_height=rest_height,rest_native22=joints)
    np.savez_compressed(root/'compact_smplx.npz',**arrays)
    model=CompactBody(root/'compact_smplx.npz',beta,float(inp['body_scale']))
    # Independent upstream nonzero-pose check validates exact LBS and regressor.
    chosen=np.array([150,400,900]);pose=inp['pose_axis_angle'][chosen]
    with torch.inference_mode():
        expected=coco(body_pose=torch.tensor(pose.reshape(1,-1,63),dtype=torch.float32),
            betas=torch.tensor(np.tile(beta,(len(chosen),1))[None],dtype=torch.float32),
            global_orient=torch.zeros((1,len(chosen),3)),transl=torch.zeros((1,len(chosen),3)))[0].numpy()
    actual=model.forward(pose,np.zeros((len(pose),len(POSE_DOFS))))[0]/float(inp['body_scale'])
    root_joint=model.J[0]
    error=float(np.max(np.abs(actual-(expected-root_joint))))
    if error>2e-5:raise ValueError('Compact SMPL-X differs from upstream: '+str(error))
    report=dict(schema='rinkspace.compact-smplx-cpu.v1',model_path=str(model_path),model_sha256=digest(model_path),
        geometry='exact SMPL-X local LBS; shared beta and stature scale fixed; original COCO vertex regressor',
        dofs=list(DOF_NAMES),upstream_nonzero_pose_max_error_model_m=error,
        rigid_skate_attachment='one ankle frame per boot; toe joint cannot bend boot; five rocker candidates; only selected lowest candidate constrained',
        effective_upright_lift_m=model.lift,rocker_end_rise_m=model.rocker,
        equipment_status='conditional existing effective 9 cm upright lift; assumed 12 mm rocker; neither source-measured',
        plantar_patches=desc,CUDA_used=False)
    (root/'compact_smplx.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


class CompactBody:
    def __init__(self,path,beta,scale,*,lift=.09,rocker=.012):
        z=load(path)
        # Preserve double precision through FK; float32 in-place joint positions
        # otherwise corrupt small local-rotation derivatives.
        z={k:(v.astype(np.float64) if v.dtype.kind=='f' else v) for k,v in z.items()}
        beta=np.asarray(beta,np.float64)
        self.z=z;self.scale=float(scale);self.lift=float(lift);self.rocker=float(rocker)
        self.parents=z['parents'].astype(int)
        self.J=z['J_template']+np.einsum('k,jck->jc',beta,z['J_shapedirs'])
        self.shaped=z['v_template']+np.einsum('k,vck->vc',beta,z['shapedirs'])
        self.posedirs=z['posedirs'].reshape(486,-1)
        self.W=z['lbs_weights'];self.reg=z['smplx2coco17_interestd']
        self.default=Rotation.from_rotvec(z['other_default_pose'].reshape(33,3)).as_matrix()
        alpha=np.linspace(0,1,5)
        rest=z['rest_patch_centroids'];height=z['rest_patch_height']
        patch=rest[:,0,None]*(1-alpha[None,:,None])+rest[:,1,None]*alpha[None,:,None]
        h=height[:,0,None]*(1-alpha[None,:])+height[:,1,None]*alpha[None,:]
        self.blade_relative=self.scale*(patch-self.J[[7,8],None])
        self.blade_relative[:,:,1]-=self.lift+self.scale*h-self.rocker*(2*alpha[None,:]-1)**2

    def forward(self,pose,delta,*,additional_rotations=None,with_frames=False):
        pose=np.asarray(pose,float);delta=np.asarray(delta,float);n=len(pose)
        local=np.broadcast_to(np.eye(3),(n,55,3,3)).copy()
        local[:,1:22]=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(n,21,3,3)
        local[:,22:]=self.default
        for d,(j,axis) in enumerate(POSE_DOFS):
            vec=np.zeros((n,3));vec[:,axis]=delta[:,d]
            local[:,j]=local[:,j]@Rotation.from_rotvec(vec).as_matrix()
        if additional_rotations is not None:
            for j,correction in additional_rotations.items():
                local[:,int(j)]=local[:,int(j)]@Rotation.from_rotvec(np.asarray(correction,float)).as_matrix()
        rotation=local.copy();position=np.broadcast_to(self.J,(n,55,3)).copy()
        for j in range(1,55):
            parent=self.parents[j];rotation[:,j]=rotation[:,parent]@local[:,j]
            position[:,j]=position[:,parent]+np.einsum('nij,j->ni',rotation[:,parent],self.J[j]-self.J[parent])
        translation=position-np.einsum('njab,jb->nja',rotation,self.J)
        feature=(local[:,1:]-np.eye(3)).reshape(n,486)
        posed=self.shaped+(feature@self.posedirs).reshape(n,-1,3)
        blend_R=np.einsum('vj,njab->nvab',self.W,rotation,optimize=True)
        blend_t=np.einsum('vj,nja->nva',self.W,translation,optimize=True)
        vertices=np.einsum('nvab,nvb->nva',blend_R,posed)+blend_t
        coco=np.einsum('vj,nvc->njc',self.reg,vertices)-self.J[0]
        native=position[:,:22]-self.J[0]
        blade=self.scale*native[:,[7,8],None]+np.einsum('nfab,fkb->nfka',rotation[:,[7,8]],self.blade_relative)
        out=(coco*self.scale,native*self.scale,blade)
        return (*out,rotation[:,:22]) if with_frames else out
