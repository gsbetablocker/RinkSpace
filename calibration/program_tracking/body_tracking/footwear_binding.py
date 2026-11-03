"""Shared foot/boot registration and clearance-driven local upper geometry.

T_XY maps Y coordinates into X. Runtime lengths use the same conditional metre
unit as the body solver; export applies the one frozen scene conversion to all
lengths. No per-frame equipment rigid motion or hidden body scaling is added.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .footwear_geometry import local_vertices,boot_alignment,decode
from .rebuild_prepare import digest,load
from .fullbody_state import FrozenReference,WindowState,matrices


def transform(R,t,points):
    return t[...,None,:]+np.einsum('...ij,...kj->...ki',R,points)


class FootwearBinding:
    def __init__(self,reference,surface,metric_binding):
        from rinkspace.viewer3d.open_boot_topology import open_boot_shell
        self.reference=reference;self.surface=surface;self.metric_binding=metric_binding
        self.original,self.parts=local_vertices(reference.geometry)
        p=decode(reference.geometry)
        self.c0=np.array([0.,-p['ankle_to_sole'],(p['shoe_front']-p['shoe_rear'])/2])
        self.template=self.original-self.c0
        self.faces,self.groups=open_boot_shell(self.template[:14])
        self.region_ids=np.r_[np.zeros(5,int),np.ones(5,int),np.full(12,2,int)]
        self.region_ids[self.groups['front_upper_tongue_proxy_faces']]=3
        self.R0=np.stack([boot_alignment(reference.geometry,f) for f in range(2)])
        self.t0=np.einsum('fij,j->fi',self.R0,self.c0)
        self.R_BL=Rotation.from_rotvec([p['blade_pitch'],0,0]).as_matrix()
        self.t_BL=np.mean(self.template[14:],axis=0)
        self.blade_L=(self.template[14:]-self.t_BL)@self.R_BL
        self.x=np.zeros(14);self.deformation_enabled=True
        # Geometric fitting assumptions, not measured lining or stiffness.
        self.lining=.003;self.heel_bed=.022;self.ball_bed=.007
        self.shank_radius=.026;self.cuff_clearance=.004
        self.maximum_modes=np.array([.07,.04,.025])
        self.mode_regularizer=np.array([.35,.5,.8])
        self._build_foot_samples()

    def _build_foot_samples(self):
        R=np.broadcast_to(np.eye(3),(1,21,3,3)).copy()
        _,j,_,frames=self.reference.body.forward_rotations(R,with_frames=True)
        v=self.surface.sample_vertices(j,frames,double_precision=True)[0]
        ids=[];self.foot_ids=[];self.patch_ids=[]
        for ankle,toe in ((7,10),(8,11)):
            chosen=np.flatnonzero((self.surface.weights[:,ankle]+self.surface.weights[:,toe]>.5)&
                (v[:,1]-j[0,ankle,1]<.015))
            z=v[chosen,2];lo,hi=np.quantile(z,[.2,.8]);ball_lo=np.quantile(z,.58)
            patches=[]
            for subset in (chosen[z<=lo],chosen[(z>=ball_lo)&(z<=hi)]):
                patches.append(subset[v[subset,1]<=np.quantile(v[subset,1],.2)+.003])
            self.foot_ids.append(chosen);self.patch_ids.append(patches);ids.extend(chosen)
        self.sample_ids=np.unique(ids)
        self.foot_indices=[np.searchsorted(self.sample_ids,a) for a in self.foot_ids]
        self.patch_indices=[[np.searchsorted(self.sample_ids,a) for a in patches] for patches in self.patch_ids]
        self.neutral_native=j;self.neutral_frames=frames
        self.neutral_samples=self.surface.sample_vertices(j,frames,self.sample_ids,double_precision=True)
        self.reference_pose_hash=__import__('hashlib').sha256(R.tobytes()).hexdigest()

    def parameters(self,x=None):
        x=self.x if x is None else np.asarray(x)
        R=self.R0@matrices(x[:12].reshape(2,6)[:,3:])
        t=self.t0+x[:12].reshape(2,6)[:,:3]
        length,width=np.exp(x[12:14]);v=self.template.copy()
        v[:14]*=[width,1.,length]  # blade and installation stay fixed
        return R,t,v

    def cavity(self,x=None):
        _,_,v=self.parameters(x);shell=v[:14]
        back=float(shell[0,2]+self.lining);front=float(shell[1,2]-self.lining)
        width=float(np.max(abs(shell[:,0]))-self.lining)
        heel=back+.018;ball=back+.68*(front-back)
        bed_z=np.array([back,heel,ball,front]);bed_y=np.array([self.heel_bed,self.heel_bed,self.ball_bed,self.ball_bed+.003])
        roof_z=np.array([back,shell[5,2],shell[4,2],shell[3,2],front])
        roof_y=shell[[6,5,4,3,2],1]-self.lining
        if not np.all(np.diff(roof_z)>0):raise ValueError('Cavity roof is not ordered')
        profile=np.array([[self.heel_bed,back],[self.ball_bed,ball],[self.ball_bed+.003,front],
            [roof_y[-1],front],[roof_y[-2],roof_z[-2]],[roof_y[-3],roof_z[-3]],
            [roof_y[-4],roof_z[-4]],[roof_y[0],back]])
        vertices=np.concatenate([np.c_[np.full(len(profile),side*width),profile] for side in (-1,1)])
        from rinkspace.viewer3d.open_boot_topology import _triangulate
        tri=_triangulate(profile,1e-14);faces=[*tri.tolist(),*(tri[:,::-1]+len(profile)).tolist()]
        for i in range(len(profile)):
            j=(i+1)%len(profile)
            if i==5:continue  # open collar; no fictitious inner lid
            faces.extend([[i,j,j+len(profile)],[i,j+len(profile),i+len(profile)]])
        bed=np.array([[side*width*.85,y,z] for side in (-1,1) for y,z in zip(bed_y,bed_z)])
        anchors=np.array([[0.,self.heel_bed,heel],[0.,self.ball_bed,ball]])
        return dict(vertices=vertices,faces=np.asarray(faces,int),bed=bed,anchors=anchors,
            back=back,front=front,width=width,bed_z=bed_z,bed_y=bed_y,roof_z=roof_z,roof_y=roof_y)

    def foot_metrics(self,joints,frames,samples=None,x=None):
        R,t,_=self.parameters(x);cavity=self.cavity(x)
        if samples is None:samples=self.surface.sample_vertices(joints,frames,self.sample_ids,double_precision=True)
        feet=[];anchors=[];violations=[];support=[]
        for f,ankle in enumerate((7,8)):
            RF=frames[:,ankle];RB=RF@R[f];O=joints[:,ankle]+np.einsum('nij,j->ni',RF,t[f])
            local=np.einsum('nji,nkj->nki',RB,samples-O[:,None])
            foot=local[:,self.foot_indices[f]];feet.append(foot)
            anchors.append(np.stack([local[:,ids].mean(1) for ids in self.patch_indices[f]],1))
            y,z=foot[...,1],foot[...,2]
            floor=np.interp(z,cavity['bed_z'],cavity['bed_y']);roof=np.interp(z,cavity['roof_z'],cavity['roof_y'])
            violations.append(np.stack([np.maximum(abs(foot[...,0])-cavity['width'],0),
                np.maximum(cavity['back']-z,0),np.maximum(z-cavity['front'],0),
                np.maximum(floor-y-self.lining,0),np.maximum(y-roof,0)],-1))
            support.append(anchors[-1][...,1]-np.interp(anchors[-1][...,2],cavity['bed_z'],cavity['bed_y']))
        return dict(feet=feet,anchors=np.stack(anchors,1),violations=violations,
            support=np.stack(support,1),cavity=cavity)

    def fit_residual(self,joints,frames,samples=None,x=None):
        m=self.foot_metrics(joints,frames,samples,x);a=m['anchors'];target=m['cavity']['anchors']
        # Equal-frame quadrature; 3 mm lining is an engineering tolerance.
        n=len(joints);scale=np.sqrt(n)
        anchor=(a-target[None,None])/.012/scale
        outside=np.concatenate([v.reshape(n,-1) for v in m['violations']],1)/.003/np.sqrt(n*len(self.sample_ids))
        bed=m['support']/.005/scale
        return np.r_[anchor.ravel(),outside.ravel(),bed.ravel()]

    def compatibility_rows(self,joints,frames,samples=None,x=None):
        m=self.foot_metrics(joints,frames,samples,x);n=len(joints)
        anchor=(m['anchors']-m['cavity']['anchors'][None,None])/.012
        outside=np.stack([np.max(v,axis=1) for v in m['violations']],1)/.003
        return np.concatenate((anchor.reshape(n,-1),outside.reshape(n,-1),m['support'].reshape(n,-1)/.005),axis=1)

    def evaluate(self,P,Q,joints,frames,x=None,deformation=None):
        R,t,vertices=self.parameters(x);n=len(P)
        RW=np.stack([Q@frames[:,a]@R[f] for f,a in enumerate((7,8))],1)
        OW=np.stack([P+np.einsum('nij,nj->ni',Q,joints[:,a]+np.einsum('nij,j->ni',frames[:,a],t[f])) for f,a in enumerate((7,8))],1)
        v=np.broadcast_to(vertices,(n,2,48,3)).copy();alpha=np.zeros((n,2,3));relative=[];shank_world=[]
        enabled=self.deformation_enabled if deformation is None else deformation
        for f,(knee,ankle) in enumerate(((4,7),(5,8))):
            footR=frames[:,ankle];bootR=footR@R[f]
            relative.append(np.swapaxes(frames[:,knee],-1,-2)@footR)
            ankle_B=np.einsum('ji,nj->ni',R[f],-np.broadcast_to(t[f],(n,3)))
            direction=np.einsum('nji,nj->ni',bootR,joints[:,knee]-joints[:,ankle])
            # Intersect the finite shank axis with the collar height; this uses
            # position AND direction, with a dead zone before contact.
            lam=np.clip((vertices[4,1]-ankle_B[:,1])/np.maximum(direction[:,1],1e-5),0,1)
            centre=ankle_B+lam[:,None]*direction
            front_gap=vertices[4,2]-centre[:,2]-self.shank_radius-self.cuff_clearance
            rear_gap=centre[:,2]-vertices[5,2]-self.shank_radius-self.cuff_clearance
            side_gap=abs(vertices[4,0])-abs(centre[:,0])-self.shank_radius-self.cuff_clearance
            penetration=np.maximum(-np.stack([front_gap,rear_gap,side_gap],1),0)
            if enabled:alpha[:,f]=np.minimum(penetration/(1+self.mode_regularizer),self.maximum_modes)
            v[:,f,[4,11],2]+=alpha[:,f,0,None]
            v[:,f,[5,12],2]-=alpha[:,f,1,None]
            v[:,f,[4,5],0]-=alpha[:,f,2,None]
            v[:,f,[11,12],0]+=alpha[:,f,2,None]
            shank_world.append(np.stack((P+np.einsum('nij,nj->ni',Q,joints[:,ankle]),P+np.einsum('nij,nj->ni',Q,joints[:,knee])),1))
        world=OW[:,:,None]+np.einsum('nfij,nfkj->nfki',RW,v)
        RWL=RW@self.R_BL;OWL=OW+np.einsum('nfij,j->nfi',RW,self.t_BL)
        return dict(gear=world,boot_R=RW,boot_O=OW,lower_boot_vertices=vertices[:14],
            upper_deformation=alpha,effective_ankle_relative_rotation=np.stack(relative,1),
            blade_R=RWL,blade_O=OWL,shank_world=np.stack(shank_world,1))

    def diagnostics(self,state):
        P,Q=state['P'],state['Q'];j=np.einsum('nji,nkj->nki',Q,state['native']-P[:,None]);frames=state['frames']
        s=self.evaluate(P,Q,j,frames);m=self.foot_metrics(j,frames);c=m['cavity'];R,O=s['boot_R'],s['boot_O']
        anchors=np.concatenate([m['anchors'],np.broadcast_to(c['anchors'],m['anchors'].shape)],2)
        world_anchors=O[:,:,None]+np.einsum('nfij,nfkj->nfki',R,anchors)
        axes=[];labels=[]
        for name,rot,origin in [('shank',np.stack([Q@frames[:,a] for a in (4,5)],1),s['shank_world'][:,:,0]),
                                ('lower_boot',R,O),('blade',s['blade_R'],s['blade_O'])]:
            for axis in range(3):
                axes.append(np.stack((origin,origin+rot[:,:,:,axis]*.08),2));labels.append(name+'_'+str(axis))
        return dict(boot_surface_world_mid=s['gear'][:,:,:14],boot_surface_faces=self.faces,
            boot_surface_region_ids=self.region_ids,
            collar_boundary=self.groups['collar_boundary'],
            footbed_world_mid=transform(R,O,np.broadcast_to(c['bed'],(len(P),2,*c['bed'].shape))),
            cavity_world_mid=transform(R,O,np.broadcast_to(c['vertices'],(len(P),2,*c['vertices'].shape))),
            cavity_faces=c['faces'],anchors_world_mid=world_anchors,
            anchor_labels=np.array(['foot_heel_patch','foot_ball_patch','boot_heel_pocket','boot_ball_support']),
            axes_world_mid=np.stack(axes,2),axis_labels=np.array(labels),
            upper_deformation_mid=s['upper_deformation'],effective_ankle_relative_rotation_mid=s['effective_ankle_relative_rotation'],
            cavity_penetration_mid=np.stack([np.max(v,axis=(1,2)) for v in m['violations']],1),
            heel_drift_mid=np.linalg.norm(m['anchors'][:,:,0]-c['anchors'][0],axis=2),
            forefoot_error_mid=np.linalg.norm(m['anchors'][:,:,1]-c['anchors'][1],axis=2),
            blade_tilt_rad_mid=np.arcsin(np.clip(s['blade_R'][:,:,2,0],-1,1)),
            blade_pose_world_mid=np.concatenate((s['blade_R'],s['blade_O'][...,None]),axis=3))

    def record(self):
        R,t,v=self.parameters();c=self.cavity();ratio=self.metric_binding['length_export_ratio']
        transforms=np.broadcast_to(np.eye(4),(2,4,4)).copy();transforms[:,:3,:3]=R;transforms[:,:3,3]=t*ratio
        TBL=np.eye(4);TBL[:3,:3]=self.R_BL;TBL[:3,3]=self.t_BL*ratio
        return dict(schema_version='rinkspace.footwear-binding.v1',
            transform_convention='column vectors: T_FB maps boot coordinates B into hindfoot F; T_WB=T_WF@T_FB; T_WL=T_WB@T_BL',
            body_model_hash=json.loads((self.reference.root/'compact_smplx.json').read_text(encoding='utf8'))['model_sha256'],
            compact_model_hash=self.reference.hashes[str(self.reference.root/'compact_smplx.npz')],
            body_shape_hash=__import__('hashlib').sha256(np.r_[self.reference.p['shared_betas'],self.reference.p['body_scale']].astype(np.float64).tobytes()).hexdigest(),
            reference_pose_hash=self.reference_pose_hash,body_scale=float(self.reference.p['body_scale']),
            foot_frame_definition=dict(hindfoot=[7,8],shank=[4,5],distal_foot=[10,11],body_local_up='Y'),
            units='current conditional metres',world_up='Z',metric_binding=self.metric_binding,
            T_FB=transforms.tolist(),T_lower_boot_to_blade=TBL.tolist(),
            inner_footbed=(c['bed']*ratio).tolist(),heel_pocket=(c['anchors'][0]*ratio).tolist(),forefoot_support=(c['anchors'][1]*ratio).tolist(),
            cavity_proxy=dict(vertices=(c['vertices']*ratio).tolist(),faces=c['faces'].tolist()),
            outer_shell_template=(v[:14]*ratio).tolist(),shell_faces=self.faces.tolist(),
            region_ids=self.region_ids.tolist(),region_labels=['side_0','side_1','bridge','front_upper_tongue_proxy'],collar_boundary=self.groups['collar_boundary'].tolist(),
            upper_deformation_modes=dict(names=['front_tongue','rear_collar','collar_opening'],
                moving_vertex_ids=[[4,11],[5,12],[4,5,11,12]],zero_weight_regions='all other shell vertices, sole, heel support, blade and installation',
                driver='finite shank axis/collar contact with clearance; positive gap gives exactly zero deformation'),
            clearance_parameters=dict(shank_radius_m=self.shank_radius*ratio,clearance_m=self.cuff_clearance*ratio,lining_allowance_m=self.lining*ratio),
            deformation_prior=dict(regularizer=self.mode_regularizer.tolist(),maximum_mode_displacements_m=(self.maximum_modes*ratio).tolist(),material_stiffness_measured=False),
            blade_edges=dict(vertices_L=(self.blade_L*ratio).tolist(),regions=['rocker_side_0','rocker_side_1']),
            toe_pick_candidates=[],toe_pick_status='not observed/calibrated; rocker endpoint never relabelled as tooth',
            parameter_provenance='geometry prior calibrated to shared-beta complete surface and cached external shoe observations',
            observability_status=dict(internal_fit='prior-constrained, not directly imaged',upper_modes='deterministic geometry prior; no independent frame parameters',foot_sliding='disabled'),
            legacy_migration=dict(c0_solver_m=self.c0.tolist(),boot_align_folded_once=True,old_geometry_frozen=True),
            fit_parameters_solver_unit=self.x.tolist(),foot_surface_vertex_ids=[a.tolist() for a in self.foot_ids],
            heel_ball_patch_vertex_ids=[[a.tolist() for a in p] for p in self.patch_ids])


class IntegratedBodyState(FrozenReference):
    """Continue the saved full-body curves, optionally attaching one binding."""
    def __init__(self,root,baseline):
        super().__init__(root);self.binding=None;self.baseline=Path(baseline);self.corrections=[];self.refinements=[]
        for i in range(1,6):
            z=load(self.baseline/'full'/f'w{i}'/'result.npz')
            ws=WindowState(self,z['seconds'],'full');self.corrections.append((ws,z['x']))

    def at(self,t):
        r=super().at(t);delta=np.zeros((len(r['P']),69))
        for ws,x in self.corrections:delta+=ws.increments(x,t)
        r=dict(P=r['P']+delta[:,:3],Q=r['Q']@matrices(delta[:,3:6]),
            local_R=r['local_R']@matrices(delta[:,6:].reshape(-1,21,3)),times=r['times'])
        for ws,x in self.refinements:
            d=ws.increments(x,t);r=dict(P=r['P']+d[:,:3],Q=r['Q']@matrices(d[:,3:6]),
                local_R=r['local_R']@matrices(d[:,6:].reshape(-1,21,3)),times=r['times'])
        return r

    def world(self,reference,delta=None):
        s=super().world(reference,delta)
        if self.binding is not None:
            local=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None])
            gear=self.binding.evaluate(s['P'],s['Q'],local,s['frames'])
            s.update(gear=gear['gear'],boot_R=gear['boot_R'],boot_O=gear['boot_O'],footwear=gear)
            s['world']=np.concatenate((s['gear'].reshape(len(local),96,3),s['coco']),axis=1)
        return s
