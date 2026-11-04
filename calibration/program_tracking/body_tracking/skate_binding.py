"""Versioned lower skate and section-offset cavity on the existing foot FK.

All internal lengths use the historical solver unit. The template's metric
parameters are converted once by the frozen scene ratio; export reverses it.
No blade pitch, independent shoe state or foot-size change is introduced.
"""
from dataclasses import asdict, replace
import copy
import json
import numpy as np
from .footwear_binding import FootwearBinding, transform
from .footwear_layout import OBSERVATION_LAYOUT
from .fullbody_state import matrices
from .skate_template import (SkateParameters, build_skate, profile_height,
    _sole_bottom, _sole_half_width)

TEMPLATE_VERSION='rinkspace.metric-lower-skate.v1'
BINDING_VERSION='rinkspace.section-foot-skate.v1'
SHAPE_NAMES=('sole_length_m','sole_width_m','front_mount_height_m',
             'rear_mount_height_m','heel_block_height_m')
# Columns of the new B basis in the historical (lateral,up,forward) frame.
NEW_B_TO_OLD=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])


def upper_topology(sections,collar_end=3):
    """Open collar and open lower boundary; the lower skate supplies the sole."""
    faces=[];regions=[]
    for i in range(sections-1):
        for j in range(1,6):  # no duplicate floor/sole at j=0
            if j==3 and i<collar_end:continue  # open ankle mouth
            k=(j+1)%6;a=i*6+j;b=i*6+k;c=(i+1)*6+k;d=(i+1)*6+j
            faces.extend([[a,b,c],[a,c,d]])
            regions.extend([1 if j in (1,2) else 0 if j in (4,5) else 2]*2)
    for i in (0,sections-1):
        for j in range(1,5):
            tri=[i*6,i*6+j,i*6+j+1]
            faces.append(tri[::-1] if i==0 else tri);regions.append(2)
    # Tongue is a distinct surface along the front cuff, not a lid over it.
    off=sections*6
    faces.extend([[off,off+1,off+2],[off,off+2,off+3]]);regions.extend([3,3])
    return np.asarray(faces,int),np.asarray(regions,int)


class SkateBinding(FootwearBinding):
    """Static 6-DOF registration per foot plus five global shape columns."""
    def __init__(self,reference,surface,metric_binding,rocker_ft=7.):
        super().__init__(reference,surface,metric_binding)
        self.layout=OBSERVATION_LAYOUT;self.ratio=float(metric_binding['length_export_ratio'])
        self.main_rocker_ft=float(rocker_ft);self.x=np.zeros(17)
        self._parameter_key=None;self._packet=None
        self.collar_constraints_enabled=False
        # Anatomical correspondences initialize a real T_FB, rather than a
        # single arbitrary ankle-to-centre displacement. These skin patches
        # are the shared generic body prior, never an athlete foot scan.
        self.R0=[];self.t0=[];lengths=[];widths=[];self.foot_correspondences=[]
        for f,ankle in enumerate((7,8)):
            pts=self.neutral_samples[0,self.foot_indices[f]]
            heel,ball=[self.neutral_samples[0,ids].mean(0) for ids in self.patch_indices[f]]
            forward=ball-heel;forward[1]=0;forward/=np.linalg.norm(forward)
            up=np.array([0.,1.,0.]);left=np.cross(up,forward)
            R=np.stack((forward,left,up),1)
            footB=(pts-heel)@R
            lengths.append(np.ptp(footB[:,0])*self.ratio+.012)
            widths.append(np.ptp(footB[:,1])*self.ratio+.009)
            self.R0.append(R);self.foot_correspondences.append(dict(heel=heel,ball=ball,
                metatarsal_negative_y=pts[np.argmin(footB[:,1])],
                metatarsal_positive_y=pts[np.argmax(footB[:,1])]))
        self.base_parameters=SkateParameters(sole_length_m=float(np.mean(lengths)),
            sole_width_m=float(np.mean(widths)),main_rocker_m=self.main_rocker_ft*.3048)
        self.R0=np.asarray(self.R0)
        p=self._solver_parameters(np.zeros(17))
        heel_target=np.array([.12*p.sole_length_m,0.,float(_sole_bottom(.12*p.sole_length_m,p))+.010/self.ratio])
        for f,a in enumerate((7,8)):
            self.t0.append(self.foot_correspondences[f]['heel']-self.neutral_native[0,a]-self.R0[f]@heel_target)
        self.t0=np.asarray(self.t0)
        geometry=self.packet();section_count=len(geometry['xs'])
        collar_end=int(np.searchsorted(geometry['xs'],.46*geometry['parameters'].sole_length_m))
        self.faces,self.region_ids=upper_topology(section_count,collar_end)
        # Compatibility shell triangles are used only to select image edges.
        from rinkspace.viewer3d.open_boot_topology import open_boot_shell
        self.adapter_faces,_=open_boot_shell(self.packet()['adapter'][:14])

    def _solver_parameters(self,x):
        values=asdict(self.base_parameters)
        for name,shift in zip(SHAPE_NAMES,np.asarray(x)[12:]):values[name]*=np.exp(shift)
        for name in values:
            if not name.endswith('_fraction'):values[name]/=self.ratio
        return SkateParameters(**values)

    def packet(self,x=None):
        x=self.x if x is None else np.asarray(x,float)
        key=x[12:].tobytes()
        if key==self._parameter_key:return self._packet
        p=self._solver_parameters(x)
        # Construct in actual metric coordinates first: the reference kernel
        # also has fixed 1/3/5mm construction constants. Scaling its parameter
        # dataclass before construction would miss those constants.
        metric_values={k:v if k.endswith('_fraction') else v*self.ratio for k,v in asdict(p).items()}
        template=build_skate(SkateParameters(**metric_values))
        template['parameters']=asdict(p)
        template['vertices']=template['vertices']/self.ratio
        for a in template['contact_arcs']:
            a['center']=(np.asarray(a['center'])/self.ratio).tolist();a['radius_m']/=self.ratio
        for g in template['contact_point_groups']:g['points']=np.asarray(g['points'])/self.ratio
        for row in template['mounting_patches']+template['post_connections']:
            for name,value in list(row.items()):
                if name.endswith('_m'):row[name]=(np.asarray(value)/self.ratio).tolist()
        for name,value in list(template['derived_dimensions'].items()):
            if name.endswith('_m'):template['derived_dimensions'][name]=value/self.ratio
        interface=template['upper_interface']
        for name in ('sole_top_stations','sole_half_widths','heel_seat','forefoot_seat'):
            interface[name]=np.asarray(interface[name])/self.ratio
        L=p.sole_length_m
        # Include every outsole width/height break: the upper's open lower
        # boundary then coincides with the actual lower-skate surface along
        # whole segments, including heel/forefoot transition corners.
        fractions=np.unique(np.round(np.r_[[0.,.12,.28,.46,.62,.76,.90,1.],
            np.asarray(template['upper_interface']['sole_top_stations'])[:,0]/L],12))
        xs=L*fractions
        floor=_sole_bottom(xs,p)+p.sole_thickness_m
        width=_sole_half_width(xs,p)
        rise=np.interp(xs/L,[0.,.12,.28,.46,.62,.76,.90,1.],
            [.105,.150,.150,.145,.085,.045,.030,.024])/self.ratio
        # Local yz rings; the cavity is an inward offset of EACH convex
        # cross-section. This retains changing heel/ball/toe widths.
        outer=[];inner=[];lining=.003/self.ratio
        for xx,w,z,h in zip(xs,width,floor,rise):
            yz=np.array([[-w,z],[w,z],[w,z+.75*h],[.70*w,z+h],[-.70*w,z+h],[-w,z+.75*h]])
            edge=np.roll(yz,-1,axis=0)-yz
            inward=np.c_[-edge[:,1],edge[:,0]]/np.linalg.norm(edge,axis=1)[:,None]
            rhs=np.einsum('ij,ij->i',inward,yz)+lining
            shifted=np.array([np.linalg.solve(inward[[j-1,j]],rhs[[j-1,j]]) for j in range(6)])
            inner_x=np.clip(xx,lining,L-lining)
            outer.extend(np.c_[np.full(6,xx),yz]);inner.extend(np.c_[np.full(6,inner_x),shifted])
        ix=lambda f:int(np.argmin(abs(xs/L-f)))
        cuff,instep=ix(.46),ix(.62)
        tongue=np.array([[.44*L,-.027/self.ratio,floor[3]+rise[3]],
                         [.44*L,.027/self.ratio,floor[cuff]+rise[cuff]],
                         [.64*L,.025/self.ratio,floor[instep]+rise[instep]+.005/self.ratio],
                         [.64*L,-.025/self.ratio,floor[instep]+rise[instep]+.005/self.ratio]])
        tongue[0,2]=floor[cuff]+rise[cuff]
        outer=np.r_[np.asarray(outer),tongue]
        inner=np.r_[np.asarray(inner),tongue-[0.,0.,lining]]
        # Newly generated observation adapter, not the old fitted shell.
        profile=np.array([[0.,floor[0]],[L,floor[-1]],[L,floor[-1]+rise[-1]],
            [.62*L,floor[instep]+rise[instep]],[.46*L,floor[cuff]+rise[cuff]],
            [.12*L,floor[ix(.12)]+rise[ix(.12)]],[0.,floor[0]+rise[0]]])
        profile_width=_sole_half_width(profile[:,0],p)*np.array([1.,1.,.70,.70,.70,.70,.70])
        adapter=np.concatenate([np.c_[profile[:,0],sign*profile_width,profile[:,1]] for sign in (-1,1)])
        blade_x=np.linspace(-p.runner_rear_extension_m,L-p.runner_front_inset_m,self.layout.edge_count)
        blade=np.concatenate([np.c_[blade_x,np.full(len(blade_x),sign*p.blade_width_m/2),profile_height(blade_x,p)] for sign in (-1,1)])
        adapter=np.r_[adapter,blade]
        edge_x=np.linspace(blade_x[0],blade_x[-1],81)
        edges=np.stack([np.c_[edge_x,np.full(len(edge_x),sign*p.blade_width_m/2),profile_height(edge_x,p)] for sign in (-1,1)])
        targets=np.array([[.12*L,0.,np.interp(.12*L,xs,floor)+lining],
                          [.72*L,0.,np.interp(.72*L,xs,floor)+lining]])
        self._parameter_key=key
        self._packet=dict(parameters=p,template=template,upper=outer,inner=inner,
            adapter=adapter,edges=edges,xs=xs,floor=floor,width=width,rise=rise,
            targets=targets,collar=np.r_[np.arange(cuff+1)*6+3,np.arange(cuff,-1,-1)*6+4,3])
        return self._packet

    def parameters(self,x=None):
        x=self.x if x is None else np.asarray(x,float)
        R=self.R0@matrices(x[:12].reshape(2,6)[:,3:])
        t=self.t0+x[:12].reshape(2,6)[:,:3]
        return R,t,self.packet(x)['adapter']

    def transforms(self,P,Q,joints,frames,x=None):
        R,t,_=self.parameters(x)
        RW=np.stack([Q@frames[:,a]@R[f] for f,a in enumerate((7,8))],1)
        OW=np.stack([P+np.einsum('nij,nj->ni',Q,joints[:,a]+np.einsum('nij,j->ni',frames[:,a],t[f])) for f,a in enumerate((7,8))],1)
        return RW,OW

    def deformation(self,joints,frames,x=None):
        R,t,_=self.parameters(x);p=self.packet(x);out=[]
        for f,(k,a) in enumerate(((4,7),(5,8))):
            ankle=-t[f]@R[f]
            direction=np.einsum('nji,nj->ni',frames[:,a]@R[f],joints[:,k]-joints[:,a])
            collar_station=int(np.argmin(abs(p['xs']/p['parameters'].sole_length_m-.28)))
            height=p['floor'][collar_station]+p['rise'][collar_station]
            lam=np.clip((height-ankle[2])/np.maximum(direction[:,2],1e-6),0,1)
            c=ankle+lam[:,None]*direction
            radius=.027/self.ratio
            out.append(np.minimum(np.maximum(np.c_[c[:,0]+radius-.46*p['parameters'].sole_length_m,
                .0-c[:,0]+radius,np.abs(c[:,1])+radius-p['width'][collar_station]],0),[.020/self.ratio,.015/self.ratio,.012/self.ratio]))
        return np.stack(out,1)

    def deform(self,vertices,alpha,x=None):
        p=self.packet(x);L=p['parameters'].sole_length_m
        v=np.broadcast_to(vertices,(*alpha.shape[:2],*vertices.shape)).copy()
        xcoord=vertices[:,0];z=vertices[:,2]
        high=np.clip((z-np.interp(xcoord,p['xs'],p['floor']))/(.08/self.ratio),0,1)
        cuff=np.clip((.65*L-xcoord)/(.20*L),0,1)*high
        front=np.clip(xcoord/(.46*L),0,1)*cuff
        rear=(1-np.clip(xcoord/(.46*L),0,1))*cuff
        v[...,0]+=alpha[:,:,0,None]*front-alpha[:,:,1,None]*rear
        v[...,1]+=alpha[:,:,2,None]*cuff*np.sign(vertices[:,1])
        return v

    def evaluate(self,P,Q,joints,frames,x=None,deformation=None):
        packet=self.packet(x);R,O=self.transforms(P,Q,joints,frames,x)
        alpha=self.deformation(joints,frames,x)
        if deformation is False:alpha*=0
        adapter=np.broadcast_to(packet['adapter'],(len(P),2,self.layout.per_foot,3)).copy()
        adapter[:,:,:self.layout.shell_count]=self.deform(packet['adapter'][:self.layout.shell_count],alpha,x)
        return dict(gear=transform(R,O,adapter),boot_R=R,boot_O=O,
            upper_deformation=alpha,template=packet['template'],
            upper_local=self.deform(packet['upper'],alpha,x),
            inner_local=self.deform(packet['inner'],alpha,x))

    def foot_metrics(self,joints,frames,samples=None,x=None,*,hand_local_R=None):
        R,t,_=self.parameters(x);p=self.packet(x)
        if samples is None:
            kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
            samples=self.surface.sample_vertices(joints,frames,self.sample_ids,double_precision=True,**kwargs)
        feet=[];anchors=[];violations=[];support=[];alpha=self.deformation(joints,frames,x)
        # Inner walls/footbed are generated from the legal outer section; no
        # maximum-width shortcut and no smaller substitute human foot.
        count=len(p['xs'])
        inner=self.deform(p['inner'],alpha,x)[:,:,:count*6].reshape(len(joints),2,count,6,3)
        for f,a in enumerate((7,8)):
            RB=frames[:,a]@R[f];O=joints[:,a]+np.einsum('nij,j->ni',frames[:,a],t[f])
            local=np.einsum('nji,nkj->nki',RB,samples-O[:,None]);foot=local[:,self.foot_indices[f]]
            feet.append(foot);anchors.append(np.stack([local[:,ids].mean(1) for ids in self.patch_indices[f]],1))
            # Batched equivalent of six per-frame np.interp calls. This
            # changes execution only; each frame retains its own co-deformed
            # inner-wall sections and its full original foot vertex set.
            walls=[];fr=np.arange(len(joints))[:,None];query=foot[...,0]
            for j in range(6):
                grid=inner[:,f,:,j,0]
                index=np.clip(np.sum(query[...,None]>=grid[:,None,:],axis=-1)-1,0,count-2)
                left=grid[fr,index];right=grid[fr,index+1]
                fraction=np.clip((query-left)/(right-left),0,1)
                yz=inner[:,f,:,j,1:]
                walls.append(yz[fr,index]*(1-fraction[...,None])+yz[fr,index+1]*fraction[...,None])
            walls=np.stack(walls,axis=2);edge=np.roll(walls,-1,axis=2)-walls
            inward=np.stack([-edge[...,1],edge[...,0]],-1)/np.linalg.norm(edge,axis=-1)[...,None]
            margins=np.sum(inward*(foot[:,:,None,1:]-walls),axis=-1)
            lining=.003/self.ratio
            violations.append(np.concatenate((np.maximum(-margins,0),
                np.maximum(lining-query,0)[...,None],
                np.maximum(query-(p['parameters'].sole_length_m-lining),0)[...,None]),axis=2))
            support.append(anchors[-1][...,2]-np.interp(anchors[-1][...,0],p['xs'],p['floor']+.003/self.ratio))
        return dict(feet=feet,anchors=np.stack(anchors,1),violations=violations,
            support=np.stack(support,1),targets=p['targets'],alpha=alpha)

    @property
    def compatibility_width(self):
        return 34 if getattr(self,'collar_constraints_enabled',False) else 32

    def enable_collar_constraints(self):
        """Add true shank skin samples without replacing the original foot."""
        if getattr(self,'collar_constraints_enabled',False):return
        from .skate_collar_factors import prepare
        spec=prepare(self);old=self.sample_ids.copy();known=set(map(int,old))
        self.sample_ids=np.r_[old,[i for i in spec.sample_ids if int(i) not in known]].astype(int)
        lookup={int(v):i for i,v in enumerate(self.sample_ids)}
        self.collar_sample_positions=np.array([lookup[int(i)] for i in spec.sample_ids],int)
        self.neutral_samples=self.surface.sample_vertices(self.neutral_native,self.neutral_frames,
            self.sample_ids,double_precision=True)
        self.collar_constraints_enabled=True

    def collar_metrics(self,joints,frames,samples=None,x=None,*,hand_local_R=None):
        from .skate_collar_factors import evaluate
        binding=self
        if x is not None:
            binding=copy.copy(self);binding.x=np.asarray(x).copy()
        chosen=None if samples is None else samples[:,self.collar_sample_positions]
        kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
        return evaluate(binding,joints,frames,chosen,**kwargs)

    def compatibility_rows(self,joints,frames,samples=None,x=None,*,hand_local_R=None):
        kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
        # The exact foot/shank union is skinned once, with the actual hands.
        # Caller-owned samples and live shape/registration remain unchanged.
        if samples is None and getattr(self,'collar_constraints_enabled',False):
            samples=self.surface.sample_vertices(joints,frames,self.sample_ids,double_precision=True,**kwargs)
        m=self.foot_metrics(joints,frames,samples,x,**kwargs);n=len(joints)
        anchor=(m['anchors']-m['targets'][None,None])/(.012/self.ratio)
        outside=np.stack([np.max(v,axis=1) for v in m['violations']],1)/(.003/self.ratio)
        rows=np.concatenate((anchor.reshape(n,-1),outside.reshape(n,-1),m['support'].reshape(n,-1)/(.005/self.ratio)),axis=1)
        if getattr(self,'collar_constraints_enabled',False):
            collar=self.collar_metrics(joints,frames,samples,x,**kwargs)
            rows=np.c_[rows,collar.get('residual_breach_solver',collar['breach_solver'])/(.003/self.ratio)]
        return rows

    def record(self):
        R,t,_=self.parameters();p=self.packet();T=np.broadcast_to(np.eye(4),(2,4,4)).copy()
        T[:,:3,:3]=R;T[:,:3,3]=t*self.ratio
        return dict(schema=BINDING_VERSION,template_version=TEMPLATE_VERSION,binding_version=BINDING_VERSION,
            T_FB=T.tolist(),units='conditional scene metres',world_up='Z',template_axes='x heel-to-toe; y local left; z up',
            metric_binding=self.metric_binding,old_axis_conversion=NEW_B_TO_OLD.tolist(),
            conversion='template metric lengths / length_export_ratio once; all exported lengths * ratio once',
            old_boot_align_applied=False,old_blade_pitch_applied=False,per_frame_equipment_parameters=0,
            shared_parameters_solver_unit=self.x.tolist(),shape_parameters={k:getattr(p['parameters'],k)*self.ratio for k in SHAPE_NAMES},
            main_rocker_ft=self.main_rocker_ft,layout=self.layout.record(),
            parameter_provenance=p['template']['parameter_provenance'],
            foot_correspondences=[{k:(v*self.ratio).tolist() for k,v in d.items()} for d in self.foot_correspondences],
            anatomical_prior='shared-beta generic SMPL-X skin; not measured athlete foot',
            cavity='3mm inward offset per convex yz section; footbed above outsole; open collar; co-deformed with upper',
            deformation='bounded geometric cuff initializer; independent gap checks still required',
            collar_section_in_solver=getattr(self,'collar_constraints_enabled',False),
            collar_section_factor='real shank triangle crossing endpoints and midpoints at 1mm below inner rim; finite surrogate, independent section review required',
            independent_3d_truth=False)
