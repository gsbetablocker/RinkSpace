"""Fixed source regions on the existing FK, exact runner and native RS chain."""
from dataclasses import replace
import numpy as np
from scipy.sparse import csr_matrix,hstack
from .support_anchor_factors import NativeRegionObservation,pixel_region_factor,lowest_runner_witness
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection


def _vee(a):
    return .5*np.stack((a[...,2,1]-a[...,1,2],a[...,0,2]-a[...,2,0],a[...,1,0]-a[...,0,1]),axis=-1)


class SupportImageFactors:
    """Rows/foot/phase/reliability fixed before solving, never height-selected."""
    def __init__(self,problem,document,*,include_review=False,arc_ids=None):
        if problem.optimize_shared:raise ValueError('This version requires frozen shared footwear')
        self.problem=problem;self.body=problem.body;self.reference=problem.reference
        self.binding=problem.binding;self.document=document;self.rows=[]
        self.source=document['source_id'];self.shot=document['shot_id'];self.tb=tuple(document['time_base'])
        lookup={int(p):i for i,p in enumerate(self.reference.p['pts'])}
        for row in document['observations']:
            o=NativeRegionObservation(**{k:row[k] for k in NativeRegionObservation.__dataclass_fields__})
            enabled=(o.role=='fit' or include_review) and o.foot in (0,1) and o.visibility=='visible' and o.reliability>0
            enabled &= o.kind!='support_contact_region' or o.contact_state=='supported'
            enabled &= include_review or row.get('optimization_enabled',True)
            if enabled:
                if o.pts not in lookup:raise ValueError('Observation outside source timeline')
                self.rows.append((row,o,lookup[o.pts]))
        if not self.rows:raise ValueError('No admitted support/shoe image rows')
        self.frames=np.array([r[2] for r in self.rows],int)
        self.feet=np.array([r[1].foot for r in self.rows],int)
        self.times=(self.reference.times[self.frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel()
        self.ref=self.reference.at(self.times);self.B=self.body.state.basis(self.times).reshape(-1,5,self.body.state.m)
        arcs=self.binding.packet()['template']['contact_arcs']
        self.arcs={a['id']:a for a in arcs if a['semantic']=='runner'}
        self.arc_ids={} if arc_ids is None else dict(arc_ids)
        state=fk_state(self.reference,self.ref,np.zeros((len(self.times),69)))
        for i,(row,o,_) in enumerate(self.rows):
            if o.kind=='support_contact_region' and row['id'] not in self.arc_ids:
                witness=lowest_runner_witness(list(self.arcs.values()),state['R'][i*5+2,o.foot],state['O'][i*5+2,o.foot],plane_normal_world=[0,0,1])
                self.arc_ids[row['id']]=witness['arc_id']
        self.include_review=include_review
        self.last=None

    def geometry(self,state):
        count=len(self.rows);points=np.empty((count,5,1,3));rotation_j=np.empty((count,5,3,3));warnings=[]
        adapter=self.binding.packet()['adapter']
        for i,(row,o,_) in enumerate(self.rows):
            for l in range(5):
                R=state['R'][i*5+l,o.foot];O=state['O'][i*5+l,o.foot]
                if o.kind=='support_contact_region':
                    w=lowest_runner_witness([self.arcs[self.arc_ids[row['id']]]],R,O,plane_normal_world=[0,0,1])
                    point=w['world_point'];J=w['jacobian_right_rotation']
                    if w['differentiability_warning']:warnings.append((row['id'],l))
                elif o.kind in ('blade_midpoint_region','blade_endpoint_region'):
                    from .skate_template import profile_height
                    p=self.binding.packet()['parameters'];xx=.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m)
                    if o.kind=='blade_endpoint_region':
                        endpoint=row.get('blade_endpoint')
                        if endpoint not in ('front','rear'):raise ValueError('Explicit blade endpoint identity required')
                        xx=p.sole_length_m-p.runner_front_inset_m if endpoint=='front' else -p.runner_rear_extension_m
                    q=np.array([xx,0.,float(profile_height(xx,p))]);point=O+R@q
                    from .fullbody_fit import skew
                    J=-R@skew(q)
                else:
                    ids=row.get('adapter_point_ids',list(range(14)))
                    q=adapter[np.asarray(ids,int)].mean(0)
                    point=O+R@q
                    from .fullbody_fit import skew
                    J=-R@skew(q)
                points[i,l,0]=point;rotation_j[i,l]=J
        return points,rotation_j,warnings

    def evaluate(self,x,jacobian=False):
        state=self.body.state;delta=state.increments(x,self.times)
        s=fk_state(self.reference,self.ref,delta);nodes,Jrotation,warnings=self.geometry(s)
        projection=moving_projection(self.problem.camera,nodes,self.frames,jacobian=jacobian)
        residual=[];D=[];metrics=[]
        for i,(row,o,_) in enumerate(self.rows):
            measured=replace(o,role='fit') if self.include_review else o
            r=pixel_region_factor(projection['uv'][i,0],measured,source_id=self.source,shot_id=self.shot,pts=o.pts,time_base=self.tb,predicted_foot=o.foot,predicted_kind=o.kind,robust_delta=self.document['robust_delta'])
            if not r['enabled']:raise ValueError('An inner source row unexpectedly disabled')
            residual.append(r['residual']);D.append(r['jacobian_native_uv'])
            offset=np.asarray(r['native_pixel_offset']);radius=r['source_ellipse_radius']
            metrics.append(dict(id=row['id'],pts=o.pts,foot=o.foot,kind=o.kind,role=o.role,uv=projection['uv'][i,0].tolist(),centre_error_px=float(np.linalg.norm(offset)),outside_region_px=float(np.linalg.norm(offset)*max(0.,1.-1./max(radius,1e-30))),ellipse_radius=radius))
        r=np.asarray(residual).ravel();J=None
        if jacobian:
            n=len(self.rows);pick=np.arange(n*5);foot=np.repeat(self.feet,5)
            R=s['R'][pick,foot];drows=[]
            def column(active):
                h=1e-6;plus=delta.copy();minus=delta.copy();plus[:,active]+=h;minus[:,active]-=h
                a=fk_state(self.reference,self.ref,plus);b=fk_state(self.reference,self.ref,minus)
                dO=(a['O'][pick,foot]-b['O'][pick,foot])/(2*h)
                dR=(a['R'][pick,foot]-b['R'][pick,foot])/(2*h)
                eta=_vee(R.transpose(0,2,1)@dR).reshape(n,5,3)
                return dO.reshape(n,5,3)+np.einsum('nlij,nlj->nli',Jrotation,eta)
            for dX in self.body.execution.map_columns(column,state.active):
                drows.append(np.einsum('nplc,nlc->npl',projection['pixel_jac_nodes'][:,0],dX))
            local=np.stack(drows,-1)
            local=np.einsum('nab,nbld->nald',np.asarray(D),local)
            body_J=csr_matrix(np.einsum('nald,nlm->namd',local,self.B).reshape(len(r),state.size))
            J=hstack((csr_matrix((len(r),17)),body_J),format='csr')
        cost=float(r@r/2)
        self.last=dict(residual=r,jac=J,cost=cost,parts={'support_image':cost},metrics=dict(observations=metrics,enabled_rows=len(metrics),fixed_arc_ids=self.arc_ids,differentiability_warnings=warnings,five_node_native_projection=True))
        return self.last
