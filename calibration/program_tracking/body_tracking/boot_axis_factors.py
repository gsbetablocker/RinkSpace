"""Finite source boot-contour axes on the existing body/ankle state.

No new shoe pose parameters. Steel subsegments are not silently mapped to the
whole blade axis or to fixed template material endpoints.
"""
import copy
from dataclasses import replace
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,hstack
from .support_image import SupportImageFactors,_vee
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection


def axis_region(heel,toe,heel_half,toe_half):
    """Unoriented centre angle and finite rectangle-induced angular envelope."""
    heel,toe,heel_half,toe_half=map(lambda v:np.asarray(v,float),(heel,toe,heel_half,toe_half))
    d=toe-heel
    if np.linalg.norm(d)<=np.linalg.norm(heel_half)+np.linalg.norm(toe_half):
        raise ValueError('Boot axis regions overlap; direction is unobservable')
    centre=np.arctan2(d[1],d[0]); corners=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    vectors=(toe+corners[:,None]*toe_half)-(heel+corners[None,:]*heel_half)
    angles=np.arctan2(vectors[...,1],vectors[...,0])-centre
    angles=.5*np.arctan2(np.sin(2*angles),np.cos(2*angles))
    return float(centre),float(np.max(np.abs(angles)))


def finite_axis_term(heel,toe,centre,half_angle,weight):
    """Scalar angular-band residual and native-pixel Jacobian, 180deg symmetric."""
    d=np.asarray(toe,float)-heel;length2=float(d@d)
    if length2<=1e-16:raise ValueError('Projected boot axis is degenerate')
    angle=np.arctan2(d[1],d[0])-centre
    angle=.5*np.arctan2(np.sin(2*angle),np.cos(2*angle))
    if half_angle<=0 or weight<=0:raise ValueError('Positive finite angular uncertainty and weight required')
    scale=np.sqrt(weight)/half_angle
    residual=scale*np.sign(angle)*max(abs(angle)-half_angle,0.)
    derivative=scale*np.r_[d[1],-d[0],-d[1],d[0]]/length2 if abs(angle)>half_angle else np.zeros(4)
    return float(residual),derivative,float(abs(angle))


class BootAxisFactors:
    """Source heel/toe contours, fixed foot identity, native five-node geometry."""
    def __init__(self,problem,document,axes,*,include_review=False):
        self.problem=problem;self.body=problem.body;self.axes=[];helpers=[]
        for row in axes:
            if row.get('role')!='fit' and not include_review:continue
            if row.get('visibility')!='visible':continue
            if not row.get('optimization_enabled',True) and not (include_review and row.get('role')=='review'):continue
            if row.get('foot') not in (0,1):raise ValueError('Unassigned source foot cannot drive an axis')
            if row.get('source_semantic')!='boot_heel_to_toe_contour_axis':
                raise ValueError('A visible steel subsegment is not a boot/full-blade axis')
            centre,half=axis_region(row['heel_uv'],row['toe_uv'],row['heel_half_extent_uv'],row['toe_half_extent_uv'])
            item=copy.deepcopy(row);item.update(centre_angle_rad=centre,half_angle_rad=half);self.axes.append(item)
            for feature,ids in [('heel',[0,7]),('toe',[1,8])]:
                h=copy.deepcopy(row['region_observations'][feature]);h['role']='fit';h['optimization_enabled']=True
                h['adapter_point_ids']=ids;helpers.append(h)
        self.helper=None
        if self.axes:
            doc=copy.deepcopy(document);doc['observations']=helpers
            self.helper=SupportImageFactors(problem,doc,arc_ids={})
        self.last=None

    def evaluate(self,x,jacobian=False):
        if not self.axes:return dict(residual=np.empty(0),jac=csr_matrix((0,self.problem.size)) if jacobian else None,parts={'boot_axis':0.},metrics={'boot_axis':[]})
        helper=self.helper;delta=self.body.state.increments(x,helper.times)
        s=fk_state(helper.reference,helper.ref,delta);nodes,Jrotation,warnings=helper.geometry(s)
        projection=moving_projection(self.problem.camera,nodes,helper.frames,jacobian=jacobian)
        uv=projection['uv'][:,0];residual=[];D=[];metrics=[]
        for i,row in enumerate(self.axes):
            weight=row['objective_weight']*row['reliability']
            r,d,error=finite_axis_term(uv[2*i],uv[2*i+1],row['centre_angle_rad'],row['half_angle_rad'],weight)
            residual.append(r);D.append(d)
            metrics.append(dict(id=row['id'],pts=row['pts'],foot=row['foot'],role=row['role'],source_semantic=row['source_semantic'],model_heel_uv=uv[2*i].tolist(),model_toe_uv=uv[2*i+1].tolist(),axis_error_degrees=float(np.rad2deg(error)),angular_tolerance_degrees=float(np.rad2deg(row['half_angle_rad'])),outside_angle_degrees=float(np.rad2deg(max(error-row['half_angle_rad'],0))),source_uncertainty=row['source_uncertainty'],not_blade_steel_or_fixed_material=True))
        r=np.asarray(residual);J=None
        if jacobian:
            n=len(helper.rows);pick=np.arange(n*5);foot=np.repeat(helper.feet,5);R=s['R'][pick,foot]
            def column(active):
                h=1e-6;plus=delta.copy();minus=delta.copy();plus[:,active]+=h;minus[:,active]-=h
                a=fk_state(helper.reference,helper.ref,plus);b=fk_state(helper.reference,helper.ref,minus)
                dO=(a['O'][pick,foot]-b['O'][pick,foot])/(2*h);dR=(a['R'][pick,foot]-b['R'][pick,foot])/(2*h)
                eta=_vee(R.transpose(0,2,1)@dR).reshape(n,5,3)
                dX=dO.reshape(n,5,3)+np.einsum('nlij,nlj->nli',Jrotation,eta)
                return np.einsum('nplc,nlc->npl',projection['pixel_jac_nodes'][:,0],dX)
            local=np.stack(list(self.body.execution.map_columns(column,self.body.state.active)),-1)
            if hasattr(delta,'chain'):
                from .absolute_motion_state import chain_exposure
                raw=chain_exposure(local,delta)
            else:raw=csr_matrix(np.einsum('nald,nlm->namd',local,helper.B).reshape(2*n,self.body.size))
            count=len(self.axes);rr=np.repeat(np.arange(count),4);cc=np.arange(count*4)
            operator=coo_matrix((np.asarray(D).ravel(),(rr,cc)),shape=(count,2*n)).tocsr()
            J=hstack((csr_matrix((count,17)),operator@raw),format='csr')
        self.last=dict(residual=r,jac=J,parts={'boot_axis':float(r@r/2)},metrics={'boot_axis':metrics,'boot_axis_geometry_warnings':warnings})
        return self.last
