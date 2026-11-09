"""Model mass-centre kinematics on the same complete body/shoe state.

Masses are a declared volume proxy, not measured subject mass distribution.
No camera back-projection or independent trajectory is introduced here.
"""
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,diags,kron,eye,vstack,hstack
from .skate_physics import fk_state
from .memory_bounded import slice_delta, slice_reference
from .fullbody_fit import right_jacobian
from .skate_contact import trapezoid_time_weights


def acceleration_operator(times):
    t=np.asarray(times,float);h=np.diff(t)
    if len(t)<3 or np.any(h<=0):raise ValueError('COM derivatives need three increasing source times')
    a=2/(h[:-1]+h[1:]);n=len(t)-2;rows=np.repeat(np.arange(n),3)
    values=np.c_[a/h[:-1],-a*(1/h[:-1]+1/h[1:]),a/h[1:]]
    return coo_matrix((values.ravel(),(rows,(np.arange(n)[:,None]+np.arange(3)).ravel())),shape=(n,len(t))).tocsr()


def com_local_jacobians(mass,base,delta):
    """Analytic COM derivative w.r.t. current local log increments and shoes."""
    s=fk_state(mass.ref,base,delta);C,_=mass.from_pose(s);com=mass.com(C)
    n=len(delta);local=np.zeros((n,3,69));ratio=mass.binding.ratio
    local[:,:,:3]=np.eye(3)*ratio*mass.mass.sum()
    axes=s['Q']@right_jacobian(np.asarray(delta)[:,3:6])
    arm=com-s['P']*ratio
    local[:,:,3:6]=np.cross(axes.swapaxes(1,2),arm[:,None,:]).swapaxes(1,2)
    parents=mass.ref.body.parents
    def descendant(k,j):
        while k>0:
            if k==j:return True
            k=int(parents[k])
        return False
    for j in range(1,22):
        affected=np.array([descendant(int(k),j) for k in mass.ids]+[descendant(7,j),descendant(8,j)])
        if not affected.any():continue
        lever=np.einsum('s,nsc->nc',mass.mass[affected],C[:,affected]-s['native_world'][:,j,None]*ratio)
        first=6+(j-1)*3
        axes=s['Q']@s['frames'][:,j]@right_jacobian(np.asarray(delta)[:,first:first+3])
        local[:,:,first:first+3]=np.cross(axes.swapaxes(1,2),lever[:,None,:]).swapaxes(1,2)
    shoe=np.zeros((n,3,6))
    for f,ankle in enumerate((7,8)):
        lever=(C[:,mass.count+f]-s['native_world'][:,ankle]*ratio)*mass.mass[mass.count+f]
        shoe[:,:,f*3:f*3+3]=np.cross(s['R'][:,f].swapaxes(1,2),lever[:,None,:]).swapaxes(1,2)
    return com,local,shoe


def com_quadrature_times(problem, config):
    """Keep native/exposure times and resolve every existing control span.

    Added knot probes closer than one source-clock tick to an existing probe
    are redundant quadrature points; the native/RS clocks themselves stay intact.
    """
    native=np.asarray(problem.reference.times,float)
    mode=config.get('quadrature','native_source_centres')
    if mode=='native_source_centres':return native.copy()
    if mode!='native_exposure_and_curve_spans':raise ValueError('Unknown COM quadrature')
    physical=np.asarray(problem.physics_times,float)
    def distance(values, grid):
        indices=np.searchsorted(grid,values)
        return np.minimum(abs(values-grid[np.minimum(indices,len(grid)-1)]),
                          abs(values-grid[np.maximum(indices-1,0)]))
    base=np.sort(np.r_[native,physical[distance(physical,native)>1e-9]])
    if np.any(np.diff(base)<=1e-9):raise ValueError('Distinct existing COM quadrature times are unresolved')
    knots=np.unique(problem.body.state.knots)
    probes=(knots[:-1,None]+np.diff(knots)[:,None]*np.arange(4)[None]/4).ravel()
    probes=np.unique(probes[(probes>=base[0])&(probes<=base[-1])])
    from .body_clock import time_base
    numerator,denominator=time_base(problem.reference.p);tick=numerator/denominator
    probes=probes[distance(probes,base)>=tick]
    added=[]
    for t in probes:
        if not added or t-added[-1]>=tick:added.append(t)
    return np.sort(np.r_[base,added])


class ModelCOMMotionFactor:
    """Finite COM curvature prior; confirmed flights have separate gravity rows."""
    def __init__(self,problem,mass,config):
        self.p=problem;self.mass=mass;self.ref=problem.reference;self.times=com_quadrature_times(problem,config)
        self.quadrature=config.get('quadrature','native_source_centres')
        self.base=self.ref.at(self.times);self.A=acceleration_operator(self.times)
        self.scale=float(config['acceleration_scale_mps2'])
        if self.scale<=0:raise ValueError('Positive finite COM engineering scale required')
        from .phase_ledger import flight_mask
        from .body_clock import time_base
        numerator,denominator=time_base(self.ref.p)
        self.source_pts_float=self.ref.p['pts'][0]+self.times*denominator/numerator
        native_rows=np.searchsorted(self.times,self.ref.times)
        np.testing.assert_array_equal(self.times[native_rows],self.ref.times)
        self.source_pts_float[native_rows]=self.ref.p['pts']
        flight=flight_mask(self.ref,self.source_pts_float)
        w=np.sqrt(trapezoid_time_weights(self.times)[1:-1])/self.scale
        w[flight[1:-1]]=0.
        self.w=w;self.operator=kron(diags(w)@self.A,eye(3),format='csr')
        self.flight_rows={};self.flight_operators={};self.flight_weights={}
        self.gravity=np.array([0.,0.,-9.81])
        self.flight_scale=config.get('confirmed_flight_acceleration_scale_mps2')
        if self.flight_scale is not None:
            self.flight_scale=float(self.flight_scale)
            if not np.isfinite(self.flight_scale) or self.flight_scale<=0:
                raise ValueError('Finite positive dense confirmed-flight acceleration scale required')
            if self.quadrature!='native_exposure_and_curve_spans':
                raise ValueError('Dense confirmed-flight physics requires exposure and curve-span quadrature')
            episodes=[e for e in getattr(self.ref,'source_document',{}).get('source_episode_review',{}).get('episodes',[])
                      if e['kind']=='flight_visible_candidate']
            if not episodes:raise ValueError('Dense flight physics requires reviewed source intervals')
            intervals=sorted((e['start_pts'],e['end_pts_inclusive'])for e in episodes)
            if any(not np.isfinite([lo,hi]).all() or lo>hi for lo,hi in intervals):
                raise ValueError('Invalid reviewed dense flight interval')
            if any(left[1]>=right[0] for left,right in zip(intervals[:-1],intervals[1:])):
                raise ValueError('Overlapping reviewed dense flight intervals')
            claimed=np.zeros(len(self.times)-2,bool)
            measure=trapezoid_time_weights(self.times)[1:-1]
            for episode in episodes:
                name=episode['episode_id']
                if name in self.flight_rows:raise ValueError('Duplicate dense flight episode')
                inside=(self.source_pts_float>=episode['start_pts'])&(self.source_pts_float<=episode['end_pts_inclusive'])
                # All three derivative samples must belong to this same reviewed
                # interior. No onset/landing force is inferred across its boundary.
                keep=inside[:-2]&inside[1:-1]&inside[2:]
                if not keep.any() or np.any(keep&claimed):
                    raise ValueError('Empty or overlapping reviewed dense flight interior')
                claimed|=keep;rows=np.flatnonzero(keep)
                weight=np.sqrt(measure[rows])/self.flight_scale
                self.flight_rows[name]=rows;self.flight_weights[name]=weight
                self.flight_operators[name]=kron(diags(weight)@self.A[rows],eye(3),format='csr')
    def evaluate(self,x,jacobian=False):
        state=self.p.body.state;d=state.increments(x,self.times)
        J=None
        if jacobian:
            centres=[];tiles=[]
            for first in range(0,len(self.times),512):
                tile=slice(first,first+512)
                dt=slice_delta(d,tile)
                base=slice_reference(self.base,tile,len(self.times))
                com,local,shoe=com_local_jacobians(self.mass,base,dt)
                centres.append(com)
                part=dt.chain(local)
                if hasattr(state,'shoe_chain'):part=part+state.shoe_chain(shoe,dt.parameters,dt.sample_times)
                tiles.append(part)
            com=np.concatenate(centres)
            Jcom=vstack(tiles,format='csr')
            J=self.operator@Jcom
            J=hstack((csr_matrix((J.shape[0],17)),J),format='csr')
        else:
            C,_=self.mass.evaluate(self.base,d);com=self.mass.com(C)
        r=np.asarray(self.operator@com.ravel())
        acceleration=self.A@com;magnitude=np.linalg.norm(acceleration,axis=1);peak=int(np.argmax(magnitude))
        from .body_clock import time_base
        numerator,denominator=time_base(self.ref.p);source_seconds=self.source_pts_float[1:-1]*numerator/denominator
        lsp=(source_seconds>=156.2)&(source_seconds<=165.2)
        from .weighted_blocks import block
        name='actual_model_com_acceleration'
        entries=[block(name,r,J,dict(
            source_pts=self.source_pts_float[1:-1],native_source_pts=self.ref.p['pts'],quadrature=self.quadrature,unit='m/s2',scale=self.scale,
            meaning='finite curvature regularizer, not measured acceleration; excluded in confirmed flight',mass_is_proxy=True))]
        metrics={name:dict(scale_mps2=self.scale,mass_is_proxy=True,whole_programme=True,quadrature=self.quadrature,
            evaluation_samples=len(self.times),native_exposures=len(self.ref.p['pts']),flight_excluded_samples=int(np.count_nonzero(self.w==0)),
            sampled_acceleration_max_mps2=float(magnitude[peak]),peak_source_seconds=float(source_seconds[peak]),
            regularized_acceleration_max_mps2=float(magnitude[self.w>0].max()) if np.any(self.w>0) else None,
            excluded_flight_acceleration_max_mps2=float(magnitude[self.w==0].max()) if np.any(self.w==0) else None,
            lsp_sampled_acceleration_max_mps2=float(magnitude[lsp].max()) if lsp.any() else None,
            sample_extrema_are_not_continuous_certificates=True)}
        flight_metrics=[]
        for episode,rows in self.flight_rows.items():
            raw=acceleration[rows]-self.gravity
            weight=self.flight_weights[episode]
            residual=(weight[:,None]*raw).ravel();matrix=None
            if jacobian:
                partial=self.flight_operators[episode]@Jcom
                matrix=hstack((csr_matrix((len(residual),17)),partial),format='csr')
            entries.append(block('episode/'+episode+'/flight_dense_com_acceleration',residual,matrix,dict(
                source_pts=self.source_pts_float[1:-1][rows],quadrature=self.quadrature,unit='m/s2',scale=self.flight_scale,
                gravity_world_mps2=self.gravity,all_three_samples_in_same_reviewed_flight=True,
                mass_is_proxy=True,meaning='Same continuous body and attached shoes; gravity residual inside reviewed flight, not an independent trajectory')))
            norm=np.linalg.norm(raw,axis=1);worst=int(np.argmax(norm));measure=weight**2
            flight_metrics.append(dict(episode_id=episode,derivative_rows=len(rows),
                source_seconds=[float(source_seconds[rows[0]]),float(source_seconds[rows[-1]])],
                rms_gravity_error_mps2=float(np.sqrt(np.sum(measure*norm**2)/measure.sum())),
                max_gravity_error_mps2=float(norm[worst]),worst_source_pts=float(self.source_pts_float[1:-1][rows[worst]])))
        if flight_metrics:
            metrics['confirmed_flight_dense_com']=dict(episodes=flight_metrics,scale_mps2=self.flight_scale,
                gravity_world_mps2=self.gravity.tolist(),mass_is_proxy=True,original_native_flight_factors_retained=True,
                whole_programme_state=True,flight_boundary_crossing_stencils_excluded=True,accuracy_accepted=False)
            r=np.concatenate([entry.residual for entry in entries])
            if jacobian:J=vstack([entry.jacobian for entry in entries],format='csr')
        return dict(residual=r,jac=J,parts={entry.name:float(entry.residual@entry.residual/2)for entry in entries},
            weighted_blocks=entries,metrics=metrics)


    def export(self,x,path,**identity):
        d=self.p.body.state.increments(x,self.times);C,_=self.mass.evaluate(self.base,d);com=self.mass.com(C)
        extra={}
        if self.flight_rows:
            groups=np.full(len(self.times)-2,-1,int)
            for group,rows in enumerate(self.flight_rows.values()):groups[rows]=group
            extra=dict(confirmed_flight_acceleration_group=groups,confirmed_flight_episode_ids=np.array(list(self.flight_rows)),
                confirmed_flight_gravity_world_mps2=self.gravity,
                confirmed_flight_acceleration_scale_mps2=np.array(self.flight_scale))
        np.savez_compressed(path,evaluation_time_s=self.times,evaluation_source_pts=self.source_pts_float,
            native_source_pts=self.ref.p['pts'],com_world_m=com,acceleration_time_s=self.times[1:-1],
            acceleration_source_pts=self.source_pts_float[1:-1],acceleration_world_mps2=self.A@com,
            regularized_acceleration_rows=self.w>0,mass_is_proxy=np.array(True),
            quadrature=np.array(self.quadrature),**extra,**{k:np.array(v)for k,v in identity.items()})


def export_com_diagnostics(mass,times,C,R):
    """Full source-clock outputs; interior derivatives are labelled explicitly."""
    t=np.asarray(times);com=mass.com(C);velocity=np.gradient(com,t,axis=0,edge_order=2)
    acc=np.empty_like(com);acc[1:-1]=acceleration_operator(t)@com
    acc[0]=acc[1];acc[-1]=acc[-2]
    H,rates,_=mass.H(t,C,R);fullH=np.full_like(com,np.nan);fullH[rates['source_indices']]=H
    return dict(com_world_mid=com,com_velocity_world_mps=velocity,com_acceleration_world_mps2=acc,
        com_specific_angular_momentum_world_m2ps=fullH,
        com_required_specific_external_force_world_mps2=acc-np.array([0.,0.,-9.81]),
        com_derivative_interior=np.r_[False,np.ones(len(t)-2,bool),False],
        com_mass_model=np.array('complete shaped body segment volume proxy plus rigid shoes; not measured subject mass'),
        com_derivative_method=np.array('native source-clock finite differences; no cut reset; endpoint acceleration copied from adjacent interior'),
        trajectory_reference=np.array('model_com'),com_segment_mass_fractions=mass.mass.copy(),
        com_segment_centres_world_m=np.asarray(C).copy(),com_segment_body_joint_ids=np.asarray(mass.ids).copy())
