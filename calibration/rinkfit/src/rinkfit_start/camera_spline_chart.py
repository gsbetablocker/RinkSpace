"""Exact uniform-cubic time subfamilies in unchanged fine control storage.

A coarser knot family is an explicit temporal model hypothesis, with no motion
penalty. Cardinal cubic knot refinement maps it exactly into the existing C++
fine blocks. Both source-readout padding and the angle function-value gauge are
preserved. A fine trajectory must first be projected into this family; merely
restricting increments of an arbitrary high-frequency parent is rejected.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
from .native_scene import camera_from_event, parameter_values

VERSION='rinkfit.exact_cardinal_cubic_time_subspace.v1'


def cubic_design(time,first,dt,n,derivative=0):
    t=np.asarray(time,float).ravel();x=(t-first)/dt
    if np.any(x<1-1e-10)or np.any(x>n-2+1e-10):raise ValueError('Outside full allocated cubic domain')
    i=np.floor(x).astype(int);i=np.where(i==n-2,i-1,i);u=x-i
    if np.any(i<1)or np.any(i>n-3):raise ValueError('Invalid four-control cubic support')
    if derivative==0:w=np.c_[(1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3]/6
    elif derivative==1:w=np.c_[-3*(1-u)**2,9*u*u-12*u,-9*u*u+6*u+3,3*u*u]/(6*dt)
    elif derivative==2:w=np.c_[6*(1-u),18*u-12,-18*u+6,6*u]/(6*dt*dt)
    else:raise ValueError('Cubic derivative must be 0,1 or2')
    return sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(len(t)),4),(i[:,None]+np.arange(-1,3)).ravel())),shape=(len(t),n))


def cardinal_refinement(first,dt,n,multiple):
    m=int(multiple)
    if m!=multiple or m<1:raise ValueError('An integer knot-spacing multiple is required')
    grid=int(round(first/dt))
    if abs(grid*dt-first)>1e-12:raise ValueError('Fine centers must share the native zero-time lattice')
    mask=np.array([1.])
    for _ in range(4):mask=np.convolve(mask,np.ones(m))
    mask=mask/(m**3);extent=2*(m-1)
    j0=int(np.ceil((grid-extent)/m));j1=int(np.floor((grid+n-1+extent)/m));nc=j1-j0+1
    rr=[];cc=[];vv=[]
    for j in range(j0,j1+1):
        fi=m*j+np.arange(-extent,extent+1)-grid;ok=(fi>=0)&(fi<n)
        rr.extend(fi[ok]);cc.extend(np.full(np.count_nonzero(ok),j-j0));vv.extend(mask[ok])
    R=sparse.csr_matrix((vv,(rr,cc)),shape=(n,nc))
    if not np.allclose(np.asarray(R.sum(axis=1)).ravel(),1.,atol=1e-15,rtol=0):raise ValueError('Incomplete endpoint padding in refinement')
    return R,dict(first_control_time=float(j0*m*dt),dt=float(m*dt),n_controls=nc,
                  degree=3,refinement_multiple=m,mask=mask.tolist(),mask_offsets=list(range(-extent,extent+1)),
                  fine_storage_first=float(first),fine_storage_dt=float(dt),fine_storage_n=int(n))


class UniformCubicTimeChart:
    version=VERSION
    def __init__(self,model,multiple):
        self.model=deepcopy(model);self.camera=model['camera'];c=self.camera
        self.first=float(c['first_control_time']);self.dt=float(c['dt']);self.n=int(c['n_controls']);self.begin=int(c.get('control_block_begin',1))
        self.R,self.basis=cardinal_refinement(self.first,self.dt,self.n,multiple);self.nc=self.R.shape[1]
        self.domain=(self.first+self.dt,self.first+(self.n-2)*self.dt)
        self.coarse_domain=(self.basis['first_control_time']+self.basis['dt'],self.basis['first_control_time']+(self.nc-2)*self.basis['dt'])
        if self.coarse_domain[0]>self.domain[0]+1e-12 or self.coarse_domain[1]<self.domain[1]-1e-12:raise ValueError('Coarse family does not cover both full readout paddings')
        self._configure(model)

    def _configure(self,model):
        self.nc=self.R.shape[1]
        self.reference=np.asarray((cubic_design([0.],self.first,self.dt,self.n)@self.R).toarray()[0])
        self.dependent=int(np.argmax(abs(self.reference)));self.angle_free=np.delete(np.arange(self.nc),self.dependent)
        rows=list(self.angle_free);cols=list(range(self.nc-1));data=[1.]*(self.nc-1)
        for j,index in enumerate(self.angle_free):
            if self.reference[index]!=0:rows.append(self.dependent);cols.append(j);data.append(-self.reference[index]/self.reference[self.dependent])
        self.angle_expansion=sparse.csr_matrix((data,(rows,cols)),shape=(self.nc,self.nc-1))
        self.coarse_free=np.delete(np.arange(3*self.nc),[3*self.dependent,3*self.dependent+1])
        mapping=np.full(3*self.nc,-1,int);mapping[self.coarse_free]=np.arange(len(self.coarse_free))
        rr=list(self.coarse_free);cc=list(range(len(self.coarse_free)));vv=[1.]*len(rr)
        for channel in (0,1):
            for i in self.angle_free:
                if self.reference[i]!=0:rr.append(3*self.dependent+channel);cc.append(mapping[3*i+channel]);vv.append(-self.reference[i]/self.reference[self.dependent])
        self.coarse_expansion=sparse.csr_matrix((vv,(rr,cc)),shape=(3*self.nc,len(self.coarse_free)))
        self.actual_fine_from_coarse=(sparse.kron(self.R,sparse.eye(3),format='csr')@self.coarse_expansion).tocsr()
        self.storage_from_coarse=self.actual_fine_from_coarse.copy().tolil()
        dep=int(self.camera['dependent_control']);self.storage_from_coarse[3*dep,:]=0.;self.storage_from_coarse[3*dep+1,:]=0.;self.storage_from_coarse=self.storage_from_coarse.tocsr();self.storage_from_coarse.eliminate_zeros()
        self.names=[('time/adaptive/'if hasattr(self,'retained')else'time/coarse/')+f'{int(self.retained[i//3])if hasattr(self,"retained")else i//3:04d}/'+['alpha','beta','logf'][i%3]for i in self.coarse_free]
        channel_scale=np.asarray(model['blocks'][self.begin]['scales'],float)
        if any(not np.array_equal(np.asarray(model['blocks'][self.begin+i]['scales']),channel_scale)for i in range(self.n)):raise ValueError('Use a single existing physical scale per camera channel')
        self.scales=np.tile(channel_scale,self.nc)[self.coarse_free]

    def expanded_fine(self,model=None):return camera_from_event(self.model if model is None else model).controls

    def recover(self,model=None,*,tolerance=2e-10):
        controls=self.expanded_fine(model);coarse=spsolve((self.R.T@self.R).tocsc(),self.R.T@controls)
        coarse[self.dependent,:2]=-(self.reference[self.angle_free,None]*coarse[self.angle_free,:2]).sum(axis=0)/self.reference[self.dependent]
        error=float(np.max(abs(self.R@coarse-controls)))
        if error>tolerance:raise ValueError('Parent trajectory is outside the coarse family; initialize by project_model first (max coefficient mismatch '+str(error)+')')
        return coarse.reshape(-1)[self.coarse_free],dict(max_fine_coefficient_membership_error=error,coarse_angle_reference_error=np.max(abs(self.reference@coarse[:,:2])).item())

    def write_controls(self,coarse_free,model=None):
        out=deepcopy(self.model if model is None else model);q=np.asarray(coarse_free,float)
        if q.shape!=(len(self.coarse_free),)or not np.isfinite(q).all():raise ValueError('Complete finite coarse coordinate vector required')
        stored=(self.storage_from_coarse@q).reshape(self.n,3)
        for i,v in enumerate(stored):out['blocks'][self.begin+i]['values']=v.tolist()
        out.setdefault('provenance',{})['time_subspace']=dict(version=self.version,basis=self.basis,angle_function_reference_time=0.,
            coarse_dependent_control=self.dependent,fine_storage_dependent_control=int(self.camera['dependent_control']),
            model_hypothesis='Uniform cubic knot family; no velocity/acceleration penalty; endpoint readout padding retained',
            allocated_fine_time_domain=list(self.domain),coarse_free_time_scalars=len(self.coarse_free),
            full_fine_storage_and_native_Cpp_registration_unchanged=True)
        return out

    def project_model(self):
        # Cubics and their products have degree3 and6 within every fine cell.
        # Four-point Gauss integration therefore gives the exact L2 objective
        # on the whole allocated domain, including the two source paddings.
        nodes,weights=leggauss(4);edges=self.first+np.arange(1,self.n-1)*self.dt
        times=((edges[:-1,None]+edges[1:,None])/2+nodes[None,:]*self.dt/2).ravel()
        weight=np.tile(weights*self.dt/2,len(edges)-1)
        B=cubic_design(times,self.first,self.dt,self.n);Phi=(B@self.R).tocsr();target=B@self.expanded_fine()
        coarse=np.empty((self.nc,3))
        for channel in (0,1,2):
            X=Phi@self.angle_expansion if channel<2 else Phi
            gram=(X.T@sparse.diags(weight)@X).tocsc();rhs=X.T@(weight*target[:,channel]);sol=spsolve(gram,rhs)
            coarse[:,channel]=self.angle_expansion@sol if channel<2 else sol
        free=coarse.reshape(-1)[self.coarse_free];out=self.write_controls(free);actual=camera_from_event(out)
        predicted=Phi@coarse;err=predicted-target;duration=float(self.domain[1]-self.domain[0])
        direct=cubic_design(times,self.basis['first_control_time'],self.basis['dt'],self.nc)@coarse
        report=dict(version=VERSION,basis=self.basis,projection='Function L2 on the full allocated fine domain; exact piecewise-cubic Gauss4 integration; angular t=0 equality eliminated',
            new_temporal_model_hypothesis=True,no_motion_regularization_strength=True,source_readout_padding_covered=True,
            fine_stored_controls=self.n,coarse_controls=self.nc,fine_time_free_scalars=3*self.n-2,coarse_time_free_scalars=len(free),
            allocated_time_domain=list(self.domain),coarse_allocated_time_domain=list(self.coarse_domain),
            function_L2_RMS_alpha_beta_logf=np.sqrt(np.sum(weight[:,None]*err*err,axis=0)/duration).tolist(),
            sampled_function_max_alpha_beta_logf=np.max(abs(err),axis=0).tolist(),
            exact_refinement_function_difference=float(np.max(abs(predicted-direct))),
            fine_native_angle_reference_error=float(actual.reference_error),projection_does_not_establish_image_fit=True)
        return out,report

    def compose_physical(self,print_chart,model=None):
        """Full print-chart physical differential = C @ combined differential.

        Columns are first every non-time physical parameter, then independent
        coarse alpha/beta/logf values. No fine temporal coefficient survives as
        an independent solver column. Must be called on a coarse-family parent.
        """
        model=self.model if model is None else model;coarse,membership=self.recover(model)
        off=print_chart.offset;start=int(off[self.begin]);stop=int(off[self.begin+self.n]);free=print_chart.free
        non_time=np.flatnonzero((free<start)|(free>=stop));rr=list(non_time);cc=list(range(len(non_time)));vv=[1.]*len(non_time)
        T=self.storage_from_coarse.tocoo()
        for r,c,v in zip(T.row,T.col,T.data):
            ambient=start+int(r)
            if ambient not in print_chart.column:raise ValueError('A fine fixed coefficient leaked into the coarse lift')
            rr.append(print_chart.column[ambient]);cc.append(len(non_time)+int(c));vv.append(float(v))
        C=sparse.csr_matrix((vv,(rr,cc)),shape=(len(free),len(non_time)+len(coarse)))
        names=[model['blocks'][int(np.searchsorted(off[1:],free[j],side='right'))]['name']+'/'+str(int(free[j]-off[int(np.searchsorted(off[1:],free[j],side='right'))]))for j in non_time]+self.names
        scales=np.concatenate([np.asarray(model['blocks'][i]['scales'])for i in range(len(model['blocks']))])[free[non_time]].copy()
        for qi,ai in print_chart.print_ad.items():
            ad=np.asarray(model['blocks'][ai]['values']);span=np.array([ad[1]]*3+[ad[3]-ad[2]]*3)
            for j in range(6):
                row=print_chart.column[int(off[qi]+j)];where=np.flatnonzero(non_time==row)
                if len(where):scales[where[0]]*=span[j]
        return C,np.r_[scales,self.scales],names,dict(non_time_columns=len(non_time),coarse_time_columns=len(coarse),combined_columns=C.shape[1],membership=membership,basis=self.basis)


LOCAL_VERSION='rinkfit.exact_adaptive_cubic_knot_removal.v1'


class LocalCubicTimeChart(UniformCubicTimeChart):
    """Remove selected fine cubic knots, retaining all other fine knots.

    The cubic third-derivative jump at fine knot j is proportional to
    c[j-2]-4c[j-1]+6c[j]-4c[j+1]+c[j+2]. Setting it to zero removes that
    knot exactly, without a motion penalty. Initialization changes only fine
    coefficient centers inside the declared local window.
    """
    version=LOCAL_VERSION
    def __init__(self,model,multiple,start_time=2.46,end_time=2.92):
        super().__init__(model,1);m=int(multiple)
        if m!=multiple or m<2:raise ValueError('Local knot removal requires an integer coarse multiple >=2')
        self.start_time=float(start_time);self.end_time=float(end_time)
        if not self.start_time<self.end_time:raise ValueError('Nonempty declared local time window required')
        times=self.first+self.dt*np.arange(self.n);ground=np.rint(times/self.dt).astype(int)
        inside=(times>=self.start_time-1e-12)&(times<=self.end_time+1e-12)
        self.removed=np.flatnonzero(inside&(ground%m!=0))
        if not len(self.removed)or np.any(self.removed<2)or np.any(self.removed>self.n-3):raise ValueError('Selected local knots lack full five-coefficient support')
        self.retained=np.delete(np.arange(self.n),self.removed);self.nc=len(self.retained)
        rr=np.repeat(np.arange(len(self.removed)),5);cc=(self.removed[:,None]+np.arange(-2,3)).ravel();vv=np.tile([1.,-4.,6.,-4.,1.],len(self.removed))
        self.knot_constraints=sparse.csr_matrix((vv,(rr,cc)),shape=(len(self.removed),self.n))
        Ar=self.knot_constraints[:,self.removed].toarray();Af=self.knot_constraints[:,self.retained]
        dependent=np.linalg.solve(Ar,-Af.toarray());r,c=np.nonzero(dependent)
        R=sparse.csr_matrix((np.r_[np.ones(self.nc),dependent[r,c]],(np.r_[self.retained,self.removed[r]],np.r_[np.arange(self.nc),c])),shape=(self.n,self.nc))
        self.R=R;self.coarse_domain=self.domain
        self.basis=dict(basis_kind='local_exact_cubic_knot_removal',refinement_multiple=m,local_window=[self.start_time,self.end_time],
            removed_fine_indices=self.removed.tolist(),removed_native_knot_times=times[self.removed].tolist(),retained_fine_indices=self.retained.tolist(),
            fine_storage_first=self.first,fine_storage_dt=self.dt,fine_storage_n=self.n,n_controls=self.nc,
            knot_jump_formula=[1,-4,6,-4,1],no_motion_penalty=True)
        self._configure(model)
        residual=self.knot_constraints@self.R
        if np.max(abs(residual.data),initial=0.)>1e-11:raise ValueError('Knot-removal lift failed its exact jump identities')

    def project_model(self):
        original=self.expanded_fine();coarse=original[self.retained].copy();times_control=self.first+self.dt*self.retained
        local=np.flatnonzero((times_control>=self.start_time-1e-12)&(times_control<=self.end_time+1e-12))
        if np.any(abs(self.reference[local])>1e-14):raise ValueError('Local initializer overlaps the angular reference gauge; use an explicit constrained endpoint chart')
        nodes,weights=leggauss(4);edges=self.first+np.arange(1,self.n-1)*self.dt
        times=((edges[:-1,None]+edges[1:,None])/2+nodes[None,:]*self.dt/2).ravel();weight=np.tile(weights*self.dt/2,len(edges)-1)
        B=cubic_design(times,self.first,self.dt,self.n);Phi=(B@self.R).tocsr();target=B@original;baseline=Phi@coarse
        if len(local):
            X=Phi[:,local];gram=(X.T@sparse.diags(weight)@X).tocsc();rhs=X.T@(weight[:,None]*(target-baseline));coarse[local]+=spsolve(gram,rhs)
        coarse[self.dependent,:2]=-(self.reference[self.angle_free,None]*coarse[self.angle_free,:2]).sum(axis=0)/self.reference[self.dependent]
        free=coarse.reshape(-1)[self.coarse_free];out=self.write_controls(free);actual=self.expanded_fine(out)
        before=np.stack([b['values']for b in self.model['blocks'][self.begin:self.begin+self.n]])
        after=np.stack([b['values']for b in out['blocks'][self.begin:self.begin+self.n]])
        outside=(self.first+self.dt*np.arange(self.n)<self.start_time-1e-12)|(self.first+self.dt*np.arange(self.n)>self.end_time+1e-12)
        if not np.array_equal(before[outside].view(np.uint64),after[outside].view(np.uint64)):raise ValueError('Local projection changed an outside fine storage value')
        jump=self.knot_constraints@actual;prediction=B@actual;err=prediction-target;duration=self.domain[1]-self.domain[0]
        changed=np.flatnonzero(np.any(before.view(np.uint64)!=after.view(np.uint64),axis=1));support=[float(self.first+(changed.min()-2)*self.dt),float(self.first+(changed.max()+2)*self.dt)]if len(changed)else None
        report=dict(version=self.version,basis=self.basis,new_temporal_model_hypothesis=True,projection='Exact function L2 on the local retained coefficients only; every outside fine storage coefficient bitwise preserved',
            fine_stored_controls=self.n,retained_control_coordinates=self.nc,removed_knots=len(self.removed),fine_time_free_scalars=3*self.n-2,coarse_time_free_scalars=len(free),
            outside_storage_bitwise_preserved=True,changed_fine_storage_indices=changed.tolist(),function_change_support=support,
            max_removed_knot_jump_numerator=float(np.max(abs(jump))),angle_reference_error=float(camera_from_event(out).reference_error),
            allocated_time_domain=list(self.domain),both_source_paddings_unchanged=True,no_motion_regularization_strength=True,
            function_L2_RMS_alpha_beta_logf=np.sqrt(np.sum(weight[:,None]*err*err,axis=0)/duration).tolist(),
            sampled_function_max_alpha_beta_logf=np.max(abs(err),axis=0).tolist(),projection_does_not_establish_image_fit=True)
        return out,report


def time_chart_from_model(model,multiple=None,window=None):
    declared=model.get('provenance',{}).get('time_subspace',{})
    if multiple is None:
        if declared.get('version')not in (VERSION,LOCAL_VERSION):return None
        multiple=int(declared['basis']['refinement_multiple'])
    if window is None and declared.get('version')==LOCAL_VERSION:window=declared['basis']['local_window']
    return LocalCubicTimeChart(model,multiple,*window)if window is not None else UniformCubicTimeChart(model,multiple)
