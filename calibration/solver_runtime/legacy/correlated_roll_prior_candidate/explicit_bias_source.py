"""Explicit installation-correlated roll-prior statistical candidate.
The existing camera state and real observation factors do not change.
"""
import numpy as np,cv2
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock

SIGMA=float(np.deg2rad(.5))

def append_selected_bias_csr(matrix,selected_rows,column,total_columns,*,sigma=SIGMA):
    m=matrix.tocsr(copy=False);rr=np.asarray(selected_rows,int)
    if (len(rr) and (np.any(np.diff(rr)<=0) or rr[0]<0 or rr[-1]>=m.shape[0])):raise ValueError("Distinct ordered original prior rows required")
    if column<m.shape[1] or column>=total_columns:raise ValueError("Explicit statistical tail required")
    if not len(rr):return csr_matrix((m.data,m.indices,m.indptr),shape=(m.shape[0],total_columns),copy=False)
    # Every original entry, including repeated source columns, keeps its order.
    ptr=m.indptr+np.searchsorted(rr,np.arange(m.shape[0]+1),side="left").astype(m.indptr.dtype)
    values=np.empty(ptr[-1],m.data.dtype);columns=np.empty(ptr[-1],m.indices.dtype)
    first=int(rr[0]);stop=int(m.indptr[first])
    values[:stop]=m.data[:stop];columns[:stop]=m.indices[:stop]
    cursor=0
    for row in range(first,m.shape[0]):
        left,right=int(m.indptr[row]),int(m.indptr[row+1]);dest=int(ptr[row]);n=right-left
        values[dest:dest+n]=m.data[left:right];columns[dest:dest+n]=m.indices[left:right]
        if cursor<len(rr) and row==rr[cursor]:
            values[dest+n]=-1./sigma;columns[dest+n]=column;cursor+=1
    result=csr_matrix((values,columns,ptr),shape=(m.shape[0],total_columns),copy=False)
    result.has_sorted_indices=m.has_sorted_indices
    result.has_canonical_format=m.has_canonical_format
    return result

def theta_and_gradient(vectors):
    theta=[];gradient=[]
    for v in np.asarray(vectors,float):
        r,raw=cv2.Rodrigues(np.ascontiguousarray(v));dr=raw.reshape(3,3,3)
        x,y=r[0,2],-r[1,2];norm=x*x+y*y
        if norm<1e-8:raise ValueError("Original roll domain undefined")
        theta.append(np.arctan2(x,y));gradient.append((y*dr[:,0,2]+x*dr[:,1,2])/norm)
    return np.asarray(theta),np.asarray(gradient)

class BiasProfileUpdate:
    def __init__(self,base,core_count):self.base=base;self.core_count=core_count
    def __call__(self,q):return self.base(np.asarray(q)[:self.core_count])
    def checkpoint_arrays(self):return self.base.checkpoint_arrays()
    def iter_checkpoint_arrays(self):return self.base.iter_checkpoint_arrays()
    def restore_checkpoint(self,data):return self.base.restore_checkpoint(data)

class ExplicitBiasSource:
    def __init__(self,cap,state,*,component_sigma_degrees=.5):
        if not np.isfinite(component_sigma_degrees) or component_sigma_degrees<=0:
            raise ValueError('Explicit positive finite engineering roll-prior sigma required')
        self.sigma=float(np.deg2rad(component_sigma_degrees))
        self.cap=cap;self.state=state;self.core_count=len(state.q);self.count=self.core_count+3
        self.order=tuple(state.ids);self.slots=np.arange(self.core_count,self.count)
        self.programme={b.name:b for b in state.blocks}
        self.external_roll_rows=[]
        self.external_active_positions=[]
        if state.external is not None:
            from all_installation_optics_candidate.all_installation_optics_graph import evaluate_external_with_optics
            external_evaluator=cap.get('current_external_evaluator')
            actual=(evaluate_external_with_optics(state,state.q,jacobian=False) if external_evaluator is None
                    else external_evaluator(state.q,jacobian=False))
            if [r["id"] for r in actual.source_row_ledger]!=state.external_contract[2]:
                raise ValueError("Original external factor ledger changed")
            for record in actual.source_row_ledger:
                if str(record["id"]).startswith("roll_"):
                    pts=int(str(record["id"])[5:])
                    if pts not in set(state.external.held_pts):
                        begin,end=map(int,record["residual_slice"])
                        if end-begin!=1:raise ValueError("One original roll scalar row required")
                        self.external_roll_rows.append(begin)
                        self.external_active_positions.append(state.external_start+4*state.external.index[pts])
            if len(self.external_roll_rows)!=len(state.external.canonical_pts)-len(state.external.held_pts):
                raise ValueError("Every active original external roll must remain identified")
        self.external_roll_rows=np.asarray(self.external_roll_rows,int)
        self.quadratic=np.r_[cap["quadratic"],np.ones(3,bool)]
        self.scales=np.r_[cap["scales"],np.ones(3)]
        self.updater=BiasProfileUpdate(cap["profile_update"],self.core_count)
        self.validated_programme=set();self.validated_external=False
        self.contract=dict(schema="rinkspace.installation-correlated-roll-explicit-bias-candidate.v1",
            original_core_parameter_count=self.core_count,parameter_count=self.count,
            statistical_bias_parameter_indices=self.slots.tolist(),installation_order=list(self.order),
            programme_actual_views=sum(len(b.bundle.stamps) for b in state.blocks),
            additional_active_external_rolls=len(self.external_roll_rows),
            unobserved_external_held_rolls_excluded_from_bias_coupling=len(state.external.held_pts) if state.external is not None else 0,
            unobserved_external_held_Rf_coordinates_fixed=4*len(state.external.held_pts) if state.external is not None else 0,
            component_sigma_radians=self.sigma,component_sigma_degrees=float(component_sigma_degrees),
            marginal_theta_prior_sigma_degrees=float(np.sqrt(2)*component_sigma_degrees),
            prior_covariance_changed_explicitly=True,three_nuisance_parameters_are_statistical_not_extra_camera_freedom=True,
            original_real_source_uv_sigma_roles_order_and_every_nonroll_prior_preserved=True,
            new_hardware_hypothesis_prior_rows=3,
            original_source_row_count=len(cap["scales"]),new_row_count=len(self.scales),
            hardware_covariance_is_candidate_not_verified_measurement=True)
        if self.sigma!=SIGMA:
            self.contract.update(inherited_roll_prior_component_sigma_degrees=.5,
                roll_prior_is_unmeasured_engineering_hypothesis=True,
                roll_prior_sigma_revision_explicit=True)
    def unpack(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.count,) or not np.isfinite(q).all():raise ValueError("Complete physical plus finite statistical state required")
        return q[:self.core_count],q[self.slots]
    def initial_bias(self,core):
        totals=np.zeros(3);counts=np.zeros(3,int)
        for b in self.state.blocks:
            i=self.order.index(b.bundle.calibration.installation_id)
            vv=core[b.start:b.start+4*len(b.bundle.stamps)].reshape(-1,4)
            theta,_=theta_and_gradient(vv[:,:3]);totals[i]+=theta.sum();counts[i]+=len(theta)
        if len(self.external_active_positions):
            vv=np.array([core[p:p+3] for p in self.external_active_positions])
            theta,_=theta_and_gradient(vv);totals[0]+=theta.sum();counts[0]+=len(theta)
        self.contract["profiled_initial_active_roll_counts"]=counts.tolist()
        return totals/(counts+1.)
    def guard(self,q):
        core,_=self.unpack(q)
        return self.cap["guard"](core)
    def blocks(self,q,jacobian=True):
        core,bias=self.unpack(q);cursor=0
        for block in self.cap["iter_source_blocks"](core,jacobian):
            identity=block.row_ledger;family=identity["family"]
            residual=np.asarray(block.r).copy();selected=np.empty(0,int);installation=None
            if family.startswith("programme:"):
                b=self.programme[family[10:]];n=len(b.bundle.stamps)
                selected=np.arange(len(residual)-n,len(residual))
                installation=self.order.index(b.bundle.calibration.installation_id)
                if family not in self.validated_programme:
                    vv=core[b.start:b.start+4*n].reshape(-1,4)
                    expected=theta_and_gradient(vv[:,:3])[0]/SIGMA
                    if not np.allclose(residual[selected],expected,rtol=0.,atol=2e-12):
                        raise ValueError("The original programme tail is not exactly its roll prior")
                    if not np.all(np.asarray(block.quadratic)[selected]) or not np.all(np.asarray(block.scales)[selected]==1.):
                        raise ValueError("Original prior role or scale changed")
                    self.validated_programme.add(family)
            elif family=="external_main_to_low":
                selected=self.external_roll_rows;installation=0
                if not self.validated_external:
                    expected=theta_and_gradient(np.array([core[p:p+3] for p in self.external_active_positions]))[0]/SIGMA
                    if not np.allclose(residual[selected],expected,rtol=0.,atol=2e-12):
                        raise ValueError("The external source ledger is not the active original roll prior")
                    self.validated_external=True
            if installation is not None:
                residual[selected]-=bias[installation]/SIGMA
                residual[selected]*=SIGMA/self.sigma
            if jacobian:
                if installation is not None:
                    original=block.J
                    if self.sigma!=SIGMA:
                        original=original.tocsr(copy=True)
                        for row in selected:
                            original.data[original.indptr[row]:original.indptr[row+1]]*=SIGMA/self.sigma
                    matrix=append_selected_bias_csr(original,selected,int(self.slots[installation]),self.count,sigma=self.sigma)
                else:
                    m=block.J.tocsr(copy=False)
                    matrix=csr_matrix((m.data,m.indices,m.indptr),shape=(m.shape[0],self.count),copy=False)
            else:matrix=None
            ledger=dict(identity,statistical_roll_bias_inst=installation,
                original_source_order_retained=True,prior_covariance_changed_explicitly=True)
            yield SourceRowBlock(residual,matrix,block.quadratic,block.scales,ledger)
            cursor+=len(residual)
        if cursor!=len(self.cap["scales"]):raise ValueError("Original source rows omitted")
        matrix=csr_matrix((np.full(3,1./self.sigma),(np.arange(3),self.slots)),shape=(3,self.count)) if jacobian else None
        yield SourceRowBlock(bias/self.sigma,matrix,np.ones(3,bool),np.ones(3),
            dict(family="installation_correlated_roll_hardware_bias",original_start=cursor,original_stop=cursor+3,
                source_role="explicit_statistical_hardware_hypothesis_candidate"))
