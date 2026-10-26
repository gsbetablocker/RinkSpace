/* Resident LM: ragged connected local systems, sparse retained CSR.
 * Scalar station elimination is division; larger dense blocks use cuSOLVER.
 * cuDSS owns the retained sparse factorization. No host numerical solve. */
inline void dense_check(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cuSOLVER status "+std::to_string(int(s)));}
inline void sparse_check(cudssStatus_t s){if(s!=CUDSS_STATUS_SUCCESS)throw std::runtime_error("cuDSS status "+std::to_string(int(s)));}
struct LocalWorkspace {size_t matrix,solved;};
// Stop only at completed stream boundaries; accepted state remains intact.
struct SolverStopped : std::exception {
    const char* what()const noexcept override{return "native solve stopped by caller";}
};

struct PrintConstraint {double margin=INFINITY,station=0.;int path=-1,side=0;};
struct PrintProbe {double station;int path;};
struct MountConstraint {double margin=INFINITY;int kind=-1;};
struct RetainedConstraint {double margin=INFINITY;int column=-1,sign=1;};
struct CoupledBoundary {
    int family;CameraConstraint camera;PrintConstraint print;MountConstraint mount;RetainedConstraint retained;
    bool same(const CoupledBoundary& b)const{
        if(family!=b.family)return false;
        if(family==3)return retained.column==b.retained.column&&retained.sign==b.retained.sign;
        if(family==2)return mount.kind==b.mount.kind;
        if(family==1)return print.path==b.print.path&&print.station==b.print.station&&print.side==b.print.side;
        const auto& a=camera;const auto& c=b.camera;
        return a.kind==c.kind&&a.cell==c.cell&&a.channel==c.channel&&a.u==c.u&&a.sign==c.sign&&
            a.minimum==c.minimum&&a.nx==c.nx&&a.ny==c.ny&&a.nz==c.nz&&a.limit==c.limit;
    }
};
// The existing bounded nonlinear restoration admits at most this many cuts.
constexpr int coupled_projection_limit=64;
struct CoupledProjectionView {double *base,*gradients,*responses,*gram,*target,*dual;};
struct SolverView {
    const int *locals,*representatives;const size_t* local_gradient;const LocalWorkspace* workspace;
    const double *lower,*upper;int* blocked;
    double *matrix,*solved,*rhs,*csr,*step,*applied,*scalars,*parts,*residual,*scratch;
    const double* damping;double* image_metric;int nlocal;
    double *delta_lower,*delta_upper,*active_delta;
    const int* rig_bound_columns;int* retained_blocked;double *retained_lower,*retained_upper,*retained_delta;
    CameraConstraint *envelope_candidates=nullptr,*envelope_choice=nullptr;
    double *envelope_gradient=nullptr,*envelope_delta=nullptr,*envelope_response=nullptr;
    PrintConstraint *print_candidates=nullptr,*print_choice=nullptr;double* print_directions=nullptr;
    const PrintProbe* print_probes=nullptr;size_t print_count=0;MountConstraint* mount_choice=nullptr;
};
__global__ void lm_start(LinearView l,const double* h,const double* shared,const double* stats,double* s){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double maximum=0.;
    if(id<size_t(l.q))maximum=shared[l.csr_offsets[id]]; // diagonal is first in upper CSR
    if(id<size_t(l.ncomponents)){auto c=l.components[id];for(int j=c.q_count;j<c.dim;++j)maximum=fmax(maximum,component_h(h,c,j,j));}
    maximum=block_reduce<1>(maximum);double cost=block_reduce<0>(id<size_t(l.ng)?stats[3*id+2]:0.);
    if(!threadIdx.x){atomic_max_nonnegative(s+2,maximum);atomicAdd(s+1,cost);}
}
__global__ void lm_set_lambda(double* s,double lambda,double tau,bool jacobi){if(!threadIdx.x&&!blockIdx.x)s[0]=lambda>0?lambda:tau*(jacobi?1.:s[2]);}
// Marquardt metric in the ORIGINAL joint coordinates. Recomputed from each
// accepted state's complete robust Hessian; no observations or directions drop.
__global__ void lm_set_damping(LinearView l,SolverView s,const double* h,const double* shared,bool jacobi){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    const double floor=64*std::numeric_limits<double>::epsilon()*s.scalars[2];
    if(i<l.q){double d=shared[l.csr_offsets[i]];const_cast<double*>(s.damping)[i]=jacobi?fmax(d,floor>0?floor:1.):1.;}
    if(i<l.ncomponents){auto c=l.components[i];for(int j=0;j<c.local_count;++j){int k=c.q_count+j,id=s.locals[c.local_begin+j];double d=component_h(h,c,k,k);const_cast<double*>(s.damping)[l.q+id]=jacobi?fmax(d,floor>0?floor:1.):1.;}}
}
// Optional trust metric for curve nuisance coordinates: full image displacement
// remains measurable even where a rank-deficient source W has a tangent null.
// This changes damping only, never W, the objective, bounds or active variables.
__global__ void lm_image_global_accumulate(View v,LinearView l,SolverView s,const double* jac){
    size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=v.n)return;auto rp=l.row_plan[row];
    const double* r=jac+rp.jac;for(int j=0;j<rp.count;++j){double x=r[j],y=r[rp.count+1+j];atomicAdd(s.image_metric+l.columns[rp.begin+j],x*x+y*y);}
}
__global__ void lm_image_global_metric(LinearView l,SolverView s){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<l.q){double* d=const_cast<double*>(s.damping)+j;*d=fmax(1.,fmax(*d,s.image_metric[j]));}
}
__global__ void lm_image_station_metric(LinearView l,SolverView s,const double* jac){
    int ci=blockIdx.x;auto c=l.components[ci];
    for(int a=threadIdx.x;a<c.local_count;a+=blockDim.x){int id=s.locals[c.local_begin+a];double norm=0.;
        for(int j=0;j<c.group_count;++j){auto g=l.groups[l.component_groups[c.group_begin+j]];
            for(int k=0;k<g.count;++k){int row=l.members[g.begin+k];if(l.row_local[row]!=id)continue;auto rp=l.row_plan[row];
                const double* v=jac+rp.jac;double x=v[rp.count],y=v[2*rp.count+1];norm+=x*x+y*y;}}
        double* d=const_cast<double*>(s.damping)+l.q+id;*d=fmax(*d,norm);
    }
}
__global__ void lm_image_local_floor(LinearView l,SolverView s){int id=blockIdx.x*blockDim.x+threadIdx.x;if(id<s.nlocal){double* d=const_cast<double*>(s.damping)+l.q+id;*d=fmax(1.,*d);}}
// Finite support and intrinsic chart pieces participate in the SAME damped
// joint system. A bound hit has a nonzero fixed increment; its coupling enters
// both the free local RHS and the retained RHS before Schur elimination.
__global__ void lm_init_bounds(View v,LinearView l,SolverView s,const double* g){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=s.nlocal)return;
    int row=s.representatives[id];double x=v.stations[row],scale=l.scales[id],gradient=g[s.local_gradient[id]];
    double lo=s.lower[id],hi=s.upper[id],piece_lo,piece_hi;bool corner;
    if(station_piece(v,row,x,piece_lo,piece_hi,corner)){lo=fmax(lo,piece_lo);hi=fmin(hi,piece_hi);}
    double dl=(lo-x)/scale,du=(hi-x)/scale;int side=l.local_sides?l.local_sides[id]:0;
    if(side==1)dl=fmax(dl,0.);else if(side==-1)du=fmin(du,0.);else if(side==2)dl=du=0.;
    s.delta_lower[id]=dl;s.delta_upper[id]=du;s.active_delta[id]=0.;
    s.blocked[id]=(dl==du)||(dl==0.&&gradient>0)||(du==0.&&gradient<0);
}
// Exact fixed charts stay in the retained solve. Finite single-column bounds
// join the coupled projection, whose dual solve can release a temporary face.
// No physical range is inferred or widened here.
__global__ void lm_init_retained_bounds(View v,LinearView l,SolverView s){
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=l.q)return;
    double lo=-std::numeric_limits<double>::infinity(),hi=std::numeric_limits<double>::infinity();
    if(l.rig_directions)for(int k=15;k<int(v.nr);++k){
        if(s.rig_bound_columns[k]!=col)continue;int chart=k<24?k-9:(k<38?k-11:k-12);
        double lower=l.rig_lower[k],upper=l.rig_upper[k];if(k==37){lower=fmax(lower,-l.max_readout);upper=fmin(upper,l.max_readout);}
        double d=l.rig_directions[size_t(chart)*l.q+col],a=(lower-v.rig[k])/d,b=(upper-v.rig[k])/d;
        lo=fmax(lo,fmin(a,b));hi=fmin(hi,fmax(a,b));
    }
    s.retained_lower[col]=lo;s.retained_upper[col]=hi;s.retained_delta[col]=0.;s.retained_blocked[col]=lo==hi;
}
__global__ void lm_constrain_retained_matrix(LinearView l,SolverView s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=l.q)return;
    const bool fixed_i=s.retained_blocked[i]!=0;
    for(int k=l.csr_offsets[i];k<l.csr_offsets[i+1];++k){int j=l.csr_columns[k];const bool fixed_j=s.retained_blocked[j]!=0;double a=s.csr[k];
        if(i!=j){if(!fixed_i&&fixed_j)atomicAdd(s.rhs+i,-a*s.retained_delta[j]);else if(fixed_i&&!fixed_j)atomicAdd(s.rhs+j,-a*s.retained_delta[i]);}
        if(fixed_i||fixed_j)s.csr[k]=i==j?1.:0.;
    }
}
__global__ void lm_constrain_retained_rhs(LinearView l,SolverView s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<l.q&&s.retained_blocked[i])s.rhs[i]=s.retained_delta[i];
}
__global__ void lm_fill_local(LinearView l,SolverView s,const double* h,const double* g,const double* jac){
    int ci=blockIdx.x;auto c=l.components[ci];auto w=s.workspace[ci];int m=c.local_count,q=c.q_count;
    if(m==1){if(threadIdx.x)return;int id=s.locals[c.local_begin];bool free=!s.blocked[id];

        double diagonal=free?component_h(h,c,q,q)+s.scalars[0]*s.damping[l.q+id]:1.;s.matrix[w.matrix]=diagonal;
        for(int j=0;j<=q;++j)s.solved[w.solved+j]=free?(j<q?component_h(h,c,q,j):g[c.g+q])/diagonal:(j==q?-s.active_delta[id]:0.);
        return;}
    for(size_t k=threadIdx.x;k<size_t(m)*m;k+=blockDim.x){int i=int(k%m),j=int(k/m);bool ai=!s.blocked[s.locals[c.local_begin+i]],aj=!s.blocked[s.locals[c.local_begin+j]];
        s.matrix[w.matrix+k]=(ai&&aj)?component_h(h,c,q+i,q+j)+(i==j?s.scalars[0]*s.damping[l.q+s.locals[c.local_begin+i]]:0.):(i==j?1.:0.);}
    for(size_t k=threadIdx.x;k<size_t(m)*(q+1);k+=blockDim.x){int i=int(k%m),j=int(k/m),id=s.locals[c.local_begin+i];bool free=!s.blocked[id];
        double value=0.;if(free){value=j<q?component_h(h,c,q+i,j):g[c.g+q+i];
            if(j==q)for(int b=0;b<m;++b){int bid=s.locals[c.local_begin+b];if(s.blocked[bid])value+=component_h(h,c,q+i,q+b)*s.active_delta[bid];}}
        else if(j==q)value=-s.active_delta[id];s.solved[w.solved+k]=value;}
}
__global__ void lm_info(const int* info,int n,int* failed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&info[i]){atomicMin(failed,i);atomicExch(failed+1,info[i]);}
}
__global__ void lm_reduce_start(LinearView l,SolverView s,const double* shared){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k<size_t(l.q)){s.csr[l.csr_offsets[k]]+=s.scalars[0]*s.damping[k];s.rhs[k]=-shared[l.nnz+k];}
}
__global__ void lm_reduce(LinearView l,SolverView s,const double* h,const double* g){
    int ci=blockIdx.x;auto c=l.components[ci];auto w=s.workspace[ci];int q=c.q_count,m=c.local_count;if(!m)return;

    for(size_t k=threadIdx.x;k<size_t(q)*q+q;k+=blockDim.x){
        if(k<size_t(q)*q){int i=int(k/q),j=int(k%q);if(j<i)continue;double ij=0.,ji=0.;
            for(int a=0;a<m;++a){ij+=component_h(h,c,i,q+a)*s.solved[w.solved+size_t(j)*m+a];ji+=component_h(h,c,j,q+a)*s.solved[w.solved+size_t(i)*m+a];}
            atomicAdd(s.csr+l.scatter[c.scatter+upper_index(q,i,j)],-.5*(ij+ji));
        }else{int i=int(k-size_t(q)*q);double value=0.;for(int a=0;a<m;++a)value+=component_h(h,c,i,q+a)*s.solved[w.solved+size_t(q)*m+a];
            atomicAdd(s.rhs+l.columns[c.q_begin+i],value);}
    }
}
__global__ void lm_backsub(LinearView l,SolverView s){
    int ci=blockIdx.x;auto c=l.components[ci];auto w=s.workspace[ci];int m=c.local_count;
    for(int a=threadIdx.x;a<m;a+=blockDim.x){int id=s.locals[c.local_begin+a];double v=s.solved[w.solved+size_t(c.q_count)*m+a];
        for(int j=0;j<c.q_count;++j)v+=s.solved[w.solved+size_t(j)*m+a]*s.step[l.columns[c.q_begin+j]];s.step[l.q+id]=s.blocked[id]?s.active_delta[id]:-v;}
}
__global__ void lm_residual_init(LinearView l,SolverView s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<l.q)s.residual[i]=-s.rhs[i];}
__global__ void lm_residual_spmv(LinearView l,SolverView s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=l.q)return;double value=0.;
    for(int k=l.csr_offsets[i];k<l.csr_offsets[i+1];++k){int j=l.csr_columns[k];value+=s.csr[k]*s.step[j];if(i!=j)atomicAdd(s.residual+j,s.csr[k]*s.step[i]);}atomicAdd(s.residual+i,value);
}
// Project the retained increment onto explicitly declared LINEAR constraints.
// This is a feasible projected LM candidate, not a claim of an exact QP solve.
// The unchanged cuDSS solution residual is checked before projection. Local
// back-substitution and the undamped predicted decrease use the projected step.
// Constraints couple parameter directions; no coordinate is permanently fixed.
__device__ bool lm_halfspace(const double* a,const double* b,int q,double base,double minimum,double* step){
    const double eps=128*std::numeric_limits<double>::epsilon()*(1+fabs(base)+fabs(minimum));
    double value=base,norm=0.;for(int j=0;j<q;++j){double d=(a?a[j]:0.)-(b?b[j]:0.);value+=d*step[j];norm+=d*d;}
    if(value>=minimum+eps*.5)return false;
    if(norm==0.)return false; // A fixed, valid state needs no numerical interior offset.
    const double amount=(minimum+eps-value)/norm;
    bool changed=false;for(int j=0;j<q;++j){double d=(a?a[j]:0.)-(b?b[j]:0.),next=step[j]+amount*d;changed|=next!=step[j];step[j]=next;}
    return changed;
}
__global__ void lm_project_shared_step(View v,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;
    bool changed;
    do{
        changed=false;
        if(l.scene_steps&&(v.options&4u)){
            auto enforce=[&](int high,int low,double minimum=0.){
                // All entries in these relations use additive physical charts.
                const double* a=l.scene_steps+size_t(6+high)*l.q;
                const double* b=l.scene_steps+size_t(6+low)*l.q;
                changed|=lm_halfspace(a,b,l.q,v.layout[high]-v.layout[low],minimum,s.step);
            };
            enforce(0,23); // curling left centre cannot pass left goal line
            for(int row=0;row<4;++row){
                enforce(10+row,23);enforce(8,10+row); // each Lane start lies between goal and right edge
                const double interior=64*std::numeric_limits<double>::epsilon();
                enforce(14+row,1,interior);enforce(8,14+row,interior); // connections pass R and stop before N
            }
        }
        if(l.rig_directions){
            for(int k=15;k<int(v.nr);++k){
                if(k==24||k==25||k==38)continue;
                if(s.rig_bound_columns&&s.rig_bound_columns[k]>=0)continue; // handled by coupled boundary solve
                int chart=k<24?k-9:(k<38?k-11:k-12);
                const double* a=l.rig_directions+size_t(chart)*l.q;
                double lower=l.rig_lower[k],upper=l.rig_upper[k];
                if(k==37){lower=fmax(lower,-l.max_readout);upper=fmin(upper,l.max_readout);}
                if(lower>=upper)continue; // Explicit fixed bounds retain exact domain-rejection semantics.
                if(finite_scalar(lower))changed|=lm_halfspace(a,nullptr,l.q,v.rig[k],lower,s.step);
                if(finite_scalar(upper))changed|=lm_halfspace(nullptr,a,l.q,-v.rig[k],-upper,s.step);
            }
        }
    }while(changed);
}
__global__ void lm_feasible_init(SolverView s){if(threadIdx.x||blockIdx.x)return;s.scalars[3]=1.;for(int k=4;k<8;++k)s.scalars[k]=0.;s.scratch[0]=s.scratch[1]=0.;}
__global__ void lm_feasible(View v,LinearView l,SolverView s){
    int id=blockIdx.x*blockDim.x+threadIdx.x;double alpha=1.,changed=0.,blocked=0.,bad=0.,rr=0.,bb=0.;
    if(id<l.q){rr=s.residual[id]*s.residual[id];bb=s.rhs[id]*s.rhs[id];double d=s.step[id],bounded=fmax(s.retained_lower[id],fmin(s.retained_upper[id],d));bad+=!finite_scalar(d);
        bad+=fabs(bounded-d)>camera_envelope_tolerance()*(1+fabs(d));}
    if(id<s.nlocal){double d=s.step[l.q+id],bounded=fmax(s.delta_lower[id],fmin(s.delta_upper[id],d));
        if(!s.blocked[id]&&bounded!=d){s.blocked[id]=1;s.active_delta[id]=bounded;changed=1.;}
        blocked=s.blocked[id]!=0;bad+=!finite_scalar(d);
    }
    alpha=block_reduce<2>(fmax(0.,alpha));changed=block_reduce<0>(changed);blocked=block_reduce<0>(blocked);bad=block_reduce<0>(bad);rr=block_reduce<0>(rr);bb=block_reduce<0>(bb);
    if(!threadIdx.x){atomic_min_nonnegative(s.scalars+3,alpha);atomicAdd(s.scalars+5,changed);atomicAdd(s.scalars+6,blocked);atomicAdd(s.scalars+7,bad);atomicAdd(s.scratch,rr);atomicAdd(s.scratch+1,bb);}
}
__global__ void lm_feasible_finish(SolverView s){if(threadIdx.x||blockIdx.x)return;double a=s.scalars[3];if(a<1&&a>0)s.scalars[3]=nextafter(a,0.);s.scalars[4]=sqrt(s.scratch[0])/fmax(1.,sqrt(s.scratch[1]));}

__global__ void lm_local_release_start(SolverView s,int* selected){if(!threadIdx.x&&!blockIdx.x){selected[0]=s.nlocal;selected[1]=0;}}
__global__ void lm_local_release(LinearView l,SolverView s,const double* h,const double* g,const double* jac,int* selected){
    if(s.scalars[5]!=0.)return;int ci=blockIdx.x*blockDim.x+threadIdx.x;if(ci>=l.ncomponents)return;
    const auto c=l.components[ci];const int q=c.q_count;
    for(int a=0;a<c.local_count;++a){int id=s.locals[c.local_begin+a];
        if(!s.blocked[id])continue;
        if(s.blocked[id]&&s.delta_lower[id]==s.delta_upper[id])continue;
        double gradient=g[c.g+q+a],scale=fabs(gradient);
        for(int j=0;j<q;++j){double term=component_h(h,c,q+a,j)*s.step[l.columns[c.q_begin+j]];gradient+=term;scale+=fabs(term);}
        for(int b=0;b<c.local_count;++b){double term=component_h(h,c,q+a,q+b)*s.step[l.q+s.locals[c.local_begin+b]];gradient+=term;scale+=fabs(term);}
        double term=s.scalars[0]*s.damping[l.q+id]*s.step[l.q+id];gradient+=term;scale+=fabs(term);
        const double eps=256*std::numeric_limits<double>::epsilon()*(1+scale);
        // A negative inequality multiplier means this boundary is no longer
        // active after the other coupled coordinates move. This applies to
        // finite curve pieces. Fixed supports stay fixed.
        bool release=false;
        if(s.blocked[id]){
            if((s.active_delta[id]==s.delta_lower[id]&&gradient< -eps)||
               (s.active_delta[id]==s.delta_upper[id]&&gradient>eps))release=true;
        }
        if(release){
            // Every multiplier uses the same completed joint step. Release
            // all wrong-sign faces together; one refactorization per point
            // made a single real S01 trial take hundreds of full passes.
            // Do not change scalars[5] until this kernel finishes, otherwise
            // its admission check would make the batch scheduling-dependent.
            atomicMin(selected,id);atomicAdd(selected+1,1);
            s.blocked[id]=0;s.active_delta[id]=0.;
        }
    }
}
__global__ void lm_local_release_finish(SolverView s,const int* selected){
    if(!threadIdx.x&&!blockIdx.x)s.scalars[5]+=selected[1];
}
// Exact working-set comparison for Brent cycle detection. A projected retained
// step can re-activate a just-released local face; such a repeated subproblem
// must reject this LM attempt instead of refactorizing forever. Keep one device
// anchor of the retained and finite local faces.
__global__ void lm_working_set(int q,SolverView s,int* anchor,bool capture,int* different){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(q)+s.nlocal)return;
    int code=0;
    if(k<size_t(q))code=s.retained_blocked[k]?(s.retained_delta[k]==s.retained_lower[k]?1:2):0;
    else{int id=int(k-q);code=s.blocked[id]?(s.active_delta[id]==s.delta_lower[id]?1:2):0;}
    if(capture)anchor[k]=code;else if(anchor[k]!=code)atomicExch(different,1);
}
// The shared physical projector does not solve the constrained joint QP.
// Re-solving that joint system after a local face release can therefore return
// to the same projected face forever. Complete a candidate directly after a
// physical shared projection when scalar local intervals permit it, or after
// a detected working-set cycle otherwise. The completion works
// by minimizing each independent scalar local quadratic CONDITIONAL on the
// already projected shared increment, with the finite material support in
// that very same increment; never widen a physical constraint or delete a
// point. The unchanged nonlinear domain, pixel cost and gain test still decide
// acceptance. This is a feasible candidate, not an exact joint-QP certificate.
__global__ void lm_conditional_start(SolverView s,int* invalid){
    if(threadIdx.x||blockIdx.x)return;*invalid=0;s.scalars[3]=1.;s.scalars[5]=s.scalars[6]=0.;
}
__global__ void lm_conditional_shared_check(LinearView l,SolverView s,int* invalid){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;double d=s.step[j];
    if(!finite_scalar(d)||d<s.retained_lower[j]||d>s.retained_upper[j])atomicAdd(invalid,1);
}
__global__ void lm_conditional_locals(View v,LinearView l,SolverView s,const double* h,const double* g,
                                     const double* output,const double* jac,int* invalid,bool apply=true){
    int ci=blockIdx.x*blockDim.x+threadIdx.x;if(ci>=l.ncomponents)return;auto c=l.components[ci];
    if(!c.local_count)return;
    if(c.local_count!=1){atomicAdd(invalid,1);return;}
    int q=c.q_count,id=s.locals[c.local_begin];double gradient=g[c.g+q];
    for(int j=0;j<q;++j)gradient+=component_h(h,c,q,j)*s.step[l.columns[c.q_begin+j]];
    const double diagonal=component_h(h,c,q,q)+s.scalars[0]*s.damping[l.q+id];
    double lo=s.delta_lower[id],hi=s.delta_upper[id];

    if(!(lo<=hi)||!(diagonal>0.)||!finite_scalar(diagonal)||!finite_scalar(gradient)){
        atomicAdd(invalid,1);return;
    }
    const double d=fmax(lo,fmin(hi,-gradient/diagonal));
    if(!finite_scalar(d)){atomicAdd(invalid,1);return;}
    if(!apply)return;
    s.step[l.q+id]=d;s.blocked[id]=(d==s.delta_lower[id]||d==s.delta_upper[id]);
    s.active_delta[id]=s.blocked[id]?d:0.;
    if(s.blocked[id])atomicAdd(s.scalars+6,1.);
}
__global__ void lm_retract(View accepted,View trial,LinearView l,SolverView s){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double alpha=s.scalars[3];
    if(k<3*accepted.nc){double d=0.;for(int j=0;j<l.q;++j)d+=l.directions[k*l.q+j]*(alpha*s.step[j]);const_cast<double*>(trial.controls)[k]=accepted.controls[k]+d;}
    if(k<accepted.n){int id=l.row_local[k];if(id>=0){double station=accepted.stations[k]+alpha*s.step[l.q+id]*l.scales[id];double lo,hi;bool corner;if(station_piece(accepted,k,accepted.stations[k],lo,hi,corner))station=fmax(lo,fmin(hi,station));const_cast<double*>(trial.stations)[k]=fmax(s.lower[id],fmin(s.upper[id],station));}}
    if(k<size_t(l.q))s.applied[k]=alpha*s.step[k];
}
__global__ void lm_retract_rig(View accepted,View trial,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x||!l.rig_directions)return;
    double delta[27]{};for(int k=0;k<27;++k)for(int j=0;j<l.q;++j)delta[k]+=l.rig_directions[size_t(k)*l.q+j]*(s.scalars[3]*s.step[j]);
    if(s.envelope_delta)for(int k=0;k<27;++k)s.envelope_delta[k]=delta[k];
    if(retract_motion_rig(accepted.rig,delta,const_cast<double*>(trial.rig))!=Status::ok)const_cast<double*>(trial.rig)[0]=std::numeric_limits<double>::quiet_NaN();
    if(accepted.nr==43){for(int i=0;i<4;++i){double d=0.;for(int j=0;j<l.q;++j)d+=l.rig_directions[size_t(27+i)*l.q+j]*(s.scalars[3]*s.step[j]);const_cast<double*>(trial.rig)[39+i]=accepted.rig[39+i]+d;}}
}
__global__ void lm_camera_project_start(SolverView s){if(!threadIdx.x&&!blockIdx.x)s.scalars[3]=1.;}
__global__ void lm_retract_camera_controls(View accepted,View trial,LinearView l,SolverView s){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=3*accepted.nc)return;
    double d=0.;for(int j=0;j<l.q;++j)d+=l.directions[k*l.q+j]*s.step[j];
    const_cast<double*>(trial.controls)[k]=accepted.controls[k]+d;
}
__global__ void lm_camera_envelope_candidates(View v,LinearView l,SolverView s){
    int cell=1+blockIdx.x*blockDim.x+threadIdx.x;if(cell>=int(v.nc)-2)return;
    s.envelope_candidates[cell-1]=camera_cell_constraint(motion_rig(v.rig),MotionSpline{v.controls,v.nc,v.first,v.dt},l.envelope,cell);
}
__global__ void lm_camera_envelope_pick(View v,SolverView s){
    if(threadIdx.x||blockIdx.x)return;auto a=s.envelope_candidates[0];
    for(int i=1;i<int(v.nc)-3;++i){auto b=s.envelope_candidates[i];if(!finite_scalar(b.margin)||b.margin<a.margin)a=b;}
    *s.envelope_choice=a;
}
__global__ void lm_camera_envelope_gradient(View accepted,View trial,LinearView l,SolverView s,bool origin){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    if(s.retained_blocked[j]&&j!=0){s.envelope_gradient[j]=0.;return;}
    LocalJet d[27],p[38];for(int k=0;k<27;++k)d[k]={(l.rig_directions&&!origin)?s.envelope_delta[k]:0.,l.rig_directions?l.rig_directions[size_t(k)*l.q+j]:0.};
    if(retract_motion_rig(accepted.rig,d,p)!=Status::ok){s.envelope_gradient[j]=std::numeric_limits<double>::quiet_NaN();return;}
    const auto a=*s.envelope_choice;LocalJet controls[12];
    for(int k=0;k<12;++k){size_t i=3*size_t(a.cell-1)+k;controls[k]={trial.controls[i],l.directions[i*l.q+j]};}
    auto value=camera_constraint_value(rig_from_values(p),controls,trial.dt,a);
    s.envelope_gradient[j]=s.retained_blocked[j]?0.:value.v;if(j==0)s.scratch[0]=value.a;
}
__global__ void lm_retained_pick(LinearView l,SolverView s,RetainedConstraint* choice){
    if(threadIdx.x||blockIdx.x)return;RetainedConstraint worst{};
    for(int j=0;j<l.q;++j){if(s.retained_lower[j]==s.retained_upper[j])continue;
        const double lower=s.step[j]-s.retained_lower[j],upper=s.retained_upper[j]-s.step[j];
        if(lower<worst.margin)worst={lower,j,1};if(upper<worst.margin)worst={upper,j,-1};}
    *choice=worst;
}
__global__ void lm_retained_gradient(LinearView l,SolverView s,RetainedConstraint constraint){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<l.q)s.envelope_gradient[j]=j==constraint.column?constraint.sign:0.;
}
__global__ void lm_coupled_add_cut(LinearView l,SolverView s,CoupledProjectionView c,int index,int count,double margin,bool origin,bool trace){
    if(threadIdx.x||blockIdx.x)return;double norm=0.,offset=0.;
    if(origin)margin=s.scratch[0];
    for(int j=0;j<l.q;++j){norm+=s.envelope_gradient[j]*s.envelope_response[j];
        offset+=s.envelope_gradient[j]*((origin?0.:s.step[j])-c.base[j]);}
    const double scale=norm>0.&&finite_scalar(norm)?1./sqrt(norm):0.;
    double* a=c.gradients+size_t(index)*l.q;double* w=c.responses+size_t(index)*l.q;
    for(int j=0;j<l.q;++j){a[j]=scale?s.envelope_gradient[j]*scale:0.;w[j]=scale?s.envelope_response[j]*scale:0.;}
    c.target[index]=scale?(offset+4*camera_envelope_tolerance()-margin)*scale:0.;c.dual[index]=0.;
    for(int i=0;i<count;++i){if(i==index)continue;double cross=0.;const double* previous=c.gradients+size_t(i)*l.q;
        for(int j=0;j<l.q;++j)cross+=previous[j]*w[j];
        c.gram[size_t(index)*coupled_projection_limit+i]=c.gram[size_t(i)*coupled_projection_limit+index]=cross;}
    c.gram[size_t(index)*coupled_projection_limit+index]=1.;
    if(trace){double zero_margin=margin;if(!origin)for(int j=0;j<l.q;++j)zero_margin-=s.envelope_gradient[j]*s.step[j];
        printf("LM_CUT index=%d at_origin=%d origin_margin=%.17g metric_norm=%.17g\n",index,int(origin),zero_margin,norm);}
}
// Hildreth's nonnegative dual coordinate solve in the CURRENT Schur metric.
// Retaining multipliers lets a later constraint release an earlier correction;
// sequential feasibility-only projections cannot do that. Numerical work stays
// on the GPU. The outer loop still verifies the original nonlinear domains and
// LM prediction/objective; this is not a generic nonlinear QP solver.
__global__ void lm_coupled_dual(CoupledProjectionView c,int count,bool trace){
    if(threadIdx.x||blockIdx.x)return;
    int sweep=0;for(;sweep<4096;++sweep){double change=0.,scale=1.;
        for(int i=0;i<count;++i){double residual=c.target[i];
            for(int j=0;j<count;++j)residual-=c.gram[size_t(i)*coupled_projection_limit+j]*c.dual[j];
            double next=fmax(0.,c.dual[i]+residual);change=fmax(change,fabs(next-c.dual[i]));
            c.dual[i]=next;scale=fmax(scale,fmax(fabs(next),fabs(c.target[i])));}
        if(change<=256*std::numeric_limits<double>::epsilon()*scale)break;
    }
    if(trace){double kkt=0.;for(int i=0;i<count;++i){double residual=c.target[i];
        for(int j=0;j<count;++j)residual-=c.gram[size_t(i)*coupled_projection_limit+j]*c.dual[j];
        kkt=fmax(kkt,c.dual[i]>0.?fabs(residual):fmax(0.,residual));}
        printf("LM_DUAL count=%d sweeps=%d kkt=%.17g\n",count,sweep,kkt);}
}
__global__ void lm_coupled_apply(LinearView l,SolverView s,CoupledProjectionView c,int count){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;double value=c.base[j];
    for(int i=0;i<count;++i)value+=c.dual[i]*c.responses[size_t(i)*l.q+j];
    s.step[j]=value;
}
__global__ void lm_camera_scale_step(LinearView l,SolverView s,double scale){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<l.q)s.step[j]*=scale;}
__global__ void lm_camera_envelope_domain(SolverView s){
    if(threadIdx.x||blockIdx.x)return;double margin=s.envelope_choice->margin;
    if(!finite_scalar(margin)||margin < -camera_envelope_tolerance()){s.scalars[8]+=1;s.scalars[9]=0.;}
}
__global__ void lm_rig_domain(View trial,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;
    bool bad=!valid_installation(trial,l.mount_side);
    if(l.rig_directions){bad|=std::abs(trial.rig[37])>l.max_readout+camera_envelope_tolerance();
    for(size_t k=0;k<trial.nr;++k){const double eps=camera_envelope_tolerance()*(1+std::abs(trial.rig[k]));
        bad|=!finite_scalar(trial.rig[k])||trial.rig[k]<l.rig_lower[k]-eps||trial.rig[k]>l.rig_upper[k]+eps;}}
    if(bad){s.scalars[8]+=1;s.scalars[9]=0.;}
}
__global__ void lm_retract_scene(View accepted,View trial,LinearView l,SolverView s){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=accepted.nscene)return;
    double value=scene_chart_retract(accepted.scene_values,l.scene_chart,k,l.scene_steps,l.q,s.step,s.scalars[3]);const_cast<double*>(trial.scene_values)[k]=value;
    if(k<6)const_cast<double*>(trial.board)[k]=value;else if(k<6+trial.nl)const_cast<double*>(trial.layout)[k-6]=value;
}
__global__ void lm_build_course(View trial){
    if(threadIdx.x||blockIdx.x)return;const double* b=trial.board;
    *const_cast<BoardCourse*>(trial.course)=BoardCourse({b[0],b[1],b[2],b[3],b[4],b[5]});
}
// Critical stations of the exact finite print chart, not extra observations.
template<class Visit> RS_HD inline void print_domain_stations(const ScenePath& path,const double* data,Visit visit){
    if(path.family==2){for(int j=0;j<=4;++j)visit(j*.25);}
    else if(path.family==3){visit(0.);visit(1.);}
    else{
        for(int j=0;j<path.count;++j)visit(double(j)/(path.count-1));
        if(path.family==1){const double* seed=data+path.vertices;const double* b=data+path.bounds;
            for(int j=0;j<path.count-1;++j)for(int axis=0;axis<2;++axis){double a=seed[2*j+axis],d=seed[2*j+2+axis]-a;
                if(d!=0.)for(int knot=1;knot<4;++knot){double t=(b[axis]+knot*.25*(b[axis+2]-b[axis])-a)/d;
                    if(t>0&&t<1)visit((j+t)/(path.count-1));}}
        }
    }
}
__global__ void lm_print_candidates(View trial,SolverView s){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=s.print_count)return;
    const auto probe=s.print_probes[i];PrintConstraint result{};Vec2Value<LocalJet> chart{},material{};
    const auto path=trial.scene_paths[probe.path];
    auto status=scene_print_chart(path,trial.scene_data,trial.scene_values,nullptr,0,0,LocalJet(probe.station),chart,&material);
    if(status!=Status::ok)result={-INFINITY,probe.station,probe.path,0};
    else{double z=chart.y.a,upper=trial.board[5]-z;result=z<=upper?PrintConstraint{z,probe.station,probe.path,0}:PrintConstraint{upper,probe.station,probe.path,1};
        if(path.code==1){double width=exp(trial.scene_values[path.size]),height=exp(trial.scene_values[path.size+1]);
            double margins[4]={width*material.x.a,width*(1.-material.x.a),height*material.y.a,height*(1.-material.y.a)};
            for(int side=0;side<4;++side)if(margins[side]<result.margin)result={margins[side],probe.station,probe.path,side+2};}}
    s.print_candidates[i]=result;
}
__global__ void lm_print_pick(View trial,SolverView s){
    if(threadIdx.x||blockIdx.x)return;PrintConstraint result{};
    for(size_t i=0;i<s.print_count;++i)if(s.print_candidates[i].margin<result.margin)result=s.print_candidates[i];
    *s.print_choice=result;
}
__global__ void lm_print_chart_directions(View trial,LinearView l,SolverView s){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    // Re-evaluate the chart pullback at the trial using the original tangent
    // directions. scene_directions already includes the accepted-state chart
    // derivative; feeding it here applies positive/ordered scaling twice.
    scene_chart_direction(trial.scene_values,trial.nscene,l.scene_chart,l.scene_steps,l.q,j,s.print_directions);
}
__global__ void lm_print_gradient(View trial,LinearView l,SolverView s){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    if(s.retained_blocked[j]&&j!=0){s.envelope_gradient[j]=0.;return;}
    auto a=*s.print_choice;Vec2Value<LocalJet> chart{},material{};const auto path=trial.scene_paths[a.path];
    auto status=scene_print_chart(path,trial.scene_data,trial.scene_values,s.print_directions,l.q,j,LocalJet(a.station),chart,&material);
    LocalJet margin(NAN,NAN);
    if(status==Status::ok){
        if(a.side<2)margin=a.side?LocalJet(trial.board[5],s.print_directions[5*size_t(l.q)+j])-chart.y:chart.y;
        else{int axis=(a.side-2)/2,index=path.size+axis;
            LocalJet size=exp(LocalJet(trial.scene_values[index],s.print_directions[size_t(index)*l.q+j]));
            LocalJet coordinate=axis?material.y:material.x;
            margin=size*((a.side%2)?1.-coordinate:coordinate);}}
    s.envelope_gradient[j]=s.retained_blocked[j]?0.:margin.v;
    if(j==0)s.scratch[0]=margin.a;
}
// The same installation domain as valid_installation, with joint camera /
// board pullbacks. A declared +/-Y side already implies rink exterior.
__device__ inline LocalJet mount_constraint_value(View v,LinearView l,SolverView s,int kind,int column){
    auto rig=[&](int i){return LocalJet(v.rig[i],column>=0&&l.rig_directions?l.rig_directions[size_t(i-9)*l.q+column]:0.);};
    auto board=[&](int i){return LocalJet(v.board[i],column>=0&&l.scene_directions?s.print_directions[size_t(i)*l.q+column]:0.);};
    if(kind==0)return rig(17);
    if(kind==1){LocalJet value(-1e-24);for(int k=18;k<21;++k){auto x=rig(k);value+=x*x;}return value;}
    if(kind==3){if(l.mount_side==2)return -rig(16)-board(1);if(l.mount_side==3)return rig(16)-board(1);return l.mount_side==4?-rig(15):rig(15);}
    auto px=rig(15),py=rig(16);auto x=px.a<0?-px:px,y=py.a<0?-py:py;
    auto rx=board(2),ry=board(3),power=board(4);
    auto u=(x-board(0)+rx)/rx,w=(y-board(1)+ry)/ry;
    return (u.a>0?exp(power*log(u)):LocalJet(0.))+(w.a>0?exp(power*log(w)):LocalJet(0.))-1.;
}
__global__ void lm_mount_pick(View trial,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;MountConstraint result{};
    for(int kind=0;kind<4;++kind){if(kind==2&&(l.mount_side==2||l.mount_side==3))continue;if(kind==3&&l.mount_side==1)continue;
        double margin=mount_constraint_value(trial,l,s,kind,-1).a;if(!finite_scalar(margin)||margin<result.margin)result={margin,kind};}
    *s.mount_choice=result;
}
__global__ void lm_mount_gradient(View trial,LinearView l,SolverView s){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    auto value=mount_constraint_value(trial,l,s,s.mount_choice->kind,j);
    s.envelope_gradient[j]=s.retained_blocked[j]?0.:value.v;if(j==0)s.scratch[0]=value.a;
}
__global__ void lm_scene_domain(View trial,LinearView l,SolverView s){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;bool bad=false;
    if(i<trial.nscene){double value=trial.scene_values[i];bad|=!finite_scalar(value)||value<l.scene_lower[i]||value>l.scene_upper[i];}
    if(i==0){const double* b=trial.board;bad|=!BoardShape{b[0],b[1],b[2],b[3],b[4],b[5]}.valid();}
    if(i<trial.npaths){auto path=trial.scene_paths[i];
        if(path.type==1){double p[8];int kind;bad|=shared_entity_definition(trial.board,6,trial.layout,trial.nl,path.code,kind,p,(trial.nl==35&&(path.code==410||path.code==411))?8:6,trial.options)!=Status::ok;}
        if(path.type==2){
            auto check=[&](double station){Vec2Value<LocalJet> chart{},material{};Vec3 xyz{},tangent{};
                auto st=scene_print_chart(path,trial.scene_data,trial.scene_values,nullptr,0,0,LocalJet(station),chart,&material);
                if(st!=Status::ok)return false;
                if(path.code==1){double width=exp(trial.scene_values[path.size]),height=exp(trial.scene_values[path.size+1]),eps=camera_envelope_tolerance();
                    if(width*material.x.a < -eps||width*(1.-material.x.a) < -eps||height*material.y.a < -eps||height*(1.-material.y.a) < -eps)return false;}
                return trial.course->point(chart.x.a,chart.y.a,xyz,tangent)==Status::ok;};
            print_domain_stations(path,trial.scene_data,[&](double station){bad|=!check(station);});
        }
    }
    if(bad){atomicAdd(s.scalars+8,1.);atomic_min_nonnegative(s.scalars+9,0.);}
}
// A curve station that leaves declared temporal support can be initialized
// at the observed-row pose. This is only a recovery proposal: the unchanged
// implicit forward and original objective decide acceptance, with no UV clamp.
__global__ void lm_restore_station_domain(View trial,LinearView l,SolverView s){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=s.nlocal||s.lower[id]==s.upper[id])return;
    const int row=s.representatives[id];
    if(trial.profile_enabled&&trial.profile_ids&&trial.profile_ids[row]>=0)return;
    const auto rig=motion_rig(trial.rig,trial.nr==43?trial.rig+38:nullptr);const MotionSpline spline{trial.controls,trial.nc,trial.first,trial.dt};
    Vec3 xyz{},tangent{};MotionProjection strict{};
    double station=trial.stations[row];auto status=geometry_at(trial,row,station,xyz,tangent);
    if(status!=Status::ok||project_motion(rig,spline,trial.height,xyz,trial.times[row],strict,trial.projection_time_half)!=Status::no_intersection)return;
    const double observed_u=trial.observed[2*row],observed_v=trial.observed[2*row+1];
    if(observed_v<0||observed_v>trial.height-1)return;
    MotionCameraState seed{};
    if(evaluate_motion_camera(rig,spline,trial.times[row]+rig.tau*(observed_v/(trial.height-1)-.5),seed)!=Status::ok)return;
    for(int iteration=0;iteration<40;++iteration){
        Vec2 uv{};double jac[6];
        if(geometry_at(trial,row,station,xyz,tangent)!=Status::ok||project(seed.camera,xyz,uv,jac)!=Status::ok)return;
        const double u=uv.x-observed_u,v=uv.y-observed_v,loss=u*u+v*v;
        const double du=jac[0]*tangent.x+jac[1]*tangent.y+jac[2]*tangent.z,dv=jac[3]*tangent.x+jac[4]*tangent.y+jac[5]*tangent.z;
        const double norm=du*du+dv*dv;if(!(norm>0)||!finite_scalar(norm))break;
        double step=-(u*du+v*dv)/norm;
        step=fmax(s.lower[id]-station,fmin(s.upper[id]-station,step));
        bool improved=false;
        for(int backtrack=0;backtrack<40;++backtrack,step*=.5){
            double next=station+step;if(next==station)break;
            Vec3 x{},t{};Vec2 q{};
            if(geometry_at(trial,row,next,x,t)==Status::ok&&project(seed.camera,x,q)==Status::ok){
                double a=q.x-observed_u,b=q.y-observed_v;
                if(a*a+b*b<loss){station=next;improved=true;break;}
            }
        }
        if(!improved)break;
    }
    if(geometry_at(trial,row,station,xyz,tangent)==Status::ok&&project_motion(rig,spline,trial.height,xyz,trial.times[row],strict,trial.projection_time_half)==Status::ok)
        const_cast<double*>(trial.stations)[row]=station;
}
__global__ void lm_share_restored_stations(View trial,LinearView l,SolverView s){
    size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=trial.n)return;
    int id=l.row_local[row];if(id>=0&&row!=size_t(s.representatives[id]))const_cast<double*>(trial.stations)[row]=trial.stations[s.representatives[id]];
}
// Restore an ORIGINAL finite source-query alpha domain. The image raster is
// not a geometric constraint; plain pixel factors have no row-edge restoration.
__global__ void lm_restore_profile_domain(View accepted,View trial,LinearView l,SolverView s,const double* jac,const int* statuses){
    if(threadIdx.x||blockIdx.x)return;
    for(int k=0;k<l.q+s.nlocal;++k)s.step[k]=s.applied[k];
    s.scratch[0]=0.;
    for(size_t row=0;row<trial.n;++row){
        if(statuses[row]!=int(Status::no_intersection))continue;
        if(trial.profile_enabled&&trial.profile_ids&&trial.profile_ids[row]>=0){
            // Diagnostic continuation locates which ORIGINAL alpha bound was
            // crossed. It is never evaluated as an accepted likelihood factor.
            const double* p=trial.profiles+17*trial.profile_ids[row];
            double station=0.,alpha=0.;Vec3 x{},t{};MotionProjection projection{};
            if(resolve_profile(trial,row,station,alpha,x,t,projection,true)!=Status::ok)continue;
            const double pad=std::max(1e-7,128*std::numeric_limits<double>::epsilon()*(1+fabs(alpha)));
            if(p[13]-p[12]<=2*pad||!(alpha<p[12]||alpha>p[13]))continue;
            double old_station=0.,old_alpha=0.;MotionProjection old_projection{};
            if(resolve_profile(accepted,row,old_station,old_alpha,x,t,old_projection)!=Status::ok)continue;
            Vec2 query{},qt{};if(profile_value(p,old_alpha,query,qt)!=Status::ok)continue;
            const double qt2=qt.x*qt.x+qt.y*qt.y;if(!(qt2>0))continue;
            auto rp=l.row_plan[row];const double* j=jac+rp.jac;double predicted=alpha,norm=0.;
            auto derivative=[&](int k){return (qt.x*j[k]+qt.y*j[rp.count+1+k])/qt2;};
            for(int k=0;k<rp.count;++k){int q=l.columns[rp.begin+k];double d=derivative(k);predicted+=d*(s.step[q]-s.applied[q]);norm+=d*d/s.damping[q];}
            if((predicted>=p[12]&&predicted<=p[13])||!(norm>0)||!finite_scalar(norm))continue;
            double target=predicted<p[12]?p[12]+pad:p[13]-pad,amount=(target-predicted)/norm;bool moved=false;
            for(int k=0;k<rp.count;++k){int q=l.columns[rp.begin+k];double next=s.step[q]+amount*derivative(k)/s.damping[q];moved|=next!=s.step[q];s.step[q]=next;}
            if(moved)s.scratch[0]+=1.;continue;
        }

    }
}
__global__ void lm_actual_locals(View accepted,View trial,LinearView l,SolverView s){int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=s.nlocal)return;int row=s.representatives[id];s.applied[l.q+id]=(trial.stations[row]-accepted.stations[row])/l.scales[id];}
__device__ double lm_component_step(LinearView l,SolverView s,ComponentPlan c,int j){return j<c.q_count?s.applied[l.columns[c.q_begin+j]]:s.applied[l.q+s.locals[c.local_begin+j-c.q_count]];}
__global__ void lm_prediction(LinearView l,SolverView s,const double* h,const double* g){
    int ci=blockIdx.x;auto c=l.components[ci];double value=0.;size_t square=size_t(c.dim)*c.dim;
    for(size_t k=threadIdx.x;k<square+c.dim;k+=blockDim.x){if(k<square){int i=int(k/c.dim),j=int(k%c.dim);value-=.5*lm_component_step(l,s,c,i)*component_h(h,c,i,j)*lm_component_step(l,s,c,j);}else{int i=int(k-square);value-=g[c.g+i]*lm_component_step(l,s,c,i);}}
    value=block_reduce<0>(value);if(!threadIdx.x)s.parts[ci]=value;
}
// Diagnostic readback of the immediately preceding linearization/candidate.
// No relinearization or accepted-state mutation is performed.
__global__ void lm_probe_changes(View v,LinearView l,SolverView s,const double* jac,const double* before,const double* after,double* out){
    size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=v.n)return;auto rp=l.row_plan[row];
    for(int d=0;d<2;++d){const double* j=jac+rp.jac+d*(rp.count+1);double predicted=0.;
        for(int k=0;k<rp.count;++k)predicted+=j[k]*s.applied[l.columns[rp.begin+k]];
        int local=l.row_local[row];if(local>=0)predicted+=j[rp.count]*s.applied[l.q+local];
        out[6*row+d]=before[11*row+3+d];out[6*row+2+d]=after[11*row+3+d];out[6*row+4+d]=predicted;
    }
}
__global__ void lm_trial_whiten(LinearView l,const double* output,double* white){int gi=blockIdx.x;auto g=l.groups[gi];int d=l.components[g.component].dim;for(int k=threadIdx.x;k<g.rank;k+=blockDim.x)white[g.white+size_t(k)*(d+1)]=whiten_element(l,gi,k,0,output,nullptr);}
__global__ void lm_finish_init(View trial,SolverView s){if(threadIdx.x||blockIdx.x)return;s.scalars[8]=0.;s.scalars[9]=double(trial.n);s.scalars[10]=s.scalars[11]=s.scalars[14]=s.scalars[23]=0.;}
__global__ void lm_finish(View trial,LinearView l,SolverView s,const int* statuses,const double* stats){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double bad=0.,first=double(trial.n),cost=0.,pred=0.,norm=0.,maximum=0.;
    if(i<trial.n&&statuses[i]){bad+=1;first=double(i);}
    if(i<size_t(s.nlocal)){double x=trial.stations[s.representatives[i]];if(!finite_scalar(x)||x<s.lower[i]||x>s.upper[i]){bad+=1;first=fmin(first,double(s.representatives[i]));}}
    if(i<size_t(l.ng))cost=stats[3*i+2];if(i<size_t(l.ncomponents))pred=s.parts[i];
    if(i<size_t(l.q)+s.nlocal){double d=s.applied[i];norm=d*d;maximum=fabs(d);}
    bad=block_reduce<0>(bad);first=block_reduce<2>(first);cost=block_reduce<0>(cost);pred=block_reduce<0>(pred);norm=block_reduce<0>(norm);maximum=block_reduce<1>(maximum);
    if(!threadIdx.x){atomicAdd(s.scalars+8,bad);atomic_min_nonnegative(s.scalars+9,first);atomicAdd(s.scalars+10,cost);atomicAdd(s.scalars+11,pred);atomicAdd(s.scalars+14,norm);atomic_max_nonnegative(s.scalars+23,maximum);}
}
__global__ void lm_finish_values(View trial,SolverView s){if(threadIdx.x||blockIdx.x)return;
    if(s.scalars[9]>=double(trial.n))s.scalars[9]=-1.;s.scalars[12]=s.scalars[1]-s.scalars[10];s.scalars[13]=s.scalars[11]>0?s.scalars[12]/s.scalars[11]:0.;s.scalars[14]=sqrt(s.scalars[14]);}
// Original robust projected gradient, including all remaining feasible directions.
__global__ void lm_stationarity_start(LinearView l,SolverView s,const double* shared){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<l.q)s.residual[i]=shared[l.nnz+i];
}

__global__ void lm_stationarity(View v,LinearView l,SolverView s,const double* g,const double* shared,const double* stats){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double maximum=0.,cost=0.,bad=0.;
    if(i<size_t(l.q)){double dg=s.residual[i];maximum=fabs(dg);bad+=!finite_scalar(dg);}
    if(i<size_t(s.nlocal)){double x=v.stations[s.representatives[i]],dg=g[s.local_gradient[i]],eps=64*std::numeric_limits<double>::epsilon();bool fixed=s.lower[i]==s.upper[i]||(x<=s.lower[i]+eps&&dg>0)||(x>=s.upper[i]-eps&&dg<0);if(!fixed)maximum=fmax(maximum,fabs(dg));bad+=!finite_scalar(dg);}
    if(i<size_t(l.ng)){cost=stats[3*i+2];bad+=!finite_scalar(cost);}
    maximum=block_reduce<1>(maximum);cost=block_reduce<0>(cost);bad=block_reduce<0>(bad);
    if(!threadIdx.x){atomicAdd(s.scalars,cost);atomic_max_nonnegative(s.scalars+1,maximum);atomicAdd(s.scalars+2,bad);}
}
struct DenseBatch {int dimension;size_t factor_begin,factor_count,solve_begin,solve_count;};
#include "gpu_station_refinement.cuh"
struct Solver {
    StationRefinement station_refinement;
    rs_lm_numeric_progress numeric_progress=nullptr;void* numeric_context=nullptr;
    size_t envelope_projections=0,envelope_backtracks=0,envelope_downloads=0,print_downloads=0,mount_downloads=0,retained_downloads=0;
    DeviceBuffer<RetainedConstraint> retained_choice;
    DeviceBuffer<CameraConstraint> envelope_candidates,envelope_choice;
    DeviceBuffer<double> envelope_gradient,envelope_delta,envelope_response,print_directions,coupled_storage;
    CoupledProjectionView coupled{};
    DeviceBuffer<PrintConstraint> print_candidates,print_choice;DeviceBuffer<PrintProbe> print_probes;DeviceBuffer<MountConstraint> mount_choice;
    DeviceBuffer<double> lower,upper,matrix,solved,rhs,csr,step,applied,scalars,parts,trial_input,trial_output,trial_white,trial_stats,residual,scratch,damping,image_metric,delta_lower,delta_upper,active_delta,retained_lower,retained_upper,retained_delta;
    DeviceBuffer<int> locals,representatives,blocked,factor_info,solve_info,check_info,trial_status,rig_bound_columns,retained_blocked;
    DeviceBuffer<int> working_set_anchor;
    DeviceBuffer<BoardCourse> trial_course;
    DeviceBuffer<size_t> local_gradient;DeviceBuffer<LocalWorkspace> workspace;
    DeviceBuffer<double*> factor_ptrs,solve_a_ptrs,solve_b_ptrs;std::vector<DenseBatch> batches;
    SolverView view{};cusolverDnHandle_t dense=nullptr;cudssHandle_t sparse=nullptr;cudssConfig_t config=nullptr;cudssData_t data=nullptr;
    cudssMatrix_t A=nullptr,X=nullptr,B=nullptr,envelope_X=nullptr,envelope_B=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
    double lambda=0.,nu=2.,initial_relative=.001,eta=.0001,linear_tolerance=1e-8;
    double device_budget=0.,host_budget=0.;int m=0,analyses=0,trials=0,accepted=0,numerical_rejections=0;
    int64_t estimates[16]{},factor_nnz=0;bool trial_available=false,analysis_ready=false,jacobi_damping=false,image_station_damping=false,image_global_damping=false;
    ~Solver(){if(A)cudssMatrixDestroy(A);if(X)cudssMatrixDestroy(X);if(B)cudssMatrixDestroy(B);if(envelope_X)cudssMatrixDestroy(envelope_X);if(envelope_B)cudssMatrixDestroy(envelope_B);if(data)cudssDataDestroy(sparse,data);if(config)cudssConfigDestroy(config);if(sparse)cudssDestroy(sparse);if(dense)cusolverDnDestroy(dense);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
    size_t device_bytes()const{return retained_choice.bytes()+coupled_storage.bytes()+mount_choice.bytes()+print_probes.bytes()+print_candidates.bytes()+print_choice.bytes()+print_directions.bytes()+working_set_anchor.bytes()+station_refinement.bytes()+envelope_candidates.bytes()+envelope_choice.bytes()+envelope_gradient.bytes()+envelope_delta.bytes()+envelope_response.bytes()+retained_lower.bytes()+retained_upper.bytes()+retained_delta.bytes()+rig_bound_columns.bytes()+retained_blocked.bytes()+delta_lower.bytes()+delta_upper.bytes()+active_delta.bytes()+image_metric.bytes()+damping.bytes()+trial_course.bytes()+lower.bytes()+upper.bytes()+matrix.bytes()+solved.bytes()+rhs.bytes()+csr.bytes()+step.bytes()+applied.bytes()+scalars.bytes()+parts.bytes()+trial_input.bytes()+trial_output.bytes()+trial_white.bytes()+trial_stats.bytes()+residual.bytes()+scratch.bytes()+locals.bytes()+representatives.bytes()+blocked.bytes()+factor_info.bytes()+solve_info.bytes()+check_info.bytes()+trial_status.bytes()+local_gradient.bytes()+workspace.bytes()+factor_ptrs.bytes()+solve_a_ptrs.bytes()+solve_b_ptrs.bytes();}
    void prepare(const View& v,Linearization& l,size_t input_count,const double* lo,const double* hi,size_t nlocal,const double* options,size_t noptions,cudaStream_t stream){
        if((noptions!=6&&noptions!=7)||(noptions==7&&options[6]!=0&&options[6]!=1&&options[6]!=2&&options[6]!=3)||nlocal!=l.scales.size()||(nlocal&&(!lo||!hi))||!finite_array(options,noptions)||options[0]<=0||options[1]<0||options[2]<=0||options[2]>=1||options[3]<=0||options[4]<=0||options[5]<=0)throw std::invalid_argument("invalid G3 bounds/options");
        jacobi_damping=noptions==7&&options[6]!=0;image_station_damping=noptions==7&&options[6]>=2;image_global_damping=noptions==7&&options[6]==3;initial_relative=options[0];lambda=options[1];eta=options[2];linear_tolerance=options[3];device_budget=options[4];host_budget=options[5];
        std::vector<int> reps(nlocal,-1);for(size_t row=0;row<v.n;++row){int id=l.row_local[row];if(id<0)continue;bool periodic=v.paths&&v.paths[7*v.codes[row]]==7;
            if(std::isnan(lo[id])||std::isnan(hi[id])||(!periodic&&(lo[id]<0||hi[id]>1))||lo[id]>hi[id]||v.stations[row]<lo[id]||v.stations[row]>hi[id])throw std::invalid_argument("original station outside declared finite support at row "+std::to_string(row));reps[id]=int(row);}
        for(int r:reps)if(r<0)throw std::invalid_argument("unreferenced station");
        int q=l.host.q,cc=l.host.ncomponents;std::vector<LocalWorkspace> work;std::vector<size_t> gradient(nlocal);std::map<int,std::vector<int>> bucket;
        size_t nm=0,nsolved=0,nfactor=0,nsolve=0;
        for(int ci=0;ci<cc;++ci){auto c=l.components[ci];work.push_back({nm,nsolved});nm+=checked_product(size_t(c.local_count),size_t(c.local_count));nsolved+=checked_product(size_t(c.local_count),size_t(c.q_count+1));m=std::max(m,c.local_count);
            for(int j=0;j<c.local_count;++j)gradient[l.component_locals[c.local_begin+j]]=c.g+c.q_count+j;
            if(c.local_count>1){bucket[c.local_count].push_back(ci);++nfactor;nsolve+=c.q_count+1;}}
        Linearization::bounded(nfactor);Linearization::bounded(nsolve);
        std::vector<PrintProbe> probes;
        if(l.host.scene_directions)for(size_t i=0;i<v.npaths;++i)if(v.scene_paths[i].type==2)
            print_domain_stations(v.scene_paths[i],v.scene_data,[&](double station){probes.push_back({station,int(i)});});
        double planned=8.*(double(nm)+nsolved+2.*q+l.host.nnz+3.*(q+nlocal)+4.*q+26+cc+input_count+11.*v.n+l.white_count+3.*l.host.ng+2.*nlocal+4.*nlocal)+
            4.*(3.*nlocal+nfactor+3+v.n+q+v.nr)+sizeof(LocalWorkspace)*double(cc)+8.*(nfactor+2.*nsolve);
        if(l.host.envelope.enabled)planned+=sizeof(CameraConstraint)*double(v.nc-2);
        const size_t coupled_count=size_t(q)+2*checked_product(size_t(q),size_t(coupled_projection_limit))+size_t(coupled_projection_limit)*(coupled_projection_limit+2);
        if(l.host.envelope.enabled||l.host.rig_directions||l.host.scene_directions||l.host.mount_side)planned+=8.*(2*q+27+double(coupled_count));
        if(l.host.rig_directions)planned+=sizeof(RetainedConstraint);
        if(l.host.scene_directions)planned+=sizeof(PrintConstraint)*double(probes.size()+1)+sizeof(PrintProbe)*double(probes.size())+8.*checked_product(v.nscene,size_t(q));
        if(l.host.mount_side)planned+=sizeof(MountConstraint);
        planned+=4.*(q+nlocal);
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));if(planned>device_budget||planned>free)throw std::runtime_error("ragged local/sparse LM workspace needs "+std::to_string(planned)+" bytes; exceeds current device budget/available memory");
        lower.upload(std::vector<double>(lo,lo+nlocal),stream);upper.upload(std::vector<double>(hi,hi+nlocal),stream);locals.upload(l.component_locals,stream);representatives.upload(reps,stream);local_gradient.upload(gradient,stream);workspace.upload(work,stream);blocked.allocate(nlocal);
        working_set_anchor.allocate(size_t(q)+nlocal);
        std::vector<int> rig_single(v.nr,-1);if(!l.rig_directions.empty())for(int k=15;k<int(v.nr);++k){
            if(k==24||k==25||k==38)continue;int chart=k<24?k-9:(k<38?k-11:k-12),found=-1;
            for(int col=0;col<q;++col)if(l.rig_directions[size_t(chart)*q+col]!=0.){if(found>=0){found=-2;break;}found=col;}
            if(found>=0)rig_single[k]=found;
        }
        rig_bound_columns.upload(rig_single,stream);retained_blocked.allocate(q);retained_lower.allocate(q);retained_upper.allocate(q);retained_delta.allocate(q);
        delta_lower.allocate(nlocal);delta_upper.allocate(nlocal);active_delta.allocate(nlocal);damping.allocate(q+nlocal);image_metric.allocate(q);matrix.allocate(nm);solved.allocate(nsolved);rhs.allocate(q);csr.allocate(l.host.nnz);step.allocate(q+nlocal);applied.allocate(q+nlocal);scalars.allocate(24);parts.allocate(cc);residual.allocate(q);scratch.allocate(2);
        if(v.scene_paths)trial_course.allocate(1);
        trial_input.allocate(input_count);trial_output.allocate(v.n*11);trial_status.allocate(v.n);trial_white.allocate(l.white_count);trial_stats.allocate(3*size_t(l.host.ng));factor_info.allocate(nfactor);solve_info.allocate(1);check_info.allocate(2);
        std::vector<double*> ap,aa,bb;ap.reserve(nfactor);aa.reserve(nsolve);bb.reserve(nsolve);
        for(auto& item:bucket){DenseBatch batch{item.first,ap.size(),item.second.size(),aa.size(),0};
            for(int ci:item.second){auto c=l.components[ci];auto w=work[ci];ap.push_back(matrix.data+w.matrix);for(int j=0;j<=c.q_count;++j){aa.push_back(ap.back());bb.push_back(solved.data+w.solved+size_t(j)*c.local_count);}}
            batch.solve_count=aa.size()-batch.solve_begin;batches.push_back(batch);}
        factor_ptrs.upload(ap,stream);solve_a_ptrs.upload(aa,stream);solve_b_ptrs.upload(bb,stream);
        view={locals.data,representatives.data,local_gradient.data,workspace.data,lower.data,upper.data,blocked.data,matrix.data,solved.data,rhs.data,csr.data,step.data,applied.data,scalars.data,parts.data,residual.data,scratch.data,damping.data,image_metric.data,int(nlocal),delta_lower.data,delta_upper.data,active_delta.data,rig_bound_columns.data,retained_blocked.data,retained_lower.data,retained_upper.data,retained_delta.data};
        if(l.host.envelope.enabled){envelope_candidates.allocate(v.nc-3);envelope_choice.allocate(1);
            view.envelope_candidates=envelope_candidates.data;view.envelope_choice=envelope_choice.data;}
        if(l.host.rig_directions)retained_choice.allocate(1);
        if(l.host.envelope.enabled||l.host.rig_directions||l.host.scene_directions||l.host.mount_side){envelope_gradient.allocate(q);envelope_delta.allocate(27);envelope_response.allocate(q);
            view.envelope_gradient=envelope_gradient.data;view.envelope_delta=envelope_delta.data;view.envelope_response=envelope_response.data;
            coupled_storage.allocate(coupled_count);coupled.base=coupled_storage.data;coupled.gradients=coupled.base+q;
            coupled.responses=coupled.gradients+size_t(q)*coupled_projection_limit;coupled.gram=coupled.responses+size_t(q)*coupled_projection_limit;
            coupled.target=coupled.gram+coupled_projection_limit*coupled_projection_limit;coupled.dual=coupled.target+coupled_projection_limit;}
        if(l.host.scene_directions){print_candidates.allocate(probes.size());print_choice.allocate(1);print_probes.upload(probes,stream);print_directions.allocate(checked_product(v.nscene,size_t(q)));
            view.print_probes=print_probes.data;view.print_count=probes.size();
            view.print_candidates=print_candidates.data;view.print_choice=print_choice.data;view.print_directions=print_directions.data;}
        if(l.host.mount_side){mount_choice.allocate(1);view.mount_choice=mount_choice.data;}
        if(nfactor){dense_check(cusolverDnCreate(&dense));dense_check(cusolverDnSetStream(dense,stream));}
        sparse_check(cudssCreate(&sparse));sparse_check(cudssSetStream(sparse,stream));sparse_check(cudssConfigCreate(&config));sparse_check(cudssDataCreate(sparse,&data));
        int disabled=0;sparse_check(cudssConfigSet(config,CUDSS_CONFIG_HYBRID_EXECUTE_MODE,&disabled,sizeof(disabled)));sparse_check(cudssConfigSet(config,CUDSS_CONFIG_HYBRID_MEMORY_MODE,&disabled,sizeof(disabled)));
        sparse_check(cudssMatrixCreateDn(&X,q,1,q,step.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));sparse_check(cudssMatrixCreateDn(&B,q,1,q,rhs.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
        if(l.host.envelope.enabled||l.host.rig_directions||l.host.scene_directions||l.host.mount_side){
            sparse_check(cudssMatrixCreateDn(&envelope_X,q,1,q,envelope_response.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
            sparse_check(cudssMatrixCreateDn(&envelope_B,q,1,q,envelope_gradient.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));}
        sparse_check(cudssMatrixCreateCsr(&A,q,q,l.host.nnz,l.d_csr_offsets.data,nullptr,l.d_csr_columns.data,csr.data,CUDSS_R_32I,CUDSS_R_32I,CUDSS_R_64F,CUDSS_MTYPE_SPD,CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
        station_refinement.prepare(v,l,lo,hi,stream);
        cuda_check(cudaEventCreate(&begin));cuda_check(cudaEventCreate(&end));cuda_check(cudaStreamSynchronize(stream));
    }
    void start(cudaStream_t s){cuda_check(cudaEventRecord(begin,s));}
    double finish(cudaStream_t s){cuda_check(cudaEventRecord(end,s));cuda_check(cudaEventSynchronize(end));float ms;cuda_check(cudaEventElapsedTime(&ms,begin,end));return ms;}
    void read_scalars(double* h,cudaStream_t s){cuda_check(cudaMemcpyAsync(h,scalars.data,24*sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));}
    bool check_dense(const int* ptr,int n,cudaStream_t s,bool retry_nonpositive=false){
        int initial[2]={n,0};cuda_check(cudaMemcpyAsync(check_info.data,initial,sizeof(initial),cudaMemcpyHostToDevice,s));lm_info<<<(n+127)/128,128,0,s>>>(ptr,n,check_info.data);cuda_check(cudaGetLastError());int result[2];cuda_check(cudaMemcpyAsync(result,check_info.data,sizeof(result),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));
        if(result[0]<n){if(retry_nonpositive&&result[1]>0)return false;throw std::runtime_error("cuSOLVER invalid local batch member "+std::to_string(result[0])+" info "+std::to_string(result[1]));}return true;
    }
    bool check_sparse(cudaStream_t s,bool retry_nonpositive=false){
        cuda_check(cudaStreamSynchronize(s));int info=0;size_t written=0;sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_INFO,&info,sizeof(info),&written));
        if(info){if(retry_nonpositive&&info>0){int clear=0;sparse_check(cudssDataSet(sparse,data,CUDSS_DATA_INFO,&clear,sizeof(clear)));return false;}throw std::runtime_error("cuDSS device numerical info "+std::to_string(info));}return true;
    }
    void evaluate_envelope(View v,LinearView l,cudaStream_t s){
        lm_camera_envelope_candidates<<<unsigned((v.nc-3+127)/128),128,0,s>>>(v,l,view);
        lm_camera_envelope_pick<<<1,1,0,s>>>(v,view);cuda_check(cudaGetLastError());
    }
    // Restore camera and observed-print boundaries in the current joint Schur
    // metric, jointly retaining discovered boundary cuts and dual multipliers.
    void project_envelope(View accepted,View trial,LinearView l,cudaStream_t s,const std::atomic<bool>& stop){
        if(!l.envelope.enabled&&!l.rig_directions&&!l.scene_directions&&!l.mount_side)return;lm_camera_project_start<<<1,1,0,s>>>(view);
        cuda_check(cudaMemcpyAsync(coupled.base,view.step,l.q*sizeof(double),cudaMemcpyDeviceToDevice,s));std::vector<CoupledBoundary> boundaries;
        for(int iteration=0;iteration<=128;++iteration){
            lm_retract_camera_controls<<<unsigned((3*accepted.nc+127)/128),128,0,s>>>(accepted,trial,l,view);
            if(l.rig_directions)lm_retract_rig<<<1,1,0,s>>>(accepted,trial,l,view);
            if(l.scene_directions)lm_retract_scene<<<unsigned((accepted.nscene+127)/128),128,0,s>>>(accepted,trial,l,view);
            CameraConstraint a{};PrintConstraint p{};MountConstraint m{};RetainedConstraint b{};bool camera_ok=true;
            if(l.envelope.enabled){evaluate_envelope(trial,l,s);
                cuda_check(cudaMemcpyAsync(&a,envelope_choice.data,sizeof(a),cudaMemcpyDeviceToHost,s));++envelope_downloads;}
            if(l.scene_directions){if(view.print_count)lm_print_candidates<<<unsigned((view.print_count+127)/128),128,0,s>>>(trial,view);lm_print_pick<<<1,1,0,s>>>(trial,view);
                cuda_check(cudaMemcpyAsync(&p,print_choice.data,sizeof(p),cudaMemcpyDeviceToHost,s));++print_downloads;}
            if(l.mount_side){lm_mount_pick<<<1,1,0,s>>>(trial,l,view);cuda_check(cudaMemcpyAsync(&m,mount_choice.data,sizeof(m),cudaMemcpyDeviceToHost,s));++mount_downloads;}
            if(l.rig_directions){lm_retained_pick<<<1,1,0,s>>>(l,view,retained_choice.data);cuda_check(cudaMemcpyAsync(&b,retained_choice.data,sizeof(b),cudaMemcpyDeviceToHost,s));++retained_downloads;}
            cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));if(stop.load())throw SolverStopped{};
            if(l.envelope.enabled)camera_ok=finite_scalar(a.margin)&&a.margin>=-camera_envelope_tolerance();
            bool strict_mount=m.kind==0||m.kind==1||(m.kind==3&&l.mount_side>=4);
            bool mount_ok=m.margin>=0.&&(!strict_mount||m.margin>0.);
            if(camera_ok&&p.margin>=0.&&mount_ok&&b.margin>=0.)return;
            if(iteration==128)throw std::runtime_error("joint camera/print boundary restoration failed at accepted-state precision");
            int selected=!camera_ok?0:(p.margin<0?1:2);double margin=selected==0?a.margin:(selected==1?p.margin:m.margin);
            if(p.margin<margin){selected=1;margin=p.margin;}if(m.margin<margin){selected=2;margin=m.margin;}
            if(b.margin<margin){selected=3;margin=b.margin;}
            if(std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS")){
                std::fprintf(stderr,"LM_COUPLED iteration=%d family=%d margin=%.17g camera_kind=%d cell=%d u=%.17g sign=%.17g print_path=%d station=%.17g side=%d mount_kind=%d retained_column=%d retained_sign=%d\n",
                    iteration,selected,margin,a.kind,a.cell,a.u,a.sign,p.path,p.station,p.side,m.kind,b.column,b.sign);std::fflush(stderr);}
            if(iteration<coupled_projection_limit&&finite_scalar(margin)){
                CoupledBoundary boundary{selected,a,p,m,b};int index=0;
                while(index<int(boundaries.size())&&!boundaries[index].same(boundary))++index;
                const bool origin=selected!=3&&index==int(boundaries.size());
                if(index==int(boundaries.size()))boundaries.push_back(boundary);
                const int cuts=int(boundaries.size());const View at=origin?accepted:trial;
                // First-order constraints use the same accepted-state chart as
                // the objective. Later nonlinear corrections replace the same
                // face instead of accumulating obsolete tangent halfspaces.
                if((selected==1||selected==2)&&l.scene_directions)lm_print_chart_directions<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                if(selected==3)lm_retained_gradient<<<(l.q+127)/128,128,0,s>>>(l,view,b);
                else if(selected==2)lm_mount_gradient<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                else if(selected==1)lm_print_gradient<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                else lm_camera_envelope_gradient<<<(l.q+127)/128,128,0,s>>>(accepted,at,l,view,origin);
                sparse_check(cudssExecute(sparse,CUDSS_PHASE_SOLVE,config,data,A,envelope_X,envelope_B));check_sparse(s);
                if(stop.load())throw SolverStopped{};
                const bool trace_projection=std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS")!=nullptr;
                lm_coupled_add_cut<<<1,1,0,s>>>(l,view,coupled,index,cuts,margin,origin,trace_projection);
                lm_coupled_dual<<<1,1,0,s>>>(coupled,cuts,trace_projection);lm_coupled_apply<<<(l.q+127)/128,128,0,s>>>(l,view,coupled,cuts);
                lm_project_shared_step<<<1,1,0,s>>>(accepted,l,view);++envelope_projections;
            }else{
                lm_camera_scale_step<<<(l.q+127)/128,128,0,s>>>(l,view,iteration==127?0.:.5);++envelope_backtracks;
            }
        }
    }
    bool run(View v,View trial,Linearization& l,size_t input_count,cudaStream_t s,const double* accepted_output,double* output,double* timings,const std::atomic<bool>& stop_requested){
        double values[24]{};auto wall=Clock::now();std::fill(output,output+32,0.);std::fill(timings,timings+12,0.);int q=l.host.q,cc=l.host.ncomponents;trial_available=false;const size_t envelope_copies_before=envelope_downloads,print_copies_before=print_downloads,mount_copies_before=mount_downloads,retained_copies_before=retained_downloads;
        const auto stop_boundary=[&](){if(stop_requested.load()){cuda_check(cudaStreamSynchronize(s));throw SolverStopped{};}};
        stop_boundary();
        if(l.device.envelope.enabled||l.device.rig_directions||l.device.scene_directions||l.device.mount_side)cuda_check(cudaMemcpyAsync(trial_input.data,v.board,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));
        start(s);cuda_check(cudaMemsetAsync(scalars.data,0,scalars.bytes(),s));int initial=std::max({q,cc,l.host.ng});
        lm_start<<<(initial+127)/128,128,0,s>>>(l.device,l.d_h.data,l.d_shared.data,l.d_stats.data,scalars.data);lm_set_lambda<<<1,1,0,s>>>(scalars.data,lambda,initial_relative,jacobi_damping);lm_set_damping<<<(initial+127)/128,128,0,s>>>(l.device,view,l.d_h.data,l.d_shared.data,jacobi_damping);
        if(image_station_damping&&view.nlocal)lm_image_station_metric<<<cc,128,0,s>>>(l.device,view,l.d_jac.data);
        if(image_global_damping){cuda_check(cudaMemsetAsync(image_metric.data,0,image_metric.bytes(),s));lm_image_global_accumulate<<<unsigned((v.n+127)/128),128,0,s>>>(v,l.device,view,l.d_jac.data);lm_image_global_metric<<<(q+127)/128,128,0,s>>>(l.device,view);if(view.nlocal)lm_image_local_floor<<<(view.nlocal+127)/128,128,0,s>>>(l.device,view);}
        lm_init_retained_bounds<<<(q+127)/128,128,0,s>>>(v,l.device,view);if(view.nlocal)lm_init_bounds<<<(view.nlocal+127)/128,128,0,s>>>(v,l.device,view,l.d_g.data);cuda_check(cudaGetLastError());read_scalars(values,s);timings[1]+=finish(s);
        if(!(values[0]>0)||!finite_scalar(values[0]))throw std::runtime_error("no observable finite damped descent system");
        double used_lambda=values[0];int bound_passes=0,scalar_copies=1,dense_checks=0,domain_copies=0;
        bool have_anchor=false,conditional_completed=false;size_t anchor_power=1,anchor_distance=0,working_set_copies=0;
        const bool trace_active=std::getenv("RINKSPACE_TRACE_LM_ACTIVESET")!=nullptr;
        const auto trace=[&](const char* phase){if(trace_active&&(bound_passes<=8||(bound_passes&(bound_passes-1))==0)){
            std::fprintf(stderr,"LM_ACTIVE pass=%d phase=%s elapsed_ms=%.3f lambda=%.12g\n",bound_passes,phase,millis(wall),used_lambda);std::fflush(stderr);}};
        auto reject_factor=[&](){++numerical_rejections;lambda=used_lambda*nu;nu*=2.;used_lambda=lambda;if(!(lambda>0)||!finite_scalar(lambda)||!finite_scalar(nu))throw std::runtime_error("LM damping overflow after numerical factorization rejection");
            have_anchor=false;anchor_power=1;anchor_distance=0;
            cuda_check(cudaMemcpyAsync(scalars.data,&used_lambda,sizeof(double),cudaMemcpyHostToDevice,s));lm_init_retained_bounds<<<(q+127)/128,128,0,s>>>(v,l.device,view);if(view.nlocal)lm_init_bounds<<<(view.nlocal+127)/128,128,0,s>>>(v,l.device,view,l.d_g.data);cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));
            if(numeric_progress)numeric_progress(numerical_rejections,lambda,nu,numeric_context);
            stop_boundary();};
        for(;;){stop_boundary();++bound_passes;trace("local");start(s);lm_fill_local<<<cc,128,0,s>>>(l.device,view,l.d_h.data,l.d_g.data,l.d_jac.data);cuda_check(cudaGetLastError());bool local_ok=true;
            for(auto b:batches){dense_check(cusolverDnDpotrfBatched(dense,CUBLAS_FILL_MODE_LOWER,b.dimension,factor_ptrs.data+b.factor_begin,b.dimension,factor_info.data+b.factor_begin,int(b.factor_count)));++dense_checks;
                if(!check_dense(factor_info.data+b.factor_begin,int(b.factor_count),s,true)){stop_boundary();local_ok=false;break;}
                stop_boundary();
                dense_check(cusolverDnDpotrsBatched(dense,CUBLAS_FILL_MODE_LOWER,b.dimension,1,solve_a_ptrs.data+b.solve_begin,b.dimension,solve_b_ptrs.data+b.solve_begin,b.dimension,solve_info.data,int(b.solve_count)));++dense_checks;check_dense(solve_info.data,1,s);stop_boundary();}
            if(!local_ok){timings[2]+=finish(s);stop_boundary();reject_factor();continue;}
            // Separate copying and diagonal addition avoids overlapping writes.
            cuda_check(cudaMemcpyAsync(csr.data,l.d_shared.data,csr.bytes(),cudaMemcpyDeviceToDevice,s));
            lm_reduce_start<<<(q+127)/128,128,0,s>>>(l.device,view,l.d_shared.data);
            lm_reduce<<<cc,128,0,s>>>(l.device,view,l.d_h.data,l.d_g.data);lm_constrain_retained_matrix<<<(q+127)/128,128,0,s>>>(l.device,view);lm_constrain_retained_rhs<<<(q+127)/128,128,0,s>>>(l.device,view);cuda_check(cudaGetLastError());timings[2]+=finish(s);
            stop_boundary();
            if(!analysis_ready){start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_ANALYSIS,config,data,A,X,B));check_sparse(s);size_t written=0;
                sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_MEMORY_ESTIMATES,estimates,sizeof(estimates),&written));sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_LU_NNZ,&factor_nnz,sizeof(factor_nnz),&written));size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
                if(estimates[1]>device_budget-double(device_bytes())||estimates[1]>free||estimates[3]>host_budget)
                    throw std::runtime_error("cuDSS symbolic factor memory exceeds current budget: device_peak="+std::to_string(estimates[1])+", device_free="+std::to_string(free)+", solver_owned="+std::to_string(device_bytes())+", device_budget="+std::to_string(device_budget)+", host_peak="+std::to_string(estimates[3])+", host_budget="+std::to_string(host_budget));
                analysis_ready=true;++analyses;timings[3]+=finish(s);}
            stop_boundary();
            start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_FACTORIZATION,config,data,A,X,B));bool factor_ok=check_sparse(s,true);timings[4]+=finish(s);stop_boundary();if(!factor_ok){reject_factor();continue;}
            start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_SOLVE,config,data,A,X,B));check_sparse(s);timings[5]+=finish(s);stop_boundary();
            const size_t projections_before=envelope_projections;
            trace("shared_projection");start(s);lm_residual_init<<<(q+127)/128,128,0,s>>>(l.device,view);lm_residual_spmv<<<(q+127)/128,128,0,s>>>(l.device,view);lm_project_shared_step<<<1,1,0,s>>>(v,l.device,view);project_envelope(v,trial,l.device,s,stop_requested);trace("backsubstitution");lm_backsub<<<cc,128,0,s>>>(l.device,view);
            lm_feasible_init<<<1,1,0,s>>>(view);int nf=std::max(q,view.nlocal);lm_feasible<<<(nf+127)/128,128,0,s>>>(v,l.device,view);
            if(view.nlocal){lm_local_release_start<<<1,1,0,s>>>(view,check_info.data);
                lm_local_release<<<(cc+127)/128,128,0,s>>>(l.device,view,l.d_h.data,l.d_g.data,l.d_jac.data,check_info.data);
                lm_local_release_finish<<<1,1,0,s>>>(view,check_info.data);}
            lm_feasible_finish<<<1,1,0,s>>>(view);cuda_check(cudaGetLastError());read_scalars(values,s);++scalar_copies;timings[6]+=finish(s);
            if(trace_active&&(bound_passes<=8||(bound_passes&(bound_passes-1))==0)){int released=view.nlocal;
                if(view.nlocal)cuda_check(cudaMemcpy(&released,check_info.data,sizeof(int),cudaMemcpyDeviceToHost));
                std::fprintf(stderr,"LM_ACTIVE pass=%d phase=feasibility changed=%.0f blocked=%.0f release=%d elapsed_ms=%.3f\n",bound_passes,values[5],values[6],released,millis(wall));std::fflush(stderr);}
            if(values[7]||!finite_scalar(values[4])||values[4]>linear_tolerance)throw std::runtime_error("nonfinite step or excessive cuDSS residual");if(values[5]==0)break;
            // The physical projection changes the shared solution after Schur
            // elimination. With independent scalar locals, solve their bounded
            // conditional quadratics once instead of cycling through face sets.
            // Probe without mutating the working set: an invalid finite-support
            // completion must fall through to the existing active-set solve.
            if(batches.empty()&&view.nlocal&&envelope_projections>projections_before){
                start(s);cuda_check(cudaMemsetAsync(check_info.data,0,sizeof(int),s));
                lm_conditional_shared_check<<<(q+127)/128,128,0,s>>>(l.device,view,check_info.data);
                lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.d_h.data,l.d_g.data,accepted_output,l.d_jac.data,check_info.data,false);
                int invalid=0;cuda_check(cudaMemcpyAsync(&invalid,check_info.data,sizeof(int),cudaMemcpyDeviceToHost,s));
                cuda_check(cudaStreamSynchronize(s));++working_set_copies;stop_boundary();
                if(!invalid){
                    lm_conditional_start<<<1,1,0,s>>>(view,check_info.data);
                    lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.d_h.data,l.d_g.data,accepted_output,l.d_jac.data,check_info.data);
                    cuda_check(cudaGetLastError());conditional_completed=true;timings[6]+=finish(s);
                    if(trace_active){std::fprintf(stderr,"LM_ACTIVE pass=%d phase=direct_conditional_local_completion lambda=%.12g\n",bound_passes,used_lambda);std::fflush(stderr);}
                    break;
                }
                timings[6]+=finish(s);
            }
            unsigned working_blocks=unsigned((size_t(q)+view.nlocal+127)/128);
            if(have_anchor){int different=0;cuda_check(cudaMemsetAsync(check_info.data,0,sizeof(int),s));
                lm_working_set<<<working_blocks,128,0,s>>>(q,view,working_set_anchor.data,false,check_info.data);
                cuda_check(cudaMemcpyAsync(&different,check_info.data,sizeof(int),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));++working_set_copies;stop_boundary();
                if(!different){
                    if(batches.empty()&&view.nlocal){
                        start(s);lm_conditional_start<<<1,1,0,s>>>(view,check_info.data);
                        lm_conditional_shared_check<<<(q+127)/128,128,0,s>>>(l.device,view,check_info.data);
                        lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.d_h.data,l.d_g.data,accepted_output,l.d_jac.data,check_info.data);
                        cuda_check(cudaGetLastError());int invalid=0;
                        cuda_check(cudaMemcpyAsync(&invalid,check_info.data,sizeof(int),cudaMemcpyDeviceToHost,s));
                        cuda_check(cudaStreamSynchronize(s));++working_set_copies;timings[6]+=finish(s);stop_boundary();
                        if(!invalid){conditional_completed=true;if(trace_active){std::fprintf(stderr,"LM_ACTIVE pass=%d phase=conditional_local_completion lambda=%.12g\n",bound_passes,used_lambda);std::fflush(stderr);}break;}
                    }
                    // No trial state has been evaluated or accepted. Report a
                    // rejected attempt and let the outer LM increase damping.
                    std::fill(values+8,values+24,0.);values[3]=0.;values[9]=-1.;values[10]=values[1];
                    lambda=used_lambda*nu;nu*=2.;++trials;std::copy(values,values+24,output);
                    output[15]=0.;output[16]=4.;output[17]=lambda;output[18]=nu;output[19]=bound_passes;output[20]=analyses;output[21]=trials;output[22]=accepted;
                    output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int)+domain_copies*sizeof(double)+working_set_copies*sizeof(int)+(envelope_downloads-envelope_copies_before)*sizeof(CameraConstraint)+(print_downloads-print_copies_before)*sizeof(PrintConstraint)+(mount_downloads-mount_copies_before)*sizeof(MountConstraint)+(retained_downloads-retained_copies_before)*sizeof(RetainedConstraint);
                    output[24]=double(device_bytes());output[25]=double(estimates[0]);output[26]=double(estimates[1]);output[27]=double(estimates[2]);output[28]=double(estimates[3]);output[29]=double(factor_nnz);output[30]=eta;output[31]=0.;timings[0]=millis(wall);return false;
                }
                if(++anchor_distance==anchor_power){anchor_power*=2;anchor_distance=0;have_anchor=false;}
            }
            if(!have_anchor){lm_working_set<<<working_blocks,128,0,s>>>(q,view,working_set_anchor.data,true,check_info.data);cuda_check(cudaGetLastError());have_anchor=true;}
        }
        stop_boundary();
        start(s);cuda_check(cudaMemcpyAsync(trial_input.data,v.board,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));size_t n=std::max({v.n,3*v.nc,size_t(q)});
        // Bounded inner numerical restoration, not an outer solve/run budget.
        // Every final trial is re-evaluated under the same original objective.
        for(int restoration=0;restoration<=40;++restoration){
            lm_retract<<<unsigned((n+127)/128),128,0,s>>>(v,trial,l.device,view);if(l.device.rig_directions)lm_retract_rig<<<1,1,0,s>>>(v,trial,l.device,view);
            if(l.device.scene_directions){lm_retract_scene<<<unsigned((v.nscene+127)/128),128,0,s>>>(v,trial,l.device,view);lm_build_course<<<1,1,0,s>>>(trial);}
            if(view.nlocal){lm_restore_station_domain<<<(view.nlocal+127)/128,128,0,s>>>(trial,l.device,view);lm_share_restored_stations<<<unsigned((v.n+127)/128),128,0,s>>>(trial,l.device,view);lm_actual_locals<<<(view.nlocal+127)/128,128,0,s>>>(v,trial,l.device,view);}
            evaluate_kernel<<<unsigned((v.n+127)/128),128,0,s>>>(trial,trial_output.data,trial_status.data);
            if(restoration==40)break;
            lm_restore_profile_domain<<<1,1,0,s>>>(v,trial,l.device,view,l.d_jac.data,trial_status.data);
            double restored=0.;++domain_copies;cuda_check(cudaMemcpyAsync(&restored,view.scratch,sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));stop_boundary();
            if(restored==0.)break;
            lm_project_shared_step<<<1,1,0,s>>>(v,l.device,view);
            project_envelope(v,trial,l.device,s,stop_requested);
        }
        lm_prediction<<<cc,128,0,s>>>(l.device,view,l.d_h.data,l.d_g.data);cuda_check(cudaGetLastError());timings[7]=finish(s);
        start(s);lm_trial_whiten<<<l.host.ng,128,0,s>>>(l.device,trial_output.data,trial_white.data);loss_kernel<<<(l.host.ng+127)/128,128,0,s>>>(l.device,trial_white.data,trial_stats.data);
        lm_finish_init<<<1,1,0,s>>>(trial,view);if(l.device.rig_directions||l.device.mount_side)lm_rig_domain<<<1,1,0,s>>>(trial,l.device,view);if(l.device.scene_directions)lm_scene_domain<<<unsigned((std::max(v.nscene,v.npaths)+127)/128),128,0,s>>>(trial,l.device,view);size_t nf=std::max({v.n,size_t(view.nlocal),size_t(q)+view.nlocal,size_t(l.host.ng),size_t(cc)});
        if(l.device.envelope.enabled){evaluate_envelope(trial,l.device,s);lm_camera_envelope_domain<<<1,1,0,s>>>(view);}
        lm_finish<<<unsigned((nf+127)/128),128,0,s>>>(trial,l.device,view,trial_status.data,trial_stats.data);lm_finish_values<<<1,1,0,s>>>(trial,view);cuda_check(cudaGetLastError());timings[8]=finish(s);trial_available=true;
        start(s);read_scalars(values,s);++scalar_copies;timings[9]=finish(s);stop_boundary();int reason=0;
        if(values[8]||!finite_scalar(values[10]))reason=3;else if(!(values[11]>0)||!(values[3]>0)||!finite_scalar(values[11]))reason=2;else if(!(values[12]>0)||!(values[13]>eta))reason=1;
        bool accept=reason==0;++trials;
        if(accept){start(s);cuda_check(cudaMemcpyAsync(const_cast<double*>(v.board),trial_input.data,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));if(v.scene_paths)cuda_check(cudaMemcpyAsync(const_cast<BoardCourse*>(v.course),trial.course,sizeof(BoardCourse),cudaMemcpyDeviceToDevice,s));timings[10]=finish(s);++accepted;l.gpu_ready=false;l.cpu_ready=false;lambda=used_lambda*std::max(1./3.,1.-std::pow(2*values[13]-1,3));nu=2.;}else{lambda=used_lambda*nu;nu*=2.;}
        std::copy(values,values+24,output);output[15]=accept;output[16]=accept&&conditional_completed?5:reason;output[17]=lambda;output[18]=nu;output[19]=bound_passes;output[20]=analyses;output[21]=trials;output[22]=accepted;
        output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int)+domain_copies*sizeof(double)+working_set_copies*sizeof(int)+(envelope_downloads-envelope_copies_before)*sizeof(CameraConstraint)+(print_downloads-print_copies_before)*sizeof(PrintConstraint)+(mount_downloads-mount_copies_before)*sizeof(MountConstraint)+(retained_downloads-retained_copies_before)*sizeof(RetainedConstraint);output[24]=double(device_bytes());output[25]=double(estimates[0]);output[26]=double(estimates[1]);output[27]=double(estimates[2]);output[28]=double(estimates[3]);output[29]=double(factor_nnz);output[30]=eta;output[31]=values[23];timings[0]=millis(wall);return accept;
    }
};
