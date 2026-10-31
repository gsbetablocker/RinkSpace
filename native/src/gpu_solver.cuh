/* Resident LM: ragged connected local systems, sparse retained CSR.
 * Scalar station elimination is division; larger dense blocks use cuSOLVER.
 * cuDSS owns the retained sparse factorization; small boundary duals use the native controller. */
inline void dense_check(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cuSOLVER status "+std::to_string(int(s)));}
inline void sparse_check(cudssStatus_t s){if(s!=CUDSS_STATUS_SUCCESS)throw std::runtime_error("cuDSS status "+std::to_string(int(s)));}
struct LocalWorkspace {size_t matrix,solved;};
// Stop only at completed stream boundaries; accepted state remains intact.
struct SolverStopped : std::exception {
    const char* what()const noexcept override{return "native solve stopped by caller";}
};

struct PrintConstraint {double margin=INFINITY,station=0.;int path=-1,side=0;};
struct PrintProbe {double station;int path;};
struct MountConstraint {double margin=INFINITY;int kind=-1,camera=0,side=0;};
struct RetainedConstraint {double margin=INFINITY;int column=-1,sign=1,kind=0,other=-1;bool equality=false;int camera=0;};
struct CoupledBoundary {
    int family;CameraConstraint camera;PrintConstraint print;MountConstraint mount;RetainedConstraint retained;
    bool same(const CoupledBoundary& b)const{
        if(family!=b.family)return false;
        if(family==3)return retained.camera==b.retained.camera&&retained.column==b.retained.column&&retained.sign==b.retained.sign&&retained.kind==b.retained.kind&&retained.other==b.retained.other;
        if(family==2)return mount.camera==b.mount.camera&&mount.kind==b.mount.kind;
        if(family==1)return print.path==b.print.path&&print.station==b.print.station&&print.side==b.print.side;
        return same_camera_constraint_face(camera,b.camera);
    }
};
// Initial allocation only. Every discovered physical cut is retained; storage
// grows under the caller's actual resource budgets instead of shrinking a
// useful shared step merely because a fixed cut array filled up.
constexpr int coupled_initial_capacity=64;
struct CoupledProjectionView {double *base,*anchor,*gradients,*responses,*gram,*target,*dual,*equality;int capacity=coupled_initial_capacity;};
inline CoupledProjectionView coupled_view(double* data,int q,int capacity){
    CoupledProjectionView c{};c.capacity=capacity;c.base=data;c.anchor=data+q;
    c.gradients=c.anchor+q;c.responses=c.gradients+size_t(q)*capacity;
    c.gram=c.responses+size_t(q)*capacity;c.target=c.gram+size_t(capacity)*capacity;
    c.dual=c.target+capacity;c.equality=c.dual+capacity;return c;
}
inline size_t coupled_storage_size(int q,int capacity){
    return 2*size_t(q)+2*checked_product(size_t(q),size_t(capacity))+checked_product(size_t(capacity),size_t(capacity)+3);
}
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
    int* print_domain_invalid=nullptr;
};
RS_HD inline SolverView camera_solver_view(SolverView s,size_t nr,size_t camera){
    if(s.rig_bound_columns)s.rig_bound_columns+=camera*nr;
    if(s.envelope_delta)s.envelope_delta+=camera*27;
    return s;
}
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
    for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(l,v.nr,camera);auto cs=camera_solver_view(s,v.nr,camera);
    if(cl.rig_directions)for(int k=15;k<int(v.nr);++k){
        if(cs.rig_bound_columns[k]!=col)continue;int chart=k<24?k-9:(k<38?k-11:k-12);
        double lower=cl.rig_lower[k],upper=cl.rig_upper[k];if(k==37){lower=fmax(lower,-cl.max_readout);upper=fmin(upper,cl.max_readout);}
        double d=cl.rig_directions[size_t(chart)*l.q+col],a=(lower-cv.rig[k])/d,b=(upper-cv.rig[k])/d;
        lo=fmax(lo,fmin(a,b));hi=fmin(hi,fmax(a,b));
    }
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
__device__ bool lm_halfspace(const double* a,const double* b,int q,double base,double minimum,double* step,double other_sign=-1.){
    double value=base,norm=0.;for(int j=0;j<q;++j){double d=(a?a[j]:0.)+other_sign*(b?b[j]:0.);value+=d*step[j];norm+=d*d;}
    // Project to the declared boundary. An unconditional inward padding can
    // make two valid opposing bounds inconsistent, even at the zero step.
    if(value>=minimum)return false;
    if(norm==0.)return false; // A fixed, valid state needs no numerical interior offset.
    const double amount=(minimum-value)/norm;
    bool changed=false;for(int j=0;j<q;++j){double d=(a?a[j]:0.)+other_sign*(b?b[j]:0.),next=step[j]+amount*d;changed|=next!=step[j];step[j]=next;}
    return changed;
}
__global__ void lm_project_shared_step(View v,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;
    s.scratch[1]=0.;
    // Bound this inner numerical projection, not the outer solve. A cycle or
    // unresolved floating correction must return to the host safe boundary.
    for(int pass=0;pass<256;++pass){
        bool changed=false;
        if(l.scene_steps&&(v.options&4u)&&!(v.options&64u)){
            auto enforce=[&](int high,int low,double minimum=0.){
                // Option16 derives the left goal from -d; its positive chart
                // is pulled back to the current physical tangent here.
                double other_sign=-1.;if((v.options&16u)&&low==23){low=24;other_sign=1.;}
                const double* a=l.scene_steps+size_t(6+high)*l.q;
                const double* b=l.scene_steps+size_t(6+low)*l.q;
                changed|=lm_halfspace(a,b,l.q,v.layout[high]+other_sign*v.layout[low],minimum,s.step,other_sign);
            };
            enforce(0,23); // curling left centre cannot pass left goal line
            for(int row=0;row<4;++row){
                enforce(10+row,23);enforce(8,10+row); // each Lane start lies between goal and right edge
                const double interior=64*std::numeric_limits<double>::epsilon();
                enforce(14+row,1,interior);enforce(8,14+row,interior); // connections pass R and stop before N
            }
        }
        for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(l,v.nr,camera);auto cs=camera_solver_view(s,v.nr,camera);
        if(cl.rig_directions){
            for(int k=15;k<int(v.nr);++k){
                if(k==24||k==25||k==38)continue;
                if(s.rig_bound_columns&&cs.rig_bound_columns[k]>=0)continue; // handled by coupled boundary solve
                int chart=k<24?k-9:(k<38?k-11:k-12);
                const double* a=cl.rig_directions+size_t(chart)*l.q;
                double lower=cl.rig_lower[k],upper=cl.rig_upper[k];
                if(k==37){lower=fmax(lower,-cl.max_readout);upper=fmin(upper,cl.max_readout);}
                if(lower>upper)continue;
                if(finite_scalar(lower))changed|=lm_halfspace(a,nullptr,l.q,cv.rig[k],lower,s.step);
                if(finite_scalar(upper))changed|=lm_halfspace(nullptr,a,l.q,-cv.rig[k],-upper,s.step);
            }
        }
        }
        if(!changed)return;
    }
    s.scratch[1]=1.;
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
    if(k<3*accepted.nc){double d=direction_row_dot(l.directions,l.q,k,l.motion_rows,s.step,alpha);const_cast<double*>(trial.controls)[k]=accepted.controls[k]+d;}
    if(k<accepted.n){int id=l.row_local[k];if(id>=0){double station=accepted.stations[k]+alpha*s.step[l.q+id]*l.scales[id];double lo,hi;bool corner;if(station_piece(accepted,k,accepted.stations[k],lo,hi,corner))station=fmax(lo,fmin(hi,station));const_cast<double*>(trial.stations)[k]=fmax(s.lower[id],fmin(s.upper[id],station));}}
    if(k<size_t(l.q))s.applied[k]=alpha*s.step[k];
}
__global__ void lm_retract_rig(View accepted,View trial,LinearView l,SolverView s){
    size_t camera=blockIdx.x;if(camera>=accepted.nrig||!l.rig_directions)return;
    auto av=camera_view(accepted,camera),tv=camera_view(trial,camera);auto cl=camera_linear_view(l,accepted.nr,camera);auto cs=camera_solver_view(s,accepted.nr,camera);
    __shared__ double delta[31];int k=threadIdx.x;
    if(k<(accepted.nr==43?31:27))delta[k]=direction_row_dot(cl.rig_directions,l.q,k,cl.rig_rows,s.step,s.scalars[3]);
    __syncthreads();if(threadIdx.x)return;
    if(cs.envelope_delta)for(int i=0;i<27;++i)cs.envelope_delta[i]=delta[i];
    bool moved=false;for(int i=0;i<27;++i)moved|=delta[i]!=0.;
    if(!moved){for(int i=0;i<38;++i)const_cast<double*>(tv.rig)[i]=av.rig[i];}
    else if(retract_motion_rig(av.rig,delta,const_cast<double*>(tv.rig))!=Status::ok)const_cast<double*>(tv.rig)[0]=std::numeric_limits<double>::quiet_NaN();
    if(accepted.nr==43)for(int i=0;i<4;++i)const_cast<double*>(tv.rig)[39+i]=av.rig[39+i]+delta[27+i];
    // Preserve exact equalities only inside the original roundoff allowance.
    for(size_t i=0;i<accepted.nr;++i)if(cl.rig_lower[i]==cl.rig_upper[i]&&finite_scalar(cl.rig_lower[i])&&
        fabs(tv.rig[i]-cl.rig_lower[i])<=camera_envelope_tolerance()*(1+fabs(cl.rig_lower[i])))
        const_cast<double*>(tv.rig)[i]=cl.rig_lower[i];
}
__global__ void lm_camera_project_start(SolverView s){if(!threadIdx.x&&!blockIdx.x)s.scalars[3]=1.;}
__global__ void lm_retract_camera_controls(View accepted,View trial,LinearView l,SolverView s){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=3*accepted.nc)return;
    double d=direction_row_dot(l.directions,l.q,k,l.motion_rows,s.step,1.);
    const_cast<double*>(trial.controls)[k]=accepted.controls[k]+d;
}
__global__ void lm_camera_envelope_candidates(View v,LinearView l,SolverView s){
    int cell=1+blockIdx.x*blockDim.x+threadIdx.x;if(cell>=int(v.nc)-2)return;
    CameraConstraint worst{};worst.margin=INFINITY;
    for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(l,v.nr,camera);if(!cl.envelope.enabled)continue;
        auto a=camera_cell_constraint(motion_rig(cv.rig),MotionSpline{v.controls,v.nc,v.first,v.dt},cl.envelope,cell);a.camera=int(camera);
        if(!finite_scalar(a.margin)||a.margin<worst.margin)worst=a;}
    s.envelope_candidates[cell-1]=worst;
}
__global__ void lm_camera_envelope_pick(View v,SolverView s){
    // Preserve the serial selector, including first-minimum ties and its
    // handling of the last nonfinite candidate, without one lane scanning
    // every spline cell at each physical-boundary correction.
    __shared__ int indices[128];const int t=threadIdx.x,n=int(v.nc)-3;
    int invalid=-1;for(int i=t;i<n;i+=blockDim.x)if(!finite_scalar(s.envelope_candidates[i].margin))invalid=i;
    indices[t]=invalid;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)indices[t]=max(indices[t],indices[t+stride]);__syncthreads();}
    const int last_invalid=indices[0];__syncthreads();
    // NaN and -inf retain the last invalid candidate; +inf may be replaced
    // by a later finite value, exactly as in the original ordered scan.
    if(last_invalid>=0&&!(s.envelope_candidates[last_invalid].margin>0.)){
        if(t==0)*s.envelope_choice=s.envelope_candidates[last_invalid];return;
    }
    int best=last_invalid;
    for(int i=last_invalid+1+t;i<n;i+=blockDim.x)
        if(best<0||s.envelope_candidates[i].margin<s.envelope_candidates[best].margin)best=i;
    indices[t]=best;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){
        if(t<stride){const int a=indices[t],b=indices[t+stride];
            if(b>=0&&(a<0||s.envelope_candidates[b].margin<s.envelope_candidates[a].margin||
                (s.envelope_candidates[b].margin==s.envelope_candidates[a].margin&&b<a)))indices[t]=b;}
        __syncthreads();
    }
    if(t==0)*s.envelope_choice=s.envelope_candidates[indices[0]];
}
__global__ void lm_camera_envelope_gradient(View accepted,View trial,LinearView l,SolverView s,bool origin){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    if(s.retained_blocked[j]&&j!=0){s.envelope_gradient[j]=0.;return;}
    const auto camera=size_t(s.envelope_choice->camera);auto av=camera_view(accepted,camera);auto cl=camera_linear_view(l,accepted.nr,camera);auto cs=camera_solver_view(s,accepted.nr,camera);
    LocalJet d[27],p[38];for(int k=0;k<27;++k)d[k]={(cl.rig_directions&&!origin)?cs.envelope_delta[k]:0.,cl.rig_directions?cl.rig_directions[size_t(k)*l.q+j]:0.};
    if(retract_motion_rig(av.rig,d,p)!=Status::ok){s.envelope_gradient[j]=std::numeric_limits<double>::quiet_NaN();return;}
    const auto a=*s.envelope_choice;LocalJet controls[12];
    for(int k=0;k<12;++k){size_t i=3*size_t(a.cell-1)+k;controls[k]={trial.controls[i],l.directions[i*l.q+j]};}
    auto value=camera_constraint_value(rig_from_values(p),controls,trial.dt,a);
    s.envelope_gradient[j]=s.retained_blocked[j]?0.:value.v;if(j==0)s.scratch[0]=value.a;
}
// All linear physical bounds enter the SAME Schur metric projection.
// Euclidean pre-projection would change the minimizer before this solve.
// Finite ice relation faces use the same native formula for values, cuts,
// trial-domain validation and the original-objective stationarity cone.
__device__ inline LocalJet ice_relation_value(View v,LinearView l,int relation,int column){
    LocalJet b[6],q[ice_tolerance_layout_size];
    auto value=[&](size_t i,double x){return LocalJet(x,column<0?0.:scene_chart_component(v.scene_values,l.scene_chart,l.scene_steps,l.q,column,i));};
    for(int i=0;i<6;++i)b[i]=value(i,v.board[i]);
    for(int i=0;i<ice_tolerance_layout_size;++i)q[i]=value(6+i,v.layout[i]);
    LocalJet result=NAN;IceToleranceGeometry<LocalJet> g(b,q);
    ice_tolerance_margins(g,[&](int id,LocalJet margin){if(id==relation)result=margin;});return result;
}
__global__ void lm_retained_pick(View accepted,View trial,LinearView l,SolverView s,RetainedConstraint* choice){
    if(threadIdx.x||blockIdx.x)return;RetainedConstraint worst{};
    auto offer=[&](double margin,int index,int sign,int kind=0,int other=-1,bool equality=false,int camera=0){if(margin<worst.margin)worst={margin,index,sign,kind,other,equality,camera};};
    for(int j=0;j<l.q;++j){if(s.retained_lower[j]==s.retained_upper[j])continue;
        offer(s.step[j]-s.retained_lower[j],j,1);offer(s.retained_upper[j]-s.step[j],j,-1);}
    for(size_t camera=0;camera<accepted.nrig;++camera){auto av=camera_view(accepted,camera),tv=camera_view(trial,camera);auto cl=camera_linear_view(l,accepted.nr,camera);auto cs=camera_solver_view(s,accepted.nr,camera);
    if(cl.rig_directions)for(int k=15;k<int(accepted.nr);++k){
        if(k==24||k==25||k==38||cs.rig_bound_columns[k]>=0)continue;
        double lo=cl.rig_lower[k],hi=cl.rig_upper[k];if(k==37){lo=fmax(lo,-cl.max_readout);hi=fmin(hi,cl.max_readout);}
        if(lo==hi&&finite_scalar(lo))offer(-fabs(tv.rig[k]-lo),k,tv.rig[k]<lo?1:-1,1,-1,true,int(camera));
        else{if(finite_scalar(lo))offer(tv.rig[k]-lo,k,1,1,-1,false,int(camera));if(finite_scalar(hi))offer(hi-tv.rig[k],k,-1,1,-1,false,int(camera));}
    }
    }
    if(l.scene_directions){
        for(size_t k=0;k<trial.nscene;++k){double lo=l.scene_lower[k],hi=l.scene_upper[k],value=trial.scene_values[k];
            if(lo==hi&&finite_scalar(lo))offer(-fabs(value-lo),int(k),value<lo?1:-1,3,-1,true);
            else{if(finite_scalar(lo))offer(value-lo,int(k),1,3);if(finite_scalar(hi))offer(hi-value,int(k),-1,3);}}
        if((accepted.options&4u)&&!(accepted.options&64u)){auto relation=[&](int high,int low,double minimum=0.){
                const bool mirrored=(accepted.options&16u)&&low==23;if(mirrored)low=24;
                offer(trial.layout[high]+(mirrored?1.:-1.)*trial.layout[low]-minimum,6+high,1,mirrored?4:2,6+low);};
            relation(0,23);for(int row=0;row<4;++row){relation(10+row,23);relation(8,10+row);
                const double interior=64*std::numeric_limits<double>::epsilon();relation(14+row,1,interior);relation(8,14+row,interior);}}
    }
    if(l.scene_directions&&(accepted.options&64u)){
        IceToleranceGeometry<double> g(trial.board,trial.layout);
        ice_tolerance_margins(g,[&](int id,double margin){offer(margin,id,1,5);});
    }
    *choice=worst;
}
__global__ void lm_retained_gradient(View v,LinearView l,SolverView s,RetainedConstraint a){
    l=camera_linear_view(l,v.nr,size_t(a.camera));
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;double value=0.;
    // Motion/camera-only columns have identically zero scene derivatives.
    // Do not build and differentiate all ice relations for those columns.
    if(a.kind==5)value=(!l.scene_active||l.scene_active[j])?ice_relation_value(v,l,a.column,j).v:0.;
    else if(a.kind==0)value=j==a.column?1.:0.;
    else if(a.kind==1){int k=a.column,chart=k<24?k-9:(k<38?k-11:k-12);value=l.rig_directions[size_t(chart)*l.q+j];}
    else if(a.kind==2||a.kind==4)value=l.scene_directions[size_t(a.column)*l.q+j]+(a.kind==4?1.:-1.)*l.scene_directions[size_t(a.other)*l.q+j];
    else value=l.scene_directions[size_t(a.column)*l.q+j];
    s.envelope_gradient[j]=a.sign*value;
}
__global__ void lm_coupled_add_cut(LinearView l,SolverView s,CoupledProjectionView c,int index,int count,double margin,bool origin,bool equality,bool trace){
    __shared__ double scale;
    if(threadIdx.x==0){double norm=0.,offset=0.,roundoff_scale=0.;if(origin)margin=s.scratch[0];
        // Keep the original column summation order for every scalar equation.
        for(int j=0;j<l.q;++j){norm+=s.envelope_gradient[j]*s.envelope_response[j];
            offset+=s.envelope_gradient[j]*((origin?0.:s.step[j])-c.base[j]);
            roundoff_scale+=fabs(s.envelope_gradient[j])*(fabs(c.base[j])+(origin?0.:fabs(s.step[j])));}
        scale=norm>0.&&finite_scalar(norm)?1./sqrt(norm):0.;
        const double inset=4*camera_envelope_tolerance()+16*std::numeric_limits<double>::epsilon()*roundoff_scale;
        c.target[index]=scale?(offset+(equality?0.:inset)-margin)*scale:0.;c.dual[index]=0.;c.equality[index]=equality;
        if(trace){double zero_margin=margin;if(!origin)for(int j=0;j<l.q;++j)zero_margin-=s.envelope_gradient[j]*s.step[j];
            printf("LM_CUT index=%d at_origin=%d origin_margin=%.17g metric_norm=%.17g\n",index,int(origin),zero_margin,norm);}
    }
    __syncthreads();
    double* a=c.gradients+size_t(index)*l.q;double* w=c.responses+size_t(index)*l.q;
    for(int j=threadIdx.x;j<l.q;j+=blockDim.x){a[j]=scale?s.envelope_gradient[j]*scale:0.;w[j]=scale?s.envelope_response[j]*scale:0.;}
    __syncthreads();
    // Cross products are independent. One thread per retained cut avoids
    // serializing the entire small Gram matrix on a single GPU lane.
    for(int i=threadIdx.x;i<count;i+=blockDim.x){
        if(i==index){c.gram[size_t(index)*c.capacity+index]=1.;continue;}
        double cross=0.;const double* previous=c.gradients+size_t(i)*l.q;
        for(int j=0;j<l.q;++j)cross+=previous[j]*w[j];
        c.gram[size_t(index)*c.capacity+i]=c.gram[size_t(i)*c.capacity+index]=cross;
    }
}
// Hildreth's nonnegative dual coordinate solve in the CURRENT Schur metric.
// Retaining multipliers lets a later constraint release an earlier correction;
// sequential feasibility-only projections cannot do that. Numerical work stays
// in the same native code on CPU or GPU. The outer loop verifies the original nonlinear domains and
// LM prediction/objective; this is not a generic nonlinear QP solver.
__global__ void lm_coupled_dual(CoupledProjectionView c,SolverView s,int count,bool trace){
    if(threadIdx.x||blockIdx.x)return;
    double kkt;int sweeps=coupled_dual_minimize(count,c.capacity,c.gram,c.target,c.dual,c.equality,&kkt);
    s.scratch[1]=coupled_dual_unresolved(count,c.target,kkt)?1.:0.;
    if(trace)printf("LM_DUAL count=%d sweeps=%d kkt=%.17g\n",count,sweeps,kkt);
}
__global__ void lm_coupled_apply(LinearView l,SolverView s,CoupledProjectionView c,int count){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;double value=c.base[j];
    for(int i=0;i<count;++i)value+=c.dual[i]*c.responses[size_t(i)*l.q+j];
    s.step[j]=value;
}
// Bundle compression preserves the current dual correction and objective.
// A nonnegative combination of supporting halfspaces is still a valid cut;
// keeping that aggregate plus the next tangent avoids a nearly singular pile
// of old pupil-vertex cuts. Exact nonlinear feasibility is checked afterward.
__global__ void lm_coupled_merge_vertex(LinearView l,CoupledProjectionView c,int first,int second,int count){
    if(threadIdx.x||blockIdx.x)return;
    const int stride=c.capacity,last=count-1;
    double a=c.dual[first],b=c.dual[second];
    double norm=sqrt(fmax(0.,a*a+b*b+2*a*b*c.gram[size_t(first)*stride+second]));
    // Two inactive cuts can be represented by the more recent one. A later
    // exact violation reintroduces any supporting direction that is needed.
    const bool active=norm>0.&&finite_scalar(norm);
    if(active){a/=norm;b/=norm;}else{a=0.;b=1.;}
    for(int j=0;j<l.q;++j){
        const size_t i=size_t(first)*l.q+j,k=size_t(second)*l.q+j;
        c.gradients[i]=a*c.gradients[i]+b*c.gradients[k];
        c.responses[i]=a*c.responses[i]+b*c.responses[k];
    }
    c.target[first]=a*c.target[first]+b*c.target[second];c.dual[first]=active?norm:0.;
    for(int k=0;k<count;++k)if(k!=first&&k!=second){
        const double cross=a*c.gram[size_t(first)*stride+k]+b*c.gram[size_t(second)*stride+k];
        c.gram[size_t(first)*stride+k]=c.gram[size_t(k)*stride+first]=cross;
    }
    c.gram[size_t(first)*stride+first]=1.;
    if(second!=last){
        for(int j=0;j<l.q;++j){c.gradients[size_t(second)*l.q+j]=c.gradients[size_t(last)*l.q+j];c.responses[size_t(second)*l.q+j]=c.responses[size_t(last)*l.q+j];}
        c.target[second]=c.target[last];c.dual[second]=c.dual[last];c.equality[second]=c.equality[last];
        for(int k=0;k<last;++k)if(k!=second)c.gram[size_t(second)*stride+k]=c.gram[size_t(k)*stride+second]=c.gram[size_t(last)*stride+k];
        c.gram[size_t(second)*stride+second]=1.;
    }
}
__global__ void lm_camera_scale_step(LinearView l,SolverView s,double scale){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<l.q)s.step[j]*=scale;}
__global__ void lm_camera_backtrack_correction(LinearView l,SolverView s,CoupledProjectionView c){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<l.q)s.step[j]=c.anchor[j]+.5*(s.step[j]-c.anchor[j]);
}
__global__ void lm_camera_envelope_domain(SolverView s){
    if(threadIdx.x||blockIdx.x)return;double margin=s.envelope_choice->margin;
    if(!finite_scalar(margin)||margin < -camera_envelope_tolerance()){s.scalars[8]+=1;s.scalars[9]=0.;}
}
__global__ void lm_rig_domain(View trial,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;
    bool bad=false;
    for(size_t camera=0;camera<trial.nrig;++camera){auto tv=camera_view(trial,camera);auto cl=camera_linear_view(l,trial.nr,camera);
    bad|=!valid_installation(tv,cl.mount_side);
    if(cl.rig_directions){bad|=std::abs(tv.rig[37])>cl.max_readout+camera_envelope_tolerance();
    for(size_t k=0;k<trial.nr;++k){const double eps=camera_envelope_tolerance()*(1+std::abs(tv.rig[k]));
        bad|=!finite_scalar(tv.rig[k])||tv.rig[k]<cl.rig_lower[k]-eps||tv.rig[k]>cl.rig_upper[k]+eps;}}
    }
    if(bad){s.scalars[8]+=1;s.scalars[9]=0.;}
}
__global__ void lm_retract_scene(View accepted,View trial,LinearView l,SolverView s){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=accepted.nscene)return;
    double value=scene_chart_retract_delta(accepted.scene_values,l.scene_chart,k,[&](size_t row){return direction_row_dot(l.scene_steps,l.q,row,l.scene_rows,s.step,s.scalars[3]);});const_cast<double*>(trial.scene_values)[k]=value;
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
    __shared__ PrintConstraint values[128];__shared__ size_t indices[128];
    int t=threadIdx.x;PrintConstraint result{};size_t index=s.print_count;
    for(size_t i=t;i<s.print_count;i+=blockDim.x)if(s.print_candidates[i].margin<result.margin){result=s.print_candidates[i];index=i;}
    values[t]=result;indices[t]=index;__syncthreads();
    for(int d=blockDim.x/2;d;d/=2){if(t<d&&(values[t+d].margin<values[t].margin||
        (values[t+d].margin==values[t].margin&&indices[t+d]<indices[t]))){values[t]=values[t+d];indices[t]=indices[t+d];}__syncthreads();}
    if(t==0)*s.print_choice=values[0];
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
    for(size_t camera=0;camera<trial.nrig;++camera){auto tv=camera_view(trial,camera);auto cl=camera_linear_view(l,trial.nr,camera);if(!cl.mount_side)continue;
    for(int kind=0;kind<4;++kind){if(kind==2&&(cl.mount_side==2||cl.mount_side==3))continue;if(kind==3&&cl.mount_side==1)continue;
        double margin=mount_constraint_value(tv,cl,s,kind,-1).a;if(!finite_scalar(margin)||margin<result.margin)result={margin,kind,int(camera),cl.mount_side};}
    }
    *s.mount_choice=result;
}
__global__ void lm_mount_gradient(View trial,LinearView l,SolverView s){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=l.q)return;
    auto camera=size_t(s.mount_choice->camera);auto tv=camera_view(trial,camera);auto cl=camera_linear_view(l,trial.nr,camera);
    auto value=mount_constraint_value(tv,cl,s,s.mount_choice->kind,j);
    s.envelope_gradient[j]=s.retained_blocked[j]?0.:value.v;if(j==0)s.scratch[0]=value.a;
}
__global__ void lm_scene_domain(View trial,LinearView l,SolverView s){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;bool bad=false;
    if(i<trial.nscene){double value=trial.scene_values[i];bad|=!finite_scalar(value)||value<l.scene_lower[i]||value>l.scene_upper[i];}
    if(i==0){const double* b=trial.board;bad|=!BoardShape{b[0],b[1],b[2],b[3],b[4],b[5]}.valid();
        if(trial.options&64u){double params[6];int kind;bad|=shared_entity_definition(b,6,trial.layout,trial.nl,100,kind,params,6,trial.options)!=Status::ok;}}
    if(i<trial.npaths){auto path=trial.scene_paths[i];
        if(path.type==1){double p[8];int kind;bad|=shared_entity_definition(trial.board,6,trial.layout,trial.nl,path.code,kind,p,(trial.nl>=35&&(path.code==410||path.code==411))?8:6,trial.options,false)!=Status::ok;}
        if(path.type==2)bad|=s.print_domain_invalid[i]!=0;
    }
    if(bad){atomicAdd(s.scalars+8,1.);atomic_min_nonnegative(s.scalars+9,0.);}
}
// The exact same precompiled critical stations, independently checked on GPU.
// One failure bit per path retains the old diagnostic counting convention.
__global__ void lm_print_domain(View trial,SolverView s){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=s.print_count)return;
    auto probe=s.print_probes[i];auto path=trial.scene_paths[probe.path];Vec2Value<LocalJet> chart{},material{};Vec3 xyz{},tangent{};
    bool good=scene_print_chart(path,trial.scene_data,trial.scene_values,nullptr,0,0,LocalJet(probe.station),chart,&material)==Status::ok;
    if(good&&path.code==1){double width=exp(trial.scene_values[path.size]),height=exp(trial.scene_values[path.size+1]),eps=camera_envelope_tolerance();
        good=width*material.x.a>=-eps&&width*(1.-material.x.a)>=-eps&&height*material.y.a>=-eps&&height*(1.-material.y.a)>=-eps;}
    if(good)good=trial.course->point(chart.x.a,chart.y.a,xyz,tangent)==Status::ok;
    if(!good)atomicExch(s.print_domain_invalid+probe.path,1);
}
// A curve station that leaves declared temporal support can be initialized
// at the observed-row pose. This is only a recovery proposal: the unchanged
// implicit forward and original objective decide acceptance, with no UV clamp.
__global__ void lm_restore_station_domain(View trial,LinearView l,SolverView s){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=s.nlocal||s.lower[id]==s.upper[id])return;
    const int row=s.representatives[id];
    trial=row_camera_view(trial,size_t(row));
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
    int ci=blockIdx.x;auto c=l.components[ci];double value=0.;
    if(c.normal_rank==2){
        double a=0.,b=0.;
        for(int j=threadIdx.x;j<c.dim;j+=blockDim.x){const double d=lm_component_step(l,s,c,j);
            value-=g[c.g+j]*d;a+=h[c.packed_h+j]*d;b+=h[c.packed_h+(c.normal_stride?c.normal_stride:c.dim)+j]*d;}
        value=block_reduce<0>(value);a=block_reduce<0>(a);b=block_reduce<0>(b);
        if(!threadIdx.x)s.parts[ci]=value-.5*(a*a+b*b);return;
    }
    size_t square=size_t(c.dim)*c.dim;
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
// Candidate likelihood needs residuals only, never a second Jacobian-sized array.
// Damping changes the candidate, not the objective or its prediction.
// Measure its energy before nonlinear station profiling changes local steps.
__global__ void lm_damping_energy(LinearView l,SolverView s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;double value=0.;
    if(i<l.q+s.nlocal)value=s.scalars[0]*s.damping[i]*s.applied[i]*s.applied[i];
    value=block_reduce<0>(value);if(!threadIdx.x)atomicAdd(s.scalars+15,value);
}
__global__ void lm_trial_whiten(LinearView l,const double* output,const size_t* offsets,double* white){
    int gi=blockIdx.x;auto g=l.groups[gi];for(int k=threadIdx.x;k<g.rank;k+=blockDim.x)
        white[offsets[gi]+k]=whiten_element(l,gi,k,0,output,nullptr);
}
__global__ void lm_trial_loss(LinearView l,const double* output,const size_t* offsets,const double* white,double* stats){
    int gi=blockIdx.x*blockDim.x+threadIdx.x;if(gi>=l.ng)return;auto g=l.groups[gi];double squared=0.;
    if(l.pixel_residuals){const double* r=output+11*size_t(l.members[g.begin])+9;squared=r[0]*r[0]+r[1]*r[1];}
    else for(int k=0;k<g.rank;++k){double r=white[offsets[gi]+k];squared+=r*r;}
    double weight=1.,rho=squared;
    if(!g.rank){weight=0.;rho=0.;}
    else if(!l.pixel_residuals&&squared>g.rank*l.delta*l.delta){double rms=sqrt(squared/g.rank);weight=l.delta/rms;rho=g.rank*(2*l.delta*rms-l.delta*l.delta);}
    stats[3*size_t(gi)]=squared;stats[3*size_t(gi)+1]=weight;stats[3*size_t(gi)+2]=.5*rho;
}
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
// Original objective gradient; shared entries are projected jointly below.
__global__ void lm_stationarity_start(LinearView l,SolverView s,const double* shared){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<l.q)s.residual[i]=shared[l.nnz+i];
}

__global__ void lm_stationarity(View v,LinearView l,SolverView s,const double* g,const double* shared,const double* stats){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double maximum=0.,cost=0.,bad=0.;
    if(i<size_t(l.q)){double dg=s.residual[i];maximum=fabs(dg);bad+=!finite_scalar(dg);}
    if(i<size_t(s.nlocal)){int row=s.representatives[i];double x=v.stations[row],dg=g[s.local_gradient[i]],eps=64*std::numeric_limits<double>::epsilon()*(1+fabs(x));
        double lo=s.lower[i],hi=s.upper[i],piece_lo,piece_hi;bool corner;
        if(station_piece(v,row,x,piece_lo,piece_hi,corner)){lo=fmax(lo,piece_lo);hi=fmin(hi,piece_hi);}
        int side=l.local_sides?l.local_sides[i]:0;
        bool fixed=lo==hi||side==2||((x<=lo+eps||side==1)&&dg>0)||((x>=hi-eps||side==-1)&&dg<0);
        if(!fixed)maximum=fmax(maximum,fabs(dg));bad+=!finite_scalar(dg);}
    if(i<size_t(l.ng)){cost=stats[3*i+2];bad+=!finite_scalar(cost);}
    maximum=block_reduce<1>(maximum);cost=block_reduce<0>(cost);bad=block_reduce<0>(bad);
    if(!threadIdx.x){atomicAdd(s.scalars,cost);atomic_max_nonnegative(s.scalars+1,maximum);atomicAdd(s.scalars+2,bad);}
}
struct DenseBatch {int dimension;size_t factor_begin,factor_count,solve_begin,solve_count;};
#include "gpu_station_refinement.cuh"
#include "gpu_stationarity.cuh"
#include "gpu_bounded_quadratic.cuh"
struct Solver {
    StationRefinement station_refinement;
    DeviceBuffer<StationarityFace> stationarity_faces;
    DeviceBuffer<int> stationarity_cell_counts,stationarity_print_counts;
    DeviceBuffer<int> stationarity_count;
    DeviceBuffer<double> stationarity_matrix,stationarity_dual;
    DeviceBuffer<double> quadratic_work;
    DeviceBuffer<size_t> trial_offsets;
    bool has_active_profiles=false;
    // Separate from residual: linear solves overwrite that scratch on every trial.
    DeviceBuffer<double> stationarity_descent,stationarity_metrics;
    bool stationarity_valid=false,stationarity_projection_complete=true,stationarity_bound_only=false;size_t stationarity_generation=0;
    double stationarity_tolerance=0.,stationarity_state_values[3]{};
    std::string stationarity_backend_key;
    double stationarity_wall_ms=0.,stationarity_faces_ms=0.,stationarity_cone_ms=0.,stationarity_readback_ms=0.;
    size_t stationarity_recomputations=0,stationarity_cache_hits=0,qp_line_evaluations=0,profile_restoration_calls=0;
    int stationarity_sweeps=-1;
    double qp_line_wall_ms=0.,trial_station_wall_ms=0.,prediction_wall_ms=0.;
    double last_damping_energy=0.,last_prediction=0.;size_t damping_recoveries=0;
    // The same small-dual algorithm can execute in the native controller;
    // GPU single-thread pair sweeps are inefficient. Capacity grows as needed.
    // Geometry, Jacobians and Schur remain GPU.
    std::vector<double> dual_host_gram,dual_host_target,dual_host_values,dual_host_equalities;
    CoupledDualCache dual_cache;
    bool dual_trace_started=false;
    DeviceBuffer<int> print_domain_invalid;
    int stationarity_active_count=0;
    rs_lm_numeric_progress numeric_progress=nullptr;void* numeric_context=nullptr;
    size_t envelope_projections=0,envelope_backtracks=0,envelope_downloads=0,print_downloads=0,mount_downloads=0,retained_downloads=0,shared_projection_downloads=0;
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
    cudssMatrix_t A=nullptr,X=nullptr,B=nullptr,envelope_X=nullptr,envelope_B=nullptr;cudaEvent_t begin=nullptr,end=nullptr,objective_events[7]{},prediction_begin=nullptr,prediction_end=nullptr;double objective_ms[6]{};
    double lambda=0.,nu=2.,initial_relative=.001,eta=.0001,linear_tolerance=1e-8;
    double device_budget=0.,host_budget=0.;int m=0,analyses=0,trials=0,accepted=0,numerical_rejections=0;
    int64_t estimates[16]{},factor_nnz=0;bool trial_available=false,analysis_ready=false,jacobi_damping=false,image_station_damping=false,image_global_damping=false,profile_trial_stations=false;
    ~Solver(){if(prediction_begin)cudaEventDestroy(prediction_begin);if(prediction_end)cudaEventDestroy(prediction_end);for(auto event:objective_events)if(event)cudaEventDestroy(event);if(A)cudssMatrixDestroy(A);if(X)cudssMatrixDestroy(X);if(B)cudssMatrixDestroy(B);if(envelope_X)cudssMatrixDestroy(envelope_X);if(envelope_B)cudssMatrixDestroy(envelope_B);if(data)cudssDataDestroy(sparse,data);if(config)cudssConfigDestroy(config);if(sparse)cudssDestroy(sparse);if(dense)cusolverDnDestroy(dense);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
    size_t device_bytes()const{return stationarity_print_counts.bytes()+stationarity_cell_counts.bytes()+stationarity_descent.bytes()+stationarity_metrics.bytes()+trial_offsets.bytes()+print_domain_invalid.bytes()+quadratic_work.bytes()+stationarity_faces.bytes()+stationarity_count.bytes()+stationarity_matrix.bytes()+stationarity_dual.bytes()+retained_choice.bytes()+coupled_storage.bytes()+mount_choice.bytes()+print_probes.bytes()+print_candidates.bytes()+print_choice.bytes()+print_directions.bytes()+working_set_anchor.bytes()+station_refinement.bytes()+envelope_candidates.bytes()+envelope_choice.bytes()+envelope_gradient.bytes()+envelope_delta.bytes()+envelope_response.bytes()+retained_lower.bytes()+retained_upper.bytes()+retained_delta.bytes()+rig_bound_columns.bytes()+retained_blocked.bytes()+delta_lower.bytes()+delta_upper.bytes()+active_delta.bytes()+image_metric.bytes()+damping.bytes()+trial_course.bytes()+lower.bytes()+upper.bytes()+matrix.bytes()+solved.bytes()+rhs.bytes()+csr.bytes()+step.bytes()+applied.bytes()+scalars.bytes()+parts.bytes()+trial_input.bytes()+trial_output.bytes()+trial_white.bytes()+trial_stats.bytes()+residual.bytes()+scratch.bytes()+locals.bytes()+representatives.bytes()+blocked.bytes()+factor_info.bytes()+solve_info.bytes()+check_info.bytes()+trial_status.bytes()+local_gradient.bytes()+workspace.bytes()+factor_ptrs.bytes()+solve_a_ptrs.bytes()+solve_b_ptrs.bytes();}
    void prepare(const View& v,Linearization& l,size_t input_count,const double* lo,const double* hi,size_t nlocal,const double* options,size_t noptions,cudaStream_t stream){
        if((noptions!=6&&noptions!=7&&noptions!=8)||(noptions>=7&&options[6]!=0&&options[6]!=1&&options[6]!=2&&options[6]!=3)||(noptions==8&&options[7]!=0&&options[7]!=1)||nlocal!=l.scales.size()||(nlocal&&(!lo||!hi))||!finite_array(options,noptions)||options[0]<=0||options[1]<0||options[2]<=0||options[2]>=1||options[3]<=0||options[4]<=0||options[5]<=0)throw std::invalid_argument("invalid G3 bounds/options");
        jacobi_damping=noptions>=7&&options[6]!=0;image_station_damping=noptions>=7&&options[6]>=2;image_global_damping=noptions>=7&&options[6]==3;profile_trial_stations=noptions==8&&options[7]!=0;
        if(profile_trial_stations&&!l.host.pixel_residuals)throw std::invalid_argument("trial station profiling requires independent raw pixel factors");
        initial_relative=options[0];lambda=options[1];eta=options[2];linear_tolerance=options[3];device_budget=options[4];host_budget=options[5];
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
        has_active_profiles=v.profile_enabled&&v.profile_ids&&std::any_of(v.profile_ids,v.profile_ids+v.n,[](int id){return id>=0;});
        std::vector<size_t> compact_offsets;size_t trial_residual_count=0;
        if(!l.host.pixel_residuals)for(auto g:l.groups){compact_offsets.push_back(trial_residual_count);trial_residual_count+=g.rank;}
        double planned=sizeof(size_t)*double(compact_offsets.size())+8.*(double(nm)+nsolved+2.*q+l.host.nnz+3.*(q+nlocal)+4.*q+26+cc+input_count+11.*v.n+trial_residual_count+3.*l.host.ng+2.*nlocal+4.*nlocal)+
            4.*(3.*nlocal+nfactor+3+v.n+q+v.nr*v.nrig)+sizeof(LocalWorkspace)*double(cc)+8.*(nfactor+2.*nsolve);
        if(l.host.envelope.enabled)planned+=sizeof(CameraConstraint)*double(v.nc-2);
        const size_t coupled_count=coupled_storage_size(q,coupled.capacity);
        if(l.host.envelope.enabled||l.host.rig_directions||l.host.scene_directions||l.host.mount_side)planned+=8.*(2*q+27*v.nrig+double(coupled_count));
        if(l.host.rig_directions||l.host.scene_directions)planned+=sizeof(RetainedConstraint);
        if(l.host.scene_directions)planned+=sizeof(PrintConstraint)*double(probes.size()+1)+sizeof(PrintProbe)*double(probes.size())+8.*checked_product(v.nscene,size_t(q));
        if(l.host.mount_side)planned+=sizeof(MountConstraint);
        // At most 220 camera faces per cell: five attitude inequalities times
        // 6x6 tied signed rate branches, plus all optical/derivative extrema.
        const size_t face_capacity=2*v.nr*v.nrig+2*v.nscene+17*v.nrig+((v.options&64u)?ice_tolerance_margin_count:0)+6*probes.size()+(l.host.envelope.enabled?220*(v.nc-3)*v.nrig:0);
        const size_t stationarity_cells=(l.host.envelope.enabled||l.host.mount_side)?(v.nc-3)*v.nrig:0;
        planned+=sizeof(StationarityFace)*double(face_capacity)+(2+3*stationarity_cells+3*probes.size())*sizeof(int);
        planned+=4.*(q+nlocal)+4*sizeof(double)+sizeof(int)*v.npaths+8.*(q+8);
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));if(planned>device_budget||planned>free)throw std::runtime_error("ragged local/sparse LM workspace needs "+std::to_string(planned)+" bytes; exceeds current device budget/available memory");
        lower.upload(std::vector<double>(lo,lo+nlocal),stream);upper.upload(std::vector<double>(hi,hi+nlocal),stream);locals.upload(l.component_locals,stream);representatives.upload(reps,stream);local_gradient.upload(gradient,stream);workspace.upload(work,stream);blocked.allocate(nlocal);
        working_set_anchor.allocate(size_t(q)+nlocal);
        std::vector<int> rig_single(v.nr*v.nrig,-1);if(!l.rig_directions.empty())for(size_t camera=0;camera<v.nrig;++camera)for(int k=15;k<int(v.nr);++k){
            if(k==24||k==25||k==38)continue;int chart=k<24?k-9:(k<38?k-11:k-12),found=-1;
            for(int col=0;col<q;++col)if(l.rig_directions[(camera*(v.nr==43?31:27)+size_t(chart))*q+col]!=0.){if(found>=0){found=-2;break;}found=col;}
            if(found>=0)rig_single[camera*v.nr+k]=found;
        }
        rig_bound_columns.upload(rig_single,stream);retained_blocked.allocate(q);retained_lower.allocate(q);retained_upper.allocate(q);retained_delta.allocate(q);
        delta_lower.allocate(nlocal);delta_upper.allocate(nlocal);active_delta.allocate(nlocal);damping.allocate(q+nlocal);image_metric.allocate(q);matrix.allocate(nm);solved.allocate(nsolved);rhs.allocate(q);csr.allocate(l.host.nnz);step.allocate(q+nlocal);applied.allocate(q+nlocal);scalars.allocate(24);parts.allocate(cc);residual.allocate(q);scratch.allocate(2);
        if(v.scene_paths)trial_course.allocate(1);
        trial_input.allocate(input_count);trial_output.allocate(v.n*11);trial_status.allocate(v.n);trial_white.allocate(trial_residual_count);trial_offsets.upload(compact_offsets,stream);trial_stats.allocate(3*size_t(l.host.ng));factor_info.allocate(nfactor);solve_info.allocate(1);check_info.allocate(2);
        std::vector<double*> ap,aa,bb;ap.reserve(nfactor);aa.reserve(nsolve);bb.reserve(nsolve);
        for(auto& item:bucket){DenseBatch batch{item.first,ap.size(),item.second.size(),aa.size(),0};
            for(int ci:item.second){auto c=l.components[ci];auto w=work[ci];ap.push_back(matrix.data+w.matrix);for(int j=0;j<=c.q_count;++j){aa.push_back(ap.back());bb.push_back(solved.data+w.solved+size_t(j)*c.local_count);}}
            batch.solve_count=aa.size()-batch.solve_begin;batches.push_back(batch);}
        factor_ptrs.upload(ap,stream);solve_a_ptrs.upload(aa,stream);solve_b_ptrs.upload(bb,stream);
        view={locals.data,representatives.data,local_gradient.data,workspace.data,lower.data,upper.data,blocked.data,matrix.data,solved.data,rhs.data,csr.data,step.data,applied.data,scalars.data,parts.data,residual.data,scratch.data,damping.data,image_metric.data,int(nlocal),delta_lower.data,delta_upper.data,active_delta.data,rig_bound_columns.data,retained_blocked.data,retained_lower.data,retained_upper.data,retained_delta.data};
        if(l.host.envelope.enabled){envelope_candidates.allocate(v.nc-3);envelope_choice.allocate(1);
            view.envelope_candidates=envelope_candidates.data;view.envelope_choice=envelope_choice.data;}
        if(l.host.rig_directions||l.host.scene_directions)retained_choice.allocate(1);
        if(l.host.envelope.enabled||l.host.rig_directions||l.host.scene_directions||l.host.mount_side){envelope_gradient.allocate(q);envelope_delta.allocate(27*v.nrig);envelope_response.allocate(q);
            view.envelope_gradient=envelope_gradient.data;view.envelope_delta=envelope_delta.data;view.envelope_response=envelope_response.data;
            coupled_storage.allocate(coupled_count);coupled=coupled_view(coupled_storage.data,q,coupled.capacity);}
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
        stationarity_faces.allocate(face_capacity);stationarity_count.allocate(2);
        stationarity_cell_counts.allocate(3*stationarity_cells);stationarity_print_counts.allocate(3*probes.size());
        quadratic_work.allocate(4);stationarity_descent.allocate(q);stationarity_metrics.allocate(8);
        print_domain_invalid.allocate(v.npaths);view.print_domain_invalid=print_domain_invalid.data;
        cuda_check(cudaEventCreate(&prediction_begin));cuda_check(cudaEventCreate(&prediction_end));
        cuda_check(cudaEventCreate(&begin));cuda_check(cudaEventCreate(&end));for(auto& event:objective_events)cuda_check(cudaEventCreate(&event));cuda_check(cudaStreamSynchronize(stream));
    }
    void grow_coupled(int q,int used,cudaStream_t stream){
        if(used<coupled.capacity)return;
        if(coupled.capacity>std::numeric_limits<int>::max()/2)throw std::runtime_error("constraint cut capacity integer overflow");
        const int capacity=2*coupled.capacity;const size_t count=coupled_storage_size(q,capacity),bytes=checked_product(count,sizeof(double));
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        // Both old and new buffers coexist until the exact state copy finishes.
        if(bytes>free||double(device_bytes())+bytes>device_budget)
            throw std::runtime_error("constraint cut growth exceeds current device budget/free memory: cuts="+std::to_string(used)+", new_capacity="+std::to_string(capacity)+", additional_bytes="+std::to_string(bytes));
        if(double(size_t(capacity)*(size_t(capacity)+3)*sizeof(double))>host_budget)
            throw std::runtime_error("constraint dual growth exceeds current host budget");
        DeviceBuffer<double> replacement;replacement.allocate(count);auto next=coupled_view(replacement.data,q,capacity);
        auto copy=[&](double* to,const double* from,size_t n){if(n)cuda_check(cudaMemcpyAsync(to,from,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));};
        copy(next.base,coupled.base,q);copy(next.anchor,coupled.anchor,q);
        copy(next.gradients,coupled.gradients,size_t(used)*q);copy(next.responses,coupled.responses,size_t(used)*q);
        if(used)cuda_check(cudaMemcpy2DAsync(next.gram,size_t(capacity)*sizeof(double),coupled.gram,size_t(coupled.capacity)*sizeof(double),size_t(used)*sizeof(double),used,cudaMemcpyDeviceToDevice,stream));
        copy(next.target,coupled.target,used);copy(next.dual,coupled.dual,used);copy(next.equality,coupled.equality,used);
        cuda_check(cudaStreamSynchronize(stream));
        if(std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS"))std::fprintf(stderr,"LM_CUT_STORAGE_GROW previous=%d capacity=%d retained=%d\n",coupled.capacity,capacity,used);
        std::swap(replacement.data,coupled_storage.data);std::swap(replacement.count,coupled_storage.count);coupled=next;
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
        lm_camera_envelope_pick<<<1,128,0,s>>>(v,view);cuda_check(cudaGetLastError());
    }
    size_t project_stationarity(View v,LinearView l,const double* shared,cudaStream_t s,double tolerance,const std::atomic<bool>& stop,bool norm_only){
        stationarity_projection_complete=true;stationarity_bound_only=false;
        const auto faces_started=Clock::now();stationarity_sweeps=-1;
        if(l.scene_directions)lm_print_chart_directions<<<(l.q+127)/128,128,0,s>>>(v,l,view);
        const char* face_backend=std::getenv("RINKSPACE_STATIONARITY_FACE_BACKEND");
        if(face_backend&&std::strcmp(face_backend,"serial")&&std::strcmp(face_backend,"parallel"))throw std::invalid_argument("RINKSPACE_STATIONARITY_FACE_BACKEND must be serial or parallel");
        if(face_backend&&std::strcmp(face_backend,"serial")==0)
            lm_stationarity_faces<<<1,1,0,s>>>(v,l,view,stationarity_faces.data,stationarity_faces.count,stationarity_count.data);
        else{
            lm_stationarity_fixed_faces<<<1,1,0,s>>>(v,l,view,stationarity_faces.data,stationarity_faces.count,stationarity_count.data);
            if(view.print_count){
                lm_stationarity_print_faces<<<unsigned((view.print_count+127)/128),128,0,s>>>(v,l,view,nullptr,0,stationarity_print_counts.data,stationarity_count.data);
                lm_stationarity_face_offsets<<<1,1,0,s>>>(int(view.print_count),stationarity_print_counts.data,stationarity_faces.count,stationarity_count.data);
                lm_stationarity_print_faces<<<unsigned((view.print_count+127)/128),128,0,s>>>(v,l,view,stationarity_faces.data,stationarity_faces.count,stationarity_print_counts.data,stationarity_count.data);
            }
            const int cells=int(stationarity_cell_counts.count/3);
            if(cells){
                lm_stationarity_cell_faces<<<(cells+127)/128,128,0,s>>>(v,l,view,nullptr,0,stationarity_cell_counts.data,stationarity_count.data);
                lm_stationarity_face_offsets<<<1,1,0,s>>>(cells,stationarity_cell_counts.data,stationarity_faces.count,stationarity_count.data);
                lm_stationarity_cell_faces<<<(cells+127)/128,128,0,s>>>(v,l,view,stationarity_faces.data,stationarity_faces.count,stationarity_cell_counts.data,stationarity_count.data);
            }
        }
        cuda_check(cudaGetLastError());int counts[2];
        cuda_check(cudaMemcpyAsync(counts,stationarity_count.data,sizeof(counts),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));
        stationarity_active_count=counts[0];stationarity_faces_ms+=millis(faces_started);
        if(stop.load())return sizeof(counts);
        if(counts[1]){
            // Keep the rejected initial state diagnosable. A generic failure
            // hides whether this is a rig, print, or scene-domain violation.
            std::vector<StationarityFace> failed_faces(static_cast<size_t>(counts[0]));
            if(counts[0])cuda_check(cudaMemcpyAsync(failed_faces.data(),stationarity_faces.data,failed_faces.size()*sizeof(StationarityFace),cudaMemcpyDeviceToHost,s));
            cuda_check(cudaStreamSynchronize(s));
            auto worst=std::min_element(failed_faces.begin(),failed_faces.end(),[](const auto& a,const auto& b){return a.margin<b.margin;});
            std::string detail="accepted state has an invalid physical stationarity constraint";
            if(worst!=failed_faces.end())detail+=" family="+std::to_string(worst->family)+" index="+std::to_string(worst->index)+" side="+std::to_string(worst->side)+" margin="+std::to_string(worst->margin)+" camera_kind="+std::to_string(worst->camera.kind)+" cell="+std::to_string(worst->camera.cell);
            throw std::runtime_error(detail);
        }
        size_t matrix_count=checked_product(size_t(counts[0]),size_t(l.q));
        if(matrix_count>stationarity_matrix.count||2*size_t(counts[0])>stationarity_dual.count){
            size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
            const size_t old=stationarity_matrix.bytes()+stationarity_dual.bytes(),wanted=(matrix_count+2*counts[0])*sizeof(double);
            if(wanted>free+old||double(device_bytes()-old+wanted)>device_budget)throw std::runtime_error("physical stationarity workspace exceeds current device budget/available memory: faces="+std::to_string(counts[0])+", q="+std::to_string(l.q)+", wanted="+std::to_string(wanted)+", free="+std::to_string(free)+", old="+std::to_string(old)+", solver_owned="+std::to_string(device_bytes())+", budget="+std::to_string(device_budget));
            cuda_check(cudaFree(stationarity_matrix.data));stationarity_matrix.data=nullptr;stationarity_matrix.count=0;
            cuda_check(cudaFree(stationarity_dual.data));stationarity_dual.data=nullptr;stationarity_dual.count=0;
            stationarity_matrix.allocate(matrix_count);stationarity_dual.allocate(2*size_t(counts[0]));
        }
        const auto cone_started=Clock::now();
        if(matrix_count)lm_stationarity_directions<<<unsigned((matrix_count+127)/128),128,0,s>>>(v,l,view,stationarity_faces.data,counts[0],stationarity_matrix.data);
        const char* cone_backend=std::getenv("RINKSPACE_STATIONARITY_BACKEND");
        if(cone_backend&&std::strcmp(cone_backend,"cpu")&&std::strcmp(cone_backend,"gpu"))
            throw std::invalid_argument("RINKSPACE_STATIONARITY_BACKEND must be cpu or gpu");
        if(cone_backend&&std::strcmp(cone_backend,"cpu")==0){
            // Identical FP64 cone projection, explicitly selected. Its serial
            // pair sweeps run poorly in one CUDA thread and cannot see stop.
            // Only the small active constraint matrix leaves the device.
            const size_t n=size_t(counts[0]),face_bytes=n*sizeof(StationarityFace);
            const size_t workspace_bytes=face_bytes+(matrix_count+2*n+2*size_t(l.q))*sizeof(double);
            if(double(workspace_bytes)>host_budget)throw std::runtime_error("stationarity host workspace exceeds declared budget");
            std::vector<StationarityFace> faces(n);
            std::vector<double> a(matrix_count),g(l.q),d(l.q),dual(2*n);
            if(n){cuda_check(cudaMemcpyAsync(faces.data(),stationarity_faces.data,face_bytes,cudaMemcpyDeviceToHost,s));
                cuda_check(cudaMemcpyAsync(a.data(),stationarity_matrix.data,matrix_count*sizeof(double),cudaMemcpyDeviceToHost,s));}
            cuda_check(cudaMemcpyAsync(g.data(),shared+l.nnz,l.q*sizeof(double),cudaMemcpyDeviceToHost,s));
            cuda_check(cudaStreamSynchronize(s));
            if(stop.load())throw SolverStopped{};
            const char* sparse_option=std::getenv("RINKSPACE_STATIONARITY_CPU_SPARSE");
            if(sparse_option&&std::strcmp(sparse_option,"0")&&std::strcmp(sparse_option,"1"))throw std::invalid_argument("RINKSPACE_STATIONARITY_CPU_SPARSE must be 0 or 1");
            std::vector<int> offsets,columns;
            if(!sparse_option||std::strcmp(sparse_option,"0")){
                size_t nonzeros=0;for(double value:a)nonzeros+=value!=0.;
                // Dense execution remains available when the declared host
                // budget has no room for this optional exact sparsity index.
                if(double(workspace_bytes+(n+1+nonzeros)*sizeof(int))<=host_budget){
                    Linearization::bounded(nonzeros);offsets.reserve(n+1);columns.reserve(nonzeros);offsets.push_back(0);
                    for(size_t i=0;i<n;++i){for(int j=0;j<l.q;++j)if(a[i*l.q+j]!=0.)columns.push_back(j);offsets.push_back(int(columns.size()));}
                }
            }
            const bool trace=std::getenv("RINKSPACE_TRACE_STATIONARITY")!=nullptr;
            auto started=Clock::now();
            if(trace){std::fprintf(stderr,"LM_STATIONARITY phase=begin backend=cpu q=%d faces=%d\n",l.q,counts[0]);std::fflush(stderr);}
            double scale=1.;for(double value:g)scale=fmax(scale,fabs(value));
            const double tol=fmin(.1*tolerance,128*std::numeric_limits<double>::epsilon()*scale);
            // Opt-in first-input capture for a reproducible CPU cone diagnosis.
            // This does not change the objective, cone, or solve tolerance.
            if(const char* dump=std::getenv("RINKSPACE_TRACE_STATIONARITY_INPUT")){
                static bool captured=false;
                if(!captured){FILE* file=std::fopen(dump,"wb");if(!file)throw std::runtime_error("cannot open stationarity diagnostic input");
                    int header[2]={l.q,counts[0]};bool good=std::fwrite(header,sizeof(int),2,file)==2;
                    good&=std::fwrite(&tol,sizeof(double),1,file)==1;
                    good&=std::fwrite(g.data(),sizeof(double),g.size(),file)==g.size();
                    good&=std::fwrite(a.data(),sizeof(double),a.size(),file)==a.size();
                    for(const auto& face:faces){int equality=face.equality;good&=std::fwrite(&equality,sizeof(int),1,file)==1;}
                    good&=std::fclose(file)==0;if(!good)throw std::runtime_error("incomplete stationarity diagnostic input");captured=true;
                }
            }
            double certificate=0.;int sweeps=0;
            const int free_certificate=norm_only?stationarity_free_coordinate_certificate(l.q,counts[0],g.data(),a.data(),tolerance,certificate,d.data()):0;
            if(free_certificate){
                stationarity_projection_complete=false;stationarity_bound_only=free_certificate==2;stationarity_sweeps=0;
                cuda_check(cudaMemcpyAsync(view.residual,d.data(),l.q*sizeof(double),cudaMemcpyHostToDevice,s));
                lm_stationarity_host_result<<<1,1,0,s>>>(view,true,certificate);
                cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));stationarity_cone_ms+=millis(cone_started);
                if(trace){std::fprintf(stderr,"LM_STATIONARITY phase=end method=%s q=%d faces=%d certificate=%.17g elapsed_ms=%.3f\n",stationarity_bound_only?"strict_nonconvergence_lower_bound":"exact_free_coordinate_norm",l.q,counts[0],certificate,millis(started));std::fflush(stderr);}
                return sizeof(counts)+face_bytes+(matrix_count+size_t(l.q))*sizeof(double);
            }
            const size_t remaining_budget=size_t(fmax(0.,host_budget-double(workspace_bytes+(offsets.size()+columns.size())*sizeof(int))));
            const bool valid=stationarity_cone_interruptible(l.q,counts[0],g.data(),a.data(),dual.data(),d.data(),
                [&](int i){return faces[i].equality;},tol,certificate,n?dual.data()+n:nullptr,
                [&]{return stop.load();},&sweeps,offsets.empty()?nullptr:offsets.data(),columns.data(),remaining_budget);
            if(trace){std::fprintf(stderr,"LM_STATIONARITY phase=end backend=cpu q=%d faces=%d sweeps=%d valid=%d certificate=%.17g elapsed_ms=%.3f\n",l.q,counts[0],sweeps,int(valid),certificate,millis(started));std::fflush(stderr);}
            stationarity_sweeps=sweeps;
            if(stop.load())throw SolverStopped{};
            if(n){cuda_check(cudaMemcpyAsync(stationarity_matrix.data,a.data(),matrix_count*sizeof(double),cudaMemcpyHostToDevice,s));
                cuda_check(cudaMemcpyAsync(stationarity_dual.data,dual.data(),2*n*sizeof(double),cudaMemcpyHostToDevice,s));}
            cuda_check(cudaMemcpyAsync(view.residual,d.data(),l.q*sizeof(double),cudaMemcpyHostToDevice,s));
            lm_stationarity_host_result<<<1,1,0,s>>>(view,valid,certificate);
            cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));
            stationarity_cone_ms+=millis(cone_started);
            return sizeof(counts)+face_bytes+(matrix_count+size_t(l.q))*sizeof(double);
        }
        lm_stationarity_project<<<1,1,0,s>>>(l,view,stationarity_faces.data,counts[0],shared,stationarity_matrix.data,stationarity_dual.data,tolerance);
        cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));stationarity_cone_ms+=millis(cone_started);
        return sizeof(counts);
    }
    size_t stationarity_state(View v,Linearization& l,cudaStream_t s,double tolerance,const std::atomic<bool>& stop,double* state,bool norm_only=false){
        const auto started=Clock::now();if(stop.load())throw SolverStopped{};
        const char* cache_option=std::getenv("RINKSPACE_STATIONARITY_CACHE");
        if(cache_option&&std::strcmp(cache_option,"0")&&std::strcmp(cache_option,"1"))throw std::invalid_argument("RINKSPACE_STATIONARITY_CACHE must be 0 or 1");
        const char* backend=std::getenv("RINKSPACE_STATIONARITY_BACKEND");const char* faces=std::getenv("RINKSPACE_STATIONARITY_FACE_BACKEND");
        const char* sparse_option=std::getenv("RINKSPACE_STATIONARITY_CPU_SPARSE");
        const std::string key=std::string(backend?backend:"gpu")+"/"+(faces?faces:"parallel")+"/"+(sparse_option?sparse_option:"1");
        if((!cache_option||std::strcmp(cache_option,"0"))&&stationarity_valid&&l.gpu_ready&&stationarity_generation==l.generation&&stationarity_tolerance==tolerance&&stationarity_backend_key==key&&(norm_only||stationarity_projection_complete)){
            std::copy_n(stationarity_state_values,3,state);
            cuda_check(cudaMemcpyAsync(residual.data,stationarity_descent.data,residual.bytes(),cudaMemcpyDeviceToDevice,s));
            ++stationarity_cache_hits;stationarity_wall_ms+=millis(started);return 0;
        }
        stationarity_valid=false;++stationarity_recomputations;
        cuda_check(cudaMemsetAsync(scalars.data,0,3*sizeof(double),s));
        int count=std::max({l.host.q,view.nlocal,l.host.ng});
        lm_stationarity_start<<<(count+127)/128,128,0,s>>>(l.device,view,l.d_shared.data);
        size_t copies=project_stationarity(v,l.device,l.d_shared.data,s,tolerance,stop,norm_only);
        if(stop.load())throw SolverStopped{};
        const auto readback_started=Clock::now();
        lm_stationarity<<<(count+127)/128,128,0,s>>>(v,l.device,view,l.d_g.data,l.d_shared.data,l.d_stats.data);
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(state,scalars.data,3*sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));
        copies+=3*sizeof(double);stationarity_readback_ms+=millis(readback_started);
        if(!state[2]&&finite_array(state,3)&&!stop.load()){
            cuda_check(cudaMemcpyAsync(stationarity_descent.data,residual.data,residual.bytes(),cudaMemcpyDeviceToDevice,s));
            std::copy_n(state,3,stationarity_state_values);stationarity_generation=l.generation;stationarity_tolerance=tolerance;stationarity_backend_key=key;stationarity_valid=true;
        }
        stationarity_wall_ms+=millis(started);return copies;
    }
    void project_shared_step(View v,LinearView l,cudaStream_t s,const std::atomic<bool>& stop){
        lm_project_shared_step<<<1,1,0,s>>>(v,l,view);cuda_check(cudaGetLastError());
        double failed=0.;cuda_check(cudaMemcpyAsync(&failed,scratch.data+1,sizeof(double),cudaMemcpyDeviceToHost,s));
        cuda_check(cudaStreamSynchronize(s));++shared_projection_downloads;
        if(stop.load())throw SolverStopped{};
        if(failed)throw std::runtime_error("shared linear boundary projection did not converge; accepted state preserved");
    }
    bool refine_bounded_quadratic(Linearization& l,cudaStream_t s,const std::atomic<bool>& stop,size_t& downloads){
        const auto line_started=Clock::now();
        auto evaluate=[&](double alpha,double* values){
            ++qp_line_evaluations;
            cuda_check(cudaMemsetAsync(quadratic_work.data,0,quadratic_work.bytes(),s));
            lm_qp_line_components<<<(l.host.ncomponents+127)/128,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data,alpha,quadratic_work.data);
            lm_qp_line_damping<<<(l.host.q+127)/128,128,0,s>>>(l.device,view,alpha,quadratic_work.data);
            cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(values,quadratic_work.data,3*sizeof(double),cudaMemcpyDeviceToHost,s));
            cuda_check(cudaStreamSynchronize(s));downloads+=3*sizeof(double);if(stop.load())throw SolverStopped{};
            if(!finite_array(values,3))throw std::runtime_error("nonfinite bounded quadratic line model");
        };
        double lower[3],upper[3],alpha=1.;evaluate(0.,lower);evaluate(1.,upper);
        if(lower[0]>=0.)alpha=0.;
        else if(upper[0]>0.){
            double lo=0.,hi=1.,x=1.,values[3];std::copy(upper,upper+3,values);
            // Safeguarded Newton on a monotone piecewise-linear derivative.
            // 64 brackets resolve a binary64 interval; this is not a run budget.
            for(int i=0;i<64;++i){
                if(fabs(values[0])<=64*std::numeric_limits<double>::epsilon()*std::max(1.,values[2]))break;
                if(values[0]>0.)hi=x;else lo=x;
                double next=values[1]>0.?x-values[0]/values[1]:(lo+hi)*.5;
                if(!(next>lo&&next<hi))next=(lo+hi)*.5;if(next==x||next==lo||next==hi)break;
                x=next;evaluate(x,values);
            }
            alpha=x;
        }
        cuda_check(cudaMemsetAsync(quadratic_work.data,0,quadratic_work.bytes(),s));cuda_check(cudaMemsetAsync(check_info.data,0,2*sizeof(int),s));
        lm_qp_line_apply<<<(l.host.q+127)/128,128,0,s>>>(l.device,view,alpha,quadratic_work.data);
        lm_qp_conditional_active<<<(l.host.ncomponents+127)/128,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data,check_info.data);
        int result[2];double change;cuda_check(cudaMemcpyAsync(result,check_info.data,sizeof(result),cudaMemcpyDeviceToHost,s));
        cuda_check(cudaMemcpyAsync(&change,quadratic_work.data+3,sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));downloads+=sizeof(result)+sizeof(double);
        if(stop.load())throw SolverStopped{};if(result[1])throw std::runtime_error("invalid bounded quadratic conditional minimum");
        qp_line_wall_ms+=millis(line_started);
        return result[0]==0&&(alpha==1.||change==0.);
    }
    // Restore camera and observed-print boundaries in the current joint Schur
    // metric, jointly retaining discovered boundary cuts and dual multipliers.
    bool project_envelope(View accepted,View trial,LinearView l,cudaStream_t s,const std::atomic<bool>& stop){
        if(!l.envelope.enabled&&!l.rig_directions&&!l.scene_directions&&!l.mount_side)return true;lm_camera_project_start<<<1,1,0,s>>>(view);
        cuda_check(cudaMemcpyAsync(coupled.base,view.step,l.q*sizeof(double),cudaMemcpyDeviceToDevice,s));std::vector<CoupledBoundary> boundaries;
        bool correcting=false,restoring=false,dual_unresolved=false;double anchor_margin=-INFINITY;
        for(size_t iteration=0;;++iteration){
            lm_retract_camera_controls<<<unsigned((3*accepted.nc+127)/128),128,0,s>>>(accepted,trial,l,view);
            if(l.rig_directions)lm_retract_rig<<<unsigned(accepted.nrig),32,0,s>>>(accepted,trial,l,view);
            if(l.scene_directions)lm_retract_scene<<<unsigned((accepted.nscene+127)/128),128,0,s>>>(accepted,trial,l,view);
            CameraConstraint a{};PrintConstraint p{};MountConstraint m{};RetainedConstraint b{};bool camera_ok=true;
            if(l.envelope.enabled){evaluate_envelope(trial,l,s);
                cuda_check(cudaMemcpyAsync(&a,envelope_choice.data,sizeof(a),cudaMemcpyDeviceToHost,s));++envelope_downloads;}
            if(l.scene_directions){if(view.print_count)lm_print_candidates<<<unsigned((view.print_count+127)/128),128,0,s>>>(trial,view);lm_print_pick<<<1,128,0,s>>>(trial,view);
                cuda_check(cudaMemcpyAsync(&p,print_choice.data,sizeof(p),cudaMemcpyDeviceToHost,s));++print_downloads;}
            if(l.mount_side){lm_mount_pick<<<1,1,0,s>>>(trial,l,view);cuda_check(cudaMemcpyAsync(&m,mount_choice.data,sizeof(m),cudaMemcpyDeviceToHost,s));++mount_downloads;}
            if(l.rig_directions||l.scene_directions){lm_retained_pick<<<1,1,0,s>>>(accepted,trial,l,view,retained_choice.data);cuda_check(cudaMemcpyAsync(&b,retained_choice.data,sizeof(b),cudaMemcpyDeviceToHost,s));++retained_downloads;}
            cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));if(stop.load())throw SolverStopped{};
            if(l.envelope.enabled)camera_ok=finite_scalar(a.margin)&&a.margin>=-camera_envelope_tolerance();
            bool strict_mount=m.kind==0||m.kind==1||(m.kind==3&&m.side>=4);
            bool mount_ok=m.margin>=0.&&(!strict_mount||m.margin>0.);
            if(camera_ok&&p.margin>=0.&&mount_ok&&(b.margin>=0.||(b.equality&&b.margin>=-camera_envelope_tolerance())))return true;
            // Bounded nonlinear restoration is an inner numerical attempt.
            // If it fails, reject and retry with LM damping; never replace the
            // proposed shared update by zero and accept station-only progress.
            if(iteration>=2*size_t(coupled.capacity))return false;
            int selected=!camera_ok?0:(p.margin<0?1:2);double margin=selected==0?a.margin:(selected==1?p.margin:m.margin);
            if(p.margin<margin){selected=1;margin=p.margin;}if(m.margin<margin){selected=2;margin=m.margin;}
            if(b.margin<margin){selected=3;margin=b.margin;}
            // If the small dual cannot resolve cancelling multipliers to its
            // own KKT precision, continue with a normal feasibility correction
            // from the candidate already found. Do not replace the original
            // descent problem merely because a nonlinear tangent overshoots.
            // Final acceptance still evaluates the original LM model and all
            // exact physical domains.
            if(dual_unresolved){
                cuda_check(cudaMemcpyAsync(coupled.base,view.step,l.q*sizeof(double),cudaMemcpyDeviceToDevice,s));
                boundaries.clear();restoring=true;correcting=false;dual_unresolved=false;
                if(std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS"))std::fprintf(stderr,"LM_DUAL_NORMAL_CORRECTION iteration=%zu margin=%.17g\n",iteration,margin);
            }
            CoupledBoundary boundary{selected,a,p,m,b};int index=0;
            while(index<int(boundaries.size())&&!boundaries[index].same(boundary))++index;
            const bool known_face=index<int(boundaries.size());
            // At an exposed quadratic pupil extremum, a violated worst face
            // is the lower bound of a convex quadratic (or the upper bound
            // of a concave one). Its margin is a concave quadratic-over-linear
            // function of the linear pupil coefficients and forward lever.
            // Retain its supporting halfspaces: replacing the preceding
            // tangent can oscillate across the curved boundary and destroy
            // an otherwise useful step. These cuts exist only in this trial;
            // the exact exposed-domain check above remains authoritative.
            const bool vertex_bundle=selected==0&&(a.kind==5||a.kind==6);
            if(vertex_bundle&&known_face){
                int second=index+1;while(second<int(boundaries.size())&&!boundaries[second].same(boundary))++second;
                if(second<int(boundaries.size())){
                    lm_coupled_merge_vertex<<<1,1,0,s>>>(l,coupled,index,second,int(boundaries.size()));
                    boundaries[second]=boundaries.back();boundaries.pop_back();
                }
                index=int(boundaries.size());
            }
            // A replacement tangent can overshoot a curved physical face.
            // Backtrack that correction, retaining the ORIGINAL quadratic
            // minimizer and all offsets in its Schur metric. Recentring base
            // on the previous candidate would replace the descent problem by
            // feasibility restoration and can stall at a nonstationary point.
            // Keep the anchor while shortening the same correction; relinearize
            // only after the exact worst violation no longer increases.
            if(correcting&&index<int(boundaries.size())&&margin<anchor_margin&&iteration+1<2*size_t(coupled.capacity)){
                if(std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS")){
                    std::fprintf(stderr,"LM_CORRECTION_BACKTRACK iteration=%zu anchor_margin=%.17g margin=%.17g\n",iteration,anchor_margin,margin);std::fflush(stderr);}
                lm_camera_backtrack_correction<<<(l.q+127)/128,128,0,s>>>(l,view,coupled);
                ++envelope_backtracks;continue;
            }
            correcting=false;
            if(std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS")){
                std::fprintf(stderr,"LM_COUPLED iteration=%zu family=%d margin=%.17g camera_kind=%d cell=%d u=%.17g sign=%.17g print_path=%d station=%.17g side=%d mount_kind=%d retained_column=%d retained_sign=%d retained_kind=%d\n",
                    iteration,selected,margin,a.kind,a.cell,a.u,a.sign,p.path,p.station,p.side,m.kind,b.column,b.sign,b.kind);std::fflush(stderr);}
            // Storage capacity limits distinct cuts, not successive nonlinear
            // corrections of the same face. A compact bundle can still need
            // further refinement to the exact physical boundary.
            if(iteration+1<2*size_t(coupled.capacity)&&finite_scalar(margin)){
                if(index==coupled.capacity)grow_coupled(l.q,int(boundaries.size()),s);
                const bool origin=!restoring&&selected!=3&&!known_face;
                if(index==int(boundaries.size()))boundaries.push_back(boundary);
                const int cuts=int(boundaries.size());const View at=origin?accepted:trial;
                // First-order constraints use the accepted objective chart.
                // Other nonlinear faces replace their previous tangent; the
                // convex pupil-vertex face keeps a compact supporting bundle.
                if((selected==1||selected==2)&&l.scene_directions)lm_print_chart_directions<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                if(selected==3)lm_retained_gradient<<<(l.q+127)/128,128,0,s>>>(b.kind==5?trial:accepted,l,view,b);
                else if(selected==2)lm_mount_gradient<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                else if(selected==1)lm_print_gradient<<<(l.q+127)/128,128,0,s>>>(at,l,view);
                else lm_camera_envelope_gradient<<<(l.q+127)/128,128,0,s>>>(accepted,at,l,view,origin);
                sparse_check(cudssExecute(sparse,CUDSS_PHASE_SOLVE,config,data,A,envelope_X,envelope_B));check_sparse(s);
                if(stop.load())throw SolverStopped{};
                const bool trace_projection=std::getenv("RINKSPACE_TRACE_LM_PROJECTIONS")!=nullptr;
                lm_coupled_add_cut<<<1,128,0,s>>>(l,view,coupled,index,cuts,margin,origin,selected==3&&b.equality,trace_projection);
                const char* dual_backend=std::getenv("RINKSPACE_SMALL_DUAL_BACKEND");
                if(dual_backend&&std::string(dual_backend)!="cpu"&&std::string(dual_backend)!="gpu")
                    throw std::invalid_argument("RINKSPACE_SMALL_DUAL_BACKEND must be cpu or gpu");
                if(!dual_backend||std::string(dual_backend)=="cpu"){
                    dual_host_gram.resize(size_t(cuts)*cuts);dual_host_target.resize(cuts);
                    dual_host_values.resize(cuts);dual_host_equalities.resize(cuts);
                    cuda_check(cudaMemcpy2DAsync(dual_host_gram.data(),size_t(cuts)*sizeof(double),
                        coupled.gram,size_t(coupled.capacity)*sizeof(double),size_t(cuts)*sizeof(double),cuts,cudaMemcpyDeviceToHost,s));
                    cuda_check(cudaMemcpyAsync(dual_host_target.data(),coupled.target,cuts*sizeof(double),cudaMemcpyDeviceToHost,s));
                    cuda_check(cudaMemcpyAsync(dual_host_values.data(),coupled.dual,cuts*sizeof(double),cudaMemcpyDeviceToHost,s));
                    cuda_check(cudaMemcpyAsync(dual_host_equalities.data(),coupled.equality,cuts*sizeof(double),cudaMemcpyDeviceToHost,s));
                    cuda_check(cudaStreamSynchronize(s));shared_projection_downloads+=dual_host_gram.size()+3*cuts;
                    if(stop.load())throw SolverStopped{};
                    const char* dual_trace_path=std::getenv("RINKSPACE_TRACE_DUAL_INPUT");
                    std::vector<double> dual_trace_seed;if(dual_trace_path)dual_trace_seed=dual_host_values;
                    const auto dual_started=Clock::now();double kkt;
                    int sweeps=coupled_dual_minimize_cached(cuts,cuts,dual_host_gram.data(),dual_host_target.data(),dual_host_values.data(),dual_host_equalities.data(),dual_cache,&kkt);
                    dual_unresolved=coupled_dual_unresolved(cuts,dual_host_target.data(),kkt);
                    if(trace_projection){std::fprintf(stderr,"LM_DUAL_HOST count=%d sweeps=%d kkt=%.17g elapsed_ms=%.6f residual_evaluations=%llu cache_hits=%llu skipped_sweeps=%llu\n",cuts,sweeps,kkt,millis(dual_started),(unsigned long long)dual_cache.evaluations,(unsigned long long)dual_cache.hits,(unsigned long long)dual_cache.skipped_sweeps);std::fflush(stderr);}
                    if(dual_trace_path&&sweeps==4096){
                        FILE* file=std::fopen(dual_trace_path,dual_trace_started?"ab":"wb");
                        if(!file)throw std::runtime_error("cannot write requested small-dual input trace");
                        int32_t n=cuts;bool ok=std::fwrite(&n,sizeof(n),1,file)==1;
                        for(const auto* values:{&dual_host_gram,&dual_host_target,&dual_trace_seed,&dual_host_equalities})
                            ok=(std::fwrite(values->data(),sizeof(double),values->size(),file)==values->size())&&ok;
                        ok=(std::fclose(file)==0)&&ok;if(!ok)throw std::runtime_error("small-dual input trace write failed");dual_trace_started=true;
                    }
                    cuda_check(cudaMemcpyAsync(coupled.dual,dual_host_values.data(),cuts*sizeof(double),cudaMemcpyHostToDevice,s));
                }else{
                    lm_coupled_dual<<<1,1,0,s>>>(coupled,view,cuts,trace_projection);
                    double unresolved=0.;cuda_check(cudaMemcpyAsync(&unresolved,view.scratch+1,sizeof(double),cudaMemcpyDeviceToHost,s));
                    cuda_check(cudaStreamSynchronize(s));++shared_projection_downloads;dual_unresolved=unresolved!=0.;
                }
                correcting=!origin&&selected!=3&&!vertex_bundle;anchor_margin=margin;
                if(correcting)cuda_check(cudaMemcpyAsync(coupled.anchor,view.step,l.q*sizeof(double),cudaMemcpyDeviceToDevice,s));
                lm_coupled_apply<<<(l.q+127)/128,128,0,s>>>(l,view,coupled,cuts);
                ++envelope_projections;
            }else{
                return false;
            }
        }
    }
    bool run(View v,View trial,Linearization& l,size_t input_count,cudaStream_t s,const double* accepted_output,double* output,double* timings,const std::atomic<bool>& stop_requested){
        double values[24]{};auto wall=Clock::now();std::fill(output,output+32,0.);std::fill(timings,timings+12,0.);std::fill(objective_ms,objective_ms+6,0.);int q=l.host.q,cc=l.host.ncomponents;trial_available=false;const size_t envelope_copies_before=envelope_downloads,print_copies_before=print_downloads,mount_copies_before=mount_downloads,retained_copies_before=retained_downloads,shared_projection_copies_before=shared_projection_downloads;
        const auto stop_boundary=[&](){if(stop_requested.load()){cuda_check(cudaStreamSynchronize(s));throw SolverStopped{};}};
        stop_boundary();
        if(l.device.envelope.enabled||l.device.rig_directions||l.device.scene_directions||l.device.mount_side)cuda_check(cudaMemcpyAsync(trial_input.data,v.board,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));
        start(s);cuda_check(cudaMemsetAsync(scalars.data,0,scalars.bytes(),s));int initial=std::max({q,cc,l.host.ng});
        lm_start<<<(initial+127)/128,128,0,s>>>(l.device,l.h_values(),l.d_shared.data,l.d_stats.data,scalars.data);lm_set_lambda<<<1,1,0,s>>>(scalars.data,lambda,initial_relative,jacobi_damping);lm_set_damping<<<(initial+127)/128,128,0,s>>>(l.device,view,l.h_values(),l.d_shared.data,jacobi_damping);
        if(image_station_damping&&view.nlocal)lm_image_station_metric<<<cc,128,0,s>>>(l.device,view,l.d_jac.data);
        if(image_global_damping){cuda_check(cudaMemsetAsync(image_metric.data,0,image_metric.bytes(),s));lm_image_global_accumulate<<<unsigned((v.n+127)/128),128,0,s>>>(v,l.device,view,l.d_jac.data);lm_image_global_metric<<<(q+127)/128,128,0,s>>>(l.device,view);if(view.nlocal)lm_image_local_floor<<<(view.nlocal+127)/128,128,0,s>>>(l.device,view);}
        lm_init_retained_bounds<<<(q+127)/128,128,0,s>>>(v,l.device,view);if(view.nlocal)lm_init_bounds<<<(view.nlocal+127)/128,128,0,s>>>(v,l.device,view,l.d_g.data);cuda_check(cudaGetLastError());read_scalars(values,s);timings[1]+=finish(s);
        if(!(values[0]>0)||!finite_scalar(values[0]))throw std::runtime_error("no observable finite damped descent system");
        double used_lambda=values[0];int bound_passes=0,scalar_copies=1,dense_checks=0,domain_copies=0;
        bool have_anchor=false,conditional_completed=false,joint_quadratic_completed=false;size_t anchor_power=1,anchor_distance=0,working_set_copies=0,quadratic_downloads=0;
        cuda_check(cudaMemsetAsync(applied.data,0,applied.bytes(),s));
        const bool trace_active=std::getenv("RINKSPACE_TRACE_LM_ACTIVESET")!=nullptr;
        const auto trace=[&](const char* phase){if(trace_active&&(bound_passes<=8||(bound_passes&(bound_passes-1))==0)){
            std::fprintf(stderr,"LM_ACTIVE pass=%d phase=%s elapsed_ms=%.3f lambda=%.12g\n",bound_passes,phase,millis(wall),used_lambda);std::fflush(stderr);}};
        auto reject_factor=[&](){++numerical_rejections;lambda=used_lambda*nu;nu*=2.;used_lambda=lambda;if(!(lambda>0)||!finite_scalar(lambda)||!finite_scalar(nu))throw std::runtime_error("LM damping overflow after numerical factorization rejection");
            have_anchor=false;anchor_power=1;anchor_distance=0;cuda_check(cudaMemsetAsync(applied.data,0,applied.bytes(),s));
            cuda_check(cudaMemcpyAsync(scalars.data,&used_lambda,sizeof(double),cudaMemcpyHostToDevice,s));lm_init_retained_bounds<<<(q+127)/128,128,0,s>>>(v,l.device,view);if(view.nlocal)lm_init_bounds<<<(view.nlocal+127)/128,128,0,s>>>(v,l.device,view,l.d_g.data);cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));
            if(numeric_progress)numeric_progress(numerical_rejections,lambda,nu,numeric_context);
            stop_boundary();};
        auto reject_boundary=[&](){
            read_scalars(values,s);++scalar_copies;std::fill(values+8,values+24,0.);
            values[3]=0.;values[9]=-1.;values[10]=values[1];lambda=used_lambda*nu;nu*=2.;++trials;
            std::copy(values,values+24,output);output[15]=0.;output[16]=7.;output[17]=lambda;output[18]=nu;
            output[19]=bound_passes;output[20]=analyses;output[21]=trials;output[22]=accepted;
            output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int)+domain_copies*sizeof(double)+working_set_copies*sizeof(int)+(envelope_downloads-envelope_copies_before)*sizeof(CameraConstraint)+(print_downloads-print_copies_before)*sizeof(PrintConstraint)+(mount_downloads-mount_copies_before)*sizeof(MountConstraint)+(retained_downloads-retained_copies_before)*sizeof(RetainedConstraint)+(shared_projection_downloads-shared_projection_copies_before)*sizeof(double)+quadratic_downloads;
            output[24]=double(device_bytes());output[25]=double(estimates[0]);output[26]=double(estimates[1]);output[27]=double(estimates[2]);output[28]=double(estimates[3]);output[29]=double(factor_nnz);output[30]=eta;output[31]=0.;timings[0]=millis(wall);return false;
        };
        for(;;){stop_boundary();++bound_passes;trace("local");start(s);
            if(batches.empty()&&view.nlocal&&l.device.pixel_residuals)lm_qp_save_active<<<(view.nlocal+127)/128,128,0,s>>>(l.device,view);
            lm_fill_local<<<cc,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data,l.d_jac.data);cuda_check(cudaGetLastError());bool local_ok=true;
            for(auto b:batches){dense_check(cusolverDnDpotrfBatched(dense,CUBLAS_FILL_MODE_LOWER,b.dimension,factor_ptrs.data+b.factor_begin,b.dimension,factor_info.data+b.factor_begin,int(b.factor_count)));++dense_checks;
                if(!check_dense(factor_info.data+b.factor_begin,int(b.factor_count),s,true)){stop_boundary();local_ok=false;break;}
                stop_boundary();
                dense_check(cusolverDnDpotrsBatched(dense,CUBLAS_FILL_MODE_LOWER,b.dimension,1,solve_a_ptrs.data+b.solve_begin,b.dimension,solve_b_ptrs.data+b.solve_begin,b.dimension,solve_info.data,int(b.solve_count)));++dense_checks;check_dense(solve_info.data,1,s);stop_boundary();}
            if(!local_ok){timings[2]+=finish(s);stop_boundary();reject_factor();continue;}
            // Separate copying and diagonal addition avoids overlapping writes.
            cuda_check(cudaMemcpyAsync(csr.data,l.d_shared.data,csr.bytes(),cudaMemcpyDeviceToDevice,s));
            lm_reduce_start<<<(q+127)/128,128,0,s>>>(l.device,view,l.d_shared.data);
            lm_reduce<<<cc,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data);lm_constrain_retained_matrix<<<(q+127)/128,128,0,s>>>(l.device,view);lm_constrain_retained_rhs<<<(q+127)/128,128,0,s>>>(l.device,view);cuda_check(cudaGetLastError());timings[2]+=finish(s);
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
            trace("shared_projection");start(s);lm_residual_init<<<(q+127)/128,128,0,s>>>(l.device,view);lm_residual_spmv<<<(q+127)/128,128,0,s>>>(l.device,view);if(!project_envelope(v,trial,l.device,s,stop_requested)){timings[6]+=finish(s);return reject_boundary();}trace("backsubstitution");lm_backsub<<<cc,128,0,s>>>(l.device,view);
            lm_feasible_init<<<1,1,0,s>>>(view);int nf=std::max(q,view.nlocal);lm_feasible<<<(nf+127)/128,128,0,s>>>(v,l.device,view);
            if(view.nlocal){lm_local_release_start<<<1,1,0,s>>>(view,check_info.data);
                lm_local_release<<<(cc+127)/128,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data,l.d_jac.data,check_info.data);
                lm_local_release_finish<<<1,1,0,s>>>(view,check_info.data);}
            lm_feasible_finish<<<1,1,0,s>>>(view);cuda_check(cudaGetLastError());read_scalars(values,s);++scalar_copies;timings[6]+=finish(s);
            if(trace_active&&(bound_passes<=8||(bound_passes&(bound_passes-1))==0)){int released=view.nlocal;
                if(view.nlocal)cuda_check(cudaMemcpy(&released,check_info.data,sizeof(int),cudaMemcpyDeviceToHost));
                std::fprintf(stderr,"LM_ACTIVE pass=%d phase=feasibility changed=%.0f blocked=%.0f release=%d elapsed_ms=%.3f\n",bound_passes,values[5],values[6],released,millis(wall));std::fflush(stderr);}
            if(values[7]||!finite_scalar(values[4])||values[4]>linear_tolerance)throw std::runtime_error("nonfinite step or excessive cuDSS residual");
            if(batches.empty()&&view.nlocal&&l.device.pixel_residuals){
                start(s);bool complete=refine_bounded_quadratic(l,s,stop_requested,quadratic_downloads);timings[6]+=finish(s);
                joint_quadratic_completed=true;if(complete)break;continue;
            }
            if(values[5]==0)break;
            // The physical projection changes the shared solution after Schur
            // elimination. With independent scalar locals, solve their bounded
            // conditional quadratics once instead of cycling through face sets.
            // Probe without mutating the working set: an invalid finite-support
            // completion must fall through to the existing active-set solve.
            if(batches.empty()&&view.nlocal&&envelope_projections>projections_before){
                start(s);cuda_check(cudaMemsetAsync(check_info.data,0,sizeof(int),s));
                lm_conditional_shared_check<<<(q+127)/128,128,0,s>>>(l.device,view,check_info.data);
                lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.h_values(),l.d_g.data,accepted_output,l.d_jac.data,check_info.data,false);
                int invalid=0;cuda_check(cudaMemcpyAsync(&invalid,check_info.data,sizeof(int),cudaMemcpyDeviceToHost,s));
                cuda_check(cudaStreamSynchronize(s));++working_set_copies;stop_boundary();
                if(!invalid){
                    lm_conditional_start<<<1,1,0,s>>>(view,check_info.data);
                    lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.h_values(),l.d_g.data,accepted_output,l.d_jac.data,check_info.data);
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
                        lm_conditional_locals<<<(cc+127)/128,128,0,s>>>(v,l.device,view,l.h_values(),l.d_g.data,accepted_output,l.d_jac.data,check_info.data);
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
                    output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int)+domain_copies*sizeof(double)+working_set_copies*sizeof(int)+(envelope_downloads-envelope_copies_before)*sizeof(CameraConstraint)+(print_downloads-print_copies_before)*sizeof(PrintConstraint)+(mount_downloads-mount_copies_before)*sizeof(MountConstraint)+(retained_downloads-retained_copies_before)*sizeof(RetainedConstraint)+(shared_projection_downloads-shared_projection_copies_before)*sizeof(double)+quadratic_downloads;
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
            lm_retract<<<unsigned((n+127)/128),128,0,s>>>(v,trial,l.device,view);if(l.device.rig_directions)lm_retract_rig<<<unsigned(v.nrig),32,0,s>>>(v,trial,l.device,view);
            if(l.device.scene_directions){lm_retract_scene<<<unsigned((v.nscene+127)/128),128,0,s>>>(v,trial,l.device,view);lm_build_course<<<1,1,0,s>>>(trial);}
            if(view.nlocal){lm_restore_station_domain<<<(view.nlocal+127)/128,128,0,s>>>(trial,l.device,view);lm_share_restored_stations<<<unsigned((v.n+127)/128),128,0,s>>>(trial,l.device,view);lm_actual_locals<<<(view.nlocal+127)/128,128,0,s>>>(v,trial,l.device,view);}
            evaluate_kernel<<<unsigned((v.n+127)/128),128,0,s>>>(trial,trial_output.data,trial_status.data);
            if(restoration==40)break;
            cuda_check(cudaMemcpyAsync(step.data,applied.data,step.bytes(),cudaMemcpyDeviceToDevice,s));
            if(!has_active_profiles)break;
            ++profile_restoration_calls;
            lm_restore_profile_domain<<<1,1,0,s>>>(v,trial,l.device,view,l.d_jac.data,trial_status.data);
            double restored=0.;++domain_copies;cuda_check(cudaMemcpyAsync(&restored,view.scratch,sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));stop_boundary();
            if(restored==0.)break;
            if(!project_envelope(v,trial,l.device,s,stop_requested)){timings[7]+=finish(s);return reject_boundary();}
        }
        cuda_check(cudaEventRecord(prediction_begin,s));lm_prediction<<<cc,128,0,s>>>(l.device,view,l.h_values(),l.d_g.data);lm_damping_energy<<<(q+view.nlocal+127)/128,128,0,s>>>(l.device,view);cuda_check(cudaEventRecord(prediction_end,s));
        if(profile_trial_stations&&station_refinement.samples.count){
            // The Schur quadratic predicts the profiled objective through its
            // linearized station minimizer. Evaluate the nonlinear candidate
            // after minimizing those independent coordinates too, rather than
            // rejecting it before the accepted-state-only curve search runs.
            // Keep that model prediction; report the actual refined coordinate
            // change in the step norm. Fixed/shared stations are never touched.
            const auto station_started=Clock::now();double refinement[4]{};station_refinement.run(trial,s,refinement,true);trial_station_wall_ms+=millis(station_started);domain_copies+=4;stop_boundary();
            lm_actual_locals<<<(view.nlocal+127)/128,128,0,s>>>(v,trial,l.device,view);
            evaluate_kernel<<<unsigned((v.n+127)/128),128,0,s>>>(trial,trial_output.data,trial_status.data);
        }
        cuda_check(cudaGetLastError());timings[7]=finish(s);float prediction_ms=0.;cuda_check(cudaEventElapsedTime(&prediction_ms,prediction_begin,prediction_end));prediction_wall_ms+=prediction_ms;
        start(s);cuda_check(cudaEventRecord(objective_events[0],s));
        if(!l.device.pixel_residuals)lm_trial_whiten<<<l.host.ng,128,0,s>>>(l.device,trial_output.data,trial_offsets.data,trial_white.data);
        lm_trial_loss<<<(l.host.ng+127)/128,128,0,s>>>(l.device,trial_output.data,trial_offsets.data,trial_white.data,trial_stats.data);
        lm_finish_init<<<1,1,0,s>>>(trial,view);cuda_check(cudaEventRecord(objective_events[1],s));
        if(l.device.rig_directions||l.device.mount_side)lm_rig_domain<<<1,1,0,s>>>(trial,l.device,view);
        cuda_check(cudaEventRecord(objective_events[2],s));
        if(l.device.scene_directions){cuda_check(cudaMemsetAsync(print_domain_invalid.data,0,print_domain_invalid.bytes(),s));
            if(view.print_count)lm_print_domain<<<unsigned((view.print_count+127)/128),128,0,s>>>(trial,view);}
        cuda_check(cudaEventRecord(objective_events[3],s));
        if(l.device.scene_directions)lm_scene_domain<<<unsigned((std::max(v.nscene,v.npaths)+127)/128),128,0,s>>>(trial,l.device,view);
        cuda_check(cudaEventRecord(objective_events[4],s));
        if(l.device.envelope.enabled){evaluate_envelope(trial,l.device,s);lm_camera_envelope_domain<<<1,1,0,s>>>(view);}
        cuda_check(cudaEventRecord(objective_events[5],s));
        size_t nf=std::max({v.n,size_t(view.nlocal),size_t(q)+view.nlocal,size_t(l.host.ng),size_t(cc)});
        lm_finish<<<unsigned((nf+127)/128),128,0,s>>>(trial,l.device,view,trial_status.data,trial_stats.data);lm_finish_values<<<1,1,0,s>>>(trial,view);
        cuda_check(cudaEventRecord(objective_events[6],s));cuda_check(cudaGetLastError());timings[8]=finish(s);
        for(int k=0;k<6;++k){float ms;cuda_check(cudaEventElapsedTime(&ms,objective_events[k],objective_events[k+1]));objective_ms[k]=ms;}
        trial_available=true;
        start(s);read_scalars(values,s);++scalar_copies;timings[9]=finish(s);stop_boundary();int reason=0;
        if(values[8]||!finite_scalar(values[10]))reason=3;else if(!(values[11]>0)||!(values[3]>0)||!finite_scalar(values[11]))reason=2;else if(!(values[12]>0)||!(values[13]>eta))reason=1;
        last_damping_energy=values[15];last_prediction=values[11];
        bool accept=reason==0;++trials;
        if(accept){start(s);cuda_check(cudaMemcpyAsync(const_cast<double*>(v.board),trial_input.data,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));if(v.scene_paths)cuda_check(cudaMemcpyAsync(const_cast<BoardCourse*>(v.course),trial.course,sizeof(BoardCourse),cudaMemcpyDeviceToDevice,s));timings[10]=finish(s);++accepted;l.gpu_ready=false;l.cpu_ready=false;lambda=used_lambda*std::max(1./3.,1.-std::pow(2*values[13]-1,3));nu=2.;}else{lambda=used_lambda*nu;nu*=2.;}
        std::copy(values,values+24,output);output[15]=accept;output[16]=accept&&joint_quadratic_completed?6:(accept&&conditional_completed?5:reason);output[17]=lambda;output[18]=nu;output[19]=bound_passes;output[20]=analyses;output[21]=trials;output[22]=accepted;
        output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int)+domain_copies*sizeof(double)+working_set_copies*sizeof(int)+(envelope_downloads-envelope_copies_before)*sizeof(CameraConstraint)+(print_downloads-print_copies_before)*sizeof(PrintConstraint)+(mount_downloads-mount_copies_before)*sizeof(MountConstraint)+(retained_downloads-retained_copies_before)*sizeof(RetainedConstraint)+(shared_projection_downloads-shared_projection_copies_before)*sizeof(double)+quadratic_downloads;output[24]=double(device_bytes());output[25]=double(estimates[0]);output[26]=double(estimates[1]);output[27]=double(estimates[2]);output[28]=double(estimates[3]);output[29]=double(factor_nnz);output[30]=eta;output[31]=values[23];timings[0]=millis(wall);return accept;
    }
};
