/* G3 uses this same resident problem. Local dense solves use cuSOLVER;
 * the retained CSR system uses cuDSS. No host numerical solve exists. */
inline void dense_check(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cuSOLVER status "+std::to_string(int(s)));}
inline void sparse_check(cudssStatus_t s){if(s!=CUDSS_STATUS_SUCCESS)throw std::runtime_error("cuDSS status "+std::to_string(int(s)));}

struct SolverView {
    const int *locals,*representatives;
    const double *lower,*upper;
    int *blocked;
    double *matrix,*solved,*reduced,*rhs,*csr,*step,*applied,*scalars,*parts;
    int m,nlocal;
};
__global__ void lm_start(LinearView l,const double* h,const double* shared,const double* stats,double lambda,double tau,double* s){
    if(threadIdx.x||blockIdx.x)return;
    for(int k=0;k<24;++k)s[k]=0.;
    double maximum=0.;
    for(int k=0;k<l.q;++k)maximum=fmax(maximum,shared[k*l.q+k]);
    for(int ci=0;ci<l.ncomponents;++ci){auto c=l.components[ci];
        for(int k=l.q;k<c.dim;++k)maximum=fmax(maximum,h[c.h+k*c.dim+k]);}
    for(int gi=0;gi<l.ng;++gi)s[1]+=stats[3*gi+2];
    s[0]=lambda>0?lambda:tau*maximum;s[2]=maximum;
}
__global__ void lm_init_bounds(View v,LinearView l,SolverView s,const double* g){
    const auto c=l.components[blockIdx.x];
    for(int j=threadIdx.x;j<c.local_count;j+=blockDim.x){
        const int id=s.locals[c.local_begin+j];const double x=v.stations[s.representatives[id]];
        const double gradient=g[c.g+l.q+j],eps=64*std::numeric_limits<double>::epsilon();
        s.blocked[id]=(s.lower[id]==s.upper[id])||
            (x<=s.lower[id]+eps&&gradient>0)||(x>=s.upper[id]-eps&&gradient<0);
    }
}
__global__ void lm_fill_local(LinearView l,SolverView s,const double* h,const double* g){
    const int ci=blockIdx.x;const auto c=l.components[ci];const int m=s.m;
    for(int k=threadIdx.x;k<m*m;k+=blockDim.x){
        const int i=k%m,j=k/m;
        const bool ai=i<c.local_count&&!s.blocked[s.locals[c.local_begin+i]];
        const bool aj=j<c.local_count&&!s.blocked[s.locals[c.local_begin+j]];
        s.matrix[ci*m*m+k]=(ai&&aj)?h[c.h+(l.q+i)*c.dim+l.q+j]+(i==j?s.scalars[0]:0.):(i==j?1.:0.);
    }
    for(int k=threadIdx.x;k<m*(l.q+1);k+=blockDim.x){
        const int i=k%m,j=k/m;
        const bool free=i<c.local_count&&!s.blocked[s.locals[c.local_begin+i]];
        s.solved[ci*m*(l.q+1)+k]=free?(j<l.q?h[c.h+(l.q+i)*c.dim+j]:g[c.g+l.q+i]):0.;
    }
}
__global__ void lm_info(const int* info,int n,int* failed){if(threadIdx.x||blockIdx.x)return;
    failed[0]=-1;failed[1]=0;for(int i=0;i<n;++i)if(info[i]){failed[0]=i;failed[1]=info[i];return;}}
__global__ void lm_reduce(LinearView l,SolverView s,const double* h,const double* shared){
    int k=blockIdx.x*blockDim.x+threadIdx.x;const int q=l.q,m=s.m;
    if(k<q*q){
        const int i=k/q,j=k%q;double ij=shared[k],ji=shared[j*q+i];
        for(int ci=0;ci<l.ncomponents;++ci){auto c=l.components[ci];
            for(int a=0;a<c.local_count;++a){ij-=h[c.h+i*c.dim+q+a]*s.solved[(ci*(q+1)+j)*m+a];
                ji-=h[c.h+j*c.dim+q+a]*s.solved[(ci*(q+1)+i)*m+a];}}
        double v=.5*(ij+ji)+(i==j?s.scalars[0]:0.);s.reduced[k]=v;
        if(j>=i)s.csr[i*q-i*(i-1)/2+j-i]=v;
    }else if(k<q*q+q){
        int i=k-q*q;double v=-shared[q*q+i];
        for(int ci=0;ci<l.ncomponents;++ci){auto c=l.components[ci];
            for(int a=0;a<c.local_count;++a)v+=h[c.h+i*c.dim+q+a]*s.solved[(ci*(q+1)+q)*m+a];}
        s.rhs[i]=v;
    }
}
__global__ void lm_backsub(LinearView l,SolverView s){
    const int ci=blockIdx.x;auto c=l.components[ci];
    for(int a=threadIdx.x;a<c.local_count;a+=blockDim.x){int id=s.locals[c.local_begin+a];
        double v=s.solved[(ci*(l.q+1)+l.q)*s.m+a];
        for(int j=0;j<l.q;++j)v+=s.solved[(ci*(l.q+1)+j)*s.m+a]*s.step[j];
        s.step[l.q+id]=s.blocked[id]?0.:-v;}
}
__global__ void lm_feasible(View v,LinearView l,SolverView s){
    if(threadIdx.x||blockIdx.x)return;
    double alpha=1.,rr=0.,bb=0.;int changed=0,blocked=0,nonfinite=0;
    for(int i=0;i<l.q;++i){double r=-s.rhs[i];for(int j=0;j<l.q;++j)r+=s.reduced[i*l.q+j]*s.step[j];rr+=r*r;bb+=s.rhs[i]*s.rhs[i];nonfinite+=!finite_scalar(s.step[i]);}
    for(int id=0;id<s.nlocal;++id){
        double x=v.stations[s.representatives[id]],d=s.step[l.q+id]*l.scales[id];
        const double eps=64*std::numeric_limits<double>::epsilon();
        if(!s.blocked[id]&&((x<=s.lower[id]+eps&&d<0)||(x>=s.upper[id]-eps&&d>0))){
            s.blocked[id]=1;++changed;
        }
        blocked+=s.blocked[id]!=0;nonfinite+=!finite_scalar(d);
        if(d>0)alpha=fmin(alpha,(s.upper[id]-x)/d);
        else if(d<0)alpha=fmin(alpha,(s.lower[id]-x)/d);
    }
    if(alpha<1&&alpha>0)alpha=nextafter(alpha,0.);
    s.scalars[3]=fmax(0.,alpha);s.scalars[4]=sqrt(rr)/fmax(1.,sqrt(bb));
    s.scalars[5]=changed;s.scalars[6]=blocked;s.scalars[7]=nonfinite;
}
__global__ void lm_retract(View accepted,View trial,LinearView l,SolverView s){
    size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;double alpha=s.scalars[3];
    if(k<3*accepted.nc){double d=0.;for(int j=0;j<l.q;++j)d+=l.directions[k*l.q+j]*(alpha*s.step[j]);
        const_cast<double*>(trial.controls)[k]=accepted.controls[k]+d;}
    if(k<accepted.n){int id=l.row_local[k];if(id>=0)
        const_cast<double*>(trial.stations)[k]=accepted.stations[k]+alpha*s.step[l.q+id]*l.scales[id];}
    if(k<static_cast<size_t>(l.q))s.applied[k]=alpha*s.step[k];
}
__global__ void lm_actual_locals(View accepted,View trial,LinearView l,SolverView s){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=s.nlocal)return;int row=s.representatives[id];
    s.applied[l.q+id]=(trial.stations[row]-accepted.stations[row])/l.scales[id];
}
__device__ double lm_component_step(LinearView l,SolverView s,ComponentPlan c,int j){
    return j<l.q?s.applied[j]:s.applied[l.q+s.locals[c.local_begin+j-l.q]];
}
__global__ void lm_prediction(LinearView l,SolverView s,const double* h,const double* g){
    int ci=blockIdx.x;if(threadIdx.x)return;auto c=l.components[ci];double value=0.;
    for(int i=0;i<c.dim;++i){double di=lm_component_step(l,s,c,i);value-=g[c.g+i]*di;
        for(int j=0;j<c.dim;++j)value-=.5*di*h[c.h+i*c.dim+j]*lm_component_step(l,s,c,j);}
    s.parts[ci]=value;
}
__global__ void lm_trial_whiten(LinearView l,const double* output,double* white){
    int gi=blockIdx.x;auto g=l.groups[gi];int d=l.components[g.component].dim;
    for(int k=threadIdx.x;k<g.rank;k+=blockDim.x)white[g.white+k*(d+1)]=whiten_element(l,gi,k,0,output,nullptr);
}
__global__ void lm_finish(View trial,LinearView l,SolverView s,const int* statuses,const double* stats){
    if(threadIdx.x||blockIdx.x)return;
    double cost=0.,pred=0.,norm=0.;int bad=0,first=-1;
    for(size_t i=0;i<trial.n;++i)if(statuses[i]){if(first<0)first=static_cast<int>(i);++bad;}
    for(int id=0;id<s.nlocal;++id){double x=trial.stations[s.representatives[id]];
        if(!finite_scalar(x)||x<s.lower[id]||x>s.upper[id]){if(first<0)first=s.representatives[id];++bad;}}
    for(int gi=0;gi<l.ng;++gi)cost+=stats[3*gi+2];
    for(int ci=0;ci<l.ncomponents;++ci)pred+=s.parts[ci];
    for(int i=0;i<l.q+s.nlocal;++i)norm+=s.applied[i]*s.applied[i];
    s.scalars[8]=bad;s.scalars[9]=first;s.scalars[10]=cost;s.scalars[11]=pred;
    s.scalars[12]=s.scalars[1]-cost;s.scalars[13]=pred>0?s.scalars[12]/pred:0.;
    s.scalars[14]=sqrt(norm);
    double maximum=0.;for(int i=0;i<l.q+s.nlocal;++i)maximum=fmax(maximum,fabs(s.applied[i]));
    s.scalars[23]=maximum;
}

// Projected gradient of the ORIGINAL robust objective in declared scaled coordinates.
// Bounds remove only outward station directions; shared gradients are summed once.
__global__ void lm_stationarity(View v,LinearView l,SolverView s,const double* g,const double* shared,const double* stats){
    if(threadIdx.x||blockIdx.x)return;
    double maximum=0.,cost=0.;int bad=0;
    for(int j=0;j<l.q;++j){double x=shared[l.q*l.q+j];maximum=fmax(maximum,fabs(x));bad+=!finite_scalar(x);}
    for(int ci=0;ci<l.ncomponents;++ci){auto c=l.components[ci];
        for(int j=0;j<c.local_count;++j){
            int id=s.locals[c.local_begin+j];double x=v.stations[s.representatives[id]],dg=g[c.g+l.q+j];
            const double eps=64*std::numeric_limits<double>::epsilon();
            bool fixed=s.lower[id]==s.upper[id]||(x<=s.lower[id]+eps&&dg>0)||(x>=s.upper[id]-eps&&dg<0);
            if(!fixed)maximum=fmax(maximum,fabs(dg));bad+=!finite_scalar(dg);
        }
    }
    for(int i=0;i<l.ng;++i)cost+=stats[3*i+2];
    s.scalars[0]=cost;s.scalars[1]=maximum;s.scalars[2]=bad+!finite_scalar(cost);
}

struct Solver {
    DeviceBuffer<double> lower,upper,matrix,solved,reduced,rhs,csr,step,applied,scalars,parts,trial_input,trial_output,trial_white,trial_stats;
    DeviceBuffer<int> locals,representatives,blocked,factor_info,solve_info,check_info,csr_offsets,csr_columns,trial_status;
    DeviceBuffer<double*> factor_ptrs,solve_a_ptrs,solve_b_ptrs;
    SolverView view{};
    cusolverDnHandle_t dense=nullptr;cudssHandle_t sparse=nullptr;cudssConfig_t config=nullptr;cudssData_t data=nullptr;
    cudssMatrix_t A=nullptr,X=nullptr,B=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    double lambda=0.,nu=2.,initial_relative=.001,eta=.0001,linear_tolerance=1e-8;
    double device_budget=0.,host_budget=0.;int m=1,analyses=0,trials=0,accepted=0,numerical_rejections=0;
    int64_t estimates[16]{},factor_nnz=0;
    bool trial_available=false,analysis_ready=false;
    ~Solver(){
        if(A)cudssMatrixDestroy(A);if(X)cudssMatrixDestroy(X);if(B)cudssMatrixDestroy(B);
        if(data)cudssDataDestroy(sparse,data);if(config)cudssConfigDestroy(config);
        if(sparse)cudssDestroy(sparse);if(dense)cusolverDnDestroy(dense);
        if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);
    }
    size_t device_bytes()const{
        return lower.bytes()+upper.bytes()+matrix.bytes()+solved.bytes()+reduced.bytes()+rhs.bytes()+csr.bytes()+step.bytes()+applied.bytes()+scalars.bytes()+parts.bytes()+trial_input.bytes()+trial_output.bytes()+trial_white.bytes()+trial_stats.bytes()+locals.bytes()+representatives.bytes()+blocked.bytes()+factor_info.bytes()+solve_info.bytes()+check_info.bytes()+csr_offsets.bytes()+csr_columns.bytes()+trial_status.bytes()+factor_ptrs.bytes()+solve_a_ptrs.bytes()+solve_b_ptrs.bytes();
    }
    void prepare(const View& v,Linearization& l,size_t input_count,const double* lo,const double* hi,size_t nlocal,const double* options,size_t noptions,cudaStream_t stream){
        if(noptions!=6||nlocal!=l.scales.size()||(nlocal&&(!lo||!hi))||!finite_array(options,noptions)||
           options[0]<=0||options[1]<0||options[2]<=0||options[2]>=1||options[3]<=0||options[4]<=0||options[5]<=0)
            throw std::invalid_argument("invalid G3 bounds/options");
        initial_relative=options[0];lambda=options[1];eta=options[2];linear_tolerance=options[3];device_budget=options[4];host_budget=options[5];
        std::vector<int> reps(nlocal,-1);
        for(size_t row=0;row<v.n;++row){int id=l.row_local[row];if(id<0)continue;
            bool periodic=v.paths&&v.paths[7*v.codes[row]]==7;
            if(std::isnan(lo[id])||std::isnan(hi[id])||(!periodic&&(lo[id]<0||hi[id]>1))||
               lo[id]>hi[id]||v.stations[row]<lo[id]||v.stations[row]>hi[id])
                throw std::invalid_argument("original station outside declared finite support at row "+std::to_string(row));
            reps[id]=static_cast<int>(row);}
        for(int r:reps)if(r<0)throw std::invalid_argument("unreferenced station");
        for(auto c:l.components)m=std::max(m,c.local_count);
        int q=l.host.q,cc=l.host.ncomponents;
        Linearization::bounded(size_t(cc)*(q+1));Linearization::bounded(size_t(q)*(q+1)/2);
        // Exact padding footprint is checked before allocation; never split a correlation component.
        double planned=8.*(double(cc)*m*m+double(cc)*m*(q+1)+q*q+q+double(q)*(q+1)/2+
            2*(q+nlocal)+24+cc+input_count+11*v.n+l.white_count+3*l.host.ng+2*nlocal)+
            4.*(3*nlocal+cc+3+q+1+double(q)*(q+1)/2+v.n)+8.*cc*(1+2*(q+1));
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        if(planned>device_budget||planned>free)throw std::runtime_error("G3 connected dense workspace exceeds available device budget");
        lower.upload(std::vector<double>(lo,lo+nlocal),stream);upper.upload(std::vector<double>(hi,hi+nlocal),stream);
        locals.upload(l.component_locals,stream);representatives.upload(reps,stream);blocked.allocate(nlocal);
        matrix.allocate(size_t(cc)*m*m);solved.allocate(size_t(cc)*m*(q+1));
        reduced.allocate(q*q);rhs.allocate(q);csr.allocate(size_t(q)*(q+1)/2);step.allocate(q+nlocal);applied.allocate(q+nlocal);
        scalars.allocate(24);parts.allocate(cc);trial_input.allocate(input_count);trial_output.allocate(v.n*11);trial_status.allocate(v.n);
        trial_white.allocate(l.white_count);trial_stats.allocate(3*l.host.ng);
        factor_info.allocate(cc);solve_info.allocate(1);check_info.allocate(2);
        std::vector<double*> ap,aa,bb;for(int i=0;i<cc;++i){ap.push_back(matrix.data+size_t(i)*m*m);
            for(int j=0;j<q+1;++j){aa.push_back(ap.back());bb.push_back(solved.data+(size_t(i)*(q+1)+j)*m);}}
        factor_ptrs.upload(ap,stream);solve_a_ptrs.upload(aa,stream);solve_b_ptrs.upload(bb,stream);
        std::vector<int> offsets{0},columns;
        for(int i=0;i<q;++i){for(int j=i;j<q;++j)columns.push_back(j);offsets.push_back(static_cast<int>(columns.size()));}
        csr_offsets.upload(offsets,stream);csr_columns.upload(columns,stream);
        view={locals.data,representatives.data,lower.data,upper.data,blocked.data,matrix.data,solved.data,reduced.data,rhs.data,csr.data,step.data,applied.data,scalars.data,parts.data,m,static_cast<int>(nlocal)};
        dense_check(cusolverDnCreate(&dense));dense_check(cusolverDnSetStream(dense,stream));
        sparse_check(cudssCreate(&sparse));sparse_check(cudssSetStream(sparse,stream));
        sparse_check(cudssConfigCreate(&config));sparse_check(cudssDataCreate(sparse,&data));
        int disabled=0;sparse_check(cudssConfigSet(config,CUDSS_CONFIG_HYBRID_EXECUTE_MODE,&disabled,sizeof(disabled)));
        sparse_check(cudssConfigSet(config,CUDSS_CONFIG_HYBRID_MEMORY_MODE,&disabled,sizeof(disabled)));
        sparse_check(cudssMatrixCreateDn(&X,q,1,q,step.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
        sparse_check(cudssMatrixCreateDn(&B,q,1,q,rhs.data,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
        sparse_check(cudssMatrixCreateCsr(&A,q,q,columns.size(),csr_offsets.data,nullptr,csr_columns.data,csr.data,CUDSS_R_32I,CUDSS_R_32I,CUDSS_R_64F,CUDSS_MTYPE_SPD,CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
        cuda_check(cudaEventCreate(&begin));cuda_check(cudaEventCreate(&end));cuda_check(cudaStreamSynchronize(stream));
    }
    void start(cudaStream_t s){cuda_check(cudaEventRecord(begin,s));}
    double finish(cudaStream_t s){cuda_check(cudaEventRecord(end,s));cuda_check(cudaEventSynchronize(end));float ms;cuda_check(cudaEventElapsedTime(&ms,begin,end));return ms;}
    void read_scalars(double* h,cudaStream_t s){cuda_check(cudaMemcpyAsync(h,scalars.data,24*sizeof(double),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));}
    bool check_dense(const int* ptr,int n,cudaStream_t s,bool retry_nonpositive=false){
        lm_info<<<1,1,0,s>>>(ptr,n,check_info.data);cuda_check(cudaGetLastError());int result[2];
        cuda_check(cudaMemcpyAsync(result,check_info.data,sizeof(result),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));
        if(result[0]>=0){
            if(retry_nonpositive&&result[1]>0)return false;
            throw std::runtime_error("cuSOLVER nonpositive/invalid local component "+std::to_string(result[0])+" info "+std::to_string(result[1]));
        }
        return true;
    }
    bool check_sparse(cudaStream_t s,bool retry_nonpositive=false){
        cuda_check(cudaStreamSynchronize(s));int info=0;size_t written=0;
        sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_INFO,&info,sizeof(info),&written));
        if(info){if(retry_nonpositive&&info>0){
            // cuDSS 0.8 INFO is sticky; clear it before refactorization.
            // https://docs.nvidia.com/cuda/cudss/0.8.0/types.html
            int clear=0;sparse_check(cudssDataSet(sparse,data,CUDSS_DATA_INFO,&clear,sizeof(clear)));
            return false;}throw std::runtime_error("cuDSS device numerical info "+std::to_string(info));}
        return true;
    }
    bool run(View v,View trial,Linearization& l,size_t input_count,cudaStream_t s,double* output,double* timings){
        double values[24]{};const auto wall=Clock::now();std::fill(output,output+32,0.);std::fill(timings,timings+12,0.);
        int q=l.host.q,cc=l.host.ncomponents;trial_available=false;
        start(s);lm_start<<<1,1,0,s>>>(l.device,l.d_h.data,l.d_shared.data,l.d_stats.data,lambda,initial_relative,scalars.data);
        lm_init_bounds<<<cc,128,0,s>>>(v,l.device,view,l.d_g.data);
        cuda_check(cudaGetLastError());read_scalars(values,s);timings[1]+=finish(s);
        if(!(values[0]>0)||!finite_scalar(values[0]))throw std::runtime_error("no observable finite damped descent system");
        double used_lambda=values[0];int bound_passes=0,scalar_copies=1,dense_checks=0;
        auto reject_factor=[&](){
            ++numerical_rejections;lambda=used_lambda*nu;nu*=2.;used_lambda=lambda;
            if(!(lambda>0)||!finite_scalar(lambda)||!finite_scalar(nu))
                throw std::runtime_error("LM damping overflow after numerical factorization rejection");
            cuda_check(cudaMemcpyAsync(scalars.data,&used_lambda,sizeof(double),cudaMemcpyHostToDevice,s));
            lm_init_bounds<<<cc,128,0,s>>>(v,l.device,view,l.d_g.data);
            cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(s));
        };
        for(;;){
            ++bound_passes;
            start(s);lm_fill_local<<<cc,128,0,s>>>(l.device,view,l.d_h.data,l.d_g.data);cuda_check(cudaGetLastError());
            dense_check(cusolverDnDpotrfBatched(dense,CUBLAS_FILL_MODE_LOWER,m,factor_ptrs.data,m,factor_info.data,cc));
            ++dense_checks;
            if(!check_dense(factor_info.data,cc,s,true)){timings[2]+=finish(s);reject_factor();continue;}
            dense_check(cusolverDnDpotrsBatched(dense,CUBLAS_FILL_MODE_LOWER,m,1,solve_a_ptrs.data,m,solve_b_ptrs.data,m,solve_info.data,cc*(q+1)));
            ++dense_checks;check_dense(solve_info.data,1,s);
            lm_reduce<<<(q*q+q+127)/128,128,0,s>>>(l.device,view,l.d_h.data,l.d_shared.data);cuda_check(cudaGetLastError());
            timings[2]+=finish(s);
            if(!analysis_ready){
                start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_ANALYSIS,config,data,A,X,B));check_sparse(s);
                size_t written=0;sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_MEMORY_ESTIMATES,estimates,sizeof(estimates),&written));
                sparse_check(cudssDataGet(sparse,data,CUDSS_DATA_LU_NNZ,&factor_nnz,sizeof(factor_nnz),&written));
                size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
                if(estimates[1]>device_budget-double(device_bytes())||estimates[1]>free||estimates[3]>host_budget)
                    throw std::runtime_error("cuDSS symbolic factor memory exceeds current budget");
                analysis_ready=true;++analyses;timings[3]+=finish(s);
            }
            start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_FACTORIZATION,config,data,A,X,B));
            bool factor_ok=check_sparse(s,true);timings[4]+=finish(s);
            if(!factor_ok){reject_factor();continue;}
            start(s);sparse_check(cudssExecute(sparse,CUDSS_PHASE_SOLVE,config,data,A,X,B));check_sparse(s);timings[5]+=finish(s);
            start(s);lm_backsub<<<cc,128,0,s>>>(l.device,view);lm_feasible<<<1,1,0,s>>>(v,l.device,view);cuda_check(cudaGetLastError());
            read_scalars(values,s);++scalar_copies;timings[6]+=finish(s);
            if(values[7]||!finite_scalar(values[4])||values[4]>linear_tolerance)throw std::runtime_error("nonfinite step or excessive cuDSS residual");
            if(values[5]==0)break;
        }
        start(s);cuda_check(cudaMemcpyAsync(trial_input.data,v.board,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));
        size_t n=std::max(v.n,3*v.nc);
        lm_retract<<<static_cast<unsigned>((n+127)/128),128,0,s>>>(v,trial,l.device,view);
        if(view.nlocal)lm_actual_locals<<<(view.nlocal+127)/128,128,0,s>>>(v,trial,l.device,view);
        lm_prediction<<<cc,1,0,s>>>(l.device,view,l.d_h.data,l.d_g.data);cuda_check(cudaGetLastError());timings[7]=finish(s);
        start(s);evaluate_kernel<<<static_cast<unsigned>((v.n+127)/128),128,0,s>>>(trial,trial_output.data,trial_status.data);
        lm_trial_whiten<<<l.host.ng,128,0,s>>>(l.device,trial_output.data,trial_white.data);
        loss_kernel<<<(l.host.ng+127)/128,128,0,s>>>(l.device,trial_white.data,trial_stats.data);
        lm_finish<<<1,1,0,s>>>(trial,l.device,view,trial_status.data,trial_stats.data);
        cuda_check(cudaGetLastError());timings[8]=finish(s);trial_available=true;
        start(s);read_scalars(values,s);++scalar_copies;timings[9]=finish(s);
        int reason=0;
        if(values[8]||!finite_scalar(values[10]))reason=3;
        else if(!(values[11]>0)||!(values[3]>0)||!finite_scalar(values[11]))reason=2;
        else if(!(values[12]>0)||!(values[13]>eta))reason=1;
        bool accept=reason==0;++trials;
        if(accept){
            start(s);cuda_check(cudaMemcpyAsync(const_cast<double*>(v.board),trial_input.data,input_count*sizeof(double),cudaMemcpyDeviceToDevice,s));
            timings[10]=finish(s);++accepted;l.gpu_ready=false;l.cpu_ready=false;
            lambda=used_lambda*std::max(1./3.,1.-std::pow(2*values[13]-1,3));nu=2.;
        }else{lambda=used_lambda*nu;nu*=2.;}
        std::copy(values,values+24,output);
        output[15]=accept;output[16]=reason;output[17]=lambda;output[18]=nu;
        output[19]=bound_passes;output[20]=analyses;output[21]=trials;output[22]=accepted;
        output[23]=scalar_copies*24*sizeof(double)+dense_checks*2*sizeof(int);
        output[24]=double(device_bytes());output[25]=double(estimates[0]);output[26]=double(estimates[1]);
        output[27]=double(estimates[2]);output[28]=double(estimates[3]);output[29]=double(factor_nnz);output[30]=eta;output[31]=values[23];
        timings[0]=millis(wall);return accept;
    }
};
