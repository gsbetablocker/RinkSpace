/* Reduced bounded quadratic merit for scalar local coordinates.
 * Every line evaluation minimizes the SAME damped quadratic over the original
 * station intervals. The shared line is safeguarded before changing its active
 * set, preventing the full-step face cycle and uphill conditional completion.
 */
__global__ void lm_qp_save_active(LinearView l,SolverView s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.nlocal)return;
    s.applied[l.q+i]=s.blocked[i]?(s.active_delta[i]==s.delta_lower[i]?1.:2.):0.;
}
__global__ void lm_qp_line_components(LinearView l,SolverView s,const double* h,const double* g,double alpha,double* out){
    int ci=blockIdx.x*blockDim.x+threadIdx.x;double slope=0.,curvature=0.,scale=0.;
    if(ci<l.ncomponents){auto c=l.components[ci];int q=c.q_count;double local=0.,bdir=0.,diagonal=1.;bool free=false;
        if(c.local_count==1){int id=s.locals[c.local_begin];double gradient=g[c.g+q];
            for(int j=0;j<q;++j){int k=l.columns[c.q_begin+j];double b=component_h(h,c,q,j);
                gradient+=b*(s.applied[k]+alpha*(s.step[k]-s.applied[k]));bdir+=b*(s.step[k]-s.applied[k]);}
            diagonal=component_h(h,c,q,q)+s.scalars[0]*s.damping[l.q+id];
            local=s.delta_lower[id]==s.delta_upper[id]?s.delta_lower[id]:fmax(s.delta_lower[id],fmin(s.delta_upper[id],-gradient/diagonal));
            free=local>s.delta_lower[id]&&local<s.delta_upper[id];
        }
        if(c.normal_rank==2){double value[2]{},direction[2]{};
            for(int j=0;j<q;++j){int k=l.columns[c.q_begin+j];double x=s.applied[k]+alpha*(s.step[k]-s.applied[k]),d=s.step[k]-s.applied[k];
                slope+=g[c.g+j]*d;for(int row=0;row<2;++row){double a=h[c.packed_h+size_t(row)*(c.normal_stride?c.normal_stride:c.dim)+j];value[row]+=a*x;direction[row]+=a*d;}}
            for(int row=0;row<2;++row){if(c.local_count)value[row]+=h[c.packed_h+size_t(row)*(c.normal_stride?c.normal_stride:c.dim)+q]*local;
                slope+=value[row]*direction[row];curvature+=direction[row]*direction[row];}
        }else{
            for(int i=0;i<q;++i){int ki=l.columns[c.q_begin+i];double di=s.step[ki]-s.applied[ki],gradient=g[c.g+i],along=0.;
                for(int j=0;j<q;++j){int kj=l.columns[c.q_begin+j];double a=component_h(h,c,i,j);
                    gradient+=a*(s.applied[kj]+alpha*(s.step[kj]-s.applied[kj]));along+=a*(s.step[kj]-s.applied[kj]);}
                if(c.local_count)gradient+=component_h(h,c,i,q)*local;slope+=gradient*di;curvature+=di*along;}
        }
        if(free)curvature-=bdir*bdir/diagonal;scale=fabs(slope);
    }
    slope=block_reduce<0>(slope);curvature=block_reduce<0>(curvature);scale=block_reduce<0>(scale);
    if(!threadIdx.x){atomicAdd(out,slope);atomicAdd(out+1,curvature);atomicAdd(out+2,scale);}
}
__global__ void lm_qp_line_damping(LinearView l,SolverView s,double alpha,double* out){
    int j=blockIdx.x*blockDim.x+threadIdx.x;double slope=0.,curvature=0.;
    if(j<l.q){double d=s.step[j]-s.applied[j],x=s.applied[j]+alpha*d,a=s.scalars[0]*s.damping[j];slope=a*x*d;curvature=a*d*d;}
    double scale=block_reduce<0>(fabs(slope));slope=block_reduce<0>(slope);curvature=block_reduce<0>(curvature);
    if(!threadIdx.x){atomicAdd(out,slope);atomicAdd(out+1,curvature);atomicAdd(out+2,scale);}
}
__global__ void lm_qp_line_apply(LinearView l,SolverView s,double alpha,double* out){
    int j=blockIdx.x*blockDim.x+threadIdx.x;double change=0.;if(j<l.q){double x=s.applied[j]+alpha*(s.step[j]-s.applied[j]);
        change=fabs(x-s.applied[j]);s.step[j]=s.applied[j]=x;}
    change=block_reduce<1>(change);if(!threadIdx.x)atomic_max_nonnegative(out+3,change);
}
__global__ void lm_qp_conditional_active(LinearView l,SolverView s,const double* h,const double* g,int* result){
    int ci=blockIdx.x*blockDim.x+threadIdx.x;if(ci>=l.ncomponents)return;auto c=l.components[ci];if(!c.local_count)return;
    if(c.local_count!=1){atomicAdd(result+1,1);return;}int id=s.locals[c.local_begin],q=c.q_count;double gradient=g[c.g+q];
    for(int j=0;j<q;++j)gradient+=component_h(h,c,q,j)*s.step[l.columns[c.q_begin+j]];
    const double diagonal=component_h(h,c,q,q)+s.scalars[0]*s.damping[l.q+id],lo=s.delta_lower[id],hi=s.delta_upper[id];
    if(!(lo<=hi)||(!finite_scalar(gradient))||(lo!=hi&&(!(diagonal>0)||!finite_scalar(diagonal)))){atomicAdd(result+1,1);return;}
    double x=lo==hi?lo:fmax(lo,fmin(hi,-gradient/diagonal));int code=x==lo?1:(x==hi?2:0);
    if(code!=int(s.applied[l.q+id]))atomicAdd(result,1);
    s.step[l.q+id]=x;s.blocked[id]=code!=0;s.active_delta[id]=code?x:0.;
}
