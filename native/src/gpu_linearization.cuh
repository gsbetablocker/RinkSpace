/* Included inside core_gpu.cu's implementation namespace. Same resident
 * Problem, stream, source inputs and native mathematics as G1. */
struct GroupPlan {int begin,count,rank,weight,component,white;};
struct ComponentPlan {int local_begin,local_count,group_begin,group_count,dim,h,g;};
static_assert(sizeof(GroupPlan)==6*sizeof(int));
static_assert(sizeof(ComponentPlan)==7*sizeof(int));
inline void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct DeviceBuffer {
    T* data=nullptr;size_t count=0;
    ~DeviceBuffer(){cudaFree(data);}
    void allocate(size_t n){count=n;if(n)cuda_check(cudaMalloc(&data,n*sizeof(T)));}
    void upload(const std::vector<T>& v,cudaStream_t stream){allocate(v.size());if(count)cuda_check(cudaMemcpyAsync(data,v.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));}
    size_t bytes()const{return count*sizeof(T);}
};
struct LinearView {
    const GroupPlan* groups;const ComponentPlan* components;
    const int *members,*component_groups,*row_local,*row_local_column;
    const double *directions,*scales,*weights;
    int q,ng,ncomponents;double delta;
};
RS_HD inline void linearize_row(const View& v,const LinearView& l,size_t row,double* output,int* status,double* jac){
    evaluate_row(v,row,output,status);
    for(int j=0;j<2*(l.q+1);++j)jac[j]=std::numeric_limits<double>::quiet_NaN();
    if(*status)return;
    Vec3 xyz{},tangent{};auto st=geometry_row(v,row,xyz,tangent);
    if(st!=Status::ok){*status=static_cast<int>(st);return;}
    MotionProjection root{{output[3],output[4]},output[5],output[6],output[7],output[8]};
    double partials[8],weights[4];int cell;
    st=motion_point_partials(motion_rig(v.rig),{v.controls,v.nc,v.first,v.dt},v.height,
        xyz,tangent,v.times[row],root,partials,cell,weights);
    if(st!=Status::ok){*status=static_cast<int>(st);return;}
    for(int d=0;d<2;++d){
        for(int q=0;q<l.q;++q){
            double total=0.;
            for(int channel=0;channel<3;++channel){
                double basis=0.;for(int k=0;k<4;++k)basis+=weights[k]*l.directions[(3*(cell-1+k)+channel)*l.q+q];
                total+=partials[4*d+channel]*basis;
            }
            jac[d*(l.q+1)+q]=total;
        }
        jac[d*(l.q+1)+l.q]=l.row_local[row]<0?0.:partials[4*d+3]*l.scales[l.row_local[row]];
    }
    if(!finite_array(jac,2*(l.q+1)))*status=static_cast<int>(Status::nonfinite);
}
RS_HD inline double whiten_element(const LinearView& l,int gi,int k,int col,const double* output,const double* jac){
    const auto g=l.groups[gi];double sum=0.;
    for(int j=0;j<g.count;++j){
        const int row=l.members[g.begin+j];
        for(int d=0;d<2;++d){
            double value=0.;
            if(col==0)value=output[11*row+9+d];
            else if(col<=l.q)value=jac[(2*row+d)*(l.q+1)+col-1];
            else if(l.row_local_column[row]==col-l.q-1)value=jac[(2*row+d)*(l.q+1)+l.q];
            sum+=l.weights[g.weight+k*2*g.count+2*j+d]*value;
        }
    }
    return sum;
}
RS_HD inline void group_loss(const LinearView& l,int gi,const double* white,double* stats){
    const auto g=l.groups[gi];const int dim=l.components[g.component].dim;
    double s=0.;for(int k=0;k<g.rank;++k){const double r=white[g.white+k*(dim+1)];s+=r*r;}
    double weight=1.,rho=s;
    if(g.rank==0){weight=0.;rho=0.;}
    else if(s>g.rank*l.delta*l.delta){const double rms=std::sqrt(s/g.rank);weight=l.delta/rms;rho=g.rank*(2*l.delta*rms-l.delta*l.delta);}
    stats[0]=s;stats[1]=weight;stats[2]=.5*rho;
}
RS_HD inline double block_element(const LinearView& l,int ci,int row,int col,bool gradient,const double* white,const double* group_stats){
    const auto component=l.components[ci];double sum=0.;
    for(int j=0;j<component.group_count;++j){
        const int gi=l.component_groups[component.group_begin+j];const auto group=l.groups[gi];
        const double weight=group_stats[3*gi+1];
        for(int k=0;k<group.rank;++k){
            const double* w=white+group.white+k*(component.dim+1);
            sum+=weight*w[row+1]*w[gradient?0:col+1];
        }
    }
    return sum;
}
__global__ void linearize_kernel(View v,LinearView l,double* output,int* status,double* jac){
    const size_t row=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row<v.n)linearize_row(v,l,row,output+11*row,status+row,jac+row*2*(l.q+1));
}
__global__ void whiten_kernel(LinearView l,const double* output,const double* jac,double* white){
    const int gi=blockIdx.x;const auto g=l.groups[gi];const int dim=l.components[g.component].dim;
    for(int i=threadIdx.x;i<g.rank*(dim+1);i+=blockDim.x)
        white[g.white+i]=whiten_element(l,gi,i/(dim+1),i%(dim+1),output,jac);
}
__global__ void loss_kernel(LinearView l,const double* white,double* stats){
    const int gi=blockIdx.x*blockDim.x+threadIdx.x;if(gi<l.ng)group_loss(l,gi,white,stats+3*gi);
}
__global__ void block_kernel(LinearView l,const double* white,const double* stats,double* h,double* gradient){
    const int ci=blockIdx.x;const auto component=l.components[ci];const int d=component.dim;
    for(int i=threadIdx.x;i<d*d+d;i+=blockDim.x){
        if(i<d*d)h[component.h+i]=block_element(l,ci,i/d,i%d,false,white,stats);
        else gradient[component.g+i-d*d]=block_element(l,ci,i-d*d,0,true,white,stats);
    }
}
__global__ void shared_kernel(LinearView l,const double* h,const double* gradient,double* shared){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=l.q*l.q+l.q)return;
    double sum=0.;for(int ci=0;ci<l.ncomponents;++ci){const auto c=l.components[ci];
        sum+=i<l.q*l.q?h[c.h+(i/l.q)*c.dim+i%l.q]:gradient[c.g+i-l.q*l.q];}
    shared[i]=sum;
}
__global__ void summary_kernel(size_t n,const int* statuses,int ng,const double* stats,double* summary){
    if(threadIdx.x||blockIdx.x)return;
    double failed=0.,raw=0.,robust=0.,down=0.;
    for(size_t i=0;i<n;++i)failed+=statuses[i]!=0;
    for(int g=0;g<ng;++g){raw+=.5*stats[3*g];robust+=stats[3*g+2];down+=stats[3*g+1]<1.;}
    summary[0]=failed;summary[1]=raw;summary[2]=robust;summary[3]=down;
}
struct Linearization {
    std::vector<double> directions,weights,scales;
    std::vector<int> members,row_local,row_local_column,component_groups,component_locals;
    std::vector<GroupPlan> groups;
    std::vector<ComponentPlan> components;
    std::vector<double> cpu_output,cpu_jac,cpu_white,cpu_stats,cpu_h,cpu_g,cpu_shared;
    std::vector<int> cpu_status;
    DeviceBuffer<double> d_directions,d_weights,d_scales,d_jac,d_white,d_stats,d_h,d_g,d_shared,d_summary;
    DeviceBuffer<int> d_members,d_row_local,d_row_local_column,d_component_groups;
    DeviceBuffer<GroupPlan> d_groups;DeviceBuffer<ComponentPlan> d_components;
    cudaEvent_t event[6]{};
    LinearView host{},device{};
    size_t h_count=0,g_count=0,white_count=0;
    bool cpu_ready=false,gpu_ready=false;
    ~Linearization(){for(auto e:event)if(e)cudaEventDestroy(e);}
    static int bounded(size_t n){if(n>static_cast<size_t>(std::numeric_limits<int>::max()))throw std::invalid_argument("G2 packed layout exceeds int32 support");return static_cast<int>(n);}
    void prepare(const View& v,int q,const double* dirs,size_t ndirs,const int* ids,size_t nids,
        const double* scale,size_t nlocal,const int* offsets,int ng,const int* rows,size_t nrows,
        const int* ranks,const int* woffsets,const double* w,size_t nw,double delta,cudaStream_t stream){
        if(q<=0||ng<=0||ndirs!=v.nc*3*q||nids!=v.n||nrows!=v.n||!ids||!offsets||!rows||!ranks||!woffsets||
           !finite_array(dirs,ndirs)||!finite_array(w,nw)||!finite_array(scale,nlocal)||!finite_scalar(delta)||delta<=0)
            throw std::invalid_argument("invalid G2 plan shape or finite inputs");
        if(offsets[0]!=0||woffsets[0]!=0||offsets[ng]!=static_cast<int>(nrows)||woffsets[ng]!=static_cast<int>(nw))
            throw std::invalid_argument("G2 group offsets do not cover source rows/W");
        for(size_t k=0;k<nlocal;++k)if(scale[k]<=0)throw std::invalid_argument("station scale must be positive");
        directions.assign(dirs,dirs+ndirs);weights.assign(w,w+nw);scales.assign(scale,scale+nlocal);
        row_local.assign(ids,ids+nids);members.assign(rows,rows+nrows);
        std::vector<int> parent(nlocal),claimed(v.n,0),local_seen(nlocal,-1);
        std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
        for(size_t row=0;row<v.n;++row){
            const int local=ids[row];
            if(local< -1||local>=static_cast<int>(nlocal))throw std::invalid_argument("station identity out of range");
            if(local>=0){
                const int old=local_seen[local];
                if(old>=0&&(v.codes[old]!=v.codes[row]||v.stations[old]!=v.stations[row]))
                    throw std::invalid_argument("shared material station has conflicting entity/value");
                local_seen[local]=static_cast<int>(row);
            }
        }
        for(int seen:local_seen)if(seen<0)throw std::invalid_argument("unreferenced local variable");
        for(int gi=0;gi<ng;++gi){
            const int begin=offsets[gi],end=offsets[gi+1],rank=ranks[gi];
            if(begin<0||end<=begin||end>static_cast<int>(nrows)||rank<0||woffsets[gi]<0||
               static_cast<size_t>(woffsets[gi+1]-woffsets[gi])!=static_cast<size_t>(rank)*2*(end-begin))
                throw std::invalid_argument("malformed complete correlation group");
            int first=-1;
            for(int j=begin;j<end;++j){const int row=rows[j];
                if(row<0||row>=static_cast<int>(v.n)||claimed[row]++)throw std::invalid_argument("correlation rows overlap/outside problem");
                if(ids[row]>=0){const int r=root(ids[row]);if(first<0)first=r;else parent[r]=root(first);}
            }
            groups.push_back({begin,end-begin,rank,woffsets[gi],-1,0});
        }
        for(int count:claimed)if(count!=1)throw std::invalid_argument("correlation partition drops source rows");
        std::map<int,int> by_root;std::vector<std::vector<int>> locals,group_ids;
        for(size_t local=0;local<nlocal;++local){
            const int r=root(static_cast<int>(local));auto inserted=by_root.emplace(r,static_cast<int>(by_root.size()));
            if(inserted.second){locals.emplace_back();group_ids.emplace_back();}
            locals[inserted.first->second].push_back(static_cast<int>(local));
        }
        int fixed=-1;row_local_column.assign(v.n,-1);
        for(int gi=0;gi<ng;++gi){
            auto& g=groups[gi];int ci=-1;
            for(int j=0;j<g.count;++j){const int local=ids[members[g.begin+j]];if(local>=0){ci=by_root.at(root(local));break;}}
            if(ci<0){if(fixed<0){fixed=static_cast<int>(locals.size());locals.emplace_back();group_ids.emplace_back();}ci=fixed;}
            g.component=ci;group_ids[ci].push_back(gi);
        }
        std::vector<int> local_column(nlocal,-1);
        for(size_t ci=0;ci<locals.size();++ci){
            const auto& ll=locals[ci];const auto& gg=group_ids[ci];const int dim=q+static_cast<int>(ll.size());
            components.push_back({bounded(component_locals.size()),static_cast<int>(ll.size()),bounded(component_groups.size()),static_cast<int>(gg.size()),dim,bounded(h_count),bounded(g_count)});
            h_count+=static_cast<size_t>(dim)*dim;g_count+=dim;
            for(size_t j=0;j<ll.size();++j)local_column[ll[j]]=static_cast<int>(j);
            component_locals.insert(component_locals.end(),ll.begin(),ll.end());component_groups.insert(component_groups.end(),gg.begin(),gg.end());
        }
        bounded(h_count);bounded(g_count);
        for(size_t row=0;row<v.n;++row)if(ids[row]>=0)row_local_column[row]=local_column[ids[row]];
        for(auto& g:groups){g.white=bounded(white_count);white_count+=static_cast<size_t>(g.rank)*(components[g.component].dim+1);}
        bounded(white_count);
        host={groups.data(),components.data(),members.data(),component_groups.data(),row_local.data(),row_local_column.data(),
            directions.data(),scales.data(),weights.data(),q,ng,static_cast<int>(components.size()),delta};
        d_directions.upload(directions,stream);d_weights.upload(weights,stream);d_scales.upload(scales,stream);
        d_members.upload(members,stream);d_row_local.upload(row_local,stream);d_row_local_column.upload(row_local_column,stream);
        d_component_groups.upload(component_groups,stream);d_groups.upload(groups,stream);d_components.upload(components,stream);
        d_jac.allocate(v.n*2*(q+1));d_white.allocate(white_count);d_stats.allocate(3*ng);
        d_h.allocate(h_count);d_g.allocate(g_count);d_shared.allocate(q*q+q);d_summary.allocate(4);
        device={d_groups.data,d_components.data,d_members.data,d_component_groups.data,d_row_local.data,d_row_local_column.data,
            d_directions.data,d_scales.data,d_weights.data,q,ng,static_cast<int>(components.size()),delta};
        for(auto& e:event)cuda_check(cudaEventCreate(&e));
    }
    size_t device_bytes()const{return d_directions.bytes()+d_weights.bytes()+d_scales.bytes()+d_members.bytes()+d_row_local.bytes()+d_row_local_column.bytes()+d_component_groups.bytes()+d_groups.bytes()+d_components.bytes()+d_jac.bytes()+d_white.bytes()+d_stats.bytes()+d_h.bytes()+d_g.bytes()+d_shared.bytes()+d_summary.bytes();}
    void cpu(const View& v,double* summary){
        const auto& l=host;cpu_output.resize(v.n*11);cpu_status.resize(v.n);cpu_jac.resize(v.n*2*(l.q+1));
        cpu_white.resize(white_count);cpu_stats.resize(3*l.ng);cpu_h.resize(h_count);cpu_g.resize(g_count);cpu_shared.resize(l.q*l.q+l.q);
        for(size_t row=0;row<v.n;++row)linearize_row(v,l,row,cpu_output.data()+11*row,cpu_status.data()+row,cpu_jac.data()+row*2*(l.q+1));
        for(int gi=0;gi<l.ng;++gi){const auto group=groups[gi];const int d=components[group.component].dim;
            for(int k=0;k<group.rank;++k)for(int col=0;col<d+1;++col)cpu_white[group.white+k*(d+1)+col]=whiten_element(l,gi,k,col,cpu_output.data(),cpu_jac.data());
            group_loss(l,gi,cpu_white.data(),cpu_stats.data()+3*gi);
        }
        for(int ci=0;ci<l.ncomponents;++ci){const auto cc=components[ci];
            for(int i=0;i<cc.dim;++i){for(int j=0;j<cc.dim;++j)cpu_h[cc.h+i*cc.dim+j]=block_element(l,ci,i,j,false,cpu_white.data(),cpu_stats.data());
                cpu_g[cc.g+i]=block_element(l,ci,i,0,true,cpu_white.data(),cpu_stats.data());}
        }
        std::fill(cpu_shared.begin(),cpu_shared.end(),0.);
        for(const auto& cc:components)for(int i=0;i<l.q*l.q+l.q;++i)
            cpu_shared[i]+=i<l.q*l.q?cpu_h[cc.h+(i/l.q)*cc.dim+i%l.q]:cpu_g[cc.g+i-l.q*l.q];
        std::fill(summary,summary+4,0.);
        for(auto st:cpu_status)summary[0]+=st!=0;
        for(int gi=0;gi<l.ng;++gi){summary[1]+=.5*cpu_stats[3*gi];summary[2]+=cpu_stats[3*gi+2];summary[3]+=cpu_stats[3*gi+1]<1.;}
        cpu_ready=summary[0]==0;
    }
};
