/* Resident sparse factor plan. A row stores only directions reachable through
 * its declared RS support. A connected station component uses their union.
 * Correlation groups and shared material stations are never split. */
struct GroupPlan {int begin,count,rank,weight,component;size_t white;};
struct ComponentPlan {
    int local_begin,local_count,group_begin,group_count,dim;
    size_t h,g;int q_begin,q_count;size_t scatter,packed_h;
    int normal_rank=0,normal_stride=0;
};
struct RowPlan {int begin,count;size_t jac;};
struct ColumnPattern {int begin,count;size_t scatter;};
inline void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline size_t checked_product(size_t a,size_t b){
    if(b&&a>std::numeric_limits<size_t>::max()/b)throw std::invalid_argument("native sparse size overflow");
    return a*b;
}
template<class T> struct DeviceBuffer {
    T* data=nullptr;size_t count=0;
    ~DeviceBuffer(){cudaFree(data);}
    void allocate(size_t n){count=n;if(n)cuda_check(cudaMalloc(&data,checked_product(n,sizeof(T))));}
    void upload(const std::vector<T>& v,cudaStream_t stream){allocate(v.size());if(count)cuda_check(cudaMemcpyAsync(data,v.data(),bytes(),cudaMemcpyHostToDevice,stream));}
    size_t bytes()const{return count*sizeof(T);}
};
// Positive reductions use bitwise IEEE ordering. All public summaries are FP64.
__device__ inline void atomic_max_nonnegative(double* p,double x){atomicMax(reinterpret_cast<unsigned long long*>(p),__double_as_longlong(x));}
__device__ inline void atomic_min_nonnegative(double* p,double x){atomicMin(reinterpret_cast<unsigned long long*>(p),__double_as_longlong(x));}
template<int Op> __device__ double block_reduce(double value){
    __shared__ double work[128];int t=threadIdx.x;work[t]=value;__syncthreads();
    for(int d=blockDim.x/2;d;d/=2){if(t<d){if constexpr(Op==0)work[t]+=work[t+d];else if constexpr(Op==1)work[t]=fmax(work[t],work[t+d]);else work[t]=fmin(work[t],work[t+d]);}__syncthreads();}
    double result=work[0];__syncthreads();return result;
}
// Exact nonzero support for repeated state updates. The original dense maps
// remain authoritative for derivatives; column order and all nonzeros are kept.
struct DirectionRowMap { const int* offsets=nullptr; const int* columns=nullptr; };
RS_HD inline double direction_row_dot(const double* dense,int q,size_t row,DirectionRowMap map,const double* step,double alpha){
    double value=0.;
    if(map.offsets){for(int k=map.offsets[row];k<map.offsets[row+1];++k){int j=map.columns[k];value+=dense[row*size_t(q)+j]*(alpha*step[j]);}}
    else for(int j=0;j<q;++j)value+=dense[row*size_t(q)+j]*(alpha*step[j]);
    return value;
}
struct DirectionRowPlan {
    std::vector<int> offsets,columns;DeviceBuffer<int> d_offsets,d_columns;
    void prepare(const std::vector<double>& dense,int q){
        if(dense.empty())return;offsets.push_back(0);
        for(size_t row=0;row<dense.size()/q;++row){
            for(int j=0;j<q;++j)if(dense[row*size_t(q)+j]!=0.)columns.push_back(j);
            if(columns.size()>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("direction map exceeds int32 support");
            offsets.push_back(int(columns.size()));
        }
    }
    void upload(cudaStream_t stream){d_offsets.upload(offsets,stream);d_columns.upload(columns,stream);}
    DirectionRowMap host()const{return {offsets.empty()?nullptr:offsets.data(),columns.data()};}
    DirectionRowMap device()const{return {d_offsets.data,d_columns.data};}
    size_t bytes()const{return sizeof(int)*(offsets.size()+columns.size());}
};
struct LinearView {
    const GroupPlan* groups;const ComponentPlan* components;const RowPlan* row_plan;
    const int *members,*component_groups,*row_local,*row_local_column,*columns,*scatter,*csr_offsets,*csr_columns;
    const double *directions,*scales,*weights;
    int q,ng,ncomponents;size_t nnz;double delta;
    const double *rig_directions=nullptr,*rig_lower=nullptr,*rig_upper=nullptr;double* rig_tangents=nullptr;double max_readout=0.;
    const double *scene_directions=nullptr,*scene_lower=nullptr,*scene_upper=nullptr;BoardCourseValue<LocalJet>* scene_courses=nullptr;const int* scene_active=nullptr;const int* scene_chart=nullptr;const double* scene_steps=nullptr;int mount_side=0;int* local_sides=nullptr;
    CameraEnvelope envelope{};
    const CameraEnvelope* camera_envelopes=nullptr;const int* mount_sides=nullptr;
    bool pixel_residuals=false;
    DirectionRowMap motion_rows{},rig_rows{},scene_rows{};
};
RS_HD inline LinearView camera_linear_view(LinearView l,size_t nr,size_t camera){
    if(l.rig_directions)l.rig_directions+=camera*(nr==43?31:27)*l.q;
    if(l.rig_rows.offsets)l.rig_rows.offsets+=camera*(nr==43?31:27);
    if(l.rig_tangents)l.rig_tangents+=camera*38*l.q;
    if(l.rig_lower)l.rig_lower+=camera*nr;if(l.rig_upper)l.rig_upper+=camera*nr;
    if(l.camera_envelopes)l.envelope=l.camera_envelopes[camera];
    if(l.mount_sides)l.mount_side=l.mount_sides[camera];
    return l;
}
RS_HD inline int column_position(const int* columns,int begin,int count,int value){
    int lo=0,hi=count;while(lo<hi){int m=lo+(hi-lo)/2;if(columns[begin+m]<value)lo=m+1;else hi=m;}
    return lo<count&&columns[begin+lo]==value?lo:-1;
}
RS_HD inline size_t upper_index(int n,int i,int j){return size_t(i)*n-size_t(i)*(i-1)/2+j-i;}
// A single unweighted pixel factor has exactly two Jacobian rows. Store those
// rows rather than its quadratic-size outer product. Correlated/multi-row
// components retain packed symmetric storage. Public diagnostics stay dense.
RS_HD inline double component_h(const double* h,const ComponentPlan& c,int i,int j){
    if(c.normal_rank==2){const double* a=h+c.packed_h;const int stride=c.normal_stride?c.normal_stride:c.dim;return a[i]*a[j]+a[stride+i]*a[stride+j];}
    return h[c.packed_h+(i<=j?upper_index(c.dim,i,j):upper_index(c.dim,j,i))];
}
RS_HD inline void linearize_row(const View& source,const LinearView& plan,size_t row,double* output,int* status,double* jac){
    const auto camera=row_camera(source,row);const auto v=camera_view(source,camera);const auto l=camera_linear_view(plan,source.nr,camera);
    const auto rp=l.row_plan[row];int width=rp.count+1;
    double resolved[2];evaluate_row(v,row,output,status,resolved);
    for(int j=0;j<2*width;++j)jac[j]=std::numeric_limits<double>::quiet_NaN();
    if(*status)return;
    Vec3 xyz{},tangent{};auto st=geometry_at(v,row,resolved[0],xyz,tangent);
    if(st!=Status::ok){*status=static_cast<int>(st);return;}
    MotionProjection root{{output[3],output[4]},output[5],output[6],output[7],output[8]};
    double partials[8],weights[4];int cell;
    st=motion_point_partials(motion_rig(v.rig,v.nr==43?v.rig+38:nullptr),{v.controls,v.nc,v.first,v.dt},v.height,xyz,tangent,v.times[row],root,partials,cell,weights);
    if(st!=Status::ok){*status=static_cast<int>(st);return;}
    for(int d=0;d<2;++d){
        for(int j=0;j<rp.count;++j){int q=l.columns[rp.begin+j];double total=0.;
            for(int channel=0;channel<3;++channel){double basis=0.;
                for(int k=0;k<4;++k)basis+=weights[k]*l.directions[(3*size_t(cell-1+k)+channel)*l.q+q];
                total+=partials[4*d+channel]*basis;}
            jac[d*width+j]=total;
        }
        jac[d*width+rp.count]=l.row_local[row]<0?0.:partials[4*d+3]*l.scales[l.row_local[row]];
    }
    if(l.rig_directions||l.scene_directions){
        for(int j=0;j<rp.count;++j){int q=l.columns[rp.begin+j];bool active=false;Vec3 dx{};
            if(l.rig_directions)for(int k=0;k<(v.nr==43?31:27);++k)active|=l.rig_directions[size_t(k)*l.q+q]!=0.;
            if(l.scene_directions&&l.scene_active[q]){
                st=scene_geometry_direction(v,row,l.scene_directions,l.q,q,l.scene_courses,dx,resolved[0]);
                if(st!=Status::ok){*status=static_cast<int>(st);return;}
                active|=dx.x!=0||dx.y!=0||dx.z!=0;
            }
            if(!active)continue;
            double value[2];st=motion_rig_projection_direction(v.rig,l.rig_tangents,l.q,q,{v.controls,v.nc,v.first,v.dt},v.height,xyz,v.times[row],root,partials,value,dx,v.nr==43?v.rig+38:nullptr,(v.nr==43&&l.rig_directions)?l.rig_directions+size_t(27)*l.q:nullptr);
            if(st!=Status::ok){*status=static_cast<int>(st);return;}
            for(int d=0;d<2;++d)jac[d*width+j]+=value[d];
        }
    }
    if(v.profile_enabled&&v.profile_ids&&v.profile_ids[row]>=0){
        Vec2 uv{},qt{};st=profile_value(v.profiles+17*v.profile_ids[row],resolved[1],uv,qt);
        if(st!=Status::ok){*status=static_cast<int>(st);return;}
        double denominator=qt.y*partials[3]-qt.x*partials[7];
        if(std::abs(denominator)<1e-14){*status=static_cast<int>(Status::degenerate);return;}
        for(int j=0;j<rp.count;++j){double ds=-(qt.y*jac[j]-qt.x*jac[width+j])/denominator;
            jac[j]+=partials[3]*ds;jac[width+j]+=partials[7]*ds;}
        jac[rp.count]=jac[width+rp.count]=0.;
    }
    if(!finite_array(jac,2*width))*status=static_cast<int>(Status::nonfinite);
}
RS_HD inline double whiten_element(const LinearView& l,int gi,int k,int col,const double* output,const double* jac){
    const auto g=l.groups[gi];const auto c=l.components[g.component];double sum=0.;
    if(l.pixel_residuals){
        // One actual UV contributes its two raw pixel components. No source
        // covariance, confidence, group compression, or error allowance.
        int row=l.members[g.begin];const auto rp=l.row_plan[row];
        if(col==0)return output[11*size_t(row)+9+k];
        int pos=-1;
        if(col<=c.q_count)pos=column_position(l.columns,rp.begin,rp.count,l.columns[c.q_begin+col-1]);
        else if(l.row_local_column[row]==col-c.q_count-1)pos=rp.count;
        return pos<0?0.:jac[rp.jac+k*(rp.count+1)+pos];
    }
    for(int j=0;j<g.count;++j){int row=l.members[g.begin+j];const auto rp=l.row_plan[row];
        int pos=-1;
        if(col>0&&col<=c.q_count)pos=column_position(l.columns,rp.begin,rp.count,l.columns[c.q_begin+col-1]);
        else if(col>c.q_count&&l.row_local_column[row]==col-c.q_count-1)pos=rp.count;
        for(int d=0;d<2;++d){double value=col==0?output[11*size_t(row)+9+d]:(pos<0?0.:jac[rp.jac+d*(rp.count+1)+pos]);
            sum+=l.weights[size_t(g.weight)+size_t(k)*2*g.count+2*j+d]*value;}
    }
    return sum;
}
RS_HD inline void group_loss(const LinearView& l,int gi,const double* white,double* stats){
    const auto g=l.groups[gi];int dim=l.components[g.component].dim;double s=0.;
    for(int k=0;k<g.rank;++k){double r=white[g.white+size_t(k)*(dim+1)];s+=r*r;}
    double weight=1.,rho=s;
    if(g.rank==0){weight=0.;rho=0.;}
    else if(!l.pixel_residuals&&s>g.rank*l.delta*l.delta){double rms=std::sqrt(s/g.rank);weight=l.delta/rms;rho=g.rank*(2*l.delta*rms-l.delta*l.delta);}
    stats[0]=s;stats[1]=weight;stats[2]=.5*rho;
}
RS_HD inline double block_element(const LinearView& l,int ci,int row,int col,bool gradient,const double* white,const double* stats){
    const auto c=l.components[ci];double sum=0.;
    for(int j=0;j<c.group_count;++j){int gi=l.component_groups[c.group_begin+j];auto g=l.groups[gi];double weight=stats[3*size_t(gi)+1];
        for(int k=0;k<g.rank;++k){const double* w=white+g.white+size_t(k)*(c.dim+1);sum+=weight*w[row+1]*w[gradient?0:col+1];}}
    return sum;
}
// At an intrinsic polygon/warp kink use the admissible one-sided analytic
// tangent. Full original W and robust group weights select descent, including
// shared material stations across rows. A local kink minimum is active only
// for this linearization and is reconsidered after every global update.
RS_HD bool corner_tangents(const View& source,const LinearView& l,int row,const double* output,double* left,double* right){
    const auto v=row_camera_view(source,size_t(row));
    double lo,hi;bool corner;if(!station_piece(v,row,v.stations[row],lo,hi,corner)||!corner)return false;
    double e=std::min(256*std::numeric_limits<double>::epsilon()*(1+fabs(v.stations[row])),.25*std::min(v.stations[row]-lo,hi-v.stations[row]));
    if(!(e>0))return false;Vec3 x=load3(output+11*size_t(row)),unused{},tangent{};
    MotionProjection root{{output[11*size_t(row)+3],output[11*size_t(row)+4]},output[11*size_t(row)+5],output[11*size_t(row)+6],output[11*size_t(row)+7],output[11*size_t(row)+8]};
    for(int side=0;side<2;++side){if(geometry_at(v,row,v.stations[row]+(side?e:-e),unused,tangent)!=Status::ok)return false;
        double partials[8],basis[4];int cell;if(motion_point_partials(motion_rig(v.rig,v.nr==43?v.rig+38:nullptr),{v.controls,v.nc,v.first,v.dt},v.height,x,tangent,v.times[row],root,partials,cell,basis)!=Status::ok)return false;
        double* a=side?right:left;a[0]=partials[3]*l.scales[l.row_local[row]];a[1]=partials[7]*l.scales[l.row_local[row]];}
    return true;
}
RS_HD void choose_corner_station(const View& v,const LinearView& l,int ci,int a,const double* output,double* jac,const double* white,const double* stats){
    auto c=l.components[ci];int id=-1,representative=-1;
    for(int j=0;j<c.group_count&&id<0;++j){auto g=l.groups[l.component_groups[c.group_begin+j]];for(int k=0;k<g.count;++k){int row=l.members[g.begin+k];if(l.row_local_column[row]==a){id=l.row_local[row];representative=row;break;}}}
    if(id<0)return;l.local_sides[id]=0;double check_l[2],check_r[2];if(!corner_tangents(v,l,representative,output,check_l,check_r))return;
    double gl=0.,gr=0.,nl=0.,nr=0.;
    for(int j=0;j<c.group_count;++j){int gi=l.component_groups[c.group_begin+j];auto g=l.groups[gi];double weight=stats[3*size_t(gi)+1];
        for(int member=0;member<g.count;++member){int row=l.members[g.begin+member];if(l.row_local[row]!=id)continue;double left[2],right[2];if(!corner_tangents(v,l,row,output,left,right))return;
            for(int d=0;d<2;++d){double cotangent=0.;
                if(l.pixel_residuals)cotangent=white[g.white+size_t(d)*(c.dim+1)];
                else for(int k=0;k<g.rank;++k)cotangent+=l.weights[size_t(g.weight)+size_t(k)*2*g.count+2*member+d]*white[g.white+size_t(k)*(c.dim+1)];
                gl+=weight*cotangent*left[d];gr+=weight*cotangent*right[d];nl+=left[d]*left[d];nr+=right[d]*right[d];}}
    }
    double ml=gl>0?gl/std::sqrt(std::max(nl,1e-300)):0.,mr=gr<0?-gr/std::sqrt(std::max(nr,1e-300)):0.;
    int side=ml==0&&mr==0?2:(ml>mr?-1:1);l.local_sides[id]=side;
    for(int j=0;j<c.group_count;++j){auto g=l.groups[l.component_groups[c.group_begin+j]];for(int k=0;k<g.count;++k){int row=l.members[g.begin+k];if(l.row_local[row]!=id)continue;
        double left[2],right[2];if(!corner_tangents(v,l,row,output,left,right))continue;auto rp=l.row_plan[row];for(int d=0;d<2;++d)jac[rp.jac+d*(rp.count+1)+rp.count]=side==2?0.:(side<0?left[d]:right[d]);}}
}
__global__ void corner_station_kernel(View v,LinearView l,const double* output,double* jac,const double* white,const double* stats){
    int ci=blockIdx.x;auto c=l.components[ci];for(int a=threadIdx.x;a<c.local_count;a+=blockDim.x)choose_corner_station(v,l,ci,a,output,jac,white,stats);
}
__global__ void scene_direction_kernel(View v,LinearView l){
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<l.q)scene_chart_direction(v.scene_values,v.nscene,l.scene_chart,l.scene_steps,l.q,q,const_cast<double*>(l.scene_directions));
}
__global__ void scene_course_kernel(View v,LinearView l){
    int k=threadIdx.x;if(k>=6)return;LocalJet b[6];for(int j=0;j<6;++j)b[j]={v.board[j],j==k?1.:0.};
    l.scene_courses[k]=BoardCourseValue<LocalJet>({b[0],b[1],b[2],b[3],b[4],b[5]});
}
__global__ void rig_tangent_kernel(View v,LinearView l){
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<l.q)for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(l,v.nr,camera);motion_rig_tangent(cv.rig,cl.rig_directions,l.q,q,cl.rig_tangents);}
}
__global__ void linearize_kernel(View v,LinearView l,double* output,int* status,double* jac){
    size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row<v.n)linearize_row(v,l,row,output+11*row,status+row,jac+l.row_plan[row].jac);
}
__global__ void whiten_kernel(LinearView l,const double* output,const double* jac,double* white){
    int gi=blockIdx.x;auto g=l.groups[gi];int dim=l.components[g.component].dim;
    for(size_t i=threadIdx.x;i<size_t(g.rank)*(dim+1);i+=blockDim.x)white[g.white+i]=whiten_element(l,gi,int(i/(dim+1)),int(i%(dim+1)),output,jac);
}
// Corner selection changes only local station columns; r, q and IRLS stay fixed.
__global__ void whiten_corner_locals_kernel(LinearView l,const double* output,const double* jac,double* white){
    int gi=blockIdx.x;auto g=l.groups[gi];auto c=l.components[g.component];
    for(int a=threadIdx.x;a<c.local_count;a+=blockDim.x){bool changed=false;
        for(int k=0;k<g.count;++k){int row=l.members[g.begin+k];if(l.row_local_column[row]==a&&l.local_sides[l.row_local[row]]!=0){changed=true;break;}}
        if(changed)for(int k=0;k<g.rank;++k)white[g.white+size_t(k)*(c.dim+1)+c.q_count+1+a]=whiten_element(l,gi,k,c.q_count+1+a,output,jac);
    }
}
__global__ void loss_kernel(LinearView l,const double* white,double* stats){int gi=blockIdx.x*blockDim.x+threadIdx.x;if(gi<l.ng)group_loss(l,gi,white,stats+3*size_t(gi));}
__global__ void block_kernel(LinearView l,const double* white,const double* stats,double* h,double* gradient){
    int ci=blockIdx.x;auto c=l.components[ci];size_t square=size_t(c.dim)*c.dim;
    if(c.normal_rank==2){auto g=l.groups[l.component_groups[c.group_begin]];
        for(int i=threadIdx.x;i<c.dim;i+=blockDim.x){
            if(!c.normal_stride){h[c.packed_h+i]=white[g.white+1+i];h[c.packed_h+c.dim+i]=white[g.white+c.dim+2+i];}
            gradient[c.g+i]=block_element(l,ci,i,0,true,white,stats);}
        return;}
    for(size_t i=threadIdx.x;i<square+c.dim;i+=blockDim.x){
        if(i<square){int row=int(i/c.dim),col=int(i%c.dim);if(col>=row)
            h[c.packed_h+upper_index(c.dim,row,col)]=block_element(l,ci,row,col,false,white,stats);}
        else gradient[c.g+i-square]=block_element(l,ci,int(i-square),0,true,white,stats);}
}
__global__ void shared_kernel(LinearView l,const double* h,const double* gradient,double* shared){
    auto c=l.components[blockIdx.x];int n=c.q_count;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)atomicAdd(shared+l.nnz+l.columns[c.q_begin+i],gradient[c.g+i]);
    for(size_t k=threadIdx.x;k<size_t(n)*n;k+=blockDim.x){int i=int(k/n),j=int(k%n);if(j>=i)
        atomicAdd(shared+l.scatter[c.scatter+upper_index(n,i,j)],component_h(h,c,i,j));}
}
__global__ void summary_kernel(size_t n,const int* statuses,int ng,const double* stats,double* summary){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    double a=block_reduce<0>(i<n&&statuses[i]?1.:0.);
    double b=block_reduce<0>(i<size_t(ng)?.5*stats[3*i]:0.);
    double c=block_reduce<0>(i<size_t(ng)?stats[3*i+2]:0.);
    double d=block_reduce<0>(i<size_t(ng)&&stats[3*i+1]<1.?1.:0.);
    if(!threadIdx.x){atomicAdd(summary,a);atomicAdd(summary+1,b);atomicAdd(summary+2,c);atomicAdd(summary+3,d);}
}
struct Linearization {
    std::vector<double> directions,weights,scales,rig_directions,rig_tangents,rig_lower,rig_upper;
    DeviceBuffer<double> d_rig_directions,d_rig_tangents,d_rig_lower,d_rig_upper;
    DirectionRowPlan motion_rows,rig_rows,scene_rows;
    std::vector<double> scene_directions,scene_lower,scene_upper;std::vector<BoardCourseValue<LocalJet>> scene_courses;std::vector<int> scene_active,scene_chart;DeviceBuffer<int> d_scene_active,d_scene_chart;std::vector<double> scene_effective;DeviceBuffer<double> d_scene_effective;
    DeviceBuffer<double> d_scene_directions,d_scene_lower,d_scene_upper;DeviceBuffer<BoardCourseValue<LocalJet>> d_scene_courses;
    std::vector<int> local_sides;DeviceBuffer<int> d_local_sides;
    std::vector<int> members,row_local,row_local_column,component_groups,component_locals,columns,scatter,csr_offsets,csr_columns;
    std::vector<GroupPlan> groups;std::vector<ComponentPlan> components;std::vector<RowPlan> row_plan;
    std::vector<double> cpu_output,cpu_jac,cpu_white,cpu_stats,cpu_h,cpu_g,cpu_shared;
    std::vector<int> cpu_status;
    DeviceBuffer<double> d_directions,d_weights,d_scales,d_jac,d_white,d_stats,d_h,d_g,d_shared,d_summary;
    DeviceBuffer<int> d_members,d_row_local,d_row_local_column,d_component_groups,d_columns,d_scatter,d_csr_offsets,d_csr_columns;
    DeviceBuffer<GroupPlan> d_groups;DeviceBuffer<ComponentPlan> d_components;DeviceBuffer<RowPlan> d_row_plan;
    cudaEvent_t event[6]{};LinearView host{},device{};
    bool rank2_white_storage=false;
    size_t h_count=0,packed_h_count=0,g_count=0,white_count=0,jac_count=0;int max_row_q=0,max_component_q=0;
    bool cpu_ready=false,gpu_ready=false;
    size_t generation=0; // Objective/configuration/layout are immutable on this plan.
    ~Linearization(){for(auto e:event)if(e)cudaEventDestroy(e);}
    static int bounded(size_t n){if(n>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("sparse index count exceeds int32 identities");return int(n);}
    void prepare(const View& v,int q,const double* dirs,size_t ndirs,const int* ids,size_t nids,
        const double* scale,size_t nlocal,const int* offsets,int ng,const int* rows,size_t nrows,
        const int* ranks,const int* woffsets,const double* w,size_t nw,double delta,cudaStream_t stream,const double* rig_dirs=nullptr,int rig_q=0,const double* rig_lo=nullptr,const double* rig_hi=nullptr,double readout_support=0.,const double* scene_dirs=nullptr,int scene_q=0,const double* scene_lo=nullptr,const double* scene_hi=nullptr,const int* charts=nullptr,int mount_side=0,bool pixels=false){
        if(q<=0||ng<=0||ndirs!=checked_product(v.nc*3,size_t(q))||nids!=v.n||nrows!=v.n||!ids||!offsets||!rows||!ranks||!woffsets||
           !finite_array(dirs,ndirs)||(nw&&!finite_array(w,nw))||(nlocal&&!finite_array(scale,nlocal))||!finite_scalar(delta)||(!pixels&&delta<=0)||(pixels&&(nw!=0||delta!=0)))
            throw std::invalid_argument("invalid G2 plan shape or finite inputs");
        bounded(v.n);bounded(nlocal);bounded(nw);
        if(offsets[0]!=0||woffsets[0]!=0||offsets[ng]!=int(nrows)||woffsets[ng]!=int(nw))throw std::invalid_argument("G2 group offsets do not cover source rows/W");
        for(size_t k=0;k<nlocal;++k)if(scale[k]<=0)throw std::invalid_argument("station scale must be positive");
        directions.assign(dirs,dirs+ndirs);if(nw)weights.assign(w,w+nw);if(nlocal)scales.assign(scale,scale+nlocal);
        row_local.assign(ids,ids+nids);members.assign(rows,rows+nrows);
        std::vector<int> parent(nlocal),claimed(v.n,0),local_seen(nlocal,-1);std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
        for(size_t row=0;row<v.n;++row){int local=ids[row];
            if(v.profile_enabled&&v.profile_ids&&v.profile_ids[row]>=0&&local>=0)throw std::invalid_argument("query station is an implicit intersection, not a free local unknown");
            if(local< -1||local>=int(nlocal))throw std::invalid_argument("station identity out of range");
            if(local>=0){int old=local_seen[local];if(old>=0&&(v.codes[old]!=v.codes[row]||v.stations[old]!=v.stations[row]))throw std::invalid_argument("shared material station has conflicting entity/value");local_seen[local]=int(row);}}
        for(int seen:local_seen)if(seen<0)throw std::invalid_argument("unreferenced local variable");
        groups.reserve(ng);
        for(int gi=0;gi<ng;++gi){int begin=offsets[gi],end=offsets[gi+1],rank=ranks[gi];
            if(begin<0||end<=begin||end>int(nrows)||rank<0||woffsets[gi]<0||woffsets[gi+1]<woffsets[gi]||size_t(woffsets[gi+1])>nw||
               (pixels?(end!=begin+1||rank!=2||woffsets[gi]!=0):size_t(woffsets[gi+1]-woffsets[gi])!=size_t(rank)*2*(end-begin)))throw std::invalid_argument("malformed observation residual group");
            int first=-1;for(int j=begin;j<end;++j){int row=rows[j];if(row<0||row>=int(v.n)||claimed[row]++)throw std::invalid_argument("correlation rows overlap/outside problem");
                if(ids[row]>=0){int r=root(ids[row]);if(first<0)first=r;else parent[r]=root(first);}}
            groups.push_back({begin,end-begin,rank,woffsets[gi],-1,0});}
        for(int count:claimed)if(count!=1)throw std::invalid_argument("correlation partition drops source rows");
        // Reuse identical support patterns; million-row input never duplicates
        // the camera column map or CSR scatter table for every observation.
        std::map<std::vector<int>,int> pattern_ids;std::vector<ColumnPattern> patterns;
        auto intern=[&](std::vector<int>& cols){std::sort(cols.begin(),cols.end());cols.erase(std::unique(cols.begin(),cols.end()),cols.end());
            auto it=pattern_ids.find(cols);if(it!=pattern_ids.end())return it->second;
            int id=bounded(patterns.size());patterns.push_back({bounded(columns.size()),bounded(cols.size()),0});columns.insert(columns.end(),cols.begin(),cols.end());pattern_ids.emplace(cols,id);return id;};
        std::vector<std::vector<int>> rig_columns(v.nrig);
        if(rig_dirs){
            if(rig_q!=q)throw std::invalid_argument("joint camera/trajectory column count mismatch");
            rig_directions.assign(rig_dirs,rig_dirs+v.nrig*size_t(v.nr==43?31:27)*q);rig_tangents.resize(v.nrig*size_t(38)*q);
            rig_lower.assign(rig_lo,rig_lo+v.nr*v.nrig);rig_upper.assign(rig_hi,rig_hi+v.nr*v.nrig);
            for(size_t camera=0;camera<v.nrig;++camera)for(int j=0;j<q;++j){bool active=false;for(int k=0;k<(v.nr==43?31:27);++k)active|=rig_dirs[(camera*(v.nr==43?31:27)+size_t(k))*q+j]!=0.;if(active)rig_columns[camera].push_back(j);}
        }
        std::vector<std::vector<int>> scene_columns(v.npaths);
        if(scene_dirs){
            if(scene_q!=q||!v.scene_paths)throw std::invalid_argument("joint scene/trajectory column count mismatch");
            scene_directions.assign(scene_dirs,scene_dirs+v.nscene*size_t(q));scene_lower.assign(scene_lo,scene_lo+v.nscene);scene_upper.assign(scene_hi,scene_hi+v.nscene);
            scene_chart.assign(v.nscene,0);if(charts)std::copy_n(charts,v.nscene,scene_chart.data());scene_effective.resize(v.nscene*size_t(q));
            for(int j=0;j<q;++j)scene_chart_direction(v.scene_values,v.nscene,scene_chart.data(),scene_dirs,q,j,scene_effective.data());
            std::vector<std::vector<int>> parameter_columns(v.nscene);scene_active.assign(q,0);
            for(size_t k=0;k<v.nscene;++k)for(int j=0;j<q;++j){
                const int type=scene_chart[k];bool active=false;
                if(type>=3&&type<=5){size_t start=k-size_t(type-3);for(int a=0;a<3;++a)active|=scene_dirs[(start+a)*q+j]!=0.;}
                else active=scene_dirs[k*q+j]!=0.;
                if(active){parameter_columns[k].push_back(j);scene_active[j]=1;}
            }
            for(size_t k=0;k<v.npaths;++k){auto path=v.scene_paths[k];auto& cols=scene_columns[k];
                auto include=[&](int begin,int count){for(int a=begin;a<begin+count;++a)cols.insert(cols.end(),parameter_columns[a].begin(),parameter_columns[a].end());};
                int kind=v.paths[7*k];if(path.type||kind==3||kind==7)include(0,6);
                if(path.type==1){
                    if(v.nl!=ice_tolerance_layout_size||v.options!=127u||
                       !ice_tolerance_layout_support(path.code,[&](int index){include(6+index,1);}))include(6,int(v.nl));
                }
                if(path.type==2){include(path.size,2);include(path.placement,3);if(path.family)include(path.shape,path.shape_count);}
                std::sort(cols.begin(),cols.end());cols.erase(std::unique(cols.begin(),cols.end()),cols.end());
            }
            for(int k=0;k<6;++k){LocalJet b[6];for(int j=0;j<6;++j)b[j]={v.board[j],j==k?1.:0.};scene_courses.emplace_back(BoardShapeValue<LocalJet>{b[0],b[1],b[2],b[3],b[4],b[5]});}
        }
        std::vector<std::vector<int>> control_columns(v.nc);
        for(size_t c=0;c<v.nc;++c)for(int j=0;j<q;++j)if(dirs[(3*c)*q+j]!=0||dirs[(3*c+1)*q+j]!=0||dirs[(3*c+2)*q+j]!=0)control_columns[c].push_back(j);
        std::map<std::tuple<int,int,int,int>,int> support_patterns;row_plan.reserve(v.n);
        MotionSpline spline{v.controls,v.nc,v.first,v.dt};
        for(size_t row=0;row<v.n;++row){int lo=0,hi=0;double basis[4];
            // Exactly the projection's declared time domain. In particular an
            // off-image root must not silently lose a crossed knot's columns.
            double begin=v.first+v.dt,end=v.first+(v.nc-2)*v.dt;
            if(v.projection_time_half>=0.){begin=std::max(begin,v.times[row]-v.projection_time_half);end=std::min(end,v.times[row]+v.projection_time_half);}
            const double actual_half=std::abs(row_camera_view(v,row).rig[37])/2;
            bool valid=finite_scalar(v.times[row])&&v.times[row]-actual_half>=begin-1e-12&&v.times[row]+actual_half<=end+1e-12&&
                spline.basis(begin,lo,basis)==Status::ok&&spline.basis(end,hi,basis)==Status::ok;
            // Invalid source times stay in the problem and fail linearization.
            auto key=std::make_tuple(valid?lo-1:-1,valid?hi+2:-1,scene_dirs?v.codes[row]:-1,int(row_camera(v,row)));auto found=support_patterns.find(key);int id;
            if(found==support_patterns.end()){std::vector<int> cols=rig_columns[row_camera(v,row)];if(scene_dirs)cols.insert(cols.end(),scene_columns[v.codes[row]].begin(),scene_columns[v.codes[row]].end());if(valid)for(int c=std::get<0>(key);c<=std::get<1>(key);++c)cols.insert(cols.end(),control_columns[c].begin(),control_columns[c].end());id=intern(cols);support_patterns.emplace(key,id);}else id=found->second;
            auto pat=patterns[id];row_plan.push_back({pat.begin,pat.count,jac_count});jac_count+=2*size_t(pat.count+1);max_row_q=std::max(max_row_q,pat.count);}
        std::vector<int> by_root(nlocal,-1);std::vector<std::vector<int>> locals,group_ids;
        for(size_t local=0;local<nlocal;++local){int r=root(int(local));if(by_root[r]<0){by_root[r]=bounded(locals.size());locals.emplace_back();group_ids.emplace_back();}locals[by_root[r]].push_back(int(local));}
        row_local_column.assign(v.n,-1);
        for(int gi=0;gi<ng;++gi){auto& g=groups[gi];int ci=-1;for(int j=0;j<g.count;++j){int local=ids[members[g.begin+j]];if(local>=0){ci=by_root[root(local)];break;}}
            // Unrelated fixed groups do not form one artificial global component.
            if(ci<0){ci=bounded(locals.size());locals.emplace_back();group_ids.emplace_back();}g.component=ci;group_ids[ci].push_back(gi);}
        std::vector<int> local_column(nlocal,-1),component_patterns;components.reserve(locals.size());
        for(size_t ci=0;ci<locals.size();++ci){const auto& ll=locals[ci];const auto& gg=group_ids[ci];std::vector<int> cols;
            for(int gi:gg){auto g=groups[gi];for(int j=0;j<g.count;++j){auto rp=row_plan[members[g.begin+j]];cols.insert(cols.end(),columns.begin()+rp.begin,columns.begin()+rp.begin+rp.count);}}
            int id=intern(cols);auto pat=patterns[id];int dim=bounded(size_t(pat.count)+ll.size());component_patterns.push_back(id);
            components.push_back({bounded(component_locals.size()),bounded(ll.size()),bounded(component_groups.size()),bounded(gg.size()),dim,h_count,g_count,pat.begin,pat.count,0,packed_h_count});
            auto& component=components.back();
            if(pixels&&gg.size()==1&&groups[gg[0]].count==1&&groups[gg[0]].rank==2)component.normal_rank=2;
            h_count+=checked_product(size_t(dim),size_t(dim));
            packed_h_count+=component.normal_rank==2?2*size_t(dim):checked_product(size_t(dim),size_t(dim)+1)/2;
            g_count+=dim;max_component_q=std::max(max_component_q,pat.count);
            for(size_t j=0;j<ll.size();++j)local_column[ll[j]]=int(j);
            component_locals.insert(component_locals.end(),ll.begin(),ll.end());component_groups.insert(component_groups.end(),gg.begin(),gg.end());}
        for(size_t row=0;row<v.n;++row)if(ids[row]>=0)row_local_column[row]=local_column[ids[row]];
        for(auto& g:groups){g.white=white_count;white_count+=checked_product(size_t(g.rank),size_t(components[g.component].dim+1));}
        // Independent raw-pixel components already own both exact Jacobian
        // rows in white. Read that immutable linearization directly instead
        // of keeping a second multi-GB rank-two copy. Mixed/shared/correlated
        // components retain their existing packed normal storage.
        rank2_white_storage=!components.empty()&&std::all_of(components.begin(),components.end(),[](const auto& c){return c.normal_rank==2;});
        if(rank2_white_storage)for(auto& c:components){
            const auto& g=groups[component_groups[c.group_begin]];c.packed_h=g.white+1;c.normal_stride=c.dim+1;
        }
        std::vector<std::vector<int>> adjacency(q);for(int i=0;i<q;++i)adjacency[i].push_back(i);
        for(const auto& pat:patterns)for(int i=0;i<pat.count;++i){auto& a=adjacency[columns[pat.begin+i]];a.insert(a.end(),columns.begin()+pat.begin+i,columns.begin()+pat.begin+pat.count);}
        csr_offsets.push_back(0);for(auto& a:adjacency){std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());csr_columns.insert(csr_columns.end(),a.begin(),a.end());csr_offsets.push_back(bounded(csr_columns.size()));}
        for(auto& pat:patterns){pat.scatter=scatter.size();for(int i=0;i<pat.count;++i){int global=columns[pat.begin+i];auto first=csr_columns.begin()+csr_offsets[global],last=csr_columns.begin()+csr_offsets[global+1];
            for(int j=i;j<pat.count;++j)scatter.push_back(int(std::lower_bound(first,last,columns[pat.begin+j])-csr_columns.begin()));}}
        for(size_t ci=0;ci<components.size();++ci)components[ci].scatter=patterns[component_patterns[ci]].scatter;
        motion_rows.prepare(directions,q);rig_rows.prepare(rig_directions,q);scene_rows.prepare(scene_directions,q);
        size_t planned=motion_rows.bytes()+rig_rows.bytes()+scene_rows.bytes()+8*(directions.size()+weights.size()+scales.size()+jac_count+white_count+3*size_t(ng)+(rank2_white_storage?0:packed_h_count)+g_count+csr_columns.size()+q+4)+
            4*(nlocal+members.size()+row_local.size()+row_local_column.size()+component_groups.size()+columns.size()+scatter.size()+csr_offsets.size()+csr_columns.size())+
            sizeof(GroupPlan)*groups.size()+sizeof(ComponentPlan)*components.size()+sizeof(RowPlan)*row_plan.size();
        planned+=8*(2*scene_directions.size()+scene_lower.size()+scene_upper.size()+rig_directions.size()+rig_tangents.size()+rig_lower.size()+rig_upper.size())+4*(scene_chart.size()+scene_active.size())+sizeof(BoardCourseValue<LocalJet>)*scene_courses.size();
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        if(planned>free)throw std::runtime_error("sparse linearization needs "+std::to_string(planned)+" device bytes; available "+std::to_string(free)+"; original correlation components retained");
        host={groups.data(),components.data(),row_plan.data(),members.data(),component_groups.data(),row_local.data(),row_local_column.data(),columns.data(),scatter.data(),csr_offsets.data(),csr_columns.data(),directions.data(),scales.data(),weights.data(),q,ng,bounded(components.size()),csr_columns.size(),delta};
        d_directions.upload(directions,stream);d_weights.upload(weights,stream);d_scales.upload(scales,stream);
        d_members.upload(members,stream);d_row_local.upload(row_local,stream);d_row_local_column.upload(row_local_column,stream);d_component_groups.upload(component_groups,stream);
        d_columns.upload(columns,stream);d_scatter.upload(scatter,stream);d_csr_offsets.upload(csr_offsets,stream);d_csr_columns.upload(csr_columns,stream);
        d_groups.upload(groups,stream);d_components.upload(components,stream);d_row_plan.upload(row_plan,stream);
        d_jac.allocate(jac_count);d_white.allocate(white_count);d_stats.allocate(3*size_t(ng));d_h.allocate(rank2_white_storage?0:packed_h_count);d_g.allocate(g_count);d_shared.allocate(csr_columns.size()+q);d_summary.allocate(4);
        device={d_groups.data,d_components.data,d_row_plan.data,d_members.data,d_component_groups.data,d_row_local.data,d_row_local_column.data,d_columns.data,d_scatter.data,d_csr_offsets.data,d_csr_columns.data,d_directions.data,d_scales.data,d_weights.data,q,ng,bounded(components.size()),csr_columns.size(),delta};
        host.pixel_residuals=device.pixel_residuals=pixels;
        motion_rows.upload(stream);rig_rows.upload(stream);scene_rows.upload(stream);
        host.motion_rows=motion_rows.host();host.rig_rows=rig_rows.host();host.scene_rows=scene_rows.host();
        device.motion_rows=motion_rows.device();device.rig_rows=rig_rows.device();device.scene_rows=scene_rows.device();
        if(rig_dirs){
            d_rig_directions.upload(rig_directions,stream);d_rig_tangents.allocate(rig_tangents.size());d_rig_lower.upload(rig_lower,stream);d_rig_upper.upload(rig_upper,stream);
            host.rig_directions=rig_directions.data();host.rig_tangents=rig_tangents.data();host.rig_lower=rig_lower.data();host.rig_upper=rig_upper.data();host.max_readout=readout_support;
            device.rig_directions=d_rig_directions.data;device.rig_tangents=d_rig_tangents.data;device.rig_lower=d_rig_lower.data;device.rig_upper=d_rig_upper.data;device.max_readout=readout_support;
        }
        if(scene_dirs){
            d_scene_chart.upload(scene_chart,stream);d_scene_effective.upload(scene_effective,stream);
            d_scene_active.upload(scene_active,stream);host.scene_active=scene_active.data();device.scene_active=d_scene_active.data;
            d_scene_directions.upload(scene_directions,stream);d_scene_lower.upload(scene_lower,stream);d_scene_upper.upload(scene_upper,stream);d_scene_courses.upload(scene_courses,stream);
            host.scene_steps=scene_directions.data();host.scene_chart=scene_chart.data();host.scene_directions=scene_effective.data();host.scene_lower=scene_lower.data();host.scene_upper=scene_upper.data();host.scene_courses=scene_courses.data();
            device.scene_steps=d_scene_directions.data;device.scene_chart=d_scene_chart.data;device.scene_directions=d_scene_effective.data;device.scene_lower=d_scene_lower.data;device.scene_upper=d_scene_upper.data;device.scene_courses=d_scene_courses.data;
        }
        local_sides.assign(nlocal,0);d_local_sides.upload(local_sides,stream);host.local_sides=local_sides.data();device.local_sides=d_local_sides.data;
        host.mount_side=device.mount_side=mount_side;
        for(auto& e:event)cuda_check(cudaEventCreate(&e));
    }
    double* h_values()const{return rank2_white_storage?d_white.data:d_h.data;}
    size_t normal_storage_count()const{return rank2_white_storage?white_count:packed_h_count;}
    size_t device_bytes()const{return motion_rows.bytes()+rig_rows.bytes()+scene_rows.bytes()+d_local_sides.bytes()+d_scene_chart.bytes()+d_scene_effective.bytes()+d_scene_active.bytes()+d_scene_directions.bytes()+d_scene_lower.bytes()+d_scene_upper.bytes()+d_scene_courses.bytes()+d_rig_directions.bytes()+d_rig_tangents.bytes()+d_rig_lower.bytes()+d_rig_upper.bytes()+d_directions.bytes()+d_weights.bytes()+d_scales.bytes()+d_members.bytes()+d_row_local.bytes()+d_row_local_column.bytes()+d_component_groups.bytes()+d_columns.bytes()+d_scatter.bytes()+d_csr_offsets.bytes()+d_csr_columns.bytes()+d_groups.bytes()+d_components.bytes()+d_row_plan.bytes()+d_jac.bytes()+d_white.bytes()+d_stats.bytes()+d_h.bytes()+d_g.bytes()+d_shared.bytes()+d_summary.bytes();}
    void cpu(const View& v,double* summary){
        const auto& l=host;cpu_output.resize(v.n*11);cpu_status.resize(v.n);cpu_jac.resize(jac_count);cpu_white.resize(white_count);cpu_stats.resize(3*size_t(l.ng));cpu_h.resize(h_count);cpu_g.resize(g_count);cpu_shared.assign(l.nnz+l.q,0.);
        if(host.scene_directions)for(int q=0;q<host.q;++q)scene_chart_direction(v.scene_values,v.nscene,host.scene_chart,host.scene_steps,host.q,q,scene_effective.data());
        if(host.scene_directions)for(int k=0;k<6;++k){LocalJet b[6];for(int j=0;j<6;++j)b[j]={v.board[j],j==k?1.:0.};scene_courses[k]=BoardCourseValue<LocalJet>({b[0],b[1],b[2],b[3],b[4],b[5]});}
        if(host.rig_directions)for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(host,v.nr,camera);for(int q=0;q<host.q;++q)motion_rig_tangent(cv.rig,cl.rig_directions,host.q,q,cl.rig_tangents);}
        for(size_t row=0;row<v.n;++row)linearize_row(v,l,row,cpu_output.data()+11*row,cpu_status.data()+row,cpu_jac.data()+row_plan[row].jac);
        for(int gi=0;gi<l.ng;++gi){auto g=groups[gi];int d=components[g.component].dim;for(int k=0;k<g.rank;++k)for(int col=0;col<d+1;++col)cpu_white[g.white+size_t(k)*(d+1)+col]=whiten_element(l,gi,k,col,cpu_output.data(),cpu_jac.data());group_loss(l,gi,cpu_white.data(),cpu_stats.data()+3*size_t(gi));}
        for(int ci=0;ci<l.ncomponents;++ci)for(int a=0;a<components[ci].local_count;++a)choose_corner_station(v,l,ci,a,cpu_output.data(),cpu_jac.data(),cpu_white.data(),cpu_stats.data());
        for(int gi=0;gi<l.ng;++gi){auto g=groups[gi];int d=components[g.component].dim;for(int k=0;k<g.rank;++k)for(int col=1;col<d+1;++col)cpu_white[g.white+size_t(k)*(d+1)+col]=whiten_element(l,gi,k,col,cpu_output.data(),cpu_jac.data());}
        for(int ci=0;ci<l.ncomponents;++ci){auto c=components[ci];for(int i=0;i<c.dim;++i){for(int j=0;j<c.dim;++j)cpu_h[c.h+size_t(i)*c.dim+j]=block_element(l,ci,i,j,false,cpu_white.data(),cpu_stats.data());cpu_g[c.g+i]=block_element(l,ci,i,0,true,cpu_white.data(),cpu_stats.data());}
            for(int i=0;i<c.q_count;++i){cpu_shared[l.nnz+columns[c.q_begin+i]]+=cpu_g[c.g+i];for(int j=i;j<c.q_count;++j)cpu_shared[scatter[c.scatter+upper_index(c.q_count,i,j)]]+=cpu_h[c.h+size_t(i)*c.dim+j];}}
        std::fill(summary,summary+4,0.);for(auto st:cpu_status)summary[0]+=st!=0;
        for(int gi=0;gi<l.ng;++gi){summary[1]+=.5*cpu_stats[3*size_t(gi)];summary[2]+=cpu_stats[3*size_t(gi)+2];summary[3]+=cpu_stats[3*size_t(gi)+1]<1.;}cpu_ready=summary[0]==0&&finite_array(summary,4);
    }
};
