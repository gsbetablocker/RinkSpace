// Warm-start the existing joint Schur QP with original accepted-state faces.
// The auxiliary cone NNLS selects faces only; it never supplies multipliers
// to a different metric and never decides convergence or nonlinear feasibility.
struct JointFaceSeed {
    std::vector<StationarityFace> faces;
    std::vector<int> offsets,columns,chosen;
    std::vector<double> values,gradient;
    DeviceBuffer<int> d_offsets,d_columns;
    DeviceBuffer<double> d_values,d_margins,d_equalities;
    bool captured=false,attempted=false,solved=false,uploaded=false;double kkt=0.,projection_norm=0.;
    size_t device_bytes()const{return d_offsets.bytes()+d_columns.bytes()+d_values.bytes()+d_margins.bytes()+d_equalities.bytes();}
    static double margin(const StationarityFace& f){return f.margin+(f.family==6?1e-14:(f.family==3?camera_envelope_tolerance():0.));}
    void capture(int q,const std::vector<StationarityFace>& input,const std::vector<double>& a,const std::vector<double>& g){
        captured=true;faces.clear();offsets.assign(1,0);columns.clear();values.clear();chosen.clear();gradient=g;attempted=solved=uploaded=false;
        std::map<std::vector<uint64_t>,int> exact;
        for(size_t i=0;i<input.size();++i){std::vector<uint64_t> key{uint64_t(input[i].equality)};std::vector<int> cols;std::vector<double> row;
            if(input[i].equality){uint64_t bits;double h=margin(input[i]);std::memcpy(&bits,&h,sizeof(bits));key.push_back(bits);}
            for(int j=0;j<q;++j){double x=a[i*size_t(q)+j];if(x==0.)continue;uint64_t bits;std::memcpy(&bits,&x,sizeof(bits));key.push_back(uint64_t(j));key.push_back(bits);cols.push_back(j);row.push_back(x);}
            if(row.empty())continue;
            auto found=exact.find(key);if(found!=exact.end()){
                if(margin(input[i])<margin(faces[found->second]))faces[found->second]=input[i];continue;}
            exact.emplace(std::move(key),int(faces.size()));faces.push_back(input[i]);
            columns.insert(columns.end(),cols.begin(),cols.end());values.insert(values.end(),row.begin(),row.end());offsets.push_back(int(columns.size()));
        }
    }
    template<class Stop> void select(double budget,Stop stop){
        if(attempted)return;attempted=true;const int n=int(faces.size());if(!n)return;
        const size_t bytes=(checked_product(size_t(n),size_t(n))+6*size_t(n)+gradient.size())*sizeof(double);
        if(double(bytes)>budget)throw std::runtime_error("joint face selection exceeds host budget");
        std::vector<double> norm(n),gram(size_t(n)*n),target(n),dual(n,0.),equality(n);
        for(int i=0;i<n;++i){if(stop())throw SolverStopped{};double nn=0.,rhs=0.;
            for(int k=offsets[i];k<offsets[i+1];++k){nn+=values[k]*values[k];rhs+=values[k]*gradient[columns[k]];}
            norm[i]=sqrt(nn);if(!(norm[i]>0.)||!finite_scalar(norm[i]))return;target[i]=rhs/norm[i];equality[i]=faces[i].equality;
        }
        for(int i=0;i<n;++i){if(stop())throw SolverStopped{};for(int j=0;j<=i;++j){int x=offsets[i],y=offsets[j];double dot=0.;
            while(x<offsets[i+1]&&y<offsets[j+1]){if(columns[x]<columns[y])++x;else if(columns[y]<columns[x])++y;else{dot+=values[x]*values[y];++x;++y;}}
            gram[size_t(i)*n+j]=gram[size_t(j)*n+i]=dot/(norm[i]*norm[j]);}}
        DualWorkingStats stats;auto result=coupled_dual_working_set(n,n,gram.data(),target.data(),dual.data(),equality.data(),kkt,stats,stop);
        if(result==DualWorkingResult::stopped)throw SolverStopped{};
        solved=result==DualWorkingResult::solved;if(!solved)return;
        auto direction=gradient;for(double& v:direction)v=-v;
        for(int i=0;i<n;++i){if(dual[i]!=0.)for(int k=offsets[i];k<offsets[i+1];++k)direction[columns[k]]+=dual[i]*values[k]/norm[i];
            // Rig boxes keep their established exact retained-coordinate map.
            // The ordinary separator still handles every unseeded rig face.
            if(faces[i].family!=0&&(dual[i]>0.||faces[i].equality))chosen.push_back(i);}
        double nn=0.;for(double v:direction)nn+=v*v;projection_norm=sqrt(nn);
    }
    template<class T> static void replace(DeviceBuffer<T>& destination,const std::vector<T>& host,cudaStream_t stream){
        DeviceBuffer<T> temporary;temporary.upload(host,stream);std::swap(destination.data,temporary.data);std::swap(destination.count,temporary.count);
    }
    size_t upload_bytes()const{size_t count=0;for(int i:chosen)count+=size_t(offsets[i+1]-offsets[i]);return (chosen.size()+1+count)*sizeof(int)+(count+2*chosen.size())*sizeof(double);}
    void upload(cudaStream_t stream){
        std::vector<int> off{0},cols;std::vector<double> val,h,eq;
        for(int i:chosen){cols.insert(cols.end(),columns.begin()+offsets[i],columns.begin()+offsets[i+1]);val.insert(val.end(),values.begin()+offsets[i],values.begin()+offsets[i+1]);off.push_back(int(cols.size()));h.push_back(margin(faces[i]));eq.push_back(faces[i].equality);}
        replace(d_offsets,off,stream);replace(d_columns,cols,stream);replace(d_values,val,stream);replace(d_margins,h,stream);replace(d_equalities,eq,stream);
        // The small host upload vectors must remain alive until H2D finishes.
        cuda_check(cudaStreamSynchronize(stream));
        uploaded=true;
    }
};
__global__ void lm_joint_seed_rhs(int q,int first,int count,DirectionRowMap map,double* rhs){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k<size_t(q)*count)rhs[k]=direction_value(nullptr,q,size_t(first)+k/q,int(k%q),map);
}
__global__ void lm_joint_seed_cut(LinearView l,SolverView s,CoupledProjectionView c,int first,int count,const double* margins,const double* equalities,const double* rhs,const double* response){
    const int row=blockIdx.x,index=first+row;if(row>=count)return;__shared__ double scale;
    const double* a=rhs+size_t(row)*l.q;const double* w=response+size_t(row)*l.q;
    if(!threadIdx.x){double norm=0.,offset=0.;for(int j=0;j<l.q;++j){norm+=a[j]*w[j];offset-=a[j]*c.base[j];}
        scale=norm>0.&&finite_scalar(norm)?1./sqrt(norm):0.;c.target[index]=scale?(offset-margins[index])*scale:0.;c.dual[index]=0.;c.equality[index]=equalities[index];}
    __syncthreads();for(int j=threadIdx.x;j<l.q;j+=blockDim.x){c.gradients[size_t(index)*l.q+j]=scale?a[j]*scale:0.;c.responses[size_t(index)*l.q+j]=scale?w[j]*scale:0.;}
}
__global__ void lm_joint_seed_gram(LinearView l,CoupledProjectionView c,int count,const int* offsets,const int* columns){
    const int row=blockIdx.x;if(row>=count)return;
    for(int i=threadIdx.x;i<=row;i+=blockDim.x){double cross=1.;if(i!=row){cross=0.;for(int k=offsets[i];k<offsets[i+1];++k){int j=columns[k];cross+=c.gradients[size_t(i)*l.q+j]*c.responses[size_t(row)*l.q+j];}}
        c.gram[size_t(row)*c.capacity+i]=c.gram[size_t(i)*c.capacity+row]=cross;}
}
