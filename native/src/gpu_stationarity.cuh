/* Original-objective stationarity in the current feasible tangent cone.
 * The cone uses every active physical face, including coupled parameter maps.
 * The projection has explicit device/host execution; LM damping is not part of this test.
 */
struct StationarityFace {
    int family=0,index=0,other=-1,side=0;bool equality=false;
    double sign=1.,station=0.,margin=0.,scale=1.;CameraConstraint camera{};
};
// Restoration deliberately puts a boundary four envelope epsilons inside.
// The active-face test must include that numerical inset; otherwise the same
// face is enforced for trial steps but disappears from the stopping cone.
RS_HD inline double stationarity_active_tolerance(double scale=1.){
    return 8*camera_envelope_tolerance()*(1.+fabs(scale));
}
__device__ inline void stationarity_faces_range(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* count,int only_camera=-1,int only_cell=0,int only_print=-1){
    count[0]=count[1]=0;
    size_t camera_index=0;
    auto offer=[&](StationarityFace a,double margin,double scale=1.){
        const double eps=stationarity_active_tolerance(scale);
        if(!finite_scalar(margin)){count[1]=1;return;}
        if(margin < -eps)count[1]=1;
        if(!a.equality&&margin>eps)return;
        if(size_t(count[0])>=capacity){count[1]=1;return;}
        if(a.family==0)a.index+=int(camera_index*v.nr);else if(a.family==3)a.index=int(camera_index);else if(a.family==5)a.index+=int(4*camera_index);
        a.margin=margin;a.scale=scale;if(faces)faces[count[0]]=a;++count[0];
    };
    auto bounds=[&](int family,int index,double value,double lo,double hi){
        StationarityFace a{};a.family=family;a.index=index;
        if(lo==hi&&finite_scalar(lo)){a.equality=true;offer(a,value-lo,fabs(value)+fabs(lo));return;}
        if(finite_scalar(lo))offer(a,value-lo,fabs(value)+fabs(lo));
        a.sign=-1.;if(finite_scalar(hi))offer(a,hi-value,fabs(value)+fabs(hi));
    };
    if(only_camera<0){
    if(only_print<0){
    for(camera_index=0;camera_index<v.nrig;++camera_index){auto cv=camera_view(v,camera_index);auto cl=camera_linear_view(l,v.nr,camera_index);
    if(cl.rig_directions)for(int k=0;k<int(v.nr);++k){
        double lo=cl.rig_lower[k],hi=cl.rig_upper[k];
        if(k==37){lo=fmax(lo,-cl.max_readout);hi=fmin(hi,cl.max_readout);}
        bounds(0,k,cv.rig[k],lo,hi);
    }
    if(cl.envelope.enabled&&cl.envelope.windowed){
        const double halo=2*v.dt*fmin(cl.envelope.frame_lo-1.,double(v.nc)-2.-cl.envelope.frame_hi);
        bounds(0,37,cv.rig[37],-halo,halo);
    }
    }
    }
    camera_index=0;
    if(l.scene_directions){
        if(only_print<0){
        for(size_t k=0;k<v.nscene;++k)bounds(1,int(k),v.scene_values[k],l.scene_lower[k],l.scene_upper[k]);
        if(v.options&64u){IceToleranceGeometry<double> g(v.board,v.layout);
            ice_tolerance_margins(g,[&](int id,double margin){StationarityFace a{};a.family=6;a.index=id;offer(a,margin);});}
        if((v.options&4u)&&!(v.options&64u)){
            auto relation=[&](int high,int low,double minimum=0.){
                StationarityFace a{};a.family=2;a.index=6+high;a.other=6+low;
                if((v.options&16u)&&low==23){low=24;a.other=6+low;a.side=1;}
                offer(a,v.layout[high]+(a.side?1.:-1.)*v.layout[low]-minimum,fabs(v.layout[high])+fabs(v.layout[low]));
            };
            relation(0,23);
            for(int row=0;row<4;++row){relation(10+row,23);relation(8,10+row);
                const double interior=64*std::numeric_limits<double>::epsilon();
                relation(14+row,1,interior);relation(8,14+row,interior);}
        }
        }
        if(only_camera!=-3)for(size_t i=only_print<0?0:size_t(only_print);i<(only_print<0?s.print_count:size_t(only_print+1));++i){
            const auto probe=s.print_probes[i];const auto path=v.scene_paths[probe.path];
            Vec2Value<LocalJet> chart{},material{};
            if(scene_print_chart(path,v.scene_data,v.scene_values,nullptr,0,0,LocalJet(probe.station),chart,&material)!=Status::ok){count[1]=1;continue;}
            StationarityFace a{};a.family=4;a.index=probe.path;a.station=probe.station;
            offer(a,chart.y.a,v.board[5]);a.side=1;offer(a,v.board[5]-chart.y.a,v.board[5]);
            if(path.code==1){double width=exp(v.scene_values[path.size]),height=exp(v.scene_values[path.size+1]);
                double margins[4]={width*material.x.a,width*(1-material.x.a),height*material.y.a,height*(1-material.y.a)};
                for(int side=0;side<4;++side){a.side=side+2;offer(a,margins[side],side<2?width:height);}}
        }
    }
    }
    if(only_camera==-2||only_camera==-3||only_print>=0)return; // Fixed/print faces precede camera cells.
    for(camera_index=only_camera<0?0:size_t(only_camera);camera_index<(only_camera<0?v.nrig:size_t(only_camera+1));++camera_index){auto cv=camera_view(v,camera_index);auto cl=camera_linear_view(l,v.nr,camera_index);
    if(only_cell<=1&&cl.mount_side)for(int kind=0;kind<4;++kind){
        if(kind==2&&(cl.mount_side==2||cl.mount_side==3))continue;if(kind==3&&cl.mount_side==1)continue;
        StationarityFace a{};a.family=5;a.index=kind;offer(a,mount_constraint_value(cv,cl,s,kind,-1).a);
    }
    if(cl.envelope.enabled){
        auto rig=motion_rig(cv.rig);MotionSpline spline{v.controls,v.nc,v.first,v.dt};
        int first_cell=1;
        if(only_cell&&cl.envelope.windowed){
            double lo=cl.envelope.frame_windows[0];
            for(size_t i=1;i<cl.envelope.frame_window_count;++i)lo=fmin(lo,cl.envelope.frame_windows[2*i]);
            first_cell=int(fmax(1.,ceil(lo-fabs(rig.tau)/(2*v.dt)-1.)));
        }
        bool static_axis_seen[2]={only_cell>first_cell,only_cell>first_cell};
        for(int cell=only_cell?only_cell:1;cell<(only_cell?only_cell+1:int(v.nc)-2);++cell)camera_cell_constraints(rig,spline,cl.envelope,cell,[&](CameraConstraint camera){
            // Axis direction limits depend only on this physical rig. The
            // windowed envelope visits them in every selected spline cell;
            // one identical face defines exactly the same tangent cone.
            // Keep time-dependent attitude, pupil and motion faces separate.
            if(camera.kind==0||camera.kind==1){
                if(static_axis_seen[camera.kind])return;
                static_axis_seen[camera.kind]=true;
            }
            StationarityFace a{};a.family=3;a.camera=camera;
            if(camera.kind!=2||camera.margin>stationarity_active_tolerance()){offer(a,camera.margin);return;}
            // A maximum absolute rate has several active derivatives at a tie.
            // All tied faces are needed, including both signs at zero speed.
            const double* c=v.controls+3*size_t(cell-1);double times[2][3];int n[2];double maximum[2];
            for(int channel=0;channel<2;++channel){n[channel]=camera_velocity_extremum_times(c,channel,0,1,times[channel]);
                maximum[channel]=fabs(camera_spline_derivative(c,channel,camera.rate_u[channel],v.dt,1));}
            for(int i=0;i<n[0];++i)for(int sign0=-1;sign0<=1;sign0+=2){
                double r0=sign0*camera_spline_derivative(c,0,times[0][i],v.dt,1);
                if(maximum[0]-r0>stationarity_active_tolerance(maximum[0]))continue;
                for(int j=0;j<n[1];++j)for(int sign1=-1;sign1<=1;sign1+=2){
                    double r1=sign1*camera_spline_derivative(c,1,times[1][j],v.dt,1);
                    if(maximum[1]-r1>stationarity_active_tolerance(maximum[1]))continue;
                    a.camera.rate_u[0]=times[0][i];a.camera.rate_u[1]=times[1][j];
                    a.camera.rate_sign[0]=sign0;a.camera.rate_sign[1]=sign1;
                    offer(a,camera_constraint_value(rig,c,v.dt,a.camera));
                }
            }
        });
    }    }

}
// Deterministic cell order is retained when compacting parallel enumeration.
// The serial path remains an exact reference for diagnostics and regressions.
constexpr int stationarity_cell_capacity=224;
__global__ void lm_stationarity_faces(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* count){
    if(threadIdx.x||blockIdx.x)return;stationarity_faces_range(v,l,s,faces,capacity,count);
}
__global__ void lm_stationarity_fixed_faces(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* count){
    if(threadIdx.x||blockIdx.x)return;stationarity_faces_range(v,l,s,faces,capacity,count,-3);
}
// Count each part first, then write directly to its deterministic compact
// offset. Re-evaluating these small parallel checks avoids a second worst-case
// StationarityFace array for all allocated spline cells and print probes.
__device__ inline void stationarity_write_part(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* counts,int* total,int camera,int cell,int print){
    const int expected=counts[0];const size_t offset=size_t(counts[2]);int result[2];
    const size_t available=offset<capacity?capacity-offset:0;
    stationarity_faces_range(v,l,s,available?faces+offset:nullptr,available<size_t(expected)?available:size_t(expected),result,camera,cell,print);
    if(result[1]||result[0]!=expected)atomicExch(total+1,1);
}
__global__ void lm_stationarity_print_faces(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* counts,int* total){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=s.print_count)return;
    if(faces)stationarity_write_part(v,l,s,faces,capacity,counts+3*i,total,-2,0,int(i));
    else stationarity_faces_range(v,l,s,nullptr,6,counts+3*i,-2,0,int(i));
}
__global__ void lm_stationarity_cell_faces(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* counts,int* total){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,cells=v.nc-3;if(i>=cells*v.nrig)return;
    if(faces)stationarity_write_part(v,l,s,faces,capacity,counts+3*i,total,int(i/cells),int(i%cells)+1,-1);
    else stationarity_faces_range(v,l,s,nullptr,stationarity_cell_capacity,counts+3*i,int(i/cells),int(i%cells)+1);
}
__global__ void lm_stationarity_face_offsets(int cells,int* counts,size_t capacity,int* total){
    if(threadIdx.x||blockIdx.x)return;int next=total[0];
    for(int i=0;i<cells;++i){counts[3*i+2]=next;next+=counts[3*i];total[1]|=counts[3*i+1];}
    if(size_t(next)>capacity){total[1]=1;next=int(capacity);}total[0]=next;
}
__device__ inline double stationarity_direction(View v,LinearView l,SolverView s,StationarityFace a,int j){
    size_t camera=a.family==0?size_t(a.index)/v.nr:(a.family==3?size_t(a.index):(a.family==5?size_t(a.index)/4:0));
    if(a.family==0)a.index%=int(v.nr);else if(a.family==5)a.index%=4;
    l=camera_linear_view(l,v.nr,camera);v=camera_view(v,camera);
    if(a.family==6)return (!l.scene_active||l.scene_active[j])?ice_relation_value(v,l,a.index,j).v:0.;
    if(a.family==1)return a.sign*l.scene_directions[size_t(a.index)*l.q+j];
    if(a.family==2)return l.scene_directions[size_t(a.index)*l.q+j]+(a.side?1.:-1.)*l.scene_directions[size_t(a.other)*l.q+j];
    if(a.family==5)return mount_constraint_value(v,l,s,a.index,j).v;
    if(a.family==4){
        Vec2Value<LocalJet> chart{},material{};const auto path=v.scene_paths[a.index];
        if(scene_print_chart(path,v.scene_data,v.scene_values,s.print_directions,l.q,j,LocalJet(a.station),chart,&material)!=Status::ok)return NAN;
        if(a.side<2)return a.side?s.print_directions[5*size_t(l.q)+j]-chart.y.v:chart.y.v;
        int axis=(a.side-2)/2,index=path.size+axis;
        LocalJet size=exp(LocalJet(v.scene_values[index],s.print_directions[size_t(index)*l.q+j]));
        LocalJet coordinate=axis?material.y:material.x;
        return (size*((a.side%2)?1.-coordinate:coordinate)).v;
    }
    if(a.family==0&&a.index>=38)return a.index==38||!l.rig_directions?0.:a.sign*l.rig_directions[size_t(a.index-12)*l.q+j];
    LocalJet d[27],p[38];for(int k=0;k<27;++k)d[k]={0.,l.rig_directions?l.rig_directions[size_t(k)*l.q+j]:0.};
    if(retract_motion_rig(v.rig,d,p)!=Status::ok)return NAN;
    if(a.family==0)return a.sign*p[a.index].v;
    LocalJet controls[12];for(int k=0;k<12;++k){size_t i=3*size_t(a.camera.cell-1)+k;controls[k]={v.controls[i],l.directions[i*l.q+j]};}
    return camera_constraint_value(rig_from_values(p),controls,v.dt,a.camera).v;
}
__global__ void lm_stationarity_directions(View v,LinearView l,SolverView s,const StationarityFace* faces,int count,double* matrix){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(count)*l.q)return;
    matrix[i]=stationarity_direction(v,l,s,faces[i/l.q],int(i%l.q));
}
// Hildreth projection onto A*d >= 0 (equalities use unrestricted multipliers).
// Return a KKT residual as well as the projected descent norm. A failed inner
// solve is explicit and cannot certify convergence or alter accepted state.
// For C=R^free x C_restricted, the free projection is exactly -g_free.
// Since 0 belongs to C_restricted, its Euclidean projection norm is bounded
// by ||g_restricted||_2. A strictly dominating free coordinate therefore
// determines the exact infinity norm without constructing the other entries.
// The scratch output is NOT a full projected vector and must never be exported.
inline int stationarity_free_coordinate_certificate(int q,int count,const double* gradient,const double* matrix,double requested,double& exact,double* scratch){
    std::vector<unsigned char> touched(q,0);
    for(int i=0;i<count;++i)for(int j=0;j<q;++j){double value=matrix[size_t(i)*q+j];if(!finite_scalar(value))return false;if(value!=0.)touched[j]=1;}
    double free_max=0.,restricted_norm=0.;
    for(int j=0;j<q;++j){double value=gradient[j];if(!finite_scalar(value))return false;
        if(touched[j]){restricted_norm=std::hypot(restricted_norm,value);scratch[j]=0.;}
        else{free_max=fmax(free_max,fabs(value));scratch[j]=-value;}}
    // Conservative rounding slack; strict comparison also handles equality.
    double upper=std::nextafter(restricted_norm*(1.+16.*q*std::numeric_limits<double>::epsilon()),std::numeric_limits<double>::infinity());
    if(!(free_max>requested))return 0;exact=free_max;return free_max>upper?1:2;
}
// Kind 1 proves the exact norm; kind 2 proves a strict lower bound above
// the unchanged convergence tolerance. Neither supplies a full projection.
inline bool stationarity_free_coordinate_norm(int q,int count,const double* gradient,const double* matrix,double requested,double& exact,double* scratch){
    return stationarity_free_coordinate_certificate(q,count,gradient,matrix,requested,exact,scratch)==1;
}
template<class Equality,class Stop> RS_HD inline bool stationarity_cone_interruptible(int q,int count,const double* gradient,double* matrix,double* dual,double* descent,Equality equality,double tolerance,double& certificate,double* norms,Stop stopped,int* completed_sweeps=nullptr,const int* row_offsets=nullptr,const int* nonzero_columns=nullptr,size_t pair_budget=std::numeric_limits<size_t>::max()){
    for(int j=0;j<q;++j){if(!finite_scalar(gradient[j]))return false;descent[j]=-gradient[j];}
    for(int i=0;i<count;++i){double norm=0.;double* a=matrix+size_t(i)*q;
        for(int j=0;j<q;++j){if(!finite_scalar(a[j]))return false;norm+=a[j]*a[j];}
        if(!finite_scalar(norm))return false;norm=sqrt(norm);if(norms)norms[i]=norm;for(int j=0;j<q;++j)a[j]=norm?a[j]/norm:0.;dual[i]=0.;}
    // Exact zero omission: active physical normals touch only a small subset
    // of motion/rig/scene columns. Preserve ascending-column summation order.
    auto dot_descent=[&](int i){const double* a=matrix+size_t(i)*q;double value=0.;
        if(row_offsets){for(int k=row_offsets[i];k<row_offsets[i+1];++k){int j=nonzero_columns[k];value+=a[j]*descent[j];}}
        else for(int j=0;j<q;++j)value+=a[j]*descent[j];return value;};
    auto cross_rows=[&](int i,int k){const double* a=matrix+size_t(i)*q;const double* b=matrix+size_t(k)*q;double value=0.;
        if(row_offsets){int row=row_offsets[i+1]-row_offsets[i]<row_offsets[k+1]-row_offsets[k]?i:k;
            for(int t=row_offsets[row];t<row_offsets[row+1];++t){int j=nonzero_columns[t];value+=a[j]*b[j];}}
        else for(int j=0;j<q;++j)value+=a[j]*b[j];return value;};
    auto update_one=[&](int i,double change){const double* a=matrix+size_t(i)*q;
        if(row_offsets){for(int k=row_offsets[i];k<row_offsets[i+1];++k){int j=nonzero_columns[k];descent[j]+=change*a[j];if(!finite_scalar(descent[j]))return false;}}
        else for(int j=0;j<q;++j){descent[j]+=change*a[j];if(!finite_scalar(descent[j]))return false;}return true;};
    auto update_pair=[&](int i,int k,double d0,double d1){const double* a=matrix+size_t(i)*q;const double* b=matrix+size_t(k)*q;
        if(row_offsets){int x=row_offsets[i],y=row_offsets[k],endx=row_offsets[i+1],endy=row_offsets[k+1];
            while(x<endx||y<endy){int ix=x<endx?nonzero_columns[x]:q,iy=y<endy?nonzero_columns[y]:q,j=ix<iy?ix:iy;
                descent[j]+=d0*a[j]+d1*b[j];if(!finite_scalar(descent[j]))return false;if(ix==j)++x;if(iy==j)++y;}}
        else for(int j=0;j<q;++j){descent[j]+=d0*a[j]+d1*b[j];if(!finite_scalar(descent[j]))return false;}return true;};
    auto certified=[&](){
        double error=0.;
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;double value=dot_descent(i);
            if(!finite_scalar(value))return -1;
            error=fmax(error,(equality(i)||dual[i]>0.)?fabs(value):fmax(0.,-value));}
        if(error>tolerance)return 0;
        certificate=error;
        for(int j=0;j<q;++j){double residual=descent[j]+gradient[j];for(int i=0;i<count;++i)residual-=dual[i]*matrix[size_t(i)*q+j];
            if(!finite_scalar(residual))return -1;
            certificate=fmax(certificate,fabs(residual));certificate=fmax(certificate,fabs(descent[j]));}
        return finite_scalar(certificate)?1:-1;
    };
#if !defined(__CUDA_ARCH__)
    // The normalized face matrix is immutable throughout this projection.
    // Cache its nonzero cross products once, preserving the original ordered
    // pair traversal and exact floating-point dot-product order.
    std::vector<size_t> pair_offsets;std::vector<int> pair_columns;
    std::vector<double> pair_cross;
    const size_t maximum_pairs=size_t(count)*size_t(count?count-1:0)/2;
    const bool cache_fits=maximum_pairs<=pair_budget/(sizeof(int)+sizeof(double))&&size_t(count+1)*sizeof(size_t)<=pair_budget-maximum_pairs*(sizeof(int)+sizeof(double));
    bool pairs_ready=false;
#endif
    for(int sweep=0;sweep<4096;++sweep){
        if(stopped())return false;
        if(completed_sweeps)*completed_sweeps=sweep+1;
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;double value=dot_descent(i);
            if(!finite_scalar(value))return false;
            double next=equality(i)?dual[i]-value:fmax(0.,dual[i]-value),change=next-dual[i];dual[i]=next;
            if(!finite_scalar(change))return false;
            if(change!=0.&&!update_one(i,change))return false;}
        // Most feasible or sparsely active cones finish in one scalar sweep.
        // Check the same KKT certificate before paying for every face pair.
        if(stopped())return false;
        int ready=certified();if(ready)return ready>0;
        // Cyclic exact two-coordinate minimization of the SAME convex dual.
        // Closely aligned active faces otherwise exchange their multipliers
        // extremely slowly under scalar Hildreth sweeps. Bound candidates keep
        // inequality multipliers nonnegative; equality multipliers stay free.
#if !defined(__CUDA_ARCH__)
        if(row_offsets&&cache_fits&&!pairs_ready){
            pair_columns.reserve(maximum_pairs);pair_cross.reserve(maximum_pairs);
            pair_offsets.reserve(size_t(count)+1);pair_offsets.push_back(0);
            for(int i=0;i<count;++i){
                if(stopped())return false;
                if(!norms||norms[i]!=0.)for(int k=i+1;k<count;++k){
                    if(norms&&norms[k]==0.)continue;
                    double cross=cross_rows(i,k);
                    if(cross!=0.){pair_columns.push_back(k);pair_cross.push_back(fmax(-1.,fmin(1.,cross)));}
                }
                pair_offsets.push_back(pair_columns.size());
            }
            pairs_ready=true;
        }
#endif
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;
            if(stopped())return false;
            size_t begin=size_t(i)+1,end=size_t(count);
#if !defined(__CUDA_ARCH__)
            if(pairs_ready){begin=pair_offsets[i];end=pair_offsets[i+1];}
#endif
            for(size_t pair=begin;pair<end;++pair){int k=int(pair);double cross=0.;
#if !defined(__CUDA_ARCH__)
                if(pairs_ready){k=pair_columns[pair];cross=pair_cross[pair];}else
#endif
                {if(norms&&norms[k]==0.)continue;cross=cross_rows(i,k);if(cross==0.)continue;cross=fmax(-1.,fmin(1.,cross));}
                double r0=dot_descent(i),r1=dot_descent(k);
                double best=0.,d0=0.,d1=0.;
                auto offer=[&](double x,double y){
                    if(!finite_scalar(x)||!finite_scalar(y))return;
                    if((!equality(i)&&dual[i]+x<0.)||(!equality(k)&&dual[k]+y<0.))return;
                    double value=.5*(x*x+y*y+2*cross*x*y)+r0*x+r1*y;
                    if(finite_scalar(value)&&value<best){best=value;d0=x;d1=y;}
                };
                const double determinant=(1.-cross)*(1.+cross);
                if(determinant>64*std::numeric_limits<double>::epsilon())
                    offer((cross*r1-r0)/determinant,(cross*r0-r1)/determinant);
                if(!equality(i)){double x=-dual[i],y=-r1-cross*x;if(!equality(k))y=fmax(-dual[k],y);offer(x,y);}
                if(!equality(k)){double y=-dual[k],x=-r0-cross*y;if(!equality(i))x=fmax(-dual[i],x);offer(x,y);}
                if(d0!=0.||d1!=0.){dual[i]+=d0;dual[k]+=d1;
                    if(!update_pair(i,k,d0,d1))return false;}
            }
        }
        ready=certified();if(ready)return ready>0;
    }
    return false;
}
template<class Equality> RS_HD inline bool stationarity_cone(int q,int count,const double* gradient,double* matrix,double* dual,double* descent,Equality equality,double tolerance,double& certificate,double* norms=nullptr){
    return stationarity_cone_interruptible(q,count,gradient,matrix,dual,descent,equality,tolerance,certificate,norms,[]{return false;});
}
__global__ void lm_stationarity_host_result(SolverView s,bool valid,double certificate){
    if(threadIdx.x||blockIdx.x)return;
    if(valid)s.scalars[1]=certificate;else s.scalars[2]+=1.;
}
__global__ void lm_stationarity_project(LinearView l,SolverView s,const StationarityFace* faces,int count,const double* shared,double* matrix,double* dual,double requested){
    if(threadIdx.x||blockIdx.x)return;double scale=1.;for(int j=0;j<l.q;++j)scale=fmax(scale,fabs(shared[l.nnz+j]));
    double certificate=0.,tol=fmin(.1*requested,128*std::numeric_limits<double>::epsilon()*scale);
    if(!stationarity_cone(l.q,count,shared+l.nnz,matrix,dual,s.residual,[&](int i){return faces[i].equality;},tol,certificate,count?dual+count:nullptr))s.scalars[2]+=1.;
    else s.scalars[1]=certificate;
}

__global__ void lm_stationarity_metrics(View v,LinearView l,SolverView s,const double* shared,const double* g,
        const StationarityFace* faces,int count,const double* matrix,const double* dual,double* out){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(l.q)){
        const double raw=shared[l.nnz+i];BoxTangentSigns signs;
        for(size_t camera=0;camera<v.nrig;++camera){auto cv=camera_view(v,camera);auto cl=camera_linear_view(l,v.nr,camera);auto cs=camera_solver_view(s,v.nr,camera);
        if(cl.rig_directions)for(int k=15;k<int(v.nr);++k){if(cs.rig_bound_columns[k]!=int(i))continue;
            int chart=k<24?k-9:(k<38?k-11:k-12);double lo=cl.rig_lower[k],hi=cl.rig_upper[k];
            if(k==37){lo=fmax(lo,-cl.max_readout);hi=fmin(hi,cl.max_readout);}
            signs.restrict(cv.rig[k],lo,hi,cl.rig_directions[size_t(chart)*l.q+i]);}
        }
        atomic_max_nonnegative(out,fabs(raw));atomic_max_nonnegative(out+2,fabs(signs.project_gradient(raw)));
        double residual=s.residual[i]+raw;for(int k=0;k<count;++k)residual-=dual[k]*matrix[size_t(k)*l.q+i];
        atomic_max_nonnegative(out+6,fabs(residual));
    }
    if(i<size_t(s.nlocal)){
        const double raw=g[s.local_gradient[i]];BoxTangentSigns signs;
        signs.restrict(v.stations[s.representatives[i]],s.lower[i],s.upper[i],l.scales[i]);
        atomic_max_nonnegative(out+1,fabs(raw));atomic_max_nonnegative(out+3,fabs(signs.project_gradient(raw)));
    }
    if(i<size_t(count)){
        double direction=0.;for(int j=0;j<l.q;++j)direction+=matrix[i*l.q+j]*s.residual[j];
        atomic_max_nonnegative(out+4,faces[i].equality?fabs(direction):fmax(0.,-direction));
        atomic_max_nonnegative(out+5,fabs(dual[i]*direction));
        atomic_max_nonnegative(out+7,fmax(0.,-faces[i].margin));
    }
}
__global__ void lm_stationarity_export_faces(const StationarityFace* faces,int count,const double* dual,double* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto a=faces[i];const auto c=a.camera;double* row=out+24*size_t(i);
    row[0]=a.family;row[1]=a.index;row[2]=a.other;row[3]=a.side;row[4]=a.equality;row[5]=a.sign;row[6]=a.station;
    row[7]=a.margin;row[8]=a.scale;row[9]=dual[i];row[10]=dual[count+i];row[11]=c.kind;row[12]=c.cell;
    row[13]=c.channel;row[14]=c.u;row[15]=c.sign;row[16]=c.minimum;row[17]=c.limit;row[18]=c.nx;row[19]=c.ny;row[20]=c.nz;
    row[21]=c.rate_u[0];row[22]=c.rate_u[1];row[23]=c.rate_sign[0]+3*c.rate_sign[1];
}
