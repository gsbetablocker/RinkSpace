/* Original-objective stationarity in the current feasible tangent cone.
 * The cone uses every active physical face, including coupled parameter maps.
 * All numerical work stays on the device; LM damping is not part of this test.
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
__global__ void lm_stationarity_faces(View v,LinearView l,SolverView s,StationarityFace* faces,size_t capacity,int* count){
    if(threadIdx.x||blockIdx.x)return;count[0]=count[1]=0;
    size_t camera_index=0;
    auto offer=[&](StationarityFace a,double margin,double scale=1.){
        const double eps=stationarity_active_tolerance(scale);
        if(!finite_scalar(margin)){count[1]=1;return;}
        if(margin < -eps)count[1]=1;
        if(!a.equality&&margin>eps)return;
        if(size_t(count[0])>=capacity){count[1]=1;return;}
        if(a.family==0)a.index+=int(camera_index*v.nr);else if(a.family==3)a.index=int(camera_index);else if(a.family==5)a.index+=int(4*camera_index);
        a.margin=margin;a.scale=scale;faces[count[0]++]=a;
    };
    auto bounds=[&](int family,int index,double value,double lo,double hi){
        StationarityFace a{};a.family=family;a.index=index;
        if(lo==hi&&finite_scalar(lo)){a.equality=true;offer(a,value-lo,fabs(value)+fabs(lo));return;}
        if(finite_scalar(lo))offer(a,value-lo,fabs(value)+fabs(lo));
        a.sign=-1.;if(finite_scalar(hi))offer(a,hi-value,fabs(value)+fabs(hi));
    };
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
    camera_index=0;
    if(l.scene_directions){
        for(size_t k=0;k<v.nscene;++k)bounds(1,int(k),v.scene_values[k],l.scene_lower[k],l.scene_upper[k]);
        if(v.options&4u){
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
        for(size_t i=0;i<s.print_count;++i){
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
    for(camera_index=0;camera_index<v.nrig;++camera_index){auto cv=camera_view(v,camera_index);auto cl=camera_linear_view(l,v.nr,camera_index);
    if(cl.mount_side)for(int kind=0;kind<4;++kind){
        if(kind==2&&(cl.mount_side==2||cl.mount_side==3))continue;if(kind==3&&cl.mount_side==1)continue;
        StationarityFace a{};a.family=5;a.index=kind;offer(a,mount_constraint_value(cv,cl,s,kind,-1).a);
    }
    if(cl.envelope.enabled){
        auto rig=motion_rig(cv.rig);MotionSpline spline{v.controls,v.nc,v.first,v.dt};
        bool static_axis_seen[2]={false,false};
        for(int cell=1;cell<int(v.nc)-2;++cell)camera_cell_constraints(rig,spline,cl.envelope,cell,[&](CameraConstraint camera){
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
__device__ inline double stationarity_direction(View v,LinearView l,SolverView s,StationarityFace a,int j){
    size_t camera=a.family==0?size_t(a.index)/v.nr:(a.family==3?size_t(a.index):(a.family==5?size_t(a.index)/4:0));
    if(a.family==0)a.index%=int(v.nr);else if(a.family==5)a.index%=4;
    l=camera_linear_view(l,v.nr,camera);v=camera_view(v,camera);
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
template<class Equality> RS_HD inline bool stationarity_cone(int q,int count,const double* gradient,double* matrix,double* dual,double* descent,Equality equality,double tolerance,double& certificate,double* norms=nullptr){
    for(int j=0;j<q;++j){if(!finite_scalar(gradient[j]))return false;descent[j]=-gradient[j];}
    for(int i=0;i<count;++i){double norm=0.;double* a=matrix+size_t(i)*q;
        for(int j=0;j<q;++j){if(!finite_scalar(a[j]))return false;norm+=a[j]*a[j];}
        if(!finite_scalar(norm))return false;norm=sqrt(norm);if(norms)norms[i]=norm;for(int j=0;j<q;++j)a[j]=norm?a[j]/norm:0.;dual[i]=0.;}
    for(int sweep=0;sweep<4096;++sweep){
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;const double* a=matrix+size_t(i)*q;double value=0.;
            for(int j=0;j<q;++j)value+=a[j]*descent[j];
            if(!finite_scalar(value))return false;
            double next=equality(i)?dual[i]-value:fmax(0.,dual[i]-value),change=next-dual[i];dual[i]=next;
            if(!finite_scalar(change))return false;
            if(change!=0.)for(int j=0;j<q;++j){descent[j]+=change*a[j];if(!finite_scalar(descent[j]))return false;}}
        // Cyclic exact two-coordinate minimization of the SAME convex dual.
        // Closely aligned active faces otherwise exchange their multipliers
        // extremely slowly under scalar Hildreth sweeps. Bound candidates keep
        // inequality multipliers nonnegative; equality multipliers stay free.
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;const double* a=matrix+size_t(i)*q;
            for(int k=i+1;k<count;++k){if(norms&&norms[k]==0.)continue;const double* b=matrix+size_t(k)*q;
                double cross=0.;for(int j=0;j<q;++j)cross+=a[j]*b[j];
                if(cross==0.)continue;cross=fmax(-1.,fmin(1.,cross));
                double r0=0.,r1=0.;for(int j=0;j<q;++j){r0+=a[j]*descent[j];r1+=b[j]*descent[j];}
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
                    for(int j=0;j<q;++j){descent[j]+=d0*a[j]+d1*b[j];if(!finite_scalar(descent[j]))return false;}}
            }
        }
        double error=0.;
        for(int i=0;i<count;++i){if(norms&&norms[i]==0.)continue;const double* a=matrix+size_t(i)*q;double value=0.;for(int j=0;j<q;++j)value+=a[j]*descent[j];
            if(!finite_scalar(value))return false;
            error=fmax(error,(equality(i)||dual[i]>0.)?fabs(value):fmax(0.,-value));}
        if(error<=tolerance){
            certificate=error;
            for(int j=0;j<q;++j){double residual=descent[j]+gradient[j];for(int i=0;i<count;++i)residual-=dual[i]*matrix[size_t(i)*q+j];
                if(!finite_scalar(residual))return false;
                certificate=fmax(certificate,fabs(residual));certificate=fmax(certificate,fabs(descent[j]));}
            return finite_scalar(certificate);
        }
    }
    return false;
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
