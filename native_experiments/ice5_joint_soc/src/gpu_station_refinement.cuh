/* Safeguarded coordinate minimization for independent raw pixel factors.
 * Samples navigate the current native curve; they are never observations.
 * A station is committed only when its exact native pixel distance decreases.
 */
struct StationSample {int row,periodic;double s;};
struct StationFrame {int camera;double time;};
struct StationSpan {int begin=0,count=0,periodic=0;double lo=0,hi=1,lower=0,upper=1;};
struct StationGeometry {Vec3 xyz;Status status;};
RS_HD inline Status station_projection(const View& source,int row,double s,MotionProjection& p,const MotionProjectionContext* context=nullptr){
    const auto v=row_camera_view(source,size_t(row));
    Vec3 x{},t{};auto status=geometry_at(v,row,s,x,t);if(status!=Status::ok)return status;
    return project_motion(motion_rig(v.rig,v.nr==43?v.rig+38:nullptr),
        {v.controls,v.nc,v.first,v.dt},v.height,x,v.times[row],p,v.projection_time_half,context);
}
RS_HD inline Status station_uv(const View& source,int row,double s,Vec2& uv,const MotionProjectionContext* context=nullptr){
    MotionProjection p{};auto status=station_projection(source,row,s,p,context);uv=p.uv;return status;
}
RS_HD inline double station_cost(Vec2 a,Vec2 b){double x=a.x-b.x,y=a.y-b.y;return .5*(x*x+y*y);}
// Periodic samples describe one circumference, while declared station bounds
// remain in the caller's unwrapped arclength chart. Only equivalent samples
// inside that original support are eligible to become accepted coordinates.
RS_HD inline bool station_in_support(const StationSpan& span,double period,double original,double& s){
    if(span.periodic){
        s+=std::round((original-s)/period)*period;
        if(s<span.lower)s+=std::ceil((span.lower-s)/period)*period;
        if(s>span.upper)s-=std::ceil((s-span.upper)/period)*period;
    }
    return finite_scalar(s)&&s>=span.lower&&s<=span.upper;
}
__global__ void station_frame_context_kernel(View v,const StationFrame* frames,size_t count,MotionProjectionContext* contexts){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=count)return;const auto frame=frames[k];auto cv=camera_view(v,size_t(frame.camera));
    prepare_motion_projection_context(motion_rig(cv.rig,cv.nr==43?cv.rig+38:nullptr),{cv.controls,cv.nc,cv.first,cv.dt},frame.time,contexts[k]);
}
__global__ void station_geometry_kernel(View v,const StationSample* samples,size_t count,StationGeometry* geometry){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=count)return;
    const auto a=samples[k];const auto cv=row_camera_view(v,size_t(a.row));Vec3 tangent{};
    const double s=a.s*(a.periodic?v.course->perimeter():1.);
    geometry[k].status=geometry_at(cv,a.row,s,geometry[k].xyz,tangent);
}
__global__ void station_sample_kernel(View v,const StationSample* samples,size_t count,Vec2* uv,const int* frame_ids,const MotionProjectionContext* contexts,
        const int* geometry_ids=nullptr,const StationGeometry* geometry=nullptr){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=count)return;
    auto a=samples[k];double s=a.s*(a.periodic?v.course->perimeter():1.);Vec2 p{};
    Status status;
    if(geometry){const auto g=geometry[geometry_ids[k]];status=g.status;
        if(status==Status::ok){const auto cv=row_camera_view(v,size_t(a.row));MotionProjection projection{};
            status=project_motion(motion_rig(cv.rig,cv.nr==43?cv.rig+38:nullptr),{cv.controls,cv.nc,cv.first,cv.dt},cv.height,g.xyz,cv.times[a.row],projection,cv.projection_time_half,contexts+frame_ids[a.row]);p=projection.uv;}}
    else status=station_uv(v,a.row,s,p,contexts+frame_ids[a.row]);
    if(status!=Status::ok)p.x=p.y=std::numeric_limits<double>::quiet_NaN();uv[k]=p;
}
// The pixel objective along a projected curve can have much more curvature
// than J_s^T J_s, especially for a foreshortened circle with nonzero residual.
// Keep the global navigation above, then minimize actual native pixel costs
// with safeguarded parabolic interpolation and golden-section contraction.
// Only strict exact-cost improvements become output stations. The interval
// termination is binary64 coordinate precision, never a fit/run iteration cap.
RS_HD inline void station_bounded_refine(const View& v,int row,const StationSpan& span,double unit,
        const MotionProjectionContext* context,double& station,double& best,MotionProjection& best_projection){
    double a=span.lower,b=span.upper;
    if(span.periodic){
        // Equivalent positions repeat every unit. One circumference covers the
        // local basin while retaining the caller's original unwrapped support.
        a=fmax(a,station-.5*unit);b=fmin(b,a+unit);a=fmax(span.lower,b-unit);
    }
    if(!finite_scalar(a)||!finite_scalar(b)||!(a<b)||station<a||station>b||best==0.)return;
    // Most warm smooth points already have no resolvable tangent-model
    // improvement. Avoid a full scalar search for these rows. At a corner a
    // zero tangent on one side says nothing about the other, so always search.
    double piece_lo,piece_hi;bool corner=false;
    if(station_piece(v,size_t(row),station,piece_lo,piece_hi,corner)&&!corner){
        Vec3 xyz{},tangent_world{};Vec2 tangent{};const auto cv=row_camera_view(v,size_t(row));
        if(geometry_at(cv,row,station,xyz,tangent_world)==Status::ok&&
           motion_station_partial(motion_rig(cv.rig,cv.nr==43?cv.rig+38:nullptr),
             {cv.controls,cv.nc,cv.first,cv.dt},cv.height,xyz,tangent_world,cv.times[row],best_projection,tangent)==Status::ok){
            const double norm=tangent.x*tangent.x+tangent.y*tangent.y;
            const double gradient=(best_projection.uv.x-v.observed[2*row])*tangent.x+
                                  (best_projection.uv.y-v.observed[2*row+1])*tangent.y;
            if(norm>0.&&gradient*gradient<=64*std::numeric_limits<double>::epsilon()*fmax(1.,best)*norm)return;
        }
    }
    double x=station,w=x,z=x,fx=best,fw=best,fz=best,d=0.,e=0.;
    constexpr double golden=.3819660112501051518;
    for(;;){
        const double middle=a+.5*(b-a);
        const double tolerance=32*std::numeric_limits<double>::epsilon()*(1.+fabs(x)+fabs(b-a));
        const double twice=2*tolerance;
        if(fabs(x-middle)<=twice-.5*(b-a))break;
        bool interpolate=false;
        if(fabs(e)>tolerance){
            const double r=(x-w)*(fx-fz),q0=(x-z)*(fx-fw);
            double p=(x-z)*q0-(x-w)*r,q=2*(q0-r);
            if(q>0.)p=-p;q=fabs(q);
            const double previous=e;e=d;
            if(q>0.&&finite_scalar(p)&&finite_scalar(q)&&fabs(p)<fabs(.5*q*previous)&&p>q*(a-x)&&p<q*(b-x)){
                d=p/q;const double proposed=x+d;
                if(proposed-a<twice||b-proposed<twice)d=copysign(tolerance,middle-x);
                interpolate=true;
            }
        }
        if(!interpolate){e=x<middle?b-x:a-x;d=golden*e;}
        const double direction=d!=0.?d:middle-x;
        const double u=x+(fabs(d)>=tolerance?d:copysign(tolerance,direction));
        if(!(u>a&&u<b)||u==x)break;
        MotionProjection projection{};
        const double fu=station_projection(v,row,u,projection,context)==Status::ok?
            station_cost(projection.uv,{v.observed[2*row],v.observed[2*row+1]}):std::numeric_limits<double>::infinity();
        if(fu<best){best=fu;station=u;best_projection=projection;}
        if(fu<=fx){
            if(u<x)b=x;else a=x;
            z=w;fz=fw;w=x;fw=fx;x=u;fx=fu;
        }else{
            if(u<x)a=u;else b=u;
            if(fu<=fw||w==x){z=w;fz=fw;w=u;fw=fu;}
            else if(fu<=fz||z==x||z==w){z=u;fz=fu;}
        }
    }
}
__global__ void station_refine_kernel(View v,const StationSpan* spans,const StationSample* samples,const Vec2* grid,const int* frame_ids,const MotionProjectionContext* contexts,double* stats,bool global_search=true,double* output=nullptr,int* statuses=nullptr){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(size_t(row)>=v.n)return;auto span=spans[row];if(!span.count)return;
    const double unit=span.periodic?v.course->perimeter():1.,lo=span.lower,hi=span.upper;
    const auto context=contexts+frame_ids[row];
    Vec2 observed{v.observed[2*row],v.observed[2*row+1]};double original=v.stations[row];
    MotionProjection best_projection{},candidate_projection{};
    double before=station_projection(v,row,original,best_projection,context)==Status::ok?station_cost(best_projection.uv,observed):std::numeric_limits<double>::infinity();
    double best=before,s=original,guess=original,estimate=before,vertex=original,vertex_cost=before;
    for(int j=0;global_search&&j<span.count;++j){int k=span.begin+j;Vec2 a=grid[k];if(!finite_scalar(a.x))continue;
        double loss=station_cost(a,observed),candidate=samples[k].s*unit;
        if(station_in_support(span,unit,original,candidate)){
            if(loss<vertex_cost){vertex_cost=loss;vertex=candidate;}
            if(loss<estimate){estimate=loss;guess=candidate;}}
        if(j+1<span.count){Vec2 b=grid[k+1];if(!finite_scalar(b.x))continue;
            double dx=b.x-a.x,dy=b.y-a.y,norm=dx*dx+dy*dy;
            if(norm>0){double t=fmax(0.,fmin(1.,((observed.x-a.x)*dx+(observed.y-a.y)*dy)/norm));
                loss=station_cost({a.x+t*dx,a.y+t*dy},observed);
                candidate=unit*(samples[k].s+t*(samples[k+1].s-samples[k].s));
                if(loss<estimate&&station_in_support(span,unit,original,candidate)){estimate=loss;guess=candidate;}}}
    }
    if(vertex_cost<best&&station_projection(v,row,vertex,candidate_projection,context)==Status::ok){double loss=station_cost(candidate_projection.uv,observed);if(loss<best){best=loss;s=vertex;best_projection=candidate_projection;}}
    if(guess!=s&&station_projection(v,row,guess,candidate_projection,context)==Status::ok){double loss=station_cost(candidate_projection.uv,observed);if(loss<best){best=loss;s=guess;best_projection=candidate_projection;}}
    // A restricted interval can be narrower than the sample grid. Its finite
    // endpoints are still eligible and are evaluated by the exact projection.
    for(int edge=0;edge<2;++edge){double endpoint=edge?hi:lo;
        if(endpoint!=s&&finite_scalar(endpoint)&&station_projection(v,row,endpoint,candidate_projection,context)==Status::ok){
            double loss=station_cost(candidate_projection.uv,observed);if(loss<best){best=loss;s=endpoint;best_projection=candidate_projection;}}}
    if(!finite_scalar(best)){atomicAdd(stats+2,1.);return;}
    station_bounded_refine(v,row,span,unit,context,s,best,best_projection);
    if(best<before&&s>=lo&&s<=hi){const_cast<double*>(v.stations)[row]=s;atomicAdd(stats,1.);
        if(finite_scalar(before))atomicAdd(stats+1,before-best);
        atomic_max_nonnegative(stats+3,fabs(s-original));
        if(output)evaluate_row(v,size_t(row),output+11*row,statuses+row);}
}
struct StationRefinement {
    DeviceBuffer<ScenePrimitive<double>> primitives;
    DeviceBuffer<StationSample> samples;DeviceBuffer<StationSpan> spans;
    DeviceBuffer<Vec2> grid;DeviceBuffer<double> stats;
    DeviceBuffer<StationFrame> frames;DeviceBuffer<int> frame_ids;DeviceBuffer<MotionProjectionContext> contexts;
    DeviceBuffer<StationSample> geometry_samples;DeviceBuffer<int> geometry_ids;DeviceBuffer<StationGeometry> geometry;
    size_t bytes()const{return primitives.bytes()+samples.bytes()+spans.bytes()+grid.bytes()+stats.bytes()+frames.bytes()+frame_ids.bytes()+contexts.bytes()+geometry_samples.bytes()+geometry_ids.bytes()+geometry.bytes();}
    void prepare(const View& v,const Linearization& l,const double* lo,const double* hi,cudaStream_t stream){
        if(!l.host.pixel_residuals)return;
        std::vector<int> uses(l.scales.size(),0);for(int id:l.row_local)if(id>=0)++uses[id];
        std::vector<StationSample> source;std::vector<StationSpan> domains(v.n);
        std::map<std::tuple<int,int,double,double,double>,StationSpan> cache;
        for(size_t row=0;row<v.n;++row){int id=l.row_local[row];if(id<0||uses[id]!=1||lo[id]==hi[id])continue;
            auto key=std::make_tuple(int(row_camera(v,row)),v.codes[row],v.times[row],lo[id],hi[id]);auto it=cache.find(key);
            if(it!=cache.end()){domains[row]=it->second;continue;}
            int kind=v.paths?v.paths[7*v.codes[row]]:0;bool periodic=kind==7;
            StationSpan span{};span.begin=Linearization::bounded(source.size());span.periodic=periodic;
            span.lower=lo[id];span.upper=hi[id];
            span.lo=periodic?0.:lo[id];span.hi=periodic?1.:hi[id];
            std::vector<double> knots;int pieces=(kind==1||kind==6||periodic)?256:128;
            for(int k=0;k<=pieces;++k)knots.push_back(span.lo+(span.hi-span.lo)*k/pieces);
            if(v.paths){const int* path=v.paths+7*v.codes[row];
                for(int k=0;k<path[6];++k){double s=v.path_data[path[5]+k];if(s>span.lo&&s<span.hi)knots.push_back(s);}
                if(kind==2||kind==4||kind==5){int count=path[4]/(kind==4?3:2);for(int k=1;k<count-1;++k){double s=double(k)/(count-1);if(s>span.lo&&s<span.hi)knots.push_back(s);}}}
            std::sort(knots.begin(),knots.end());knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
            for(double s:knots)source.push_back({int(row),int(periodic),s});span.count=Linearization::bounded(knots.size());
            cache.emplace(key,span);domains[row]=span;
        }
        std::map<std::pair<int,double>,int> frame_cache;std::vector<StationFrame> frame_source;std::vector<int> frame_indices(v.n,-1);
        for(size_t row=0;row<v.n;++row)if(domains[row].count){auto key=std::make_pair(int(row_camera(v,row)),v.times[row]);auto found=frame_cache.find(key);
            if(found==frame_cache.end()){int id=Linearization::bounded(frame_source.size());frame_source.push_back({key.first,key.second});frame_cache.emplace(key,id);frame_indices[row]=id;}
            else frame_indices[row]=found->second;}
        // Source geometry is camera/time independent. Share only exact path,
        // periodic chart and sample-coordinate identities; recompute values
        // for EVERY trial scene, never reuse values across scene generations.
        std::vector<StationSample> geometry_source;std::vector<int> geometry_indices;
        const char* geometry_option=std::getenv("RINKSPACE_STATION_GEOMETRY_CACHE");
        if(geometry_option&&std::string(geometry_option)!="0"&&std::string(geometry_option)!="1")throw std::invalid_argument("RINKSPACE_STATION_GEOMETRY_CACHE must be 0 or 1");
        if(!geometry_option||std::string(geometry_option)=="1"){
            std::map<std::tuple<int,int,double>,int> geometry_cache;geometry_indices.reserve(source.size());
            for(const auto a:source){auto key=std::make_tuple(v.codes[a.row],a.periodic,a.s);auto found=geometry_cache.find(key);
                if(found==geometry_cache.end()){int id=Linearization::bounded(geometry_source.size());geometry_source.push_back(a);geometry_cache.emplace(key,id);geometry_indices.push_back(id);}
                else geometry_indices.push_back(found->second);}}
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        size_t planned=source.size()*(sizeof(StationSample)+sizeof(Vec2))+domains.size()*sizeof(StationSpan)+4*sizeof(double)+
            frame_source.size()*(sizeof(StationFrame)+sizeof(MotionProjectionContext))+frame_indices.size()*sizeof(int)+
            geometry_source.size()*(sizeof(StationSample)+sizeof(StationGeometry))+geometry_indices.size()*sizeof(int)+(v.scene_paths?v.npaths*sizeof(ScenePrimitive<double>):0);
        if(planned>free)throw std::runtime_error("native curve search exceeds available device memory");
        samples.upload(source,stream);spans.upload(domains,stream);grid.allocate(source.size());stats.allocate(4);
        frames.upload(frame_source,stream);frame_ids.upload(frame_indices,stream);contexts.allocate(frame_source.size());
        geometry_samples.upload(geometry_source,stream);geometry_ids.upload(geometry_indices,stream);geometry.allocate(geometry_source.size());
        if(v.scene_paths)primitives.allocate(v.npaths);
    }
    void run(View v,cudaStream_t stream,double* out,bool global_search=true,double* output=nullptr,int* statuses=nullptr){
        if(primitives.count){scene_primitive_value_kernel<<<unsigned((v.npaths+127)/128),128,0,stream>>>(v,primitives.data);v.primitives=primitives.data;}
        cuda_check(cudaMemsetAsync(stats.data,0,stats.bytes(),stream));
        if(samples.count){station_frame_context_kernel<<<unsigned((frames.count+127)/128),128,0,stream>>>(v,frames.data,frames.count,contexts.data);
            if(global_search){
                if(geometry_samples.count)station_geometry_kernel<<<unsigned((geometry_samples.count+127)/128),128,0,stream>>>(v,geometry_samples.data,geometry_samples.count,geometry.data);
                station_sample_kernel<<<unsigned((samples.count+127)/128),128,0,stream>>>(v,samples.data,samples.count,grid.data,frame_ids.data,contexts.data,geometry_ids.data,geometry.data);}
            station_refine_kernel<<<unsigned((v.n+127)/128),128,0,stream>>>(v,spans.data,samples.data,grid.data,frame_ids.data,contexts.data,stats.data,global_search,output,statuses);}
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(out,stats.data,stats.bytes(),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));
    }
};
