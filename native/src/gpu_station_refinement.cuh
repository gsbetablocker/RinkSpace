/* Safeguarded coordinate minimization for independent raw pixel factors.
 * Samples navigate the current native curve; they are never observations.
 * A station is committed only when its exact native pixel distance decreases.
 */
struct StationSample {int row,periodic;double s;};
struct StationSpan {int begin=0,count=0,periodic=0;double lo=0,hi=1;};
RS_HD inline Status station_uv(const View& v,int row,double s,Vec2& uv){
    Vec3 x{},t{};auto status=geometry_at(v,row,s,x,t);if(status!=Status::ok)return status;
    MotionProjection p{};status=project_motion(motion_rig(v.rig,v.nr==43?v.rig+38:nullptr),
        {v.controls,v.nc,v.first,v.dt},v.height,x,v.times[row],p,v.projection_time_half);uv=p.uv;return status;
}
RS_HD inline double station_cost(Vec2 a,Vec2 b){double x=a.x-b.x,y=a.y-b.y;return .5*(x*x+y*y);}
__global__ void station_sample_kernel(View v,const StationSample* samples,size_t count,Vec2* uv){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=count)return;
    auto a=samples[k];double s=a.s*(a.periodic?v.course->perimeter():1.);Vec2 p{};
    if(station_uv(v,a.row,s,p)!=Status::ok)p.x=p.y=std::numeric_limits<double>::quiet_NaN();uv[k]=p;
}
__global__ void station_refine_kernel(View v,const StationSpan* spans,const StationSample* samples,const Vec2* grid,double* stats){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(size_t(row)>=v.n)return;auto span=spans[row];if(!span.count)return;
    const double unit=span.periodic?v.course->perimeter():1.,lo=span.lo*unit,hi=span.hi*unit;
    Vec2 observed{v.observed[2*row],v.observed[2*row+1]},uv{};double original=v.stations[row];
    double before=station_uv(v,row,original,uv)==Status::ok?station_cost(uv,observed):std::numeric_limits<double>::infinity();
    double best=before,s=original,guess=original,estimate=before,vertex=original,vertex_cost=before;
    for(int j=0;j<span.count;++j){int k=span.begin+j;Vec2 a=grid[k];if(!finite_scalar(a.x))continue;
        double loss=station_cost(a,observed);
        if(loss<vertex_cost){vertex_cost=loss;vertex=samples[k].s*unit;}
        if(loss<estimate){estimate=loss;guess=samples[k].s*unit;}
        if(j+1<span.count){Vec2 b=grid[k+1];if(!finite_scalar(b.x))continue;
            double dx=b.x-a.x,dy=b.y-a.y,norm=dx*dx+dy*dy;
            if(norm>0){double t=fmax(0.,fmin(1.,((observed.x-a.x)*dx+(observed.y-a.y)*dy)/norm));
                loss=station_cost({a.x+t*dx,a.y+t*dy},observed);
                if(loss<estimate){estimate=loss;guess=unit*(samples[k].s+t*(samples[k+1].s-samples[k].s));}}}
    }
    if(vertex_cost<best){best=vertex_cost;s=vertex;}
    if(station_uv(v,row,guess,uv)==Status::ok){double loss=station_cost(uv,observed);if(loss<best){best=loss;s=guess;}}
    if(!finite_scalar(best)){atomicAdd(stats+2,1.);return;}
    // Analytic tangent, exact projection and monotone backtracking. This is
    // only a coordinate step; every rig/scene/motion parameter remains fixed.
    for(int iteration=0;iteration<32;++iteration){
        Vec3 x{},t{};Vec2 tangent{};MotionProjection p{};
        if(profile_projection(v,row,s,x,t,p,tangent)!=Status::ok)break;
        double norm=tangent.x*tangent.x+tangent.y*tangent.y;if(!(norm>0))break;
        double ds=-((p.uv.x-observed.x)*tangent.x+(p.uv.y-observed.y)*tangent.y)/norm;
        ds=fmax(-.25*(hi-lo),fmin(.25*(hi-lo),ds));
        bool moved=false;
        for(int line=0;line<32;++line){double next=fmax(lo,fmin(hi,s+ds));if(next==s)break;
            if(station_uv(v,row,next,uv)==Status::ok){double loss=station_cost(uv,observed);
                if(loss<best){s=next;best=loss;moved=true;break;}}
            ds*=.5;
        }
        if(!moved)break;
    }
    if(best<before){const_cast<double*>(v.stations)[row]=s;atomicAdd(stats,1.);
        if(finite_scalar(before))atomicAdd(stats+1,before-best);
        atomic_max_nonnegative(stats+3,fabs(s-original));}
}
struct StationRefinement {
    DeviceBuffer<StationSample> samples;DeviceBuffer<StationSpan> spans;
    DeviceBuffer<Vec2> grid;DeviceBuffer<double> stats;
    size_t bytes()const{return samples.bytes()+spans.bytes()+grid.bytes()+stats.bytes();}
    void prepare(const View& v,const Linearization& l,const double* lo,const double* hi,cudaStream_t stream){
        if(!l.host.pixel_residuals)return;
        std::vector<int> uses(l.scales.size(),0);for(int id:l.row_local)if(id>=0)++uses[id];
        std::vector<StationSample> source;std::vector<StationSpan> domains(v.n);
        std::map<std::tuple<int,double,double,double>,StationSpan> cache;
        for(size_t row=0;row<v.n;++row){int id=l.row_local[row];if(id<0||uses[id]!=1||lo[id]==hi[id])continue;
            auto key=std::make_tuple(v.codes[row],v.times[row],lo[id],hi[id]);auto it=cache.find(key);
            if(it!=cache.end()){domains[row]=it->second;continue;}
            int kind=v.paths?v.paths[7*v.codes[row]]:0;bool periodic=kind==7;
            StationSpan span{};span.begin=Linearization::bounded(source.size());span.periodic=periodic;
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
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        size_t planned=source.size()*(sizeof(StationSample)+sizeof(Vec2))+domains.size()*sizeof(StationSpan)+4*sizeof(double);
        if(planned>free)throw std::runtime_error("native curve search exceeds available device memory");
        samples.upload(source,stream);spans.upload(domains,stream);grid.allocate(source.size());stats.allocate(4);
    }
    void run(const View& v,cudaStream_t stream,double* out){
        cuda_check(cudaMemsetAsync(stats.data,0,stats.bytes(),stream));
        if(samples.count){station_sample_kernel<<<unsigned((samples.count+127)/128),128,0,stream>>>(v,samples.data,samples.count,grid.data);
            station_refine_kernel<<<unsigned((v.n+127)/128),128,0,stream>>>(v,spans.data,samples.data,grid.data,stats.data);}
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(out,stats.data,stats.bytes(),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));
    }
};
