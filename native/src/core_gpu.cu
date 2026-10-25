#include "rinkspace/core_api.h"
#include "rinkspace/entities.hpp"
#include "rinkspace/camera_state.hpp"
#include "rinkspace/motion_derivatives.hpp"
#include <cuda_runtime.h>
#ifdef RINKSPACE_HAVE_CUDSS
#include <cusolverDn.h>
#include <cudss.h>
#endif
#include <chrono>
#include <vector>
#include <string>
#include <exception>
#include <map>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <atomic>

namespace {
using namespace rinkspace;
using Clock=std::chrono::steady_clock;
double millis(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
thread_local std::string last_error;
struct View {
    const double *board,*layout,*rig,*controls,*stations,*times,*observed;
    const int* codes;
    const int* paths=nullptr;const double* path_data=nullptr;const BoardCourse* course=nullptr;size_t npaths=0;
    size_t nl,nc,n;unsigned options;double first,dt;int height;
};
RS_HD Status geometry_row(const View& v,size_t i,Vec3& xyz,Vec3& tangent){
    int code=v.codes[i],kind=-1;double parameters[8];
    if(v.paths){
        if(code<0||static_cast<size_t>(code)>=v.npaths)return static_cast<Status>(11);
        const int* path=v.paths+7*code;kind=path[0];
        const double* p=v.path_data+path[1];const double* vertices=v.path_data+path[3];
        if(kind==8)return sample_knot_board(*v.course,v.path_data+path[5],vertices,path[6],v.stations[i],xyz,tangent);
        return sample_entity_point(v.course,v.board[5],kind,p,vertices,path[4],v.stations[i],xyz,tangent);
    }
    if(!((code>=100&&code<132)||(code>=200&&code<204)||(code>=300&&code<312)||(code>=400&&code<=415)))
        return static_cast<Status>(11);
    const bool ellipse=v.nl==35&&(code==410||code==411);
    auto st=shared_entity_definition(v.board,6,v.layout,v.nl,code,kind,parameters,ellipse?8:6,v.options);
    if(st!=Status::ok)return st;
    return sample_entity_point(nullptr,v.board[5],kind,parameters,nullptr,0,v.stations[i],xyz,tangent);
}
RS_HD void evaluate_row(const View& v,size_t i,double* out,int* status){
    const double nan=std::numeric_limits<double>::quiet_NaN();for(int j=0;j<11;++j)out[j]=nan;
    Vec3 xyz{},tangent{};
    auto st=geometry_row(v,i,xyz,tangent);
    if(!finite_scalar(v.observed[2*i])||!finite_scalar(v.observed[2*i+1]))st=Status::invalid_argument;
    if(st==Status::ok){
        MotionProjection p{};const MotionSpline spline{v.controls,v.nc,v.first,v.dt};
        st=project_motion(motion_rig(v.rig),spline,v.height,xyz,v.times[i],p);
        if(st==Status::ok){
            store3(out,xyz);out[3]=p.uv.x;out[4]=p.uv.y;out[5]=p.depth;
            out[6]=p.row;out[7]=p.equality;out[8]=p.contraction;
            out[9]=p.uv.x-v.observed[2*i];out[10]=p.uv.y-v.observed[2*i+1];
        }
    }
    *status=static_cast<int>(st);
}
__global__ void evaluate_kernel(View view,double* output,int* statuses){
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<view.n)evaluate_row(view,i,output+11*i,statuses+i);
}
#include "gpu_linearization.cuh"
#ifdef RINKSPACE_HAVE_CUDSS
#include "gpu_solver.cuh"
#endif
struct Problem {
#ifdef RINKSPACE_HAVE_CUDSS
    std::unique_ptr<Solver> solver;
#endif
    std::atomic<bool> stop_requested{false};
    bool host_stale=false;
    void sync_host(){if(host_stale){cuda_check(cudaMemcpyAsync(input.data(),device_input,input.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));host_stale=false;}}
    std::unique_ptr<Linearization> linearization;
    std::vector<double> input;std::vector<int> codes;
    std::vector<int> path_meta;std::vector<double> geometry_data;
    std::unique_ptr<BoardCourse> host_course;
    DeviceBuffer<int> d_path_meta;DeviceBuffer<double> d_geometry_data;DeviceBuffer<BoardCourse> d_course;
    double *device_input=nullptr,*device_output=nullptr;int *device_codes=nullptr,*device_status=nullptr;
    cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,middle=nullptr,end=nullptr;
    View host{},device{};
    ~Problem(){
        if(stream)cudaStreamSynchronize(stream);
        linearization.reset();
#ifdef RINKSPACE_HAVE_CUDSS
        solver.reset();
#endif
        cudaFree(device_status);cudaFree(device_output);cudaFree(device_codes);cudaFree(device_input);
        if(start)cudaEventDestroy(start);if(middle)cudaEventDestroy(middle);if(end)cudaEventDestroy(end);
        if(stream)cudaStreamDestroy(stream);
    }
    bool checked(cudaError_t status){if(status==cudaSuccess)return true;last_error=cudaGetErrorString(status);return false;}
    View bind(const double* p,const int* c)const{
        View v=host;v.codes=c;v.board=p;v.layout=p+6;v.rig=v.layout+v.nl;
        v.controls=v.rig+38;v.stations=v.controls+3*v.nc;v.times=v.stations+v.n;v.observed=v.times+v.n;
        if(!path_meta.empty()){
            const bool gpu=c==device_codes;
            v.paths=gpu?d_path_meta.data:path_meta.data();
            v.path_data=gpu?d_geometry_data.data:geometry_data.data();
            v.course=gpu?d_course.data:host_course.get();v.npaths=path_meta.size()/7;
        }
        return v;
    }
};
}
extern "C" {
RS_API const char* rs_g1_last_error(){return last_error.c_str();}
RS_API void rs_g1_destroy(void* handle){delete static_cast<Problem*>(handle);}
RS_API int rs_g1_create(const double* board,size_t nb,const double* layout,size_t nl,unsigned options,
    const double* rig,size_t nr,const double* controls,size_t nc,double first,double dt,int height,
    const int* codes,const double* stations,const double* times,const double* observed,size_t count,
    void** handle,double* metrics,size_t nm){
    last_error.clear();if(handle)*handle=nullptr;
    const auto start=Clock::now();
    if(!handle||!metrics||nm!=10||nb!=6||(nl!=33&&nl!=35)||nr!=38||height<=1||!count||
       !codes||!stations||!times||!observed||!finite_array(board,nb)||!finite_array(layout,nl)||
       !finite_array(rig,nr)||!motion_rig(rig).valid()||
       !MotionSpline{controls,nc,first,dt}.valid()||(options&~15u)){
        last_error="invalid G1 immutable problem";return 1;
    }
    std::unique_ptr<Problem> p;
    try {
        p.reset(new Problem);
        p->host.nl=nl;p->host.nc=nc;p->host.n=count;p->host.options=options;
        p->host.first=first;p->host.dt=dt;p->host.height=height;
        auto append=[&](const double* b,size_t n){p->input.insert(p->input.end(),b,b+n);};
        append(board,nb);append(layout,nl);append(rig,nr);append(controls,3*nc);
        append(stations,count);append(times,count);append(observed,2*count);
        p->codes.assign(codes,codes+count);p->host=p->bind(p->input.data(),p->codes.data());
        auto context_start=Clock::now();
        if(!p->checked(cudaFree(nullptr)))return 20;
        metrics[1]=millis(context_start);
        size_t free_before,total,free_after,unused;
        int device;cudaDeviceProp properties{};
        if(!p->checked(cudaGetDevice(&device))||!p->checked(cudaGetDeviceProperties(&properties,device))||
           !p->checked(cudaStreamCreateWithFlags(&p->stream,cudaStreamNonBlocking))||
           !p->checked(cudaEventCreate(&p->start))||!p->checked(cudaEventCreate(&p->middle))||
           !p->checked(cudaEventCreate(&p->end))||!p->checked(cudaMemGetInfo(&free_before,&total)))return 20;
        const size_t inputs=p->input.size()*sizeof(double),ids=count*sizeof(int),outputs=count*11*sizeof(double);
        if(!p->checked(cudaMalloc(&p->device_input,inputs))||!p->checked(cudaMalloc(&p->device_codes,ids))||
           !p->checked(cudaMalloc(&p->device_output,outputs))||!p->checked(cudaMalloc(&p->device_status,ids))||
           !p->checked(cudaMemGetInfo(&free_after,&unused)))return 20;
        if(!p->checked(cudaEventRecord(p->start,p->stream))||
           !p->checked(cudaMemcpyAsync(p->device_input,p->input.data(),inputs,cudaMemcpyHostToDevice,p->stream))||
           !p->checked(cudaMemcpyAsync(p->device_codes,p->codes.data(),ids,cudaMemcpyHostToDevice,p->stream))||
           !p->checked(cudaEventRecord(p->end,p->stream))||!p->checked(cudaEventSynchronize(p->end)))return 20;
        float h2d;if(!p->checked(cudaEventElapsedTime(&h2d,p->start,p->end)))return 20;
        p->device=p->bind(p->device_input,p->device_codes);
        metrics[0]=millis(start);metrics[2]=h2d;metrics[3]=static_cast<double>(inputs+outputs+2*ids);
        metrics[4]=static_cast<double>(free_before);metrics[5]=static_cast<double>(free_after);metrics[6]=static_cast<double>(total);
        metrics[7]=device;metrics[8]=properties.major;metrics[9]=properties.minor;
        *handle=p.release();return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_gpu_attach_paths(void* handle,const int* paths,size_t np,const double* values,size_t nv,double* bytes){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||!p->path_meta.empty()||!paths||!np||np%7||!values||!finite_array(values,nv)||!bytes){
        last_error="invalid frozen geometry plan or already attached";return 1;}
    try{
        const auto* b=p->host.board;BoardShape shape{b[0],b[1],b[2],b[3],b[4],b[5]};
        if(!shape.valid())throw std::invalid_argument("invalid current board");
        auto course=std::make_unique<BoardCourse>(shape);
        for(size_t j=0;j<np;j+=7){
            const int* r=paths+j;int kind=r[0];
            if(kind<0||kind>8||r[1]<0||r[2]<0||r[3]<0||r[4]<0||r[5]<0||r[6]<0||
               size_t(r[1])+r[2]>nv||size_t(r[3])+r[4]>nv||size_t(r[5])+r[6]>nv)
                throw std::invalid_argument("invalid native path offsets/kind");
            if(kind==8){
                if(r[6]<2||r[4]!=2*r[6]||values[r[5]]!=0||values[r[5]+r[6]-1]!=1)
                    throw std::invalid_argument("compiled native chart needs exact 0..1 break stations");
                for(int k=1;k<r[6];++k)if(!(values[r[5]+k]>values[r[5]+k-1]))
                    throw std::invalid_argument("nonincreasing native chart breaks");
            }else{
                double station=.5,xyz[3];int status;
                auto valid=sample_entity_path(b,6,kind,values+r[1],r[2],values+r[3],r[4],&station,1,xyz,&status);
                if(valid!=Status::ok)throw std::invalid_argument("invalid native primitive definition "+std::to_string(j/7));
            }
        }
        for(int id:p->codes)if(id<0||size_t(id)>=np/7)throw std::invalid_argument("row references missing native path");
        p->host_course=std::move(course);p->path_meta.assign(paths,paths+np);p->geometry_data.assign(values,values+nv);
        p->d_path_meta.upload(p->path_meta,p->stream);p->d_geometry_data.upload(p->geometry_data,p->stream);
        p->d_course.upload(std::vector<BoardCourse>{*p->host_course},p->stream);
        cuda_check(cudaStreamSynchronize(p->stream));
        p->host=p->bind(p->input.data(),p->codes.data());p->device=p->bind(p->device_input,p->device_codes);
        *bytes=p->d_path_meta.bytes()+p->d_geometry_data.bytes()+p->d_course.bytes();return 0;
    }catch(const std::exception& e){last_error=e.what();return 24;}
}
RS_API int rs_g1_evaluate(void* handle,int backend,double* output,size_t no,int* statuses,size_t ns,
    double* metrics,size_t nm){
    last_error.clear();auto* p=static_cast<Problem*>(handle);
    if(!p||!output||!statuses||!metrics||nm!=3||no!=11*p->host.n||ns!=p->host.n||(backend!=0&&backend!=1)){
        last_error="invalid G1 evaluation output or backend";return 1;
    }
    const auto start=Clock::now();metrics[1]=metrics[2]=0.;
    if(backend==0){
        try{p->sync_host();}catch(const std::exception& e){last_error=e.what();return 20;}
        for(size_t i=0;i<p->host.n;++i)evaluate_row(p->host,i,output+11*i,statuses+i);
    }else{
        if(!p->checked(cudaEventRecord(p->start,p->stream)))return 20;
        evaluate_kernel<<<static_cast<unsigned>((p->host.n+127)/128),128,0,p->stream>>>(p->device,p->device_output,p->device_status);
        if(!p->checked(cudaGetLastError())||!p->checked(cudaEventRecord(p->middle,p->stream))||
           !p->checked(cudaMemcpyAsync(output,p->device_output,no*sizeof(double),cudaMemcpyDeviceToHost,p->stream))||
           !p->checked(cudaMemcpyAsync(statuses,p->device_status,ns*sizeof(int),cudaMemcpyDeviceToHost,p->stream))||
           !p->checked(cudaEventRecord(p->end,p->stream))||!p->checked(cudaEventSynchronize(p->end)))return 20;
        float kernel,d2h;
        if(!p->checked(cudaEventElapsedTime(&kernel,p->start,p->middle))||
           !p->checked(cudaEventElapsedTime(&d2h,p->middle,p->end)))return 20;
        metrics[1]=kernel;metrics[2]=d2h;
    }
    metrics[0]=millis(start);return 0;
}
RS_API int rs_g2_attach(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,const int* offsets,int ng,const int* members,size_t nmembers,
    const int* ranks,const int* weight_offsets,const double* weights,size_t nweights,double huber_delta,
    double* info,size_t ninfo){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!info||ninfo!=12||p->linearization){last_error="invalid or already-attached G2 problem";return 1;}
    const auto start=Clock::now();
    try{
        auto linear=std::make_unique<Linearization>();
        cuda_check(cudaEventRecord(p->start,p->stream));
        linear->prepare(p->host,q,dirs,ndirs,row_local,nrows,local_scales,nlocal,offsets,ng,members,nmembers,ranks,weight_offsets,weights,nweights,huber_delta,p->stream);
        cuda_check(cudaEventRecord(p->end,p->stream));cuda_check(cudaEventSynchronize(p->end));
        float transfer;cuda_check(cudaEventElapsedTime(&transfer,p->start,p->end));
        int maximum=0,total_rank=0;for(auto c:linear->components)maximum=std::max(maximum,c.local_count);
        for(auto g:linear->groups)total_rank+=g.rank;
        const double values[]={double(q),double(nlocal),double(ng),double(linear->components.size()),double(linear->white_count),
            double(linear->h_count),double(linear->g_count),double(linear->device_bytes()),millis(start),transfer,double(maximum),double(total_rank)};
        std::copy(values,values+12,info);p->linearization=std::move(linear);return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 20;}
}
RS_API int rs_g2_layout(void* handle,int* groups,size_t ng,int* components,size_t nc,int* locals,size_t nl){
    auto* p=static_cast<Problem*>(handle);
    if(!p||!p->linearization){last_error="missing G2 plan";return 1;}const auto& l=*p->linearization;
    if(!groups||!components||(nl&&!locals)||ng!=6*l.groups.size()||nc!=7*l.components.size()||nl!=l.component_locals.size()){
        last_error="G2 metadata output shape";return 1;}
    std::memcpy(groups,l.groups.data(),ng*sizeof(int));std::memcpy(components,l.components.data(),nc*sizeof(int));
    if(nl)std::memcpy(locals,l.component_locals.data(),nl*sizeof(int));return 0;
}
RS_API int rs_g2_linearize(void* handle,int backend,double* summary,size_t ns,double* timings,size_t nt){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->linearization||!summary||!timings||ns!=4||nt!=8||(backend!=0&&backend!=1)){
        last_error="invalid G2 linearization arguments";return 1;}
    auto& l=*p->linearization;const auto start=Clock::now();std::fill(timings,timings+8,0.);
    try{
        if(backend==0){p->sync_host();l.cpu(p->host,summary);}
        else{
            l.gpu_ready=false;
            cuda_check(cudaEventRecord(l.event[0],p->stream));
            linearize_kernel<<<static_cast<unsigned>((p->host.n+127)/128),128,0,p->stream>>>(p->device,l.device,p->device_output,p->device_status,l.d_jac.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[1],p->stream));
            whiten_kernel<<<l.host.ng,128,0,p->stream>>>(l.device,p->device_output,l.d_jac.data,l.d_white.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[2],p->stream));
            loss_kernel<<<(l.host.ng+127)/128,128,0,p->stream>>>(l.device,l.d_white.data,l.d_stats.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[3],p->stream));
            block_kernel<<<l.host.ncomponents,128,0,p->stream>>>(l.device,l.d_white.data,l.d_stats.data,l.d_h.data,l.d_g.data);
            cuda_check(cudaGetLastError());
            shared_kernel<<<(l.host.q*l.host.q+l.host.q+127)/128,128,0,p->stream>>>(l.device,l.d_h.data,l.d_g.data,l.d_shared.data);
            cuda_check(cudaGetLastError());
            summary_kernel<<<1,1,0,p->stream>>>(p->host.n,p->device_status,l.host.ng,l.d_stats.data,l.d_summary.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[4],p->stream));
            cuda_check(cudaMemcpyAsync(summary,l.d_summary.data,4*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaEventRecord(l.event[5],p->stream));cuda_check(cudaEventSynchronize(l.event[5]));
            for(int stage=0;stage<5;++stage){float ms;cuda_check(cudaEventElapsedTime(&ms,l.event[stage],l.event[stage+1]));timings[stage+1]=ms;}
            timings[7]=4*sizeof(double);l.gpu_ready=summary[0]==0;
        }
        timings[0]=millis(start);timings[6]=double(l.device_bytes());
        if(summary[0]!=0||!finite_array(summary,4)){last_error="G2 invalid rows retained; no valid block system (failed="+std::to_string(summary[0])+")";return 21;}
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_export_blocks(void* handle,int backend,double* h,size_t nh,double* g,size_t ng,double* shared,size_t nq){
    auto* p=static_cast<Problem*>(handle);
    if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;
    if(!h||!g||!shared||nh!=l.h_count||ng!=l.g_count||nq!=static_cast<size_t>(l.host.q*l.host.q+l.host.q)||
       (backend!=0&&backend!=1)||(backend==0?!l.cpu_ready:!l.gpu_ready)){last_error="invalid G2 explicit block export";return 1;}
    try{
        if(backend==0){std::copy(l.cpu_h.begin(),l.cpu_h.end(),h);std::copy(l.cpu_g.begin(),l.cpu_g.end(),g);std::copy(l.cpu_shared.begin(),l.cpu_shared.end(),shared);}
        else{cuda_check(cudaMemcpyAsync(h,l.d_h.data,nh*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaMemcpyAsync(g,l.d_g.data,ng*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaMemcpyAsync(shared,l.d_shared.data,nq*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));}
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_probe_rows(void* handle,int backend,const int* rows,size_t n,double* jac,size_t nj){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;
    const size_t width=2*(l.host.q+1);
    if(!rows||!jac||nj!=n*width||(backend!=0&&backend!=1)||(backend==0?!l.cpu_ready:!l.gpu_ready)){last_error="invalid G2 row probe";return 1;}
    for(size_t i=0;i<n;++i)if(rows[i]<0||rows[i]>=static_cast<int>(p->host.n)){last_error="G2 row probe outside source";return 1;}
    try{
        for(size_t i=0;i<n;++i){
            if(backend==0)std::copy_n(l.cpu_jac.data()+rows[i]*width,width,jac+i*width);
            else cuda_check(cudaMemcpyAsync(jac+i*width,l.d_jac.data+rows[i]*width,width*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        }
        if(backend==1)cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_probe_group(void* handle,int backend,int group,double* white,size_t nw,double* loss,size_t ns){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;
    if(group<0||group>=l.host.ng||!white||!loss||ns!=3||(backend!=0&&backend!=1)||(backend==0?!l.cpu_ready:!l.gpu_ready)){last_error="invalid G2 group probe";return 1;}
    const auto g=l.groups[group];const size_t count=static_cast<size_t>(g.rank)*(l.components[g.component].dim+1);
    if(nw!=count){last_error="G2 group probe shape";return 1;}
    try{
        if(backend==0){std::copy_n(l.cpu_white.data()+g.white,count,white);std::copy_n(l.cpu_stats.data()+3*group,3,loss);}
        else{if(count)cuda_check(cudaMemcpyAsync(white,l.d_white.data+g.white,count*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaMemcpyAsync(loss,l.d_stats.data+3*group,3*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));}
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}


#ifdef RINKSPACE_HAVE_CUDSS
RS_API int rs_g3_attach(void* handle,const double* lower,const double* upper,size_t nl,const double* options,size_t no,double* info,size_t ni){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->linearization||p->solver||!info||ni!=8){last_error="invalid or already-attached G3 solver";return 1;}
    const auto start=Clock::now();
    try{
        p->sync_host();auto s=std::make_unique<Solver>();
        s->prepare(p->host,*p->linearization,p->input.size(),lower,upper,nl,options,no,p->stream);
        int major,minor,patch;sparse_check(cudssGetProperty(MAJOR_VERSION,&major));sparse_check(cudssGetProperty(MINOR_VERSION,&minor));sparse_check(cudssGetProperty(PATCH_LEVEL,&patch));
        double values[]={millis(start),double(s->device_bytes()),double(s->m),double(p->linearization->host.q*(p->linearization->host.q+1)/2),double(major),double(minor),double(patch),0.};
        std::copy(values,values+8,info);p->solver=std::move(s);return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 20;}
}
RS_API int rs_g3_step(void* handle,double* summary,size_t ns,double* timings,size_t nt){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!summary||!timings||ns!=32||nt!=12){last_error="invalid G3 step arguments";return 1;}
    auto start=Clock::now();double lin_time=0.;
    try{
        if(!p->linearization->gpu_ready){double v[4],t[8];int status=rs_g2_linearize(handle,1,v,4,t,8);if(status)return status;lin_time=t[0];}
        auto trial=p->bind(p->solver->trial_input.data,p->device_codes);
        bool accepted=p->solver->run(p->device,trial,*p->linearization,p->input.size(),p->stream,summary,timings);
        if(accepted)p->host_stale=true;
        timings[11]=lin_time;timings[0]=millis(start);return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 22;}
}
RS_API int rs_g4_request_stop(void* handle){
    auto* p=static_cast<Problem*>(handle);
    if(!p)return 1;
    p->stop_requested.store(true);return 0;
}
RS_API int rs_g4_solve(void* handle,const double* tol,size_t n,rs_lm_progress callback,void* context,double* out,size_t no){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!tol||n!=3||!out||no!=17||!finite_array(tol,n)||tol[0]<=0||tol[1]<=0||tol[2]<=0){
        last_error="invalid G4 controller arguments";return 1;
    }
    std::fill(out,out+17,0.);auto started=Clock::now();
    auto& solver=*p->solver;auto& lin=*p->linearization;
    const int initial_trials=solver.trials,initial_accepted=solver.accepted;
    bool accepted_small=false;double latest_step=0.,relative=0.,copies=0.,total_lin=0.,total_steps=0.;
    int linearizations=0;
    try{
        for(;;){
            double lin_ms=0.;
            if(!lin.gpu_ready){
                double v[4],t[8];int status=rs_g2_linearize(handle,1,v,4,t,8);if(status)return status;
                lin_ms=t[0];total_lin+=lin_ms;copies+=32;++linearizations;
            }
            lm_stationarity<<<1,1,0,p->stream>>>(p->device,lin.device,solver.view,lin.d_g.data,lin.d_shared.data,lin.d_stats.data);
            cuda_check(cudaGetLastError());double state[3];
            cuda_check(cudaMemcpyAsync(state,solver.scalars.data,sizeof(state),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaStreamSynchronize(p->stream));copies+=sizeof(state);
            out[4]=state[0];out[5]=state[1];
            if(state[2])throw std::runtime_error("nonfinite original objective/projected gradient");
            if(p->stop_requested.exchange(false)){out[0]=3;break;}
            if(state[1]<=tol[0]){out[0]=0;break;}
            if(accepted_small){out[0]=1;break;}
            if(!finite_scalar(solver.lambda)||!finite_scalar(solver.nu))
                throw std::runtime_error("LM damping overflow without convergence");
            double metrics[32],timings[12];
            int status=rs_g3_step(handle,metrics,32,timings,12);if(status)return status;
            timings[11]=lin_ms;timings[0]+=lin_ms;total_steps+=timings[0];
            copies+=metrics[23];latest_step=metrics[31];
            relative=fabs(metrics[12])/std::max(1.,fabs(metrics[1]));
            accepted_small=metrics[15]&&latest_step<=tol[1]&&relative<=tol[2];
            if(callback)callback(metrics,timings,state[1],solver.numerical_rejections,context);
            // Rejected, unresolvable objective differences are reported explicitly,
            // never labelled as a successful small-step convergence.
            if(p->stop_requested.load())continue;
            if(!metrics[15]&&latest_step<=tol[1]&&metrics[8]==0&&
               fabs(metrics[11])<=64*std::numeric_limits<double>::epsilon()*std::max(1.,fabs(metrics[1]))){
                out[0]=2;break;
            }
        }
        out[1]=solver.trials-initial_trials;out[2]=solver.accepted-initial_accepted;out[3]=out[1]-out[2];
        out[6]=latest_step;out[7]=relative;out[8]=solver.lambda;out[9]=solver.nu;out[10]=millis(started);
        out[11]=linearizations;out[12]=solver.analyses;out[13]=copies;out[14]=total_lin;out[15]=total_steps;out[16]=solver.numerical_rejections;
        return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 23;}
}
RS_API int rs_g3_export_state(void* handle,int trial,double* controls,size_t nc,double* stations,size_t ns){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!controls||!stations||(trial!=0&&trial!=1)||nc!=3*p->host.nc||ns!=p->host.n||(trial&&!p->solver->trial_available)){last_error="G3 state snapshot unavailable/shape";return 1;}
    try{
        auto v=trial?p->bind(p->solver->trial_input.data,p->device_codes):p->device;
        cuda_check(cudaMemcpyAsync(controls,v.controls,nc*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaMemcpyAsync(stations,v.stations,ns*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g3_export_system(void* handle,double* values,size_t nv,int* blocked,size_t nb){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->solver->trial_available||!values){last_error="G3 system snapshot unavailable";return 1;}
    auto& s=*p->solver;const int q=p->linearization->host.q;size_t nl=p->linearization->scales.size(),step=q+nl;
    if(nv!=size_t(q)*q+q+2*step||nb!=nl||(nl&&!blocked)){last_error="G3 system snapshot shape";return 1;}
    try{
        size_t off=0;auto copy=[&](const double* d,size_t n){cuda_check(cudaMemcpyAsync(values+off,d,n*sizeof(double),cudaMemcpyDeviceToHost,p->stream));off+=n;};
        copy(s.reduced.data,q*q);copy(s.rhs.data,q);copy(s.step.data,step);copy(s.applied.data,step);
        if(nl)cuda_check(cudaMemcpyAsync(blocked,s.blocked.data,nl*sizeof(int),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
#endif

}
