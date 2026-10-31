#include "rinkspace/core_api.h"
#include "rinkspace/entities.hpp"
#include "rinkspace/camera_state.hpp"
#include "rinkspace/motion_derivatives.hpp"
#include "rinkspace/camera_chart.hpp"
#include "rinkspace/camera_envelope.hpp"
#include "rinkspace/box_stationarity.hpp"
#include "rinkspace/coupled_dual.hpp"
#include "rinkspace/scene_derivatives.hpp"
#include <tuple>
#include <cuda_runtime.h>
#ifdef RINKSPACE_HAVE_CUDSS
#include <cusolverDn.h>
#include <cudss.h>
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
    size_t nl,nc,n;size_t nr=38;unsigned options;double first,dt;int height;
    const int* profile_ids=nullptr;const double* profiles=nullptr;bool profile_enabled=true;
    const double *scene_values=nullptr,*scene_data=nullptr;const ScenePath* scene_paths=nullptr;size_t nscene=0;
    double projection_time_half=-1.;
    size_t nrig=1;const int* camera_ids=nullptr;
};
// Select only the physical camera. Geometry, source row and spline clock stay shared.
RS_HD inline View camera_view(View v,size_t camera){v.rig+=camera*v.nr;v.nrig=1;v.camera_ids=nullptr;return v;}
RS_HD inline size_t row_camera(const View& v,size_t row){return v.camera_ids?size_t(v.camera_ids[row]):0;}
RS_HD inline View row_camera_view(View v,size_t row){return camera_view(v,row_camera(v,row));}
RS_HD bool valid_installation(const View& v,int side){
    if(!side)return true;const auto r=motion_rig(v.rig,v.nr==43?v.rig+38:nullptr);
    if(!(r.pivot.z>0)||dot(r.lever,r.lever)<=1e-24||!finite(r.pivot)||!finite(r.lever))return false;
    const double* b=v.board;
    double u=std::max(0.,(std::abs(r.pivot.x)-(b[0]-b[2]))/b[2]);
    double w=std::max(0.,(std::abs(r.pivot.y)-(b[1]-b[3]))/b[3]);
    if(std::pow(u,b[4])+std::pow(w,b[4])<1.)return false;
    return (side!=2||r.pivot.y<=-b[1])&&(side!=3||r.pivot.y>=b[1])&&
           (side!=4||r.pivot.x<0)&&(side!=5||r.pivot.x>0);
}
RS_HD Status geometry_at(const View& v,size_t i,double station,Vec3& xyz,Vec3& tangent){
    int code=v.codes[i],kind=-1;double parameters[8];
    if(v.scene_paths){
        const auto& path=v.scene_paths[code];
        if(path.type==1){
            auto st=shared_entity_definition(v.board,6,v.layout,v.nl,path.code,kind,parameters,(v.nl>=35&&(path.code==410||path.code==411))?8:6,v.options,false);
            if(st!=Status::ok)return st;return sample_entity_point(v.course,v.board[5],kind,parameters,nullptr,0,station,xyz,tangent);
        }
        if(path.type==2){
            Vec2Value<LocalJet> chart{};auto st=scene_print_chart(path,v.scene_data,v.scene_values,nullptr,0,0,{station,1.},chart);
            if(st!=Status::ok)return st;Vec3 along{};st=v.course->point(chart.x.a,chart.y.a,xyz,along);
            tangent={along.x*chart.x.v,along.y*chart.x.v,chart.y.v};return st;
        }
    }
    if(v.paths){
        if(code<0||static_cast<size_t>(code)>=v.npaths)return static_cast<Status>(11);
        const int* path=v.paths+7*code;kind=path[0];
        const double* p=v.path_data+path[1];const double* vertices=v.path_data+path[3];
        if(kind==8)return sample_knot_board(*v.course,v.path_data+path[5],vertices,path[6],station,xyz,tangent);
        return sample_entity_point(v.course,v.board[5],kind,p,vertices,path[4],station,xyz,tangent);
    }
    if(!((code>=100&&code<132)||(code>=200&&code<204)||(code>=300&&code<312)||(code>=400&&code<=415)))
        return static_cast<Status>(11);
    const bool ellipse=v.nl>=35&&(code==410||code==411);
    auto st=shared_entity_definition(v.board,6,v.layout,v.nl,code,kind,parameters,ellipse?8:6,v.options,false);
    if(st!=Status::ok)return st;
    return sample_entity_point(nullptr,v.board[5],kind,parameters,nullptr,0,station,xyz,tangent);
}
RS_HD Status scene_geometry_direction(const View& v,size_t row,const double* directions,int q,int column,
    const BoardCourseValue<LocalJet>* courses,Vec3& derivative,double station){
    derivative={0,0,0};int code=v.codes[row];const auto& path=v.scene_paths[code];
    if(path.type==1){Vec3 xyz{};return shared_scene_point(v.board,v.layout,v.nl,v.options,path.code,station,directions,q,column,xyz,derivative);}
    Vec2Value<LocalJet> chart{};Vec3 xyz{},along{};
    int kind=v.paths[7*code];const double* p=v.path_data+v.paths[7*code+1];
    if(path.type==2){auto st=scene_print_chart(path,v.scene_data,v.scene_values,directions,q,column,LocalJet(station),chart);if(st!=Status::ok)return st;
        st=v.course->point(chart.x.a,chart.y.a,xyz,along);if(st!=Status::ok)return st;
        derivative={along.x*chart.x.v,along.y*chart.x.v,chart.y.v};
    }else if(kind!=3&&kind!=7)return Status::ok;
    for(int k=0;k<6;++k){double coefficient=directions[size_t(k)*q+column];if(coefficient==0.)continue;
        Vec3Value<LocalJet> x{},tangent{};Status st;
        if(path.type==2)st=courses[k].point(LocalJet(chart.x.a),LocalJet(chart.y.a),x,tangent);
        else{
            LocalJet height(v.board[5],k==5?1.:0.),station_jet(station);
            if(kind==7)st=courses[k].point(station_jet,p[0]*height,x,tangent);
            else{const auto& breaks=courses[k].breaks();int segment=int(p[0]);LocalJet position;
                if(segment==-2){double u=station*8;int j=std::min(int(std::floor(u)),7);position=breaks[j]+(u-j)*(breaks[j+1]-breaks[j]);}
                else if(segment>=0)position=breaks[segment]+station_jet*(breaks[segment+1]-breaks[segment]);
                else position=station_jet*courses[k].perimeter();
                st=courses[k].point(position,p[1]*height,x,tangent);}
        }
        if(st!=Status::ok)return st;derivative=add(derivative,mul(coefficient,Vec3{x.x.v,x.y.v,x.z.v}));
    }
    return finite(derivative)?Status::ok:Status::nonfinite;
}
// Neighbouring intrinsic piecewise-linear chart knots. These are geometry
// breakpoints, not extra observations or permanent station restrictions.
RS_HD bool station_piece(const View& v,size_t row,double station,double& lo,double& hi,bool& corner){
    lo=0.;hi=1.;corner=false;bool piecewise=false;
    const double eps=64*std::numeric_limits<double>::epsilon()*(1+fabs(station));
    auto knot=[&](double k){if(k<=0||k>=1)return;if(fabs(k-station)<=eps)corner=true;else if(k<station)lo=std::max(lo,k);else hi=std::min(hi,k);};
    int code=v.codes[row];
    if(v.scene_paths&&v.scene_paths[code].type==2){auto path=v.scene_paths[code];
        if(path.family==3)return false;piecewise=true;
        int count=path.family==2?5:path.count;if(count<2)return false;
        int j=int(std::floor(station*(count-1)));for(int k=std::max(1,j-1);k<=std::min(count-2,j+2);++k)knot(double(k)/(count-1));
        if(path.family==1){const double* seed=v.scene_data+path.vertices;const double* b=v.scene_data+path.bounds;
            for(int e=std::max(0,j-1);e<=std::min(count-2,j+1);++e)for(int axis=0;axis<2;++axis){double a=seed[2*e+axis],d=seed[2*e+2+axis]-a;
                if(d!=0.)for(int k=1;k<4;++k){double t=(b[axis]+k*.25*(b[axis+2]-b[axis])-a)/d;if(t>0&&t<1)knot((e+t)/(count-1));}}
        }
    }else if(v.paths){const int* path=v.paths+7*code;int kind=path[0];
        if(kind==8){piecewise=true;const double* knots=v.path_data+path[5];int count=path[6],a=0,b=count;
            while(a<b){int m=(a+b)/2;if(knots[m]<station)a=m+1;else b=m;}
            for(int k=std::max(1,a-2);k<=std::min(count-2,a+2);++k)knot(knots[k]);
        }else if(kind==2||kind==4||kind==5){int count=path[4]/(kind==4?3:2);if(count<3)return false;piecewise=true;int j=int(std::floor(station*(count-1)));
            for(int k=std::max(1,j-1);k<=std::min(count-2,j+2);++k)knot(double(k)/(count-1));}
    }
    return piecewise;
}
RS_HD Status geometry_row(const View& v,size_t i,Vec3& xyz,Vec3& tangent){return geometry_at(v,i,v.stations[i],xyz,tangent);}
// A source query is an immutable rational cubic in the original source image.
// Its alpha is an internal intersection coordinate, never a fitted unknown.
RS_HD Status profile_value(const double* p,double alpha,Vec2& uv,Vec2& tangent){
    double v[3],d[3];for(int k=0;k<3;++k){const double* c=p+4*k;v[k]=((c[3]*alpha+c[2])*alpha+c[1])*alpha+c[0];d[k]=(3*c[3]*alpha+2*c[2])*alpha+c[1];}
    if(!finite_array(v,3)||!finite_array(d,3)||std::abs(v[2])<1e-14)return Status::degenerate;
    uv={v[0]/v[2],v[1]/v[2]};tangent={(d[0]-uv.x*d[2])/v[2],(d[1]-uv.y*d[2])/v[2]};
    return finite_scalar(uv.x)&&finite_scalar(uv.y)&&finite_scalar(tangent.x)&&finite_scalar(tangent.y)?Status::ok:Status::nonfinite;
}
RS_HD Status profile_projection(const View& source,size_t row,double station,Vec3& xyz,Vec3& tangent,MotionProjection& projection,Vec2& image_tangent){
    const auto v=row_camera_view(source,row);
    auto st=geometry_at(v,row,station,xyz,tangent);if(st!=Status::ok)return st;
    const auto rig=motion_rig(v.rig,v.nr==43?v.rig+38:nullptr);const MotionSpline spline{v.controls,v.nc,v.first,v.dt};
    st=project_motion(rig,spline,v.height,xyz,v.times[row],projection,v.projection_time_half);if(st!=Status::ok)return st;
    return motion_station_partial(rig,spline,v.height,xyz,tangent,v.times[row],projection,image_tangent);
}
RS_HD Status resolve_profile(const View& v,size_t row,double& station,double& alpha,Vec3& xyz,Vec3& tangent,MotionProjection& projection,bool allow_alpha_extension=false){
    const double* p=v.profiles+17*v.profile_ids[row];station=v.stations[row];alpha=p[16];
    if(station<p[14]||station>p[15])return Status::outside_support;
    Vec2 qt{},q{},it{};auto st=Status::ok;
    for(int iteration=0;iteration<40;++iteration){
        st=profile_projection(v,row,station,xyz,tangent,projection,it);if(st!=Status::ok)return st;
        st=profile_value(p,alpha,q,qt);if(st!=Status::ok)return st;
        double fu=projection.uv.x-q.x,fv=projection.uv.y-q.y,loss=fu*fu+fv*fv;
        double determinant=qt.x*it.y-it.x*qt.y;
        if(std::abs(determinant)<=1e-10*std::sqrt((qt.x*qt.x+qt.y*qt.y)*(it.x*it.x+it.y*it.y)))return Status::degenerate;
        if(loss<1e-14)return Status::ok;
        double ds=(qt.y*fu-qt.x*fv)/determinant,da=(it.y*fu-it.x*fv)/determinant,step=1.;
        if(ds>0)step=std::min(step,(p[15]-station)/ds);else if(ds<0)step=std::min(step,(p[14]-station)/ds);
        if(!allow_alpha_extension){if(da>0)step=std::min(step,(p[13]-alpha)/da);else if(da<0)step=std::min(step,(p[12]-alpha)/da);}
        bool accepted=false;
        for(int backtrack=0;backtrack<24&&step>0.;++backtrack,step*=.5){
            double ns=station+step*ds,na=alpha+step*da;Vec3 x{},t{};Vec2 iq{},qq{},tt{};MotionProjection candidate{};
            auto candidate_status=profile_projection(v,row,ns,x,t,candidate,iq);
            if(candidate_status==Status::ok&&profile_value(p,na,qq,tt)==Status::ok){
                double u=candidate.uv.x-qq.x,w=candidate.uv.y-qq.y;
                if(u*u+w*w<loss){station=ns;alpha=na;accepted=true;break;}
            }
        }
        if(!accepted)return Status::no_intersection;
    }
    return Status::no_intersection;
}
RS_HD void evaluate_row(const View& source,size_t i,double* out,int* status,double* resolved=nullptr){
    const auto v=row_camera_view(source,i);
    const double nan=std::numeric_limits<double>::quiet_NaN();for(int j=0;j<11;++j)out[j]=nan;
    Vec3 xyz{},tangent{};MotionProjection p{};double station=v.stations[i],alpha=nan;
    auto st=Status::ok;
    if(!finite_scalar(v.observed[2*i])||!finite_scalar(v.observed[2*i+1]))st=Status::invalid_argument;
    else if(v.profile_enabled&&v.profile_ids&&v.profile_ids[i]>=0)st=resolve_profile(v,i,station,alpha,xyz,tangent,p);
    else{st=geometry_row(v,i,xyz,tangent);if(st==Status::ok)st=project_motion(motion_rig(v.rig,v.nr==43?v.rig+38:nullptr),{v.controls,v.nc,v.first,v.dt},v.height,xyz,v.times[i],p,v.projection_time_half);}
    if(st==Status::ok){store3(out,xyz);out[3]=p.uv.x;out[4]=p.uv.y;out[5]=p.depth;out[6]=p.row;out[7]=p.equality;out[8]=p.contraction;out[9]=p.uv.x-v.observed[2*i];out[10]=p.uv.y-v.observed[2*i+1];}
    if(resolved){resolved[0]=station;resolved[1]=alpha;}*status=static_cast<int>(st);
}
__global__ void profile_roots_kernel(View view,double* roots,int* status){
    size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=view.n)return;
    if(!view.profile_ids||view.profile_ids[row]<0){roots[2*row]=view.stations[row];roots[2*row+1]=std::numeric_limits<double>::quiet_NaN();status[row]=0;return;}
    view.profile_enabled=true;double output[11];evaluate_row(view,row,output,status+row,roots+2*row);
}
__global__ void evaluate_kernel(View view,double* output,int* statuses){
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<view.n)evaluate_row(view,i,output+11*i,statuses+i);
}
__global__ void evaluate_errors_kernel(View view,double* errors){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=view.n)return;
    double values[11];int status;evaluate_row(view,i,values,&status);
    errors[i]=status?INFINITY:hypot(values[9],values[10]);
}
__global__ void evaluate_selected_kernel(View view,const int* rows,size_t n,double* values,int* statuses){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)evaluate_row(view,size_t(rows[i]),values+11*i,statuses+i);
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
    void sync_host(){if(host_stale){cuda_check(cudaMemcpyAsync(input.data(),device_input,input.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));if(!scene_paths.empty()){auto b=input.data();host_course=std::make_unique<BoardCourse>(BoardShape{b[0],b[1],b[2],b[3],b[4],b[5]});host=bind(input.data(),codes.data());}host_stale=false;}}
    std::unique_ptr<Linearization> linearization;
    std::vector<int> camera_ids,mount_sides;DeviceBuffer<int> d_camera_ids,d_mount_sides;
    std::vector<CameraEnvelope> camera_envelopes;DeviceBuffer<CameraEnvelope> d_camera_envelopes;
    std::vector<std::vector<double>> camera_windows;std::vector<std::unique_ptr<DeviceBuffer<double>>> d_camera_windows;
    std::vector<double> rig_directions,rig_lower,rig_upper;int rig_q=0;double max_readout=0.;int mount_side=0;CameraEnvelope envelope{};
    std::vector<double> envelope_windows;std::unique_ptr<DeviceBuffer<double>> d_envelope_windows;
    std::vector<double> input;std::vector<int> codes;
    std::vector<int> profile_ids;std::vector<double> profiles;DeviceBuffer<int> d_profile_ids;DeviceBuffer<double> d_profiles;
    std::vector<int> path_meta;std::vector<double> geometry_data;
    std::vector<ScenePath> scene_paths;std::vector<double> scene_data,scene_directions,scene_lower,scene_upper;int scene_q=0;std::vector<int> scene_chart;
    DeviceBuffer<ScenePath> d_scene_paths;DeviceBuffer<double> d_scene_data;
    std::unique_ptr<BoardCourse> host_course;
    DeviceBuffer<int> d_path_meta;DeviceBuffer<double> d_geometry_data;DeviceBuffer<BoardCourse> d_course;
    double *device_input=nullptr,*device_output=nullptr;int *device_codes=nullptr,*device_status=nullptr;
    cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,middle=nullptr,end=nullptr;
    View host{},device{};
    ~Problem(){
        if(stream)cudaStreamSynchronize(stream);
#ifdef RINKSPACE_HAVE_CUDSS
        solver.reset();
#endif
        linearization.reset();
        cudaFree(device_status);cudaFree(device_output);cudaFree(device_codes);cudaFree(device_input);
        if(start)cudaEventDestroy(start);if(middle)cudaEventDestroy(middle);if(end)cudaEventDestroy(end);
        if(stream)cudaStreamDestroy(stream);
    }
    bool checked(cudaError_t status){if(status==cudaSuccess)return true;last_error=cudaGetErrorString(status);return false;}
    View bind(const double* p,const int* c)const{
        View v=host;v.codes=c;v.board=p;v.layout=p+6;v.rig=v.layout+v.nl;
        v.controls=v.rig+v.nr*v.nrig;v.stations=v.controls+3*v.nc;v.times=v.stations+v.n;v.observed=v.times+v.n;
        if(!path_meta.empty()){
            const bool gpu=c==device_codes;
            v.paths=gpu?d_path_meta.data:path_meta.data();
            v.path_data=gpu?d_geometry_data.data:geometry_data.data();
            v.course=gpu?d_course.data:host_course.get();v.npaths=path_meta.size()/7;
        }
        if(!profile_ids.empty()){bool gpu=c==device_codes;v.profile_ids=gpu?d_profile_ids.data:profile_ids.data();v.profiles=gpu?d_profiles.data:profiles.data();}
        if(!scene_paths.empty()){bool gpu=c==device_codes;v.scene_paths=gpu?d_scene_paths.data:scene_paths.data();v.scene_data=gpu?d_scene_data.data:scene_data.data();v.scene_values=v.observed+2*v.n;}
        if(!camera_ids.empty())v.camera_ids=c==device_codes?d_camera_ids.data:camera_ids.data();
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
    if(!handle||!metrics||nm!=10||nb!=6||!shared_layout_size_valid(nl)||(nr!=38&&nr!=43)||height<=1||!count||
       !codes||!stations||!times||!observed||!finite_array(board,nb)||!finite_array(layout,nl)||
       !finite_array(rig,nr)||!motion_rig(rig,nr==43?rig+38:nullptr).valid()||
       !MotionSpline{controls,nc,first,dt}.valid()||!shared_options_valid(nl,options)){
        last_error="invalid G1 immutable problem";return 1;
    }
    if(options&64u){int kind;double parameters[6];
        if(shared_entity_definition(board,6,layout,nl,100,kind,parameters,6,options)!=Status::ok){last_error="invalid bounded ice geometry";return 1;}}
    std::unique_ptr<Problem> p;
    try {
        p.reset(new Problem);
        p->host.nl=nl;p->host.nr=nr;p->host.nc=nc;p->host.n=count;p->host.options=options;
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
RS_API int rs_gpu_set_camera_groups(void* handle,const double* rigs,size_t nr,const int* ids,size_t count,double* bytes){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||!p->path_meta.empty()||!p->scene_paths.empty()||p->host.nrig!=1||!p->camera_ids.empty()||
       !rigs||nr%p->host.nr||nr/p->host.nr<2||!ids||count!=p->host.n||!bytes||!finite_array(rigs,nr)){
        last_error="invalid camera groups or camera groups attached too late";return 1;}
    const size_t cameras=nr/p->host.nr;
    for(size_t j=0;j<cameras;++j)if(!motion_rig(rigs+j*p->host.nr,p->host.nr==43?rigs+j*p->host.nr+38:nullptr).valid()){
        last_error="invalid physical camera in camera groups";return 1;}
    for(size_t i=0;i<count;++i)if(ids[i]<0||size_t(ids[i])>=cameras){last_error="source row has no declared physical camera";return 1;}
    try{
        std::vector<double> input;const size_t offset=6+p->host.nl;
        input.insert(input.end(),p->input.begin(),p->input.begin()+offset);input.insert(input.end(),rigs,rigs+nr);
        input.insert(input.end(),p->input.begin()+offset+p->host.nr,p->input.end());
        std::vector<int> cameras_by_row(ids,ids+count);DeviceBuffer<int> device_ids;device_ids.upload(cameras_by_row,p->stream);
        double* replacement=nullptr;cuda_check(cudaMalloc(&replacement,input.size()*sizeof(double)));
        auto status=cudaMemcpyAsync(replacement,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice,p->stream);
        if(status!=cudaSuccess){cudaFree(replacement);cuda_check(status);}
        cuda_check(cudaStreamSynchronize(p->stream));cudaFree(p->device_input);p->device_input=replacement;
        p->input.swap(input);p->camera_ids.swap(cameras_by_row);
        p->d_camera_ids.data=device_ids.data;p->d_camera_ids.count=device_ids.count;device_ids.data=nullptr;device_ids.count=0;
        p->host.nrig=cameras;p->host=p->bind(p->input.data(),p->codes.data());p->device=p->bind(p->device_input,p->device_codes);
        *bytes=double((nr-p->host.nr)*sizeof(double)+count*sizeof(int));return 0;
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
RS_API int rs_g1_evaluate_errors(void* handle,double* errors,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!errors||n!=p->host.n){last_error="invalid compact error output";return 1;}
    try{DeviceBuffer<double> output;output.allocate(n);
        evaluate_errors_kernel<<<unsigned((n+127)/128),128,0,p->stream>>>(p->device,output.data);
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(errors,output.data,n*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g1_evaluate_rows(void* handle,const int* rows,size_t n,double* values,size_t nv,int* statuses,size_t ns){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||nv!=11*n||ns!=n||(n&&(!rows||!values||!statuses))){last_error="invalid selected row output";return 1;}
    for(size_t i=0;i<n;++i)if(rows[i]<0||size_t(rows[i])>=p->host.n){last_error="selected row outside problem";return 1;}
    if(!n)return 0;
    try{DeviceBuffer<int> ids,status;DeviceBuffer<double> output;ids.allocate(n);status.allocate(n);output.allocate(nv);
        cuda_check(cudaMemcpyAsync(ids.data,rows,n*sizeof(int),cudaMemcpyHostToDevice,p->stream));
        evaluate_selected_kernel<<<unsigned((n+127)/128),128,0,p->stream>>>(p->device,ids.data,n,output.data,status.data);
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(values,output.data,nv*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaMemcpyAsync(statuses,status.data,n*sizeof(int),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_gpu_attach_profiles(void* handle,const int* ids,size_t ni,const double* data,size_t nd){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||!p->profile_ids.empty()||ni!=p->host.n||!ids||nd==0||nd%17||!finite_array(data,nd)){last_error="invalid source profile plan";return 1;}
    try{
        std::vector<int> seen(nd/17,0);
        for(size_t i=0;i<ni;++i){int id=ids[i];if(id<-1||id>=int(nd/17))throw std::invalid_argument("profile identity outside immutable source table");if(id>=0)++seen[id];}
        for(size_t j=0;j<nd/17;++j){const double* r=data+17*j;
            if(!seen[j]||r[12]>=r[13]||r[14]>=r[15]||r[16]<r[12]||r[16]>r[13])throw std::invalid_argument("invalid profile finite interval or unused query");}
        p->profile_ids.assign(ids,ids+ni);p->profiles.assign(data,data+nd);p->d_profile_ids.upload(p->profile_ids,p->stream);p->d_profiles.upload(p->profiles,p->stream);cuda_check(cudaStreamSynchronize(p->stream));
        p->host=p->bind(p->input.data(),p->codes.data());p->device=p->bind(p->device_input,p->device_codes);return 0;
    }catch(const std::exception& e){last_error=e.what();return 24;}
}
RS_API int rs_gpu_set_profile_mode(void* handle,int enabled){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||(enabled!=0&&enabled!=1)){last_error="profile mode is immutable after factor-plan compilation";return 1;}
    p->host.profile_enabled=p->device.profile_enabled=enabled!=0;return 0;
}
RS_API int rs_gpu_profile_roots(void* handle,int backend,double* roots,size_t nr,int* status,size_t ns){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||nr!=2*p->host.n||ns!=p->host.n||!roots||!status||(backend!=0&&backend!=1)){last_error="invalid profile root output";return 1;}
    try{
        if(backend==0){p->sync_host();for(size_t row=0;row<p->host.n;++row){
            if(p->host.profile_ids&&p->host.profile_ids[row]>=0){View exact=p->host;exact.profile_enabled=true;double out[11];evaluate_row(exact,row,out,status+row,roots+2*row);}
            else{roots[2*row]=p->host.stations[row];roots[2*row+1]=std::numeric_limits<double>::quiet_NaN();status[row]=0;}}}
        else{DeviceBuffer<double> values;DeviceBuffer<int> statuses;values.allocate(nr);statuses.allocate(ns);
            profile_roots_kernel<<<unsigned((ns+127)/128),128,0,p->stream>>>(p->device,values.data,statuses.data);
            cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(roots,values.data,nr*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaMemcpyAsync(status,statuses.data,ns*sizeof(int),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));}
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 24;}
}
RS_API int rs_gpu_attach_scene(void* handle,const double* values,size_t nv,const int* paths,size_t np,
    const double* data,size_t nd,int q,const double* directions,size_t nj,const double* lower,const double* upper){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||p->scene_q||p->path_meta.empty()||nv<6+p->host.nl||np!=12*p->host.npaths||!paths||
       !finite_array(values,nv)||!finite_array(data,nd)||q<=0||nj!=nv*size_t(q)||!finite_array(directions,nj)||!lower||!upper){last_error="invalid common scene plan";return 1;}
    try{
        for(size_t i=0;i<6+p->host.nl;++i)if(values[i]!=p->input[i])throw std::invalid_argument("scene attachment cannot replace selected native board/layout");
        for(size_t i=0;i<nv;++i)if(std::isnan(lower[i])||std::isnan(upper[i])||values[i]<lower[i]||values[i]>upper[i])throw std::invalid_argument("scene state violates declared bounds");
        std::vector<ScenePath> records(np/12);std::memcpy(records.data(),paths,np*sizeof(int));
        for(auto path:records){
            if(path.type<0||path.type>2)throw std::invalid_argument("unsupported scene path relation");
            if(path.type==2){
                if(path.code<0||path.code>1)throw std::invalid_argument("unsupported print material domain");
                if(path.size<0||path.placement<0||size_t(path.size)+2>nv||size_t(path.placement)+3>nv||path.family<0||path.family>3||path.vertices<0||path.count<0||size_t(path.vertices)+2*path.count>nd)throw std::invalid_argument("print state references outside common scene");
                if(path.family&&(path.shape<0||size_t(path.shape)+path.shape_count>nv))throw std::invalid_argument("print shape reference outside common scene");
                if((path.family==1&&(path.shape_count!=10||path.bounds<0||size_t(path.bounds)+4>nd))||(path.family==2&&path.shape_count!=3)||(path.family==3&&(path.shape_count!=4||path.side<0||path.side>3)))throw std::invalid_argument("print shape family mismatch");
            }
        }
        p->scene_paths=std::move(records);p->scene_data.assign(data,data+nd);p->scene_directions.assign(directions,directions+nj);p->scene_lower.assign(lower,lower+nv);p->scene_upper.assign(upper,upper+nv);p->scene_q=q;p->host.nscene=nv;
        if(p->host.options&64u)for(size_t i=6+ice_nominal_size;i<6+ice_tolerance_layout_size;++i){
            p->scene_lower[i]=std::max(p->scene_lower[i],-.01);p->scene_upper[i]=std::min(p->scene_upper[i],.01);}
        p->input.insert(p->input.end(),values,values+nv);p->d_scene_paths.upload(p->scene_paths,p->stream);p->d_scene_data.upload(p->scene_data,p->stream);
        cuda_check(cudaFree(p->device_input));p->device_input=nullptr;cuda_check(cudaMalloc(&p->device_input,p->input.size()*sizeof(double)));
        cuda_check(cudaMemcpyAsync(p->device_input,p->input.data(),p->input.size()*sizeof(double),cudaMemcpyHostToDevice,p->stream));cuda_check(cudaStreamSynchronize(p->stream));
        p->host=p->bind(p->input.data(),p->codes.data());p->device=p->bind(p->device_input,p->device_codes);return 0;
    }catch(const std::exception& e){last_error=e.what();return 24;}
}
RS_API int rs_gpu_set_scene_chart(void* handle,const int* chart,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||!p->scene_q||n!=p->host.nscene||!chart){last_error="invalid scene chart";return 1;}
    for(size_t i=0;i<n;++i){int type=chart[i];double v=p->host.scene_values[i];
        if(type<0||type>5||(type==1&&v<=0)||(type==2&&v<=1)){last_error="scene value outside coordinate chart";return 1;}
        if(type>=3){if(i<size_t(type-3)){last_error="ordered chart start";return 1;}size_t start=i-(type-3);
            if(start+2>=n||chart[start]!=3||chart[start+1]!=4||chart[start+2]!=5||!(0<p->host.scene_values[start]&&p->host.scene_values[start]<p->host.scene_values[start+1]&&p->host.scene_values[start+1]<p->host.scene_values[start+2]&&p->host.scene_values[start+2]<1)){last_error="ordered radius chart mismatch";return 1;}}
    }
    p->scene_chart.assign(chart,chart+n);return 0;
}
RS_API int rs_gpu_export_scene(void* handle,int trial,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();if(!p||!values||n!=p->host.nscene||!n||(trial!=0&&trial!=1)){last_error="scene export shape";return 1;}
    try{View v=p->device;
#ifdef RINKSPACE_HAVE_CUDSS
        if(trial){if(!p->solver||!p->solver->trial_available){last_error="no trial scene";return 1;}v=p->bind(p->solver->trial_input.data,p->device_codes);}
#else
        if(trial)return 1;
#endif
        cuda_check(cudaMemcpyAsync(values,v.scene_values,n*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_gpu_set_mount_constraints_for_camera(void* handle,int camera,int side){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||camera<0||size_t(camera)>=p->host.nrig||side<1||side>5||
       !valid_installation(camera_view(p->host,size_t(camera)),side)){last_error="invalid selected physical installation";return 1;}
    if(p->mount_sides.empty())p->mount_sides.resize(p->host.nrig,0);
    p->mount_sides[camera]=side;p->mount_side=side;return 0;
}
RS_API int rs_gpu_set_mount_constraints(void* handle,int side){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||side<1||side>5||!valid_installation(p->host,side)){last_error="selected installation violates confirmed outside/positive-pivot/nonzero-lever relations";return 1;}
    for(size_t camera=1;camera<p->host.nrig;++camera)if(!valid_installation(camera_view(p->host,camera),side)){last_error="physical camera violates installation side";return 1;}
    p->mount_side=side;p->mount_sides.assign(p->host.nrig,side);return 0;
}
RS_API int rs_gpu_set_camera_envelope_for_camera(void* handle,int camera,const double* limits,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();CameraEnvelope e{};
    if(!p||p->linearization||camera<0||size_t(camera)>=p->host.nrig||!camera_envelope(limits,n,e)){last_error="invalid camera envelope or already attached linearization";return 1;}
    if(e.windowed)for(size_t i=0;i<p->host.n;++i){
        if(row_camera(p->host,i)!=size_t(camera))continue;
        const double frame=(p->host.times[i]-p->host.first)/p->host.dt;
        const double eps=128*std::numeric_limits<double>::epsilon()*(1+std::abs(frame));
        bool covered=false;for(size_t j=0;j<e.frame_window_count;++j)
            covered|=frame>=e.frame_windows[2*j]-eps&&frame<=e.frame_windows[2*j+1]+eps;
        if(!covered){last_error="camera envelope time window omits a source exposure";return 1;}}
    const auto a=camera_envelope_check(motion_rig(camera_view(p->host,size_t(camera)).rig),MotionSpline{p->host.controls,p->host.nc,p->host.first,p->host.dt},e);
    if(!finite_scalar(a.margin)||a.margin < -camera_envelope_tolerance()){
        last_error="current camera violates complete envelope: kind="+std::to_string(a.kind)+" cell="+std::to_string(a.cell)+" margin="+std::to_string(a.margin);return 1;}
    try{
        std::vector<double> windows;if(e.windowed)windows.assign(e.frame_windows,e.frame_windows+2*e.frame_window_count);
        auto device_windows=std::make_unique<DeviceBuffer<double>>();device_windows->upload(windows,p->stream);
        cuda_check(cudaStreamSynchronize(p->stream));
        if(p->camera_envelopes.empty()){p->camera_envelopes.resize(p->host.nrig);p->camera_windows.resize(p->host.nrig);p->d_camera_windows.resize(p->host.nrig);}
        p->camera_windows[camera].swap(windows);p->d_camera_windows[camera]=std::move(device_windows);
        e.frame_windows=p->camera_windows[camera].empty()?nullptr:p->camera_windows[camera].data();p->camera_envelopes[camera]=e;
        p->envelope=e;return 0;
    }catch(const std::exception& error){last_error=error.what();return 20;}
}
RS_API int rs_gpu_set_camera_envelope(void* handle,const double* limits,size_t n){
    auto* p=static_cast<Problem*>(handle);
    if(!p||p->host.nrig!=1){last_error="multi-camera envelope requires an explicit camera index";return 1;}
    return rs_gpu_set_camera_envelope_for_camera(handle,0,limits,n);
}
RS_API int rs_gpu_set_projection_time_support(void* handle,double half_seconds){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||!finite_scalar(half_seconds)||half_seconds<0.||
        2*half_seconds<std::abs(p->host.rig[37])||(p->rig_q&&2*half_seconds<p->max_readout)){
        last_error="projection time support must contain admissible readout and precede linearization";return 1;
    }
    for(size_t camera=0;camera<p->host.nrig;++camera)if(2*half_seconds<std::abs(p->host.rig[camera*p->host.nr+37])){last_error="projection support omits a camera readout";return 1;}
    p->host.projection_time_half=p->device.projection_time_half=half_seconds;return 0;
}
RS_API int rs_gpu_configure_camera(void* handle,int q,const double* directions,size_t n,
    const double* lower,const double* upper,size_t nb,double max_readout){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||p->linearization||p->rig_q||q<=0||n!=p->host.nrig*size_t(p->host.nr==43?31:27)*q||nb!=p->host.nr*p->host.nrig||!lower||!upper||
       !finite_array(directions,n)||!finite_scalar(max_readout)||max_readout<0){last_error="invalid joint camera plan";return 1;}
    for(size_t i=0;i<p->host.nr*p->host.nrig;++i)if(std::isnan(lower[i])||std::isnan(upper[i])||lower[i]>upper[i]||p->host.rig[i]<lower[i]||p->host.rig[i]>upper[i]){
        last_error="current rig outside declared physical bounds at component "+std::to_string(i);return 1;}
    for(size_t camera=0;camera<p->host.nrig;++camera)if(std::abs(p->host.rig[camera*p->host.nr+37])>max_readout){last_error="current readout exceeds declared row-time support";return 1;}
    p->rig_directions.assign(directions,directions+n);p->rig_lower.assign(lower,lower+nb);p->rig_upper.assign(upper,upper+nb);
    p->rig_q=q;p->max_readout=max_readout;
    p->host.projection_time_half=p->device.projection_time_half=std::max(p->host.projection_time_half,max_readout/2);return 0;
}
RS_API int rs_gpu_export_model(void* handle,int trial,double* board,size_t nb,double* layout,size_t nl,double* rig,size_t nr){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!board||!layout||!rig||nb!=6||nl!=p->host.nl||nr!=p->host.nr*p->host.nrig||(trial!=0&&trial!=1)){last_error="model export shape";return 1;}
    try{View v=p->device;
#ifdef RINKSPACE_HAVE_CUDSS
        if(trial){if(!p->solver||!p->solver->trial_available){last_error="no trial model";return 1;}v=p->bind(p->solver->trial_input.data,p->device_codes);}
#else
        if(trial){last_error="no trial solver";return 1;}
#endif
        cuda_check(cudaMemcpyAsync(board,v.board,nb*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaMemcpyAsync(layout,v.layout,nl*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaMemcpyAsync(rig,v.rig,nr*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
static int attach_linearization(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,const int* offsets,int ng,const int* members,size_t nmembers,
    const int* ranks,const int* weight_offsets,const double* weights,size_t nweights,double huber_delta,
    double* info,size_t ninfo,bool pixels){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!info||ninfo!=12||p->linearization){last_error="invalid or already-attached G2 problem";return 1;}
    const auto start=Clock::now();
    try{
        auto linear=std::make_unique<Linearization>();
        cuda_check(cudaEventRecord(p->start,p->stream));
        linear->prepare(p->host,q,dirs,ndirs,row_local,nrows,local_scales,nlocal,offsets,ng,members,nmembers,ranks,weight_offsets,weights,nweights,huber_delta,p->stream,p->rig_directions.empty()?nullptr:p->rig_directions.data(),p->rig_q,p->rig_lower.data(),p->rig_upper.data(),p->max_readout,p->scene_directions.empty()?nullptr:p->scene_directions.data(),p->scene_q,p->scene_lower.data(),p->scene_upper.data(),p->scene_chart.empty()?nullptr:p->scene_chart.data(),p->mount_side,pixels);
        linear->host.envelope=p->envelope;linear->device.envelope=p->envelope;
        if(!p->camera_envelopes.empty()){
            auto device_envelopes=p->camera_envelopes;
            for(size_t camera=0;camera<device_envelopes.size();++camera)
                device_envelopes[camera].frame_windows=p->d_camera_windows[camera]?p->d_camera_windows[camera]->data:nullptr;
            p->d_camera_envelopes.upload(device_envelopes,p->stream);
            linear->host.camera_envelopes=p->camera_envelopes.data();linear->device.camera_envelopes=p->d_camera_envelopes.data;
        }
        if(!p->mount_sides.empty()){p->d_mount_sides.upload(p->mount_sides,p->stream);linear->host.mount_sides=p->mount_sides.data();linear->device.mount_sides=p->d_mount_sides.data;}
        cuda_check(cudaEventRecord(p->end,p->stream));cuda_check(cudaEventSynchronize(p->end));
        float transfer;cuda_check(cudaEventElapsedTime(&transfer,p->start,p->end));
        int maximum=0,total_rank=0;for(auto c:linear->components)maximum=std::max(maximum,c.local_count);
        for(auto g:linear->groups)total_rank+=g.rank;
        const double values[]={double(q),double(nlocal),double(ng),double(linear->components.size()),double(linear->white_count),
            double(linear->h_count),double(linear->g_count),double(linear->device_bytes()),millis(start),transfer,double(maximum),double(total_rank)};
        std::copy(values,values+12,info);p->linearization=std::move(linear);return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 20;}
}
RS_API int rs_g2_attach(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,const int* offsets,int ng,const int* members,size_t nmembers,
    const int* ranks,const int* weight_offsets,const double* weights,size_t nweights,double huber_delta,
    double* info,size_t ninfo){
    return attach_linearization(handle,q,dirs,ndirs,row_local,nrows,local_scales,nlocal,offsets,ng,members,nmembers,ranks,weight_offsets,weights,nweights,huber_delta,info,ninfo,false);
}
RS_API int rs_g2_attach_pixels(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,double* info,size_t ninfo){
    try{
        auto* p=static_cast<Problem*>(handle);
        if(!p||!p->profile_ids.empty())throw std::invalid_argument("plain pixel residuals cannot inherit source query constraints");
        int n=Linearization::bounded(nrows);std::vector<int> offsets(n+1),members(n),ranks(n,2),unused(n+1,0);
        std::iota(offsets.begin(),offsets.end(),0);std::iota(members.begin(),members.end(),0);
        return attach_linearization(handle,q,dirs,ndirs,row_local,nrows,local_scales,nlocal,offsets.data(),n,members.data(),n,ranks.data(),unused.data(),nullptr,0,0.,info,ninfo,true);
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_layout64(void* handle,int64_t* groups,size_t ng,int64_t* components,size_t nc,int* locals,size_t nl){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}const auto& l=*p->linearization;
    if(!groups||!components||(nl&&!locals)||ng!=6*l.groups.size()||nc!=7*l.components.size()||nl!=l.component_locals.size()){last_error="G2 metadata output shape";return 1;}
    for(size_t i=0;i<l.groups.size();++i){auto g=l.groups[i];int64_t row[]={g.begin,g.count,g.rank,g.weight,g.component,int64_t(g.white)};std::copy_n(row,6,groups+6*i);}
    for(size_t i=0;i<l.components.size();++i){auto c=l.components[i];int64_t row[]={c.local_begin,c.local_count,c.group_begin,c.group_count,c.dim,int64_t(c.h),int64_t(c.g)};std::copy_n(row,7,components+7*i);}
    if(nl)std::copy(l.component_locals.begin(),l.component_locals.end(),locals);return 0;
}
RS_API int rs_g2_layout(void* handle,int* groups,size_t ng,int* components,size_t nc,int* locals,size_t nl){
    if(!groups||!components){last_error="G2 metadata output shape";return 1;}
    try{std::vector<int64_t> gg(ng),cc(nc);int status=rs_g2_layout64(handle,gg.data(),ng,cc.data(),nc,locals,nl);if(status)return status;
        for(size_t i=0;i<ng;++i)groups[i]=Linearization::bounded(size_t(gg[i]));for(size_t i=0;i<nc;++i)components[i]=Linearization::bounded(size_t(cc[i]));return 0;
    }catch(const std::exception& e){last_error=e.what();return 1;}
}
RS_API int rs_g2_sparse_info(void* handle,int64_t* info,size_t n){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization||!info||n!=6){last_error="invalid sparse info";return 1;}auto& l=*p->linearization;
    int64_t values[]={int64_t(l.host.nnz),int64_t(l.columns.size()),int64_t(l.scatter.size()),int64_t(l.jac_count),l.max_row_q,l.max_component_q};std::copy_n(values,6,info);return 0;
}
RS_API int rs_g2_sparse_layout(void* handle,int64_t* component_columns,size_t nc,int* columns,size_t ncol,int* offsets,size_t no,int* indices,size_t ni){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing sparse plan";return 1;}auto& l=*p->linearization;
    if(!component_columns||(ncol&&!columns)||!offsets||!indices||nc!=2*l.components.size()||ncol!=l.columns.size()||no!=l.csr_offsets.size()||ni!=l.csr_columns.size()){last_error="sparse metadata shape";return 1;}
    for(size_t i=0;i<l.components.size();++i){component_columns[2*i]=l.components[i].q_begin;component_columns[2*i+1]=l.components[i].q_count;}
    std::copy(l.columns.begin(),l.columns.end(),columns);std::copy(l.csr_offsets.begin(),l.csr_offsets.end(),offsets);std::copy(l.csr_columns.begin(),l.csr_columns.end(),indices);return 0;
}

RS_API int rs_g2_linearize(void* handle,int backend,double* summary,size_t ns,double* timings,size_t nt){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->linearization||!summary||!timings||ns!=4||nt!=8||(backend!=0&&backend!=1)){
        last_error="invalid G2 linearization arguments";return 1;}
    auto& l=*p->linearization;const auto start=Clock::now();std::fill(timings,timings+8,0.);
    try{
        if(backend==0){p->sync_host();l.cpu(p->host,summary);}
        else{
            l.gpu_ready=false;++l.generation;
            cuda_check(cudaEventRecord(l.event[0],p->stream));
            if(l.device.scene_directions){scene_direction_kernel<<<(l.host.q+127)/128,128,0,p->stream>>>(p->device,l.device);scene_course_kernel<<<1,6,0,p->stream>>>(p->device,l.device);}
            if(l.device.rig_directions)rig_tangent_kernel<<<(l.host.q+127)/128,128,0,p->stream>>>(p->device,l.device);
            linearize_kernel<<<static_cast<unsigned>((p->host.n+127)/128),128,0,p->stream>>>(p->device,l.device,p->device_output,p->device_status,l.d_jac.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[1],p->stream));
            whiten_kernel<<<l.host.ng,128,0,p->stream>>>(l.device,p->device_output,l.d_jac.data,l.d_white.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[2],p->stream));
            loss_kernel<<<(l.host.ng+127)/128,128,0,p->stream>>>(l.device,l.d_white.data,l.d_stats.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[3],p->stream));
            if(!l.scales.empty()){
                corner_station_kernel<<<l.host.ncomponents,128,0,p->stream>>>(p->device,l.device,p->device_output,l.d_jac.data,l.d_white.data,l.d_stats.data);
                whiten_corner_locals_kernel<<<l.host.ng,128,0,p->stream>>>(l.device,p->device_output,l.d_jac.data,l.d_white.data);
            }
            block_kernel<<<l.host.ncomponents,128,0,p->stream>>>(l.device,l.d_white.data,l.d_stats.data,l.h_values(),l.d_g.data);
            cuda_check(cudaGetLastError());
            cuda_check(cudaMemsetAsync(l.d_shared.data,0,l.d_shared.bytes(),p->stream));
            shared_kernel<<<l.host.ncomponents,128,0,p->stream>>>(l.device,l.h_values(),l.d_g.data,l.d_shared.data);
            cuda_check(cudaGetLastError());
            cuda_check(cudaMemsetAsync(l.d_summary.data,0,l.d_summary.bytes(),p->stream));
            summary_kernel<<<unsigned((std::max(p->host.n,size_t(l.host.ng))+127)/128),128,0,p->stream>>>(p->host.n,p->device_status,l.host.ng,l.d_stats.data,l.d_summary.data);
            cuda_check(cudaGetLastError());cuda_check(cudaEventRecord(l.event[4],p->stream));
            cuda_check(cudaMemcpyAsync(summary,l.d_summary.data,4*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaEventRecord(l.event[5],p->stream));cuda_check(cudaEventSynchronize(l.event[5]));
            for(int stage=0;stage<5;++stage){float ms;cuda_check(cudaEventElapsedTime(&ms,l.event[stage],l.event[stage+1]));timings[stage+1]=ms;}
            timings[7]=4*sizeof(double);l.gpu_ready=summary[0]==0&&finite_array(summary,4);
        }
        timings[0]=millis(start);timings[6]=double(l.device_bytes());
        if(summary[0]!=0||!finite_array(summary,4)){last_error="G2 invalid rows retained; no valid block system (failed="+std::to_string(summary[0])+")";return 21;}
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_export_sparse_blocks(void* handle,int backend,double* h,size_t nh,double* g,size_t ng,double* shared,size_t nq){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;
    if((nh&&!h)||(ng&&!g)||!shared||nh!=l.h_count||ng!=l.g_count||nq!=l.host.nnz+l.host.q||(backend!=0&&backend!=1)||(backend==0?!l.cpu_ready:!l.gpu_ready)){last_error="invalid G2 explicit block export";return 1;}
    try{if(backend==0){std::copy(l.cpu_h.begin(),l.cpu_h.end(),h);std::copy(l.cpu_g.begin(),l.cpu_g.end(),g);std::copy(l.cpu_shared.begin(),l.cpu_shared.end(),shared);}
        else{std::vector<double> packed(l.normal_storage_count());
            if(!packed.empty())cuda_check(cudaMemcpyAsync(packed.data(),l.h_values(),packed.size()*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            if(ng)cuda_check(cudaMemcpyAsync(g,l.d_g.data,ng*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaMemcpyAsync(shared,l.d_shared.data,nq*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));
            for(const auto& c:l.components)for(int i=0;i<c.dim;++i)for(int j=0;j<c.dim;++j)h[c.h+size_t(i)*c.dim+j]=component_h(packed.data(),c,i,j);
        }return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_export_blocks(void* handle,int backend,double* h,size_t nh,double* g,size_t ng,double* shared,size_t nq){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;size_t q=l.host.q;
    if(!shared||nq!=q*q+q){last_error="dense diagnostic shape";return 1;}
    try{std::vector<double> sparse(l.host.nnz+q);int status=rs_g2_export_sparse_blocks(handle,backend,h,nh,g,ng,sparse.data(),sparse.size());if(status)return status;
        std::fill(shared,shared+nq,0.);for(int i=0;i<l.host.q;++i)for(int k=l.csr_offsets[i];k<l.csr_offsets[i+1];++k){int j=l.csr_columns[k];shared[size_t(i)*q+j]=shared[size_t(j)*q+i]=sparse[k];}std::copy_n(sparse.data()+l.host.nnz,q,shared+q*q);return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g2_probe_rows(void* handle,int backend,const int* rows,size_t n,double* jac,size_t nj){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G2 plan";return 1;}auto& l=*p->linearization;size_t width=2*size_t(l.host.q+1);
    if(!rows||!jac||nj!=n*width||(backend!=0&&backend!=1)||(backend==0?!l.cpu_ready:!l.gpu_ready)){last_error="invalid G2 row probe";return 1;}
    size_t total=0;for(size_t i=0;i<n;++i){if(rows[i]<0||rows[i]>=int(p->host.n)){last_error="G2 row probe outside source";return 1;}total+=2*size_t(l.row_plan[rows[i]].count+1);}
    try{std::vector<double> packed(total);size_t off=0;for(size_t i=0;i<n;++i){auto rp=l.row_plan[rows[i]];size_t count=2*size_t(rp.count+1);
            if(backend==0)std::copy_n(l.cpu_jac.data()+rp.jac,count,packed.data()+off);else cuda_check(cudaMemcpyAsync(packed.data()+off,l.d_jac.data+rp.jac,count*sizeof(double),cudaMemcpyDeviceToHost,p->stream));off+=count;}
        if(backend==1)cuda_check(cudaStreamSynchronize(p->stream));std::fill(jac,jac+nj,0.);off=0;
        for(size_t i=0;i<n;++i){auto rp=l.row_plan[rows[i]];for(int d=0;d<2;++d){for(int j=0;j<rp.count;++j)jac[i*width+d*(l.host.q+1)+l.columns[rp.begin+j]]=packed[off+d*(rp.count+1)+j];jac[i*width+d*(l.host.q+1)+l.host.q]=packed[off+d*(rp.count+1)+rp.count];}off+=2*size_t(rp.count+1);}return 0;
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
        double values[]={millis(start),double(s->device_bytes()),double(s->m),double(p->linearization->host.nnz),double(major),double(minor),double(patch),double(s->matrix.count)};
        std::copy(values,values+8,info);p->solver=std::move(s);return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 20;}
}
RS_API int rs_g3_refine_stations_ex(void* handle,int global_search,double* info,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!p->linearization->host.pixel_residuals||!info||n!=6||(global_search!=0&&global_search!=1)){last_error="independent raw pixel solver required for curve refinement";return 1;}
    auto start=Clock::now();std::fill(info,info+6,0.);
    try{
        p->solver->station_refinement.run(p->device,p->stream,info,global_search!=0);
        info[4]=millis(start);info[5]=global_search?double(p->solver->station_refinement.samples.count):0.;
        if(info[0]){p->host_stale=true;p->linearization->gpu_ready=false;p->linearization->cpu_ready=false;p->solver->trial_available=false;}
        if(info[2]){last_error="curve refinement has unprojectable rows; no observations removed";return 21;}
        return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 22;}
}
RS_API int rs_g3_refine_stations(void* handle,double* info,size_t n){return rs_g3_refine_stations_ex(handle,1,info,n);}
RS_API int rs_g3_step(void* handle,double* summary,size_t ns,double* timings,size_t nt){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!summary||!timings||ns!=32||nt!=12){last_error="invalid G3 step arguments";return 1;}
    auto start=Clock::now();double lin_time=0.;
    try{
        if(!p->linearization->gpu_ready){double v[4],t[8];int status=rs_g2_linearize(handle,1,v,4,t,8);if(status)return status;lin_time=t[0];}
        auto trial=p->bind(p->solver->trial_input.data,p->device_codes);
        if(p->host.nscene)trial.course=p->solver->trial_course.data;
        bool accepted=p->solver->run(p->device,trial,*p->linearization,p->input.size(),p->stream,p->device_output,summary,timings,p->stop_requested);
        if(accepted)p->host_stale=true;
        timings[11]=lin_time;timings[0]=millis(start);return 0;
    }catch(const SolverStopped& e){last_error=e.what();return 25;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 22;}
}
RS_API int rs_g4_set_numeric_progress(void* handle,rs_lm_numeric_progress callback,void* context){
    auto* p=static_cast<Problem*>(handle);
    if(!p||!p->solver){last_error="missing solver for numeric progress";return 1;}
    p->solver->numeric_progress=callback;p->solver->numeric_context=context;return 0;
}
RS_API int rs_g4_request_stop(void* handle){
    auto* p=static_cast<Problem*>(handle);
    if(!p)return 1;
    p->stop_requested.store(true);return 0;
}
RS_API int rs_g4_stationarity_version(void){return 2;}
RS_API int rs_g4_stationarity_cached_norm(void* handle,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->solver||!p->linearization||!values||n!=5){last_error="invalid cached stationarity norm query";return 1;}
    auto& s=*p->solver;auto& l=*p->linearization;bool valid=s.stationarity_valid&&l.gpu_ready&&s.stationarity_generation==l.generation;
    values[0]=valid&&!s.stationarity_bound_only?1.:0.;values[1]=s.stationarity_projection_complete?1.:0.;
    values[2]=s.stationarity_state_values[0];values[3]=s.stationarity_state_values[1];values[4]=s.stationarity_active_count;return 0;
}
RS_API int rs_g4_stationarity_cached_certificate(void* handle,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->solver||!p->linearization||!values||n!=6){last_error="invalid cached stationarity certificate query";return 1;}
    auto& s=*p->solver;auto& l=*p->linearization;bool valid=s.stationarity_valid&&l.gpu_ready&&s.stationarity_generation==l.generation;
    values[0]=valid?1.:0.;values[1]=s.stationarity_projection_complete?1.:0.;
    values[2]=s.stationarity_state_values[0];values[3]=s.stationarity_state_values[1];values[4]=s.stationarity_active_count;
    values[5]=s.stationarity_projection_complete?1.:(s.stationarity_bound_only?3.:2.);return 0;
}
RS_API int rs_g4_stationarity_diagnostics(void* handle,double tolerance,double* out,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!out||n!=12||!finite_scalar(tolerance)||tolerance<=0){last_error="invalid stationarity diagnostic";return 1;}
    try{
        auto& solver=*p->solver;auto& lin=*p->linearization;
        if(!lin.gpu_ready){double summary[4],timings[8];int code=rs_g2_linearize(handle,1,summary,4,timings,8);if(code)return code;}
        double state[3];const std::atomic<bool> no_stop{false};solver.stationarity_state(p->device,lin,p->stream,tolerance,no_stop,state);
        auto& metrics=solver.stationarity_metrics;cuda_check(cudaMemsetAsync(metrics.data,0,metrics.bytes(),p->stream));
        const int width=std::max({lin.host.q,solver.view.nlocal,solver.stationarity_active_count});
        lm_stationarity_metrics<<<(width+127)/128,128,0,p->stream>>>(p->device,lin.device,solver.view,lin.d_shared.data,lin.d_g.data,
            solver.stationarity_faces.data,solver.stationarity_active_count,solver.stationarity_matrix.data,solver.stationarity_dual.data,metrics.data);
        cuda_check(cudaMemcpyAsync(out,metrics.data,metrics.bytes(),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));
        out[8]=state[0];out[9]=state[1];out[10]=solver.stationarity_active_count;out[11]=state[2];
        if(state[2]){last_error="stationarity projection failed; accepted state unchanged";return 23;}return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 23;}
}
RS_API int rs_g4_stationarity_export(void* handle,double* faces,size_t nf,double* jacobian,size_t nj,double* gradient,size_t ng){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization){last_error="missing stationarity problem";return 1;}
    auto& solver=*p->solver;auto& lin=*p->linearization;size_t count=solver.stationarity_active_count,q=lin.host.q;
    if(nf!=24*count||nj!=count*q||ng!=2*q||(nf&&!faces)||(nj&&!jacobian)||!gradient){last_error="stationarity export shape";return 1;}
    try{
        if(!solver.stationarity_valid||!solver.stationarity_projection_complete||!lin.gpu_ready||solver.stationarity_generation!=lin.generation){last_error="stationarity export requires a completed diagnostic for the current state";return 1;}
        if(count){DeviceBuffer<double> records;records.allocate(nf);
            lm_stationarity_export_faces<<<unsigned((count+127)/128),128,0,p->stream>>>(solver.stationarity_faces.data,int(count),solver.stationarity_dual.data,records.data);
            cuda_check(cudaMemcpyAsync(faces,records.data,nf*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
            cuda_check(cudaMemcpyAsync(jacobian,solver.stationarity_matrix.data,nj*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));}
        cuda_check(cudaMemcpyAsync(gradient,lin.d_shared.data+lin.host.nnz,q*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_check(cudaMemcpyAsync(gradient+q,solver.stationarity_descent.data,q*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(p->stream);last_error=e.what();return 23;}
}
RS_API int rs_g3_objective_timings(void* handle,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!values||n!=6){last_error="invalid objective timing output";return 1;}
    std::copy(p->solver->objective_ms,p->solver->objective_ms+6,values);return 0;
}
// Cumulative host timing/counters; event spans remain in the existing step ABI.
RS_API int rs_g4_performance_metrics(void* handle,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->solver||!values||(n!=16&&n!=19)){last_error="invalid solver performance output";return 1;}
    auto& s=*p->solver;double out[]={s.stationarity_wall_ms,double(s.stationarity_recomputations),double(s.stationarity_cache_hits),
        s.stationarity_faces_ms,s.stationarity_cone_ms,s.stationarity_readback_ms,double(s.stationarity_active_count),double(s.stationarity_sweeps),
        s.qp_line_wall_ms,double(s.qp_line_evaluations),s.trial_station_wall_ms,s.prediction_wall_ms,double(s.profile_restoration_calls),
        double(s.trial_white.bytes()),double(s.envelope_projections),double(s.envelope_backtracks),
        s.last_damping_energy,s.last_prediction,double(s.damping_recoveries)};
    std::copy_n(out,n,values);return 0;
}
RS_API int rs_g4_solve(void* handle,const double* tol,size_t n,rs_lm_progress callback,void* context,double* out,size_t no){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!tol||n!=3||!out||no!=17||!finite_array(tol,n)||tol[0]<=0||tol[1]<=0||tol[2]<=0){
        last_error="invalid G4 controller arguments";return 1;
    }
    std::fill(out,out+17,0.);auto started=Clock::now();
    auto& solver=*p->solver;auto& lin=*p->linearization;
    const int initial_trials=solver.trials,initial_accepted=solver.accepted;
    bool accepted_small=false,check_cost_progress=false;
    double accepted_origin_cost=0.,accepted_prediction=0.,accepted_lambda=0.;
    double latest_step=0.,relative=0.,copies=0.,total_lin=0.,total_steps=0.;
    int linearizations=0;
    try{
        for(;;){
            double lin_ms=0.;
            if(!lin.gpu_ready){
                double v[4],t[8];int status=rs_g2_linearize(handle,1,v,4,t,8);if(status)return status;
                lin_ms=t[0];total_lin+=lin_ms;copies+=32;++linearizations;
            }
            double state[3];const auto stationarity_started=Clock::now();
            try{copies+=solver.stationarity_state(p->device,lin,p->stream,tol[0],p->stop_requested,state,true);}
            catch(const SolverStopped&){p->stop_requested.exchange(false);out[0]=3;break;}
            const double stationarity_ms=millis(stationarity_started);
            out[4]=state[0];out[5]=state[1];
            if(state[2])throw std::runtime_error("original objective/constraint stationarity evaluation failed");
            if(p->stop_requested.exchange(false)){out[0]=3;break;}
            // A truncated/damped small step is not a stationarity certificate.
            // Recompute the projected gradient after acceptance before success.
            if(state[1]<=tol[0]){out[0]=accepted_small?1:0;break;}
            // Re-evaluate after coordinate refinement. An over-damped tiny
            // step is recoverable: retain the ordinary LM damping reduction.
            // Only model stagnation not dominated by damping stops here.
            if(check_cost_progress&&fabs(accepted_origin_cost-state[0])<=tol[2]*fabs(accepted_origin_cost)&&
               fabs(accepted_prediction)<=tol[2]*fabs(accepted_origin_cost)){
                if(solver.last_damping_energy>.5*accepted_prediction&&solver.lambda<accepted_lambda)++solver.damping_recoveries;
                else{out[0]=4;break;}
            }
            check_cost_progress=false;
            if(!finite_scalar(solver.lambda)||!finite_scalar(solver.nu))
                throw std::runtime_error("LM damping overflow without convergence");
            double metrics[32],timings[12];
            int status=rs_g3_step(handle,metrics,32,timings,12);
            if(status==25){p->stop_requested.exchange(false);last_error.clear();out[0]=3;break;}
            if(status)return status;
            timings[11]=lin_ms;total_steps+=timings[0];timings[0]+=lin_ms+stationarity_ms;
            copies+=metrics[23];latest_step=metrics[31];
            relative=fabs(metrics[12])/std::max(1.,fabs(metrics[1]));
            accepted_small=metrics[15]&&metrics[3]==1.&&latest_step<=tol[1]&&relative<=tol[2];
            check_cost_progress=metrics[15]!=0.;accepted_origin_cost=metrics[1];accepted_prediction=metrics[11];accepted_lambda=metrics[0];
            if(callback)callback(metrics,timings,state[1],solver.numerical_rejections,context);
            // Rejected, unresolvable objective differences are reported explicitly,
            // never labelled as a successful small-step convergence.
            if(p->stop_requested.load())continue;
            if(!metrics[15]&&metrics[16]!=4&&metrics[16]!=7&&latest_step<=tol[1]&&metrics[8]==0&&
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
RS_API int rs_g3_export_sparse_system(void* handle,double* values,size_t nv,int* blocked,size_t nb){
    auto* p=static_cast<Problem*>(handle);last_error.clear();if(!p||!p->solver||!p->solver->trial_available||!values){last_error="G3 system snapshot unavailable";return 1;}
    auto& s=*p->solver;auto& l=*p->linearization;size_t q=l.host.q,nl=l.scales.size(),step=q+nl;
    if(nv!=l.host.nnz+q+2*step||nb!=nl||(nl&&!blocked)){last_error="G3 sparse system snapshot shape";return 1;}
    try{size_t off=0;auto copy=[&](const double* d,size_t n){if(n)cuda_check(cudaMemcpyAsync(values+off,d,n*sizeof(double),cudaMemcpyDeviceToHost,p->stream));off+=n;};copy(s.csr.data,l.host.nnz);copy(s.rhs.data,q);copy(s.step.data,step);copy(s.applied.data,step);if(nl)cuda_check(cudaMemcpyAsync(blocked,s.blocked.data,nl*sizeof(int),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;}catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g3_probe_changes(void* handle,double* values,size_t n){
    auto* p=static_cast<Problem*>(handle);last_error.clear();
    if(!p||!p->solver||!p->linearization||!p->solver->trial_available||!values||n!=6*p->host.n){last_error="last-step probe unavailable/shape";return 1;}
    try{DeviceBuffer<double> out;out.allocate(n);auto& l=*p->linearization;auto& s=*p->solver;
        lm_probe_changes<<<unsigned((p->host.n+127)/128),128,0,p->stream>>>(p->device,l.device,s.view,l.d_jac.data,p->device_output,s.trial_output.data,out.data);
        cuda_check(cudaGetLastError());cuda_check(cudaMemcpyAsync(values,out.data,n*sizeof(double),cudaMemcpyDeviceToHost,p->stream));cuda_check(cudaStreamSynchronize(p->stream));return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}
RS_API int rs_g3_export_system(void* handle,double* values,size_t nv,int* blocked,size_t nb){
    auto* p=static_cast<Problem*>(handle);if(!p||!p->linearization){last_error="missing G3 system";return 1;}auto& l=*p->linearization;size_t q=l.host.q,step=q+l.scales.size();
    if(!values||nv!=q*q+q+2*step){last_error="G3 dense system snapshot shape";return 1;}
    try{std::vector<double> sparse(l.host.nnz+q+2*step);int status=rs_g3_export_sparse_system(handle,sparse.data(),sparse.size(),blocked,nb);if(status)return status;
        std::fill(values,values+q*q,0.);for(int i=0;i<l.host.q;++i)for(int k=l.csr_offsets[i];k<l.csr_offsets[i+1];++k){int j=l.csr_columns[k];values[size_t(i)*q+j]=values[size_t(j)*q+i]=sparse[k];}
        std::copy(sparse.begin()+l.host.nnz,sparse.end(),values+q*q);return 0;
    }catch(const std::exception& e){last_error=e.what();return 20;}
}

#endif

}
