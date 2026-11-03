// Production G2 entry point with observable copies; no new exported test API.
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <array>
#include <limits>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace audit {
const void* status_source=nullptr;
size_t status_downloads=0,all_downloads=0;
cudaError_t copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream=0){
    if(kind==cudaMemcpyDeviceToHost){++all_downloads;if(src==status_source)++status_downloads;}
    return cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
void reset(){status_downloads=all_downloads=0;}
}
#define cudaMemcpyAsync audit::copy
#include "../src/core_gpu.cu"
#undef cudaMemcpyAsync
#include "rinkspace/ice_actual_chart.hpp"

struct Fixture {
    void* handle=nullptr;
    Fixture(){
        const double board[]{.5,.25,.1,.1,2.,.02};
        const double layout[]{-.3,.3,-.15,.1,.03,.1,.35,.67,.4,.07,-.4,-.4,-.4,-.4,.35,.35,.35,.35,
            0.,.1,.07,-.25,.25,-.4,.4,.3,0.,.1,0.,.03,.03,.03,.03,.05,.07};
        const double rig[]{1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,0.,0.,0.,0.,-3.,0.,0.,0.,0.,0.,0.,
            100.,1.,50.,50.,0.,0.,0.,0.,0.,0.,0.,0.,0.,.02};
        double controls[24],directions[24]{};
        for(int i=0;i<8;++i){controls[3*i]=-.01+.02*i/7;controls[3*i+1]=-.02+.04*i/7;
            controls[3*i+2]=std::log(100.);directions[3*i]=.001;}
        const int codes[]{100,100,100},locals[]{-1,-1,-1};
        const double stations[]{.1,.3,.7},times[]{0.,.1,.2},observed[6]{};
        double metrics[10],info[12];
        assert(rs_g1_create(board,6,layout,35,0,rig,38,controls,8,-2.,1.,101,
            codes,stations,times,observed,3,&handle,metrics,10)==0);
        assert(rs_g2_attach_pixels(handle,1,directions,24,locals,3,nullptr,0,info,12)==0);
    }
    ~Fixture(){rs_g1_destroy(handle);audit::status_source=nullptr;}
    Problem& problem(){return *static_cast<Problem*>(handle);}
};

std::string expected_error(const std::vector<int>& statuses,double failed){
    std::string message="G2 invalid rows retained; no valid block system (failed="+std::to_string(failed)+")";
    std::map<int,size_t> census;
    for(int code:statuses)++census[code];
    for(auto item:census)message+=" status_"+std::to_string(item.first)+"="+std::to_string(item.second);
    return message;
}

void check_backend_diagnostics(){
    Fixture f;auto& p=f.problem();auto& l=*p.linearization;
    audit::status_source=p.device_status;
    const std::vector<double> times(p.host.times,p.host.times+3);
    auto* host_times=const_cast<double*>(p.host.times);
    const double invalid=std::numeric_limits<double>::quiet_NaN();
    host_times[0]=host_times[2]=invalid;
    const std::vector<int> gpu_sentinels{73,74,73};
    cuda_check(cudaMemcpy(p.device_status,gpu_sentinels.data(),3*sizeof(int),cudaMemcpyHostToDevice));
    double summary[4],timings[8];audit::reset();
    assert(rs_g2_linearize(f.handle,0,summary,4,timings,8)==21);
    assert(summary[0]==2.&&l.cpu_status[0]!=0&&l.cpu_status[1]==0&&l.cpu_status[2]!=0);
    assert(rs_g1_last_error()==expected_error(l.cpu_status,2.));
    assert(audit::status_downloads==0&&audit::all_downloads==0&&!l.cpu_ready);
    std::vector<int> device_status(3);
    cuda_check(cudaMemcpy(device_status.data(),p.device_status,3*sizeof(int),cudaMemcpyDeviceToHost));
    assert(device_status==gpu_sentinels); // CPU diagnostics must not touch the device statuses.

    std::copy(times.begin(),times.end(),host_times);
    cuda_check(cudaMemcpy(const_cast<double*>(p.device.times)+1,&invalid,sizeof(double),cudaMemcpyHostToDevice));
    l.cpu_status.assign(3,92);audit::reset();
    assert(rs_g2_linearize(f.handle,1,summary,4,timings,8)==21);
    assert(summary[0]==1.&&audit::status_downloads==1&&!l.gpu_ready);
    const std::string gpu_message=rs_g1_last_error();
    cuda_check(cudaMemcpy(device_status.data(),p.device_status,3*sizeof(int),cudaMemcpyDeviceToHost));
    assert(device_status[0]==0&&device_status[1]!=0&&device_status[2]==0);
    assert(gpu_message==expected_error(device_status,1.));
    assert(l.cpu_status==std::vector<int>(3,92));

    cuda_check(cudaMemcpy(const_cast<double*>(p.device.times),times.data(),3*sizeof(double),cudaMemcpyHostToDevice));
    double cpu_summary[4];audit::reset();
    assert(rs_g2_linearize(f.handle,0,cpu_summary,4,timings,8)==0&&l.cpu_ready);
    assert(cpu_summary[0]==0.&&audit::status_downloads==0&&audit::all_downloads==0);
    assert(l.cpu_status==std::vector<int>(3,0));
    audit::reset();
    assert(rs_g2_linearize(f.handle,1,summary,4,timings,8)==0&&l.gpu_ready);
    assert(summary[0]==0.&&audit::status_downloads==0);
    for(int i=0;i<4;++i)assert(std::isfinite(summary[i])&&std::abs(summary[i]-cpu_summary[i])<=1e-10*std::max(1.,std::abs(cpu_summary[i])));
    assert(std::string(rs_g1_last_error()).empty());
    std::cout<<"PASS B06: CPU failures=2, GPU sentinels unchanged, no CPU D2H; GPU failures=1, device census only; return 21/raw statuses and both success branches preserved\n";
}

struct RadiusStep { RS_HD double operator()(size_t i)const{return i==61?.001:0.;} };
__global__ void boundary_kernel(const double* values,rinkspace::IceActualWitness* out,int* success){
    int trial=threadIdx.x;
    if(trial<2)success[trial]=rinkspace::ice_actual_chart_retract(values,RadiusStep{},out[trial],trial!=0);
}
void check_boundary_retraction(){
    std::array<double,152> values{};
    values[6]=.3;values[7]=-.2;values[9]=.1;values[10]=.03;
    values[11]=.25;values[12]=.5;values[13]=.75;
    values[49]=-.01;values[50]=.01;values[51]=0.;values[52]=.005;
    double* device_values=nullptr;rinkspace::IceActualWitness* device_witness=nullptr;int* device_success=nullptr;
    cuda_check(cudaMalloc(&device_values,sizeof(values)));
    cuda_check(cudaMalloc(&device_witness,2*sizeof(rinkspace::IceActualWitness)));
    cuda_check(cudaMalloc(&device_success,2*sizeof(int)));
    cuda_check(cudaMemcpy(device_values,values.data(),sizeof(values),cudaMemcpyHostToDevice));
    boundary_kernel<<<1,2>>>(device_values,device_witness,device_success);
    cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
    rinkspace::IceActualWitness witness[2]{};int success[2]{};
    cuda_check(cudaMemcpy(witness,device_witness,sizeof(witness),cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(success,device_success,sizeof(success),cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(device_success));cuda_check(cudaFree(device_witness));cuda_check(cudaFree(device_values));
    for(int trial=0;trial<2;++trial){
        assert(success[trial]);
        rinkspace::IceActualWitness host{};
        assert(rinkspace::ice_actual_chart_retract(values.data(),RadiusStep{},host,trial!=0));
        for(int i=0;i<8;++i){
            assert(witness[trial].x_deviation[i]>=-.01&&witness[trial].x_deviation[i]<=.01);
            assert(witness[trial].c[i/4]+values[9]*witness[trial].x_deviation[i]==host.c[i/4]+values[9]*host.x_deviation[i]);
        }
        for(int i=0;i<32;++i)assert(witness[trial].radius_deviation[i]>=-.01&&witness[trial].radius_deviation[i]<=.01);
        const double radius=witness[trial].radius[0]*(1.+witness[trial].radius_deviation[0]);
        const double expected=values[10]*values[11]*std::exp(.001);
        assert(std::abs(radius-expected)<=8*std::numeric_limits<double>::epsilon()*expected);
    }
    std::cout<<"PASS B01: actual CUDA ordinary/constrained nonzero radius retractions keep inclusive X and radius bounds\n";
}
int main(){
    int count=0;auto status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess||count==0){
        std::cout<<"NOT RUN: CUDA device unavailable: "<<cudaGetErrorString(status)<<"\n";return 77;
    }
    check_boundary_retraction();
    check_backend_diagnostics();
}
