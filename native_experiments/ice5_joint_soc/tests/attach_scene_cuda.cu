// Real CUDA operations, with recoverable errors injected only in this test TU.
// Including the implementation lets us inspect ownership/views without adding
// fault switches or diagnostic APIs to the production library.
#include <cuda_runtime.h>
#ifdef RINKSPACE_HAVE_CUDSS
#include <cusolverDn.h>
#include <cudss.h>
#endif
#include <array>
#include <map>
#include <iostream>
#include "rinkspace/coupled_dual.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace fault {
// For the outer LM check only: feed the real device helper the finite overflow
// counterexample at its call boundary. The production controller and kernels
// still handle its actual partial multipliers and unresolved certificate.
__device__ bool inject_dual=false;
__device__ int dual_failures=0;
RS_HD inline int minimize(int count,int stride,const double* gram,const double* target,
        double* dual,const double* equality,double* kkt=nullptr){
#ifdef __CUDA_ARCH__
    if(inject_dual){
        double g[]{1.,.5,.5,1.},t[]{1e308,-1e308},d[]{0.,0.},e[]{1.,1.};
        int sweeps=rinkspace::coupled_dual_minimize(2,2,g,t,d,e,kkt);
        for(int i=0;i<count;++i)dual[i]=d[i%2];
        ++dual_failures;return sweeps;
    }
#endif
    return rinkspace::coupled_dual_minimize(count,stride,gram,target,dual,equality,kkt);
}
enum class Op { none, allocation, copy, sync };
Op operation=Op::none;int ordinal=0,seen=0;bool triggered=false;
std::map<void*,size_t> owned;
bool fail(Op op){
    if(op!=operation||++seen!=ordinal)return false;
    triggered=true;operation=Op::none;return true;
}
template<class T> cudaError_t allocate(T** ptr,size_t bytes){
    if(fail(Op::allocation))return cudaErrorMemoryAllocation;
    auto result=cudaMalloc(reinterpret_cast<void**>(ptr),bytes);
    if(result==cudaSuccess&&*ptr)assert(owned.emplace(*ptr,bytes).second);
    return result;
}
cudaError_t release(void* ptr){
    if(ptr)assert(owned.erase(ptr)==1);
    return cudaFree(ptr);
}
cudaError_t copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream=0){
    if(fail(Op::copy))return cudaErrorInvalidValue;
    return cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
cudaError_t sync(cudaStream_t stream){
    if(fail(Op::sync))return cudaErrorInvalidValue;
    return cudaStreamSynchronize(stream);
}
}
#define cudaMalloc fault::allocate
#define cudaFree fault::release
#define cudaMemcpyAsync fault::copy
#define cudaStreamSynchronize fault::sync
#define coupled_dual_minimize fault::minimize
#include "../src/core_gpu.cu"
#undef coupled_dual_minimize
#undef cudaMalloc
#undef cudaFree
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize

struct Fixture {
    void* handle=nullptr;
    std::vector<double> values,data{0.,1.},directions,lower,upper;
    std::array<int,12> record{}; // Frozen path: scene attachment changes no geometry.
    Fixture(){
        const double board[]{.5,.25,.1,.1,2.,.02};
        const double layout[]{-.3,.3,-.15,.1,.03,.1,.35,.67,.4,.07,-.4,-.4,-.4,-.4,.35,.35,.35,.35,
            0.,.1,.07,-.25,.25,-.4,.4,.3,0.,.1,0.,.03,.03,.03,.03,.05,.07};
        const double rig[]{1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,0.,0.,0.,0.,-3.,0.,0.,0.,0.,0.,0.,
            100.,1.,50.,50.,0.,0.,0.,0.,0.,0.,0.,0.,0.,.02};
        double controls[24];for(int i=0;i<8;++i){controls[3*i]=-.01+.02*i/7;controls[3*i+1]=-.02+.04*i/7;controls[3*i+2]=std::log(100.);}
        const int codes[]{0,0,0};const double stations[]{.2,.5,.8},times[]{0.,.05,.1},observed[6]{};
        double metrics[10];
        assert(rs_g1_create(board,6,layout,35,0,rig,38,controls,8,-2.,1.,101,codes,stations,times,observed,3,&handle,metrics,10)==0);
        // Include existing multi-rig ownership in the rollback checks.
        std::vector<double> rigs(rig,rig+38);rigs.insert(rigs.end(),rig,rig+38);
        const int cameras[]{0,1,0};double bytes;
        assert(rs_gpu_set_camera_groups(handle,rigs.data(),rigs.size(),cameras,3,&bytes)==0);
        const int path[]{0,0,6,0,0,0,0};const double segment[]{-.2,-.1,0.,.2,.1,0.};
        assert(rs_gpu_attach_paths(handle,path,7,segment,6,&bytes)==0);
        values.assign(board,board+6);values.insert(values.end(),layout,layout+35);
        directions.assign(values.size(),0.);lower.assign(values.size(),-INFINITY);upper.assign(values.size(),INFINITY);
    }
    ~Fixture(){rs_g1_destroy(handle);}
    Problem& problem(){return *static_cast<Problem*>(handle);}
    int attach(){return rs_gpu_attach_scene(handle,values.data(),values.size(),record.data(),record.size(),
        data.data(),data.size(),1,directions.data(),directions.size(),lower.data(),upper.data());}
    std::vector<double> evaluate(int backend){
        std::vector<double> result(33);int status[3];double metrics[3];
        assert(rs_g1_evaluate(handle,backend,result.data(),result.size(),status,3,metrics,3)==0);
        for(int code:status)assert(code==0);return result;
    }
};

template<class T> auto bytes_of(const T& value){
    std::array<unsigned char,sizeof(T)> bytes;std::memcpy(bytes.data(),&value,sizeof(T));return bytes;
}
std::vector<double> download(Problem& p){
    std::vector<double> values(p.input.size());
    cuda_check(cudaMemcpy(values.data(),p.device_input,values.size()*sizeof(double),cudaMemcpyDeviceToHost));return values;
}
void check_attach(fault::Op op,int ordinal,bool retry){
    const auto initial_allocations=fault::owned;
    {
        Fixture f;auto& p=f.problem();const auto input=p.input,device_values=download(p);const auto codes=p.codes;
        const auto host=bytes_of(p.host),device=bytes_of(p.device);const auto owners=fault::owned;
        const auto old_host=p.input.data(),old_device=p.device_input;
        const auto old_capacity=p.input.capacity();const auto old_course=p.host_course.get();
        const auto before=f.evaluate(1),before_cpu=f.evaluate(0);
        for(size_t i=0;i<before.size();++i)assert(std::abs(before[i]-before_cpu[i])<1e-10);
        fault::operation=op;fault::ordinal=ordinal;fault::seen=0;fault::triggered=false;
        int result=f.attach();fault::operation=fault::Op::none;
        if(op!=fault::Op::none){
            assert(fault::triggered&&result==24&&!last_error.empty());
            // Includes every board/layout/rig/control/station/source value and
            // all pointer fields of both views, not just exported coordinates.
            assert(p.input==input&&download(p)==device_values&&p.codes==codes);
            assert(p.input.data()==old_host&&p.input.capacity()==old_capacity&&p.device_input==old_device);
            assert(bytes_of(p.host)==host&&bytes_of(p.device)==device&&p.host_course.get()==old_course);
            assert(p.scene_q==0&&p.host.nscene==0&&p.device.nscene==0);
            assert(p.scene_paths.empty()&&p.scene_data.empty()&&p.scene_directions.empty()&&p.scene_lower.empty()&&p.scene_upper.empty());
            assert(!p.d_scene_paths.data&&!p.d_scene_data.data&&p.d_scene_paths.count==0&&p.d_scene_data.count==0);
            assert(fault::owned==owners);assert(f.evaluate(0)==before_cpu&&f.evaluate(1)==before);
            if(!retry)return; // Destructor also has to be safe immediately after failure.
            result=f.attach();
        }
        assert(result==0&&p.scene_q==1&&p.host.nscene==f.values.size()&&p.device.nscene==f.values.size());
        assert(p.host.board==p.input.data()&&p.device.board==p.device_input);
        assert(p.host.scene_paths==p.scene_paths.data()&&p.device.scene_paths==p.d_scene_paths.data);
        assert(p.host.scene_data==p.scene_data.data()&&p.device.scene_data==p.d_scene_data.data);
        assert(p.host.scene_values==p.input.data()+input.size()&&p.device.scene_values==p.device_input+input.size());
        assert(std::equal(input.begin(),input.end(),p.input.begin()));
        assert(std::equal(f.values.begin(),f.values.end(),p.host.scene_values));
        assert(download(p)==p.input&&p.scene_directions==f.directions&&p.scene_lower==f.lower&&p.scene_upper==f.upper);
        assert(f.evaluate(0)==before_cpu&&f.evaluate(1)==before);
    }
    assert(fault::owned==initial_allocations);
}

#ifdef RINKSPACE_HAVE_CUDSS
void check_device_dual(){
    const auto initial_allocations=fault::owned;
    {
        DeviceBuffer<double> gram,target,dual,equality,scratch;
        gram.upload({1.,.5,.5,1.},0);equality.upload({1.,1.},0);scratch.allocate(2);
        target.allocate(2);dual.allocate(2);
        CoupledProjectionView c{};c.capacity=2;c.gram=gram.data;c.target=target.data;c.dual=dual.data;c.equality=equality.data;
        SolverView s{};s.scratch=scratch.data;
        for(const auto& t:std::vector<std::vector<double>>{{1e308,-1e308},{NAN,1.},{1.,0.}}){
            double seed[2]{},cpu_dual[2]{},kkt,status,result[2];
            coupled_dual_minimize(2,2,std::vector<double>{1.,.5,.5,1.}.data(),t.data(),cpu_dual,std::vector<double>{1.,1.}.data(),&kkt);
            cuda_check(cudaMemcpy(target.data,t.data(),2*sizeof(double),cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(dual.data,seed,2*sizeof(double),cudaMemcpyHostToDevice));
            lm_coupled_dual<<<1,1>>>(c,s,2,false);cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
            cuda_check(cudaMemcpy(&status,scratch.data+1,sizeof(double),cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(result,dual.data,2*sizeof(double),cudaMemcpyDeviceToHost));
            assert(status==double(coupled_dual_unresolved(2,t.data(),kkt)));
            if(std::isfinite(kkt))assert(std::memcmp(result,cpu_dual,sizeof(result))==0);
            else assert(status==1.);
        }
    }
    assert(fault::owned==initial_allocations);
    std::cout<<"PASS B03: production lm_coupled_dual on CUDA: finite overflow, NaN, finite CPU/device control\n";
}
void check_dual_failure_is_not_published(){
#ifdef _WIN32
    _putenv_s("RINKSPACE_SMALL_DUAL_BACKEND","gpu");
#else
    setenv("RINKSPACE_SMALL_DUAL_BACKEND","gpu",1);
#endif
    const auto initial_allocations=fault::owned;
    {
        // Same tiny lane/goal-boundary counterexample as the existing
        // test_gpu_joint_constraints fixture; no scene file or media is used.
        Fixture base;auto v=base.problem().host;
        int codes[]{0,0,0,0,1,1,1,1};double stations[]{0.,.1,.2,.3,0.,.1,.2,.3},times[8]{},observed[16]{},metrics[10];
        void* handle=nullptr;
        assert(rs_g1_create(v.board,6,v.layout,v.nl,6,v.rig,v.nr,v.controls,v.nc,v.first,v.dt,v.height,
            codes,stations,times,observed,8,&handle,metrics,10)==0);
        std::unique_ptr<Problem> p(static_cast<Problem*>(handle));
        std::vector<int> paths;std::vector<double> parameters;
        for(int code:{300,403}){
            int kind;double data[6];assert(shared_entity_definition(v.board,6,v.layout,v.nl,code,kind,data,6,6)==Status::ok);
            paths.insert(paths.end(),{kind,int(parameters.size()),6,0,0,0,0});parameters.insert(parameters.end(),data,data+6);
        }
        double bytes;assert(rs_gpu_attach_paths(handle,paths.data(),paths.size(),parameters.data(),parameters.size(),&bytes)==0);
        std::vector<double> scene(v.board,v.board+6+v.nl),directions(scene.size()*2,0.);
        directions[16*2]=.001;directions[29*2+1]=.001;
        std::vector<double> lo(scene.size(),-INFINITY),hi(scene.size(),INFINITY);int records[24]{};
        records[0]=records[12]=1;records[1]=300;records[13]=403;double scene_data=0.;
        assert(rs_gpu_attach_scene(handle,scene.data(),scene.size(),records,24,&scene_data,1,2,directions.data(),directions.size(),lo.data(),hi.data())==0);
        double output[88],evaluation[3];int status[8];
        assert(rs_g1_evaluate(handle,0,output,88,status,8,evaluation,3)==0);
        for(int i=0;i<8;++i){assert(status[i]==0);auto uv=const_cast<double*>(p->host.observed)+2*i;uv[0]=output[11*i+3]-(i<4?.1:0.);uv[1]=output[11*i+4];}
        cuda_check(cudaMemcpy(p->device_input,p->input.data(),p->input.size()*sizeof(double),cudaMemcpyHostToDevice));
        double controls[48]{},linear_info[12];int locals[8];std::fill(locals,locals+8,-1);
        assert(rs_g2_attach_pixels(handle,2,controls,48,locals,8,nullptr,0,linear_info,12)==0);
        double options[]{1e-3,0.,1e-4,1e-8,double(1ull<<30),double(1ull<<30)},solver_info[8];
        assert(rs_g3_attach(handle,nullptr,nullptr,0,options,6,solver_info,8)==0);
        const auto accepted=download(*p);bool inject=true;int failures=0;
        cuda_check(cudaMemcpyToSymbol(fault::inject_dual,&inject,sizeof(inject)));
        cuda_check(cudaMemcpyToSymbol(fault::dual_failures,&failures,sizeof(failures)));
        double summary[32],timings[12];int result=rs_g3_step(handle,summary,32,timings,12);
        inject=false;cuda_check(cudaMemcpyToSymbol(fault::inject_dual,&inject,sizeof(inject)));
        cuda_check(cudaMemcpyFromSymbol(&failures,fault::dual_failures,sizeof(failures)));
        assert(failures>0&&result==0&&summary[15]==0.&&!p->solver->trial_available);
        assert(download(*p)==accepted);
        // The same controller/handle can accept a finite step after disabling
        // the injected helper input; no failed dual state becomes accepted.
        assert(rs_g3_step(handle,summary,32,timings,12)==0&&summary[15]==1.&&p->solver->trial_available);
        assert(download(*p)!=accepted);
        std::cout<<"PASS B03: "<<failures<<" injected finite-overflow device dual calls: accepted state unchanged, no trial published; same-handle finite step accepted\n";
    }
    assert(fault::owned==initial_allocations);
}
#endif
int main(){
    check_attach(fault::Op::none,0,true);
    int cases=0;
    for(bool retry:{false,true})for(auto op:{fault::Op::allocation,fault::Op::copy,fault::Op::sync})
        for(int ordinal=1;ordinal<=(op==fault::Op::sync?1:3);++ordinal){
            check_attach(op,ordinal,retry);++cases;
            assert(fault::owned.empty());
        }
    std::cout<<"PASS B06: "<<cases<<" real-CUDA recoverable failures (3 allocation, 3 upload, sync; close/retry), success control; views/state/owners unchanged; no tracked leaks\n";
#ifdef RINKSPACE_HAVE_CUDSS
    check_device_dual();
    check_dual_failure_is_not_published();
#endif
}
