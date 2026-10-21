#include "rinkfit/joint_gpu_driver.hpp"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#error Joint CUDA Driver bridge currently targets the reviewed Windows runtime.
#endif

namespace rinkfit::joint_gpu {
namespace {
using Result=int;
using Context=void*;
using Module=void*;
using Function=void*;
using Event=void*;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::vector<char> read_ptx(const std::string& file) {
 std::ifstream f(file,std::ios::binary|std::ios::ate);
 if(!f)throw std::runtime_error("Joint GPU: cannot open PTX: "+file);
 const auto n=f.tellg();
 if(n<=0 || n>std::streamoff(64*1024*1024))throw std::runtime_error("Joint GPU: invalid PTX size");
 std::vector<char> bytes(static_cast<std::size_t>(n)+1,0);f.seekg(0);f.read(bytes.data(),n);
 if(!f)throw std::runtime_error("Joint GPU: incomplete PTX read");
 return bytes;
}
}
struct DriverContext::Impl {
 HMODULE driver=nullptr; Context context=nullptr; Module module=nullptr; Function kernel=nullptr;
 std::unordered_map<DevicePtr,std::size_t> allocations;
 DriverTelemetry telemetry;Event kernel_begin=nullptr,kernel_end=nullptr;
 Result (WINAPI *init)(unsigned)=nullptr;
 Result (WINAPI *device_get)(int*,int)=nullptr;
 Result (WINAPI *context_create)(Context*,unsigned,int)=nullptr;
 Result (WINAPI *context_set)(Context)=nullptr;
 Result (WINAPI *context_get)(Context*)=nullptr;
 Result (WINAPI *context_destroy)(Context)=nullptr;
 Result (WINAPI *context_sync)()=nullptr;
 Result (WINAPI *event_create)(Event*,unsigned)=nullptr;
 Result (WINAPI *event_destroy)(Event)=nullptr;
 Result (WINAPI *event_record)(Event,void*)=nullptr;
 Result (WINAPI *event_elapsed)(float*,Event,Event)=nullptr;
 Result (WINAPI *module_load)(Module*,const void*,unsigned,int*,void**)=nullptr;
 Result (WINAPI *module_unload)(Module)=nullptr;
 Result (WINAPI *function_get)(Function*,Module,const char*)=nullptr;
 Result (WINAPI *memory_allocate)(DevicePtr*,std::size_t)=nullptr;
 Result (WINAPI *memory_free)(DevicePtr)=nullptr;
 Result (WINAPI *copy_htod)(DevicePtr,const void*,std::size_t)=nullptr;
 Result (WINAPI *copy_dtoh)(void*,DevicePtr,std::size_t)=nullptr;
 Result (WINAPI *launch)(Function,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)=nullptr;
 Result (WINAPI *error_name)(Result,const char**)=nullptr;
 Result (WINAPI *error_string)(Result,const char**)=nullptr;
 template<class T> void bind(T& destination,const char* symbol) {
  destination=reinterpret_cast<T>(GetProcAddress(driver,symbol));
  if(!destination)throw std::runtime_error(std::string("Joint GPU: system driver lacks ")+symbol);
 }
 void check(Result r,const char* action) const {
  if(r==0)return;const char *name=nullptr,*message=nullptr;
  if(error_name)error_name(r,&name);if(error_string)error_string(r,&message);
  throw std::runtime_error(std::string("Joint GPU ")+action+": "+std::to_string(r)+" "+(name?name:"")+" "+(message?message:""));
 }
 void current()const{check(context_set(context),"select context");}
 // CUDA runtime solvers own a separate caller context; never leave our private one selected.
 struct SavedContext {
  const Impl& owner; Context previous=nullptr;
  explicit SavedContext(const Impl& x):owner(x){owner.check(owner.context_get(&previous),"read caller context");}
  ~SavedContext(){owner.context_set(previous);}
  SavedContext(const SavedContext&)=delete;SavedContext& operator=(const SavedContext&)=delete;
 };
 void range(DevicePtr p,std::size_t offset,std::size_t bytes)const {
  const auto it=allocations.find(p);
  if(it==allocations.end()||offset>it->second||bytes>it->second-offset)throw std::runtime_error("Joint GPU: transfer outside owned buffer");
 }
 ~Impl(){
  if(context){Context previous=nullptr;const bool read_ok=context_get&&context_get(&previous)==0;context_set(context);for(const auto& a:allocations)memory_free(a.first);if(kernel_begin)event_destroy(kernel_begin);if(kernel_end)event_destroy(kernel_end);if(module)module_unload(module);context_destroy(context);if(read_ok)context_set(previous==context?nullptr:previous);}
  if(driver)FreeLibrary(driver);
 }
};
DriverContext::DriverContext(const std::string& ptx_file,int ordinal):DriverContext(ptx_file,ordinal,"joint_raw_uv"){}
DriverContext::DriverContext(const std::string& ptx_file,int ordinal,const char* kernel_symbol):impl_(new Impl) {
 auto& x=*impl_;x.driver=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
 if(!x.driver)throw std::runtime_error("Joint GPU: cannot load installed NVIDIA system driver");
 x.bind(x.init,"cuInit");x.bind(x.device_get,"cuDeviceGet");x.bind(x.context_create,"cuCtxCreate_v2");x.bind(x.context_set,"cuCtxSetCurrent");x.bind(x.context_get,"cuCtxGetCurrent");x.bind(x.context_destroy,"cuCtxDestroy_v2");x.bind(x.context_sync,"cuCtxSynchronize");x.bind(x.event_create,"cuEventCreate");x.bind(x.event_destroy,"cuEventDestroy_v2");x.bind(x.event_record,"cuEventRecord");x.bind(x.event_elapsed,"cuEventElapsedTime");x.bind(x.module_load,"cuModuleLoadDataEx");x.bind(x.module_unload,"cuModuleUnload");x.bind(x.function_get,"cuModuleGetFunction");x.bind(x.memory_allocate,"cuMemAlloc_v2");x.bind(x.memory_free,"cuMemFree_v2");x.bind(x.copy_htod,"cuMemcpyHtoD_v2");x.bind(x.copy_dtoh,"cuMemcpyDtoH_v2");x.bind(x.launch,"cuLaunchKernel");x.bind(x.error_name,"cuGetErrorName");x.bind(x.error_string,"cuGetErrorString");
 x.check(x.init(0),"initialize");const Impl::SavedContext caller(x);int device=-1;x.check(x.device_get(&device,ordinal),"get device");
 x.check(x.context_create(&x.context,4u,device),"create blocking-sync context");
 x.check(x.event_create(&x.kernel_begin,0),"create begin timing event");x.check(x.event_create(&x.kernel_end,0),"create end timing event");
 const auto ptx=read_ptx(ptx_file);x.check(x.module_load(&x.module,ptx.data(),0,nullptr,nullptr),"load reviewed PTX");if(!kernel_symbol||std::string(kernel_symbol)!="joint_raw_uv"&&std::string(kernel_symbol)!="joint_raw_uv_width_v1")throw std::runtime_error("Joint GPU: unsupported explicit kernel ABI symbol");x.check(x.function_get(&x.kernel,x.module,kernel_symbol),"resolve reviewed raw UV ABI symbol");
}
DriverContext::~DriverContext()=default;
void DriverContext::set_current(){impl_->current();}
void DriverContext::synchronize(){auto&x=*impl_;const Impl::SavedContext caller(x);x.current();x.check(x.context_sync(),"synchronize");}
DevicePtr DriverContext::allocate(std::size_t bytes){if(bytes==0)return 0;auto&x=*impl_;const Impl::SavedContext caller(x);x.current();DevicePtr pointer=0;x.check(x.memory_allocate(&pointer,bytes),"allocate");x.allocations.emplace(pointer,bytes);x.telemetry.allocated_bytes+=bytes;x.telemetry.peak_allocated_bytes=std::max(x.telemetry.peak_allocated_bytes,x.telemetry.allocated_bytes);return pointer;}
void DriverContext::free(DevicePtr pointer){if(!pointer)return;auto&x=*impl_;const Impl::SavedContext caller(x);x.current();if(x.allocations.find(pointer)==x.allocations.end())throw std::runtime_error("Joint GPU: free of unowned buffer");x.check(x.memory_free(pointer),"free");x.telemetry.allocated_bytes-=x.allocations.at(pointer);x.allocations.erase(pointer);}
void DriverContext::upload(DevicePtr pointer,const void* source,std::size_t bytes,std::size_t offset){if(!bytes)return;auto&x=*impl_;x.range(pointer,offset,bytes);if(!source)throw std::runtime_error("Joint GPU: null upload source");const Impl::SavedContext caller(x);x.current();const auto start=Clock::now();x.check(x.copy_htod(pointer+offset,source,bytes),"upload");x.telemetry.h2d_completed_seconds+=elapsed(start);x.telemetry.h2d_bytes+=bytes;++x.telemetry.h2d_calls;}
void DriverContext::download(void* destination,DevicePtr pointer,std::size_t bytes,std::size_t offset){if(!bytes)return;auto&x=*impl_;x.range(pointer,offset,bytes);if(!destination)throw std::runtime_error("Joint GPU: null download destination");const Impl::SavedContext caller(x);x.current();const auto start=Clock::now();x.check(x.copy_dtoh(destination,pointer+offset,bytes),"download");x.telemetry.d2h_completed_seconds+=elapsed(start);x.telemetry.d2h_bytes+=bytes;++x.telemetry.d2h_calls;}
double DriverContext::launch_raw(const GPUMeta& source_meta,const RawArrays& source_arrays,int begin,int end,unsigned block_size){
 if(begin<0||end<begin||block_size==0||block_size>1024)throw std::runtime_error("Joint GPU: invalid raw launch range");
 if(begin==end)return 0.;auto&x=*impl_;const Impl::SavedContext caller(x);x.current();GPUMeta meta=source_meta;RawArrays a=source_arrays;
 void* args[]={&meta,&a.offsets,&a.values,&a.dims,&a.paths,&a.points,&a.rows,&a.block_ids,&a.controls,&a.slots,&a.control_offsets,&a.geometry_indices,&a.layouts,&a.jobs,&begin,&end,&a.output};
 const auto start=std::chrono::steady_clock::now();const unsigned grid=(static_cast<unsigned>(end-begin)+block_size-1)/block_size;
 x.check(x.event_record(x.kernel_begin,nullptr),"record raw UV begin");
 x.check(x.launch(x.kernel,grid,1,1,block_size,1,1,0,nullptr,args,nullptr),"launch raw UV");x.check(x.event_record(x.kernel_end,nullptr),"record raw UV end");
 const auto wait_start=Clock::now();x.check(x.context_sync(),"finish raw UV");x.telemetry.kernel_sync_wait_seconds+=elapsed(wait_start);
 float device_ms=0;x.check(x.event_elapsed(&device_ms,x.kernel_begin,x.kernel_end),"read completed raw UV device time");
 const double seconds=elapsed(start);x.telemetry.kernel_device_seconds+=double(device_ms)*.001;x.telemetry.kernel_completed_seconds+=seconds;++x.telemetry.kernel_calls;return seconds;
}
std::size_t DriverContext::allocated_bytes()const{return impl_->telemetry.allocated_bytes;}
DriverTelemetry DriverContext::telemetry()const{return impl_->telemetry;}
} // namespace rinkfit::joint_gpu
