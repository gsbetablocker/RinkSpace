// FP64 persistent camera.at C ABI. Same exact_math.cuh as validated gpu_full.
#include "exact_math.cuh"
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <memory>
using namespace gpu_exact;
struct Spec {FixedHead<double> head; ZoomOptics<double> optics; int count; double first,dt;};
struct Context {Spec spec; double *controls=nullptr,*world=nullptr,*time=nullptr,*out=nullptr; unsigned char* flags=nullptr; std::uint64_t capacity=0;};
static thread_local std::string error;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void free_buffers(Context* c){cudaFree(c->world);cudaFree(c->time);cudaFree(c->out);cudaFree(c->flags);c->world=c->time=c->out=nullptr;c->flags=nullptr;c->capacity=0;}
static void reserve(Context* c,std::uint64_t n){if(n<=c->capacity)return;free_buffers(c);try{check(cudaMalloc((void**)&c->world,n*3*sizeof(double)));check(cudaMalloc((void**)&c->time,n*sizeof(double)));check(cudaMalloc((void**)&c->out,n*9*sizeof(double)));check(cudaMalloc((void**)&c->flags,n));c->capacity=n;}catch(...){free_buffers(c);throw;}}
extern "C" __declspec(dllexport) const char* rf_gpu_camera_error(){return error.c_str();}
extern "C" __declspec(dllexport) void rf_gpu_camera_destroy(void* raw){auto* c=static_cast<Context*>(raw);if(!c)return;free_buffers(c);cudaFree(c->controls);delete c;}
extern "C" __declspec(dllexport) void* rf_gpu_camera_create(const double* h,const double* o,const double* controls,int count,double first,double dt){
 auto c=std::make_unique<Context>();try{if(count<4||!(dt>0))throw std::domain_error("invalid cubic allocation");auto& s=c->spec;s.count=count;s.first=first;s.dt=dt;
 std::copy(h,h+9,s.head.R0.begin());std::copy(h+9,h+12,s.head.n.begin());std::copy(h+12,h+15,s.head.b.begin());std::copy(h+15,h+18,s.head.P.begin());std::copy(h+18,h+21,s.head.d.begin());s.head.a=h[21];s.head.pupil_linear=h[22];s.head.pupil_quadratic=h[23];
 auto& z=s.optics;z.f_ref=o[0];z.radius_max=o[1];z.cx=o[2];z.cy=o[3];z.eta=o[4];z.cx_linear=o[5];z.cy_linear=o[6];z.k1=o[7];z.k1_linear=o[8];z.k2=o[9];z.k2_linear=o[10];z.p1=o[11];z.p2=o[12];
 if(!camera_pose(s.head,0.,0.,0.).valid())throw std::domain_error("invalid existing fixed head");check(cudaMalloc((void**)&c->controls,3*count*sizeof(double)));check(cudaMemcpy(c->controls,controls,3*count*sizeof(double),cudaMemcpyHostToDevice));return c.release();
 }catch(const std::exception& e){error=e.what();cudaFree(c->controls);return nullptr;}}
__device__ bool evaluate(const Spec& c,const double* controls,const double* X,double t,double* z,unsigned char& valid){
 double x=(t-c.first)/c.dt;int cell=int(floor(x));if(cell==c.count-2)--cell;if(!scalar_finite(t)||cell<1||cell>c.count-3){valid=3;z[0]=t;z[1]=x;z[2]=c.first;z[3]=c.dt;z[4]=c.count;z[5]=cell;return false;}
 auto weights=cubic_weights(x-cell);std::array<double,3> tr{};for(int j=0;j<4;++j)for(int k=0;k<3;++k)tr[k]+=weights[j]*controls[3*(cell-1+j)+k];
 auto optical=zoom_optics(c.optics,tr[2]);auto pose=camera_pose(c.head,tr[0],tr[1],optical.z);if(!pose.valid()){valid=10+(unsigned char)pose.status;return false;}
 auto Y=matvec(pose.R,sub(Vec3<double>{X[0],X[1],X[2]},pose.C));double depth=Y[2];Vec2<double> xy{Y[0]/depth,Y[1]/depth};double r2=xy[0]*xy[0]+xy[1]*xy[1];auto distorted=brown_distort(xy,optical.brown);
 double u=optical.intr.cx+optical.intr.fx*distorted[0],v=optical.intr.cy+optical.intr.fy*distorted[1];auto normal=cross(c.head.n,matvec(transpose(pose.R),c.head.b));double hn=sqrt(dot(normal,normal));
 valid=(depth>0&&finite_array(Y)&&hn>1e-10&&scalar_finite(u)&&scalar_finite(v)&&r2<=c.optics.radius_max*c.optics.radius_max);z[0]=u;z[1]=v;z[2]=depth;z[3]=Y[0];z[4]=Y[1];z[5]=Y[2];z[6]=tr[2];z[7]=optical.z;z[8]=sqrt(r2);return true;}
__global__ void instant(Spec c,const double* controls,std::uint64_t n,const double* X,const double* time,double* out,unsigned char* flags){std::uint64_t i=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned char valid=0;evaluate(c,controls,X+3*i,time[i],out+9*i,valid);flags[i]=valid;}
extern "C" __declspec(dllexport) int rf_gpu_camera_at(void* raw,std::uint64_t n,const double* world,const double* time,double* out,unsigned char* valid){
 try{if(!raw)throw std::runtime_error("retired camera context");if(!n)return 0;auto* c=static_cast<Context*>(raw);reserve(c,n);check(cudaMemcpy(c->world,world,n*3*sizeof(double),cudaMemcpyHostToDevice));check(cudaMemcpy(c->time,time,n*sizeof(double),cudaMemcpyHostToDevice));instant<<<(unsigned int)((n+127)/128),128>>>(c->spec,c->controls,n,c->world,c->time,c->out,c->flags);check(cudaGetLastError());check(cudaMemcpy(out,c->out,n*9*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(valid,c->flags,n,cudaMemcpyDeviceToHost));return 0;}catch(const std::exception& e){error=e.what();return 1;}}
extern "C" __declspec(dllexport) std::uint64_t rf_gpu_camera_capacity(void* raw){return raw?static_cast<Context*>(raw)->capacity:0;}
