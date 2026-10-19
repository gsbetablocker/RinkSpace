// SciPy-compatible Gaussian reflect and cubic-spline mirror prefilter.
// Arithmetic follows scipy.ndimage 1.17.1 ni_splines.c and ni_filters.c.
// SciPy BSD-3-Clause attribution is retained in SCIPY_LICENSE.txt.
#include <cuda_runtime.h>
#include <cmath>
#include <string>
static thread_local std::string last_error;
struct Context {int h,w,batch;float *src,*tmp,*g;double *c;};
static bool ok(cudaError_t e){if(e!=cudaSuccess){last_error=cudaGetErrorString(e);return false;}return true;}
__device__ int reflect(int i,int n){while(i<0||i>=n){if(i<0)i=-i-1;else i=2*n-i-1;}return i;}
__global__ void gauss(const float* a,float* b,int h,int w,int count,int axis,double k0,double k1,double k2,double k3){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;int y=(p/w)%h,x=p%w;double t=double(a[p])*k0;
 for(int j=3;j>=1;--j){int lo=axis==0?p+(reflect(y-j,h)-y)*w:p+reflect(x-j,w)-x;int hi=axis==0?p+(reflect(y+j,h)-y)*w:p+reflect(x+j,w)-x;double k=j==1?k1:(j==2?k2:k3);t+=(double(a[lo])+double(a[hi]))*k;}
 b[p]=float(t);
}
__global__ void convert(const float* a,double* b,int count){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)b[p]=double(a[p]);}
__global__ void spline(double* c,int h,int w,int batch,int axis,double z,double gain){
 int line=blockIdx.x*blockDim.x+threadIdx.x,lines=batch*(axis==0?w:h);if(line>=lines)return;int image=line/(axis==0?w:h),q=line%(axis==0?w:h),n=axis==0?h:w,stride=axis==0?w:1;double* a=c+image*h*w+(axis==0?q:q*w);
 for(int i=0;i<n;++i)a[i*stride]*=gain;
 double zi=z,zn1=pow(z,n-1);a[0]=a[0]+zn1*a[(n-1)*stride];
 for(int i=1;i<n-1;++i){a[0]+=zi*(a[i*stride]+zn1*a[(n-1-i)*stride]);zi*=z;}
 a[0]/=1-zn1*zn1;
 for(int i=1;i<n;++i)a[i*stride]+=z*a[(i-1)*stride];
 a[(n-1)*stride]=(z*a[(n-2)*stride]+a[(n-1)*stride])*z/(z*z-1);
 for(int i=n-2;i>=0;--i)a[i*stride]=z*(a[(i+1)*stride]-a[i*stride]);
}
extern "C" __declspec(dllexport) const char* rf_source_filter_error(){return last_error.c_str();}
extern "C" __declspec(dllexport) void* rf_source_filter_create(int h,int w,int batch){
 if(h<2||w<2||batch<1){last_error="dimensions must exceed one";return nullptr;}Context* c=new Context{h,w,batch,nullptr,nullptr,nullptr,nullptr};size_t n=size_t(h)*w*batch;
 if(!ok(cudaMalloc(&c->src,n*sizeof(float)))||!ok(cudaMalloc(&c->tmp,n*sizeof(float)))||!ok(cudaMalloc(&c->g,n*sizeof(float)))||!ok(cudaMalloc(&c->c,n*sizeof(double)))){cudaFree(c->src);cudaFree(c->tmp);cudaFree(c->g);cudaFree(c->c);delete c;return nullptr;}return c;
}
extern "C" __declspec(dllexport) int rf_source_filter_apply(void* v,const float* src,float* gaussian,double* coefficients,int batch,const double* kernel){
 Context* c=static_cast<Context*>(v);if(!c||batch<1||batch>c->batch){last_error="invalid batch/context";return 1;}int count=c->h*c->w*batch,blocks=(count+255)/256;double z=-0.267949192431122706472553658494127633;double gain=(1-z)*(1-1/z);
 if(!ok(cudaMemcpy(c->src,src,size_t(count)*sizeof(float),cudaMemcpyHostToDevice)))return 2;
 gauss<<<blocks,256>>>(c->src,c->tmp,c->h,c->w,count,0,kernel[3],kernel[4],kernel[5],kernel[6]);
 gauss<<<blocks,256>>>(c->tmp,c->g,c->h,c->w,count,1,kernel[3],kernel[4],kernel[5],kernel[6]);
 convert<<<blocks,256>>>(c->g,c->c,count);
 spline<<<(batch*c->w+127)/128,128>>>(c->c,c->h,c->w,batch,0,z,gain);
 spline<<<(batch*c->h+127)/128,128>>>(c->c,c->h,c->w,batch,1,z,gain);
 if(!ok(cudaGetLastError())||!ok(cudaMemcpy(gaussian,c->g,size_t(count)*sizeof(float),cudaMemcpyDeviceToHost))||!ok(cudaMemcpy(coefficients,c->c,size_t(count)*sizeof(double),cudaMemcpyDeviceToHost)))return 3;return 0;
}
extern "C" __declspec(dllexport) void rf_source_filter_destroy(void* v){Context* c=static_cast<Context*>(v);if(c){cudaFree(c->src);cudaFree(c->tmp);cudaFree(c->g);cudaFree(c->c);delete c;}}
