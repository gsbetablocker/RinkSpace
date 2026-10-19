#include "abi.hpp"
#include "exact_math.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
namespace ge=gpu_exact;
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(cudaError_t e){if(e!=cudaSuccess)throw e;}
template<class T>void alloc(T*&p,std::size_t n){if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),sizeof(T)*n));}
template<class T>void upload(T*d,const T*h,std::size_t n){if(n)check(cudaMemcpy(d,h,sizeof(T)*n,cudaMemcpyHostToDevice));}
struct Context {
 GPUMeta meta{};int nb=0,nv=0,n=0,nj=0;int *off=nullptr,*dims=nullptr,*bids=nullptr,*ci=nullptr,*cs=nullptr;
 double *values=nullptr,*points=nullptr;GPUPath*paths=nullptr;ge::Path*adapted=nullptr;UVRow*rows=nullptr;UVLayout*layout=nullptr;UVJob*jobs=nullptr;UVPartial*out=nullptr;cudaEvent_t a=nullptr,b=nullptr;
 ~Context(){cudaFree(off);cudaFree(dims);cudaFree(bids);cudaFree(ci);cudaFree(cs);cudaFree(values);cudaFree(points);cudaFree(paths);cudaFree(adapted);cudaFree(rows);cudaFree(layout);cudaFree(jobs);cudaFree(out);if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}
};
__device__ ge::Model make_model(const GPUMeta&m,const ge::Path*paths){ge::Model x{};for(int i=0;i<9;i++)x.R0_seed[i]=m.R0[i];for(int i=0;i<3;i++){x.n_seed[i]=m.n[i];x.b_seed[i]=m.b[i];x.n_tangent1[i]=m.t1[i];x.n_tangent2[i]=m.t2[i];}for(int i=0;i<4;i++){x.reference_indices[i]=m.ref[i];x.reference_weights[i]=m.refw[i];}x.f_ref=m.fref;x.radius_max=m.Rmax;x.dt=m.dt;x.first_control_time=m.first;x.n_controls=m.nc;x.control_block_begin=m.cb;x.dependent_control=m.dep;x.paths=paths;return x;}
__global__ void adapt_paths(const GPUPath*in,ge::Path*out,int np,const double*points){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=np)return;const auto&p=in[i];ge::Path q{};q.kind=p.kind;q.radius_index=p.ri;q.boundary_segment=p.bs;q.start_angle=p.a0;q.end_angle=p.a1;q.fixed_height_fraction=p.h;q.points={reinterpret_cast<const ge::Vec2<double>*>(points+2*p.begin),p.count};out[i]=q;}
__global__ void uv_kernel(GPUMeta meta,const int*offsets,const double*values,const int*dims,const ge::Path*paths,const UVRow*rows,const int*block_ids,const int*controls,const int*slots,const UVLayout*layout,const UVJob*jobs,int begin,int end,UVPartial*out){
 int index=begin+blockIdx.x*blockDim.x+threadIdx.x;if(index>=end)return;const auto job=jobs[index];const auto row=rows[job.row];const auto l=layout[job.row];ge::Jet storage[160];const ge::Jet*ptrs[32];int cursor=0;
 for(int k=0;k<l.block_count;k++){int b=block_ids[l.block_begin+k];ptrs[k]=storage+cursor;for(int j=0;j<dims[b];j++){storage[cursor]=ge::Jet(values[offsets[b]+j],cursor==job.column?1.:0.);++cursor;}}
 // Register the complete four-auxiliary column ABI: station, RS row, alpha, b.
 // UV has no alpha/b dependency; exact Jet evaluation proves those zeros.
 for(int k=0;k<4;k++){ptrs[l.block_count+k]=storage+cursor;storage[cursor]=ge::Jet(k==0?row.s:k==1?row.v:k==2?row.alpha:row.budget,cursor==job.column?1.:0.);++cursor;}
 ge::Layout local{};local.global_slot=l.global;local.geometry_slot=l.geom;local.advertisement_slot=l.ad;local.flatpath=l.path;local.frame_time=row.t;local.control_ids={controls+l.control_begin,l.control_count};local.control_slots={slots+l.control_begin,l.control_count};
 const auto model=make_model(meta,paths);const auto p=model.predict_for_al(local,ptrs[l.block_count][0],ptrs[l.block_count+1][0],ptrs);UVPartial r{};r.arithmetic=p.arithmetic_valid;r.status=int(p.physical_status);if(p.arithmetic_valid){r.u=p.uv[0].v;r.v=p.uv[1].v;if(!ge::isfinite(p.uv[0])||!ge::isfinite(p.uv[1]))r.arithmetic=0;}out[index]=r;
}
}
extern "C" int rf_uv_create(const GPUMeta*m,const int*off,const int*dims,int nb,int nv,const GPUPath*paths,int np,const double*points,int npoints,const int*bids,int nbids,const int*ci,const int*cs,int ncontrols,const UVLayout*layouts,int n,const UVJob*jobs,int nj,void**handle,UVTiming*timing,char*error,int size){
 auto start=Clock::now();Context*c=nullptr;*handle=nullptr;try{c=new Context;c->meta=*m;c->nb=nb;c->nv=nv;c->n=n;c->nj=nj;check(cudaEventCreate(&c->a));check(cudaEventCreate(&c->b));alloc(c->off,nb);alloc(c->dims,nb);alloc(c->values,nv);alloc(c->points,npoints);alloc(c->paths,np);alloc(c->adapted,np);alloc(c->rows,n);alloc(c->bids,nbids);alloc(c->ci,ncontrols);alloc(c->cs,ncontrols);alloc(c->layout,n);alloc(c->jobs,nj);alloc(c->out,nj);
 upload(c->off,off,nb);upload(c->dims,dims,nb);upload(c->points,points,npoints);upload(c->paths,paths,np);upload(c->bids,bids,nbids);upload(c->ci,ci,ncontrols);upload(c->cs,cs,ncontrols);upload(c->layout,layouts,n);upload(c->jobs,jobs,nj);adapt_paths<<<(np+127)/128,128>>>(c->paths,c->adapted,np,c->points);check(cudaGetLastError());check(cudaDeviceSynchronize());timing->setup_ms=elapsed(start);*handle=c;return 0;
 }catch(cudaError_t e){snprintf(error,size,"CUDA row UV setup: %s",cudaGetErrorString(e));delete c;return int(e);}catch(const std::exception&e){snprintf(error,size,"%s",e.what());delete c;return -1;}
}
extern "C" int rf_uv_evaluate(void*handle,const double*values,int nv,const UVRow*rows,int n,UVPartial*out,int nj,UVTiming*timing,char*error,int size){
 const auto start=Clock::now();auto*c=static_cast<Context*>(handle);try{if(!c||nv!=c->nv||n!=c->n||nj!=c->nj){snprintf(error,size,"CUDA row UV dimensions changed; rebuild cache");return -1;}
 timing->batches=0;check(cudaEventRecord(c->a));upload(c->values,values,nv);upload(c->rows,rows,n);check(cudaEventRecord(c->b));check(cudaEventSynchronize(c->b));check(cudaEventElapsedTime(&timing->h2d_ms,c->a,c->b));check(cudaEventRecord(c->a));
 for(int begin=0;begin<nj;begin+=131072){int end=std::min(nj,begin+131072);uv_kernel<<<(end-begin+127)/128,128>>>(c->meta,c->off,c->values,c->dims,c->adapted,c->rows,c->bids,c->ci,c->cs,c->layout,c->jobs,begin,end,c->out);++timing->batches;}
 check(cudaGetLastError());check(cudaEventRecord(c->b));check(cudaEventSynchronize(c->b));check(cudaEventElapsedTime(&timing->kernel_ms,c->a,c->b));check(cudaEventRecord(c->a));check(cudaMemcpy(out,c->out,sizeof(UVPartial)*nj,cudaMemcpyDeviceToHost));check(cudaEventRecord(c->b));check(cudaEventSynchronize(c->b));check(cudaEventElapsedTime(&timing->d2h_ms,c->a,c->b));timing->total_ms=elapsed(start);return 0;
 }catch(cudaError_t e){snprintf(error,size,"CUDA row UV evaluation: %s",cudaGetErrorString(e));return int(e);}
}
extern "C" void rf_uv_destroy(void*handle){delete static_cast<Context*>(handle);}
