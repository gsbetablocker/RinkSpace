// Exact current full-W residual/block-Jacobian assembly, independent C ABI.
// No scene, camera, robust loss, covariance estimation, or solver dependencies.
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <omp.h>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
struct WhiteningInput {
 int32_t abi_version,nblocks,nraw,nstat;
 int64_t raw_edges,stat_edges,nnz,raw_jac_values,stat_jac_values;
 const int32_t *block_sizes;
 const int64_t *raw_edge_offsets;const int32_t *raw_edge_blocks;const int64_t *raw_jac_offsets;
 const int64_t *stat_edge_offsets;const int32_t *stat_edge_blocks;const int64_t *stat_jac_offsets;
 const int64_t *stat_residual_offsets;const int32_t *raw_stat_owner;
 const int64_t *csr_indptr,*csr_indices;const double *csr_values;
};
struct Plan {
 int32_t nraw,nstat;int64_t raw_jac_values,stat_jac_values;
 std::vector<int32_t> sizes,raw_blocks;
 std::vector<int64_t> raw_edge_offsets,raw_jac_offsets,stat_edge_offsets,stat_jac_offsets,stat_residual_offsets;
 std::vector<int64_t> indptr,indices,destination_base;
 std::vector<double> weights;
};
static void error(char* buffer,int64_t n,const char* message){if(buffer&&n>0)std::snprintf(buffer,(size_t)n,"%s",message);}
template<class T> static std::vector<T> copy(const T* p,int64_t n){
 if(n<0||(!p&&n))throw std::runtime_error("null/current array length");
 return n?std::vector<T>(p,p+n):std::vector<T>();
}
API int32_t wh_abi_version(){return 1;}
API void* wh_create(const WhiteningInput* in,char* err,int64_t err_size){
 try {
  if(!in||in->abi_version!=1||in->nblocks<=0||in->nraw<=0||in->nstat<=0)throw std::runtime_error("invalid whitening ABI or layout counts");
  Plan p;p.nraw=in->nraw;p.nstat=in->nstat;p.raw_jac_values=in->raw_jac_values;p.stat_jac_values=in->stat_jac_values;
  p.sizes=copy(in->block_sizes,in->nblocks);p.raw_blocks=copy(in->raw_edge_blocks,in->raw_edges);
  p.raw_edge_offsets=copy(in->raw_edge_offsets,in->nraw+1);p.raw_jac_offsets=copy(in->raw_jac_offsets,in->raw_edges+1);
  p.stat_edge_offsets=copy(in->stat_edge_offsets,in->nstat+1);p.stat_jac_offsets=copy(in->stat_jac_offsets,in->stat_edges+1);
  p.stat_residual_offsets=copy(in->stat_residual_offsets,in->nstat+1);
  auto sb=copy(in->stat_edge_blocks,in->stat_edges);auto owner=copy(in->raw_stat_owner,in->nraw);
  if(p.raw_edge_offsets.front()!=0||p.raw_edge_offsets.back()!=in->raw_edges||p.stat_edge_offsets.front()!=0||p.stat_edge_offsets.back()!=in->stat_edges||p.raw_jac_offsets.front()!=0||p.raw_jac_offsets.back()!=p.raw_jac_values||p.stat_jac_offsets.front()!=0||p.stat_jac_offsets.back()!=p.stat_jac_values||p.stat_residual_offsets.front()!=0)throw std::runtime_error("layout terminal offsets differ");
  auto offsets=[](const std::vector<int64_t>& a,int64_t end,const char* label){
   if(a.empty()||a.front()!=0||a.back()!=end)throw std::runtime_error(label);
   for(size_t k=1;k<a.size();++k)if(a[k]<a[k-1]||a[k]<0||a[k]>end)throw std::runtime_error(label);
  };
  offsets(p.raw_edge_offsets,in->raw_edges,"raw edge offset range invalid");
  offsets(p.stat_edge_offsets,in->stat_edges,"statistical edge offset range invalid");
  offsets(p.raw_jac_offsets,p.raw_jac_values,"raw Jacobian offset range invalid");
  offsets(p.stat_jac_offsets,p.stat_jac_values,"statistical Jacobian offset range invalid");
  int64_t residuals=p.stat_residual_offsets.back();
  offsets(p.stat_residual_offsets,residuals,"statistical residual offset range invalid");
  p.indptr=copy(in->csr_indptr,residuals+1);p.indices=copy(in->csr_indices,in->nnz);p.weights=copy(in->csr_values,in->nnz);
  offsets(p.indptr,in->nnz,"W CSR offset range invalid");
  p.destination_base.assign((size_t)in->raw_edges,-1);
  for(int b:p.sizes)if(b<=0)throw std::runtime_error("nonpositive parameter block size");
  for(int64_t f=0;f<in->nstat;++f){
   if(p.stat_edge_offsets[f]>p.stat_edge_offsets[f+1]||p.stat_residual_offsets[f]>=p.stat_residual_offsets[f+1])throw std::runtime_error("statistical factor offsets invalid");
   int64_t dim=p.stat_residual_offsets[f+1]-p.stat_residual_offsets[f];
   for(int64_t e=p.stat_edge_offsets[f];e<p.stat_edge_offsets[f+1];++e){int b=sb[e];
    if(b<0||b>=in->nblocks||p.stat_jac_offsets[e+1]-p.stat_jac_offsets[e]!=dim*p.sizes[b])throw std::runtime_error("statistical Jacobian block dimension mismatch");
   }
  }
  for(int f=0;f<in->nraw;++f){
   if(p.raw_edge_offsets[f]>p.raw_edge_offsets[f+1])throw std::runtime_error("raw factor edge offsets invalid");
   int stat=owner[f];if(stat<-1||stat>=in->nstat)throw std::runtime_error("raw source statistical owner invalid");
   for(int64_t e=p.raw_edge_offsets[f];e<p.raw_edge_offsets[f+1];++e){int b=p.raw_blocks[e];
    if(b<0||b>=in->nblocks||p.raw_jac_offsets[e+1]-p.raw_jac_offsets[e]!=2*p.sizes[b])throw std::runtime_error("raw Jacobian block dimension mismatch");
    if(stat<0)continue;
    bool found=false;
    for(int64_t se=p.stat_edge_offsets[stat];se<p.stat_edge_offsets[stat+1];++se)if(sb[se]==b){p.destination_base[e]=p.stat_jac_offsets[se];found=true;break;}
    if(!found)throw std::runtime_error("statistical factor omits a raw dependency block");
   }
  }
  for(int64_t r=0,f=0;r<residuals;++r){
   while(r>=p.stat_residual_offsets[f+1])++f;
   if(p.indptr[r]>p.indptr[r+1])throw std::runtime_error("W CSR offsets not increasing");
   for(int64_t k=p.indptr[r];k<p.indptr[r+1];++k){auto c=p.indices[k];
    if(c<0||c>=2*int64_t(in->nraw)||owner[c/2]!=f||!std::isfinite(p.weights[k]))throw std::runtime_error("full W source owner or actual coefficient invalid");
   }
  }
  error(err,err_size,"");return new Plan(std::move(p));
 } catch(const std::exception& e){error(err,err_size,e.what());return nullptr;}
}
API void wh_destroy(void* pointer){delete static_cast<Plan*>(pointer);}
API int32_t wh_apply(void* pointer,const double* raw_r,const double* raw_j,double* out_r,double* out_j,int32_t threads,char* err,int64_t err_size){
 if(!pointer||!raw_r||!out_r||threads<=0||(out_j&&!raw_j)){error(err,err_size,"invalid whitening evaluation buffers");return -1;}
 const auto& p=*static_cast<const Plan*>(pointer);int invalid=0;
 #pragma omp parallel for schedule(dynamic,32) num_threads(threads) reduction(|:invalid)
 for(int f=0;f<p.nstat;++f){
  const int64_t first=p.stat_residual_offsets[f],last=p.stat_residual_offsets[f+1];
  int64_t jb=p.stat_jac_offsets[p.stat_edge_offsets[f]],je=p.stat_jac_offsets[p.stat_edge_offsets[f+1]];
  if(out_j)std::fill(out_j+jb,out_j+je,0.0);
  for(int64_t row=first;row<last;++row){
   double value=0.0;int64_t local_row=row-first;
   for(int64_t k=p.indptr[row];k<p.indptr[row+1];++k){
    int64_t column=p.indices[k],source=column/2,component=column%2;double w=p.weights[k];value+=w*raw_r[column];
    if(out_j){
     for(int64_t e=p.raw_edge_offsets[source];e<p.raw_edge_offsets[source+1];++e){
      int width=p.sizes[p.raw_blocks[e]];int64_t dst=p.destination_base[e]+local_row*width,src=p.raw_jac_offsets[e]+component*width;
      for(int j=0;j<width;++j)out_j[dst+j]+=w*raw_j[src+j];
     }
    }
   }
   out_r[row]=value;if(!std::isfinite(value))invalid=1;
  }
  if(out_j)for(int64_t k=jb;k<je;++k)if(!std::isfinite(out_j[k])){invalid=1;break;}
 }
 if(invalid){error(err,err_size,"nonfinite active source residual/Jacobian; reject current evaluation");return -2;}
 error(err,err_size,"");return 0;
}
