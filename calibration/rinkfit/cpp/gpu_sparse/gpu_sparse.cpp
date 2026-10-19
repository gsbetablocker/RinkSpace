// Optional, immutable FP64 CSR SpMV. No optimization, factorization, or model state.
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
void cuda_ok(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(s)); }
void sparse_ok(cusparseStatus_t s) { if (s != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cuSPARSE status " + std::to_string(static_cast<int>(s))); }
struct Matrix {
    cudaStream_t stream{};
    cusparseHandle_t handle{};
    cusparseSpMatDescr_t matrix{};
    cusparseDnVecDescr_t vx{}, vy{};
    int32_t *indptr{}, *indices{};
    double *values{}, *x{}, *y{};
    void* workspace{};
    int64_t rows{}, cols{};
    double alpha{1.0}, beta{0.0};
    ~Matrix() {
        if (stream) cudaStreamSynchronize(stream);
        if (vx) cusparseDestroyDnVec(vx);
        if (vy) cusparseDestroyDnVec(vy);
        if (matrix) cusparseDestroySpMat(matrix);
        if (handle) cusparseDestroy(handle);
        cudaFree(workspace); cudaFree(y); cudaFree(x);
        cudaFree(values); cudaFree(indices); cudaFree(indptr);
        if (stream) cudaStreamDestroy(stream);
    }
};
template<class T> void alloc(T*& p, size_t n) { if (n) cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
}
#define EXPORT extern "C" __declspec(dllexport)
EXPORT int rinkfit_gpu_sparse_abi() noexcept { return 1; }
EXPORT const char* rinkfit_gpu_sparse_error() noexcept { return last_error.c_str(); }
EXPORT int rinkfit_gpu_sparse_create(int64_t rows, int64_t cols, int64_t nnz,
    const int32_t* indptr, const int32_t* indices, const double* values, void** result) noexcept {
    if (result) *result = nullptr;
    try {
        last_error.clear();
        constexpr int64_t imax = std::numeric_limits<int32_t>::max();
        if (!result || rows <= 0 || cols <= 0 || rows > imax || cols > imax || nnz < 0 || nnz > imax || !indptr || (nnz && (!indices || !values)))
            throw std::runtime_error("Invalid FP64/CSR32 input dimensions or pointers");
        auto p = std::make_unique<Matrix>(); p->rows=rows; p->cols=cols;
        cuda_ok(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
        sparse_ok(cusparseCreate(&p->handle)); sparse_ok(cusparseSetStream(p->handle,p->stream));
        alloc(p->indptr,static_cast<size_t>(rows+1)); alloc(p->indices,static_cast<size_t>(nnz)); alloc(p->values,static_cast<size_t>(nnz));
        alloc(p->x,static_cast<size_t>(cols)); alloc(p->y,static_cast<size_t>(rows));
        cuda_ok(cudaMemcpyAsync(p->indptr,indptr,(rows+1)*sizeof(int32_t),cudaMemcpyHostToDevice,p->stream));
        if (nnz) {
            cuda_ok(cudaMemcpyAsync(p->indices,indices,nnz*sizeof(int32_t),cudaMemcpyHostToDevice,p->stream));
            cuda_ok(cudaMemcpyAsync(p->values,values,nnz*sizeof(double),cudaMemcpyHostToDevice,p->stream));
        }
        sparse_ok(cusparseCreateCsr(&p->matrix,rows,cols,nnz,p->indptr,p->indices,p->values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
        sparse_ok(cusparseCreateDnVec(&p->vx,cols,p->x,CUDA_R_64F));
        sparse_ok(cusparseCreateDnVec(&p->vy,rows,p->y,CUDA_R_64F));
        size_t bytes{};
        sparse_ok(cusparseSpMV_bufferSize(p->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&p->alpha,p->matrix,p->vx,&p->beta,p->vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,&bytes));
        if (bytes) cuda_ok(cudaMalloc(&p->workspace,bytes));
        sparse_ok(cusparseSpMV_preprocess(p->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&p->alpha,p->matrix,p->vx,&p->beta,p->vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,p->workspace));
        cuda_ok(cudaStreamSynchronize(p->stream)); *result=p.release(); return 0;
    } catch(const std::exception& e) { last_error=e.what(); return 1; }
    catch(...) { last_error="Unknown GPU sparse creation error"; return 1; }
}
EXPORT int rinkfit_gpu_sparse_matvec(void* handle, const double* x, double* y) noexcept {
    try {
        last_error.clear(); auto* p=static_cast<Matrix*>(handle);
        if (!p || !x || !y) throw std::runtime_error("Null GPU sparse handle/vector");
        cuda_ok(cudaMemcpyAsync(p->x,x,p->cols*sizeof(double),cudaMemcpyHostToDevice,p->stream));
        sparse_ok(cusparseSpMV(p->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&p->alpha,p->matrix,p->vx,&p->beta,p->vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,p->workspace));
        cuda_ok(cudaMemcpyAsync(y,p->y,p->rows*sizeof(double),cudaMemcpyDeviceToHost,p->stream));
        cuda_ok(cudaStreamSynchronize(p->stream)); return 0;
    } catch(const std::exception& e) { last_error=e.what(); return 1; }
    catch(...) { last_error="Unknown GPU sparse matvec error"; return 1; }
}
EXPORT void rinkfit_gpu_sparse_destroy(void* handle) noexcept { delete static_cast<Matrix*>(handle); }
