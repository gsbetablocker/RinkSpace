#pragma once
#include "rinkfit/joint_gpu_driver.hpp"
#include "rinkfit/joint_gpu_cache.hpp"
namespace rinkfit::joint_gpu {
// Explicit successor ABI. GPUMeta304 and all old structs remain unchanged.
struct UVLayoutWidthV1 {int global,geom,ad,ref,path,block_begin,block_count,control_begin,control_count,geometry_begin,geometry_count,geometry_offset,width_slot,width_sign;};
static_assert(sizeof(UVLayoutWidthV1)==56&&sizeof(GPUMeta)==304,"Line-width raw CUDA ABI differs");
inline constexpr const char* ice_line_width_kernel_abi="ice_line_width_v1";
inline constexpr const char* ice_line_width_kernel_symbol="joint_raw_uv_width_v1";
}
namespace rinkfit {
std::shared_ptr<JointUVCache>make_joint_gpu_width_uv_cache(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file);
}
