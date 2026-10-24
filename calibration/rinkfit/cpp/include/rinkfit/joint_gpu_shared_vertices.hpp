#pragma once
#include "rinkfit/joint_gpu_shared_shape.hpp"
namespace rinkfit::joint_gpu {
inline constexpr const char* shared_vertices_kernel_abi="shared_print_vertices_v2";
inline constexpr const char* shared_vertices_kernel_symbol="joint_raw_uv_shared_vertices_v2";
struct UVLayoutSharedVerticesV2 {int global,geom,ad,ref,path,block_begin,block_count,control_begin,control_count,geometry_begin,geometry_count,geometry_offset,width_slot,width_sign,shape_segment,shape_vertex0,shape_vertex1;};
static_assert(sizeof(UVLayoutSharedVerticesV2)==68,"local shared vertex layout ABI");
}
namespace rinkfit {
std::shared_ptr<JointUVCache>make_joint_gpu_shared_vertices_uv_cache(JointProblemBuilder&,double*,const std::string&,int,const std::filesystem::path&);
}
