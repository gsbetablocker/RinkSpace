#pragma once
#include "rinkfit/joint_gpu_line_width.hpp"
namespace rinkfit::joint_gpu {
inline constexpr const char* shared_shape_kernel_abi="shared_print_shape_v1";
inline constexpr const char* shared_shape_kernel_symbol="joint_raw_uv_shared_shape_v1";
// Same wire widths, versioned kind6 path semantics and -2..-5 derivative jobs.
inline std::array<int,4> shared_shape_selected_indices_v1(const JointEventModel&m,const JointRowLayout&l,double station){
 const auto&p=m.materials.at(l.material).paths.at(l.path);
 if(p.kind!=6||l.geometry_slot<0||!std::isfinite(station)||station<0||station>1)throw std::runtime_error("invalid shared shape selection");
 const int count=l.geometry_indices.empty()?int(m.blocks.at(l.block_ids.at(l.geometry_slot)).values.size())-l.geometry_offset:int(l.geometry_indices.size());
 if(count<2||count%2)throw std::runtime_error("invalid shared shape dimension");
 const int n=count/2,segments=p.closed?n:n-1,k=n==1?0:std::min(segments-1,int(std::floor(station*segments))),next=n==1?0:(k+1)%n;
 std::array<int,4>result{2*k,2*k+1,2*next,2*next+1};
 for(auto&index:result)index=l.geometry_indices.empty()?l.geometry_offset+index:l.geometry_indices.at(index);
 return result;
}
}
namespace rinkfit {
std::shared_ptr<JointUVCache>make_joint_gpu_shared_shape_uv_cache(JointProblemBuilder&,double*,const std::string&,int,const std::filesystem::path&);
}
