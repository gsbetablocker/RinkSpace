#pragma once
#include "rinkfit/joint_uv_cache.hpp"
#include "rinkfit/joint_builder.hpp"
namespace rinkfit {
std::shared_ptr<JointUVCache>make_joint_gpu_uv_cache(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file);
std::shared_ptr<JointUVCache>make_joint_source_uv_capture_gpu_cache_v2(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file,const std::string&abi);
std::shared_ptr<JointUVCache>make_joint_source_uv_capture_gpu_cache_v1(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file,const std::string&abi);
}
