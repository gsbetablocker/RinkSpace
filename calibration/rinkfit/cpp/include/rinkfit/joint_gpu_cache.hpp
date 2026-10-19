#pragma once
#include "rinkfit/joint_uv_cache.hpp"
#include "rinkfit/joint_builder.hpp"
namespace rinkfit {
std::shared_ptr<JointUVCache>make_joint_gpu_uv_cache(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file);
}
