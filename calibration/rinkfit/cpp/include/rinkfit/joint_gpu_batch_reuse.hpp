#pragma once
#include "rinkfit/joint_gpu_cache.hpp"
namespace rinkfit {
// Opt-in one-slot per-shot raw UV/partial reuse. All layout dependencies are byte exact;
// alpha, budget, Query/PHR and native-domain consumers remain current on every point.
std::shared_ptr<JointUVCache>make_joint_gpu_batch_reuse_uv_cache(JointProblemBuilder&,double*budget,const std::string&ptx,int threads,const std::filesystem::path&log_file,const std::string&kernel_abi);
}
