#pragma once
#include "rinkfit/joint_uv_cache.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace rinkfit {
// The cache and all Ceres callbacks retain their addresses. Call only outside
// Solve/Evaluate; source branch changes and graph edits remain a caller transaction.
struct JointGPULocalRowRebindV1 {
 const void* owner=nullptr;std::uint64_t token=0;bool active=false;eventio::Json receipt;
};
// Excludes detached row jobs before launch, rather than evaluating an old cell
// and discarding its output. Each row is {first_job,job_count,enabled}.
inline std::vector<std::array<std::size_t,2>> joint_gpu_live_job_ranges_v1(const std::vector<std::array<std::size_t,3>>&rows) {
 std::vector<std::array<std::size_t,2>>out;std::size_t previous=0;bool first=true;
 for(const auto&r:rows){if((!first&&r[0]<previous)||r[1]>std::numeric_limits<std::size_t>::max()-r[0])throw std::runtime_error("overlapping/overflowed GPU row job range");first=false;previous=r[0]+r[1];if(!r[2]||!r[1])continue;if(!out.empty()&&out.back()[1]==r[0])out.back()[1]=previous;else out.push_back({r[0],previous});}
 return out;
}
JointGPULocalRowRebindV1 begin_joint_gpu_local_row_rebind_v1(ceres::EvaluationCallback*,const std::vector<std::size_t>&raw_rows,bool outside_evaluation);
eventio::Json patch_joint_gpu_local_row_rebind_v1(ceres::EvaluationCallback*,JointGPULocalRowRebindV1&,bool outside_evaluation);
eventio::Json finish_joint_gpu_local_row_rebind_v1(ceres::EvaluationCallback*,JointGPULocalRowRebindV1&,bool accepted,bool outside_evaluation);
// This is caller-measured affected residual CPU wall time, not a GPU timing estimate.
void note_joint_gpu_local_fallback_cpu_seconds_v1(ceres::EvaluationCallback*,double seconds,std::size_t evaluations);
}
