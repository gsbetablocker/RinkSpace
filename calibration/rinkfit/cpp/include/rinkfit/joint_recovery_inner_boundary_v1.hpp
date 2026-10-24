#pragma once
#include <ceres/ceres.h>
namespace rinkfit {
enum class JointRecoveryInnerBoundaryV1 { kCoordinateRechart, kNumericalWorkingSet };
// An accepted physical update can reverse temporary face multipliers while
// the current reduced problem is still making progress. Reconsider those
// faces after its convergence; coordinate validity changes need a safe boundary.
inline bool joint_recovery_interrupt_after_accepted_step_v1(JointRecoveryInnerBoundaryV1 kind,const ceres::IterationSummary&s){
 return kind==JointRecoveryInnerBoundaryV1::kCoordinateRechart&&s.iteration>0&&s.step_is_successful;
}
}
