#pragma once
#include "rinkfit/joint_engineering_outer.hpp"
#include <ceres/types.h>

namespace rinkfit {
// A termination boundary is a safety filter, not an inexact-AL stationarity
// certificate. In particular, CONVERGENCE alone does not prove that certificate.
struct EngineeringInnerBoundary {
 bool failed=false,stopped=false,budget_exhausted=false;
 bool permits_dual_update=false;
 const char* status="incomplete_inner";
};
inline EngineeringInnerBoundary engineering_inner_boundary(ceres::TerminationType termination,
 bool callback_failed,bool requested_stop){
 EngineeringInnerBoundary b;
 b.failed=callback_failed||termination==ceres::FAILURE||termination==ceres::USER_FAILURE;
 b.stopped=requested_stop||termination==ceres::USER_SUCCESS;
 b.budget_exhausted=termination==ceres::NO_CONVERGENCE;
 b.permits_dual_update=termination==ceres::CONVERGENCE&&!b.failed&&!b.stopped;
 b.status=b.failed?"failed":b.stopped?"stopped":b.budget_exhausted?"inner_budget_exhausted":b.permits_dual_update?"converged_termination_only":"incomplete_inner";
 return b;
}
struct EngineeringInnerTransition {
 bool dual_update_applied=false,penalty_limit_reached=false;
};
inline EngineeringInnerTransition finish_engineering_inner(EngineeringOuterState& outer,
 const EngineeringInnerBoundary& boundary,const std::vector<double>&g,const std::vector<double>&h,
 double rho_max,double improvement_ratio,double penalty_factor){
 outer.previous_inner_incomplete=!boundary.permits_dual_update;
 if(!boundary.permits_dual_update)return {};
 auto next=update_phr(g,h,outer.lambda,outer.nu,outer.rho,outer.previous_progress,
                      rho_max,improvement_ratio,penalty_factor);
 outer.lambda=std::move(next.lambda);outer.nu=std::move(next.nu);
 outer.previous_progress=next.progress;outer.rho=next.next_rho;++outer.outer_iteration;
 return {true,next.penalty_limit_reached};
}
}
