#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include "rinkfit/joint_observed_branch_restart_v1.hpp"
#include "rinkfit/joint_gpu_local_rebind_v1.hpp"
#include <functional>
#include <optional>
namespace rinkfit {
// Only same-source, same-path split-vertex cell changes enter this transaction.
// The issued W group closure and every original row domain are retained. The
// surrounding problem, global PHR, motion factors and GPU callback stay alive.
inline eventio::Json transact_joint_observed_local_cells_v1(JointALBuilder&b,ceres::Problem&problem,BuildInventory&inv,JointObservedBranchPlanV1&plan,const JointObservedLocalClosureV1&closure,JointCompactAcceptedState&accepted,int threads,const std::filesystem::path&out,const std::function<eventio::Json()>&debt,const std::function<std::function<void()>(const std::vector<std::pair<ceres::ResidualBlockId,ceres::ResidualBlockId>>&)>&prepare_remap,bool removal_incidence_enabled){
 using Json=eventio::Json;std::filesystem::create_directories(out);accepted.capture();const auto before_token=accepted.token();auto cost=[&](){if(auto*c=b.gpu_evaluation_callback())c->PrepareForEvaluation(false,true);ceres::Problem::EvaluateOptions options;options.num_threads=threads;double value=NAN;if(!problem.Evaluate(options,&value,nullptr,nullptr,nullptr)||!std::isfinite(value))throw std::runtime_error("local observed complete actual objective evaluation failed");return value;};
 auto bounds=[&](){for(const auto&change:plan.rows){const auto&r=b.core.packet.rows.at(change.row_index);const auto interval=joint_shape_cell_station_interval_v1(b.core.model,r,b.core.station.at(change.row_index));auto*x=&b.core.station[change.row_index];if(!problem.HasParameterBlock(x)||problem.GetManifold(x))throw std::runtime_error("local cell lost its original scalar station block");problem.SetParameterLowerBound(x,0,interval[0]);problem.SetParameterUpperBound(x,0,interval[1]);}};
 const double before=cost();const auto before_debt=debt();double candidate=NAN;Json candidate_debt,proof,rejected=Json::array(),gpu_reports=Json::array();bool committed=false;std::string failure;const auto proposed=plan.rows.size();std::size_t round=0;
 for(;;){
  JointGPULocalRowRebindV1 gpu;JointObservedLocalGraphRebindV1 edit;bool have_edit=false;std::vector<std::size_t>changed;for(const auto&change:plan.rows)changed.push_back(change.row_index);bool accept_round=false;std::set<std::size_t>blocked;std::optional<JointCompactPathStateV1>prepared_path;std::function<void()>commit_remap;
  try{
   gpu=begin_joint_gpu_local_row_rebind_v1(b.native_gpu_evaluation_callback_v1(),changed,true);edit=b.begin_observed_local_cell_rebind_v1(problem,inv,changed,closure.active_row_factors);have_edit=true;apply_joint_observed_branches_local_v1(b,plan,edit.detached);gpu_reports.push_back(patch_joint_gpu_local_row_rebind_v1(b.native_gpu_evaluation_callback_v1(),gpu,true));b.bind_observed_local_rows_v1(problem,inv,edit);bounds();candidate=cost();candidate_debt=debt();proof=joint_observed_cell_inward_descent_proof_v1(b,problem,plan,removal_incidence_enabled);
   if(plan.cell_boundary_only)for(const auto&row:proof.at("rows"))if(!row.at("admissible_cell_release").get<bool>()){blocked.insert(row.at("raw_row").get<std::size_t>());rejected.push_back(row);}
   accept_round=blocked.empty()&&joint_observed_branch_transaction_acceptance_v1(plan,before,candidate,before_debt,candidate_debt,proof);
   if(!accept_round)failure=blocked.empty()?"candidate failed the unchanged complete objective or physical-feasibility gate":"neighbor cell lacks actual complete-objective inward descent";
   else{prepared_path.emplace(accepted.prepare_local_path_rebind_v1(b,!edit.detached));commit_remap=prepare_remap(edit.replacements);gpu_reports.push_back(nullptr);gpu_reports.back()=finish_joint_gpu_local_row_rebind_v1(b.native_gpu_evaluation_callback_v1(),gpu,true,true);}
  }catch(const std::exception&e){failure=e.what();accept_round=false;}
  ++round;
  if(accept_round){commit_remap();accepted.capture_prepared_local_path_rebind_v1(std::move(*prepared_path));committed=true;failure.clear();break;}
  eventio::atomic_json(out/("candidate_failure_"+std::to_string(round)+".json"),{{"schema","rinkfit.observed_local_cell_candidate_failure.v1"},{"candidate_error",failure},{"before_compact_state_token",before_token},{"candidate_paths_applied",plan.applied},{"rollback_completed",false}});
  if(have_edit)b.detach_observed_local_rows_v1(problem,inv,edit);
  if(plan.applied)rollback_joint_observed_branches_local_with_numeric_state_v1(b,plan,have_edit&&edit.detached,[&](){accepted.restore_after_local_path_rebind_v1(b,edit.detached);});else accepted.restore(b);
  std::function<void()>commit_rollback_remap;if(have_edit){b.bind_observed_local_rows_v1(problem,inv,edit);bounds();commit_rollback_remap=prepare_remap(edit.replacements);}
  if(gpu.active)gpu_reports.push_back(finish_joint_gpu_local_row_rebind_v1(b.native_gpu_evaluation_callback_v1(),gpu,false,true));if(commit_rollback_remap)commit_rollback_remap();
  const double restored=cost();if(!accepted.matches_current_values()||std::abs(restored-before)>128*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(before)))throw std::runtime_error("local observed cell rollback failed exact accepted state/objective restoration");
  if(blocked.empty())break;plan.rows.erase(std::remove_if(plan.rows.begin(),plan.rows.end(),[&](const auto&change){return blocked.count(change.row_index)!=0;}),plan.rows.end());if(plan.rows.empty()){failure="every neighboring cell lacked actual complete-objective inward descent";break;}
 }
 auto report=joint_observed_branch_receipt_v1(plan,committed,before,candidate,before_debt,candidate_debt,failure,proof);report["schema"]="rinkfit.observed_local_cell_transaction.v1";report["originally_proposed_rows"]=proposed;report["candidate_rounds"]=round;report["complete_W_groups"]=closure.complete_group_indices.size();report["complete_group_rows"]=closure.complete_group_rows.size();report["locally_rebound_row_factors"]=closure.active_row_factors.size();report["rejected_cell_candidates_with_actual_full_gradient"]=rejected;report["GPU_local_rebind"]=gpu_reports;report["before_compact_state_token"]=before_token;report["final_compact_state_token"]=accepted.token();report["complete_checkpoint_written"]=false;report["complete_checkpoint_policy"]="normal accepted checkpoint cadence and final current checkpoint";report["Ceres_problem_rebuilt"]=false;report["Ceres_fast_removal_incidence_index_enabled"]=removal_incidence_enabled;report["removal_incidence_index_memory_bytes"]=nullptr;report["fallback_CPU_timing_provided_by_this_transaction"]=false;report["global_PHR_or_motion_residuals_rebuilt"]=false;eventio::atomic_json(out/"receipt.json",report);return report;
}
}
