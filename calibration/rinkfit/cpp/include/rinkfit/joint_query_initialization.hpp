#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_query_witness.hpp"
#include <set>
namespace rinkfit {
// A numerical initial state, never a source edit or a native certificate. The
// complete original Phase-I AL is evaluated before and after all proposals.
// Every original physical value, branch, budget and unselected auxiliary stays
// bit identical; multiplier/counter/runtime metadata are owned by the caller.
inline eventio::Json initialize_appended_genuine_query_witnesses(
    JointALBuilder& builder,const RunCheckpoint& checkpoint,std::size_t first_new_row){
 if(first_new_row>builder.core.packet.rows.size())throw std::runtime_error("new Query prefix out of range");
 const auto parent=builder.capture_state();const double budget=*builder.budget_parameter();
 if(parent.token()!=checkpoint.state.token())throw std::runtime_error("new Query initializer is not at the complete rebased state");
 auto merit=[&](const StateSnapshot& state){const auto value=builder.evaluate_constraints(state);
  if(value.g.size()!=checkpoint.outer.lambda.size()||value.h.size()!=checkpoint.outer.nu.size())throw std::runtime_error("new Query original AL multiplier scope differs");
  long double total=budget+.5L;for(std::size_t i=0;i<value.g.size();++i){double r=inequality_residual(value.g[i],checkpoint.outer.lambda[i],checkpoint.outer.rho);total+=.5L*r*r;}
  for(std::size_t i=0;i<value.h.size();++i){double r=equality_residual(value.h[i],checkpoint.outer.nu[i],checkpoint.outer.rho);total+=.5L*r*r;}return double(total);};
 std::vector<std::size_t> rows;for(std::size_t i=first_new_row;i<builder.core.packet.rows.size();++i){const auto&r=builder.core.packet.rows[i];if(r.active&&r.query&&r.kind!=1)rows.push_back(i);}
 eventio::Json report={{"schema","rinkfit.appended_finite_Query_original_AL_numerical_initialization.v1"},{"first_new_row",first_new_row},{"requested_rows",rows.size()},{"parent_state_token",parent.token()},{"source_SHA",builder.core.packet.source_hash},{"source_query_windows_widths_world_sides_and_finite_branches_unchanged",true},{"all_old_named_primal_aux_budget_and_physical_values_bit_identical",true},{"native_pass_or_fitting_convergence_claimed",false},{"rows",eventio::Json::array()}};
 if(rows.empty()){report["accepted"]=true;report["changed_rows"]=0;report["state_token"]=parent.token();return report;}
 if(checkpoint.outer.phase!=Phase::kFeasibility)throw std::runtime_error("new Query witness initialization requires the complete original Phase-I objective; Phase-II source appends need an explicit full-W acceptance path");
 const double before=merit(parent);std::set<std::size_t> changed;
 for(auto i:rows){const auto&r=builder.core.packet.rows[i];const std::array<double,3>old{builder.core.station[i],builder.core.row[i],builder.core.alpha[i]};const auto proposal=propose_joint_query_witness(builder.core.model,r,old);
  eventio::Json row={{"row_index",i},{"source_row_id",r.source_row_id},{"shot_id",r.shot_id},{"native_pts",r.pts},{"material",r.material},{"path",r.path},{"width_block",r.width_block},{"world_side",r.width_sign},{"old_auxiliary",old},{"proposal_available",proposal.available},{"query_numerically_closed",proposal.query_closed},{"finite_starts",proposal.starts},{"closed_starts",proposal.closed_starts}};
  if(proposal.available&&proposal.auxiliary!=old){builder.core.station[i]=proposal.auxiliary[0];builder.core.row[i]=proposal.auxiliary[1];builder.core.alpha[i]=proposal.auxiliary[2];changed.insert(i);row["proposed_auxiliary"]=proposal.auxiliary;row["native_error_px"]=std::sqrt(proposal.native_error_sq);row["RS_Query_equation_norm_px"]=std::sqrt(proposal.query_error_sq);}report["rows"].push_back(std::move(row));}
 auto candidate=builder.capture_state();
 if(!(candidate.data().identity==parent.data().identity)||candidate.data().discrete!=parent.data().discrete||candidate.data().root_charts!=parent.data().root_charts||*builder.budget_parameter()!=budget||candidate.data().parameters.size()!=parent.data().parameters.size())throw std::runtime_error("new Query initializer changed complete identity, branch, chart or budget");
 for(std::size_t k=0;k<parent.data().parameters.size();++k){const auto&a=parent.data().parameters[k];const auto&z=candidate.data().parameters[k];if(a.name!=z.name||a.equality_auxiliary!=z.equality_auxiliary)throw std::runtime_error("new Query initializer changed named registry");bool allowed=false;
  for(const auto&prefix:{std::string("station/"),std::string("row/"),std::string("alpha/")})if(a.name.rfind(prefix,0)==0)allowed=changed.count(std::stoull(a.name.substr(prefix.size())))!=0;
  if(!allowed&&a.values!=z.values)throw std::runtime_error("new Query initializer changed unselected parameter "+a.name);}
 const double after=merit(candidate);const bool accepted=std::isfinite(after)&&after<=before+1e-8;
 if(!accepted){builder.restore_state(parent);if(builder.capture_state().token()!=parent.token())throw std::runtime_error("rejected new Query initialization did not restore exact initial state");}
 report["same_complete_original_AL_before"]=before;report["candidate_same_complete_original_AL_after"]=after;report["accepted"]=accepted;report["changed_rows"]=accepted?changed.size():0;report["state_token"]=builder.capture_state().token();report["rejected_candidate_restored_exact_rebased_state"]=!accepted;return report;
}
}
