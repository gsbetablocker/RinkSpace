#pragma once
#include "rinkfit/joint_al.hpp"
#include <mutex>
#include <unordered_set>
#include "rinkfit/joint_factor_failure_v1.hpp"
#include "rinkfit/joint_global_phr_epoch_v1.hpp"
namespace rinkfit {
// Reuse the original PHR factor implementation. The only adapter here shares
// the original scalar factor between read-only feasibility and the live graph.
class JointSharedPhysicalScalarV1:public ceres::CostFunction{
 protected:std::shared_ptr<ceres::CostFunction>source_;
 public:explicit JointSharedPhysicalScalarV1(std::shared_ptr<ceres::CostFunction>source):source_(std::move(source)){
  if(!source_||source_->num_residuals()!=1)throw std::runtime_error("global PHR needs one original scalar constraint");set_num_residuals(1);*mutable_parameter_block_sizes()=source_->parameter_block_sizes();
 }
 bool Evaluate(double const*const*p,double*r,double**j)const override{return source_->Evaluate(p,r,j);}
};
class JointSharedPhysicalADScalarV1 final:public JointSharedPhysicalScalarV1,public PHRADValueProbe{
 public:explicit JointSharedPhysicalADScalarV1(std::shared_ptr<ceres::CostFunction>source):JointSharedPhysicalScalarV1(std::move(source)){}
 bool EvaluateADValue(double const*const*p,double*r)const override{return dynamic_cast<const PHRADValueProbe*>(source_.get())->EvaluateADValue(p,r);}
};
inline std::unique_ptr<ceres::CostFunction>share_joint_physical_scalar_v1(const std::shared_ptr<ceres::CostFunction>&source){
 if(dynamic_cast<const PHRADValueProbe*>(source.get()))return std::make_unique<JointSharedPhysicalADScalarV1>(source);
 return std::make_unique<JointSharedPhysicalScalarV1>(source);
}
class JointMutablePhysicalPHRCostV1 final:public ceres::CostFunction{
 std::shared_ptr<ceres::CostFunction>source_;JointGlobalPHRStateAccessV1 state_;std::size_t index_;bool equality_;std::string id_;std::vector<double*>origins_;
 mutable std::mutex mutex_;mutable std::unique_ptr<ceres::CostFunction>phr_;mutable double rho_=NAN,multiplier_=NAN;
 public:JointMutablePhysicalPHRCostV1(std::shared_ptr<ceres::CostFunction>source,JointGlobalPHRStateAccessV1 state,std::size_t index,bool equality,std::string id={},std::vector<double*>origins={}):source_(std::move(source)),state_(state),index_(index),equality_(equality),id_(std::move(id)),origins_(std::move(origins)){set_num_residuals(1);*mutable_parameter_block_sizes()=source_->parameter_block_sizes();}
 bool Evaluate(double const*const*p,double*r,double**j)const override{
  auto fail=[&](const char*reason){return joint_actual_factor_failure_v1([&](eventio::Json&record){record["factor"]="mutable_global_PHR";record["constraint_id"]=id_;record["constraint_index"]=index_;record["equality"]=equality_;record["failure_kind"]=reason;for(std::size_t k=0;k<parameter_block_sizes().size();++k)append_joint_actual_failure_parameter_v1(record,k<origins_.size()?origins_[k]:nullptr,p[k],parameter_block_sizes()[k],"PHR_parameter/"+std::to_string(k));});};
  std::lock_guard<std::mutex>lock(mutex_);try{const double rho=state_.at("rho").get<double>(),multiplier=state_.at(equality_?"nu":"lambda").at(index_).get<double>();
   if(!phr_||rho!=rho_||multiplier!=multiplier_){phr_=make_phr_cost(share_joint_physical_scalar_v1(source_),multiplier,rho,equality_,true);rho_=rho;multiplier_=multiplier;}
   return phr_->Evaluate(p,r,j)?true:fail("original_scalar_or_PHR_evaluation_false");
  }catch(const std::exception&e){return fail(e.what());}catch(...){return fail("non_std_exception");}
 }
};
class JointPhysicalFeasibilityV1 {
 using Json=eventio::Json;struct Term{std::string id;bool equality;std::size_t index;std::shared_ptr<ceres::CostFunction>cost;std::vector<double*>parameters;std::vector<const double*>read_parameters;};
 JointGlobalPHRStateAccessV1 state_;Json policy_;std::string source_sha_;std::vector<Term>terms_;std::vector<std::string>inequality_ids_,equality_ids_;std::unordered_set<std::string>constraint_ids_;bool direct_build_finalized_=false;
 static std::string sha(const Json&j){const auto text=j.dump();return eventio::sha256(std::vector<std::uint8_t>(text.begin(),text.end()));}
 static void require(bool ok,const std::string&why){if(!ok)throw std::runtime_error("global physical PHR: "+why);}
 static void validate_policy(const Json&p){require(p.at("schema")=="rinkfit.engineering_global_phr_policy.v1","unsupported policy");for(const char*key:{"initial_rho","rho_growth","required_progress_ratio","primal_tolerance","complementarity_tolerance"})require(std::isfinite(p.at(key).get<double>()),std::string("nonfinite policy ")+key);require(p.at("initial_rho").get<double>()>0&&p.at("rho_growth").get<double>()>1&&p.at("required_progress_ratio").get<double>()>0&&p.at("required_progress_ratio").get<double>()<1&&p.at("primal_tolerance").get<double>()>=0&&p.at("complementarity_tolerance").get<double>()>=0,"invalid numeric policy");}
 void validate_state()const{const auto&s=state_.json();require(s.at("schema")=="rinkfit.engineering_global_phr_state.v1"&&s.at("source_packet_sha256")==source_sha_&&s.at("policy_sha256")==sha(policy_),"saved source/policy identity differs");require(s.at("inequality_ids")==inequality_ids_&&s.at("equality_ids")==equality_ids_,"saved ordered original constraint IDs differ");require(s.at("lambda").size()==inequality_ids_.size()&&s.at("nu").size()==equality_ids_.size(),"saved multiplier shape differs");require(std::isfinite(s.at("rho").get<double>())&&s.at("rho").get<double>()>0,"saved rho invalid");for(const auto&x:s.at("lambda"))require(std::isfinite(x.get<double>())&&x.get<double>()>=0,"saved inequality multiplier invalid");for(const auto&x:s.at("nu"))require(std::isfinite(x.get<double>()),"saved equality multiplier invalid");require(s.at("outer_iteration").get<long long>()>=0,"saved outer iteration invalid");if(!s.at("previous_primal_inf").is_null())require(std::isfinite(s.at("previous_primal_inf").get<double>())&&s.at("previous_primal_inf").get<double>()>=0,"saved progress invalid");}
 void initialize_or_validate(){validate_policy(policy_);if(state_.is_null()){require(policy_.at("initialization")=="explicit_new_objective_zero_duals","missing explicit new-objective dual initialization");state_.publish(Json{{"schema","rinkfit.engineering_global_phr_state.v1"},{"source_packet_sha256",source_sha_},{"policy_sha256",sha(policy_)},{"inequality_ids",inequality_ids_},{"equality_ids",equality_ids_},{"rho",policy_.at("initial_rho")},{"lambda",std::vector<double>(inequality_ids_.size(),0.)},{"nu",std::vector<double>(equality_ids_.size(),0.)},{"outer_iteration",0},{"previous_primal_inf",nullptr},{"last_progress",nullptr}});}validate_state();}
 public:
 JointPhysicalFeasibilityV1(JointGlobalPHRStateAccessV1 state,Json policy,std::string source_sha):state_(state),policy_(std::move(policy)),source_sha_(std::move(source_sha)){}
 void append_original(std::string id,bool equality,std::unique_ptr<ceres::CostFunction>cost,std::vector<double*>parameters){
  require(!direct_build_finalized_,"original factor appended after direct graph finalization");require(cost&&cost->num_residuals()==1&&cost->parameter_block_sizes().size()==parameters.size(),"original scalar factor shape");auto&ids=equality?equality_ids_:inequality_ids_;require(!id.empty()&&constraint_ids_.insert(id).second,"duplicate original constraint ID");const std::size_t index=ids.size();ids.push_back(id);std::vector<const double*>read(parameters.begin(),parameters.end());terms_.push_back({std::move(id),equality,index,std::shared_ptr<ceres::CostFunction>(std::move(cost)),std::move(parameters),std::move(read)});
 }
 ceres::ResidualBlockId add_original_final_factor(ceres::Problem&p,std::string id,bool equality,std::unique_ptr<ceres::CostFunction>cost,std::vector<double*>parameters){
  append_original(std::move(id),equality,std::move(cost),std::move(parameters));const auto&term=terms_.back();
  return p.AddResidualBlock(new JointMutablePhysicalPHRCostV1(term.cost,state_,term.index,term.equality,joint_actual_factor_failure_audit_v1().enabled?term.id:std::string{},joint_actual_factor_failure_audit_v1().enabled?term.parameters:std::vector<double*>{}),nullptr,term.parameters);
 }
 void finish_direct_build(std::size_t expected_inequalities,std::size_t expected_equalities){
  require(!direct_build_finalized_&&expected_inequalities>0&&inequality_ids_.size()==expected_inequalities&&equality_ids_.size()==expected_equalities,"direct global factor count differs from original physical registry");initialize_or_validate();direct_build_finalized_=true;
 }
 Json installation_report()const{
  require(direct_build_finalized_,"direct global graph is incomplete");
  return {{"schema","rinkfit.actual_global_phr_installation.v1"},{"inequality_ids",inequality_ids_},{"equality_ids",equality_ids_},{"direct_final_global_phr_residuals",terms_.size()},{"replaced_old_soft_global_residuals",0},{"removed_residual_blocks",0},{"recreated_residual_blocks",0},{"duplicated_global_penalty_factors",0},{"actual_original_factor_math_reused",true},{"original_PHR_factory_reused",true},{"pixel_graph_rebuilt",false},{"full_source_or_pixel_membership_changed",false},{"state_persisted_in_complete_primal",true},{"observed_initialization_only",true},{"formal_Query_or_true_RS_closed",false},{"quality_accepted",false}};
 }
 Json evaluate()const{validate_state();Json values=Json::array();double gmax=0,hmax=0,complementarity=0;for(const auto&term:terms_){double value=0.;require(term.cost->Evaluate(term.read_parameters.data(),&value,nullptr)&&std::isfinite(value),"original global constraint arithmetic failed: "+term.id);if(term.equality)hmax=std::max(hmax,std::abs(value));else{gmax=std::max(gmax,std::max(0.,value));complementarity=std::max(complementarity,std::abs(state_.at("lambda").at(term.index).get<double>()*value));}values.push_back({{"id",term.id},{"equality",term.equality},{"index",term.index},{"value",value}});}const double primal=std::max(gmax,hmax);return {{"g_max_positive",gmax},{"h_max_abs",hmax},{"primal_inf",primal},{"complementarity_inf",complementarity},{"global_primal_within_tolerance",primal<=policy_.at("primal_tolerance").get<double>()},{"global_complementarity_within_tolerance",complementarity<=policy_.at("complementarity_tolerance").get<double>()},{"constraint_values",values},{"stationarity_certified",false},{"formal_Query_or_true_RS_closed",false},{"quality_accepted",false}};}
 Json advance_outer(){const auto measured=evaluate();Json next=state_.json();const double rho=next.at("rho");double maximum_dual_change=0.;for(const auto&term:measured.at("constraint_values")){const bool equality=term.at("equality");const auto index=term.at("index").get<std::size_t>();auto&dual=next[equality?"nu":"lambda"][index];const double before=dual.get<double>(),proposal=before+rho*term.at("value").get<double>(),after=equality?proposal:std::max(0.,proposal);require(std::isfinite(after),"nonfinite updated multiplier");dual=after;maximum_dual_change=std::max(maximum_dual_change,std::abs(after-before));}
  const double primal=measured.at("primal_inf");bool grew=false;if(primal>policy_.at("primal_tolerance").get<double>()&&!next.at("previous_primal_inf").is_null()&&primal>policy_.at("required_progress_ratio").get<double>()*next.at("previous_primal_inf").get<double>()){const double candidate=rho*policy_.at("rho_growth").get<double>();require(std::isfinite(candidate)&&candidate>rho,"rho growth overflow");next["rho"]=candidate;grew=true;}
  next["outer_iteration"]=next.at("outer_iteration").get<std::size_t>()+1;next["previous_primal_inf"]=primal;next["last_progress"]=measured;state_.publish(std::move(next));validate_state();auto report=evaluate();report["schema"]="rinkfit.actual_global_phr_outer_update.v1";report["outer_iteration"]=state_.at("outer_iteration");report["rho_before"]=rho;report["rho_after"]=state_.at("rho");report["rho_increased_for_insufficient_primal_progress"]=grew;report["maximum_dual_change"]=maximum_dual_change;report["multipliers_follow_original_PHR_update"]=true;report["complete_state_recapture_required_before_accepted_checkpoint"]=true;report["live_small_factors_consume_updated_state_without_pixel_rebuild"]=true;return report;
 }
 // Numerical gauges may inspect these same original factors, never a copied
 // physical formula or a new penalty. Ordinals stay in the original order.
 template<class Visitor>void visit_original_terms_v1(Visitor&&visitor)const{for(std::size_t i=0;i<terms_.size();++i)visitor(i,terms_[i].parameters);}
 bool original_term_blocks_numerical_gauge_v1(std::size_t ordinal)const{
  const auto&t=terms_.at(ordinal);if(t.equality)return true;double g=0.;
  if(!t.cost->Evaluate(t.read_parameters.data(),&g,nullptr)||!std::isfinite(g))return true;
  const double branch=state_.at("lambda").at(t.index).get<double>()+state_.at("rho").get<double>()*g;
  return !std::isfinite(branch)||g>=0.||branch>=0.;
 }
 const Json&state()const{return state_.json();}
 template<class Builder>static std::shared_ptr<JointPhysicalFeasibilityV1>begin_direct_build(Builder&b,const Json&cfg){
  if(!cfg.contains("engineering_global_phr_v1"))return {};require((b.source_uv_observed_row_initialization()&&!b.source_uv_capture_v1())||(!b.source_uv_observed_row_initialization()&&b.source_uv_capture_v3()),"this adapter requires an actual observed-primal or trueRS v3/v4 capture graph");return std::make_shared<JointPhysicalFeasibilityV1>(b.engineering_global_phr_state_v1,cfg.at("engineering_global_phr_v1"),b.core.packet.source_hash);
 }
};
}
