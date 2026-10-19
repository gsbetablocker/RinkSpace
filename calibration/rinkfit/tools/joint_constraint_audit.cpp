// Compact read-only decomposition of an exact saved joint PHR state.
#include "rinkfit/joint_al.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
using namespace rinkfit;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("usage MODEL PACKET COMPLETE_CHECKPOINT NEW_REPORT");
 if(std::filesystem::exists(argv[4]))throw std::runtime_error("immutable diagnostic report exists");
 auto model=JointEventModel::load(argv[1]);auto packet=JointPacket::load(argv[2],model);
 std::ifstream input(argv[3]);auto checkpoint=read_checkpoint(input);const bool initialization=checkpoint.state.data().identity.objective_id.find("geometry_initialization")!=std::string::npos;JointALBuilder b(std::move(model),std::move(packet),initialization,false);b.restore_state(checkpoint.state);
 if(b.capture_state().token()!=checkpoint.state.token())throw std::runtime_error("complete continuous/discrete checkpoint did not roundtrip");
 auto values=b.evaluate_constraints(checkpoint.state);const auto&schema=b.schema();const auto&o=checkpoint.outer;
 if(values.g.size()!=schema.inequality_ids.size()||values.h.size()!=schema.equality_ids.size()||values.g.size()!=o.lambda.size()||values.h.size()!=o.nu.size())throw std::runtime_error("constraint/dual schema mismatch");
 eventio::Json groups=eventio::Json::object();
 auto bucket=[&](const std::string&id,bool equality){if(equality)return id.rfind("RS/",0)==0?std::string("RS"):id.rfind("frozen_query_0/",0)==0?std::string("frozen_query_0"):id.rfind("frozen_query_1/",0)==0?std::string("frozen_query_1"):std::string("physical_equality");if(id.rfind("pixel/",0)==0)return std::string("pixel");if(id.rfind("native_domain/",0)==0)return std::string("native_domain_kind_")+id.substr(id.find_last_of('/')+1);if(id.find("whole_cubic_hull")!=std::string::npos)return std::string("whole_zoom_control_enclosure");if(id.find("whole_zoom_Brown")!=std::string::npos)return std::string("whole_zoom_Brown_certificate");return std::string("finite_physical_geometry");};
 std::vector<std::pair<double,eventio::Json>>worst;double cost=0;
 auto add=[&](double raw,double residual,const std::string&id,bool equality,double dual){auto key=bucket(id,equality);auto&g=groups[key];if(g.is_null())g={{"count",0},{"positive_or_absolute_violation_count",0},{"max_positive_or_absolute_violation",0.},{"PHR_half_cost",0.}};g["count"]=g["count"].get<std::size_t>()+1;double violation=equality?std::abs(raw):std::max(0.,raw);if(violation>0)g["positive_or_absolute_violation_count"]=g["positive_or_absolute_violation_count"].get<std::size_t>()+1;g["max_positive_or_absolute_violation"]=std::max(g["max_positive_or_absolute_violation"].get<double>(),violation);const double c=.5*residual*residual;g["PHR_half_cost"]=g["PHR_half_cost"].get<double>()+c;cost+=c;if(worst.size()<40)worst.push_back({c,{{"id",id},{"constraint_value",raw},{"multiplier",dual},{"residual",residual},{"half_cost",c}}});else{auto it=std::min_element(worst.begin(),worst.end(),[](const auto&a,const auto&b){return a.first<b.first;});if(c>it->first)*it={c,{{"id",id},{"constraint_value",raw},{"multiplier",dual},{"residual",residual},{"half_cost",c}}};}};
 for(std::size_t i=0;i<values.g.size();++i)add(values.g[i],inequality_residual(values.g[i],o.lambda[i],o.rho),schema.inequality_ids[i],false,o.lambda[i]);
 for(std::size_t i=0;i<values.h.size();++i)add(values.h[i],equality_residual(values.h[i],o.nu[i],o.rho),schema.equality_ids[i],true,o.nu[i]);
 double budget=*b.budget_parameter(),budget_cost=0;if(o.phase==Phase::kFeasibility){const auto r=phase_one_budget_residual(budget);budget_cost=.5*r*r;cost+=budget_cost;}
 std::partial_sort(worst.begin(),worst.begin()+std::min<std::size_t>(40,worst.size()),worst.end(),[](const auto&a,const auto&b){return a.first>b.first;});eventio::Json top=eventio::Json::array();for(std::size_t i=0;i<std::min<std::size_t>(40,worst.size());++i)top.push_back(worst[i].second);
 eventio::Json report={{"schema","rinkfit.readonly_joint_constraint_decomposition.v1"},{"model_sha256",eventio::file_sha256(argv[1])},{"packet_sha256",eventio::file_sha256(argv[2])},{"checkpoint_sha256",eventio::file_sha256(argv[3])},{"state_token",checkpoint.state.token()},{"complete_snapshot_roundtrip_equal",true},{"phase",int(o.phase)},{"outer",o.outer_iteration},{"rho",o.rho},{"budget_sq",budget},{"budget_objective_half_cost",budget_cost},{"double_evaluation_PHRI_half_cost",cost},{"canonical_AD_merit_or_formal_certificate",false},{"groups",groups},{"largest_PHRI_factors",top},{"state_or_source_modified",false}};
 std::ofstream out(argv[4]);out<<report.dump(2);std::cout<<report.dump(2);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();return 1;}}
