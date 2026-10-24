#pragma once
#include "rinkfit/event_io.hpp"
#include <atomic>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <functional>
namespace rinkfit {
// Failure-only telemetry. No state, row scan or projection work is added to a
// successful evaluation; the first actual false return emits one bounded record.
inline std::atomic_flag& joint_factor_failure_once_v1(){static std::atomic_flag seen=ATOMIC_FLAG_INIT;return seen;}
inline std::string joint_factor_failure_text_v1(const char*text){return text?std::string(text).substr(0,512):std::string();}
inline eventio::Json joint_factor_failure_number_v1(double x){if(std::isfinite(x))return x;return std::isnan(x)?eventio::Json("NaN"):eventio::Json(x>0?"+Inf":"-Inf");}

// Explicit first-proposal audit only. The base views borrow the probe's one
// immutable arm-point buffer; successful production evaluation does no scan.
struct JointActualFactorFailureAuditV1 {
 struct Base{std::string name;const double*value=nullptr;int size=0;};
 bool enabled=false,stop_after_first_failure=false;std::filesystem::path output;const double*station=nullptr;const double*row=nullptr;const double*alpha=nullptr;
 std::unordered_map<const double*,Base> bases;
 std::function<Base(const double*,int)> resolve_accepted_base;
 std::atomic_flag emitted=ATOMIC_FLAG_INIT;
};
inline JointActualFactorFailureAuditV1& joint_actual_factor_failure_audit_v1(){static JointActualFactorFailureAuditV1 audit;return audit;}
inline void configure_joint_actual_factor_failure_audit_v1(bool enabled,const std::filesystem::path&out,const double*station=nullptr,const double*row=nullptr,const double*alpha=nullptr){auto&a=joint_actual_factor_failure_audit_v1();a.enabled=enabled;a.output=out;a.station=station;a.row=row;a.alpha=alpha;a.bases.clear();a.resolve_accepted_base={};a.emitted.clear(std::memory_order_relaxed);}
inline void bind_joint_actual_factor_failure_base_v1(const double*address,std::string name,const double*value,int size){auto&a=joint_actual_factor_failure_audit_v1();if(a.enabled)a.bases.emplace(address,JointActualFactorFailureAuditV1::Base{std::move(name),value,size});}
inline void append_joint_actual_failure_parameter_v1(eventio::Json&record,const double*origin,const double*actual,int size,const std::string&fallback_name={}){
 auto&a=joint_actual_factor_failure_audit_v1();auto it=a.bases.find(origin);eventio::Json before=eventio::Json::array(),after=eventio::Json::array();
 JointActualFactorFailureAuditV1::Base resolved;const bool registered=it!=a.bases.end();if(registered)resolved=it->second;else if(a.resolve_accepted_base)resolved=a.resolve_accepted_base(origin,size);const bool has_base=resolved.value!=nullptr;if(has_base&&resolved.size!=size)throw std::runtime_error("factor failure base parameter dimension");
 for(int k=0;k<size;++k){before.push_back(joint_factor_failure_number_v1(has_base?resolved.value[k]:actual[k]));after.push_back(joint_factor_failure_number_v1(actual[k]));}
 record["parameters"].push_back({{"name",has_base?resolved.name:fallback_name},{"base_values",before},{"actual_values",after},{"base_source",has_base?(registered?"actual_solver_arm_point":"current_complete_accepted_state"):"current_value_no_saved_base"}});
}
template<class Fill>bool joint_actual_factor_failure_v1(Fill&&fill)noexcept{
 auto&a=joint_actual_factor_failure_audit_v1();if(!a.enabled||a.emitted.test_and_set(std::memory_order_relaxed))return false;
 try{eventio::Json record={{"schema","rinkfit.first_actual_factor_failure_with_point.v1"},{"returned_false",true},{"extra_full_problem_evaluations",0},{"parameters",eventio::Json::array()}};fill(record);eventio::atomic_json(a.output/"actual_first_factor_failure.json",record);if(a.stop_after_first_failure){std::ofstream stop(a.output/"STOP_REQUESTED");stop<<"Saved first runtime factor failure and accepted base; stop at the next solver callback boundary.\n";}const auto line="[actual_factor_failure] "+record.dump(-1,' ',false,eventio::Json::error_handler_t::replace)+"\n";std::fwrite(line.data(),1,line.size(),stderr);std::fflush(stderr);}catch(...){std::fputs("[actual_factor_failure] first false captured; serialization failed\n",stderr);std::fflush(stderr);}return false;
}

template<class Row>bool joint_factor_failure_v1(const Row*row,double const*const*p,const char*factor,const char*stage,const char*kind="evaluation_false",int domain_kind=-1)noexcept{
 if(joint_actual_factor_failure_audit_v1().enabled)joint_actual_factor_failure_v1([&](eventio::Json&record){record["factor"]=factor;record["stage"]=stage;record["failure_kind"]=kind;record["domain_kind"]=domain_kind;if(row&&row->source){const auto&s=*row->source;record["raw_row"]=row->cache_row;record["source_row_id"]=s.source_row_id;record["shot_id"]=s.shot_id;record["native_pts"]=s.pts;record["material"]=s.material;record["path"]=row->layout.path;record["layout_shape_cell"]=row->layout.shape_segment;if(row->model&&p){for(std::size_t k=0;k<row->layout.block_ids.size();++k){const auto&block=row->model->blocks.at(row->layout.block_ids[k]);append_joint_actual_failure_parameter_v1(record,block.values.data(),p[row->physical_slots.at(k)],int(block.values.size()),block.name);}const auto&a=joint_actual_factor_failure_audit_v1();if(row->station_slot>=0)append_joint_actual_failure_parameter_v1(record,a.station? a.station+row->cache_row:nullptr,p[row->station_slot],1,"station/"+std::to_string(row->cache_row));if(row->row_slot>=0)append_joint_actual_failure_parameter_v1(record,a.row?a.row+row->cache_row:nullptr,p[row->row_slot],1,"row/"+std::to_string(row->cache_row));if(row->alpha_slot>=0)append_joint_actual_failure_parameter_v1(record,a.alpha?a.alpha+row->cache_row:nullptr,p[row->alpha_slot],1,"alpha/"+std::to_string(row->cache_row));}}});
 if(joint_factor_failure_once_v1().test_and_set(std::memory_order_relaxed))return false;
 try{
  using J=eventio::Json;J record={{"schema","rinkfit.first_actual_factor_evaluation_failure.v1"},{"factor",factor},{"stage",stage},{"failure_kind",kind},{"domain_kind",domain_kind},{"returned_false",true},{"state_changed",false},{"failure_only_single_row_double_recheck",false}};
  if(row&&row->source){
   const auto&s=*row->source;record["raw_row"]=row->cache_row;record["source_dataset_id"]=s.source_dataset_id;record["source_row_id"]=s.source_row_id;record["shot_id"]=s.shot_id;record["native_pts"]=s.pts;record["material"]=s.material;record["path"]=row->layout.path;record["layout_shape_cell"]=row->layout.shape_segment;record["source_shape_cell"]=s.numerical_shape_segment;record["source_uv"]=s.uv;record["source_seed_station"]=joint_factor_failure_number_v1(s.station);
   const double station=p&&row->station_slot>=0?p[row->station_slot][0]:s.station,sensor=p&&row->row_slot>=0?p[row->row_slot][0]:row->fixed_row_value;
   record["trial_station"]=joint_factor_failure_number_v1(station);record["trial_sensor_row"]=joint_factor_failure_number_v1(sensor);
   if(row->model&&p){
    // The failed CostFunction is not retried. This one double projection only
    // diagnoses its exact row/cell at the same trial point; AD may fail separately.
    record["failure_only_single_row_double_recheck"]=true;
    try{const auto projection=row->model->predict_for_al(row->layout,station,sensor,p);record["double_projection_arithmetic_valid"]=projection.arithmetic_valid;record["double_projection_uv"]=J::array({joint_factor_failure_number_v1(projection.uv[0]),joint_factor_failure_number_v1(projection.uv[1])});if(!projection.arithmetic_valid&&std::string(kind)=="evaluation_false")record["failure_kind"]="projection_false";}
    catch(const std::exception&e){const auto why=joint_factor_failure_text_v1(e.what());record["double_projection_exception"]=why;if(why.find("shared split print station left declared numerical cell")!=std::string::npos)record["failure_kind"]="local_cell_domain";else record["double_projection_failed"]=true;}
    catch(...){record["double_projection_exception"]="non_std_exception";}
   }
  }
  const auto line="[factor_failure] "+record.dump(-1,' ',false,J::error_handler_t::replace)+"\n";std::fwrite(line.data(),1,line.size(),stderr);std::fflush(stderr);
 }catch(...){std::fputs("[factor_failure] first actual factor returned false; record unavailable\n",stderr);std::fflush(stderr);}
 return false;
}
}
