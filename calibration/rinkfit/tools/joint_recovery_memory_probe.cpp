// Read-only sequential-process comparison of runtime representations.
// No Ceres graph or solver is built. Physical/native constraints are optional
// because their existing frozen() implementation copies the complete packet.
#include "rinkfit/joint_recovery_metrics_export.hpp"
#include "rinkfit/joint_source_guard.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include "rinkfit/joint_memory.hpp"
#include <iostream>
#include <set>
#include <chrono>

using namespace rinkfit;
namespace {
using Clock=std::chrono::steady_clock;
struct JsonAllocationCounts {
 std::size_t nodes=0,string_capacity=0,object_key_capacity=0,array_capacity=0;
 void count(const eventio::Json&j){++nodes;
  if(j.is_string())string_capacity+=j.get_ref<const std::string&>().capacity()+1;
  else if(j.is_array()){array_capacity+=j.get_ref<const eventio::Json::array_t&>().capacity()*sizeof(eventio::Json);for(const auto&v:j)count(v);}
  else if(j.is_object())for(auto i=j.begin();i!=j.end();++i){object_key_capacity+=i.key().capacity()+1;count(i.value());}
 }
 eventio::Json json()const{return {{"JSON_nodes",nodes},{"string_capacity_bytes",string_capacity},{"object_key_capacity_bytes",object_key_capacity},{"array_element_capacity_bytes",array_capacity},{"allocator_and_tree_node_overhead_included",false}};}
};
eventio::Json packet_buffers(const JointPacket&p){
 std::size_t ids=0,intervals=0,group_rows=0,W=0,group_strings=0,queries=0,alpha=0,uv=0,nx=0,ny=0,den=0,active=0;
 std::set<const JointQuery*>seen;
 for(const auto&r:p.rows){active+=r.active;for(const auto*s:{&r.source_dataset_id,&r.source_row_id,&r.shot_id,&r.source_group_id,&r.fit_role})ids+=s->capacity()+1;
  intervals+=r.source_station_intervals.capacity()*sizeof(std::array<double,2>);
  if(r.query&&seen.insert(r.query.get()).second){++queries;const auto&q=*r.query;alpha+=q.alpha.capacity()*sizeof(double);uv+=q.uv.capacity()*sizeof(Vec2<double>);nx+=q.nx.capacity()*sizeof(double);ny+=q.ny.capacity()*sizeof(double);den+=q.den.capacity()*sizeof(double);}
 }
 for(const auto&g:p.groups){group_rows+=g.rows.capacity()*sizeof(std::size_t);W+=g.W.capacity()*sizeof(double);group_strings+=g.id.capacity()+1+g.statistic_kind.capacity()+1;}
 JsonAllocationCounts json;json.count(p.source_json);
 return {{"raw_rows",p.rows.size()},{"active_rows",active},{"rows_capacity",p.rows.capacity()},{"typed_row_inline_bytes",p.rows.capacity()*sizeof(JointObservation)},{"typed_row_string_capacity_bytes",ids},{"typed_source_station_interval_bytes",intervals},
 {"statistical_groups",p.groups.size()},{"typed_group_inline_bytes",p.groups.capacity()*sizeof(JointStatisticalGroup)},{"typed_group_row_index_bytes",group_rows},{"typed_group_W_f64_bytes",W},{"typed_group_string_capacity_bytes",group_strings},
 {"unique_Query_objects",queries},{"Query_object_bytes",queries*sizeof(JointQuery)},{"Query_alpha_f64_bytes",alpha},{"Query_uv_f64x2_bytes",uv},{"Query_numerator_x_f64_bytes",nx},{"Query_numerator_y_f64_bytes",ny},{"Query_denominator_f64_bytes",den},
 {"source_json_allocations",json.json()},{"raw_array_cache_capacity_bytes",joint_packet_raw_cache_bytes()},
 {"CPU_typed_consumers","JointALBuilder/core, native source/RS/Query/visibility evaluators; groups/W retained without editing"},{"GPU_copies_in_this_probe",false}};
}
std::string string_sha(const std::string&s){eventio::SHA256Stream h;h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());return h.finish();}
std::string registry_sha(const std::vector<ParameterBlockBinding>&reg){eventio::SHA256Stream h;for(const auto&r:reg){const std::uint64_t n=r.name.size(),size=r.size;h.update(reinterpret_cast<const std::uint8_t*>(&n),sizeof(n));h.update(reinterpret_cast<const std::uint8_t*>(r.name.data()),r.name.size());h.update(reinterpret_cast<const std::uint8_t*>(&size),sizeof(size));h.update(reinterpret_cast<const std::uint8_t*>(r.data),r.size*sizeof(double));}return h.finish();}
std::string names_sha(const std::vector<std::string>&names){eventio::SHA256Stream h;for(const auto&s:names){const std::uint64_t n=s.size();h.update(reinterpret_cast<const std::uint8_t*>(&n),sizeof(n));h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}return h.finish();}
void verify_registry(const StateSnapshot&s,const std::vector<ParameterBlockBinding>&r){if(s.data().parameters.size()!=r.size())throw std::runtime_error("registered block count mismatch");for(std::size_t i=0;i<r.size();++i){const auto&p=s.data().parameters[i];if(p.name!=r[i].name||p.values.size()!=r[i].size||std::memcmp(p.values.data(),r[i].data,r[i].size*sizeof(double)))throw std::runtime_error("registered block name/shape/bitwise value mismatch");}}
void mutate_trial(std::vector<ParameterBlockBinding>&reg){for(auto&r:reg)for(std::size_t j=0;j<r.size;++j)r.data[j]=std::nextafter(r.data[j],INFINITY);}
}
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("usage: joint_recovery_memory_probe legacy|compact MODEL PACKET CONFIG_JSON NEW_OUTPUT");
 const std::string mode=argv[1];if(mode!="legacy"&&mode!="compact")throw std::runtime_error("unknown comparison mode");
 const std::filesystem::path out=argv[5];if(std::filesystem::exists(out))throw std::runtime_error("immutable probe output exists");std::filesystem::create_directories(out);
 eventio::Json report={{"schema","rinkfit.recovery_runtime_representation_probe.v1"},{"mode",mode},{"Ceres_graph_or_solver_created",false},{"native_precision","float64 unchanged"},{"source_packet",argv[3]},{"source_model",argv[2]}};
 const auto start=Clock::now();auto previous=start;
 auto mark=[&](const std::string&name,eventio::Json details=eventio::Json::object()){const auto now=Clock::now();auto j=joint_memory_sample(name);j["stage_seconds"]=std::chrono::duration<double>(now-previous).count();j["elapsed_seconds"]=std::chrono::duration<double>(now-start).count();j["details"]=std::move(details);std::ofstream f(out/"memory_stages.jsonl",std::ios::app);f<<j.dump()<<std::endl;std::cout<<j.dump()<<std::endl;previous=now;return j;};
 mark("before_load");const auto cfg=eventio::read_json(argv[4]);
 auto model=JointEventModel::load(argv[2]);mark("model_loaded");
 auto packet=JointPacket::load(argv[3],model);mark("packet_loaded",{{"raw_cache_capacity_bytes",joint_packet_raw_cache_bytes()}});
 report["loaded_packet_buffers"]=packet_buffers(packet);report["source_packet_sha256"]=packet.source_hash;report["source_model_sha256"]=eventio::file_sha256(argv[2]);mark("packet_buffers_counted");
 assert_joint_board_source_qualification(packet,argv[3]);
 const auto guard=joint_source_guard_snapshot(packet.source_json);
 const auto full_fingerprint=joint_source_membership_fingerprint(packet.source_json);
 if(joint_source_membership_fingerprint(guard)!=full_fingerprint)throw std::runtime_error("compact membership fingerprint differs");
 const auto recalled=joint_source_recall_marker(packet.source_json,argv[3],true),guard_recalled=joint_source_recall_marker(guard,argv[3],true);
 if(recalled!=guard_recalled||!recalled.empty())throw std::runtime_error("source recall guard mismatch or recall");
 report["source_guard_membership_fingerprint"]=full_fingerprint;report["native_source_qualification_guard_passed"]=true;report["full_and_compact_recall_markers_equal"]=true;mark("qualification_verified");
 if(mode=="compact"){const auto released=joint_packet_raw_cache_bytes();release_joint_packet_raw_cache();packet.source_json=guard;packet.source_json["schema"]=packet.schema;report["raw_cache_released_bytes"]=released;}
 report["runtime_packet_buffers"]=packet_buffers(packet);mark("runtime_representation_ready",{{"raw_cache_capacity_bytes",joint_packet_raw_cache_bytes()}});
 auto objective_cfg=cfg;objective_cfg.erase("resume_engineering_checkpoint");objective_cfg.erase("max_inner_iterations");objective_cfg.erase("memory_probe_capture_repeats");objective_cfg.erase("memory_probe_full_constraints");
 JointALBuilder b(std::move(model),std::move(packet),false,false);b.set_engineering_identity(string_sha(objective_cfg.dump()));
 if(cfg.contains("resume_engineering_checkpoint")){std::ifstream in(cfg.at("resume_engineering_checkpoint").get<std::string>());const auto state=read_checkpoint(in);b.restore_state(state.state);}
 mark("AL_builder_ready");auto registry=b.parameter_blocks();std::size_t scalars=0;for(const auto&r:registry)scalars+=r.size;
 const auto initial_registry_sha=registry_sha(registry);report["registered_blocks"]=registry.size();report["registered_scalar_count"]=scalars;report["registered_f64_bytes"]=scalars*sizeof(double);report["registered_names_and_bitwise_values_sha256"]=initial_registry_sha;
 report["inequality_schema_sha256"]=names_sha(b.schema().inequality_ids);report["equality_schema_sha256"]=names_sha(b.schema().equality_ids);
 report["constraint_pixel_count"]=b.schema().pixel_count;report["constraint_g_count"]=b.schema().inequality_ids.size();report["constraint_h_count"]=b.schema().equality_ids.size();
 StateSnapshot legacy;std::string initial_token;
 {auto baseline=b.capture_state();initial_token=baseline.token();verify_registry(baseline,registry);if(mode=="legacy")legacy=std::move(baseline);}
 report["complete_initial_state_token"]=initial_token;report["every_registered_parameter_bitwise_equal_to_complete_snapshot"]=true;mark("complete_snapshot_verified");
 std::unique_ptr<JointCompactAcceptedState>compact;
 if(mode=="compact"){compact=std::make_unique<JointCompactAcceptedState>(b);if(!compact->matches_current_values())throw std::runtime_error("compact initial values differ");report["compact_value_capacity_bytes"]=compact->value_capacity_bytes();report["compact_registry_blocks"]=compact->registry_blocks();}
 mark("accepted_state_ready");
 const int repeats=cfg.value("memory_probe_capture_repeats",3);double captures=0;
 for(int k=0;k<repeats;++k){const auto t=Clock::now();if(compact)compact->capture();else legacy=b.capture_state();captures+=std::chrono::duration<double>(Clock::now()-t).count();}
 report["accepted_capture_repeats"]=repeats;report["accepted_capture_total_seconds"]=captures;report["accepted_capture_mean_seconds"]=captures/std::max(1,repeats);mark("accepted_capture_repeated");
 mutate_trial(registry);if(registry_sha(registry)==initial_registry_sha)throw std::runtime_error("candidate isolation probe did not mutate state");
 if(compact){if(compact->matches_current_values())throw std::runtime_error("accepted state aliases mutable trial");compact->restore(b);}else b.restore_state(legacy);
 if(registry_sha(registry)!=initial_registry_sha||b.capture_state().token()!=initial_token)throw std::runtime_error("accepted rollback changed full state");
 report["accepted_candidate_mutable_values_isolated"]=true;report["complete_state_token_after_rollback"]=initial_token;report["every_registered_parameter_bitwise_restored"]=true;mark("rollback_verified");
 report["explicit_metrics"]=export_joint_recovery_explicit_metrics(b,out/"explicit_metrics",cfg.value("num_threads",8));mark("explicit_metrics_complete");
 const auto before_checkpoint=mark("before_checkpoint");const auto checkpoint_start=Clock::now();
 {std::ofstream f(out/"engineering_checkpoint.json");OuterState outer;outer.phase=Phase::kPrecision;outer.rho=0;write_checkpoint(f,{b.capture_state(),{},outer,string_sha(objective_cfg.dump())});if(!f)throw std::runtime_error("checkpoint write failed");}
 report["checkpoint_seconds"]=std::chrono::duration<double>(Clock::now()-checkpoint_start).count();report["checkpoint_bytes"]=std::filesystem::file_size(out/"engineering_checkpoint.json");report["checkpoint_sha256"]=eventio::file_sha256(out/"engineering_checkpoint.json");report["checkpoint_start_memory"]=before_checkpoint;
 report["final_memory"]=mark("checkpoint_complete");report["complete_final_state_token"]=b.capture_state().token();if(report["complete_final_state_token"]!=initial_token)throw std::runtime_error("read-only probe changed state");
 if(cfg.value("memory_probe_full_constraints",false)){
  auto frozen=b.capture_state();const auto values=b.evaluate_constraints(frozen);auto write=[&](const char*name,const std::vector<double>&v){std::ofstream f(out/name,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(double));f.close();return eventio::Json{{"sha256",eventio::file_sha256(out/name)},{"elements",v.size()}};};
  report["full_constraints"]={{"g",write("full_constraints_g.f64",values.g)},{"h",write("full_constraints_h.f64",values.h)}};mark("full_constraints_evaluated_with_existing_frozen_copy");
 }
 report["total_seconds"]=std::chrono::duration<double>(Clock::now()-start).count();eventio::atomic_json(out/"receipt.json",report);std::cout<<"probe_complete "<<mode<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
