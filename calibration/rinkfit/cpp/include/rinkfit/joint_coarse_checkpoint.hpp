#pragma once
#include "rinkfit/joint_builder.hpp"
#include <ceres/ceres.h>
#include <iostream>
#if defined(_WIN32)
// Declare the two required Win32 calls without windows.h/min/max macros.
extern "C" __declspec(dllimport) int __stdcall MoveFileExW(const wchar_t*,const wchar_t*,unsigned long);
extern "C" __declspec(dllimport) unsigned long __stdcall GetLastError();
#endif
namespace rinkfit {
inline void joint_coarse_atomic_json(const std::filesystem::path&file,const eventio::Json&value){
 const auto pending=std::filesystem::path(file.string()+".pending");
 {std::ofstream stream(pending,std::ios::binary|std::ios::trunc);if(!stream)throw std::runtime_error("cannot open coarse checkpoint commit");stream<<value.dump(2)<<'\n';stream.flush();if(!stream)throw std::runtime_error("cannot flush coarse checkpoint commit");}
#if defined(_WIN32)
 // MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH. Never unlink old first.
 if(!::MoveFileExW(pending.c_str(),file.c_str(),1ul|8ul))throw std::runtime_error("coarse checkpoint atomic replace failed Win32="+std::to_string(::GetLastError()));
#else
 std::filesystem::rename(pending,file);
#endif
}

inline eventio::Json joint_coarse_checkpoint_ref(const std::filesystem::path&file){const auto p=std::filesystem::weakly_canonical(file);return {{"file",p.string()},{"sha256",eventio::file_sha256(p)},{"size_bytes",std::filesystem::file_size(p)}};}
inline eventio::Json joint_coarse_checkpoint_context(const std::filesystem::path&out,const std::filesystem::path&model,const std::filesystem::path&packet,const std::filesystem::path&executable,const std::filesystem::path&stage){
 return {{"output",std::filesystem::absolute(out).string()},{"input_model",joint_coarse_checkpoint_ref(model)},{"source_packet",joint_coarse_checkpoint_ref(packet)},{"executable",joint_coarse_checkpoint_ref(executable)},{"stage",joint_coarse_checkpoint_ref(stage)}};
}
// A bounded two-slot journal. The mutable manifest always names a fully written
// immutable-for-that-generation model/station pair, with fixed row/alpha/path
// referenced from this run's genuine initial_nuisance_state. No replay per step.
class JointCoarseCheckpointCallback final:public ceres::IterationCallback {
 JointProblemBuilder&b_;std::filesystem::path out_,root_;eventio::Json dependencies_,initial_,seed_files_,latest_;std::size_t generation_=0;int slot_=1;std::size_t saves_=0;
 static void binary(const std::filesystem::path&file,const double*data,std::size_t count){std::ofstream f(file,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write coarse checkpoint station");f.write(reinterpret_cast<const char*>(data),static_cast<std::streamsize>(count*sizeof(double)));f.flush();if(!f)throw std::runtime_error("cannot flush coarse checkpoint station");}
 void verify_fixed_seed(const std::string&name,const void*data,std::size_t bytes){const auto file=out_/"initial_nuisance_state"/name;const auto&d=initial_.at("files").at(name);const auto contents=eventio::bytes(file);if(contents.size()!=bytes||contents.size()!=d.at("size_bytes").get<std::size_t>()||eventio::sha256(contents)!=d.at("sha256").get<std::string>()||std::memcmp(contents.data(),data,bytes)!=0)throw std::runtime_error("coarse checkpoint fixed seed differs from actual initial "+name);seed_files_[name]=joint_coarse_checkpoint_ref(file);}
 void save(int iteration,double cost,const std::string&kind){
  const int next=1-slot_;const auto folder=root_/("slot"+std::to_string(next));std::filesystem::create_directories(folder);
  joint_coarse_atomic_json(folder/"model.json",b_.model.exact_json());binary(folder/"station.f64",b_.station.data(),b_.station.size());
  auto files=seed_files_;files["model.json"]=joint_coarse_checkpoint_ref(folder/"model.json");files["station.f64"]=joint_coarse_checkpoint_ref(folder/"station.f64");
  latest_={{"schema","rinkfit.joint_source_capture_checkpoint.v1"},{"generation",generation_++},{"state_kind",kind},{"iteration",iteration},{"cost",std::isfinite(cost)?eventio::Json(cost):eventio::Json(nullptr)},{"raw_rows",b_.packet.rows.size()},{"source_packet_sha256",b_.packet.source_hash},{"input_packet_sha256",b_.packet.source_hash},{"objective_mode",objective_mode},{"dependencies",dependencies_},{"initial_nuisance_manifest",joint_coarse_checkpoint_ref(out_/"initial_nuisance_state/manifest.json")},{"files",files},{"station_is_current_Ceres_parameter_state",true},{"row_alpha_path_are_unchanged_numerical_seeds",true},{"new_true_RS_or_finite_Query_rebinding_performed",false},{"source_UV_E_W_Query_membership_changed",false},{"likelihood_groups_deferred",objective_mode=="source_capture_feasibility"},{"quality_accepted",false},{"formal_Phase_I_II_passed",false},{"optimizer_internal_trust_region_history_saved",false},{"restart_semantics","exact saved physical/numerical primal warm start with same source, objective and stage; fresh Ceres trust region, no quality attestation"},{"journal_slots",2},{"Windows_manifest_commit","MoveFileExW REPLACE_EXISTING|WRITE_THROUGH; old manifest never unlinked"},{"live_reader_policy","pin/copy referenced model+station and verify recorded SHA before diagnostics; two-slot files rotate"},{"stop_requested",false}};
  // Commit the small manifest last; a crash before this leaves the other slot valid.
  joint_coarse_atomic_json(root_/"manifest.json",latest_);slot_=next;++saves_;
 }
 public:
 std::string objective_mode,error;bool stopped=false;
 JointCoarseCheckpointCallback(JointProblemBuilder&b,const eventio::Json&context,const std::string&mode):b_(b),out_(context.at("output").get<std::string>()),root_(out_/"latest_checkpoint"),dependencies_(context),objective_mode(mode){
  if(mode!="source_capture_engineering"&&mode!="source_capture_feasibility")throw std::runtime_error("coarse checkpoint requires explicit source capture objective");
  for(const auto*key:{"input_model","source_packet","executable","stage"}){const auto&d=context.at(key);eventio::verify(d.at("file").get<std::string>(),d.at("sha256").get<std::string>());}
  if(context.at("source_packet").at("sha256")!=b_.packet.source_hash)throw std::runtime_error("coarse checkpoint source packet mismatch");
  if(std::filesystem::exists(root_))throw std::runtime_error("coarse checkpoint journal already exists for this immutable run");std::filesystem::create_directories(root_);
  dependencies_["before_model"]=joint_coarse_checkpoint_ref(out_/"before/model.json");
  initial_=eventio::read_json(out_/"initial_nuisance_state/manifest.json");
  if(initial_.at("schema")!="rinkfit.joint_initial_nuisance_state.v1"||initial_.at("input_packet_sha256")!=b_.packet.source_hash||initial_.at("raw_rows")!=b_.packet.rows.size()||initial_.at("source_UV_changed")!=false||initial_.at("original_s01_nuisance_changed")!=false)throw std::runtime_error("coarse checkpoint requires genuine protected initial nuisance state");
  verify_fixed_seed("row.f64",b_.row.data(),b_.row.size()*sizeof(double));verify_fixed_seed("alpha.f64",b_.alpha.data(),b_.alpha.size()*sizeof(double));std::vector<std::uint32_t>paths;paths.reserve(b_.packet.rows.size());for(const auto&o:b_.packet.rows)paths.push_back(static_cast<std::uint32_t>(o.path));verify_fixed_seed("path.u32",paths.data(),paths.size()*sizeof(std::uint32_t));
  const auto station_file=out_/"initial_nuisance_state/station.f64";eventio::verify(station_file,initial_.at("files").at("station.f64").at("sha256").get<std::string>());
  if(std::filesystem::file_size(station_file)!=b_.station.size()*sizeof(double))throw std::runtime_error("coarse initial station length mismatch");
  save(-1,NAN,"complete_initial_primal_before_Ceres");
 }
 ceres::CallbackReturnType operator()(const ceres::IterationSummary&s)override{try{
  if(s.step_is_successful)save(s.iteration,s.cost,"last_accepted_Ceres_source_capture_primal");
  if(std::filesystem::exists(out_/"STOP_REQUESTED")){
   stopped=true;latest_["stop_requested"]=true;latest_["stopped_after_iteration"]=s.iteration;latest_["stopped_on_accepted_step"]=s.step_is_successful;joint_coarse_atomic_json(root_/"manifest.json",latest_);
   joint_coarse_atomic_json(out_/"source_capture_stop.json",{{"schema","rinkfit.coarse_capture_safe_stop.v1"},{"last_accepted_checkpoint",joint_coarse_checkpoint_ref(root_/"manifest.json")},{"source_packet_sha256",b_.packet.source_hash},{"quality_accepted",false},{"finite_Query_or_true_RS_passed",false}});return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
  }
  return ceres::SOLVER_CONTINUE;
 }catch(const std::exception&e){error=e.what();std::cerr<<"coarse checkpoint failure: "<<error<<std::endl;return ceres::SOLVER_ABORT;}}
 eventio::Json report()const{return {{"enabled",true},{"checkpoint_manifest",(root_/"manifest.json").string()},{"complete_primal_saves",saves_},{"bounded_journal_slots",2},{"stop_requested",stopped},{"callback_error",error},{"new_true_RS_or_finite_Query_rebinding_performed",false},{"quality_accepted",false}};}
};
} // namespace rinkfit
