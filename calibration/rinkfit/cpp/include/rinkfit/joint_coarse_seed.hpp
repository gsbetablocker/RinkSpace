#pragma once
#include "rinkfit/joint_builder.hpp"
#include <cmath>
#include <set>

namespace rinkfit {
// Only numerical auxiliaries are imported. Immutable pixel/identity/statistical
// source fields remain the original loaded packet throughout this operation.
struct JointCoarseSeedScope {
 eventio::Json original_recapture_rows;
 bool overridden=false;
 void restore(JointProblemBuilder&b){if(overridden){b.packet.source_json["recapture_rows"]=original_recapture_rows;overridden=false;}}
};
namespace coarse_seed_detail {
inline std::filesystem::path verified_ref(const eventio::Json&ref,const std::filesystem::path&root){
 if(!ref.is_object()||!ref.contains("file")||!ref.contains("sha256"))throw std::runtime_error("coarse seed dependency requires file and SHA256");
 auto file=std::filesystem::path(ref.at("file").get<std::string>());if(file.is_relative())file=root/file;
 const auto digest=ref.at("sha256").get<std::string>();if(digest.size()!=64)throw std::runtime_error("coarse seed dependency SHA256 length");
 eventio::verify(file,digest);return std::filesystem::weakly_canonical(file);
}
template<class T>std::vector<T> array(const std::filesystem::path&root,const eventio::Json&files,const std::string&name,std::size_t count){
 const auto&ref=files.at(name);const auto file=root/name;const auto data=eventio::bytes(file);
 if(data.size()!=count*sizeof(T)||data.size()!=ref.at("size_bytes").get<std::size_t>()||eventio::sha256(data)!=ref.at("sha256").get<std::string>())throw std::runtime_error("coarse seed binary size/SHA mismatch: "+file.string());
 const std::uint16_t endian=1;if(*reinterpret_cast<const std::uint8_t*>(&endian)!=1)throw std::runtime_error("coarse seed requires little-endian host");
 std::vector<T>out(count);if(!data.empty())std::memcpy(out.data(),data.data(),data.size());return out;
}
inline bool same_double(double a,double b){return std::memcmp(&a,&b,sizeof(double))==0;}
inline std::size_t index(const eventio::Json&value){if(!value.is_number_integer()||(value.is_number_integer()&&!value.is_number_unsigned()&&value.get<std::int64_t>()<0))throw std::runtime_error("coarse seed row must be a nonnegative integer");return value.get<std::size_t>();}
}
inline JointCoarseSeedScope install_joint_coarse_nuisance_seed(JointProblemBuilder&b,const eventio::Json&stage,const std::filesystem::path&current_model,const std::filesystem::path&current_packet,const std::filesystem::path&out,const std::filesystem::path&stage_root){
 JointCoarseSeedScope scope;if(!stage.contains("coarse_nuisance_seed")&&!stage.contains("recapture_rows_override"))return scope;
 eventio::Json receipt={{"schema","rinkfit.joint_coarse_nuisance_seed_install.v1"},{"passed",false},{"source_file_changed",false},{"source_UV_PTS_identity_E_W_Query_changed",false},{"strict_Query_or_physical_solution_claimed",false}};
 try{
  if(!stage.value("source_capture_engineering",false)||!stage.value("recapture_new_sources",false)||stage.value("observed_row_time_new_sources",false)||!stage.contains("coarse_nuisance_seed")||!stage.contains("recapture_rows_override"))throw std::runtime_error("coarse seed and explicit recapture override require protected source_capture_engineering stage");
  const auto&refs=stage.at("coarse_nuisance_seed");receipt["dependencies"]=refs;
  const auto manifest_file=coarse_seed_detail::verified_ref(refs.at("manifest"),stage_root),binding_file=coarse_seed_detail::verified_ref(refs.at("binding"),stage_root);
  const auto execution_file=coarse_seed_detail::verified_ref(refs.at("execution"),stage_root),report_file=coarse_seed_detail::verified_ref(refs.at("recapture_report"),stage_root),ledger_file=coarse_seed_detail::verified_ref(refs.at("recapture_ledger"),stage_root),old_stage_file=coarse_seed_detail::verified_ref(refs.at("stage"),stage_root);
  const auto old_out=manifest_file.parent_path().parent_path();
  if(execution_file.parent_path()!=old_out||report_file.parent_path()!=old_out||ledger_file.parent_path()!=old_out)throw std::runtime_error("coarse seed evidence must belong to its actual capture output");
  const auto manifest=eventio::read_json(manifest_file),binding=eventio::read_json(binding_file),execution=eventio::read_json(execution_file),report=eventio::read_json(report_file),old_stage=eventio::read_json(old_stage_file);
  if(manifest.at("schema")!="rinkfit.joint_initial_nuisance_state.v1"||manifest.at("source_UV_changed")!=false||manifest.at("original_s01_nuisance_changed")!=false)throw std::runtime_error("coarse seed is not unchanged-source initial nuisance capture");
  if(execution.at("schema")!="rinkfit.joint_backend_execution.v1"||execution.at("source_capture_engineering")!=true||execution.at("formal_Phase_I_II_passed")!=false||report.at("schema")!="rinkfit.joint_initial_nuisance_recapture.v1"||report.at("source_UV_changed")!=false||report.at("material_identity_changed")!=false||report.at("original_s01_nuisance_changed")!=false||report.at("projection_policy")!="true_sensor_RS"||report.at("finite_Query_closure_required_for_success")!=false)throw std::runtime_error("coarse seed lacks actual true-RS capture evidence");
  const auto source_model=coarse_seed_detail::verified_ref(binding.at("parent_model"),binding_file.parent_path()),source_packet=coarse_seed_detail::verified_ref(binding.at("packet"),binding_file.parent_path()),source_executable=coarse_seed_detail::verified_ref(binding.at("executable"),binding_file.parent_path());
  const auto model_sha=eventio::file_sha256(current_model),packet_sha=eventio::file_sha256(current_packet);
  if(model_sha!=binding.at("parent_model").at("sha256")||packet_sha!=binding.at("packet").at("sha256")||packet_sha!=b.packet.source_hash||manifest.at("input_packet_sha256")!=packet_sha||execution.at("model_sha256")!=model_sha||execution.at("packet_sha256")!=packet_sha||execution.at("backend_sha256")!=binding.at("executable").at("sha256"))throw std::runtime_error("coarse seed parent model/packet/executable binding mismatch");
  const auto&command=execution.at("command");
  if(command.size()<7||std::filesystem::weakly_canonical(command.at(0).get<std::string>())!=source_executable||std::filesystem::weakly_canonical(command.at(1).get<std::string>())!=source_model||std::filesystem::weakly_canonical(command.at(2).get<std::string>())!=source_packet||std::filesystem::weakly_canonical(command.at(3).get<std::string>())!=old_out||std::filesystem::weakly_canonical(command.at(6).get<std::string>())!=old_stage_file)throw std::runtime_error("coarse seed actual execution command provenance mismatch");
  const auto old_fixed=old_stage.at("fix_shot_nuisance").get<std::vector<std::string>>();
  if(!old_stage.value("source_capture_engineering",false)||!old_stage.value("recapture_new_sources",false)||old_stage.value("observed_row_time_new_sources",false)||std::find(old_fixed.begin(),old_fixed.end(),"s01")==old_fixed.end())throw std::runtime_error("coarse seed original stage did not protect S01 nuisance");
  const auto count=b.packet.rows.size(),start=b.packet.source_json.at("new_source_start").get<std::size_t>();
  if(manifest.at("raw_rows")!=count||manifest.at("new_source_start")!=start||start>count)throw std::runtime_error("coarse seed raw-row/prefix identity mismatch");
  const auto original=b.packet.source_json.at("recapture_rows");const auto original_rows=original.get<std::vector<std::size_t>>();std::set<std::size_t>original_set(original_rows.begin(),original_rows.end()),overrides;
  if(original_set.size()!=original_rows.size())throw std::runtime_error("immutable original recapture scope has duplicate rows");
  const auto&requested=stage.at("recapture_rows_override");if(!requested.is_array()||requested.empty())throw std::runtime_error("coarse seed requires an explicit nonempty subset recapture scope");
  for(const auto&value:requested){const auto i=coarse_seed_detail::index(value);if(i<start||i>=count||!b.packet.rows.at(i).active||!original_set.count(i)||!overrides.insert(i).second)throw std::runtime_error("coarse override must be unique active new-source subset of immutable packet recapture_rows");}
  auto station=coarse_seed_detail::array<double>(manifest_file.parent_path(),manifest.at("files"),"station.f64",count),row=coarse_seed_detail::array<double>(manifest_file.parent_path(),manifest.at("files"),"row.f64",count),alpha=coarse_seed_detail::array<double>(manifest_file.parent_path(),manifest.at("files"),"alpha.f64",count);auto path=coarse_seed_detail::array<std::uint32_t>(manifest_file.parent_path(),manifest.at("files"),"path.u32",count);
  for(std::size_t i=0;i<count;++i){const auto&o=b.packet.rows[i];if(!std::isfinite(station[i])||station[i]<0||station[i]>1||!std::isfinite(row[i])||row[i]<0||row[i]>1079||!std::isfinite(alpha[i])||o.material<0||std::size_t(o.material)>=b.model.materials.size()||path[i]>=b.model.materials.at(o.material).paths.size())throw std::runtime_error("coarse seed auxiliary range/path invalid at row "+std::to_string(i));
   if(i<start&&(o.shot_id!="s01"||!coarse_seed_detail::same_double(station[i],b.station[i])||!coarse_seed_detail::same_double(row[i],b.row[i])||!coarse_seed_detail::same_double(alpha[i],b.alpha[i])||path[i]!=std::uint32_t(o.path)))throw std::runtime_error("coarse seed changes original S01 nuisance prefix at row "+std::to_string(i));
   if(o.kind==1&&!coarse_seed_detail::same_double(station[i],o.station))throw std::runtime_error("coarse seed moved fixed entity station at row "+std::to_string(i));
  }
  std::ifstream ledger(ledger_file,std::ios::binary);std::string line;std::set<std::size_t>seen;std::size_t valid=0,RSvalid=0,failed=0;eventio::Json inherited_failures=eventio::Json::array();
  while(std::getline(ledger,line)){if(line.empty())continue;const auto item=eventio::Json::parse(line);const auto i=coarse_seed_detail::index(item.at("joint_row"));if(!original_set.count(i)||!seen.insert(i).second)throw std::runtime_error("coarse capture ledger row scope/duplicate mismatch");const auto&o=b.packet.rows.at(i);
   if(item.at("source_dataset_id")!=o.source_dataset_id||item.at("source_row_id")!=o.source_row_id||item.at("shot_id")!=o.shot_id||item.at("native_pts")!=o.pts||item.at("source_uv").get<Vec2<double>>()!=o.uv||item.at("source_UV_unchanged")!=true||item.at("material_identity_unchanged")!=true||item.at("current_path")!=path[i]||!coarse_seed_detail::same_double(item.at("current_station").get<double>(),station[i]))throw std::runtime_error("coarse capture ledger identity/UV/numerical state mismatch at row "+std::to_string(i));
   const bool ok=item.at("numerical_initialization_seed_valid").get<bool>(),rs=item.at("RS_valid").get<bool>();valid+=ok;RSvalid+=rs;failed+=!ok;
   if(!ok||!rs){inherited_failures.push_back(i);if(!overrides.count(i))throw std::runtime_error("coarse seed would silently reuse an invalid uncaptured source row "+std::to_string(i));}
  }
  if(!ledger.eof()||seen!=original_set||report.at("requested_new_rows")!=seen.size()||report.at("valid_rows")!=valid||report.at("invalid_rows")!=failed||report.at("RS_valid_rows")!=RSvalid)throw std::runtime_error("coarse capture complete ledger/report counts mismatch");
  receipt["raw_rows"]=count;receipt["new_source_start"]=start;receipt["parent_model_sha256"]=model_sha;receipt["input_packet_sha256"]=packet_sha;receipt["source_capture_executable_sha256"]=binding.at("executable").at("sha256");receipt["seed_files"]=manifest.at("files");receipt["original_S01_prefix_bitwise_equal"]=true;receipt["original_recapture_rows"]=original;receipt["original_recapture_row_count"]=original_rows.size();receipt["recapture_rows_override"]=requested;receipt["scope_is_memory_only"]=true;receipt["inherited_true_RS_rows"]=RSvalid;receipt["inherited_invalid_rows_requiring_recapture"]=inherited_failures;receipt["complete_original_source_denominator_preserved"]=true;receipt["passed"]=true;
  // Install only after every dependency, source identity and old failure is checked.
  b.station=std::move(station);b.row=std::move(row);b.alpha=std::move(alpha);
  for(std::size_t i=0;i<count;++i){auto&o=b.packet.rows[i];o.station=b.station[i];o.row=b.row[i];o.alpha=b.alpha[i];o.path=int(path[i]);}
  scope.original_recapture_rows=original;scope.overridden=true;b.packet.source_json["recapture_rows"]=requested;
  eventio::atomic_json(out/"coarse_nuisance_seed_install.json",receipt);return scope;
 }catch(const std::exception&e){scope.restore(b);receipt["passed"]=false;receipt["failure"]=e.what();eventio::atomic_json(out/"coarse_nuisance_seed_install.json",receipt);throw;}
}
} // namespace rinkfit
