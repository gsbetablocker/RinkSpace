#pragma once
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/event_io.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace rinkfit::event_output {
using eventio::Json;
inline void atomic_text(const std::filesystem::path& path,const std::string& content){
  auto temp=path;temp += ".tmp";
  {std::ofstream out(temp,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("checkpoint open failed: "+temp.string());
   out.write(content.data(),static_cast<std::streamsize>(content.size()));out.flush();if(!out)throw std::runtime_error("checkpoint flush failed: "+temp.string());}
  std::error_code ec;std::filesystem::rename(temp,path,ec);
  if(ec){std::filesystem::remove(path,ec);std::filesystem::rename(temp,path);}
}
inline void save_checkpoint(const std::filesystem::path& run_dir,const RunCheckpoint& checkpoint,Json extra_state){
  if(checkpoint.state.empty())throw std::runtime_error("refuse empty physical checkpoint");
  const auto clock=std::chrono::system_clock::now().time_since_epoch();
  const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(clock).count();
  const std::string generation="p"+std::to_string(static_cast<int>(checkpoint.outer.phase))+"_o"+std::to_string(checkpoint.outer.outer_iteration)+"_"+std::to_string(micros);
  const auto base=run_dir/"checkpoints",folder=base/generation;
  std::filesystem::create_directories(folder);
  std::ostringstream serialized;write_checkpoint(serialized,checkpoint);
  atomic_text(folder/"text.chk",serialized.str());
  extra_state["state_token"]=checkpoint.state.token();
  extra_state["checkpoint_schema"]="rinkfit.complete_event_checkpoint.v1.1";
  extra_state["checkpoint_generation"]=generation;
  extra_state["solver_resume_checkpoint"]="text.chk";
  extra_state["solver_resume_checkpoint_sha256"]=eventio::file_sha256(folder/"text.chk");
  extra_state["outer_resume_state"]={{"phase",static_cast<int>(checkpoint.outer.phase)},{"outer_iteration",checkpoint.outer.outer_iteration},
    {"inner_attempt",checkpoint.outer.inner_attempt},{"chart_restarts",checkpoint.outer.chart_restarts},
    {"elapsed_seconds",checkpoint.outer.elapsed_seconds},{"rho",checkpoint.outer.rho},
    {"previous_progress",std::isfinite(checkpoint.outer.previous_progress)?Json(checkpoint.outer.previous_progress):Json(nullptr)},
    {"lambda",checkpoint.outer.lambda},{"nu",checkpoint.outer.nu},{"runtime_fingerprint",checkpoint.runtime_fingerprint}};
  extra_state["best_physical_state_token"]=checkpoint.best_physical_state.empty()?Json(nullptr):Json(checkpoint.best_physical_state.token());
  extra_state["resume_includes_full_current_and_best_parameters"]=true;
  eventio::atomic_json(folder/"state.json",extra_state);
  Json manifest={{"schema","rinkfit.atomic_checkpoint_generation.v1"},{"generation",generation},
    {"current_state_token",checkpoint.state.token()},{"best_state_token",extra_state["best_physical_state_token"]},
    {"text",{{"file","text.chk"},{"sha256",eventio::file_sha256(folder/"text.chk")}}},
    {"state",{{"file","state.json"},{"sha256",eventio::file_sha256(folder/"state.json")}}},
    {"contains_multipliers",true},{"contains_full_best_state",!checkpoint.best_physical_state.empty()}};
  eventio::atomic_json(folder/"manifest.json",manifest);
  // A generation is immutable and complete before its pointer becomes visible.
  // Consumers open current.json once, then use only that generation's files.
  eventio::atomic_json(base/"current.json",{{"generation",generation},{"manifest",(folder/"manifest.json").string()},
    {"manifest_sha256",eventio::file_sha256(folder/"manifest.json")}});
  atomic_text(base/"current.chk",serialized.str());
  eventio::atomic_json(base/"current_state.json",extra_state);
}
inline void write_report(const std::filesystem::path& run_dir,const Json& report){
  std::filesystem::create_directories(run_dir);eventio::atomic_json(run_dir/"physical_report.json",report);
}
} // namespace rinkfit::event_output
