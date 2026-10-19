#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/event_io.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
using namespace rinkfit;
static RuntimeLimits settings(const eventio::Json&j){RuntimeLimits l;l.max_inner_iterations=j.value("max_inner_iterations",l.max_inner_iterations);l.max_outer_iterations_per_phase=j.value("max_outer_iterations_per_phase",l.max_outer_iterations_per_phase);l.max_inner_attempts_per_outer=j.value("max_inner_attempts_per_outer",l.max_inner_attempts_per_outer);l.max_total_seconds=j.value("max_total_seconds",l.max_total_seconds);l.num_threads=j.value("num_threads",l.num_threads);l.log_ceres_progress=j.value("log_ceres_progress",true);l.inequality_kink_active=j.value("inequality_kink_active",l.inequality_kink_active);l.validate();return l;}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("CHECKPOINT ORIGINAL_RUNTIME_CONFIG NEW_OUT NEW_STOP_FILE");
 const std::filesystem::path out=argv[3];if(std::filesystem::exists(out))throw std::runtime_error("immutable runtime continuation bridge output exists");
 auto config=eventio::read_json(argv[2]);auto original=settings(config);std::ifstream stream(argv[1]);auto checkpoint=read_checkpoint(stream);
 if(checkpoint.runtime_fingerprint!=original.fingerprint())throw std::runtime_error("runtime bridge original frozen settings mismatch");
 const auto token=checkpoint.state.token();const auto old_fingerprint=checkpoint.runtime_fingerprint;auto next=original;next.max_inner_attempts_per_outer=std::numeric_limits<std::size_t>::max();next.validate();
 if(next.max_inner_attempts_per_outer<=checkpoint.outer.inner_attempt)throw std::runtime_error("no additional same-objective continuation available");
 checkpoint.runtime_fingerprint=next.fingerprint();std::filesystem::create_directories(out);{std::ofstream f(out/"resume_checkpoint.json");write_checkpoint(f,checkpoint);}
 config["max_inner_attempts_per_outer"]=next.max_inner_attempts_per_outer;config["resume_checkpoint"]=(out/"resume_checkpoint.json").string();config["stop_file"]=argv[4];
 if(settings(config).fingerprint()!=next.fingerprint())throw std::runtime_error("runtime bridge config roundtrip mismatch");
 {std::ofstream f(out/"config.json");f<<config.dump(2);}
 {std::ifstream f(out/"resume_checkpoint.json");auto check=read_checkpoint(f);if(check.state.token()!=token||check.outer.lambda!=checkpoint.outer.lambda||check.outer.nu!=checkpoint.outer.nu||check.outer.rho!=checkpoint.outer.rho||check.outer.outer_iteration!=checkpoint.outer.outer_iteration||check.outer.inner_attempt!=checkpoint.outer.inner_attempt||check.outer.phase!=checkpoint.outer.phase||check.outer.chart_restarts!=checkpoint.outer.chart_restarts||check.outer.previous_progress!=checkpoint.outer.previous_progress||check.outer.elapsed_seconds!=checkpoint.outer.elapsed_seconds)throw std::runtime_error("runtime bridge changed actual state or AL history");}
 const eventio::Json receipt={{"schema","rinkfit.same_original_objective_unbounded_inner_continuation_runtime_bridge.v1"},{"original_checkpoint",{{"file",argv[1]},{"sha256",eventio::file_sha256(argv[1])}}},{"original_runtime_config",{{"file",argv[2]},{"sha256",eventio::file_sha256(argv[2])}}},{"old_runtime_fingerprint",old_fingerprint},{"new_runtime_fingerprint",next.fingerprint()},{"old_inner_attempt_cap",original.max_inner_attempts_per_outer},{"technical_unbounded_inner_attempt_sentinel",next.max_inner_attempts_per_outer},{"original_current_attempt_counter_preserved",checkpoint.outer.inner_attempt},{"state_token_before_and_after",token},{"all_theta_four_aux_budget_duals_rho_phase_outer_history_and_elapsed_exact",true},{"all_original_stationarity_feasibility_native_E_Query_KKT_thresholds_unchanged",true},{"objective_source_geometry_statistics_or_precision_changed",false},{"multiplier_update_still_requires_original_fresh_complete_gradient",true},{"manual_safe_stop_boundary_preserved",argv[4]},{"new_checkpoint_sha256",eventio::file_sha256(out/"resume_checkpoint.json")},{"new_runtime_config_sha256",eventio::file_sha256(out/"config.json")},{"completion_claimed",false}};
 {std::ofstream f(out/"receipt.json");f<<receipt.dump(2);}std::cout<<receipt.dump(2);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();return 1;}}
