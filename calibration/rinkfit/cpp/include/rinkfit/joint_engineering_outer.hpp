#pragma once
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/event_io.hpp"
#include <algorithm>
#include <fstream>
#include <limits>

namespace rinkfit {
// Engineering multipliers belong to the nonpixel physical/native g and RS/Query
// h only. They are independent of the legacy PHR OuterState in the primal CP.
struct EngineeringOuterState {
 std::vector<double> lambda,nu;
 double rho=10.,previous_progress=std::numeric_limits<double>::max();
 std::size_t outer_iteration=0;
 bool previous_inner_incomplete=false;
 std::string saved_stage,status="ready_to_solve";
};
namespace engineering_outer_detail {
inline void require(bool ok,const std::string&why){if(!ok)throw std::runtime_error("engineering outer state: "+why);}
inline void put(eventio::SHA256Stream&h,const std::string&s){h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
inline eventio::Json identity(const FrozenIdentity&i){return {{"source_id",i.source_id},{"model_id",i.model_id},{"budget_id",i.budget_id},{"statistics_id",i.statistics_id},{"scales_id",i.scales_id},{"objective_id",i.objective_id}};}
inline std::size_t g_count(const ProblemBuilder&b){const auto&s=b.schema();require(s.pixel_count<=s.inequality_ids.size(),"pixel prefix exceeds g schema");return s.inequality_ids.size()-s.pixel_count;}
// Stream the complete ordered schema; never duplicate the large ID vectors.
inline std::string schema_sha(const ProblemBuilder&b){const auto&s=b.schema();g_count(b);eventio::SHA256Stream h;put(h,"rinkfit.engineering_constraint_schema_order.v1|"+std::to_string(s.pixel_count)+"|");for(const auto*v:{&s.inequality_ids,&s.equality_ids}){put(h,std::to_string(v->size())+"[");for(const auto&id:*v){require(!id.empty(),"empty constraint identity");put(h,std::to_string(id.size())+":"+id);}put(h,"]");}return h.finish();}
inline void validate(const ProblemBuilder&b,const EngineeringOuterState&s){
 require(b.identity().objective_id=="engineering_recovery_native_pixel_v1","wrong objective; old PHR outer state is forbidden");
 require(s.lambda.size()==g_count(b)&&s.nu.size()==b.schema().equality_ids.size(),"multiplier/schema shape mismatch");
 require(std::isfinite(s.rho)&&s.rho>0&&std::isfinite(s.previous_progress)&&s.previous_progress>=0,"nonfinite/invalid rho or previous progress");
 for(const auto x:s.lambda)require(std::isfinite(x)&&x>=0,"nonfinite or negative inequality multiplier");
 for(const auto x:s.nu)require(std::isfinite(x),"nonfinite equality multiplier");
}
struct CheckpointIdentity{std::string sha,token;};
inline CheckpointIdentity checkpoint(const ProblemBuilder&b,const std::filesystem::path&path,const std::string&config){
 require(!config.empty(),"empty engineering configuration identity");
 // End this temporary snapshot before materializing the saved CP, so only one
 // complete parameter registry is live at a time alongside the builder.
 const std::string current_token=b.capture_state().token();
 const auto hash=eventio::file_sha256(path);std::ifstream stream(path);require(bool(stream),"missing complete primal checkpoint");const auto cp=read_checkpoint(stream);
 require(cp.runtime_fingerprint==config,"checkpoint configuration mismatch");
 require(cp.outer.lambda.empty()&&cp.outer.nu.empty()&&cp.outer.rho==0&&cp.best_physical_state.empty(),"legacy PHR dual/rho/best state must remain absent");
 require(cp.state.data().identity==b.identity()&&cp.state.token()==current_token,"checkpoint is not the current complete primal state");
 return {hash,cp.state.token()};
}
inline std::filesystem::path verified(const eventio::Json&ref){auto p=std::filesystem::path(ref.at("file").get<std::string>());require(p.is_absolute(),"sidecar references must be absolute");eventio::verify(p,ref.at("sha256"));return p;}
inline eventio::Json write_vector(const std::filesystem::path&p,const std::vector<double>&v){
 require(!std::filesystem::exists(p),"immutable multiplier binary exists");std::ofstream f(p,std::ios::binary);require(bool(f),"cannot create multiplier binary");if(!v.empty())f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(double));f.flush();require(bool(f),"cannot complete multiplier binary");f.close();return {{"file",std::filesystem::absolute(p).lexically_normal().string()},{"sha256",eventio::file_sha256(p)},{"dtype","<f8"},{"shape",{v.size()}},{"layout","C"},{"byte_order","little"}};
}
inline std::vector<double>read_vector(const eventio::Json&d,std::size_t n){
 require(d.at("dtype")=="<f8"&&d.at("shape")==eventio::Json::array({n})&&d.value("layout",std::string("C"))=="C"&&d.value("byte_order",std::string("little"))=="little"&&!d.contains("offset_bytes"),"multiplier binary ABI/shape mismatch");
 const auto p=std::filesystem::path(d.at("file").get<std::string>());require(p.is_absolute()&&std::filesystem::file_size(p)==n*sizeof(double),"multiplier binary size mismatch");return eventio::read_array<double>(p.parent_path(),d,"<f8");
}
}

inline EngineeringOuterState fresh_engineering_outer(const ProblemBuilder&b,const eventio::Json&cfg){
 EngineeringOuterState s;s.lambda.assign(engineering_outer_detail::g_count(b),0.);s.nu.assign(b.schema().equality_ids.size(),0.);
 s.rho=cfg.value("engineering_outer_initial_rho",10.);engineering_outer_detail::validate(b,s);return s;
}
// Save accepted primal plus the exact context BEFORE the next inner solve.
// A completed inner has already updated multipliers/penalty; STOP/FAIL retains
// its unchanged context and sets previous_inner_incomplete. Never update again
// before resuming. The ordinary rho0/empty-dual primal CP is never edited.
inline eventio::Json save_engineering_outer(const ProblemBuilder&b,const EngineeringOuterState&s,
 const std::filesystem::path&cp_path,const std::filesystem::path&sidecar_path,
 const std::string&config_identity,const std::string&stage,const std::string&status){
 using namespace engineering_outer_detail;validate(b,s);require(!stage.empty(),"missing parameter stage");require(status=="ready_to_solve","only an updated ready_to_solve outer state is resumable");
 const auto out=std::filesystem::absolute(sidecar_path).lexically_normal();require(!std::filesystem::exists(out),"immutable outer sidecar exists");
 const auto cp=checkpoint(b,cp_path,config_identity);const auto stem=out.stem().string();const auto lambda_file=out.parent_path()/(stem+".lambda.f64"),nu_file=out.parent_path()/(stem+".nu.f64");
 require(!std::filesystem::exists(lambda_file)&&!std::filesystem::exists(nu_file),"immutable outer binary exists");std::filesystem::create_directories(out.parent_path());
 const auto lambda=write_vector(lambda_file,s.lambda),nu=write_vector(nu_file,s.nu);
 const eventio::Json doc={{"schema","rinkfit.engineering_outer_state.v1"},{"status",status},{"ready_to_solve",true},{"state_placement","accepted_primal_and_context_before_next_inner_solve"},{"stage",stage},
 {"checkpoint",{{"file",std::filesystem::absolute(cp_path).lexically_normal().string()},{"sha256",cp.sha}}},{"state_token",cp.token},{"identity",identity(b.identity())},{"configuration_identity",config_identity},
 {"constraint_schema_order_sha256",schema_sha(b)},{"pixel_inequalities_excluded",b.schema().pixel_count},{"g_count",s.lambda.size()},{"h_count",s.nu.size()},
 {"rho",s.rho},{"previous_progress",s.previous_progress},{"outer_iteration",s.outer_iteration},{"previous_inner_incomplete",s.previous_inner_incomplete},{"arrays",{{"lambda",lambda},{"nu",nu}}},
 {"legacy_PHR_outer_state_imported",false},{"primal_CP_duals_or_rho_changed",false},{"source_qualification_issued",false},{"formal_acceptance_claimed",false}};
 eventio::atomic_json(out,doc);return {{"file",out.string()},{"sha256",eventio::file_sha256(out)}};
}

inline EngineeringOuterState load_engineering_outer(const ProblemBuilder&b,const eventio::Json&ref,
 const std::filesystem::path&cp_path,const std::string&config_identity){
 using namespace engineering_outer_detail;const auto sidecar=verified(ref);const auto j=eventio::read_json(sidecar);
 require(j.at("schema")=="rinkfit.engineering_outer_state.v1"&&j.at("status")=="ready_to_solve"&&j.value("ready_to_solve",false)&&j.at("state_placement")=="accepted_primal_and_context_before_next_inner_solve","outer state is not an updated next-solve checkpoint");
 require(!j.value("legacy_PHR_outer_state_imported",true)&&!j.value("primal_CP_duals_or_rho_changed",true)&&!j.value("source_qualification_issued",true)&&!j.value("formal_acceptance_claimed",true),"forbidden legacy/source/pass claim");
 require(j.at("configuration_identity")==config_identity&&j.at("identity")==identity(b.identity()),"configuration/source/model/scales/objective identity mismatch");
 require(j.at("constraint_schema_order_sha256")==schema_sha(b)&&j.at("pixel_inequalities_excluded")==b.schema().pixel_count&&j.at("g_count")==g_count(b)&&j.at("h_count")==b.schema().equality_ids.size(),"ordered constraint schema mismatch");
 eventio::verify(cp_path,j.at("checkpoint").at("sha256"));const auto cp=checkpoint(b,cp_path,config_identity);require(cp.sha==j.at("checkpoint").at("sha256")&&cp.token==j.at("state_token"),"complete checkpoint SHA/token mismatch");
 EngineeringOuterState s;s.lambda=read_vector(j.at("arrays").at("lambda"),g_count(b));s.nu=read_vector(j.at("arrays").at("nu"),b.schema().equality_ids.size());
 s.rho=j.at("rho");s.previous_progress=j.at("previous_progress");require(j.at("outer_iteration").is_number_unsigned()||(j.at("outer_iteration").is_number_integer()&&j.at("outer_iteration").get<std::int64_t>()>=0),"invalid outer iteration");s.outer_iteration=j.at("outer_iteration");
 require(j.contains("previous_inner_incomplete")&&j.at("previous_inner_incomplete").is_boolean(),"missing/inaccurate incomplete-inner context flag");s.previous_inner_incomplete=j.at("previous_inner_incomplete").get<bool>();
 s.saved_stage=j.at("stage").get<std::string>();s.status=j.at("status").get<std::string>();require(!s.saved_stage.empty(),"saved parameter stage missing");validate(b,s);return s;
}
}
