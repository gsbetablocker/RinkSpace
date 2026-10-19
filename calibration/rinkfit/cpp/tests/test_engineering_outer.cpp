#include "rinkfit/joint_engineering_outer.hpp"
#include <iostream>
#include <functional>
#include <iterator>
using namespace rinkfit;
namespace fs=std::filesystem;
using Json=eventio::Json;
struct TinyBuilder final:ProblemBuilder {
 FrozenIdentity id{std::string(64,'a'),std::string(64,'b'),std::string(64,'c'),std::string(64,'d'),std::string(64,'e'),"engineering_recovery_native_pixel_v1"};
 ConstraintSchema constraints{{"pixel/original","physical/a","physical/b"},{"RS/original","Query/original"},1};StateSnapshot state;std::vector<ObservationSpec>obs;double budget=0;
 TinyBuilder(){reset();}void reset(){StatePayload p;p.identity=id;p.parameters={{"physical",{.25,1.5},false},{"row/0",{500.},false}};p.discrete={{"finite_path","0"}};state=StateSnapshot(std::move(p));}
 const FrozenIdentity&identity()const override{return id;}const ConstraintSchema&schema()const override{return constraints;}const std::vector<ObservationSpec>&observations()const override{return obs;}double*budget_parameter()override{return &budget;}
 std::vector<ParameterBlockBinding>parameter_blocks()override{return {};}StateSnapshot capture_state()const override{return state;}void restore_state(const StateSnapshot&s)override{state=s;}
 NativeReplay replay_native(const StateSnapshot&)const override{return {};}ConstraintValues evaluate_constraints(const StateSnapshot&)const override{return {};}GradientReport lagrangian_gradient(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const override{return {};}BuildInventory build_ceres_problem(ceres::Problem&,const ObjectiveContext&)override{return {};}
};
static void check(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
static Json ref(const fs::path&p){return {{"file",fs::absolute(p).string()},{"sha256",eventio::file_sha256(p)}};}
static void write_CP(const TinyBuilder&b,const fs::path&p,const std::string&config){OuterState legacy;legacy.phase=Phase::kPrecision;legacy.rho=0;std::ofstream f(p);write_checkpoint(f,{b.capture_state(),{},legacy,config});}
int main(int argc,char**argv){try{
 check(argc==2,"output Data directory required");const fs::path out=fs::absolute(argv[1]);check(!fs::exists(out),"immutable native test output exists");fs::create_directories(out);
 TinyBuilder b;const std::string config(64,'f');const auto cp=out/"primal_CP.txt";write_CP(b,cp,config);const auto cp_sha=eventio::file_sha256(cp);
 auto state=fresh_engineering_outer(b,Json::object());check(state.lambda==std::vector<double>({0,0})&&state.nu==std::vector<double>({0,0})&&state.rho==10&&!state.previous_inner_incomplete&&std::isfinite(state.previous_progress),"fresh engineering shape/defaults");
 state.lambda={1.,2.};state.nu={-3.,4.};state.previous_progress=.125;state.outer_iteration=1;
 const auto saved=save_engineering_outer(b,state,cp,out/"outer.json",config,"motion","ready_to_solve");auto loaded=load_engineering_outer(b,saved,cp,config);
 check(loaded.lambda==state.lambda&&loaded.nu==state.nu&&loaded.rho==10&&loaded.previous_progress==.125&&loaded.outer_iteration==1&&loaded.saved_stage=="motion"&&loaded.status=="ready_to_solve","exact native sidecar roundtrip");
 check(eventio::file_sha256(cp)==cp_sha,"primal CP changed");{std::ifstream f(cp);const auto untouched=read_checkpoint(f);check(untouched.outer.rho==0&&untouched.outer.lambda.empty()&&untouched.outer.nu.empty(),"engineering sidecar leaked into old CP");}
 std::size_t rejected=0;Json cases=Json::array();auto reject=[&](const std::string&name,const std::function<void()>&fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}check(failed,("failed to reject "+name).c_str());++rejected;cases.push_back(name);};
 auto mutation=[&](const std::string&name,const std::function<void(Json&)>&edit){auto j=eventio::read_json(out/"outer.json");edit(j);auto file=out/(name+".json");eventio::atomic_json(file,j);return ref(file);};
 reject("changed_sidecar_SHA",[&](){auto bad=saved;bad["sha256"]=std::string(64,'0');load_engineering_outer(b,bad,cp,config);});
 reject("wrong_config",[&](){load_engineering_outer(b,saved,cp,std::string(64,'1'));});
 for(const auto&key:{"source","model","scales","objective"})reject(std::string("changed_")+key,[&](){TinyBuilder other;if(std::string(key)=="source")other.id.source_id="different";else if(std::string(key)=="model")other.id.model_id="different";else if(std::string(key)=="scales")other.id.scales_id="different";else other.id.objective_id="old_PHR";other.reset();load_engineering_outer(other,saved,cp,config);});
 reject("same_shape_reordered_g",[&](){TinyBuilder other;std::swap(other.constraints.inequality_ids[1],other.constraints.inequality_ids[2]);load_engineering_outer(other,saved,cp,config);});
 reject("same_shape_reordered_h",[&](){TinyBuilder other;std::swap(other.constraints.equality_ids[0],other.constraints.equality_ids[1]);load_engineering_outer(other,saved,cp,config);});
 reject("stale_primal_state",[&](){TinyBuilder other;auto p=other.state.data();p.parameters[0].values[0]+=.1;other.state=StateSnapshot(std::move(p));load_engineering_outer(other,saved,cp,config);});
 reject("not_ready_to_solve",[&](){auto r=mutation("not_ready",[](Json&j){j["status"]="inner_finished_before_multiplier_update";});load_engineering_outer(b,r,cp,config);});
 reject("rho_zero",[&](){auto r=mutation("zero_rho",[](Json&j){j["rho"]=0;});load_engineering_outer(b,r,cp,config);});
 reject("negative_previous_progress",[&](){auto r=mutation("negative_progress",[](Json&j){j["previous_progress"]=-1;});load_engineering_outer(b,r,cp,config);});
 reject("negative_outer_iteration",[&](){auto r=mutation("negative_iteration",[](Json&j){j["outer_iteration"]=-1;});load_engineering_outer(b,r,cp,config);});
 reject("wrong_multiplier_shape",[&](){auto r=mutation("wrong_shape",[](Json&j){j["arrays"]["lambda"]["shape"]={3};});load_engineering_outer(b,r,cp,config);});
 auto altered_array=[&](const std::string&name,const std::string&array,const std::vector<double>&values){const auto p=out/(name+".f64");{std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));}return mutation(name,[&](Json&j){j["arrays"][array]["file"]=p.string();j["arrays"][array]["sha256"]=eventio::file_sha256(p);});};
 reject("negative_lambda_with_valid_SHA",[&](){auto r=altered_array("negative_lambda","lambda",{-1,2});load_engineering_outer(b,r,cp,config);});
 reject("nonfinite_nu_with_valid_SHA",[&](){auto r=altered_array("NaN_nu","nu",{std::numeric_limits<double>::quiet_NaN(),0});load_engineering_outer(b,r,cp,config);});
 reject("truncated_binary_with_valid_SHA",[&](){auto r=altered_array("short_lambda","lambda",{1});load_engineering_outer(b,r,cp,config);});
 reject("changed_CP_SHA",[&](){auto p=out/"different_CP.txt";std::ifstream in(cp);std::string text((std::istreambuf_iterator<char>(in)),{});text.back()='x';std::ofstream(p)<<text;load_engineering_outer(b,saved,p,config);});
 reject("corrupt_complete_CP_token_even_with_new_SHA",[&](){auto p=out/"bad_token_CP.txt";std::ifstream in(cp);std::string text((std::istreambuf_iterator<char>(in)),{});const auto token=b.capture_state().token();auto bad=token;bad[0]=bad[0]=='0'?'1':'0';text.replace(text.find(token),token.size(),bad);std::ofstream(p)<<text;auto r=mutation("bad_CP_token",[&](Json&j){j["checkpoint"]=ref(p);});load_engineering_outer(b,r,p,config);});
 reject("legacy_PHR_duals_not_importable",[&](){auto p=out/"old_dual_CP.txt";OuterState old;old.phase=Phase::kPrecision;old.rho=10;old.lambda={1};{std::ofstream f(p);write_checkpoint(f,{b.capture_state(),{},old,config});}save_engineering_outer(b,state,p,out/"forbidden_legacy.json",config,"motion","ready_to_solve");});
 reject("save_requires_after_update_status",[&](){save_engineering_outer(b,state,cp,out/"forbidden_status.json",config,"motion","updating");});
 reject("save_immutable_sidecar",[&](){save_engineering_outer(b,state,cp,out/"outer.json",config,"motion","ready_to_solve");});

 auto incomplete=loaded;incomplete.previous_inner_incomplete=true;const auto incomplete_ref=save_engineering_outer(b,incomplete,cp,out/"incomplete_inner.json",config,"motion","ready_to_solve");const auto retained_context=load_engineering_outer(b,incomplete_ref,cp,config);check(retained_context.previous_inner_incomplete&&retained_context.lambda==loaded.lambda&&retained_context.nu==loaded.nu&&retained_context.rho==loaded.rho&&retained_context.outer_iteration==loaded.outer_iteration,"STOP/FAIL exact context changed or lost incomplete flag");
 auto next=loaded;next.outer_iteration=2;next.lambda[0]+=1;const auto nextref=save_engineering_outer(b,next,cp,out/"next_outer.json",config,"motion_shared_geometry","ready_to_solve");const auto switched=load_engineering_outer(b,nextref,cp,config);check(switched.saved_stage=="motion_shared_geometry"&&switched.lambda[0]==2&&switched.outer_iteration==2,"stage switch with independent engineering state");
 const Json report={{"passed",true},{"fresh_shape_and_rho10",true},{"exact_roundtrip",true},{"legacy_primal_CP_unchanged",true},{"stage_switch_allowed_and_recorded",true},{"STOP_FAIL_context_preserved_for_next_solve",true},{"rejection_cases_passed",rejected},{"rejection_cases",cases},{"full_recovery_or_904k_packet_loaded",false},{"fit_started",false}};eventio::atomic_json(out/"report.json",report);std::cout<<report.dump()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
