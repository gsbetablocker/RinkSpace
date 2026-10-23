#include "rinkfit/joint_engineering_inner_boundary.hpp"
#include <ceres/ceres.h>
#include <chrono>
#include <iostream>
using namespace rinkfit;
namespace fs=std::filesystem;
using Json=eventio::Json;
namespace {
void require(bool ok,const std::string&message){if(!ok)throw std::runtime_error(message);}
struct TinyBuilder final:ProblemBuilder {
 FrozenIdentity id{std::string(64,'a'),std::string(64,'b'),std::string(64,'c'),std::string(64,'d'),std::string(64,'e'),"engineering_recovery_native_pixel_v1"};
 ConstraintSchema constraints{{"physical/x_le_1"},{"physical/y_eq_0"},0};std::vector<ObservationSpec>obs;double q[2]={-1.3,.5},budget=0;
 const FrozenIdentity&identity()const override{return id;}const ConstraintSchema&schema()const override{return constraints;}
 const std::vector<ObservationSpec>&observations()const override{return obs;}double*budget_parameter()override{return &budget;}
 std::vector<ParameterBlockBinding>parameter_blocks()override{return {{"physical",q,2,false}};}
 StateSnapshot capture_state()const override{StatePayload p;p.identity=id;p.parameters={{"physical",{q[0],q[1]},false}};return StateSnapshot(std::move(p));}
 void restore_state(const StateSnapshot&s)override{require(s.data().identity==id,"tiny CP identity changed");const auto&v=s.data().parameters.at(0).values;q[0]=v.at(0);q[1]=v.at(1);}
 NativeReplay replay_native(const StateSnapshot&)const override{return {};}
 ConstraintValues evaluate_constraints(const StateSnapshot&s)const override{const auto&v=s.data().parameters[0].values;return {s.token(),{v[0]-1},{v[1]}};}
 GradientReport lagrangian_gradient(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const override{return {};}
 BuildInventory build_ceres_problem(ceres::Problem&,const ObjectiveContext&)override{return {};}
};
struct ToyResidual {
 double lambda,nu,rho;
 template<class T>bool operator()(const T*q,T*r)const{
  using std::sqrt;r[0]=sqrt(T(2))*(q[0]-T(2));
  const T inequality=T(lambda)+T(rho)*(q[0]-T(1));
  r[1]=inequality>T(0)?inequality/sqrt(T(rho)):T(0);
  r[2]=sqrt(T(rho))*(q[1]+T(nu/rho));return true;
 }
};
EngineeringOuterState context(TinyBuilder&b){auto s=fresh_engineering_outer(b,Json::object());s.lambda={.4};s.nu={-.3};s.rho=10;s.previous_progress=.125;s.outer_iteration=2;return s;}
bool same_context(const EngineeringOuterState&a,const EngineeringOuterState&b){return a.lambda==b.lambda&&a.nu==b.nu&&a.rho==b.rho&&a.previous_progress==b.previous_progress&&a.outer_iteration==b.outer_iteration;}
void checkpoint_roundtrip(TinyBuilder&b,const EngineeringOuterState&outer,const fs::path&out){
 fs::create_directories(out);const std::string config(64,'f');const auto cp=out/"primal_CP.txt";
 {OuterState unused;unused.phase=Phase::kPrecision;unused.rho=0;std::ofstream stream(cp);write_checkpoint(stream,{b.capture_state(),{},unused,config});}
 const auto before_hash=eventio::file_sha256(cp);const auto ref=save_engineering_outer(b,outer,cp,out/"outer.json",config,"toy","ready_to_solve");
 TinyBuilder restored;{std::ifstream stream(cp);const auto saved=read_checkpoint(stream);restored.restore_state(saved.state);require(saved.outer.lambda.empty()&&saved.outer.nu.empty()&&saved.outer.rho==0,"primal CP imported engineering duals");}
 auto loaded=load_engineering_outer(restored,ref,cp,config);require(same_context(outer,loaded)&&outer.previous_inner_incomplete==loaded.previous_inner_incomplete,"complete CP/outer continuation changed context");
 require(restored.capture_state().token()==b.capture_state().token()&&eventio::file_sha256(cp)==before_hash,"accepted primal was changed during complete CP roundtrip");
}
}
int main(int argc,char**argv){try{
 require(argc==2,"Data output directory required");const auto out=fs::absolute(argv[1])/std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());fs::create_directories(out);
 Json cases=Json::array();
 for(int budget:{0,1}){
  TinyBuilder b;auto outer=context(b);const auto before=outer;ceres::Problem problem;
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ToyResidual,3,2>(new ToyResidual{outer.lambda[0],outer.nu[0],outer.rho}),nullptr,b.q);
  ceres::Solver::Options options;options.max_num_iterations=budget;options.linear_solver_type=ceres::DENSE_QR;options.function_tolerance=0;options.parameter_tolerance=0;options.gradient_tolerance=1e-12;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  require(summary.termination_type==ceres::NO_CONVERGENCE,"small budget did not exercise actual NO_CONVERGENCE");
  const auto boundary=engineering_inner_boundary(summary.termination_type,false,false);const auto constraints=b.evaluate_constraints(b.capture_state());
  const auto advanced=finish_engineering_inner(outer,boundary,constraints.g,constraints.h,RuntimeLimits{}.rho_max,RuntimeLimits{}.improvement_ratio,RuntimeLimits{}.penalty_factor);
  require(!advanced.dual_update_applied&&boundary.budget_exhausted&&outer.previous_inner_incomplete&&same_context(outer,before),"budget exhaustion advanced AL context");
  checkpoint_roundtrip(b,outer,out/("budget_"+std::to_string(budget)));
  cases.push_back({{"budget",budget},{"Ceres_termination",ceres::TerminationTypeToString(summary.termination_type)},{"accepted_primal",{b.q[0],b.q[1]}},{"all_five_context_members_unchanged",true},{"complete_CP_context_roundtrip",true}});
 }
 TinyBuilder b;auto outer=context(b);const auto before=outer;ceres::Problem problem;
 problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ToyResidual,3,2>(new ToyResidual{outer.lambda[0],outer.nu[0],outer.rho}),nullptr,b.q);
 ceres::Solver::Options options;options.max_num_iterations=int(RuntimeLimits{}.max_inner_iterations);options.linear_solver_type=ceres::DENSE_QR;options.function_tolerance=0;options.parameter_tolerance=0;options.gradient_tolerance=1e-12;
 ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);require(summary.termination_type==ceres::CONVERGENCE,"solved toy did not converge");
 ceres::Problem::EvaluateOptions eo;double cost=0;std::vector<double>gradient;require(problem.Evaluate(eo,&cost,nullptr,&gradient,nullptr),"fresh actual objective gradient unavailable");
 const double fresh_gradient=infinity_norm(gradient);require(fresh_gradient<=RuntimeLimits{}.inner_stationarity_late,"toy convergence did not meet fresh gradient accuracy");
 const auto constraints=b.evaluate_constraints(b.capture_state());const auto boundary=engineering_inner_boundary(summary.termination_type,false,false);
 const auto advanced=finish_engineering_inner(outer,boundary,constraints.g,constraints.h,RuntimeLimits{}.rho_max,RuntimeLimits{}.improvement_ratio,RuntimeLimits{}.penalty_factor);
 require(advanced.dual_update_applied&&!outer.previous_inner_incomplete&&outer.outer_iteration==before.outer_iteration+1&&!same_context(outer,before),"converged toy did not apply exactly one update");checkpoint_roundtrip(b,outer,out/"converged");
 for(auto termination:{ceres::FAILURE,ceres::USER_FAILURE,ceres::USER_SUCCESS}){auto stopped=context(b);const auto prior=stopped;const auto gate=engineering_inner_boundary(termination,false,false);auto change=finish_engineering_inner(stopped,gate,{1.},{1.},1e6,.5,5.);require(!change.dual_update_applied&&stopped.previous_inner_incomplete&&same_context(stopped,prior),"stop/failure advanced multipliers");}
 auto callback=context(b);const auto prior=callback;const auto gate=engineering_inner_boundary(ceres::CONVERGENCE,true,false);require(!finish_engineering_inner(callback,gate,{1.},{1.},1e6,.5,5.).dual_update_applied&&same_context(callback,prior),"callback failure advanced multipliers");
 const Json report={{"passed",true},{"budget_exhaustion_cases",cases},{"converged_toy_fresh_gradient",fresh_gradient},{"converged_toy_dual_updates",1},{"stop_failure_callback_context_preserved",true},{"production_fresh_scaled_inexact_AL_gate_implemented",false},{"production_fit_started",false}};eventio::atomic_json(out/"report.json",report);std::cout<<report.dump()<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
