#include "rinkfit/joint_representative_resume_identity_v1.hpp"
#include "rinkfit/joint_recovery_write_cadence_v1.hpp"
#include <fstream>
#include <iostream>
#include <limits>
using namespace rinkfit;
void need(bool x,const char*why){if(!x)throw std::runtime_error(why);}
template<class F>void rejects(F f,const char*why){bool failed=false;try{f();}catch(const std::exception&){failed=true;}need(failed,why);}
int main(int argc,char**argv){try{
 StatePayload p;p.identity={"source","model","budget","statistics","scales","observed"};p.parameters.push_back({"shape",{1.,2.},false});
 RunCheckpoint c;c.state=StateSnapshot(p);c.outer.rho=0.;c.runtime_fingerprint="old";
 eventio::Json saved={{"state_token",c.state.token()},{"checkpoint",{{"sha256","checkpoint"}}},{"model",{{"sha256","model"}}}};
 eventio::Json cfg={{"source_uv_capture_objective","observed"}};
 need(validate_joint_representative_resume_identity_v1(c,saved,cfg,"old",false).at("exact_objective_continuation"),"same objective rejected");
 rejects([&]{validate_joint_representative_resume_identity_v1(c,saved,cfg,"new",false);},"unbound target accepted");
 cfg["representative_objective_transition_v1"]={{"schema","rinkfit.representative_objective_transition.v1"},{"kind","complete_primal_warm_start"},{"from_state_token",c.state.token()},{"from_runtime_fingerprint","old"},{"from_checkpoint_sha256","checkpoint"},{"from_model_sha256","model"},{"to_objective_id","observed"},{"reason","withdraw protected training observations"}};
 need(!validate_joint_representative_resume_identity_v1(c,saved,cfg,"new",false).at("exact_objective_continuation").get<bool>(),"new objective called exact");
 auto altered=cfg;altered["representative_objective_transition_v1"]["from_state_token"]="wrong";rejects([&]{validate_joint_representative_resume_identity_v1(c,saved,altered,"new",false);},"wrong state accepted");
 auto d=c;d.outer.lambda={1.};rejects([&]{validate_joint_representative_resume_identity_v1(d,saved,cfg,"new",false);},"dual discarded");d=c;d.outer.rho=1.;rejects([&]{validate_joint_representative_resume_identity_v1(d,saved,cfg,"new",false);},"penalty discarded");
 altered=cfg;altered["source_uv_capture_objective"]="formal";rejects([&]{validate_joint_representative_resume_identity_v1(c,saved,altered,"new",false);},"mode silently changed");
 JointRecoveryWriteCadenceV1 cadence;cadence.configure(10.);need(cadence.due(100.),"first save missing");cadence.completed(100.);need(!cadence.due(109.),"early write");need(cadence.due(110.),"due save missing");need(cadence.due(101.,true),"forced latest save missing");rejects([&]{cadence.configure(std::numeric_limits<double>::quiet_NaN());},"NaN cadence accepted");
 if(argc>1){std::ifstream f(argv[1]);auto actual=read_checkpoint(f);need(actual.outer.rho==0.&&actual.outer.lambda.empty()&&actual.outer.nu.empty(),"r9 outer contract changed");std::cout<<"actual_token="<<actual.state.token()<<" fingerprint="<<actual.runtime_fingerprint<<" blocks="<<actual.state.data().parameters.size()<<"\n";}
 std::cout<<"recovery identity and write cadence passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
