"""Native small-state regressions shared by manual import and accepted rollback."""
from pathlib import Path
import os,shutil,subprocess,unittest

ROOT=Path(__file__).resolve().parents[3]
SOURCE=r'''
#include "rinkfit/joint_global_phr_state_v1.hpp"
#include "rinkfit/joint_current_parameter_transport_v1.hpp"
#include <iostream>
using J=rinkfit::eventio::Json;
struct Saved{std::map<std::string,std::string>discrete;};
struct Snapshot{Saved value;const Saved&data()const{return value;}std::string token()const{return "source-token";}};
struct Builder{J engineering_global_phr_state_v1;struct Core{struct Packet{std::string source_hash=std::string(64,'a');}packet;}core;J registry={{"inequality_ids",{"c0/order","tripod/outside"}},{"equality_ids",{"physical_relation/test"}}};J engineering_global_constraint_registry_v1()const{return registry;}};
void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}need(rejected,"invalid migration unexpectedly accepted");}
int main(){using namespace rinkfit;try{
 
 {struct Old{std::string name;std::vector<double>values;};struct New{std::string name;std::size_t size;};
  const std::vector<Old>saved={{"shape",{1.,2.}},{"station/2",{.25}},{"station/9",{.75}},{"row/2",{33.}},{"row/9",{66.}},{"alpha/9",{.5}},{"budget_sq",{123.}},{"static_motion/0",{.2,.7}}};
  std::vector<New>target={{"shape",2},{"station/2",1},{"row/2",1},{"budget_sq",1},{"static_motion/0",2}};std::size_t checked=0;
  auto raw=[&](const std::string&name,std::size_t row,double value){need(row==9,"wrong withdrawn raw ID");need(value==(name=="station/9"?.75:name=="row/9"?66.:.5),"withdrawn complete raw value changed");++checked;};
  const auto map=current_parameter_transport_map_v1(saved,target,std::vector<std::size_t>{9},raw);need(map==std::vector<std::size_t>({0,1,3,6,7})&&checked==3,"complete withdrawal did not retain ordered model/motion state");
  rejects([&]{current_parameter_transport_map_v1(saved,target,std::vector<std::size_t>{},raw);});
  rejects([&]{current_parameter_transport_map_v1(saved,target,std::vector<std::size_t>{2,9},raw);});
  auto wrong=target;std::swap(wrong[1],wrong[2]);rejects([&]{current_parameter_transport_map_v1(saved,wrong,std::vector<std::size_t>{9},raw);});
  wrong=target;wrong[0].size=3;rejects([&]{current_parameter_transport_map_v1(saved,wrong,std::vector<std::size_t>{9},raw);});
  wrong=target;wrong.pop_back();rejects([&]{current_parameter_transport_map_v1(saved,wrong,std::vector<std::size_t>{9},raw);});
  auto bad=saved;bad[2].values[0]=.9;rejects([&]{current_parameter_transport_map_v1(bad,target,std::vector<std::size_t>{9},raw);});
 }
 J policy={{"schema","rinkfit.engineering_global_phr_policy.v1"},{"initialization","explicit_new_objective_zero_duals"},{"initial_rho",10.}};J cfg={{"engineering_global_phr_v1",policy}};
 J state={{"schema","rinkfit.engineering_global_phr_state.v1"},{"source_packet_sha256",std::string(64,'a')},{"policy_sha256",joint_global_phr_digest_v1(policy)},{"inequality_ids",{"c0/order","tripod/outside"}},{"equality_ids",{"physical_relation/test"}},{"rho",20.},{"lambda",{.25,.5}},{"nu",{-.3}},{"outer_iteration",3},{"previous_primal_inf",.2},{"last_progress",nullptr}};
 Snapshot s;s.value.discrete={{"joint_packet_sha256",std::string(64,'a')},{"engineering_global_phr_state_v1",state.dump()}};Builder b;
 auto receipt=restore_joint_global_phr_small_state_v1(b,s,cfg,false);need(b.engineering_global_phr_state_v1==state&&receipt.at("multipliers_restored_exactly")==true,"same objective lost rho/dual/progress");
 {auto missing=s;missing.value.discrete.erase("engineering_global_phr_state_v1");rejects([&]{restore_joint_global_phr_small_state_v1(b,missing,cfg,false);});}
 rejects([&]{restore_joint_global_phr_small_state_v1(b,s,cfg,true);});
 auto transition=[&](J target,const std::string&action){target["representative_objective_transition_v1"]={{"schema","rinkfit.representative_objective_transition.v1"},{"kind","complete_primal_warm_start"},{"from_state_token",s.token()},{"global_phr_state",{{"action",action},{"from_state_sha256",joint_global_phr_digest_v1(state)},{"reason","actual changed target"}}}};return target;};
 {auto changed=b;changed.core.packet.source_hash=std::string(64,'b');const auto c=transition(cfg,"preserve_exact");rejects([&]{restore_joint_global_phr_small_state_v1(changed,s,c,true);});}
 {auto changed=b;changed.registry["inequality_ids"][0]="other/id";rejects([&]{restore_joint_global_phr_small_state_v1(changed,s,cfg,false);});}
 {auto c=cfg;c["engineering_global_phr_v1"]["initial_rho"]=30.;c=transition(c,"preserve_exact");rejects([&]{restore_joint_global_phr_small_state_v1(b,s,c,true);});}
 {auto c=transition(cfg,"reset_for_new_objective");c["representative_objective_transition_v1"]["global_phr_state"]["from_state_sha256"]=std::string(64,'0');rejects([&]{restore_joint_global_phr_small_state_v1(b,s,c,true);});}
 {auto target=b;auto c=cfg;c["engineering_global_phr_v1"]["initial_rho"]=30.;c=transition(c,"reset_for_new_objective");const auto r=restore_joint_global_phr_small_state_v1(target,s,c,true);need(target.engineering_global_phr_state_v1.is_null()&&r.at("new_objective_zero_duals_required")==true,"explicit new-objective reset was not recorded");need(s.value.discrete.at("engineering_global_phr_state_v1")==state.dump(),"reset changed archived source state");}
 {Snapshot r9;r9.value.discrete={{"joint_packet_sha256",std::string(64,'a')}};Builder target;restore_joint_global_phr_small_state_v1(target,r9,cfg,true);need(target.engineering_global_phr_state_v1.is_null(),"r9 invented duals before policy initialization");rejects([&]{restore_joint_global_phr_small_state_v1(target,r9,cfg,false);});}
 {auto wrong=s;auto changed=state;changed["lambda"]={-1.,.5};wrong.value.discrete["engineering_global_phr_state_v1"]=changed.dump();rejects([&]{restore_joint_global_phr_small_state_v1(b,wrong,cfg,false);});changed=state;changed["inequality_ids"]={"x","x"};wrong.value.discrete["engineering_global_phr_state_v1"]=changed.dump();rejects([&]{restore_joint_global_phr_small_state_v1(b,wrong,cfg,false);});}
 {J live=state;JointCompactSmallStateV1 accepted(live);std::uint64_t initial=1469598103934665603ull;accepted.append_token(initial);live["rho"]=40.;live["lambda"][0]=3.;live["nu"][0]=2.;need(!accepted.matches(),"accepted snapshot aliased current duals");accepted.restore();need(live==state&&accepted.matches(),"accepted rollback failed to restore dual/rho");std::uint64_t restored=1469598103934665603ull;accepted.append_token(restored);need(restored==initial,"restored compact token changed");live["rho"]=80.;accepted.capture();std::uint64_t changed=1469598103934665603ull;accepted.append_token(changed);need(changed!=initial,"global state missing from compact token");J absent;JointCompactSmallStateV1 legacy(absent);std::uint64_t base=123;legacy.append_token(base);need(base==123,"legacy null state changed numeric compact token");}
 {J registry={{"inequality_ids",{"c0/order","physical_inequality/Lane_width","tripod/outside"}},{"equality_ids",{"physical_relation/test","physical_relation/Lane_right"}}};J extension={{"schema","rinkfit.current_physical_constraint_extension.v1"},{"new_objective_identity_required",true},{"from_state_token",s.token()},{"from_global_state_sha256",joint_global_phr_digest_v1(state)},{"target_registry_sha256",joint_global_phr_digest_v1(registry)},{"added_inequality_ids",{"physical_inequality/Lane_width"}},{"added_equality_ids",{"physical_relation/Lane_right"}},{"reason","confirmed current Lane relations"}};
  const auto after=extend_joint_current_physical_constraints_v1(state,registry,extension,s.token(),b.core.packet.source_hash);need(after.at("rho")==state.at("rho")&&after.at("outer_iteration")==state.at("outer_iteration")&&after.at("lambda")==J::array({.25,0.,.5})&&after.at("nu")==J::array({-.3,0.}),"new current physical relations lost existing dual/penalty state");need(after.at("previous_primal_inf").is_null()&&after.at("current_constraint_extension_v1").at("previous_primal_inf_before_extension")==.2,"old progress reused as new-domain comparison");
  auto wrong=extension;wrong["from_global_state_sha256"]=std::string(64,'0');rejects([&]{extend_joint_current_physical_constraints_v1(state,registry,wrong,s.token(),b.core.packet.source_hash);});
  wrong=extension;wrong["added_equality_ids"]=J::array();rejects([&]{extend_joint_current_physical_constraints_v1(state,registry,wrong,s.token(),b.core.packet.source_hash);});
  auto reordered=registry;reordered["inequality_ids"]={"tripod/outside","physical_inequality/Lane_width","c0/order"};wrong=extension;wrong["target_registry_sha256"]=joint_global_phr_digest_v1(reordered);rejects([&]{extend_joint_current_physical_constraints_v1(state,reordered,wrong,s.token(),b.core.packet.source_hash);});
  rejects([&]{extend_joint_current_physical_constraints_v1(state,registry,extension,s.token(),std::string(64,'b'));});
 }
 std::cout<<"global PHR strict/manual restore, explicit migration, accepted rollback/token passed"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
'''

class GlobalPhrRestoreTests(unittest.TestCase):
 def test_native_restore_and_accepted_rollback(self):
  out=Path('D:/CodexData/RinkSpace_Data/tests/global_phr_restore_v1');out.mkdir(parents=True,exist_ok=True);source=out/'check.cpp';source.write_text(SOURCE,encoding='utf8')
  vc=Path('D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat');r=subprocess.run('cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34),capture_output=True,text=True,check=True)
  env={k.upper():v for k,v in os.environ.items()};env.update({k.upper():v for k,v in(x.split('=',1)for x in r.stdout.splitlines()if'='in x and not x.startswith('='))});compiler=shutil.which('cl.exe',path=env['PATH']);self.assertIsNotNone(compiler)
  exe=out/'check.exe';command=[compiler,'/nologo','/std:c++17','/EHsc','/O1','/MD','/utf-8','/I'+str(ROOT/'calibration/rinkfit/cpp/include'),'/ID:/CodexData/RinkSpace_Data/runtime_tools/rf_deps/nlohmann_json/Library/include',str(source),'/Fo:'+str(out/'check.obj'),'/Fe:'+str(exe)]
  r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/'build.log').write_text(r.stdout+r.stderr,encoding='utf8');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
  r=subprocess.run([str(exe)],cwd=out,capture_output=True,text=True,errors='replace');(out/'test.log').write_text(r.stdout+r.stderr,encoding='utf8');self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':unittest.main()
