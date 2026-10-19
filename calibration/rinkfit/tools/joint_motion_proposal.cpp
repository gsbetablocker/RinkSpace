#define NOMINMAX
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/event_io.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <filesystem>
using namespace rinkfit;
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("MODEL COMPLETE_CHECKPOINT PROPOSAL OUT");
 std::filesystem::path out=argv[4];if(std::filesystem::exists(out))throw std::runtime_error("immutable numeric proposal state exists");
 eventio::Json model,proposal;{std::ifstream f(argv[1]);f>>model;}{std::ifstream f(argv[3]);f>>proposal;}
 if(proposal.at("schema")!="rinkfit.joint_shot_motion_numerical_proposal.v1"||proposal.at("shot_id")!="s09")throw std::runtime_error("unsupported numeric proposal contract");
 if(eventio::file_sha256(argv[1])!=proposal.at("parent_model").at("sha256")||eventio::file_sha256(argv[2])!=proposal.at("parent_checkpoint").at("sha256"))throw std::runtime_error("numeric proposal exact parent hash mismatch");
 std::ifstream input(argv[2]);auto checkpoint=read_checkpoint(input);const auto parent_token=checkpoint.state.token();StatePayload payload=checkpoint.state.data();
 std::map<std::string,std::vector<double>> after,before;
 for(const auto&c:proposal.at("changes")){
  const int index=c.at("block");auto&b=model.at("blocks").at(index);const std::string name=c.at("name");
  if(b.at("name")!=name||name.rfind("shot/s09/",0)!=0||(name!="shot/s09/reference_angles"&&name.rfind("shot/s09/control/",0)!=0)||b.value("constant",false))throw std::runtime_error("proposal must only change original free s09 motion blocks");
  auto old=c.at("before_values").get<std::vector<double>>(),next=c.at("after_values").get<std::vector<double>>();
  if(old!=b.at("values").get<std::vector<double>>()||old.size()!=next.size()||!before.emplace(name,old).second)throw std::runtime_error("proposal changed fixed registry or stale values");
  for(std::size_t j=0;j<next.size();++j)if(!std::isfinite(next[j])||next[j]<b.at("lower").at(j).get<double>()||next[j]>b.at("upper").at(j).get<double>())throw std::runtime_error("proposal outside original parameter domain");
  for(const auto&j:b.at("fixed_indices"))if(next.at(j.get<std::size_t>())!=old.at(j.get<std::size_t>()))throw std::runtime_error("proposal modified original fixed component");
  after.emplace(name,next);b["values"]=next;
 }
 std::size_t replaced=0,fixed=0;for(auto&p:payload.parameters){auto it=after.find(p.name);if(it==after.end()){++fixed;continue;}if(p.equality_auxiliary||p.values!=before.at(p.name))throw std::runtime_error("proposal physical state differs from exact parent");p.values=it->second;++replaced;}
 if(replaced!=after.size()||!replaced)throw std::runtime_error("proposal omitted actual registered shot parameters");
 auto proposed=StateSnapshot(std::move(payload));const auto&original=checkpoint.state.data();const auto&trial=proposed.data();
 if(original.discrete!=trial.discrete||original.root_charts!=trial.root_charts||original.parameters.size()!=trial.parameters.size())throw std::runtime_error("proposal changed source discrete/root/state registry");
 for(std::size_t i=0;i<trial.parameters.size();++i){const auto&a=original.parameters[i];const auto&b=trial.parameters[i];if(a.name!=b.name||a.equality_auxiliary!=b.equality_auxiliary||(!after.count(a.name)&&a.values!=b.values))throw std::runtime_error("proposal changed source/budget/non-s09 complete state");}
 checkpoint.state=std::move(proposed);std::filesystem::create_directories(out);{std::ofstream f(out/"model.json");f<<model.dump(2);}{std::ofstream f(out/"numerical_resume_checkpoint.json");write_checkpoint(f,checkpoint);}
 eventio::Json receipt={{"schema","rinkfit.unaccepted_same_scope_camera_numerical_proposal_state.v1"},{"parent_state_token",parent_token},{"proposal_state_token",checkpoint.state.token()},{"source_proposal",{{"file",argv[3]},{"sha256",eventio::file_sha256(argv[3])}}},{"source_packet",proposal.at("source_packet")},{"changed_S09_motion_blocks",replaced},{"exactly_unchanged_other_named_blocks",fixed},{"budget_source_UV_E_W_queries_all_four_aux_AL_duals_rho_phase_and_fingerprint_unchanged",true},{"source_branches_root_charts_and_frozen_identity_unchanged",true},{"model_sha256",eventio::file_sha256(out/"model.json")},{"checkpoint_sha256",eventio::file_sha256(out/"numerical_resume_checkpoint.json")},{"accepted_original_full_AL_or_physical_closure_claimed",false},{"required_next","Optimize only allowed shot motion/source auxiliaries on unchanged original full AL and compare original parent merit, E, RS, Query, domains and S01 before acceptance"}};
 {std::ofstream f(out/"proposal_state_receipt.json");f<<receipt.dump(2);}std::cout<<receipt.dump(2);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();return 1;}}
