#include "rinkfit/joint_report_identity.hpp"
#include <iostream>
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("one JSON fixture required");
  const auto value=rinkfit::eventio::read_json(argv[1]);
  std::cout<<rinkfit::joint_replay_objective_binding(value.at("packet"),
   value.at("source_packet_sha256"),value.at("report"),value.at("exclusions")).dump()<<'\n';
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
