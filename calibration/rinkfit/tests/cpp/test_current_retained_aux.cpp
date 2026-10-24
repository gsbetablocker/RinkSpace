#include "rinkfit/joint_packet.hpp"
#include <cstring>
#include <iostream>
using namespace rinkfit;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("requires retained-aux fixture index");
 const auto request=eventio::read_json(argv[1]);JointEventModel model;
 model.materials.resize(1);model.materials[0].paths.resize(1);
 model.source_json={{"source_clock",{{"source_video_sha256",std::string(64,'a')}}}};
 int rejected=0;
 for(const auto&c:request.at("cases")){
  model.materials[0].label="fixture-print";model.materials[0].paths.resize(3);model.source_json.erase("printed_material_path_refinements");
  if(c.value("refine",false))model.source_json["printed_material_path_refinements"]={{"0",{{"label","fixture-print"},{"original_path_count",1},{"children",{0,1}}}}};
  const auto mp=c.at("model").get<std::string>(),pp=c.at("packet").get<std::string>();
  const bool reject=c.at("reject");bool caught=false;
  try{
   const auto p=c.value("legacy",false)?JointPacket::load(pp,model):JointPacket::load_current_v1(c.at("binding"),model,mp);
   if(reject)throw std::runtime_error("invalid retained auxiliary accepted");
   const auto expected=std::vector<double>{.5,-.01602934459078972,1.0143455807336128};
   if(p.rows.size()!=expected.size())throw std::runtime_error("raw rows lost");
   for(std::size_t i=0;i<expected.size();++i)if(std::memcmp(&p.rows[i].station,&expected[i],sizeof(double)))throw std::runtime_error("retained station mutated");
   if(p.rows[1].active||p.rows[2].active||p.rows[1].uv!=Vec2<double>{200,300})throw std::runtime_error("retained membership/pixels mutated");
  }catch(const std::runtime_error&e){
   if(!reject||std::string(e.what()).find(c.value("expected_error",std::string("invalid joint observation/nuisance")))==std::string::npos)throw;
   caught=true;++rejected;
  }
  if(reject&&!caught)throw std::runtime_error("missing rejection");
 }
 std::cout<<"actual current packet loader: complete inactive raw station bytes retained; physical/legacy/nonfinite/row/UV checks rejected "<<rejected<<" cases\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
