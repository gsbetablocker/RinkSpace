#include "rinkfit/event_packet.hpp"
#include <cassert>
#include <functional>
#include <iostream>

using rinkfit::eventio::Json;
int main(int argc,char** argv)try{
  if(argc!=5)throw std::runtime_error("requires full packet, resolved semantics, native query and writable test directory");
  const auto packet_path=std::filesystem::path(argv[1]),semantic_path=std::filesystem::path(argv[2]);
  const auto query_path=std::filesystem::path(argv[3]),work=std::filesystem::path(argv[4]);
  std::filesystem::create_directories(work);
  const auto original=rinkfit::eventio::read_json(semantic_path);
  auto load=[&](Json sem,const std::string& name){
    for(const char* key:{"original_material_index","resolved_material_index","resolved_variant_index","confirmed_bindings"}){auto& value=sem.at(key).at("file");const auto path=std::filesystem::path(value.get<std::string>());if(path.is_relative())value=(semantic_path.parent_path()/path).string();}
    const auto file=work/(name+".json");rinkfit::eventio::atomic_json(file,sem);
    rinkfit::EventPacket packet(packet_path);packet.manifest["semantic_packet"]["sha256"]=rinkfit::eventio::file_sha256(file);
    packet.load_semantics(file);return packet;
  };
  auto positive=load(original,"positive");
  positive.load_query(query_path,query_path.parent_path().parent_path()/"query_initialization.json");
  assert(positive.rows.size()==182017&&positive.material_labels.size()==171&&positive.queries.size()==961);
  assert(positive.groups.size()==9173+87967&&positive.formal_residual_rows==214112);
  assert(positive.material[11534]==30&&positive.model_material(11534)==24);
  assert(positive.variant_ids.at(positive.variant[11534])=="w015"&&positive.model_variant_id(11534)=="w012");
  for(std::size_t i=0;i<positive.rows.size();++i)if(i!=11534){assert(positive.model_material(i)==positive.material[i]);assert(positive.model_variant_id(i)==positive.variant_ids[positive.variant[i]]);}
  std::size_t rejected=0;
  auto reject=[&](Json sem,const std::string& name,const std::string& expected){
    bool caught=false;try{load(sem,name);}catch(const std::runtime_error& error){caught=std::string(error.what()).find(expected)!=std::string::npos;if(!caught)std::cerr<<name<<": unexpected rejection: "<<error.what()<<"\n";}
    assert(caught);++rejected;
  };
  const auto binding_path=semantic_path.parent_path()/original.at("confirmed_bindings").at("file").get<std::string>();
  const auto binding=rinkfit::eventio::read_json(binding_path);
  auto attach_binding=[&](Json& sem,const Json& content,const std::string& name){const auto file=work/(name+"_bindings.json");rinkfit::eventio::atomic_json(file,content);sem["confirmed_bindings"]={{"file",file.string()},{"sha256",rinkfit::eventio::file_sha256(file)}};};
  auto invalid=original;auto forged=binding;
  const auto human_ref=binding.at("bindings").at(0).at("direct_human_confirmation");
  auto human=rinkfit::eventio::read_json(human_ref.at("path").get<std::string>());human["source"]="generated guess, no direct user confirmation";
  const auto human_path=work/"forged_human.json";rinkfit::eventio::atomic_json(human_path,human);
  forged["bindings"][0]["direct_human_confirmation"]={{"path",human_path.string()},{"sha256",rinkfit::eventio::file_sha256(human_path)}};
  attach_binding(invalid,forged,"forged");reject(invalid,"forged_confirmation","lacks direct same-row human confirmation");
  invalid=original;invalid["original_material_index"]["sha256"]=std::string(64,'0');reject(invalid,"changed_raw_material","SHA256 mismatch");
  invalid=original;invalid["arrays"]["weights"]["sha256"]=std::string(64,'0');reject(invalid,"changed_W","changed frozen source/W/query descriptor");
  auto write_u32=[&](Json& sem,const std::string& name,std::vector<std::uint32_t> values){const auto file=work/(name+".u32");std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));out.close();sem["resolved_material_index"]["file"]=file.string();sem["resolved_material_index"]["sha256"]=rinkfit::eventio::file_sha256(file);};
  invalid=original;auto material=positive.resolved_material;material[11533]=24;write_u32(invalid,"extra_row",material);reject(invalid,"unconfirmed_second_row","unconfirmed material change");
  invalid=original;auto profile=binding;std::size_t row=0;while(positive.kind[row]!=2)++row;profile["bindings"][0]["source_row"]=row;attach_binding(invalid,profile,"profile");reject(invalid,"profile_override","duplicated/profile/ordered");
  assert(rejected==5);
  Json report={{"passed",true},{"observations",182017},{"raw_material_labels",171},{"resolved_rows",1},{"profile_queries",961},{"formal_W_rank",214112},{"negative_cases_rejected",rejected},{"semantic_sha256",rinkfit::eventio::file_sha256(semantic_path)}};
  rinkfit::eventio::atomic_json(work/"material_resolution_loader_report.json",report);
  std::cout<<report.dump()<<"\n";
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 2;}
