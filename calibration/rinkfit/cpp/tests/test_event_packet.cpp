#include "rinkfit/event_packet.hpp"
#include <cassert>
#include <iostream>

int main(int argc,char** argv) {
  if(argc!=4)throw std::runtime_error("requires full event, semantic and query manifests");
  const auto code= rinkfit::eventpacket_detail::utf32(u8"A☃😀");
  assert((code==std::vector<std::uint32_t>{0x41,0x2603,0x1f600}));
  bool rejected=false;try{rinkfit::eventpacket_detail::utf32(std::string("\xc0\xaf",2));}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  rinkfit::EventPacket packet(argv[1]);packet.load_semantics(argv[2]);
  packet.load_query(argv[3],std::filesystem::path(argv[3]).parent_path().parent_path()/"query_initialization.json");
  assert(packet.observations.size()==182017&&packet.runs.size()==15419&&packet.queries.size()==961&&packet.statistics_complete);
  const auto semantic_path=std::filesystem::path(argv[2]);const auto original=rinkfit::eventio::read_json(semantic_path);
  auto invalid=original;invalid["immutable_source_arrays"]["source_point_ids"]["sha256_c_contiguous_array_bytes"]=std::string(64,'0');
  const auto temporary=semantic_path.parent_path().parent_path()/"model_checks"/"invalid_source_identity_test.json";
  rinkfit::eventio::atomic_json(temporary,invalid);
  packet.groups.clear();packet.runs.clear();packet.manifest["semantic_packet"]["sha256"]=rinkfit::eventio::file_sha256(temporary);
  rejected=false;try{packet.load_semantics(temporary);}catch(const std::runtime_error& error){rejected=std::string(error.what()).find("immutable source array identity mismatch")!=std::string::npos;}assert(rejected);
  std::filesystem::remove(temporary);
  std::cout<<"full source identity, kinds, W shapes/hash/rank, ordered runs and native queries passed; altered source identity rejected\n";
}
