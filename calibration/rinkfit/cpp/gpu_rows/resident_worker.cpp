#include "resident_dataset.hpp"
#include "build_policy.hpp"
#include <chrono>
#include <iostream>
#include <memory>
namespace resident {
using namespace rinkfit;
using J=eventio::Json;
static std::unique_ptr<EventPacket> immutable;
static J authority,closure;
static double initialization_seconds=0;
static std::size_t request_generation=0;
static std::string hash_json(const J&j){const auto s=j.dump();return eventio::sha256(std::vector<std::uint8_t>(s.begin(),s.end()));}
static std::filesystem::path checked(const J&d){const auto p=std::filesystem::u8path(d.at(d.contains("path")?"path":"file").get<std::string>());eventio::verify(p,d.at("sha256"));return p;}
const EventPacket& dataset(const std::filesystem::path&pk,const std::filesystem::path&sm){
 if(!immutable)throw std::runtime_error("dataset not initialized");
 if(eventio::file_sha256(pk)!=authority.at("packet_sha256")||eventio::file_sha256(sm)!=authority.at("semantic_sha256"))throw std::runtime_error("dataset authority changed; start a new worker and fully validate");
 return *immutable;
}
static J initialize(const J&req){
 if(immutable)throw std::runtime_error("worker authority is immutable; reinitialize with a fresh worker");
 const auto begin=std::chrono::steady_clock::now();
 if(req.at("reader_policy_sha256")!=ORIGINAL_READER_SHA)throw std::runtime_error("unapproved reader policy");
 const auto pk=checked(req.at("packet")),sm=checked(req.at("semantic")),qm=checked(req.at("query"));
 try{
  eventio::snapshot_capture_enabled=true;
  auto p=std::make_unique<EventPacket>(pk);p->load_semantics(sm);
  if(p->manifest.at("query_packet").at("sha256")!=req.at("query").at("sha256"))throw std::runtime_error("query identity differs from packet");
  p->load_query(qm,p->root.parent_path()/"query_initialization.json");
  eventio::snapshot_capture_enabled=false;
  J assets=J::array();std::size_t retained_bytes=0;
  for(auto& [path,b]:eventio::snapshot_blobs){if(b.hash.empty())b.hash=eventio::sha256(b.bytes);retained_bytes+=b.bytes.size();assets.push_back({{"path",path},{"sha256",b.hash},{"bytes",b.bytes.size()}});}
  closure={{"schema","rinkfit.verified_immutable_source_closure.v1"},{"packet",req.at("packet")},{"semantic",req.at("semantic")},{"query",req.at("query")},{"reader_policy_sha256",ORIGINAL_READER_SHA},{"assets",assets}};
  authority={{"packet_sha256",req.at("packet").at("sha256")},{"semantic_sha256",req.at("semantic").at("sha256")},{"query_sha256",req.at("query").at("sha256")},{"reader_policy_sha256",ORIGINAL_READER_SHA},{"closure_sha256",hash_json(closure)}};
  immutable=std::move(p);initialization_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  const auto out=std::filesystem::u8path(req.at("output").get<std::string>());std::filesystem::create_directories(out);eventio::atomic_json(out/"source_closure.json",closure);
  return {{"authority",authority},{"initialization_seconds",initialization_seconds},{"retained_files",assets.size()},{"retained_bytes",retained_bytes},{"raw_rows",immutable->rows.size()},{"active_rows",immutable->active_rows.size()},{"complete_source_validation_performed",true}};
 }catch(...){eventio::snapshot_capture_enabled=false;eventio::snapshot_blobs.clear();immutable.reset();throw;}
}
static J transport_state_identity(const J&req){
 const auto cp=checked(req.at("input"));const auto doc=eventio::read_json(cp);
 if(doc.at("schema")!="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1")throw std::runtime_error("complete transport checkpoint required");
 if(req.at("authority")!=authority)throw std::runtime_error("authority closure/policy mismatch; new worker/full validation required");
 checked(doc.at("model"));const auto lin=checked(doc.at("linearization"));const auto lm=eventio::read_json(lin);
 checked(lm.at("packet"));checked(lm.at("semantic"));
 if(lm.at("packet").at("sha256")!=authority.at("packet_sha256")||lm.at("semantic").at("sha256")!=authority.at("semantic_sha256"))throw std::runtime_error("transport source authority mismatch; new worker required");
 const auto qm=std::filesystem::u8path(immutable->manifest.at("query_packet").at("manifest").get<std::string>());
 if(eventio::file_sha256(qm)!=authority.at("query_sha256"))throw std::runtime_error("query file authority changed");
 J identity{{"checkpoint_sha256",eventio::file_sha256(cp)},{"model_sha256",doc.at("model").at("sha256")},{"model",doc.at("model")},{"packet",lm.at("packet")},{"semantic",lm.at("semantic")},{"linearization",doc.at("linearization")},{"arrays",J::object()}};
 for(const char*key:{"station","branch","row","alpha"}){
  const auto& d=doc.at("row_state_files").at(key);
  if(d.at("shape")!=J::array({182017})||d.at("dtype")!=(std::string(key)=="branch"?"<u4":"<f8"))throw std::runtime_error("transport full raw ledger dtype/shape required");
  eventio::verify(cp.parent_path()/std::filesystem::u8path(d.at("file").get<std::string>()),d.at("sha256"));identity["arrays"][key]=d.at("sha256");
 }
 if(req.at("expected_state")!=identity)throw std::runtime_error("transport checkpoint/model/four-auxiliary identity mismatch");
 return identity;
}
static J state_identity(const J&req){
 if(req.at("op")=="whole_uv"||req.at("op")=="profile_implicit")return transport_state_identity(req);
 const auto path=checked(req.at("input"));const auto doc=eventio::read_json(path);const bool values=req.at("op")=="values";
 const auto inpath=values?checked(doc.at("input")):path;
 const auto input=eventio::read_json(inpath);const auto dir=inpath.parent_path();
 if(values){for(const char* key:{"model","packet","semantic"}){checked(doc.at(key));if(doc.at(key)!=input.at(key))throw std::runtime_error(std::string("values qbar/input full reference mismatch: ")+key);}}
 if(input.at("schema")!="rinkfit.qbar_sparse_linearization_input.v1")throw std::runtime_error("state schema");
 if(req.at("authority")!=authority)throw std::runtime_error("authority closure/policy mismatch; new worker/full validation required");
 checked(input.at("packet"));checked(input.at("semantic"));checked(input.at("model"));
 if(input.at("packet").at("sha256")!=authority.at("packet_sha256")||input.at("semantic").at("sha256")!=authority.at("semantic_sha256"))throw std::runtime_error("state uses a different source authority");
 const auto qpath=std::filesystem::u8path(immutable->manifest.at("query_packet").at("manifest").get<std::string>());
 if(eventio::file_sha256(qpath)!=authority.at("query_sha256"))throw std::runtime_error("query file authority changed");
 J identity{{"input_sha256",eventio::file_sha256(inpath)},{"model_sha256",input.at("model").at("sha256")},{"arrays",J::object()}};
 for(const char* key:{"station","path","row","alpha","mode"}){const auto&d=input.at("arrays").at(key);if(d.at("shape")!=J::array({182017}))throw std::runtime_error("full raw ledger arrays required");eventio::verify(dir/std::filesystem::u8path(d.at("file").get<std::string>()),d.at("sha256"));identity["arrays"][key]=d.at("sha256");}
 if(input.contains("source_rows")){
  const auto& d=input.at("source_rows");if(d.is_array())identity["selected_rows_sha256"]=hash_json(d);else {eventio::verify(dir/std::filesystem::u8path(d.at("file").get<std::string>()),d.at("sha256"));identity["selected_rows_sha256"]=d.at("sha256");}
 }else identity["selected_rows_sha256"]="all_active";
 if(values){for(const char* key:{"linearization_station","linearization_row","source_rows"}){const auto&d=doc.at(key);eventio::verify(path.parent_path()/std::filesystem::u8path(d.at("file").get<std::string>()),d.at("sha256"));identity[std::string("qbar_")+key+"_sha256"]=d.at("sha256");}identity["qbar_manifest_sha256"]=eventio::file_sha256(path);}
 if(req.at("expected_state")!=identity)throw std::runtime_error("request state/model/four auxiliaries/refinement identity mismatch");
 return identity;
}
static J evaluate(const J&req){
 if(!immutable)throw std::runtime_error("initialize the fully verified dataset first");
 const auto begin=std::chrono::steady_clock::now();if(!std::filesystem::u8path(req.at("input").at("path").get<std::string>()).is_absolute()||!std::filesystem::u8path(req.at("output").get<std::string>()).is_absolute())throw std::runtime_error("absolute request input/output required");const auto state=state_identity(req);const auto op=req.at("op").get<std::string>();
 auto in=req.at("input").at("path").get<std::string>(),out=req.at("output").get<std::string>();
 if(std::filesystem::exists(std::filesystem::u8path(out)/"manifest.json"))throw std::runtime_error("immutable evaluation output already exists");
 char name[]="resident";char*args[]={name,in.data(),out.data()};++request_generation;
 const int result=op=="qbar"?resident_qbar_main(3,args):op=="values"?resident_values_main(3,args):op=="whole_uv"?resident_whole_uv_main(3,args):resident_profile_implicit_main(3,args);
 if(result)throw std::runtime_error("original frozen evaluator rejected request, code="+std::to_string(result));
 const auto report=std::filesystem::u8path(out)/"manifest.json";
 const auto result_doc=eventio::read_json(report);
 if(result_doc.at("model").at("sha256")!=state.at("model_sha256")||result_doc.at("packet").at("sha256")!=authority.at("packet_sha256")||result_doc.at("semantic").at("sha256")!=authority.at("semantic_sha256"))throw std::runtime_error("evaluator output model/source identity differs from request");
 if(op=="whole_uv"||op=="profile_implicit"){
  if(result_doc.at("checkpoint")!=req.at("input"))throw std::runtime_error("transport output checkpoint reference mismatch");
  for(const char*key:{"model","packet","semantic"}){checked(result_doc.at(key));if(result_doc.at(key)!=state.at(key))throw std::runtime_error(std::string("transport output full reference mismatch: ")+key);}
  if(transport_state_identity(req)!=state)throw std::runtime_error("transport state changed during evaluation");
 }
 if(eventio::file_sha256(std::filesystem::u8path(req.at("input").at("path").get<std::string>()))!=req.at("input").at("sha256"))throw std::runtime_error("request manifest changed during evaluation");
 J response{{"generation",request_generation},{"state_sha256",hash_json(state)},{"state",state},{"authority",authority},{"request_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},{"report",{{"path",report.u8string()},{"sha256",eventio::file_sha256(report)}}},{"no_Solve",true},{"no_root_or_J_result_cache",true},{"parameter_model_fresh_per_request",true}};
 eventio::atomic_json(std::filesystem::u8path(out)/"resident_receipt.json",response);return response;
}
}
int main(){using J=rinkfit::eventio::Json;std::string line;while(std::getline(std::cin,line)){if(line.empty())continue;J result;try{const auto req=J::parse(line);const auto op=req.at("op").get<std::string>();if(op=="close"){std::cout<<"RESIDENT_RESPONSE "<<J{{"ok",true},{"closed",true}}.dump()<<std::endl;break;}if(op!="init"&&op!="qbar"&&op!="values"&&op!="whole_uv"&&op!="profile_implicit")throw std::runtime_error("unknown operation");result={{"ok",true},{"result",op=="init"?resident::initialize(req):resident::evaluate(req)}};}catch(const std::exception&e){result={{"ok",false},{"error",e.what()},{"no_Solve",true}};}std::cout<<"RESIDENT_RESPONSE "<<result.dump()<<std::endl;}return 0;}
