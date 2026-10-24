#pragma once
#include "rinkfit/event_io.hpp"
#include "rinkfit/joint_exact_dependency_alias_v1.hpp"
#include <set>
#include <map>
#include <memory>
#include <cmath>
#ifndef RINKFIT_SOURCE_VALIDATOR_SHA
#define RINKFIT_SOURCE_VALIDATOR_SHA ""
#endif
namespace rinkfit {
// Explicit effective-input ABI. Legacy entry points never install this context.
// Historical code is identified by its original JSON field, not its extension.
class JointEffectiveInputV1 {
 using J=eventio::Json;
 struct State {J binding,manifest,prepared,history=J::array();std::map<std::string,std::size_t> historical;std::map<std::string,std::string> ancestry;std::map<std::string,J> payload;};
 static std::unique_ptr<State>& state(){static std::unique_ptr<State> s;return s;}
 static void need(bool x,const std::string&why){if(!x)throw std::runtime_error("effective input: "+why);}
 static std::string key(const std::filesystem::path&p,const std::string&sha){return std::filesystem::absolute(p).lexically_normal().u8string()+"|"+sha;}
 static void length(eventio::FastSHA256Stream&h,std::uint64_t n){std::uint8_t b[8];for(int k=0;k<8;++k)b[k]=std::uint8_t(n>>(8*k));h.update(b,8);}
 static void put(eventio::FastSHA256Stream&h,const J&v,bool portable){
  auto tag=[&](char c){h.update(reinterpret_cast<const std::uint8_t*>(&c),1);};
  if(v.is_null()){tag('N');return;}if(v.is_boolean()){tag(v.get<bool>()?'T':'F');return;}
  if(v.is_number_float()){const double d=v.get<double>();need(std::isfinite(d),"nonfinite canonical double");tag('D');std::uint64_t u;std::memcpy(&u,&d,8);std::uint8_t b[8];for(int k=0;k<8;++k)b[k]=std::uint8_t(u>>(8*k));h.update(b,8);return;}
  if(v.is_number_integer()||v.is_number_unsigned()){tag('I');const auto s=v.dump();length(h,s.size());h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());return;}
  if(v.is_string()){tag('S');const auto&s=v.get_ref<const std::string&>();length(h,s.size());h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());return;}
  if(v.is_array()){tag('A');length(h,v.size());for(const auto&x:v)put(h,x,portable);return;}
  need(v.is_object(),"canonical JSON type");tag('O');length(h,v.size());for(const auto&x:v.items()){put(h,J(x.key()),portable);if(portable&&x.key()=="file"&&v.contains("sha256"))put(h,J("sha256:"+v.at("sha256").get<std::string>()),portable);else put(h,x.value(),portable);}
 }
 // Build the ordinary Json DOM directly. The callback DOM parser scans the
 // entire parent container at every object_end, even when nothing is discarded.
 // Object insertion already supplies exact decoded-key duplicate detection.
 struct StrictJsonDomSax {
  const std::filesystem::path&path;J value;std::vector<J*> containers;J*pending=nullptr;
  explicit StrictJsonDomSax(const std::filesystem::path&p):path(p){}
  J*append(J next){
   if(containers.empty()){value=std::move(next);return &value;}
   auto&parent=*containers.back();
   if(parent.is_array()){parent.push_back(std::move(next));return &parent.back();}
   auto*slot=pending;pending=nullptr;*slot=std::move(next);return slot;
  }
  bool null(){append(nullptr);return true;}
  bool boolean(bool v){append(v);return true;}
  bool number_integer(J::number_integer_t v){append(v);return true;}
  bool number_unsigned(J::number_unsigned_t v){append(v);return true;}
  bool number_float(J::number_float_t v,const J::string_t&){append(v);return true;}
  bool string(J::string_t&v){append(std::move(v));return true;}
  bool binary(J::binary_t&v){append(std::move(v));return true;}
  bool start_object(std::size_t){containers.push_back(append(J::object()));return true;}
  bool key(J::string_t&v){
   auto inserted=containers.back()->emplace(std::move(v),nullptr);
   if(!inserted.second)need(false,"duplicate JSON key in "+path.u8string());
   pending=&inserted.first.value();return true;
  }
  bool end_object(){containers.pop_back();return true;}
  bool start_array(std::size_t){containers.push_back(append(J::array()));return true;}
  bool end_array(){containers.pop_back();return true;}
  // The parser passes its concrete parse_error/out_of_range type here.
  template<class Exception>bool parse_error(std::size_t,const std::string&,const Exception&ex){throw ex;}
 };
 public:
 static std::string compiled_collection_sha256(const J&refs){J values=J::object();for(const auto&r:refs){const auto file=r.at("file").get<std::string>();need(!values.contains(file),"repeated validator source path");values[file]=r.at("sha256");}std::string text="{";bool first=true;for(const auto&v:values.items()){if(!first)text+=", ";first=false;text+=J(v.key()).dump(-1,' ',true)+": "+v.value().dump(-1,' ',true);}text+="}";return eventio::sha256(std::vector<std::uint8_t>(text.begin(),text.end()));}
 static bool& legacy_read(){static thread_local bool x=false;return x;}
 struct LegacyReadScope {bool before;LegacyReadScope():before(legacy_read()){legacy_read()=true;}~LegacyReadScope(){legacy_read()=before;}};
 static std::filesystem::path runtime_path(const std::filesystem::path&p,const std::string&sha){if(!active()||legacy_read())return p;auto it=state()->payload.find(key(p,sha));if(it==state()->payload.end())return p;const auto actual=std::filesystem::u8path(it->second.at("actual").at("file").get<std::string>());eventio::verify(actual,sha);it->second["consumed"]=true;return actual;}
 static bool active(){return bool(state());}
 static void clear(){eventio::effective_reference_resolver()={};state().reset();}
 static std::string digest(const J&j,bool portable=true){eventio::FastSHA256Stream h;put(h,j,portable);return h.finish();}
 static J strict_json(const std::filesystem::path&p){
  const auto bytes=eventio::bytes(p);StrictJsonDomSax sax(p);
  if(!J::sax_parse(bytes.begin(),bytes.end(),&sax,J::input_format_t::json,true,false))need(false,"invalid JSON in "+p.u8string());
  return std::move(sax.value);
 }
 static std::filesystem::path pin(const J&r,const std::filesystem::path&base={}){need(r.is_object()&&r.contains("file")&&r.contains("sha256"),"missing exact file/SHA");auto p=std::filesystem::u8path(r.at("file").get<std::string>());if(p.is_relative())p=base/p;need(r.at("sha256").get<std::string>().size()==64,"SHA width");p=runtime_path(p,r.at("sha256"));eventio::verify(p,r.at("sha256"));if(r.contains("byte_count"))need(std::filesystem::file_size(p)==r.at("byte_count").get<std::uintmax_t>(),"byte count "+p.u8string());return p;}
 static void install(const J&binding){
  need(!active(),"context already installed");const auto f=pin(binding);auto s=std::make_unique<State>();s->binding=binding;s->manifest=strict_json(f);const auto&j=s->manifest;
  need(!std::string(RINKFIT_SOURCE_VALIDATOR_SHA).empty()&&j.at("compiled_source_validator_sha256")==RINKFIT_SOURCE_VALIDATOR_SHA,"effective ABI does not match this compiled source validator");
  need(j.at("schema")=="rinkfit.effective_joint_input.v1"&&j.at("abi")=="rinkfit.effective_input_abi.v1"&&j.at("canonical_encoding")=="rinkfit.tagged_json_sha256.v1","schema/ABI/encoding");
  pin(j.at("source_packet"),f.parent_path());pin(j.at("source_model"),f.parent_path());
  need(j.at("raw_rows").get<std::size_t>()>0&&j.at("qualification").at("schema")=="rinkfit.effective_source_qualification.v1","qualification/row count");
  const auto&qualification=j.at("qualification");pin(qualification.at("original_qualification"),f.parent_path());pin(qualification.at("selection"),f.parent_path());pin(qualification.at("frame_roles"),f.parent_path());pin(qualification.at("source_identity_verdicts"),f.parent_path());pin(qualification.at("physical_mask"),f.parent_path());pin(qualification.at("training_mask"),f.parent_path());
  need(qualification.at("source_packet_sha256")==j.at("source_packet").at("sha256")&&qualification.at("resolved_metadata_sha256")==j.at("resolved_metadata_sha256"),"qualification effective payload binding");
  for(const auto&r:j.at("ancestry")){const auto p=pin(r,f.parent_path());need(s->ancestry.emplace(std::filesystem::absolute(p).lexically_normal().u8string(),r.at("sha256")).second,"duplicate ancestry path");}
  need(!s->ancestry.empty()&&j.at("ancestry").back().at("sha256")==j.at("source_packet").at("sha256"),"ancestry head");
  for(const auto&g:j.value("qualification_generations",J::array())){
   const auto cf=pin(g.at("container"),f.parent_path());const auto proof=strict_json(cf);const auto schema=proof.value("schema",std::string{});need(schema.find("qualification")!=std::string::npos,"additional generation lacks original qualification container");const auto pointer=g.at("json_pointer").get<std::string>();need(pointer=="/source_packet"||pointer=="/parent_packet","additional generation is not an original qualification packet field");need(proof.at(J::json_pointer(pointer))==g.at("reference"),"additional generation changed original packet ref");
   J current=g.at("reference");auto base=cf.parent_path();std::set<std::string>seen;for(;;){const auto path=pin(current,base);const auto address=std::filesystem::absolute(path).lexically_normal().u8string();need(seen.insert(address).second,"cyclic qualification packet ancestry");auto known=s->ancestry.find(address);if(known!=s->ancestry.end()){need(known->second==current.at("sha256"),"qualification generation reused path with different original bytes");break;}s->ancestry.emplace(address,current.at("sha256"));const auto doc=strict_json(path);need(doc.at("schema")=="rinkfit.joint_packet.v1","qualified generation packet schema");if(!doc.contains("dense_parent_packet"))break;current=doc.at("dense_parent_packet");base=path.parent_path();}
  }
  for(const auto&r:j.at("validator_sources"))pin(r,f.parent_path());need(compiled_collection_sha256(j.at("validator_sources"))==j.at("compiled_source_validator_sha256"),"declared validator collection differs from compiled native ABI");need(digest(j.at("validator_sources"))==j.at("validator_code_sha256"),"validator source digest");need(qualification.at("validator_code_sha256")==j.at("validator_code_sha256"),"qualification validator binding");
  const std::set<std::string> roles={"issuer","reader","source_profile_reader","native_frame_extractor","actual_review_consumer","consumer_module_ref","issuer_module_ref","reader_module_refs","recursive_parent_guard_module_ref","actual_input_dependencies","actual_dependencies","source_validator_module_ref"};
  for(const auto&e:j.at("historical_audit_only")){
   need(e.at("role")=="historical_issuance_code"&&e.at("used_as_runtime_data")==false,"unsupported historical role");const auto cf=pin(e.at("container"),f.parent_path());const auto container=strict_json(cf);const auto pointer=e.at("json_pointer").get<std::string>();const auto&ref=container.at(J::json_pointer(pointer));need(ref==e.at("reference"),"historical reference changed at original field");
   bool purpose=false;std::string part;std::istringstream parts(pointer);while(std::getline(parts,part,'/'))if(roles.count(part))purpose=true;
   need(purpose&&ref.is_object()&&ref.contains("file")&&ref.contains("sha256"),"historical field is not a known code purpose");auto original=std::filesystem::u8path(ref.at("file").get<std::string>());if(original.is_relative())original=cf.parent_path()/original;need(original.extension()==".py","historical code must be Python issuer/reader, never a data file");
   J record=e;record["status"]="missing_expected_historical_bytes";record["actual_runtime_execution"]=false;
   try{auto actual=JointExactDependencyAliasesV1::resolve(original,ref.at("sha256").get<std::string>());eventio::verify(actual,ref.at("sha256"));record["status"]="available_exact_original_bytes";record["available_file"]=actual.u8string();}catch(const std::exception&){if(std::filesystem::exists(original))record["current_path_has_other_sha256"]=eventio::file_sha256(original);}
   const auto idx=s->history.size();need(s->historical.emplace(key(original,ref.at("sha256")),idx).second,"duplicate historical code binding");s->history.push_back(std::move(record));
  }
  for(const auto&e:j.value("runtime_payload",J::array())){need(e.at("schema")=="rinkfit.exact_runtime_payload_file.v1"&&e.at("original").at("sha256")==e.at("actual").at("sha256"),"runtime payload may not replace original bytes");const auto original=std::filesystem::u8path(e.at("original").at("file").get<std::string>());const auto actual=pin(e.at("actual"),f.parent_path());need(original.is_absolute()&&actual.is_absolute()&&original.lexically_normal()!=actual.lexically_normal(),"runtime payload needs separate absolute byte storage");need(!s->ancestry.count(std::filesystem::absolute(original).lexically_normal().u8string()),"original packet JSON ancestry cannot be substituted");need(s->payload.emplace(key(original,e.at("original").at("sha256")),e).second,"duplicate runtime payload binding");}
  state()=std::move(s);eventio::effective_reference_resolver()=[](const auto&p,const auto&sha){return runtime_path(p,sha);};
 }
 static bool prepared(){return active()&&!state()->prepared.is_null();}
 static void install_prepared(const J&binding,const J&prepared_binding){
  need(!active(),"context already installed");const auto mf=pin(binding),pf=pin(prepared_binding);auto s=std::make_unique<State>();s->binding=binding;s->manifest=strict_json(mf);s->prepared=strict_json(pf);const auto&m=s->manifest;const auto&p=s->prepared;
  need(m.at("schema")=="rinkfit.effective_joint_input.v1"&&m.at("abi")=="rinkfit.effective_input_abi.v1"&&m.at("canonical_encoding")=="rinkfit.tagged_json_sha256.v1","prepared effective schema/ABI");
  need(p.at("schema")=="rinkfit.prepared_effective_input.v1"&&p.at("effective_input")==binding,"prepared manifest binding");
  const auto proof=strict_json(pin(p.at("cold_native_proof")));need(proof.at("passed")==true&&proof.at("full_native_numerical_payload_consumed")==true&&proof.at("binding")==binding,"prepared original native proof incomplete or belongs to another input");
  const auto&q=proof.at("qualification");need(q.at("full_original_native_source_mathematics_reexecuted")==true&&q.at("legacy_validation_cache_reused")==false&&proof.at("typed_rows")==m.at("raw_rows"),"prepared proof did not consume full original source mathematics");
  for(const char*k:{"source_packet","source_model","raw_rows","resolved_metadata_sha256","arrays","qualification","engineering_support_budget_v2"})need(p.at(k)==m.at(k),std::string("prepared current field differs: ")+k);
  for(const char*k:{"source_packet","raw_rows","resolved_metadata_sha256","validator_code_sha256","compiled_source_validator_sha256"})need(proof.at(k)==m.at(k),std::string("prepared proof field differs: ")+k);
  for(const char*k:{"validator_code_sha256","compiled_source_validator_sha256"})need(p.at("proof_validator").at(k)==m.at(k),"prepared original validator identity changed");
  need(compiled_collection_sha256(m.at("validator_sources"))==m.at("compiled_source_validator_sha256")&&digest(m.at("validator_sources"))==m.at("validator_code_sha256"),"prepared original validator collection identity differs");
  need(p.at("original_validator_identity_preserved")==true&&p.at("native_source_mathematics_reexecuted_during_preparation")==false&&p.at("quality_accepted")==false&&p.at("source_image_identity_proven")==false,"prepared proof scope promoted");
  pin(m.at("source_packet"));pin(m.at("source_model"));pin(p.at("resolved_payload"));
  const auto closure=strict_json(pin(p.at("dependency_closure")));need(closure.at("schema")=="rinkfit.prepared_source_dependencies.v1"&&closure.at("dependencies").size()==p.at("source_dependency_count"),"prepared dependency closure schema/count");
  std::set<std::string>seen;for(const auto&e:closure.at("dependencies")){const auto&old=e.at("original");const auto&actual=e.at("actual");need(old.at("sha256")==actual.at("sha256"),"prepared source alias changed bytes");need(seen.insert(key(std::filesystem::u8path(old.at("file").get<std::string>()),old.at("sha256"))).second,"prepared duplicate source dependency");pin(actual);}
  for(const auto&e:m.value("runtime_payload",J::array())){need(e.at("schema")=="rinkfit.exact_runtime_payload_file.v1"&&e.at("original").at("sha256")==e.at("actual").at("sha256"),"prepared runtime payload changed original bytes");pin(e.at("actual"));need(s->payload.emplace(key(std::filesystem::u8path(e.at("original").at("file").get<std::string>()),e.at("original").at("sha256")),e).second,"prepared duplicate runtime payload");}
  s->history=proof.at("historical_audit_only");for(const auto&e:m.at("historical_audit_only")){auto original=std::filesystem::u8path(e.at("reference").at("file").get<std::string>());if(original.is_relative())original=std::filesystem::u8path(e.at("container").at("file").get<std::string>()).parent_path()/original;need(s->historical.emplace(key(original,e.at("reference").at("sha256")),s->historical.size()).second,"prepared historical ref repeated");}s->prepared["binding"]=prepared_binding;state()=std::move(s);eventio::effective_reference_resolver()=[](const auto&path,const auto&sha){return runtime_path(path,sha);};
 }
 static const J& prepared_receipt(){need(prepared(),"no prepared input context");return state()->prepared;}
 static const J& manifest(){need(active(),"missing explicit effective context");return state()->manifest;}
 static const J& binding(){need(active(),"missing explicit effective context");return state()->binding;}
 static bool historical(const J&r,const std::filesystem::path&base){if(!active()||!r.is_object()||!r.contains("sha256")||(!r.contains("file")&&!r.contains("path")))return false;auto p=std::filesystem::u8path(r.value("file",r.value("path",std::string{})));if(p.is_relative())p=base/p;const auto it=state()->historical.find(key(p,r.at("sha256")));if(it==state()->historical.end())return false;state()->history.at(it->second)["native_numeric_validator_requested_only_historical_pin"]=true;return true;}
 static void authorize_generation(const std::filesystem::path&p){need(active(),"generation outside effective context");const auto it=state()->ancestry.find(std::filesystem::absolute(p).lexically_normal().u8string());need(it!=state()->ancestry.end(),"generation is outside original authenticated ancestry: "+p.u8string());eventio::verify(p,it->second);}
 static J receipt(){const auto&j=manifest();if(prepared()){auto r=prepared_receipt();r["schema"]="rinkfit.actual_prepared_effective_input_consumption.v1";r["input_validation_mode"]="effective_input_v1";r["prepared_binding"]=r.at("binding");r["binding"]=binding();r["abi"]=j.at("abi");r["validator_code_sha256"]=j.at("validator_code_sha256");r["compiled_source_validator_sha256"]=j.at("compiled_source_validator_sha256");r["solver_compiled_source_validator_sha256"]=RINKFIT_SOURCE_VALIDATOR_SHA;r["current_solver_reexecuted_historical_qualification"]=false;return r;}bool complete=true;for(const auto&e:state()->history)complete=complete&&e.at("status")=="available_exact_original_bytes";return {{"schema","rinkfit.actual_effective_input_consumption.v1"},{"binding",binding()},{"input_validation_mode","effective_input_v1"},{"abi",j.at("abi")},{"source_packet",j.at("source_packet")},{"raw_rows",j.at("raw_rows")},{"resolved_metadata_sha256",j.at("resolved_metadata_sha256")},{"validator_code_sha256",j.at("validator_code_sha256")},{"compiled_source_validator_sha256",j.at("compiled_source_validator_sha256")},{"historical_audit_only",state()->history},{"exact_runtime_payload",state()->payload},{"historical_issuance_dependencies_available",complete},{"historical_issuance_replay_complete",false},{"source_image_identity_proven",false},{"quality_accepted",false}};}
};

template<class Packet,class Model>auto joint_effective_packet_generation_v1(const std::string&p,const Model&m,int)->decltype(Packet::load_effective_generation_v1(p,m)){return JointEffectiveInputV1::active()?Packet::load_effective_generation_v1(p,m):Packet::load(p,m);}
template<class Packet,class Model>auto joint_effective_packet_generation_v1(const std::string&p,const Model&m,long)->decltype(Packet::load(p,m)){if(JointEffectiveInputV1::active())throw std::runtime_error("effective qualification proxy lacks authenticated-generation loader");return Packet::load(p,m);}
}
