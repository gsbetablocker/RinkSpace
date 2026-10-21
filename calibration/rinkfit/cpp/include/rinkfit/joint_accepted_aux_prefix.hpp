#pragma once
#include "rinkfit/joint_builder.hpp"
#include "rinkfit/augmented_lagrangian.hpp"
#include <cstring>
#include <set>

namespace rinkfit {
struct JointAcceptedAuxiliaryPreservation {
 std::vector<std::uint8_t> preserved;
 eventio::Json receipt=eventio::Json::object();
};
namespace accepted_aux_detail {
inline void require(bool ok,const std::string&why){if(!ok)throw std::runtime_error("accepted engineering auxiliary prefix: "+why);}
inline std::filesystem::path verified(const eventio::Json&ref){auto p=std::filesystem::path(ref.at("file").get<std::string>());require(p.is_absolute(),"state reference must be absolute");eventio::verify(p,ref.at("sha256"));return p;}
inline void put(eventio::SHA256Stream&h,const std::string&s){h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
inline void typed(eventio::SHA256Stream&h,const eventio::Json&v){
 if(v.is_null())put(h,"n");else if(v.is_boolean())put(h,v.get<bool>()?"t":"f");
 else if(v.is_number_float()){put(h,"d");const double x=v.get<double>();require(std::isfinite(x),"nonfinite typed JSON");h.update(reinterpret_cast<const std::uint8_t*>(&x),sizeof(x));}
 else if(v.is_number_integer()||v.is_number_unsigned())put(h,"i"+v.dump()+";");
 else if(v.is_string()){const auto s=v.get<std::string>();put(h,"s"+std::to_string(s.size())+":"+s);}
 else if(v.is_array()){put(h,"[");for(const auto&x:v)typed(h,x);put(h,"]");}
 else if(v.is_object()){put(h,"{");for(auto it=v.begin();it!=v.end();++it){typed(h,eventio::Json(it.key()));typed(h,it.value());}put(h,"}");}
 else throw std::runtime_error("unsupported prefix JSON value");
}
inline eventio::Json projection_definition(eventio::Json j){j.erase("source_supported_motion_flags");j.erase("linear_control_manifolds");for(auto&b:j.at("blocks")){b.erase("constant");b.erase("fixed_indices");}return j;}
inline std::string digest(const eventio::Json&v){eventio::SHA256Stream h;typed(h,v);return h.finish();}
inline std::string bytes_digest(const void*p,std::size_t n){eventio::SHA256Stream h;h.update(reinterpret_cast<const std::uint8_t*>(p),n);return h.finish();}
inline std::size_t dtype_size(const std::string&s){if(s=="<f8"||s=="<i8")return 8;if(s=="<u4"||s=="<i4")return 4;if(s=="<u2")return 2;if(s=="|u1"||s=="u1"||s=="|i1"||s=="i1")return 1;throw std::runtime_error("unsupported accepted prefix dtype "+s);}
inline std::string array_prefix_hash(const eventio::Json&d,std::size_t n){
 const auto p=verified(d);const auto shape=d.at("shape").get<std::vector<std::size_t>>();require(!shape.empty()&&shape[0]>=n,"prefix shape");
 std::size_t bytes=n*dtype_size(d.at("dtype"));for(std::size_t k=1;k<shape.size();++k)bytes*=shape[k];
 std::ifstream f(p,std::ios::binary);f.seekg(d.value("offset_bytes",std::size_t(0)));eventio::SHA256Stream h;std::vector<std::uint8_t>b(1024*1024);
 while(bytes){const auto take=std::min(bytes,b.size());require(bool(f.read(reinterpret_cast<char*>(b.data()),take)),"short prefix binary");h.update(b.data(),take);bytes-=take;}return h.finish();
}
template<class T>inline std::vector<T> prefix(const eventio::Json&d,std::size_t n){
 const auto p=verified(d);require(dtype_size(d.at("dtype"))==sizeof(T),"auxiliary dtype size");std::ifstream f(p,std::ios::binary);f.seekg(d.value("offset_bytes",std::size_t(0)));std::vector<T>v(n);require(bool(f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))),"short explicit auxiliary");return v;
}
template<class T,class F>inline std::string runtime_array_hash(std::size_t n,std::size_t columns,F value){std::vector<T>v(n*columns);for(std::size_t i=0;i<n;++i)for(std::size_t k=0;k<columns;++k)v[i*columns+k]=T(value(i,k));return bytes_digest(v.data(),v.size()*sizeof(T));}
inline std::string query_digest(const JointPacket&p,std::size_t n){eventio::SHA256Stream h;for(std::size_t i=0;i<n;++i){const auto&r=p.rows[i];if(!r.query)continue;const auto&q=*r.query;const eventio::Json j={{"row_index",i},{"alpha",q.alpha},{"uv",q.uv},{"nx",q.nx},{"ny",q.ny},{"den",q.den},{"origin",q.origin},{"scale",q.scale},{"normal",r.normal}};put(h,digest(j));}return h.finish();}
inline std::string group_digest(const JointPacket&p,std::size_t n,std::size_t old_rows){require(n<=p.groups.size(),"incomplete original W groups");eventio::SHA256Stream h;std::size_t selected=0;for(const auto&g:p.groups){bool all_old=true,any_old=false;for(const auto i:g.rows){all_old=all_old&&i<old_rows;any_old=any_old||i<old_rows;}require(all_old||!any_old,"appended W group crosses original prefix");if(!all_old)continue;++selected;const eventio::Json j={{"id",g.id},{"rows",g.rows},{"rank",g.rank},{"statistic_kind",g.statistic_kind},{"W_values_sha256",bytes_digest(g.W.data(),g.W.size()*sizeof(double))}};put(h,digest(j));}require(selected==n,"original W group membership changed");return h.finish();}
}
// Numerical preservation only; this function never issues source qualification.
// Call after ordinary source validation and builder construction, before fresh
// initializer/GPU preparation. Only the small accepted binding must survive
// documentary JSON trimming; it reads no second full packet/model allocation.
inline JointAcceptedAuxiliaryPreservation preserve_joint_accepted_auxiliary_prefix(
 JointProblemBuilder&core,const std::filesystem::path&input_model,const std::filesystem::path&input_packet){
 using namespace accepted_aux_detail;JointAcceptedAuxiliaryPreservation result;result.preserved.resize(core.packet.rows.size(),0);
 if(!core.packet.source_json.contains("accepted_explicit_state_binding"))return result;
 const auto&binding=core.packet.source_json.at("accepted_explicit_state_binding");
 require(binding.at("schema")=="rinkfit.accepted_explicit_state_packet_binding.v2"&&binding.at("state_kind")=="engineering_candidate","unsupported state binding");
 require(!binding.value("native_replay_pass_claimed",true)&&!binding.value("formal_acceptance_claimed",true)&&!binding.value("old_AL_duals_resumed",true)&&!binding.value("source_UV_W_E_admission_roles_changed",true),"forbidden pass/dual/source claim");
 eventio::verify(input_packet,core.packet.source_hash); // exact currently loaded wrapper
 const auto contract_file=verified(binding.at("prefix_contract"));const auto contract=eventio::read_json(contract_file);
 require(contract.at("schema")=="rinkfit.engineering_accepted_aux_prefix_contract.v1"&&contract.value("complete_Query_material_kind_normal_E_ID_PTS_UV_and_W_prefix_verified",false)&&contract.value("full_candidate_explicit_arrays_preserved",false),"incomplete prefix contract");
 const std::size_t old_n=binding.at("preserved_raw_rows"),n=core.packet.rows.size();
 require(old_n>0&&old_n<=n&&contract.at("old_raw_rows")==old_n&&contract.at("new_raw_rows")==n,"raw prefix count");
 const auto parent=verified(binding.at("accepted_source_packet"));const auto appended=verified(binding.at("parent_packet"));
 require(contract.at("old_source_packet")==binding.at("accepted_source_packet")&&contract.at("appended_source_packet")==binding.at("parent_packet"),"exact parent generation mismatch");
 const auto receipt=eventio::read_json(verified(binding.at("accepted_receipt")));
 require(receipt.at("schema")=="rinkfit.joint_engineering_recovery_run.v1"&&receipt.at("objective_id")=="engineering_recovery_native_pixel_v1"&&binding.at("objective_id")==receipt.at("objective_id"),"engineering objective mismatch");
 require(!receipt.value("source_recalled",true)&&!receipt.value("formal_Phase_II_KKT_passed",true)&&!receipt.value("common_worst_budget_consumed",true)&&!receipt.value("original_W_likelihood_consumed",true)&&receipt.value("callback_error",std::string()).empty(),"recalled/incompatible candidate");
 require(receipt.at("source_packet_sha256")==binding.at("accepted_source_packet").at("sha256")&&receipt.at("model_sha256")==binding.at("model_sha256"),"receipt source/model mismatch");
 const auto model_file=verified(binding.at("accepted_model"));require(binding.at("accepted_model").at("sha256")==receipt.at("model_sha256"),"candidate model reference mismatch");
 const auto old_model=eventio::read_json(model_file),current_model=eventio::read_json(input_model);const auto&blocks=old_model.at("blocks");
 const bool qualified_suffix=contract.contains("qualified_physical_suffix");
 if(qualified_suffix){
  const auto&suffix=contract.at("qualified_physical_suffix");require(binding.value("qualified_physical_suffix",eventio::Json{})==suffix,"physical suffix binding/contract mismatch");
  require(suffix.at("schema")=="rinkfit.qualified_physical_suffix_for_accepted_aux.v1"&&suffix.value("old_CP_model_is_not_replaced",false)&&suffix.value("old_physical_prefix_values_exact",false)&&suffix.value("old_projection_roles_and_topology_exact",false)&&suffix.value("suffix_is_not_part_of_old_checkpoint",false)&&suffix.value("source_qualification_required_before_consume",false),"invalid physical suffix claims");
  require(verified(suffix.at("current_model"))==std::filesystem::absolute(input_model)&&suffix.at("old_physical_blocks")==blocks.size()&&suffix.at("new_physical_blocks")==core.model.blocks.size()&&current_model.at("blocks").size()==core.model.blocks.size(),"physical suffix current model/count mismatch");
  const auto&certref=suffix.at("source_qualification_certificate");require(core.packet.source_json.at("board_material_source_qualification")==certref,"physical suffix changed current source authority");const auto cert=eventio::read_json(verified(certref));
  require(cert.at("schema")=="rinkfit.board_material_fit_qualification.v6_static_board_prefix"&&cert.at("original_raw_rows")==old_n&&cert.at("raw_rows")==n&&cert.at("appended_raw_rows")==n-old_n,"physical suffix source scope not exact V6");
  require(cert.at("actual_append_source_bundle")==suffix.at("actual_source_bundle"),"physical suffix source bundle mismatch");const auto source=eventio::read_json(verified(suffix.at("actual_source_bundle")));
  require(source.at("schema")=="rinkfit.native_static_board_tracklet_qualified_bundle.v1"&&source.at("model")==suffix.at("source_issued_model")&&source.at("parent_model")==suffix.at("source_issued_parent_model")&&source.at("track_material_bindings")==suffix.at("track_material_bindings"),"physical suffix issuer dependencies changed");
  const auto issued=eventio::read_json(verified(suffix.at("source_issued_model"))),issued_parent=eventio::read_json(verified(suffix.at("source_issued_parent_model"))),latents=eventio::read_json(verified(suffix.at("track_material_bindings")));const auto old_m=old_model.at("materials").size();
  require(issued_parent.at("blocks").size()==blocks.size()&&issued_parent.at("materials").size()==old_m&&suffix.at("old_materials")==old_m&&suffix.at("new_materials")==current_model.at("materials").size()&&issued.at("blocks").size()==current_model.at("blocks").size()&&issued.at("materials").size()==current_model.at("materials").size()&&latents.size()==core.model.blocks.size()-blocks.size()&&latents.size()==current_model.at("materials").size()-old_m&&suffix.at("qualified_suffix_blocks")==latents.size()&&suffix.at("qualified_suffix_materials")==latents.size()&&suffix.at("qualified_new_source_rows")==n-old_n,"physical suffix has unqualified allocation");
  auto prefix=current_model;prefix["blocks"]=eventio::Json::array();for(std::size_t i=0;i<blocks.size();++i)prefix["blocks"].push_back(current_model.at("blocks")[i]);prefix["materials"]=eventio::Json::array();for(std::size_t i=0;i<old_m;++i)prefix["materials"].push_back(current_model.at("materials")[i]);
  if(!old_model.contains("static_board_shared_uv_append")){require(prefix.value("static_board_shared_uv_append",eventio::Json{})==issued.value("static_board_shared_uv_append",eventio::Json{}),"physical suffix provenance changed");prefix.erase("static_board_shared_uv_append");}
  require(prefix.value("linear_control_manifolds",eventio::Json{})==old_model.value("linear_control_manifolds",eventio::Json{})&&projection_definition(prefix)==projection_definition(old_model),"accepted physical prefix values/topology/roles changed");
  std::set<std::size_t>seen_blocks,seen_materials;std::size_t source_rows=0;
  for(const auto&latent:latents){const auto mi=latent.at("material_index").get<std::size_t>(),bi=latent.at("geometry_block").get<std::size_t>();const auto ad=latent.at("advertisement_block").get<int>();
   require(mi>=old_m&&mi<current_model.at("materials").size()&&bi>=blocks.size()&&bi<core.model.blocks.size()&&seen_blocks.insert(bi).second&&seen_materials.insert(mi).second&&current_model.at("materials")[mi]==issued.at("materials")[mi],"physical suffix track/material identity changed");
   const auto&path=current_model.at("materials")[mi].at("paths");require(path.size()==1&&path[0].at("kind")==2&&path[0].at("points")==eventio::Json::array({eventio::Json::array({0,0})})&&path[0].value("physical_point",false)&&path[0].at("geometry_block")==bi&&path[0].at("advertisement_block")==ad&&(ad==7||ad==8),"physical suffix is not the qualified board material point");
   const auto&actual=core.model.blocks[bi];const auto&definition=current_model.at("blocks")[bi];auto a=definition,b=issued.at("blocks")[bi];for(const char*key:{"values","constant"}){a.erase(key);b.erase(key);}require(a==b&&actual.name==definition.at("name")&&actual.values==definition.at("values").get<std::vector<double>>()&&actual.values.size()==6&&actual.fixed_indices==std::vector<int>({0,1,3,4}),"physical suffix parameter roles or current coordinates changed");
   for(const int k:{0,1,3,4})require(actual.values[k]==((k==0||k==4)?1.:0.),"physical suffix affine basis changed");for(const int k:{2,5})require(std::isfinite(actual.values[k])&&actual.values[k]>=0&&actual.values[k]<=1&&actual.lower[k]==0&&actual.upper[k]==1,"physical suffix latent outside material bounds");
   source_rows+=latent.at("shared_native_rows").get<std::size_t>();
  }
  require(source_rows==n-old_n,"physical suffix source denominator changed");for(std::size_t i=old_n;i<n;++i){const auto&r=core.packet.rows[i];require(r.active&&r.kind==1&&r.path==0&&r.station==0&&seen_materials.count(r.material)&&!r.query,"new physical suffix source is not a fixed observed point");}
 }else{
  require(!binding.contains("qualified_physical_suffix"),"unverified physical suffix binding");
  require(projection_definition(old_model)==projection_definition(current_model),"current projection/material/clock definition differs from accepted candidate");
  require(blocks.size()==core.model.blocks.size()&&current_model.at("blocks").size()==blocks.size(),"physical block allocation changed");
 }
 for(std::size_t i=0;i<blocks.size();++i){const auto&b=core.model.blocks[i];const auto&j=blocks[i];const auto&current=current_model.at("blocks")[i];require(b.name==j.at("name")&&b.values==j.at("values").get<std::vector<double>>()&&current.at("name")==j.at("name")&&current.at("values")==j.at("values"),"physical block name/value changed; source support flags may be recomputed");}
 const auto manifest_file=verified(binding.at("explicit_nuisance_manifest"));const auto manifest=eventio::read_json(manifest_file);
 require(manifest.at("packet_sha256")==receipt.at("source_packet_sha256")&&manifest.at("raw_rows")==old_n,"complete explicit manifest generation mismatch");
 for(auto it=contract.at("source_arrays").begin();it!=contract.at("source_arrays").end();++it)require(array_prefix_hash(it.value().at("current_descriptor"),old_n)==it.value().at("prefix_sha256"),"source binary prefix changed "+it.key());

 const auto&source_arrays=contract.at("source_arrays");const auto&observed=core.packet.rows;
 auto matches=[&](const std::string&name,const std::string&hash){require(source_arrays.contains(name)&&hash==source_arrays.at(name).at("prefix_sha256"),"loaded runtime source prefix differs "+name);};
 matches("native_pts",runtime_array_hash<std::int64_t>(old_n,1,[&](std::size_t i,std::size_t){return observed[i].pts;}));
 matches("uv",runtime_array_hash<double>(old_n,2,[&](std::size_t i,std::size_t k){return observed[i].uv[k];}));
 matches("kind",runtime_array_hash<std::uint8_t>(old_n,1,[&](std::size_t i,std::size_t){return observed[i].kind;}));
 matches("material_index",runtime_array_hash<std::uint32_t>(old_n,1,[&](std::size_t i,std::size_t){return observed[i].material;}));
 matches("E",runtime_array_hash<double>(old_n,1,[&](std::size_t i,std::size_t){return observed[i].E;}));
 const auto shot_ids=contract.at("shot_ids").get<std::vector<std::string>>();std::map<std::string,std::uint16_t>shot_indices;for(std::size_t i=0;i<shot_ids.size();++i){require(i<65536&&shot_indices.emplace(shot_ids[i],std::uint16_t(i)).second,"shot index identity");}
 matches("shot_index",runtime_array_hash<std::uint16_t>(old_n,1,[&](std::size_t i,std::size_t){return shot_indices.at(observed[i].shot_id);}));
 const auto&aux=contract.at("auxiliaries");const auto station=prefix<double>(aux.at("station").at("current_descriptor"),old_n),row=prefix<double>(aux.at("row").at("current_descriptor"),old_n),alpha=prefix<double>(aux.at("alpha").at("current_descriptor"),old_n);const auto path=prefix<std::uint32_t>(aux.at("path").at("current_descriptor"),old_n);
 for(const auto&name:{"station","row","alpha","path"}){const auto&a=aux.at(name);verified(a.at("candidate_file"));require(a.at("candidate_file").at("sha256")==manifest.at("files").at(std::string(name)+(std::string(name)=="path"?".u32":".f64")).at("sha256"),"candidate auxiliary manifest mismatch");require(array_prefix_hash(a.at("current_descriptor"),old_n)==a.at("prefix_sha256")&&a.at("prefix_sha256")==a.at("candidate_file").at("sha256"),"candidate/merged auxiliary prefix mismatch");}
 std::ifstream stream(verified(binding.at("complete_engineering_checkpoint")));const auto checkpoint=read_checkpoint(stream);const auto&state=checkpoint.state;
 require(checkpoint.outer.lambda.empty()&&checkpoint.outer.nu.empty()&&checkpoint.outer.rho==0&&checkpoint.best_physical_state.empty(),"old AL dual/best state is forbidden");
 require(checkpoint.runtime_fingerprint==receipt.at("configuration_sha256")&&binding.at("configuration_sha256")==receipt.at("configuration_sha256")&&state.token()==receipt.at("candidate_state_token")&&state.token()==binding.at("state_token"),"configuration or complete state token mismatch");
 const auto&payload=state.data();require(payload.identity.source_id==receipt.at("source_packet_sha256")&&payload.identity.objective_id==receipt.at("objective_id"),"checkpoint source/objective mismatch");
 require(payload.parameters.size()>=blocks.size(),"incomplete physical checkpoint");for(std::size_t i=0;i<blocks.size();++i)require(payload.parameters[i].name==core.model.blocks[i].name&&payload.parameters[i].values==core.model.blocks[i].values,"checkpoint physical values differ");
 std::vector<std::uint8_t>have_station(old_n),have_row(old_n),have_alpha(old_n);
 for(std::size_t k=blocks.size();k<payload.parameters.size();++k){const auto&p=payload.parameters[k];for(const auto&name:{"station/","row/","alpha/"})if(p.name.rfind(name,0)==0){const auto i=std::stoull(p.name.substr(std::strlen(name)));require(i<old_n&&core.packet.rows[i].active&&p.values.size()==1,"checkpoint auxiliary index/active membership");const auto key=std::string(name);const double value=key=="station/"?station[i]:key=="row/"?row[i]:alpha[i];require(p.values[0]==value,"checkpoint and complete explicit auxiliary differ");(key=="station/"?have_station:key=="row/"?have_row:have_alpha)[i]=1;}}
 const auto discrete=payload.discrete.find("joint_packet_sha256");require(discrete!=payload.discrete.end()&&discrete->second==payload.identity.source_id,"discrete source identity");
 const auto paths=payload.discrete.find("finite_material_paths_RLE_u32_v1");require(paths!=payload.discrete.end(),"complete discrete paths missing");std::size_t offset=0,pos=0;while(pos<paths->second.size()){const auto colon=paths->second.find(':',pos),end=paths->second.find(';',pos);require(colon!=std::string::npos&&end!=std::string::npos&&colon<end,"invalid path RLE");const auto count=std::stoull(paths->second.substr(pos,colon-pos)),value=std::stoull(paths->second.substr(colon+1,end-colon-1));require(count>0&&count<=old_n-offset,"invalid path run count");for(std::size_t j=0;j<count;++j)require(path[offset++]==value,"checkpoint explicit discrete path differs");pos=end+1;}require(offset==old_n,"incomplete discrete path prefix");
 eventio::SHA256Stream ids;put(ids,"[");put(ids,"[");for(std::size_t i=0;i<old_n;++i)typed(ids,eventio::Json(core.packet.rows[i].source_dataset_id));put(ids,"]");put(ids,"[");for(std::size_t i=0;i<old_n;++i)typed(ids,eventio::Json(core.packet.rows[i].source_row_id));put(ids,"]");put(ids,"]");require(ids.finish()==contract.at("source_ID_prefix_sha256"),"source raw ID prefix changed");
 require(query_digest(core.packet,old_n)==contract.at("runtime_query_prefix_sha256"),"current frozen Query/normal prefix differs");require(group_digest(core.packet,contract.at("runtime_old_group_count"),old_n)==contract.at("runtime_group_prefix_sha256"),"current W group prefix differs");
 std::size_t preserved=0;for(std::size_t i=0;i<old_n;++i){const auto&r=core.packet.rows[i];require(core.station[i]==station[i]&&core.row[i]==row[i]&&core.alpha[i]==alpha[i]&&r.path==int(path[i]),"runtime merged four-auxiliary prefix changed");require(bool(have_row[i])==r.active&&bool(have_station[i])==(r.active&&r.kind!=1)&&bool(have_alpha[i])==(r.active&&bool(r.query)),"incomplete active auxiliary snapshot");if(r.active){result.preserved[i]=1;++preserved;}}
 require(preserved==contract.at("old_active_rows"),"original active membership changed");
 result.receipt={{"schema","rinkfit.production_engineering_accepted_auxiliary_prefix.v1"},{"source_packet_sha256",core.packet.source_hash},{"accepted_candidate_state_token",state.token()},{"objective_id",payload.identity.objective_id},{"prefix_contract",binding.at("prefix_contract")},{"preserved_raw_rows",old_n},{"preserved_active_rows",preserved},{"fresh_appended_rows",n-old_n},{"all_physical_block_names_values_exact",!qualified_suffix},{"complete_four_auxiliary_prefix_exact",true},{"fresh_initializer_must_skip_preserved_rows",true},{"source_qualification_issued",false},{"formal_acceptance_claimed",false},{"old_AL_duals_or_objective_resumed",false},{"old_complete_CP_physical_prefix_exact",true},{"old_complete_CP_physical_blocks",blocks.size()},{"qualified_new_physical_suffix",qualified_suffix},{"current_physical_blocks",core.model.blocks.size()}};return result;
}
}
