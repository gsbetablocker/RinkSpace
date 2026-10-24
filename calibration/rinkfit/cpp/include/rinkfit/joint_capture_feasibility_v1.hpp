#pragma once
#include "rinkfit/joint_capture_certified_zoom_v1.hpp"
#include "rinkfit/joint_source_append_provenance_v1.hpp"
#include <cstring>
namespace rinkfit {
// This is an explicit new-initialization operation, never an exact resume.
// Plan everything before changing a scalar; an affine/shared violation needs
// a constrained subspace restoration and is deliberately rejected here.
inline eventio::Json restore_joint_capture_zoom_values_v1(JointEventModel&m,
 const std::map<std::string,std::array<double,2>>&ranges,const std::set<std::pair<int,int>>&explicitly_free){
 using Json=eventio::Json;struct Coordinate{int block,index;double before,after,lower,upper;Json sources=Json::array();};
 std::map<std::pair<int,int>,Coordinate>coordinates;std::set<int>affine;for(const auto&d:m.control_manifolds())affine.insert(d.at("block").get<int>());
 for(const auto&s:m.shots){const auto range=ranges.at(s.rig_id);for(std::size_t k=0;k<s.control_sources.size();++k){const int bi=s.control_sources[k],offset=s.control_offsets[k],index=offset+2;auto&v=m.blocks.at(bi);
  if(offset<0||offset+3>int(v.values.size()))throw std::runtime_error("zoom feasibility source coordinate outside block");
  const double lo=std::max(v.lower.at(index),range[0]),hi=std::min(v.upper.at(index),range[1]),value=v.values.at(index);
  if(!(lo<=hi)||!std::isfinite(value))throw std::runtime_error("zoom feasibility invalid original bound/value: "+v.name);
  const double next=std::clamp(value,lo,hi);const auto key=std::make_pair(bi,index);auto found=coordinates.find(key);
  if(found==coordinates.end())found=coordinates.emplace(key,Coordinate{bi,index,value,next,lo,hi,Json::array()}).first;
  else if(found->second.lower!=lo||found->second.upper!=hi)throw std::runtime_error("aliased zoom coordinate has conflicting certified rig domains");
  found->second.sources.push_back({{"shot",s.id},{"rig",s.rig_id},{"control_index",k},{"source_block",bi},{"source_offset",offset},{"source_component",index}});
  if(next!=value&&(affine.count(bi)||v.values.size()!=3||offset!=0))throw std::runtime_error("shared/affine logf restoration requires a constrained original-subspace solve; scalar clipping refused: "+v.name);
  if(next!=value&&!explicitly_free.count(key))throw std::runtime_error("zoom feasibility requires the coordinate to be explicitly free in the new runtime stage: "+v.name);
 }}
 Json changes=Json::array();std::map<std::string,std::size_t>checked;for(const auto&item:coordinates){const auto&c=item.second;for(const auto&source:c.sources)++checked[source.at("shot").get<std::string>()];if(c.before!=c.after)changes.push_back({{"block",c.block},{"name",m.blocks.at(c.block).name},{"component",c.index},{"before",c.before},{"after",c.after},{"lower",c.lower},{"upper",c.upper},{"source_controls",c.sources}});}
 // A small complete physical snapshot proves every other scalar remains exact.
 std::vector<std::vector<double>>expected;expected.reserve(m.blocks.size());for(const auto&v:m.blocks)expected.push_back(v.values);
 for(const auto&change:changes){const int bi=change.at("block"),index=change.at("component");const double value=change.at("after");expected.at(bi).at(index)=value;m.blocks.at(bi).values.at(index)=value;}
 for(std::size_t bi=0;bi<m.blocks.size();++bi)if(m.blocks[bi].values.size()!=expected[bi].size()||std::memcmp(m.blocks[bi].values.data(),expected[bi].data(),expected[bi].size()*sizeof(double))!=0)throw std::runtime_error("zoom feasibility changed an unlisted physical scalar");
 return {{"schema","rinkfit.capture_zoom_feasibility_values.v1"},{"changes",changes},{"changed_scalar_count",changes.size()},{"checked_source_controls",checked},{"all_unlisted_physical_values_bitwise_preserved",true},{"shared_affine_scalar_clipping_used",false},{"certified_domain_expanded",false},{"quality_accepted",false}};
}
inline std::string joint_capture_feasibility_aux_sha_v1(const JointALBuilder&b){
 eventio::FastSHA256Stream hash;auto append=[&](const void*p,std::size_t bytes){hash.update(static_cast<const std::uint8_t*>(p),bytes);};
 const std::size_t n=b.core.packet.rows.size();append(&n,sizeof(n));for(const auto*v:{&b.core.station,&b.core.row,&b.core.alpha}){const auto size=v->size();append(&size,sizeof(size));append(v->data(),size*sizeof(double));}
 for(const auto&row:b.core.packet.rows){const std::uint32_t path=std::uint32_t(row.path);append(&path,sizeof(path));}
 if(b.static_motion)for(const auto&row:b.static_motion->rows)append(row.latent.data(),2*sizeof(double));
 return hash.finish();
}
inline eventio::Json joint_capture_zoom_recovery_metadata_v1(const JointEventModel&m){const char*key="capture_zoom_feasibility_recovery_v1";return m.runtime_model_metadata.value(key,m.source_json.value(key,eventio::Json()));}
inline void validate_joint_capture_zoom_recovery_metadata_v1(const JointEventModel&m,const eventio::Json&record,
 const eventio::Json&policy,const eventio::Json&original,const std::map<std::string,std::array<double,2>>&ranges){
 auto require=[](bool ok,const char*why){if(!ok)throw std::runtime_error(std::string("saved zoom feasibility transition: ")+why);};
 require(record.at("schema")=="rinkfit.applied_capture_zoom_feasibility_recovery.v1"&&record.at("policy")==policy&&record.at("original_imported_state_token")==policy.at("from_state_token"),"original SHA-bound transition policy/token differs");
 require(record.at("original_model_sha256")==policy.at("from_model").at("sha256")&&record.at("certified_domain_sha256")==policy.at("certified_domain").at("sha256"),"original model/domain binding differs");
 std::set<std::pair<int,int>>seen;for(const auto&change:record.at("changes")){const int bi=change.at("block"),index=change.at("component");require(seen.insert({bi,index}).second&&index==2&&m.blocks.at(bi).values.size()==3&&change.at("name")==m.blocks.at(bi).name,"invalid or duplicate direct scalar record");require(original.at("blocks").at(bi).at("name")==change.at("name")&&original.at("blocks").at(bi).at("values").at(index)==change.at("before"),"recorded original scalar differs from SHA-bound original model");const double before=change.at("before"),after=change.at("after"),lo=change.at("lower"),hi=change.at("upper");require(std::isfinite(before)&&std::isfinite(after)&&lo<=hi&&before!=after&&after==std::clamp(before,lo,hi),"record is not the declared initial interval projection");}
 require(record.at("changed_scalar_count").get<std::size_t>()==seen.size(),"transition change count differs");
 for(const auto&s:m.shots)for(std::size_t k=0;k<s.control_sources.size();++k){const int bi=s.control_sources[k],index=s.control_offsets[k]+2;const auto&old=original.at("blocks").at(bi);const auto range=ranges.at(s.rig_id);const double before=old.at("values").at(index),lo=std::max(old.at("lower").at(index).get<double>(),range[0]),hi=std::min(old.at("upper").at(index).get<double>(),range[1]);if(before<lo||before>hi)require(seen.count({bi,index})==1,"record omits an originally infeasible authoritative control");for(const auto&change:record.at("changes"))if(change.at("block")==bi&&change.at("component")==index)require(change.at("lower")==lo&&change.at("upper")==hi,"recorded projection used a different physical/certified interval");}

}
inline eventio::Json restore_joint_capture_zoom_feasibility_v1(JointALBuilder&b,const eventio::Json&cfg,
 const JointCaptureCertifiedDomainV1&domain,const std::string&verified_imported_state_token){
 using Json=eventio::Json;if(!cfg.contains("capture_zoom_feasibility_recovery_v1"))return {{"enabled",false}};
 const auto&policy=cfg.at("capture_zoom_feasibility_recovery_v1");auto require=[](bool ok,const std::string&why){if(!ok)throw std::runtime_error("explicit zoom feasibility recovery: "+why);};
 require(b.source_uv_observed_row_initialization()&&!b.source_uv_capture_v1(),"only a declared observed-primal new initialization may recover these coordinates");
 require(policy.at("schema")=="rinkfit.capture_zoom_feasibility_recovery.v1"&&policy.at("allowed_change")=="project_only_direct_control_logf_to_existing_certified_interval","unsupported recovery policy");
 require(!verified_imported_state_token.empty(),"complete imported token missing");
 if(policy.at("from_source_packet_sha256")!=b.core.packet.source_hash){const auto append=source_append_v1::authorization(cfg,policy.at("from_source_packet_sha256").get<std::string>(),b.core.packet.source_hash);(void)append;}
 const auto&model_ref=policy.at("from_model"),domain_ref=policy.at("certified_domain");const auto model_path=model_ref.at("file").get<std::string>();eventio::verify(model_path,model_ref.at("sha256"));
 eventio::verify(domain_ref.at("file").get<std::string>(),domain_ref.at("sha256"));require(domain_ref==cfg.at("capture_certified_numeric_domain")&&domain_ref.at("sha256")==domain.manifest_sha,"existing domain binding differs");domain.assert_dependencies();
 const auto original=eventio::read_json(model_path);const auto saved_transition=joint_capture_zoom_recovery_metadata_v1(b.core.model);
 if(!saved_transition.is_null()){
  validate_joint_capture_zoom_recovery_metadata_v1(b.core.model,saved_transition,policy,original,domain.ranges);const auto before_token=b.capture_state().token(),before_aux=joint_capture_feasibility_aux_sha_v1(b);
  // Empty permission set makes every out-of-domain coordinate fail before any
  // mutation. A completed transition is verified, never projected a second time.
  auto report=restore_joint_capture_zoom_values_v1(b.core.model,domain.ranges,{});require(report.at("changed_scalar_count")==0,"saved feasible transition became infeasible");
  require(before_token==b.capture_state().token()&&before_aux==joint_capture_feasibility_aux_sha_v1(b),"same-objective restart changed complete physical/raw auxiliary state");report["schema"]="rinkfit.explicit_capture_zoom_feasibility_recovery.v1";report["enabled"]=true;report["original_transition_metadata_reused"]=true;report["policy"]=policy;report["current_imported_state_token"]=verified_imported_state_token;report["state_token"]=before_token;report["new_initialization_only"]=false;report["new_initialization_performed"]=false;report["raw_aux_and_source_preserved"]=true;return report;
 }
 require(policy.at("from_state_token")==verified_imported_state_token,"first recovery requires its exact complete original imported token");require(original==b.core.model.exact_json(),"runtime model no longer equals the SHA-bound complete imported model");
 require(!policy.at("reason").get<std::string>().empty(),"new-initialization reason missing");
 std::set<std::pair<int,int>>explicitly_free;for(const auto&entry:cfg.at("runtime_shared_parameter_stage_v2").at("blocks")){const int bi=entry.at("block_index");require(entry.at("block_name")==b.core.model.blocks.at(bi).name,"runtime stage block identity differs");for(const auto&index:entry.at("free_ambient_indices"))explicitly_free.insert({bi,index.get<int>()});}
 const auto before_token=b.capture_state().token(),before_aux=joint_capture_feasibility_aux_sha_v1(b);auto report=restore_joint_capture_zoom_values_v1(b.core.model,domain.ranges,explicitly_free);const auto after_aux=joint_capture_feasibility_aux_sha_v1(b);
 require(before_aux==after_aux,"raw station/row/alpha/path or motion latents changed");
 auto expected=original;for(const auto&change:report.at("changes"))expected["blocks"][change.at("block").get<std::size_t>()]["values"][change.at("component").get<std::size_t>()]=change.at("after");require(expected==b.core.model.exact_json(),"unlisted model metadata or physical scalar changed");
 const Json transition={{"schema","rinkfit.applied_capture_zoom_feasibility_recovery.v1"},{"policy",policy},{"original_imported_state_token",verified_imported_state_token},{"original_model_sha256",model_ref.at("sha256")},{"certified_domain_sha256",domain_ref.at("sha256")},{"changes",report.at("changes")},{"changed_scalar_count",report.at("changed_scalar_count")}};b.core.model.runtime_model_metadata["capture_zoom_feasibility_recovery_v1"]=transition;expected["capture_zoom_feasibility_recovery_v1"]=transition;require(expected==b.core.model.exact_json(),"zoom transition metadata altered an unrelated model field");
 report["schema"]="rinkfit.explicit_capture_zoom_feasibility_recovery.v1";report["enabled"]=true;report["policy"]=policy;report["imported_state_token"]=verified_imported_state_token;report["new_objective_state_token_before_recovery"]=before_token;report["new_objective_state_token_after_recovery"]=b.capture_state().token();report["raw_aux_and_motion_SHA256_before"]=before_aux;report["raw_aux_and_motion_SHA256_after"]=after_aux;report["raw_aux_and_source_preserved"]=true;report["exact_objective_continuation_claimed"]=false;report["optimizer_history_restored"]=false;report["new_initialization_only"]=true;return report;
}
}
