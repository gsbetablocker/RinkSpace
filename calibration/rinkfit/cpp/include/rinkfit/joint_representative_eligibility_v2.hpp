#pragma once
#include "rinkfit/event_io.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <map>
#include <array>
#include <fstream>
#include <vector>

namespace rinkfit {
// Physical admission is the current packet's immutable domain. Training is a
// separate, SHA-bound subset; omitted physical rows remain inspection rows.
struct JointRepresentativeEligibilityV2 {
 bool enabled=false;
 std::vector<unsigned char> training,original_fit;
 std::size_t physical_count=0,training_count=0,fit_cap=0;
 eventio::Json binding,provenance,frame_roles,identity_verdicts=eventio::Json::object(),append_proof=eventio::Json::object();
 std::size_t original_source_rows=0,appended_fit_count=0;

 static std::vector<std::uint8_t> checked_bytes(const eventio::Json&ref,const std::filesystem::path&base={}){
  auto file=std::filesystem::u8path(ref.at("file").get<std::string>());if(file.is_relative())file=base/file;
  const auto bytes=eventio::bytes(file);
  if(eventio::sha256(bytes)!=ref.at("sha256"))throw std::runtime_error("representative v2 reference SHA mismatch: "+file.u8string());
  return bytes;
 }
 static eventio::Json checked_json(const eventio::Json&ref){const auto bytes=checked_bytes(ref);return eventio::Json::parse(bytes.begin(),bytes.end());}
 static bool same_ref(const eventio::Json&a,const eventio::Json&b){return a.at("sha256")==b.at("sha256")&&std::filesystem::u8path(a.at("file").get<std::string>()).lexically_normal()==std::filesystem::u8path(b.at("file").get<std::string>()).lexically_normal();}
 static void exact_append_prefix(const eventio::Json&parent,const eventio::Json&current,std::size_t before,std::size_t after){
  const auto&pa=parent.at("arrays"),&ca=current.at("arrays");if(pa.size()!=ca.size())throw std::runtime_error("training append changed original array registry");
  for(const auto&item:pa.items()){const auto&p=item.value();const auto&c=ca.at(item.key());const auto ps=p.at("shape").get<std::vector<std::size_t>>(),cs=c.at("shape").get<std::vector<std::size_t>>();if(ps.empty()||ps[0]!=before||cs.empty()||cs[0]!=after||ps.size()!=cs.size()||p.at("dtype")!=c.at("dtype"))throw std::runtime_error("training append changed original array ABI");std::size_t bytes=before;for(std::size_t k=1;k<ps.size();++k){if(ps[k]!=cs[k])throw std::runtime_error("training append changed original array tail shape");bytes*=ps[k];}const auto type=p.at("dtype").get<std::string>();if(type=="<f8"||type=="<u8"||type=="<i8")bytes*=8;else if(type=="<u4"||type=="<i4")bytes*=4;else if(type=="<u2"||type=="<i2")bytes*=2;else if(type!="|u1"&&type!="|i1")throw std::runtime_error("training append unsupported original array scalar");eventio::verify(std::filesystem::u8path(p.at("file").get<std::string>()),p.at("sha256"));eventio::verify(std::filesystem::u8path(c.at("file").get<std::string>()),c.at("sha256"));std::ifstream old(std::filesystem::u8path(p.at("file").get<std::string>()),std::ios::binary),now(std::filesystem::u8path(c.at("file").get<std::string>()),std::ios::binary);old.seekg(p.value("offset_bytes",std::size_t(0)));now.seekg(c.value("offset_bytes",std::size_t(0)));std::array<char,65536>a{},b{};while(bytes){const auto count=std::min(bytes,a.size());old.read(a.data(),count);now.read(b.data(),count);if(old.gcount()!=std::streamsize(count)||now.gcount()!=std::streamsize(count)||std::memcmp(a.data(),b.data(),count))throw std::runtime_error("training append original source prefix bytes differ");bytes-=count;}}
 }
 static bool training_role(const std::string&role){return role=="historical_regression"||role=="development_candidate";}
 static void typed(const eventio::Json&ref,const char*dtype,std::size_t count){if(ref.at("dtype")!=dtype||ref.at("shape")!=eventio::Json::array({count}))throw std::runtime_error("representative v2 binary ABI mismatch");}
 static JointRepresentativeEligibilityV2 load(const eventio::Json&m,const std::string&packet_sha,const std::vector<unsigned char>&physical){
  JointRepresentativeEligibilityV2 e;e.training=physical;
  for(auto value:physical){if(value>1)throw std::runtime_error("nonbinary physical source membership");e.physical_count+=value;}
  e.training_count=e.physical_count;e.fit_cap=e.physical_count;
  const auto schema=m.at("schema").get<std::string>();
  if(schema=="rinkfit.per_frame_representative_source_selection.v1")return e;
  if(schema!="rinkfit.per_frame_representative_source_selection.v2")throw std::runtime_error("unsupported representative selection schema");
  e.enabled=true;e.binding=m.at("training_eligibility");const auto&b=e.binding;
  if(b.at("schema")!="rinkfit.source_training_eligibility_mask.v1"||b.at("source_packet_sha256")!=packet_sha||b.at("row_count")!=physical.size()||m.at("physical_admitted_rows")!=e.physical_count)throw std::runtime_error("representative v2 training domain source/count binding");
  const auto&mask=b.at("mask");typed(mask,"|u1",physical.size());
  e.training=checked_bytes(mask);if(e.training.size()!=physical.size())throw std::runtime_error("representative v2 full training mask size");
  e.training_count=0;for(std::size_t i=0;i<physical.size();++i){const auto value=e.training[i];if(value>1||(value&&!physical[i]))throw std::runtime_error("representative v2 training mask is not a subset of current physical admission");e.training_count+=value;}
  if(b.at("eligible_count")!=e.training_count||m.at("eligible_rows")!=e.training_count)throw std::runtime_error("representative v2 eligible count mismatch");
  e.provenance=m.at("eligibility_provenance");e.frame_roles=checked_json(e.provenance.at("frame_roles"));
  if(e.frame_roles.at("schema")!="rinkfit.joint_source_frame_protection.v1"||e.frame_roles.at("source_video_sha256").get<std::string>().size()!=64||!e.frame_roles.at("frame_roles").is_object())throw std::runtime_error("representative v2 requires original native frame role authority");
  if(e.provenance.contains("source_video")){const auto&r=e.provenance.at("source_video");eventio::verify(std::filesystem::u8path(r.at("file").get<std::string>()),r.at("sha256"));if(e.provenance.at("source_video").at("sha256")!=e.frame_roles.at("source_video_sha256"))throw std::runtime_error("representative v2 frame role source video mismatch");}
  if(e.provenance.contains("source_identity_verdicts")){
   const auto&ref=e.provenance.at("source_identity_verdicts");e.identity_verdicts=checked_json(ref);
   if(e.identity_verdicts.at("schema")!="rinkfit.source_identity_verdicts.v1"||e.identity_verdicts.at("source_packet_sha256")!=packet_sha)throw std::runtime_error("representative v2 identity verdict source mismatch");
   const auto base=std::filesystem::u8path(ref.at("file").get<std::string>()).parent_path();
   for(const auto&proof:e.identity_verdicts.value("evidence_files",eventio::Json::array()))checked_bytes(proof,base);
  }else throw std::runtime_error("representative v2 requires SHA-bound source identity verdicts with declared scope");
  const auto baseline=checked_json(m.at("original_selection"));
  e.original_source_rows=physical.size();
  eventio::Json append=eventio::Json::object();
  if(m.contains("source_append_selection_v1")){
   append=m.at("source_append_selection_v1");e.original_source_rows=append.at("parent_raw_rows").get<std::size_t>();
   if(append.at("schema")!="rinkfit.source_append_selection.v1"||append.at("current_raw_rows")!=physical.size()||e.original_source_rows>=physical.size()||append.at("old_domain_no_backfill")!=true||!same_ref(append.at("original_selection"),m.at("original_selection")))throw std::runtime_error("representative v2 appended selection range/baseline contract");
   e.append_proof=checked_json(append.at("append_source_proof"));const auto&p=e.append_proof;
   if(p.at("schema")!="rinkfit.authenticated_native_source_append_proof.v1"||p.at("passed")!=true||p.at("source_packet").at("sha256")!=packet_sha||!same_ref(p.at("parent_packet"),append.at("parent_packet"))||!same_ref(p.at("original_state_packet"),append.at("original_state_packet"))||p.at("parent_raw_rows")!=e.original_source_rows||p.at("current_raw_rows")!=physical.size()||p.at("new_source_rows")!=physical.size()-e.original_source_rows||p.at("source_model_projection_used")!=false||p.at("failed_or_missing_measurements_filled")!=false||p.at("old_failed_queries_restored")!=false||p.at("new_known_W_created")!=false||p.at("all_new_groups_rank_zero")!=true||p.at("maximum_midpoint_replay_difference_px")!=0||!same_ref(p.at("frame_roles"),e.provenance.at("frame_roles")))throw std::runtime_error("representative v2 append source proof contract");
   for(const char*key:{"source_evidence","programme_scope","f29_review"})checked_json(p.at(key));
   const auto current=checked_json(p.at("source_packet")),parent=checked_json(p.at("parent_packet"));checked_json(p.at("original_state_packet"));
   if(!same_ref(current.at("dense_parent_packet"),p.at("parent_packet"))||current.at("row_count")!=physical.size()||parent.at("row_count")!=e.original_source_rows||current.at("native_ad_source_binding")!=p.at("source_registry"))throw std::runtime_error("representative v2 append packet source registry differs");
   const auto&registry=p.at("source_registry");if(registry.at("row_start")!=e.original_source_rows||registry.at("count")!=physical.size()-e.original_source_rows||!same_ref(registry.at("source_evidence"),p.at("source_evidence")))throw std::runtime_error("representative v2 append source registry range");for(const char*key:{"source_id_utf8","source_id_offsets","source_family_index","source_store_ordinal"})checked_bytes(registry.at(key));
   exact_append_prefix(parent,current,e.original_source_rows,physical.size());
   const auto&policy=append.at("representative_policy");if(policy.at("selection_uses_fit_residual")!=false||policy.at("static_contour_support")!=true||policy.at("component_extrema_retained")!=true||policy.at("pixel_grid_spacing")!=48||policy.at("orientation_bins")!=2)throw std::runtime_error("representative v2 append changed original representative policy");
  }
  const auto original_packet_sha=append.empty()?packet_sha:append.at("original_state_packet").at("sha256").get<std::string>();
  if(baseline.at("schema")!="rinkfit.per_frame_representative_source_selection.v1"||baseline.at("source_packet").at("sha256")!=original_packet_sha||baseline.at("source_row_count")!=e.original_source_rows)throw std::runtime_error("representative v2 original fit-cap source binding");
  const auto&original_rows=m.at("original_fit_rows");
  if(original_rows.at("sha256")!=baseline.at("fit_rows").at("sha256"))throw std::runtime_error("representative v2 relocated original row identity differs");
  const auto original=checked_bytes(original_rows);const auto original_count=baseline.at("fit_count").get<std::size_t>();typed(original_rows,"<u8",original_count);
  if(original.size()/8!=original_count||original.size()%8)throw std::runtime_error("representative v2 original fit count bytes");
  e.original_fit.assign(physical.size(),0);std::uint64_t previous=0;
  for(std::size_t k=0;k<original_count;++k){std::uint64_t row;std::memcpy(&row,original.data()+8*k,8);if(row>=e.original_source_rows||(k&&row<=previous))throw std::runtime_error("representative v2 original rows outside source or not strictly sorted");e.original_fit[row]=1;previous=row;}
  if(!append.empty()){const auto&ref=append.at("appended_fit_rows");const auto bytes=checked_bytes(ref);if(bytes.size()%8)throw std::runtime_error("appended training row byte shape");e.appended_fit_count=bytes.size()/8;typed(ref,"<u8",e.appended_fit_count);if(e.appended_fit_count>append.at("maximum_added_fit_rows").get<std::size_t>()||append.at("maximum_added_fit_rows").get<std::size_t>()>physical.size()-e.original_source_rows)throw std::runtime_error("appended training fit cap exceeds real source suffix");std::uint64_t prior=0;for(std::size_t k=0;k<e.appended_fit_count;++k){std::uint64_t row;std::memcpy(&row,bytes.data()+8*k,8);if(row<e.original_source_rows||row>=physical.size()||(k&&row<=prior)||!e.training[row])throw std::runtime_error("appended training representative outside exact eligible source suffix");e.original_fit[row]=1;prior=row;}}
  e.fit_cap=m.at("fit_cap").get<std::size_t>();
  if(e.fit_cap>original_count+e.appended_fit_count||m.at("fit_count").get<std::size_t>()>e.fit_cap)throw std::runtime_error("representative v2 exceeds original thin fit cap plus authenticated suffix");
  const auto&method=m.at("method");if(method.at("no_backfill")!=true||method.at("selection_uses_fit_residual")!=false||method.at("all_physically_admitted_nonfit_rows_remain_inspection")!=true)throw std::runtime_error("representative v2 must preserve source selection and inspection policy");
  return e;
 }
 template<class Rows>void validate_source_rows(const Rows&rows,const std::vector<std::string>&shot_ids)const{
  if(!enabled)return;if(rows.size()!=training.size())throw std::runtime_error("representative v2 source row denominator");
  const auto&roles=frame_roles.at("frame_roles");
  for(std::size_t i=0;i<rows.size();++i)if(training[i]){const auto&r=rows[i];const auto key=std::to_string(r.pts);if(!roles.contains(key))throw std::runtime_error("representative v2 unknown native training exposure");const auto&role=roles.at(key);if(role.at("shot_id")!=r.shot_id||!training_role(role.at("role").template get<std::string>()))throw std::runtime_error("representative v2 protected or cross-shot source in training domain");}
  if(!append_proof.empty()){
   std::set<std::size_t>append_seen;std::map<std::int64_t,std::size_t>frame_rows;
   for(const auto&v:append_proof.at("rows")){const auto i=v.at("raw_row").template get<std::size_t>();if(i<original_source_rows||i>=rows.size()||!append_seen.insert(i).second)throw std::runtime_error("appended source proof missing/duplicate raw ordinal");const auto&r=rows[i];const auto key=std::to_string(r.pts);if(v.at("source_row_id")!=r.source_row_id||v.at("native_pts")!=r.pts||v.at("shot_id")!=r.shot_id||v.at("material_index")!=r.material||v.at("path")!=r.path||v.at("source_uv")!=eventio::Json(r.uv)||v.at("actual_replay_midpoint_uv")!=eventio::Json(r.uv)||v.at("source_identity_confirmed")!=true||v.at("max_absolute_replay_difference_px")!=0||v.at("reading_status")!="measured_paired_stroke_midline"||v.at("role")!=roles.at(key).at("role")||!training_role(v.at("role").template get<std::string>()))throw std::runtime_error("appended source proof changed original raw midline identity");++frame_rows[r.pts];}
   if(append_seen.size()!=rows.size()-original_source_rows)throw std::runtime_error("appended source proof omits raw source suffix");
   std::set<std::int64_t>seen_frames;for(const auto&f:append_proof.at("frames")){const auto pts=f.at("native_pts").template get<std::int64_t>();const auto key=std::to_string(pts);if(!frame_rows.count(pts)||!seen_frames.insert(pts).second||f.at("rows")!=frame_rows.at(pts)||f.at("role")!=roles.at(key).at("role")||f.at("source_video_sha256")!=frame_roles.at("source_video_sha256")||f.at("native_RGB_sha256").template get<std::string>().size()!=64||f.at("native_BGR_sha256").template get<std::string>().size()!=64)throw std::runtime_error("appended source original frame/role/video proof differs");}if(seen_frames.size()!=frame_rows.size())throw std::runtime_error("appended source proof omits original frames");
  }
  std::set<std::size_t>rejected,seen;
  for(const auto&v:identity_verdicts.value("rows",eventio::Json::array())){const auto i=v.at("raw_row").template get<std::size_t>();if(i>=rows.size()||!seen.insert(i).second)throw std::runtime_error("representative v2 identity verdict row outside source or duplicate");const auto&r=rows[i];if(v.at("source_row_id")!=r.source_row_id||v.at("native_pts")!=r.pts||v.at("material_index")!=r.material||v.at("path")!=r.path||v.at("shot_index").template get<std::size_t>()>=shot_ids.size()||shot_ids.at(v.at("shot_index").template get<std::size_t>())!=r.shot_id||v.at("uv")!=eventio::Json(r.uv)||v.at("uses_fitted_model_projection")!=false)throw std::runtime_error("representative v2 identity verdict changed source measurement or material");const auto decision=v.at("decision").template get<std::string>();if(decision!="confirmed"&&decision!="rejected"&&decision!="ambiguous"&&decision!="unchecked")throw std::runtime_error("representative v2 unknown identity verdict");if(decision!="confirmed"&&training[i])throw std::runtime_error("representative v2 unresolved or rejected source still trains");if(decision=="rejected")rejected.insert(i);}
  const auto declared=provenance.value("rejected_source_rows",std::vector<std::size_t>{});if(std::set<std::size_t>(declared.begin(),declared.end())!=rejected||declared.size()!=rejected.size())throw std::runtime_error("representative v2 rejected source partition differs from verdicts");
 }
 template<class Rows>void validate_correlated_group(const Rows&rows,const std::vector<unsigned char>&physical,const std::vector<unsigned char>&keep)const{
  std::size_t physical_members=0,eligible_members=0,retained=0;
  for(auto i:rows){if(physical.at(i)){++physical_members;eligible_members+=training.at(i)!=0;retained+=keep.at(i)!=0;}}
  if(enabled&&eligible_members&&eligible_members!=physical_members)throw std::runtime_error("representative v2 eligibility partially trims a positive-rank covariance group");
  if(retained&&retained!=eligible_members)throw std::runtime_error("representative selection split a correlated source covariance group");
 }
 void validate_inspection(const eventio::Json&m,const std::vector<unsigned char>&physical,const std::vector<unsigned char>&keep)const{
  if(!enabled)return;std::size_t fit=0;for(std::size_t i=0;i<keep.size();++i)if(keep[i]){if(!training.at(i)||!original_fit.at(i))throw std::runtime_error("representative v2 fit is ineligible or backfilled");++fit;}
  if(!append_proof.empty())for(std::size_t i=original_source_rows;i<keep.size();++i)if(keep[i]!=original_fit[i])throw std::runtime_error("appended representative list differs from exact declared new-source selection");
  if(fit!=m.at("fit_count").get<std::size_t>()||fit>fit_cap)throw std::runtime_error("representative v2 fit count/cap mismatch");
  const auto&ref=m.at("inspection_rows");const auto bytes=checked_bytes(ref);const auto expected=m.at("inspection_count").get<std::size_t>();typed(ref,"<u8",expected);
  if(bytes.size()/8!=expected||bytes.size()%8)throw std::runtime_error("representative v2 inspection size");
  std::size_t cursor=0;
  for(std::size_t i=0;i<physical.size();++i)if(physical[i]&&!keep[i]){if(cursor>=expected)throw std::runtime_error("representative v2 inspection omits admitted sources");std::uint64_t row;std::memcpy(&row,bytes.data()+8*cursor,8);if(row!=i)throw std::runtime_error("representative v2 inspection is not the exact admitted nonfit complement");++cursor;}
  if(cursor!=expected)throw std::runtime_error("representative v2 inspection contains extra sources");
 }
};
}
