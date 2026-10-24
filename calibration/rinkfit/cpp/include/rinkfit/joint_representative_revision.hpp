#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_path_state.hpp"
#include "rinkfit/joint_station_alias_v1.hpp"
#include <cstring>
#include <set>

namespace rinkfit {
// A reviewed source withdrawal changes the objective, never the saved primal.
// The full source guard proves the new packet differs only in reviewed membership.
template<class Pin> inline eventio::Json representative_identity_ancestry_v1(
    const std::string& saved_sha,eventio::Json packet_ref,eventio::Json revision_ref,eventio::Json qualification_ref,Pin pin){
 using J=eventio::Json;auto need=[](bool ok,const char*why){if(!ok)throw std::runtime_error(why);};
 auto same=[](const J&a,const J&c){return a.is_object()&&c.is_object()&&a.contains("sha256")&&c.contains("sha256")&&a.at("sha256")==c.at("sha256")&&std::filesystem::u8path(a.at("file").get<std::string>()).lexically_normal()==std::filesystem::u8path(c.at("file").get<std::string>()).lexically_normal();};
 auto inherited=[&](J doc,const std::string&name,bool array=false){std::set<std::string>seen;while(array?(!doc.contains("arrays")||!doc.at("arrays").contains(name)):!doc.contains(name)){need(doc.contains("dense_parent_packet"),"missing continuation ancestry field");const auto ref=doc.at("dense_parent_packet");need(seen.insert(ref.at("sha256").get<std::string>()).second,"cyclic continuation source ancestry");doc=eventio::read_json(pin(ref));}return array?doc.at("arrays").at(name):doc.at(name);};
 auto current_revision=[&](const J&doc){const bool rebind=doc.contains("native_print_identity_rebind"),withdrawal=doc.contains("native_material_identity_revision");need(rebind!=withdrawal,"exactly one current source revision required");return doc.at(rebind?"native_print_identity_rebind":"native_material_identity_revision");};
 std::set<std::pair<std::string,std::string>>proofs;auto proof=[&](const J&ref){auto key=std::make_pair(ref.at("file").get<std::string>(),ref.at("sha256").get<std::string>());if(proofs.insert(key).second)pin(ref);};
 J generations=J::array();std::set<std::string>seen;std::size_t raw_rows=0;
 while(packet_ref.at("sha256")!=saved_sha){
  need(seen.insert(packet_ref.at("sha256").get<std::string>()).second,"cyclic continuation source ancestry");
  const auto packet=eventio::read_json(pin(packet_ref)),revision=eventio::read_json(pin(revision_ref)),qualification=eventio::read_json(pin(qualification_ref));
  const auto parent_ref=revision.at("parent_packet"),parent=eventio::read_json(pin(parent_ref)),parent_qual_ref=revision.at("parent_qualification"),parent_qual=eventio::read_json(pin(parent_qual_ref));
  const bool rebind=revision.at("schema")=="rinkfit.native_print_identity_rebind.v1",withdrawal=revision.at("schema")=="rinkfit.native_material_identity_revision.v1";
  need((rebind||withdrawal)&&revision.at("source_rows_deleted")==0&&revision.at(rebind?"source_UV_ID_PTS_and_membership_unchanged":"source_UV_ID_PTS_and_operators_unchanged")==true&&same(current_revision(packet),revision_ref)&&same(packet.at("dense_parent_packet"),parent_ref)&&qualification.at("schema")== (rebind?"rinkfit.native_print_identity_rebind_qualification.v1":"rinkfit.native_material_identity_revision_qualification.v1")&&same(qualification.at("source_packet"),packet_ref)&&same(qualification.at("revision"),revision_ref)&&same(qualification.at("parent_packet"),parent_ref)&&same(qualification.at("parent_qualification"),parent_qual_ref)&&same(parent_qual.at("source_packet"),parent_ref),"continuation mixed revision ancestry/qualification mismatch");
  const auto count=inherited(packet,"row_count").get<std::size_t>();if(!raw_rows)raw_rows=count;
  need(count==raw_rows&&inherited(parent,"row_count")==count&&revision.at("raw_rows")==count,"continuation revision changed raw row denominator");
  const auto review=eventio::read_json(pin(revision.at("review")));const auto&rows=review.at("rows");std::set<std::size_t>ids;
  if(rebind)need(review.at("source_pixels_changed")==false&&review.at("source_model_projection_used")==false,"continuation review changed/projected source pixels");
  else{need(review.at("schema")=="rinkfit.native_material_identity_review.v1"&&review.at("qualification_uses_model_residual")==false,"continuation withdrawal requires original RGB review");proof(review.at("native_frame_extractor"));}
  for(const auto&row:rows){const auto i=row.at("raw_row").get<std::size_t>();need(i<count&&ids.insert(i).second,"invalid continuation reviewed row");if(rebind){const double station=row.at("new_station");need(std::isfinite(station)&&station>=0&&station<=1,"nonfinite continuation station");}else{need(row.at("qualification")=="unknown"&&!row.at("finding").get<std::string>().empty()&&!row.at("native_images").empty(),"withdrawal lacks native source finding");for(const auto&image:row.at("native_images"))proof(image);}}
  const auto declared=revision.at(rebind?"rebound_rows":"withheld_rows").get<std::vector<std::size_t>>();need(!ids.empty()&&std::set<std::size_t>(declared.begin(),declared.end())==ids&&declared.size()==ids.size(),"continuation reviewed row partition mismatch");
  for(const char*name:{"active_rows","admitted_rows"}){const auto before=inherited(parent,name).get<std::size_t>();need(before>=(rebind?0:ids.size())&&inherited(packet,name)==before-(rebind?0:ids.size()),"continuation revision changed unreviewed membership");}
  for(const auto&item:packet.at("arrays").items()){const bool permitted=rebind?(item.key()=="material_index"||item.key()=="path"):(item.key()=="admission"||item.key()=="fit_role");if(!permitted)need(item.value()==inherited(parent,item.key(),true),"continuation revision changed UV/ID/PTS/E/W array");}
  const std::set<std::string>metadata{"schema","dense_parent_packet","row_count","arrays","active_rows","admitted_rows","native_print_identity_rebind","native_material_identity_revision","dense_parent_append"};for(const auto&item:packet.items())if(!metadata.count(item.key()))need(item.value()==inherited(parent,item.key()),"continuation revision changed source semantics");
  generations.push_back({{"kind",rebind?"rebind":"withdrawal"},{"packet",packet_ref},{"revision",revision_ref},{"review",revision.at("review")},{"rows",rows}});
  packet_ref=parent_ref;qualification_ref=parent_qual_ref;if(packet_ref.at("sha256")!=saved_sha)revision_ref=current_revision(parent);
 }
 std::map<std::size_t,J>merged,withdrawn;
 for(auto generation=generations.rbegin();generation!=generations.rend();++generation)for(auto row:generation->at("rows")){const auto i=row.at("raw_row").get<std::size_t>();need(!withdrawn.count(i),"continuation revised already withdrawn observation");auto old=merged.find(i);if(generation->at("kind")=="withdrawal"){if(old!=merged.end())for(const char*key:{"source_row_id","native_pts","uv"})need(old->second.at(key)==row.at(key),"continuation withdrawal source discontinuity");withdrawn[i]=std::move(row);continue;}if(old!=merged.end()){const auto&before=old->second;need(before.at("new_material_index")==row.at("old_material_index")&&before.value("new_path",before.value("old_path",0))==row.value("old_path",0)&&before.at("source_row_id")==row.at("source_row_id")&&before.at("native_pts")==row.at("native_pts")&&before.at("uv")==row.at("uv"),"continuation reviewed identity transition discontinuity");row["old_material_index"]=before.at("old_material_index");row["old_path"]=before.value("old_path",0);}merged[i]=std::move(row);}
 J rows=J::array(),ordered=J::array(),withheld=J::array(),withdrawal_rows=J::array();for(const auto&item:merged)rows.push_back(item.second);for(const auto&item:withdrawn){withheld.push_back(item.first);withdrawal_rows.push_back(item.second);}for(auto i=generations.rbegin();i!=generations.rend();++i){auto g=*i;g.erase("rows");ordered.push_back(std::move(g));}
 return {{"generations",ordered},{"rows",rows},{"withheld_rows",withheld},{"withdrawal_rows",withdrawal_rows},{"raw_rows",raw_rows}};
}

template<class Pin> inline bool representative_source_revision_v1(
    JointALBuilder& b,const eventio::Json&saved,const eventio::Json&cfg,Pin pin,eventio::Json*reviewed_rows=nullptr,eventio::Json*ancestry_report=nullptr){
 using J=eventio::Json;if(saved.at("source_packet_sha256")==b.core.packet.source_hash)return false;
 const auto qual=eventio::read_json(pin(cfg.at("dense_source_qualification")));if(qual.at("source_packet").at("sha256")!=b.core.packet.source_hash)throw std::runtime_error("continuation qualification binds a different current packet");
 const auto packet=eventio::read_json(pin(qual.at("source_packet")));const bool rebind=packet.contains("native_print_identity_rebind"),withdrawal=packet.contains("native_material_identity_revision");if(rebind==withdrawal)throw std::runtime_error("exactly one actual current source revision required");const auto key=rebind?"representative_material_rebind_v1":"representative_source_withdrawal_v1";if(!cfg.contains(key)||cfg.at(key)!=packet.at(rebind?"native_print_identity_rebind":"native_material_identity_revision"))throw std::runtime_error("continuation must authorize actual current revision");
 auto chain=representative_identity_ancestry_v1(saved.at("source_packet_sha256").get<std::string>(),qual.at("source_packet"),cfg.at(key),cfg.at("dense_source_qualification"),pin);if(chain.at("raw_rows")!=b.core.packet.rows.size())throw std::runtime_error("continuation raw source size mismatch");
 for(const auto&e:chain.at("rows")){const auto&r=b.core.packet.rows.at(e.at("raw_row").get<std::size_t>());if(r.material!=e.at("new_material_index")||r.source_row_id!=e.at("source_row_id")||r.pts!=e.at("native_pts")||r.uv!=e.at("uv").get<Vec2<double>>())throw std::runtime_error("merged reviewed continuation source mismatch");}
 for(const auto&e:chain.at("withdrawal_rows")){const auto&r=b.core.packet.rows.at(e.at("raw_row").get<std::size_t>());if(r.active||r.source_row_id!=e.at("source_row_id")||r.pts!=e.at("native_pts")||r.uv!=e.at("uv").get<Vec2<double>>())throw std::runtime_error("merged withdrawn continuation source mismatch");}
 if(reviewed_rows)*reviewed_rows=chain.at("rows");
 const auto previous=eventio::read_json(pin(saved.at("representative_source_selection_v1"))),current=eventio::read_json(pin(cfg.at("representative_source_selection_v1")));
 if(previous.at("source_packet").at("sha256")!=saved.at("source_packet_sha256")||current.at("source_packet").at("sha256")!=b.core.packet.source_hash)throw std::runtime_error("revised selection ancestry");
 const auto old_bytes=eventio::bytes(pin(previous.at("fit_rows"))),new_bytes=eventio::bytes(pin(current.at("fit_rows")));if(old_bytes.size()%8||new_bytes.size()%8)throw std::runtime_error("revised selection binary size");
 std::set<std::uint64_t>withdrawn;for(const auto&v:chain.at("withheld_rows"))withdrawn.insert(v.get<std::uint64_t>());std::size_t cursor=0;
 for(std::size_t k=0;k<old_bytes.size();k+=8){std::uint64_t row;std::memcpy(&row,old_bytes.data()+k,8);if(withdrawn.count(row))continue;std::uint64_t next;if(cursor+8>new_bytes.size())throw std::runtime_error("revision lost an unreviewed observation");std::memcpy(&next,new_bytes.data()+cursor,8);if(next!=row)throw std::runtime_error("revision changed retained selection");cursor+=8;}if(cursor!=new_bytes.size())throw std::runtime_error("revision added unreviewed fit observations");if(ancestry_report){chain.erase("rows");chain.erase("withdrawal_rows");*ancestry_report=std::move(chain);}return true;
}
inline eventio::Json restore_representative_revised_primal_v1(
    JointALBuilder& b,const StateSnapshot& state,const eventio::Json& saved,const std::string& restored_raw_paths,const eventio::Json&cfg=eventio::Json::object()) {
  auto need=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
  need(state.token()==saved.at("state_token"),"revised continuation state token");
  need(state.data().identity.objective_id=="rinkfit.sourceUV_observed_row_finite_initialization.v1","revised continuation objective");
  need(state.data().discrete.at("joint_packet_sha256")==saved.at("source_packet_sha256"),"revised continuation original source");
  need(state.data().discrete.at("finite_material_paths_RLE_u32_v1")==restored_raw_paths,"revised continuation full raw paths");
  const auto& p=state.data().parameters;const auto n=b.core.model.blocks.size();need(p.size()>=n,"revised continuation physical prefix");
  for(std::size_t i=0;i<n;++i)need(p[i].name==b.core.model.blocks[i].name&&p[i].values==b.core.model.blocks[i].values,"revised continuation physical values differ");
  std::size_t checked=0;
  for(std::size_t k=n;k<p.size();++k){const auto& v=p[k];const auto slash=v.name.find('/');const auto family=v.name.substr(0,slash);
    if(family=="station"||family=="row"||family=="alpha"){
      const auto i=std::stoull(v.name.substr(slash+1));const auto& a=family=="station"?b.core.station:family=="row"?b.core.row:b.core.alpha;
      need(i<a.size()&&v.values.size()==1&&std::memcmp(v.values.data(),&a[i],8)==0,"revised continuation raw auxiliary differs");++checked;
    }else need(v.name=="budget_sq"||family=="static_motion"||v.name.find("/numerical_zoom_enclosure")!=std::string::npos,"revised continuation unknown parameter family");
  }
  auto extras=b.restore_native_repair_extras_v1(state,cfg);
  return {{"complete_saved_primal_verified",true},{"saved_nuisance_parameters_checked",checked},{"all_raw_nuisance_columns_restored",true},{"source_withdrawal_is_new_objective",true},{"extras",extras}};
}

inline eventio::Json apply_reviewed_print_station_navigation_v1(JointALBuilder& b,const eventio::Json& cfg,const eventio::Json*inherited_rows=nullptr,const std::vector<std::uint32_t>*saved_raw_paths=nullptr){
 eventio::Json rows;if(inherited_rows)rows=*inherited_rows;else{const auto ref=cfg.at("representative_material_rebind_v1");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto revision=eventio::read_json(ref.at("file").get<std::string>());const auto rr=revision.at("review");eventio::verify(rr.at("file").get<std::string>(),rr.at("sha256"));rows=eventio::read_json(rr.at("file").get<std::string>()).at("rows");}std::size_t changed=0,preserved=0;
 for(const auto&e:rows){const auto i=e.at("raw_row").get<std::size_t>();auto&r=b.core.packet.rows.at(i);const auto s=e.at("new_station").get<double>();if(r.material!=e.at("new_material_index")||r.source_row_id!=e.at("source_row_id")||!std::isfinite(s)||s<0||s>1)throw std::runtime_error("reviewed print navigation source mismatch");const int path=e.value("new_path",e.value("old_path",0));if(path<0||std::size_t(path)>=b.core.model.materials.at(r.material).paths.size())throw std::runtime_error("reviewed print finite path mismatch");double station=s;if(saved_raw_paths){const int old_material=e.at("old_material_index").get<int>(),old_path=int(saved_raw_paths->at(i));station=joint_reviewed_station_initializer_v1(b.core.model,old_material,old_path,r.material,path,b.core.station[i],s);if(joint_same_station_parameterization_v1(b.core.model,old_material,old_path,r.material,path))++preserved;}r.path=path;b.core.station[i]=r.station=station;++changed;}
 return {{"schema","rinkfit.reviewed_print_identity_navigation.v1"},{"rows",changed},{"identical_parameterization_saved_stations_preserved",preserved},{"changed_parameterization_review_initializers",changed-preserved},{"complete_saved_primal_restored_first",true},{"source_UV_unchanged",true},{"physical_parameters_changed",false},{"source_projection_used_for_observations",false}};
}
}
