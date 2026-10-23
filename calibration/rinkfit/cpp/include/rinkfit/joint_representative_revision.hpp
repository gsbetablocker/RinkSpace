#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_path_state.hpp"
#include <cstring>
#include <set>

namespace rinkfit {
// A reviewed source withdrawal changes the objective, never the saved primal.
// The full source guard proves the new packet differs only in reviewed membership.
template<class Pin> inline bool representative_source_revision_v1(
    JointALBuilder& b, const eventio::Json& saved, const eventio::Json& cfg, Pin pin) {
  using J=eventio::Json;
  if(saved.at("source_packet_sha256")==b.core.packet.source_hash)return false;
  const bool rebind=cfg.contains("representative_material_rebind_v1");
  if(!rebind&&!cfg.contains("representative_source_withdrawal_v1"))throw std::runtime_error("source revision requires explicit native RGB review");
  const auto ref=cfg.at(rebind?"representative_material_rebind_v1":"representative_source_withdrawal_v1");const auto revision=eventio::read_json(pin(ref));
  const auto qual=eventio::read_json(pin(cfg.at("dense_source_qualification")));
  if(revision.at("schema")!=(rebind?"rinkfit.native_print_identity_rebind.v1":"rinkfit.native_material_identity_revision.v1")||revision.at("parent_packet").at("sha256")!=saved.at("source_packet_sha256")||revision.at("source_rows_deleted")!=0||revision.at(rebind?"source_UV_ID_PTS_and_membership_unchanged":"source_UV_ID_PTS_and_operators_unchanged")!=true||qual.at("schema")!=(rebind?"rinkfit.native_print_identity_rebind_qualification.v1":"rinkfit.native_material_identity_revision_qualification.v1")||qual.at("revision")!=ref||qual.at("source_packet").at("sha256")!=b.core.packet.source_hash)throw std::runtime_error("source withdrawal ancestry/measurement mismatch");
  pin(revision.at("parent_packet"));pin(revision.at("review"));
  const auto previous=eventio::read_json(pin(saved.at("representative_source_selection_v1")));const auto current=eventio::read_json(pin(cfg.at("representative_source_selection_v1")));
  if(previous.at("source_packet").at("sha256")!=saved.at("source_packet_sha256")||current.at("source_packet").at("sha256")!=b.core.packet.source_hash)throw std::runtime_error("revised selection ancestry");
  const auto old_bytes=eventio::bytes(pin(previous.at("fit_rows"))),new_bytes=eventio::bytes(pin(current.at("fit_rows")));
  if(old_bytes.size()%8||new_bytes.size()%8)throw std::runtime_error("revised selection binary size");
  std::set<std::uint64_t> withdrawn;for(const auto& v:revision.value("withheld_rows",J::array()))withdrawn.insert(v.get<std::uint64_t>());
  std::size_t cursor=0;
  for(std::size_t k=0;k<old_bytes.size();k+=8){std::uint64_t row;std::memcpy(&row,old_bytes.data()+k,8);if(withdrawn.count(row))continue;std::uint64_t next;if(cursor+8>new_bytes.size())throw std::runtime_error("revision lost an unreviewed observation");std::memcpy(&next,new_bytes.data()+cursor,8);if(next!=row)throw std::runtime_error("revision changed the retained selection");cursor+=8;}
  if(cursor!=new_bytes.size())throw std::runtime_error("revision added unreviewed fit observations");
  return true;
}
inline eventio::Json restore_representative_revised_primal_v1(
    JointALBuilder& b,const StateSnapshot& state,const eventio::Json& saved,const std::string& restored_raw_paths) {
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
  auto extras=b.restore_native_repair_extras_v1(state);
  return {{"complete_saved_primal_verified",true},{"saved_nuisance_parameters_checked",checked},{"all_raw_nuisance_columns_restored",true},{"source_withdrawal_is_new_objective",true},{"extras",extras}};
}

inline eventio::Json apply_reviewed_print_station_navigation_v1(JointALBuilder& b,const eventio::Json& cfg){
 const auto ref=cfg.at("representative_material_rebind_v1");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto revision=eventio::read_json(ref.at("file").get<std::string>());const auto rr=revision.at("review");eventio::verify(rr.at("file").get<std::string>(),rr.at("sha256"));const auto review=eventio::read_json(rr.at("file").get<std::string>());std::size_t changed=0;
 for(const auto&e:review.at("rows")){const auto i=e.at("raw_row").get<std::size_t>();auto&r=b.core.packet.rows.at(i);const auto s=e.at("new_station").get<double>();if(r.material!=e.at("new_material_index")||r.source_row_id!=e.at("source_row_id")||s<0||s>1)throw std::runtime_error("reviewed print navigation source mismatch");const int path=e.value("new_path",e.value("old_path",0));if(path<0||std::size_t(path)>=b.core.model.materials.at(r.material).paths.size())throw std::runtime_error("reviewed print finite path mismatch");r.path=path;b.core.station[i]=r.station=s;++changed;}
 return {{"schema","rinkfit.reviewed_print_identity_navigation.v1"},{"rows",changed},{"complete_saved_primal_restored_first",true},{"source_UV_unchanged",true},{"physical_parameters_changed",false},{"source_projection_used_for_observations",false}};
}
}
