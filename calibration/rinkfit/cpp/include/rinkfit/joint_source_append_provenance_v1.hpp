#pragma once
#include "rinkfit/event_io.hpp"
namespace rinkfit {
namespace source_append_v1 {
using Json=eventio::Json;
inline void need(bool v,const std::string&w){if(!v)throw std::runtime_error("authenticated source append: "+w);}
inline bool same(const Json&a,const Json&b){return a.at("sha256")==b.at("sha256")&&std::filesystem::absolute(std::filesystem::u8path(a.at("file").get<std::string>())).lexically_normal()==std::filesystem::absolute(std::filesystem::u8path(b.at("file").get<std::string>())).lexically_normal();}
inline std::filesystem::path pin(const Json&r){const auto p=std::filesystem::u8path(r.at("file").get<std::string>());need(p.is_absolute(),"absolute reference required");eventio::verify(p,r.at("sha256"));if(r.contains("byte_count"))need(std::filesystem::file_size(p)==r.at("byte_count").get<std::size_t>(),"reference byte count");return p;}
// Authorization is based on exact old and current packet bytes plus an explicit
// migration policy. This is never an alias for source data or a SHA override.
inline Json authorization(const Json&cfg,const std::string&old_sha,const std::string&current_sha){
 need(cfg.contains("representative_source_append_primal_v1"),"missing explicit new-source primal policy");const auto&p=cfg.at("representative_source_append_primal_v1");
 need(p.at("schema")=="rinkfit.authenticated_source_append_primal.v1"&&p.at("from_source_packet").at("sha256")==old_sha&&p.at("source_packet").at("sha256")==current_sha&&old_sha!=current_sha,"source/primal migration identity");
 const auto proof=eventio::read_json(pin(p.at("append_source_proof")));const auto current=eventio::read_json(pin(p.at("source_packet")));const auto parent=eventio::read_json(pin(p.at("parent_packet")));pin(p.at("from_source_packet"));
 need(proof.at("schema")=="rinkfit.authenticated_native_source_append_proof.v1"&&proof.at("passed")==true&&same(proof.at("source_packet"),p.at("source_packet"))&&same(proof.at("original_state_packet"),p.at("from_source_packet"))&&same(proof.at("parent_packet"),p.at("parent_packet")),"source proof exact original/parent/current refs");
 need(same(current.at("dense_parent_packet"),p.at("parent_packet"))&&same(parent.at("dense_parent_packet"),p.at("from_source_packet"))&&proof.at("source_registry")==current.at("native_ad_source_binding"),"actual original->withdrawal->append ancestry");
 const auto n0=p.at("from_raw_rows").get<std::size_t>(),n=p.at("to_raw_rows").get<std::size_t>();need(n>n0&&parent.at("row_count")==n0&&current.at("row_count")==n&&proof.at("parent_raw_rows")==n0&&proof.at("current_raw_rows")==n&&current.at("native_ad_source_binding").at("row_start")==n0&&current.at("native_ad_source_binding").at("count")==n-n0,"full raw prefix/append denominator");
 need(p.at("old_domain_reseeded")==false&&p.at("tail_initialization")=="exact_authenticated_packet_station_row_alpha_path"&&p.at("same_objective_resume")==false&&!p.at("reason").get<std::string>().empty(),"explicit initialization semantics");return proof;
}
} // namespace source_append_v1
}
