#pragma once
#include "rinkfit/joint_representative_resume_identity_v1.hpp"
#include <array>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>
namespace rinkfit {
// An internal fact from strict native checkpoint restoration, never a config
// switch. It does not waive source, state, physical or settled branch checks.
template<class Builder>inline eventio::Json verify_joint_observed_same_objective_resume_v1(const Builder&b,const RunCheckpoint&checkpoint,const eventio::Json&saved,const eventio::Json&identity_report,const eventio::Json&raw_manifest,bool source_revision,bool selection_transition){
 using J=eventio::Json;if(source_revision||selection_transition||!identity_report.at("exact_objective_continuation").get<bool>())return J();
 const auto n=b.core.packet.rows.size();auto need=[](bool v,const char*why){if(!v)throw std::runtime_error(std::string("verified observed continuation: ")+why);};need(raw_manifest.at("schema")=="rinkfit.joint_explicit_PHR_nuisance_state.v1"&&raw_manifest.at("raw_rows")==n&&raw_manifest.at("packet_sha256")==b.core.packet.source_hash&&saved.at("source_packet_sha256")==b.core.packet.source_hash,"raw/source denominator mismatch");need(checkpoint.state.data().identity.objective_id=="rinkfit.sourceUV_observed_row_finite_initialization.v1"&&b.capture_state().token()==checkpoint.state.token()&&checkpoint.state.token()==saved.at("state_token"),"complete native checkpoint roundtrip failed");J columns=J::object();
 for(const auto&column:std::vector<std::pair<std::string,const std::vector<double>*>>{{"station.f64",&b.core.station},{"row.f64",&b.core.row},{"alpha.f64",&b.core.alpha}}){need(column.second->size()==n,"complete raw scalar column size");const auto&spec=raw_manifest.at("files").at(column.first);need(spec.at("size_bytes")==n*sizeof(double),"raw column declared byte count");eventio::FastSHA256Stream h;h.update(reinterpret_cast<const std::uint8_t*>(column.second->data()),n*sizeof(double));const auto hash=h.finish();need(spec.at("sha256")==hash,"native CP and complete raw scalar sidecar disagree");columns[column.first]=hash;}
 eventio::FastSHA256Stream paths;std::array<std::uint32_t,4096>buffer;for(std::size_t i=0;i<n;){const auto count=std::min(buffer.size(),n-i);for(std::size_t k=0;k<count;++k)buffer[k]=static_cast<std::uint32_t>(b.core.packet.rows[i+k].path);paths.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),count*sizeof(std::uint32_t));i+=count;}const auto path_hash=paths.finish();need(raw_manifest.at("files").at("path.u32").at("size_bytes")==n*sizeof(std::uint32_t)&&raw_manifest.at("files").at("path.u32").at("sha256")==path_hash,"native CP and complete raw path sidecar disagree");columns["path.u32"]=path_hash;
 return {{"schema","rinkfit.verified_same_objective_observed_complete_resume.v1"},{"source_packet_sha256",b.core.packet.source_hash},{"state_token",checkpoint.state.token()},{"runtime_fingerprint",checkpoint.runtime_fingerprint},{"model",saved.at("model")},{"checkpoint",saved.at("checkpoint")},{"raw_rows",n},{"complete_raw_columns_sha256",columns},{"native_complete_state_roundtrip_exact",true},{"raw_cell_path_global_PHR_and_motion_restored_by_native_checkpoint",true},{"same_objective_startup_navigation_deferred_to_accepted_and_settled_boundaries",true},{"quality_accepted",false}};
}
}
