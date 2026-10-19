#pragma once
#include "rinkfit/joint_builder.hpp"

namespace rinkfit {
struct JointBaselineAuxiliaryPreservation {
 std::vector<std::uint8_t> preserved;
 eventio::Json receipt=eventio::Json::object();
};
// Numerical initializer only. The normal current-generation source guard must
// still run; this receipt never qualifies source pixels, W or measurements.
// Call before fresh auxiliary recapture and before preparing a GPU row layout.
inline JointBaselineAuxiliaryPreservation restore_joint_baseline_auxiliaries(
 JointProblemBuilder&core,const eventio::Json&manifest_ref,
 const std::filesystem::path&input_model,const std::filesystem::path&input_packet) {
 const auto manifest_file=std::filesystem::path(manifest_ref.at("file").get<std::string>());
 eventio::verify(manifest_file,manifest_ref.at("sha256"));const auto j=eventio::read_json(manifest_file);
 if(j.at("schema")!="rinkfit.exact_native_baseline_auxiliary_import.v1"||!j.value("all_original_UV_ID_PTS_material_kind_normal_Query_exact",false)||j.value("v98_auxiliary_state_loaded",true)||j.value("source_membership_UV_W_E_or_qualification_changed",true))throw std::runtime_error("unverified exact baseline auxiliary import");
 const auto packet_sha=j.at("current_packet").at("sha256").get<std::string>();
 if(core.packet.source_hash!=packet_sha)throw std::runtime_error("baseline auxiliary map belongs to a different exact qualified source generation; regenerate by ID/UV/operator proof");
 eventio::verify(input_packet,packet_sha);eventio::verify(input_model,j.at("model").at("sha256"));
 // Reject use after any physical numerical step, even when its file path did
 // not change. Old auxiliaries belong to the exact imported baseline model.
 const auto&source=core.model.source_json.at("blocks");if(source.size()!=core.model.blocks.size())throw std::runtime_error("baseline auxiliary model allocation changed");
 for(std::size_t i=0;i<core.model.blocks.size();++i){const auto original=source[i].at("values").get<std::vector<double>>();if(original!=core.model.blocks[i].values)throw std::runtime_error("baseline auxiliary import cannot overwrite a changed physical candidate");}
 const auto&files=j.at("files");const auto base=manifest_file.parent_path();
 const auto rows=eventio::read_array<std::uint64_t>(base,files.at("rows.u64"),"<u8");
 const auto station=eventio::read_array<double>(base,files.at("station.f64"),"<f8");
 const auto path=eventio::read_array<std::uint32_t>(base,files.at("path.u32"),"<u4");
 const auto row=eventio::read_array<double>(base,files.at("row.f64"),"<f8");
 const auto alpha=eventio::read_array<double>(base,files.at("alpha.f64"),"<f8");
 if(rows.empty()||rows.size()!=station.size()||rows.size()!=path.size()||rows.size()!=row.size()||rows.size()!=alpha.size()||rows.size()!=j.at("preserved_active_rows").get<std::size_t>())throw std::runtime_error("incomplete exact baseline auxiliary vectors");
 JointBaselineAuxiliaryPreservation result;result.preserved.resize(core.packet.rows.size(),0);
 // Validate every row before changing any runtime state: malformed maps cannot
 // leave a partially restored scene or silently revive an inactive source.
 for(std::size_t k=0;k<rows.size();++k){const auto i=rows[k];if(i>=core.packet.rows.size()||result.preserved[i])throw std::runtime_error("baseline auxiliary global row out of range or duplicated");
  const auto&o=core.packet.rows[i];if(!o.active||o.shot_id!="s01"||o.source_dataset_id!="s01_original"||o.width_block>=0)throw std::runtime_error("baseline auxiliary row lacks unchanged original S01 source/operator membership");
  if(path[k]>=core.model.materials.at(o.material).paths.size()||!o.station_allowed(core.model,int(path[k]),station[k],1e-12))throw std::runtime_error("baseline auxiliary outside current finite source path domain");
  if(o.kind==1&&!core.model.materials.at(o.material).paths[path[k]].physical_point)throw std::runtime_error("baseline fixed entity cannot become a sliding finite path");
  if(!std::isfinite(row[k])||row[k]<0||row[k]>1079||!std::isfinite(alpha[k]))throw std::runtime_error("baseline auxiliary outside native sensor or Query parameter domain");
  if(o.query){Vec2<double>query;if(!o.query->point(alpha[k],&query))throw std::runtime_error("baseline auxiliary outside original frozen finite Query");}
  result.preserved[i]=1;
 }
 for(std::size_t k=0;k<rows.size();++k){const auto i=rows[k];auto&o=core.packet.rows[i];o.path=int(path[k]);o.station=core.station[i]=station[k];o.row=core.row[i]=row[k];o.alpha=core.alpha[i]=alpha[k];}
 result.receipt={{"schema","rinkfit.production_exact_baseline_auxiliary_restore.v1"},{"manifest",manifest_ref},{"source_packet_sha256",packet_sha},{"input_model_sha256",j.at("model").at("sha256")},{"preserved_native_rows",rows.size()},{"original_Ad2_baseline",j.at("Ad2_same623")},{"preserved_saved_prediction_max_delta_px",j.at("maximum_saved_d78_prediction_change_px")},{"all_current_source_and_finite_Query_domains_verified",true},{"fresh_recapture_must_skip_preserved_rows",true},{"source_qualification_UV_W_or_membership_changed",false},{"old_AL_duals_or_objective_resumed",false}};
 return result;
}
}
