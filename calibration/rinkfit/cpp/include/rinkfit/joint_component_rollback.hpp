#pragma once
#include "rinkfit/joint_builder.hpp"
namespace rinkfit {
struct JointShotComponentRollback {std::vector<std::uint8_t>preserved;eventio::Json receipt;};
// Call after exact accepted-prefix verification, before fresh append recapture
// and GPU preparation. Only numerical values change. Existing source support,
// variable flags and affine motion manifolds remain open and unchanged.
inline JointShotComponentRollback rollback_joint_s01_component(
 JointProblemBuilder&core,const eventio::Json&manifest_ref,
 const std::filesystem::path&input_model,const std::filesystem::path&input_packet){
 const auto manifest=std::filesystem::path(manifest_ref.at("file").get<std::string>());eventio::verify(manifest,manifest_ref.at("sha256"));const auto j=eventio::read_json(manifest);
 if(j.at("schema")!="rinkfit.exact_S01_component_rollback.v1"||j.at("shot_id")!="s01"||!j.value("immutable_source_prefix_verified",false)||j.value("old_duals_or_objective_resumed",true))throw std::runtime_error("unverified S01 component rollback contract");
 const auto packet_sha=j.at("packet").at("sha256").get<std::string>();if(core.packet.source_hash!=packet_sha)throw std::runtime_error("component rollback source generation differs");eventio::verify(input_packet,packet_sha);eventio::verify(input_model,j.at("input_model").at("sha256"));
 auto load_bound=[&](const char*key){const auto&r=j.at(key);const auto file=std::filesystem::path(r.at("file").get<std::string>());eventio::verify(file,r.at("sha256"));return eventio::read_json(file);};
 const auto accepted=load_bound("accepted_model"),baseline=load_bound("baseline_model");const auto&ab=accepted.at("blocks");const auto&bb=baseline.at("blocks");
 if(ab.size()!=core.model.blocks.size()||bb.size()!=ab.size())throw std::runtime_error("component model allocation differs");
 const auto owns=[](const std::string&name){return name.rfind("shot/s01/",0)==0;};
 std::vector<std::size_t>blocks;std::vector<std::vector<double>>replacement;std::vector<std::uint8_t>selected(ab.size());std::size_t open=0;
 for(std::size_t i=0;i<ab.size();++i){const auto&b=core.model.blocks[i];if(ab[i].at("name")!=b.name||bb[i].at("name")!=b.name)throw std::runtime_error("component physical block identity differs");const auto a=ab[i].at("values").get<std::vector<double>>(),z=bb[i].at("values").get<std::vector<double>>();if(a!=b.values||a.size()!=z.size())throw std::runtime_error("component rollback must start at exact accepted physical values");
  if(owns(b.name)){if(!std::all_of(z.begin(),z.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("nonfinite baseline component");blocks.push_back(i);replacement.push_back(z);selected[i]=1;if(!b.constant)++open;}
  else if(b.name.rfind("shot/",0)!=0&&a!=z)throw std::runtime_error("shared scene/rig/geometry changed; S01-only rollback is not equivalent");
 }
 if(blocks.empty())throw std::runtime_error("missing S01 physical component");
 if(accepted.at("shots").at("s01")!=baseline.at("shots").at("s01")||core.model.source_json.at("shots").at("s01")!=accepted.at("shots").at("s01"))throw std::runtime_error("S01 camera function allocation/time scope changed");
 for(const auto&s:core.model.shots)if(s.id!="s01"){if(selected.at(s.reference_angles_block))throw std::runtime_error("S01 block shared with another shot reference");for(auto block:s.control_sources)if(selected.at(block))throw std::runtime_error("S01 control shared with another shot");}
 for(const auto&r:core.model.rigs)if(selected.at(r.head_block)||selected.at(r.optics_block))throw std::runtime_error("S01 block shared with rig");
 const auto&files=j.at("files");const auto dir=manifest.parent_path();const auto rows=eventio::read_array<std::uint64_t>(dir,files.at("rows.u64"),"<u8");const auto station=eventio::read_array<double>(dir,files.at("station.f64"),"<f8"),row=eventio::read_array<double>(dir,files.at("row.f64"),"<f8"),alpha=eventio::read_array<double>(dir,files.at("alpha.f64"),"<f8");const auto path=eventio::read_array<std::uint32_t>(dir,files.at("path.u32"),"<u4");
 if(rows.size()!=j.at("component_raw_rows").get<std::size_t>()||rows.empty()||station.size()!=rows.size()||row.size()!=rows.size()||alpha.size()!=rows.size()||path.size()!=rows.size())throw std::runtime_error("incomplete S01 component auxiliaries");
 JointShotComponentRollback result;result.preserved.resize(core.packet.rows.size(),0);std::size_t active=0;
 for(std::size_t k=0;k<rows.size();++k){const auto i=rows[k];if(i>=core.packet.rows.size()||result.preserved[i])throw std::runtime_error("duplicate/out-of-range rollback row");const auto&o=core.packet.rows[i];if(o.shot_id!="s01"||o.source_dataset_id!="s01_original"||o.width_block>=0||o.stroke_pair_index>=0)throw std::runtime_error("rollback row is not the original S01 source/operator");if(!std::isfinite(station[k])||!std::isfinite(row[k])||!std::isfinite(alpha[k])||row[k]<0||row[k]>1079||path[k]>=core.model.materials.at(o.material).paths.size())throw std::runtime_error("invalid component auxiliary");if(o.active){++active;if(!o.station_allowed(core.model,path[k],station[k],1e-12))throw std::runtime_error("rollback outside finite source station domain");if(o.kind==1&&!core.model.materials.at(o.material).paths[path[k]].physical_point)throw std::runtime_error("rollback fixed entity identity changed");if(o.query){Vec2<double>q;if(!o.query->point(alpha[k],&q))throw std::runtime_error("rollback outside frozen finite Query");}}result.preserved[i]=1;}
 for(std::size_t i=0;i<core.packet.rows.size();++i)if((core.packet.rows[i].shot_id=="s01")!=bool(result.preserved[i]))throw std::runtime_error("rollback does not cover the complete current S01 component");
 if(active!=j.at("component_active_rows").get<std::size_t>())throw std::runtime_error("S01 source membership differs");
 auto fingerprint=[&](bool target){eventio::SHA256Stream h;auto push=[&](const auto&x){h.update(reinterpret_cast<const std::uint8_t*>(&x),sizeof(x));};for(std::size_t i=0;i<core.model.blocks.size();++i)if(bool(selected[i])==target){push(i);const auto&b=core.model.blocks[i];push(b.constant);for(auto x:b.values)push(x);}for(std::size_t i=0;i<core.packet.rows.size();++i)if(bool(result.preserved[i])==target){push(i);const auto&o=core.packet.rows[i];push(o.path);push(o.station);push(o.row);push(o.alpha);push(core.station[i]);push(core.row[i]);push(core.alpha[i]);}return h.finish();};
 const auto unchanged_before=fingerprint(false),target_before=fingerprint(true);const auto metadata=core.model.source_json.dump();
 // All source, model, domain and ownership checks finish before any mutation.
 for(std::size_t k=0;k<blocks.size();++k)std::copy(replacement[k].begin(),replacement[k].end(),core.model.blocks[blocks[k]].values.begin());
 for(std::size_t k=0;k<rows.size();++k){const auto i=rows[k];auto&o=core.packet.rows[i];o.path=path[k];o.station=core.station[i]=station[k];o.row=core.row[i]=row[k];o.alpha=core.alpha[i]=alpha[k];}
 const auto unchanged_after=fingerprint(false);if(unchanged_after!=unchanged_before||metadata!=core.model.source_json.dump())throw std::runtime_error("rollback altered a non-target value or source/manifold metadata");
 result.receipt={{"schema","rinkfit.production_S01_component_rollback.v1"},{"manifest",manifest_ref},{"source_packet_sha256",packet_sha},{"input_model_sha256",j.at("input_model").at("sha256")},{"restored_component_raw_rows",rows.size()},{"restored_component_active_rows",active},{"restored_physical_blocks",blocks.size()},{"target_variable_blocks_before_and_after",open},{"non_target_value_sha256_before",unchanged_before},{"non_target_value_sha256_after",unchanged_after},{"target_value_sha256_before",target_before},{"target_value_sha256_after",fingerprint(true)},{"shared_scene_mainrig_other_shots_bit_identical",true},{"source_UV_ID_membership_W_or_qualification_changed",false},{"source_support_manifold_or_variable_flags_changed",false},{"S01_permanently_frozen",false},{"old_duals_or_objective_resumed",false},{"caller_must_start_fresh_engineering_objective_without_old_duals",true},{"fresh_initializer_must_skip_preserved_component",true}};return result;
}
}
