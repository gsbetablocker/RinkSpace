#pragma once
#include "rinkfit/joint_engineering_identity_v1.hpp"
#include "rinkfit/joint_legacy_line_rechart_v1.hpp"
#include "rinkfit/joint_paired_advertisements_v1.hpp"
#include "rinkfit/joint_continuous_board_station_v1.hpp"
#include "rinkfit/joint_full_s01_subspace_v1.hpp"
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_native_repair_restart_v1.hpp"
#include "rinkfit/joint_shared_print_restore_v1.hpp"
#include "rinkfit/joint_exact_dependency_alias_v1.hpp"
#include <cstring>
#include <fstream>

namespace rinkfit {
// Restores a complete observed-row engineering primal, not Ceres/LM history
// and not a formal true-RS checkpoint. Strict mode is the same problem; the
// explicit warm-start mode permits an audited new engineering stage. Replay the original model/packet preparation
// first; restore exactly once, after mode/identity/domain setup and before GPU.
inline eventio::Json restore_joint_observed_primal_restart_v1(
    JointALBuilder& b, const eventio::Json& cfg,
    const std::string& model_file, const std::string& packet_file,
    const std::filesystem::path& out, const std::string& fingerprint) {
  using Json = eventio::Json;
  const bool warm = cfg.contains("observed_primal_warm_start_v1");
  const char* key = warm ? "observed_primal_warm_start_v1" : "observed_primal_restart_v1";
  if (!cfg.contains(key)) return {{"requested", false}};
  if (warm && cfg.contains("observed_primal_restart_v1")) throw std::runtime_error("ambiguous observed import mode");
  auto require = [](bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error("observed primal restart: " + why);
  };
  auto pin = [&](const Json& ref) {
    require(ref.is_object() && ref.contains("file") && ref.contains("sha256"),
            "missing file/SHA256 reference");
    auto p = std::filesystem::u8path(ref.at("file").get<std::string>());
    require(p.is_absolute(), "references must be absolute");
    p = JointExactDependencyAliasesV1::resolve(p, ref.at("sha256").get<std::string>());
    eventio::verify(p, ref.at("sha256").get<std::string>());
    return p;
  };
  require(b.source_uv_observed_row_initialization() && !b.source_uv_capture_v1(),
          "only the observed-row engineering objective may import this state");
  require(cfg.value("dense_source_fresh_capture_v1", false) &&
          !cfg.value("reinitialize_native_auxiliaries", true) &&
          !cfg.value("source_uv_capture_reseed_new_sources", false),
          "must retain original qualified preparation without reseeding");
  for (const auto* forbidden : {"resume_checkpoint", "resume_engineering_checkpoint",
          "resume_engineering_outer_state", "resume_source_uv_capture_checkpoint",
          "source_capture_init", "source_uv_capture_source_append_v4",
          "source_uv_capture_source_append_v5"})
    require(!cfg.contains(forbidden), std::string("conflicting entry: ") + forbidden);

  const auto manifest = eventio::read_json(pin(cfg.at(key)));
  require(manifest.at("schema") == (warm ? "rinkfit.observed_primal_warm_start.v1" : "rinkfit.observed_primal_restart.v1") &&
          manifest.at("objective_id") == "rinkfit.sourceUV_observed_row_finite_initialization.v1" &&
          manifest.at("optimizer_history_restored") == false &&
          manifest.at("quality_accepted") == false,
          "unsupported or falsely promoted restart manifest");
  const auto previous_cfg = eventio::read_json(pin(manifest.at("original_configuration")));
  if(warm&&manifest.value("allowed_semantic_change",std::string{})=="native_source_and_print_repair_v2")return restore_joint_native_repair_v1(b,cfg,manifest,previous_cfg,model_file,packet_file,out,pin);
  // Python issuer relocation is metadata only: preserve the old registry and
  // every expected byte hash, append archives for changed installed code only.
  const bool code_archive_changed = previous_cfg.at("archived_dependency_aliases") != cfg.at("archived_dependency_aliases");
  if (code_archive_changed) {
    auto previous = eventio::read_json(pin(previous_cfg.at("archived_dependency_aliases")));
    auto current = eventio::read_json(pin(cfg.at("archived_dependency_aliases")));
    const auto old_aliases=previous.at("aliases"), new_aliases=current.at("aliases");
    previous.erase("aliases"); current.erase("aliases");
    require(previous==current && previous.at("schema")=="rinkfit.exact_dependency_aliases.v1" &&
            new_aliases.size()>=old_aliases.size(), "archive registry must preserve old metadata");
    for (std::size_t i=0;i<old_aliases.size();++i)
      require(old_aliases.at(i)==new_aliases.at(i), "old code archive entry changed");
    std::set<std::pair<std::string,std::string>> seen;
    for (const auto& item:new_aliases) {
      const auto legacy=std::filesystem::u8path(item.at("legacy_path").get<std::string>());
      const auto actual=std::filesystem::u8path(item.at("actual_file").get<std::string>());
      const auto sha=item.at("sha256").get<std::string>();
      require(legacy.is_absolute() && actual.is_absolute() && legacy.extension()==".py" && actual.extension()==".py" &&
              JointExactDependencyAliasesV1::key(legacy)!=JointExactDependencyAliasesV1::key(actual) &&
              seen.emplace(JointExactDependencyAliasesV1::key(legacy),sha).second,
              "only separate unique exact Python code archives are permitted");
      eventio::verify(actual,sha);
    }
  }
  const bool source_append=warm&&manifest.at("allowed_semantic_change")=="explicit_native_ad_append_v1";
  const bool source_requalification=warm&&manifest.at("allowed_semantic_change")=="explicit_source_requalification_v1";
  const bool source_delta=source_append||source_requalification||(warm&&manifest.at("allowed_semantic_change")=="explicit_source_withdrawal_v1");
  const bool stage_v2=warm&&(source_delta||manifest.at("allowed_semantic_change")=="explicit_engineering_stage_v2");
  const bool shared_stage=stage_v2&&cfg.contains("runtime_shared_parameter_stage_v2");
  auto scrub = [warm,stage_v2,shared_stage,source_delta](Json j) {
    // These control execution/diagnostics only. All other fields, including
    // stage freezes, weights, source/domain refs, and PTX qualification match.
    for (const auto* name : {"observed_primal_restart_v1", "observed_primal_warm_start_v1", "num_threads",
            "max_inner_iterations", "audit_accepted_cost_breakdown_v1", "function_tolerance", "parameter_tolerance", "gradient_tolerance", "initial_trust_region_radius", "periodic_camera_angle_rebase",
            "linear_solver_backend", "write_accepted_complete_checkpoint",
            "defer_complete_continuation_auxiliary_v1", "defer_auxiliary_minimization_to_joint_phase_v1",
            "write_accepted_physical_preview", "accepted_source_anchor_review",
            "source_uv_capture_construct_problem_only", "archived_dependency_aliases", "audit_group_gradients_v2", "audit_actual_directional_v1", "audit_actual_family_directional_v1", "audit_actual_gradient_locations_v1", "observed_polyline_kkt_working_set_v1", "audit_actual_solver_step_v1"}) j.erase(name);
    if (warm) for (const auto* name : {"runtime_atlas_gauge_v1", "audit_expected_free_physical_blocks", "joint6_atlas_gauge_transition"}) j.erase(name);
    if(stage_v2)for(const auto*name:{"engineering_support_budget_v2","static_surface_motion_v1"})j.erase(name);
    if(shared_stage)for(const auto* name:{"runtime_shared_parameter_stage_v2","temporary_runtime_fixed_physical_blocks","temporary_runtime_fixed_physical_prefixes","temporary_runtime_fixed_nuisance_shots"})j.erase(name);
    if(source_delta)for(const auto* name:{"dense_source_qualification","engineering_dense_source_weights_v1","capture_certified_numeric_domain","audit_expected_source_counts","runtime_source_curve_coverage_v1"})j.erase(name);
    return j;
  };
  if(stage_v2){require(cfg.contains("engineering_support_budget_v2"),"v2 stage requires explicit common budget");if(previous_cfg.contains("static_surface_motion_v1"))require(cfg.at("static_surface_motion_v1")==previous_cfg.at("static_surface_motion_v1"),"warm stage cannot replace motion sources");}
  if (warm&&!stage_v2) require(cfg.contains("runtime_atlas_gauge_v1") && !previous_cfg.contains("runtime_atlas_gauge_v1") &&
                    manifest.at("allowed_semantic_change") == "runtime_atlas_coordinate_gauge_only",
                    "warm-start mode supports only the explicit atlas coordinate transition");
  auto old_semantics=scrub(previous_cfg),new_semantics=scrub(cfg);
  const auto gpu_stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
  if(gpu_stage.contains("gpu_kernel_transition_v1")){
    require(stage_v2,"GPU implementation transition requires explicit engineering stage");const auto proof=eventio::read_json(pin(gpu_stage.at("gpu_kernel_transition_v1")));
    require(proof.at("schema")=="rinkfit.same_objective_GPU_implementation_transition.v1"&&proof.at("passed")==true&&proof.at("previous_ptx_sha256")==previous_cfg.at("gpu_raw_uv").at("ptx_sha256")&&proof.at("current_ptx_sha256")==cfg.at("gpu_raw_uv").at("ptx_sha256")&&proof.at("shadow_receipt")==cfg.at("gpu_raw_uv").at("shadow_receipt"),"GPU old/new implementation identity");
    const auto shadow=eventio::read_json(pin(proof.at("shadow_receipt")));require(shadow.at("passed")==true&&shadow.at("ptx_sha256")==proof.at("current_ptx_sha256"),"GPU implementation shadow failed");auto a=previous_cfg.at("gpu_raw_uv"),c=cfg.at("gpu_raw_uv");for(auto*j:{&a,&c})for(const char*k:{"ptx","ptx_sha256","shadow_receipt"})j->erase(k);require(a==c,"GPU transition changed cache or objective semantics");old_semantics.erase("gpu_raw_uv");new_semantics.erase("gpu_raw_uv");
  }
  require(old_semantics == new_semantics,
          "objective, stage, source, domain or qualification configuration changed");
  const auto input_model = pin(manifest.at("input_model"));
  const auto input_packet = pin(manifest.at("source_packet"));

  const bool material_append=manifest.contains("material_append_topology");
  if(material_append){
    require(source_append,"material topology append requires the actual source append");
    const auto saved=eventio::read_json(pin(manifest.at("model"))),input=eventio::read_json(input_model);
    const auto& proof=manifest.at("material_append_topology");
    auto a=saved,c=input;const auto old=a.at("materials"),now=c.at("materials");a.erase("materials");c.erase("materials");
    require(now.size()>old.size()&&proof.at("schema")=="rinkfit.append_only_native_material_topology.v1"&&proof.at("parent_model")==manifest.at("model")&&proof.at("model")==manifest.at("input_model")&&proof.at("existing_physical_values_bounds_and_metadata_unchanged")==true&&proof.at("existing_material_count")==old.size()&&proof.at("appended_material_count")==now.size()-old.size()&&a==c,"material append changed the saved physical model");
    for(std::size_t i=0;i<old.size();++i)require(old[i]==now[i],"material append changed saved source identity "+std::to_string(i));
    std::set<std::string> labels;for(const auto& q:old)labels.insert(q.at("label"));
    for(std::size_t i=old.size();i<now.size();++i){const auto& q=now[i];const auto label=q.at("label").get<std::string>();require(label.rfind("Ad",0)==0&&label.find(".print.")!=std::string::npos&&labels.insert(label).second&&!q.at("paths").empty(),"invalid appended print identity");
      for(const auto& p:q.at("paths")){const int g=p.at("geometry_block"),ad=p.at("advertisement_block");require(p.at("kind")==2&&!p.at("physical_point").get<bool>()&&g>=0&&ad>=0&&g<int(b.core.model.blocks.size())&&ad<int(b.core.model.blocks.size())&&b.core.model.blocks.at(g).values.size()==6&&b.core.model.blocks.at(ad).values.size()==4&&p.at("points").size()>=2&&p.contains("provenance"),"appended print changed physical topology");for(const auto& xy:p.at("points"))require(xy.size()==2&&std::isfinite(xy[0].get<double>())&&std::isfinite(xy[1].get<double>()),"nonfinite new print contour");}
    }
  }

  require(eventio::file_sha256(model_file) == eventio::file_sha256(input_model) &&
          eventio::file_sha256(packet_file) == eventio::file_sha256(input_packet) &&
          b.core.packet.source_hash == manifest.at("source_packet").at("sha256"),
          "must replay the original producer model and identical source packet");
  const auto producer_source=source_delta?manifest.at("producer_source_packet"):manifest.at("source_packet");
  const auto producer_source_hash=producer_source.at("sha256").get<std::string>();
  std::size_t producer_rows=b.core.packet.rows.size();std::size_t withdrawn_rows=0;std::vector<std::size_t> restored_source_rows;
  if(source_delta){
    // Re-read the raw source contract, not inferred world coordinates. Numeric
    // overlays are restored independently from the complete saved state below.
    const auto prior=JointPacket::load(pin(producer_source).string(),b.core.model);
    producer_rows=prior.rows.size();
    if(source_append){
      require(producer_rows<b.core.packet.rows.size(),"native source append is empty");const auto&append=b.core.packet.source_json.at("native_ad_source_append");std::set<std::size_t>reviewed_prefix_withdrawals;auto expected_parent=producer_source;if(manifest.contains("reviewed_prefix_source_packet")){expected_parent=manifest.at("reviewed_prefix_source_packet");const auto prefix=eventio::read_json(pin(expected_parent));const auto&rr=manifest.at("reviewed_prefix_source_revision");const auto review=eventio::read_json(pin(rr));require(prefix.at("row_count")==producer_rows&&prefix.at("native_material_identity_revision")==rr&&b.core.packet.source_json.at("native_material_identity_revision")==rr&&review.at("schema")=="rinkfit.native_material_identity_revision.v1"&&review.at("source_rows_deleted")==0&&review.at("source_UV_ID_PTS_and_operators_unchanged")==true,"combined append prefix source review differs");for(const auto&i:review.at("withheld_rows"))require(reviewed_prefix_withdrawals.insert(i.get<std::size_t>()).second,"duplicate reviewed prefix withdrawal");}require(append.at("schema")=="rinkfit.native_ad_source_append.v1"&&append.at("parent_packet")==expected_parent&&append.at("row_start")==producer_rows&&append.at("count").get<std::size_t>()==b.core.packet.rows.size()-producer_rows,"native source exact prefix/suffix identity");
      for(std::size_t i=0;i<producer_rows;++i){const auto&a=prior.rows[i];const auto&c=b.core.packet.rows[i];require(a.source_row_id==c.source_row_id&&a.source_dataset_id==c.source_dataset_id&&a.uv==c.uv&&a.pts==c.pts&&a.shot_id==c.shot_id&&a.material==c.material&&a.kind==c.kind&&a.E==c.E&&c.active==(a.active&&!reviewed_prefix_withdrawals.count(i))&&a.path==c.path&&a.station==c.station&&((c.active&&c.shot_id!="s01")?c.row==c.uv[1]:a.row==c.row)&&a.alpha==c.alpha,"native ad append changed original raw state at row "+std::to_string(i));}
      // Raw JSON operators were verified by the source gate before compaction.
      // Here compare the live typed operators; observed-row mode only replaces
      // the numerical row seed with the measured sensor row.
      for(std::size_t i=0;i<producer_rows;++i){const auto&a=prior.rows[i];const auto&c=b.core.packet.rows[i];require(a.normal==c.normal&&a.width_block==c.width_block&&a.width_sign==c.width_sign&&a.source_station_intervals==c.source_station_intervals&&a.station_domain_path==c.station_domain_path&&bool(a.query)==bool(c.query),"native ad append changed original operator at row "+std::to_string(i));if(a.query)require(a.query->alpha==c.query->alpha&&a.query->uv==c.query->uv&&a.query->nx==c.query->nx&&a.query->ny==c.query->ny&&a.query->den==c.query->den&&a.query->origin==c.query->origin&&a.query->scale==c.query->scale,"native ad append changed original Query");}
      std::map<std::string,const JointStatisticalGroup*> current_groups;for(const auto&g:b.core.packet.groups)require(current_groups.emplace(g.id,&g).second,"duplicate runtime statistical group");for(const auto&a:prior.groups){const auto f=current_groups.find(a.id);require(f!=current_groups.end(),"native ad append lost original W group "+a.id);const auto&c=*f->second;require(a.statistic_kind==c.statistic_kind&&a.rows==c.rows&&a.rank==c.rank&&a.W==c.W,"native ad append changed original W group "+a.id);}
    }else{
    require(prior.rows.size()==b.core.packet.rows.size(),"withdrawal changed raw denominator");
    for(const auto& item:prior.source_json.at("arrays").items())
      if(item.key()!="admission"&&item.key()!="fit_role")
        require(item.value()==b.core.packet.source_json.at("arrays").at(item.key()),"withdrawal changed raw column: "+item.key());
    for(const auto* name:{"stroke_pairs","dense_source_binding","engineering_pixel_dependency_groups"})
      require(prior.source_json.value(name,Json())==b.core.packet.source_json.value(name,Json()),"withdrawal changed source semantics");
    require(prior.groups.size()==b.core.packet.groups.size(),"withdrawal changed statistical groups");
    for(std::size_t i=0;i<prior.groups.size();++i){const auto&a=prior.groups[i];const auto&c=b.core.packet.groups[i];
      require(a.id==c.id&&a.statistic_kind==c.statistic_kind&&a.rows==c.rows&&a.rank==c.rank&&a.W==c.W,"withdrawal changed original W/group");}
    const auto prefix=prior.source_json.at("dense_source_binding").at("row_start").get<std::size_t>();
    for(std::size_t i=0;i<prior.rows.size();++i){const auto&a=prior.rows[i];const auto&c=b.core.packet.rows[i];
      require(a.source_row_id==c.source_row_id&&a.uv==c.uv&&a.pts==c.pts&&a.shot_id==c.shot_id&&a.material==c.material&&a.kind==c.kind&&a.E==c.E,"withdrawal changed source values");
      require(a.normal==c.normal&&a.width_block==c.width_block&&a.width_sign==c.width_sign&&bool(a.query)==bool(c.query),"withdrawal changed source operator");
      if(a.query)require(a.query->alpha==c.query->alpha&&a.query->uv==c.query->uv&&a.query->nx==c.query->nx&&a.query->ny==c.query->ny&&a.query->den==c.query->den&&a.query->origin==c.query->origin&&a.query->scale==c.query->scale,"withdrawal changed original Query");
      require(!c.active||a.active||source_requalification,"withdrawal reactivated unknown source");
      if(c.active&&!a.active)restored_source_rows.push_back(i);
      require(i>=prefix||a.active==c.active,"withdrawal changed retained prefix");
      withdrawn_rows+=a.active&&!c.active;
    }
    require(withdrawn_rows>0||!restored_source_rows.empty(),"source membership transition is empty");
    if(source_requalification){
      require(withdrawn_rows==0,"source recovery also withdrew existing observations");
      const auto qualification=eventio::read_json(pin(cfg.at("dense_source_qualification")));const auto delta_file=pin(qualification.at("qualification_delta"));const auto delta=eventio::read_json(delta_file);
      require(delta.at("schema")=="rinkfit.dense_source_requalification_delta.v1"&&delta.at("source_packet")==producer_source&&!delta.at("selected_by_model_residual").get<bool>(),"source recovery evidence parent differs");
      pin(delta.at("restored_packet_rows"));const auto restored=eventio::read_array<std::uint64_t>(delta_file.parent_path(),delta.at("restored_packet_rows"),"<u8");
      require(restored==std::vector<std::uint64_t>(restored_source_rows.begin(),restored_source_rows.end()),"source recovery differs from native material evidence");
    }
    const bool reviewed_geometry_transition=cfg.value("runtime_shared_parameter_stage_v2",Json::object()).contains("native_identity_revision");if(reviewed_geometry_transition){const auto&ref=cfg.at("runtime_shared_parameter_stage_v2").at("native_identity_revision");const auto revision=eventio::read_json(pin(ref));const auto qualification=eventio::read_json(pin(cfg.at("dense_source_qualification")));require(revision.at("schema")=="rinkfit.native_material_identity_revision.v1"&&qualification.at("revision")==ref&&b.core.packet.source_json.at("native_material_identity_revision")==ref&&revision.at("source_rows_deleted")==0,"combined geometry repair requires preserved native identity review");}
    if(!source_requalification&&!reviewed_geometry_transition)for(const auto* key:{"engineering_support_budget_v2","runtime_shared_parameter_stage_v2"}){
      auto before=previous_cfg.at(key),after=cfg.at(key);before.erase("source_packet_sha256");after.erase("source_packet_sha256");
      if(std::string(key)=="runtime_shared_parameter_stage_v2")for(auto* stage:{&before,&after}){
        stage->erase("observed_circle_station_navigation");stage->erase("observed_finite_curve_station_navigation");
        for(auto& row:stage->at("blocks"))if(row.contains("source_support")){
          const auto ref=row.at("source_support");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
          auto support=eventio::read_json(ref.at("file").get<std::string>());support.erase("source_packet_sha256");row["source_support"]=std::move(support);
        }
      }
      require(before==after,"withdrawal changed physical stage or budget");
    }
  }
  }
  const auto binding = eventio::read_json(pin(manifest.at("state_binding")));
  const auto nuisance = eventio::read_json(pin(manifest.at("nuisance_manifest")));
  require(binding.at("schema") == "rinkfit.provisional_measured_row_initialization_state_binding.v1" &&
          binding.at("objective_id") == manifest.at("objective_id") &&
          binding.at("true_RS_verified") == false && binding.at("formal_Query_closed") == false &&
          binding.at("initialization_only") == true && binding.at("phase_II_allowed") == false &&
          binding.at("fresh_trueRS_v3_transition_required") == true &&
          binding.at("source_packet_sha256") == producer_source_hash &&
          binding.at("state_token") == manifest.at("expected_state_token"),
          "state binding does not describe this unaccepted observed objective");
  require(nuisance.at("schema") == "rinkfit.joint_explicit_PHR_nuisance_state.v1" &&
          nuisance.at("packet_sha256") == producer_source_hash &&
          nuisance.at("raw_rows") == producer_rows,
          "full raw-row auxiliary manifest differs");
  const auto cp_file = pin(manifest.at("checkpoint"));
  std::ifstream stream(cp_file);
  auto cp = read_checkpoint(stream);  // Existing parser validates full state token.
  const auto producer_fingerprint=verified_joint_engineering_producer_fingerprint_v1(previous_cfg,manifest,cp.runtime_fingerprint,cp.state.token(),producer_source_hash,pin);
  require(cp.runtime_fingerprint == producer_fingerprint && (warm || cp.runtime_fingerprint == fingerprint) && cp.outer.lambda.empty() &&
          cp.outer.nu.empty() && cp.outer.rho == 0 &&
          cp.state.data().identity.objective_id == manifest.at("objective_id") &&
          cp.state.token() == manifest.at("expected_state_token"),
          "checkpoint objective fingerprint, token, or AL state is incompatible");
  std::shared_ptr<JointMotionStoreV1> new_motion;
  if(stage_v2&&!previous_cfg.contains("static_surface_motion_v1"))new_motion=std::exchange(b.static_motion,nullptr);
  if(previous_cfg.value("runtime_shared_parameter_stage_v2",Json::object()).contains("paired_advertisement_templates")){
    restore_joint_confirmed_print_templates_v1(b.core.model,previous_cfg,false);
    restore_joint_paired_advertisements_v1(b.core.model,previous_cfg,false);
  }
  if (warm) b.set_engineering_identity(producer_fingerprint);
  if(source_delta){
    auto expected=b.capture_state().data();auto payload=cp.state.data();
    decltype(payload.parameters) retained;retained.reserve(expected.parameters.size());
    std::map<std::string,std::size_t> original_parameters;std::set<std::string> used;
    for(std::size_t i=0;i<payload.parameters.size();++i)require(original_parameters.emplace(payload.parameters[i].name,i).second,"duplicate archived primal parameter");
    std::map<std::string,std::vector<std::uint8_t>> archived_aux;
    for(const auto& parameter:expected.parameters){
      auto found=original_parameters.find(parameter.name);
      if(found!=original_parameters.end()){
        const auto& old=payload.parameters[found->second];require(old.values.size()==parameter.values.size(),"source transition changed primal shape");retained.push_back(old);used.insert(old.name);continue;
      }
      const auto slash=parameter.name.find('/');const auto family=parameter.name.substr(0,slash);
      require((source_requalification||source_append)&&(family=="station"||family=="row"||family=="alpha")&&slash!=std::string::npos,"source recovery introduced a physical parameter");
      const auto index=std::stoull(parameter.name.substr(slash+1));require(((source_append&&index>=producer_rows&&index<b.core.packet.rows.size())||std::binary_search(restored_source_rows.begin(),restored_source_rows.end(),index))&&parameter.values.size()==1,"source recovery introduced an unsupported auxiliary");
      if(source_append){retained.push_back(parameter);continue;}
      const auto name=family+".f64";if(!archived_aux.count(name))archived_aux[name]=eventio::bytes(pin(manifest.at("files").at(name)));
      const auto& bytes=archived_aux.at(name);require(bytes.size()==b.core.packet.rows.size()*sizeof(double),"raw restored auxiliary denominator differs");
      auto restored=parameter;std::memcpy(restored.values.data(),bytes.data()+index*sizeof(double),sizeof(double));require(std::isfinite(restored.values[0]),"nonfinite restored raw auxiliary");retained.push_back(std::move(restored));
    }
    for(const auto& parameter:payload.parameters){
      if(used.count(parameter.name))continue;const auto slash=parameter.name.find('/');const auto family=parameter.name.substr(0,slash);
      require((family=="station"||family=="row"||family=="alpha")&&slash!=std::string::npos,"source transition lost a physical parameter");const auto index=std::stoull(parameter.name.substr(slash+1));require(index<b.core.packet.rows.size()&&!b.core.packet.rows[index].active,"source transition lost an active auxiliary");
    }
    if(source_append){const auto bytes=eventio::bytes(pin(manifest.at("files").at("path.u32")));require(bytes.size()==producer_rows*sizeof(std::uint32_t),"append original path bytes differ");std::vector<std::uint32_t>paths(producer_rows);std::memcpy(paths.data(),bytes.data(),bytes.size());require(payload.discrete.at("finite_material_paths_RLE_u32_v1")==joint_path_RLE(paths),"append original path encoding differs from full checkpoint");for(std::size_t i=producer_rows;i<b.core.packet.rows.size();++i)paths.push_back(std::uint32_t(b.core.packet.rows[i].path));payload.discrete["finite_material_paths_RLE_u32_v1"]=joint_path_RLE(paths);}
    payload.parameters=std::move(retained);payload.identity=expected.identity;
    payload.discrete["joint_packet_sha256"]=b.core.packet.source_hash;
    const StateSnapshot transferred(std::move(payload));b.restore_state(transferred);
    require(b.capture_state().token()==transferred.token(),"withdrawal primal transfer round-trip failed");
  }else{
    b.restore_state(cp.state);  // Checks complete physical topology/registry/order.
    require(b.capture_state().token() == cp.state.token(), "full primal round-trip failed");
  }
  if(previous_cfg.value("runtime_shared_parameter_stage_v2",Json::object()).contains("restore_original_print_vertices"))
    restore_joint_original_print_shapes_v1(b.core.model,previous_cfg,false);
  restore_joint_confirmed_print_templates_v1(b.core.model,previous_cfg,false);
  restore_joint_paired_advertisements_v1(b.core.model,previous_cfg,false);
  restore_joint_saved_s01_subspace_v1(b.core.model,previous_cfg,eventio::read_json(pin(manifest.at("model"))));
  restore_joint_sparse_s01_storage_v1(b.core.model,previous_cfg,false);
  restore_joint_continuous_board_stations_v1(b,previous_cfg,false);
  if(previous_cfg.value("runtime_shared_parameter_stage_v2",Json::object()).contains("confirmed_tripod_outside_rink_v1"))b.attach_confirmed_tripod_outside_rink(previous_cfg.at("runtime_shared_parameter_stage_v2").at("confirmed_tripod_outside_rink_v1"));
  const auto saved_model=eventio::read_json(material_append?input_model:pin(manifest.at("model")));
  // These are SHA-bound completed numeric transitions. Reconstruct their
  // metadata/domain views without projecting a physical scalar a second time.
  for(const auto*key:{"capture_zoom_feasibility_recovery_v1","legacy_line_coordinate_recharts_v1"})if(saved_model.contains(key)){const auto current=b.core.model.runtime_model_metadata.value(key,b.core.model.source_json.value(key,Json()));require(current.is_null()||current==saved_model.at(key),"saved numeric transition metadata differs");b.core.model.runtime_model_metadata[key]=saved_model.at(key);}
  restore_joint_legacy_line_domains_v1(b.core);
  const auto actual_model=b.core.model.exact_json();if(actual_model!=saved_model)eventio::atomic_json(out/"observed_restart_model_difference.json",Json::diff(saved_model,actual_model));
  require(actual_model==saved_model,"checkpoint physical values differ from its candidate model; see observed_restart_model_difference.json");

  const auto n = b.core.packet.rows.size();
  for (const auto* filename : {"station.f64", "row.f64", "alpha.f64", "path.u32"}) {
    const auto& ref = manifest.at("files").at(filename);
    const auto path = pin(ref);
    const auto& expected = nuisance.at("files").at(filename);
    const bool is_path = std::string(filename) == "path.u32";
    const auto bytes = eventio::bytes(path);
    require(ref.at("sha256") == expected.at("sha256") &&
            bytes.size() == expected.at("size_bytes").get<std::size_t>() &&
            bytes.size() == producer_rows * (is_path ? sizeof(std::uint32_t) : sizeof(double)),
            std::string("auxiliary SHA/size mismatch: ") + filename);
    for (std::size_t i = 0; i < producer_rows; ++i) {
      if (is_path) {
        std::uint32_t value; std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        require(value == std::uint32_t(b.core.packet.rows[i].path),
                "raw path mismatch at row " + std::to_string(i));
      } else {
        if(!b.core.packet.rows[i].active){
          double archived;std::memcpy(&archived,bytes.data()+i*sizeof(double),sizeof(double));
          require(std::isfinite(archived),"nonfinite archived raw auxiliary");
          if(std::string(filename)=="station.f64")b.core.station[i]=archived;
          else if(std::string(filename)=="row.f64"){b.core.row[i]=archived;b.core.packet.rows[i].row=archived;}
          else{b.core.alpha[i]=archived;b.core.packet.rows[i].alpha=archived;}
        }
        const double value = std::string(filename) == "station.f64" ? b.core.station[i] :
                             std::string(filename) == "row.f64" ? b.core.row[i] : b.core.alpha[i];
        require(std::memcmp(&value, bytes.data() + i * sizeof(value), sizeof(value)) == 0,
                std::string("raw auxiliary bit mismatch: ") + filename + "/" + std::to_string(i));
      }
    }
  }
  if(new_motion){b.static_motion=std::move(new_motion);eventio::atomic_json(out/"motion_initialization.json",initialize_joint_motion_domain_v1(*b.static_motion,b.core.model,false));}
  if (warm) b.set_engineering_identity(fingerprint);
  Json receipt = {{"schema", warm ? "rinkfit.observed_primal_warm_start_receipt.v1" : "rinkfit.observed_primal_restart_receipt.v1"},
      {"passed", true}, {"manifest", cfg.at(key)}, {"source_state_token", cp.state.token()},
      {"state_token", b.capture_state().token()}, {"same_problem_restart", !warm},
      {"previous_objective_fingerprint", producer_fingerprint},
      {"objective_fingerprint", fingerprint}, {"raw_rows", n},
      {"all_four_auxiliaries_bitwise_equal", !source_append},{"all_original_auxiliary_prefix_bits_equal",true},{"appended_raw_source_rows",n-producer_rows}, {"complete_primal_restored", true},
      {"optimizer_history_restored", false}, {"Ceres_trust_region_history_restored", false}, {"exact_code_archive_extension",code_archive_changed},
      {"source_UV_Query_E_W_or_qualification_changed", source_delta}, {"source_membership_withdrawn_rows",withdrawn_rows}, {"source_membership_restored_rows",restored_source_rows.size()}, {"original_source_UV_Query_E_W_unchanged",true}, {"complete_physical_motion_and_all_raw_auxiliaries_preserved",true},
      {"true_RS_verified", false}, {"quality_accepted", false}, {"phase_II_allowed", false}};
  eventio::atomic_json(out / (warm ? "observed_primal_warm_start_receipt.json" : "observed_primal_restart_receipt.json"), receipt);
  return receipt;
}
}  // namespace rinkfit
