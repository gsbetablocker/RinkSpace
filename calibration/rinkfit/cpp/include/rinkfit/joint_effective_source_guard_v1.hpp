#pragma once
#include "rinkfit/joint_effective_input_v1.hpp"
#include "rinkfit/joint_dense_source_guard_v1.hpp"
#include "rinkfit/joint_representative_eligibility_v2.hpp"
namespace rinkfit {
class JointEffectiveSourceQualificationV1 {
 std::unique_ptr<JointDenseSourceQualificationV1> original_;eventio::Json report_;
 public:
 void assert_dependencies()const{if(JointEffectiveInputV1::prepared()){if(report_.is_null())throw std::runtime_error("prepared source proof absent");return;}if(!original_)throw std::runtime_error("effective source proof absent");original_->assert_dependencies();const auto&m=JointEffectiveInputV1::manifest();for(const char*k:{"physical_mask","training_mask","frame_roles","source_identity_verdicts","selection","original_qualification"})JointEffectiveInputV1::pin(m.at("qualification").at(k));}
 eventio::Json report()const{return report_;}
 static JointEffectiveSourceQualificationV1 load(const JointPacket&p,const JointEventModel&m){
  const auto&manifest=JointEffectiveInputV1::manifest();const auto&q=manifest.at("qualification");const auto path=JointEffectiveInputV1::pin(manifest.at("source_packet"));
  if(p.source_hash!=manifest.at("source_packet").at("sha256")||p.rows.size()!=manifest.at("raw_rows")||!p.source_json.contains("runtime_effective_input_v1"))throw std::runtime_error("effective source proof not bound to actual effective loader");
  JointEffectiveSourceQualificationV1 result;
  // Cold effective validation deliberately has no legacy validation-cache hit.
  // This executes the original per-generation source/RGB/membership mathematics.
  if(!JointEffectiveInputV1::prepared())result.original_=std::make_unique<JointDenseSourceQualificationV1>(JointDenseSourceQualificationV1::load(p,path,m,q.at("original_qualification"),{},8));
  std::vector<unsigned char>physical(p.rows.size());for(std::size_t i=0;i<p.rows.size();++i)physical[i]=p.rows[i].active?1:0;
  const auto mask=eventio::bytes(JointEffectiveInputV1::pin(q.at("physical_mask")));if(mask!=physical)throw std::runtime_error("effective physical mask differs from original actual native qualification");
  const auto selection=JointEffectiveInputV1::strict_json(JointEffectiveInputV1::pin(q.at("selection")));auto eligible=JointRepresentativeEligibilityV2::load(selection,p.source_hash,physical);eligible.validate_source_rows(p.rows,p.source_json.at("shot_ids").get<std::vector<std::string>>());
  const auto training=eventio::bytes(JointEffectiveInputV1::pin(q.at("training_mask")));if(training!=eligible.training||eligible.provenance.at("frame_roles")!=q.at("frame_roles")||eligible.provenance.at("source_identity_verdicts")!=q.at("source_identity_verdicts"))throw std::runtime_error("effective training eligibility differs from source role/verdict authority");
  const auto roles=JointEffectiveInputV1::strict_json(JointEffectiveInputV1::pin(q.at("frame_roles")));if(roles.at("source_video_sha256")!=m.source_json.at("source_clock").at("source_video_sha256"))throw std::runtime_error("effective source role clock differs");
  result.report_=JointEffectiveInputV1::receipt();result.report_["schema"]="rinkfit.actual_effective_source_qualification.v1";result.report_["effective_payload_complete"]=true;result.report_["full_original_native_source_mathematics_reexecuted"]=!JointEffectiveInputV1::prepared();result.report_["legacy_validation_cache_reused"]=false;result.report_["physical_rows"]=eligible.physical_count;result.report_["training_rows"]=eligible.training_count;if(result.original_)result.report_["original_mathematical_consumption"]=result.original_->report();else{result.report_["prior_completed_native_source_proof"]=JointEffectiveInputV1::prepared_receipt().at("cold_native_proof");result.report_["prepared_dependency_closure_verified_once"]=true;}result.report_["source_image_identity_proven"]=false;result.assert_dependencies();return result;
 }
};
}
