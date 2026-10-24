#include "rinkfit/joint_effective_source_guard_v1.hpp"
#include "rinkfit/joint_dense_capture_weights_v1.hpp"
#include "rinkfit/joint_engineering_budget_v2.hpp"
#include <iostream>
int main(int argc,char**argv){try{using namespace rinkfit;using J=eventio::Json;
 if(argc<4)throw std::runtime_error("request, result, packet|qualify required");const auto req=JointEffectiveInputV1::strict_json(argv[1]);if(req.contains("archived_dependency_aliases"))JointExactDependencyAliasesV1::install(req.at("archived_dependency_aliases"));
 if(req.contains("native_source_archive_members"))JointArchivedSourceMembersV1::install(req.at("native_source_archive_members"));
 auto model=JointEventModel::load(req.at("model").at("file"));auto packet=JointPacket::load_effective_v1(req.at("effective_input_v1"),model,req.at("model").at("file").get<std::string>());
 J result=JointEffectiveInputV1::receipt();result["effective_payload_complete"]=true;result["typed_rows"]=packet.rows.size();result["typed_groups"]=packet.groups.size();result["full_native_numerical_payload_consumed"]=true;std::cout<<"effective_packet_consumed "<<packet.rows.size()<<std::endl;
 if(std::string(argv[3])=="qualify"){auto qualification=JointEffectiveSourceQualificationV1::load(packet,model);result["qualification"]=qualification.report();qualification.assert_dependencies();}
 result["cross_language_fixture_sha256"]=JointEffectiveInputV1::digest(req.at("canonical_fixture"));if(result.at("cross_language_fixture_sha256")!=req.at("canonical_fixture_sha256"))throw std::runtime_error("cross-language canonical fixture mismatch");
 auto cfg=req.at("config");JointProblemBuilder core(std::move(model),std::move(packet));auto budget=load_joint_dense_capture_weights_v1(cfg,core);if(!budget.enabled||budget.receipt.at("schema")!="rinkfit.actual_effective_source_budget_validation.v1")throw std::runtime_error("effective budget consumer missing");result["effective_budget"]=budget.receipt;
 result["passed"]=true;result["solver_started"]=false;eventio::atomic_json(argv[2],result);JointEffectiveInputV1::clear();return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
