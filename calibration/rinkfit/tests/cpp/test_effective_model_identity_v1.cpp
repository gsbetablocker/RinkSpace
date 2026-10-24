#include "rinkfit/joint_effective_model_identity_v1.hpp"
#include <iostream>
int main(int argc,char**argv){try{using namespace rinkfit;using J=eventio::Json;
 if(argc!=3)throw std::runtime_error("request and result required");const auto req=JointEffectiveInputV1::strict_json(argv[1]);const auto ref=req.at("original_model");const auto file=ref.at("file").get<std::string>();auto model=JointEventModel::load(file);
 J result={{"schema","rinkfit.actual_effective_model_identity_regression.v1"},{"original_model",ref},{"block_count",model.blocks.size()},{"solver_started",false}};
 auto report=assert_joint_effective_model_load_boundary_v1(model,file,ref);if(!report.at("raw_document_required_default_normalization").get<bool>())throw std::runtime_error("actual noncanonical df3 fixture did not exercise default normalization");result["actual_df3_load_boundary"]=report;
 auto reject=[&](const std::string&name,auto action){bool rejected=false;std::string error;try{action();}catch(const std::exception&e){rejected=true;error=e.what();}if(!rejected)throw std::runtime_error("mutation accepted: "+name);result["rejected"][name]=error;};
 auto block_changed=model;block_changed.blocks.at(0).values.at(0)+=1e-6;reject("live_block_value",[&]{assert_joint_effective_model_load_boundary_v1(block_changed,file,ref);});
 auto metadata_changed=model;metadata_changed.runtime_model_metadata["unauthorized_load_boundary_edit"]=true;reject("live_runtime_metadata",[&]{assert_joint_effective_model_load_boundary_v1(metadata_changed,file,ref);});
 auto source_changed=model;source_changed.source_json["unauthorized_source_edit"]=true;reject("source_document",[&]{assert_joint_effective_model_load_boundary_v1(source_changed,file,ref);});
 reject("different_cli_model",[&]{assert_joint_effective_model_load_boundary_v1(model,req.at("alternative_model").get<std::string>(),ref);});
 auto wrong_sha=ref;wrong_sha["sha256"]=std::string(64,'0');reject("wrong_original_sha",[&]{assert_joint_effective_model_load_boundary_v1(model,file,wrong_sha);});
 result["passed"]=true;result["arbitrary_later_material_or_rig_mutations_proven"]=false;eventio::atomic_json(argv[2],result);std::cout<<result.dump(2)<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
