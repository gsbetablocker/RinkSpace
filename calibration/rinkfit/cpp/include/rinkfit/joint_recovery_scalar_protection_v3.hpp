#pragma once
#include "rinkfit/joint_recovery_source_capture.hpp"
#include <cstring>
namespace rinkfit {
inline void install_joint_recovery_scalar_prefix_manifolds_v3(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg){
 for(const auto&rec:cfg.value("protected_S01_scalar_prefixes",eventio::Json::array())){const auto i=rec.at("block_index").get<std::size_t>();auto&bl=b.core.model.blocks.at(i);if(!((i==50&&bl.name=="ice/circle/C0")||(i==1787&&bl.name=="ice/circle/A2"))||bl.name!=rec.at("block_name")||bl.values.size()!=4||bl.constant||!problem.HasParameterBlock(bl.values.data())||problem.IsParameterBlockConstant(bl.values.data()))throw std::runtime_error("invalid/free-circle suffix runtime manifold installation");problem.SetManifold(bl.values.data(),new ceres::SubsetManifold(4,{0,1,2}));}
}

inline eventio::Json protect_joint_recovery_s01_scalar_prefix_v3(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg){
 if(!cfg.contains("protected_S01_scalar_prefixes"))return protect_joint_recovery_s01(b,problem,cfg);
 if(!cfg.contains("engineering_active_shot_ids")||cfg.at("engineering_active_shot_ids")!=eventio::Json::array({"s01","s02","s03","s04"}))throw std::runtime_error("scalar-prefix recovery protection requires explicit first4 source scope");
 const auto fixed=recovery_source_capture_detail::fixed_shots(cfg,b.core.model);if(!fixed.count("s01"))throw std::runtime_error("scalar-prefix protection lost S01 nuisance freeze");
 const auto&records=cfg.at("protected_S01_scalar_prefixes");if(!records.is_array()||records.size()!=2)throw std::runtime_error("source role split requires its two exact protected circles");
 std::map<std::size_t,eventio::Json>permitted;
 for(const auto&rec:records){
  const auto bi=rec.at("block_index").get<std::size_t>();const auto name=rec.at("block_name").get<std::string>();
  if(!((bi==50&&name=="ice/circle/C0")||(bi==1787&&name=="ice/circle/A2"))||!permitted.emplace(bi,rec).second)throw std::runtime_error("unsupported/duplicate protected scalar suffix");
  const auto&bl=b.core.model.blocks.at(bi);const auto prefix=rec.at("preserved_prefix_size").get<std::size_t>();const auto expected=rec.at("original_prefix_values").get<std::vector<double>>();
  if(bl.name!=name||prefix!=3||bl.values.size()!=4||expected.size()!=3||rec.at("new_free_scalar_indices")!=eventio::Json::array({3})||std::memcmp(bl.values.data(),expected.data(),3*sizeof(double)))throw std::runtime_error("source circle old scalar prefix changed");
  for(const auto&pair:std::vector<std::pair<std::string,const std::vector<double>*>>{{"original_prefix_lower",&bl.lower},{"original_prefix_upper",&bl.upper},{"original_prefix_scales",&bl.scales}}){auto v=rec.at(pair.first).get<std::vector<double>>();if(v.size()!=3||pair.second->size()!=4||std::memcmp(pair.second->data(),v.data(),3*sizeof(double)))throw std::runtime_error("source circle old prefix metadata changed");}
  const auto&ref=rec.at("source_ref");const std::string f=ref.contains("file")?ref.at("file").get<std::string>():ref.at("path").get<std::string>();eventio::verify(f,ref.at("sha256"));
  if(!problem.HasParameterBlock(bl.values.data())||problem.IsParameterBlockConstant(bl.values.data()))throw std::runtime_error("declared new source circle suffix is not actually free");
  for(const auto&r:b.core.packet.rows)if(r.active&&r.shot_id=="s01"){
   const auto&p=b.core.model.materials.at(r.material).paths.at(r.path);if(p.geometry_block==int(bi)&&(p.kind!=1||p.radius_index<0||p.radius_index>=3))throw std::runtime_error("S01 source consumes newly free circle suffix");
  }
 }
 std::size_t nuisance=0,protected_rows=0;
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active||!fixed.count(r.shot_id))continue;++protected_rows;for(double*value:{&b.core.station[i],&b.core.row[i],&b.core.alpha[i]})if(problem.HasParameterBlock(value)){if(!problem.IsParameterBlockConstant(value))problem.SetParameterBlockConstant(value);++nuisance;}}
 std::size_t covered=0;const auto blocks=recovery_source_capture_detail::s01_dependencies(b.core,&covered);if(!covered)throw std::runtime_error("scalar-prefix S01 protection has no active source");eventio::Json suffixes=eventio::Json::array();
 for(auto id:blocks){auto&bl=b.core.model.blocks.at(id);auto*value=bl.values.data();if(!problem.HasParameterBlock(value))throw std::runtime_error("S01 physical dependency not registered");if(problem.IsParameterBlockConstant(value))continue;
  auto it=permitted.find(id);if(it==permitted.end())throw std::runtime_error("unprotected S01 physical dependency is variable: "+bl.name);
  const auto*manifold=problem.GetManifold(value);const int tangent=problem.ParameterBlockTangentSize(value);if(!manifold||tangent!=1)throw std::runtime_error("circle suffix requires exactly its one-dimensional manifold");
  std::vector<double>J(4*tangent);if(!manifold->PlusJacobian(value,J.data()))throw std::runtime_error("circle suffix manifold Jacobian failed");for(int k=0;k<3;++k)for(int j=0;j<tangent;++j)if(J[k*tangent+j]!=0)throw std::runtime_error("S01 original scalar is variable in actual manifold");if(!std::isfinite(J[3])||J[3]==0)throw std::runtime_error("circle suffix has no actual free radius direction");
  suffixes.push_back({{"block",id},{"name",bl.name},{"protected_ambient_indices",{0,1,2}},{"free_suffix_index",3},{"actual_tangent_size",tangent},{"old_scalar_PlusJacobian_rows_exact_zero",true}});
 }
 return {{"schema","rinkfit.S01_original_scalar_projection_protection.v3"},{"temporary_fixed_nuisance_shots",fixed},{"protected_active_rows",protected_rows},{"actual_registered_nuisance_blocks_constant",nuisance},{"S01_active_rows_covered",covered},{"S01_unique_physical_blocks_checked",blocks.size()},{"complete_S01_projection_scalar_dependencies_constant",true},{"explicit_source_circle_free_suffixes",suffixes},{"source_or_model_values_changed",false},{"passed",true}};
}
}
