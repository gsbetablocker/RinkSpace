#pragma once
#include "rinkfit/joint_model.hpp"
#include <Eigen/Dense>
namespace rinkfit {
// Restore all directions in the original cubic subspace. Numerical bump modes
// were an initializer restriction, not observed camera motion or a new datum.
inline eventio::Json restore_joint_full_s01_subspace_v1(JointEventModel& m,const eventio::Json& cfg){
 using Json=eventio::Json;const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.value("restore_full_original_s01_subspace",false))return {{"enabled",false}};
 auto require=[](bool b,const std::string&s){if(!b)throw std::runtime_error("original S01 motion space: "+s);};
 require(stage.value("release_s01_trajectory",false),"trajectory must be explicitly variable");
 const std::string proof_key="s01_full_original_subspace_v1";
 auto record_end_controls=[&](Json report){if(stage.value("release_original_s01_end_controls",false)){int extra=0;std::set<int>seen,sparse_restored;const auto sparse=m.runtime_model_metadata.value("s01_original_sparse_storage_v1",m.source_json.value("s01_original_sparse_storage_v1",Json::object()));for(const auto& bi:sparse.value("restored_original_control_blocks",Json::array()))sparse_restored.insert(bi.get<int>());Json controls=Json::array();for(const auto& coordinate:m.source_json.at("shots").at("s01").at("control_coordinates")){const int bi=coordinate.at("block");const auto&b=m.blocks.at(bi);if(b.name.rfind("shot/s01/control/",0)!=0||sparse_restored.count(bi)||!seen.insert(bi).second)continue;require(b.values.size()==3,"original end-control shape");const int free=int(b.values.size()-b.fixed_indices.size());extra+=free;controls.push_back({{"block",bi},{"free",free}});}const int total=report.at("tangent_scalars").get<int>()+extra;require(total==m.source_json.at("s01_time_subspace_provenance").at("coarse_free_time_scalars").get<int>(),"complete original motion dimension differs");report["original_end_controls"]=controls;report["complete_original_tangent_scalars"]=total;report["dependent_angle_gauge_preserved"]=true;m.runtime_model_metadata[proof_key]=report;}return report;};
 if(m.runtime_model_metadata.contains(proof_key))return record_end_controls(m.runtime_model_metadata.at(proof_key));
 if(m.source_json.contains(proof_key))return record_end_controls(m.source_json.at(proof_key));
 const auto& provenance=m.source_json.at("s01_affine_recovery_corrections");
 require(provenance.at("original_exact_subspace_preserved")==true&&provenance.at("baseline_function_resampled")==false,"original trajectory provenance differs");
 const auto& shot=m.shots.at(m.shot_map.at("s01"));std::set<int> authoritative(shot.control_sources.begin(),shot.control_sources.end());
 Json manifolds=Json::array(),groups=Json::array();int before=0,after=0;
 for(const auto& old:m.source_json.at("linear_control_manifolds")){
  const int bi=old.at("block");const auto& block=m.blocks.at(bi);
  if(block.name.rfind("shot/s01/control_group/",0)!=0){manifolds.push_back(old);continue;}
  require(authoritative.count(bi)&&block.fixed_indices.empty(),"not an original active group");
  const int n=int(block.values.size());require(n==old.at("ambient_size"),"ambient shape changed");
  before+=old.at("tangent_size").get<int>();int tangent=n;double membership=0.;
  const auto& rows=old.at("constraint_matrix");
  if(!rows.empty()){
   Eigen::MatrixXd C(rows.size(),n);for(int i=0;i<C.rows();++i){require(rows[i].size()==std::size_t(n),"original constraint shape");for(int j=0;j<n;++j)C(i,j)=rows[i][j];}
   Eigen::FullPivLU<Eigen::MatrixXd> rank(C);rank.setThreshold(1e-11);require(rank.rank()==C.rows()&&C.rows()<n,"original constraints are not independent");
   tangent=n-int(C.rows());Eigen::HouseholderQR<Eigen::MatrixXd> qr(C.transpose());Eigen::MatrixXd Q=qr.householderQ()*Eigen::MatrixXd::Identity(n,n);Eigen::MatrixXd B=Q.rightCols(tangent);
   for(int j=0;j<tangent;++j){Eigen::Index index;B.col(j).cwiseAbs().maxCoeff(&index);if(B(index,j)<0)B.col(j)*=-1;}
   require((C*B).cwiseAbs().maxCoeff()<1e-10&&(B.transpose()*B-Eigen::MatrixXd::Identity(tangent,tangent)).cwiseAbs().maxCoeff()<1e-10,"full original nullspace construction failed");
   membership=(C*Eigen::Map<const Eigen::VectorXd>(block.values.data(),n)).cwiseAbs().maxCoeff();require(membership<2e-10,"saved values left original cubic subspace");
   Json basis=Json::array();for(int i=0;i<n;++i){std::vector<double>row(tangent);for(int j=0;j<tangent;++j)row[j]=B(i,j);basis.push_back(row);}
   auto full=old;full["basis"]=std::move(basis);full["tangent_size"]=tangent;full["correction_role"]="complete original cubic subspace; artificial polynomial bump restriction removed";manifolds.push_back(std::move(full));
  }
  after+=tangent;groups.push_back({{"block",bi},{"block_name",block.name},{"ambient",n},{"original_constraint_rows",rows.size()},{"tangent",tangent},{"membership_error",membership}});
 }
 require(groups.size()==provenance.at("groups").get<std::size_t>()&&before==provenance.at("tangent_scalars").get<int>()&&after>before,"incomplete original source-supported motion scope");
 Json report={{"schema","rinkfit.restored_full_original_s01_subspace.v1"},{"enabled",true},{"groups",groups},{"previous_tangent_scalars",before},{"tangent_scalars",after},{"initial_physical_values_unchanged",true},{"native_time_basis_and_original_constraints_preserved",true},{"source_UV_unchanged",true},{"temporal_penalty_added",false}};
 m.runtime_model_metadata["linear_control_manifolds"]=std::move(manifolds);m.runtime_model_metadata[proof_key]=report;return record_end_controls(report);
}
// Reuse the pinned producer's exact basis and initialization report. Membership
// roundoff changes after fitting and is not a different physical model.
inline void restore_joint_saved_s01_subspace_v1(JointEventModel& m,const eventio::Json& cfg,const eventio::Json& saved){
 using Json=eventio::Json;const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.value("restore_full_original_s01_subspace",false))return;
 restore_joint_full_s01_subspace_v1(m,cfg);
 auto require=[](bool b,const std::string&s){if(!b)throw std::runtime_error("saved original S01 motion space: "+s);};
 const std::string key="s01_full_original_subspace_v1";
 auto expected=m.runtime_model_metadata.at(key),recorded=saved.at(key);
 auto scrub=[](Json v){for(auto& group:v.at("groups"))group.erase("membership_error");return v;};
 require(scrub(expected)==scrub(recorded),"recorded original motion scope differs");
 const auto& live=m.control_manifolds();const auto& prior=saved.at("linear_control_manifolds");require(live.size()==prior.size(),"saved manifold count differs");
 for(std::size_t i=0;i<live.size();++i){auto a=live[i],b=prior[i];a.erase("basis");b.erase("basis");require(a==b,"saved original subspace identity differs");
  const int n=prior[i].at("ambient_size"),k=prior[i].at("tangent_size");const auto& basis=prior[i].at("basis");require(basis.size()==std::size_t(n),"saved basis rows");Eigen::MatrixXd B(n,k);for(int r=0;r<n;++r){require(basis[r].size()==std::size_t(k),"saved basis columns");for(int c=0;c<k;++c)B(r,c)=basis[r][c];}
  require(B.allFinite()&&(B.transpose()*B-Eigen::MatrixXd::Identity(k,k)).cwiseAbs().maxCoeff()<1e-10,"saved basis is not orthonormal");
  for(const auto& row:prior[i].at("constraint_matrix"))for(int c=0;c<k;++c){double value=0;for(int r=0;r<n;++r)value+=row[r].get<double>()*B(r,c);require(std::abs(value)<1e-10,"saved basis leaves original cubic subspace");}
 }
 m.runtime_model_metadata["linear_control_manifolds"]=prior;m.runtime_model_metadata[key]=saved.at(key);
}

// The original 3-scalar controls already exist in the registry. Undo only the
// artificial 32-control packaging; retain the genuine knot-removal manifold.
inline eventio::Json restore_joint_sparse_s01_storage_v1(JointEventModel& m,const eventio::Json& cfg,bool transfer_values){
 using Json=eventio::Json;const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.value("restore_original_s01_sparse_storage",false))return {{"enabled",false}};
 auto require=[](bool b,const std::string&s){if(!b)throw std::runtime_error("original sparse S01 storage: "+s);};
 require(stage.value("restore_full_original_s01_subspace",false)&&stage.value("release_original_s01_end_controls",false),"complete original motion space must already be released");
 const std::string key="s01_original_sparse_storage_v1";
 if(m.runtime_model_metadata.contains(key))return m.runtime_model_metadata.at(key);
 if(m.source_json.contains(key))return m.source_json.at(key);
 std::set<int> constrained;for(const auto& d:m.control_manifolds())constrained.insert(d.at("block").get<int>());
 auto& shot=m.shots.at(m.shot_map.at("s01"));std::set<int> unused,restored;std::size_t scalars=0;
 for(std::size_t i=0;i<shot.control_sources.size();++i){const int old=shot.control_sources[i],offset=shot.control_offsets[i],target=shot.control_blocks[i];
  if(constrained.count(old)||old==target)continue;
  const auto& source=m.blocks.at(old);auto& dest=m.blocks.at(target);
  require(source.name.rfind("shot/s01/control_group/",0)==0&&dest.name=="shot/s01/control/"+std::to_string(i)&&dest.values.size()==3&&dest.fixed_indices.empty(),"not an original unconstrained fine control");
  if(transfer_values)for(int j=0;j<3;++j){const double v=source.values.at(offset+j);require(v>=dest.lower[j]&&v<=dest.upper[j],"original value outside original scalar box");dest.values[j]=v;}
  unused.insert(old);restored.insert(target);scalars+=3;shot.control_sources[i]=target;shot.control_offsets[i]=0;
 }
 require(!unused.empty()&&restored.size()*3==scalars,"empty sparse conversion");
 auto shots=m.source_json.at("shots");Json coordinates=Json::array();for(std::size_t i=0;i<shot.control_sources.size();++i)coordinates.push_back({{"block",shot.control_sources[i]},{"offset",shot.control_offsets[i]}});shots["s01"]["control_coordinates"]=std::move(coordinates);
 Json report={{"schema","rinkfit.original_sparse_s01_storage.v1"},{"enabled",true},{"unused_group_blocks",unused},{"restored_original_control_blocks",restored},{"restored_scalars",scalars},{"entire_original_motion_space_preserved",true},{"initial_authoritative_control_values_bitwise_preserved",true},{"source_pixels_or_time_basis_changed",false}};
 m.runtime_model_metadata["shots"]=std::move(shots);m.runtime_model_metadata[key]=report;return report;
}

}
