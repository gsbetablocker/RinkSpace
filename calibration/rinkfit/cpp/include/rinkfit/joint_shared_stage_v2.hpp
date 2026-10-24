#pragma once
#include "rinkfit/joint_collinear_carrier_v1.hpp"
#include "rinkfit/joint_ceres_bounds.hpp"
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_shared_print_shape_transition_v1.hpp"
namespace rinkfit {
inline bool shared_scene_requested_v2(const JointALBuilder& b,const eventio::Json& cfg){
 if(!cfg.contains("runtime_shared_parameter_stage_v2"))return false;
 for(const auto& row:cfg.at("runtime_shared_parameter_stage_v2").at("blocks"))if(row.at("block_index").get<int>()==b.core.model.scene_block)return true;
 return false;
}

// Explicit runtime freedom, never a rewrite of historical measurements or seeds.
// The actual source contract is replayed before this function; its coverage is
// unchanged. Corner incidence proves support, not numerical identifiability.
inline eventio::Json install_joint_shared_stage_v2(JointALBuilder& b,
    ceres::Problem& problem, const eventio::Json& cfg) {
 using Json=eventio::Json;
 const char* key="runtime_shared_parameter_stage_v2";
 if(!cfg.contains(key))return {{"enabled",false}};
 auto require=[](bool ok,const std::string& message){
   if(!ok)throw std::runtime_error("explicit shared stage: "+message);
 };
 const auto& request=cfg.at(key);
 require(request.at("schema")=="rinkfit.explicit_shared_parameter_stage.v2" &&
         request.at("world_values_measured")==false &&
         request.at("source_packet_sha256")==b.core.packet.source_hash,
         "unbound or falsely measured parameter policy");
 require(cfg.contains("audit_expected_free_physical_blocks") &&
         cfg.value("temporary_runtime_fixed_nuisance_shots",std::vector<std::string>{}).empty() &&
         !cfg.contains("protected_S01_scalar_prefixes") && !cfg.contains("source_capture_init"),
         "shared stage must explicitly replace the old whole-S01 protection");
 const auto before=b.capture_state().token();
 struct MaterialSupport {std::size_t rows=0;std::set<std::string> shots;std::set<std::pair<std::string,std::int64_t>> exposures;bool unmeasured=false,chart_alignment=true;};
 std::map<int,MaterialSupport> shape_support,camera_support,ice_support,advertisement_support;
 std::set<int> coherent_groups;
 if(request.contains("confirmed_print_template_restore")){const auto&ref=request.at("confirmed_print_template_restore");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto source_template=eventio::read_json(ref.at("file").get<std::string>());for(const auto&g:source_template.at("groups"))coherent_groups.insert(g.at("block_index").get<int>());}
 if(request.contains("paired_advertisement_templates")){const auto&ref=request.at("paired_advertisement_templates");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto paired=eventio::read_json(ref.at("file").get<std::string>());for(const auto&g:paired.value("shared_chart_blocks",Json::array()))coherent_groups.insert(g.at("block_index").get<int>());}
 std::map<int,int> s01_dimensions;
 if(request.value("release_s01_trajectory",false)){
  require(cfg.value("source_uv_capture_objective",std::string())=="rinkfit.sourceUV_observed_row_finite_initialization.v1","S01 trajectory release is an explicit provisional stage");
  const auto& provenance=b.core.model.source_json.at("s01_affine_recovery_corrections");
  require(provenance.at("original_exact_subspace_preserved")==true&&provenance.at("baseline_function_resampled")==false,"S01 baseline/subspace provenance");
  for(int i:b.core.model.shots.at(b.core.model.shot_map.at("s01")).control_sources)if(b.core.model.blocks.at(i).name.rfind("shot/s01/control_group/",0)==0)s01_dimensions[i]=int(b.core.model.blocks.at(i).values.size());
  for(const auto& m:b.core.model.control_manifolds()){const int i=m.at("block");if(s01_dimensions.count(i))s01_dimensions[i]=m.at("tangent_size");}
  if(!request.value("restore_original_s01_sparse_storage",false))require(s01_dimensions.size()==provenance.at("groups").get<std::size_t>(),"complete existing S01 trajectory scope");
  else require(b.core.model.runtime_model_metadata.contains("s01_original_sparse_storage_v1")||b.core.model.source_json.contains("s01_original_sparse_storage_v1"),"original sparse motion storage was not restored");
  if(request.value("release_original_s01_end_controls",false))for(int i:b.core.model.shots.at(b.core.model.shot_map.at("s01")).control_sources){const auto& b0=b.core.model.blocks.at(i);if(b0.name.rfind("shot/s01/control/",0)==0){require(b0.values.size()==3,"original end-control shape");s01_dimensions[i]=int(b0.values.size()-b0.fixed_indices.size());}}
 }
 for(const auto& material:b.core.model.materials)if(material.label.rfind("ice.",0)==0)for(const auto& path:material.paths)if(!path.physical_point&&path.geometry_block>=0&&(path.kind==0||path.kind==1||path.kind==4||path.kind==5)){
  const auto& p=path.provenance;auto& s=ice_support[path.geometry_block];s.unmeasured=s.unmeasured||p.find("initializer")!=std::string::npos||p.find("initialization")!=std::string::npos||p.find("legacy_static_seed_only_not_world_truth")!=std::string::npos||p.find("not a physical measurement")!=std::string::npos;
 }
 // Provenance belongs to the shared shape, including aliases of its contours.
 for(const auto& material:b.core.model.materials)for(const auto& path:material.paths){
   if((path.kind!=2&&path.kind!=6)||path.geometry_block<0)continue;
   const auto& p=path.provenance;auto& s=shape_support[path.geometry_block];
   s.unmeasured=s.unmeasured||p.find("initializer")!=std::string::npos||p.find("initialization")!=std::string::npos||p.find("not_world_truth")!=std::string::npos||p.find("not_new_world_measurements")!=std::string::npos;
 }
 for(const auto& row:b.core.packet.rows){
   if(!row.active)continue;
   const auto& path=b.core.model.materials.at(row.material).paths.at(row.path);
   if(path.kind==6&&!path.vertex_blocks.empty()&&path.advertisement_block>=0){auto&s=advertisement_support[path.advertisement_block];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});}
   if(path.kind==2&&path.advertisement_block>=0){bool known=path.geometry_block<0;if(path.geometry_block>=0){const auto&shape=b.core.model.blocks.at(path.geometry_block);known=shape.name.rfind("print/",0)==0&&(shape.constant||coherent_groups.count(path.geometry_block));}if(known){auto&s=advertisement_support[path.advertisement_block];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});}}
   if(b.core.model.materials.at(row.material).label.rfind("ice.",0)==0&&!path.physical_point&&ice_support.count(path.geometry_block)){auto& s=ice_support[path.geometry_block];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});}
   if((path.kind==2||path.kind==6)&&path.geometry_block>=0){
     const auto& shape=b.core.model.blocks.at(path.geometry_block);
     if((shape.name.rfind("print/",0)==0&&shape.values.size()==6)||(path.kind==6&&shape.name.rfind("print_component/polyline/",0)==0)){
       auto& s=shape_support[path.geometry_block];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});
       const auto& p=path.provenance;
       s.chart_alignment=s.chart_alignment&&p.find("local_points_are_chart_coordinates_not_new_world_measurements")!=std::string::npos;
     }
   }
   const auto& shot=b.core.model.shots.at(b.core.model.shot_map.at(row.shot_id));
   const auto& rig=b.core.model.rigs.at(b.core.model.rig_map.at(shot.rig_id));
   for(int index:{rig.optics_block,rig.head_block}){auto& s=camera_support[index];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});}
   if(!s01_dimensions.empty()&&shot.id=="s01")for(int index:row.layout(b.core.model).block_ids)if(s01_dimensions.count(index)){auto& s=camera_support[index];++s.rows;s.shots.insert(row.shot_id);s.exposures.insert({row.shot_id,row.pts});}
 }
 std::set<int> requested;Json released=Json::array();
 for(const auto& rec:request.at("blocks")){
   const int index=rec.at("block_index");
   require(index>=0 && std::size_t(index)<b.core.model.blocks.size() && requested.insert(index).second,
           "invalid or duplicate block index");
   auto& block=b.core.model.blocks.at(index);
   require(block.name==rec.at("block_name"),"block name/index mismatch");
   const bool ad=block.name.rfind("ad/",0)==0;const bool paired_ad=block.name.rfind("ad/pair/",0)==0;
   const bool scene=index==b.core.model.scene_block;
   const bool head=block.name.rfind("rig/",0)==0 && block.name.find("/head15")!=std::string::npos;
   const bool ice=block.name.rfind("ice/",0)==0;
   const bool collinear_carrier=rec.value("release_collinear_ice_carrier",false);
   require(!collinear_carrier||(ice&&block.name.rfind("ice/line/",0)==0&&block.values.size()==6&&JointCollinearCarrierV1::collinear(block.values.data())),"collinear carrier release requires the existing straight six-coordinate geometry");
   const bool print_affine=block.name.rfind("print/",0)==0&&block.values.size()==6;
   const bool restored_print=rec.value("restore_original_print_vertices",false);
   require(!restored_print||(request.contains("restore_original_print_vertices")&&block.name.rfind("print_component/polyline/",0)==0),"restored shape is outside original polyline scope");
   const bool optics=block.name.rfind("rig/",0)==0&&block.name.find("/optics12")!=std::string::npos;
   const bool trajectory=rec.value("release_initialization_s01_trajectory",false);
   require(!trajectory||s01_dimensions.count(index),"S01 release is outside the existing affine trajectory");
   require(ad||scene||head||ice||print_affine||optics||trajectory||restored_print,"unsupported explicit shared geometry block");
   const bool release_alignment=rec.value("release_unmeasured_chart_alignment",false);
   const bool release_camera=rec.value("release_initialization_camera_indices",false);
   const auto& original=b.core.model.source_json;
   require(!release_camera||((head||optics)&&original.contains("camera_bootstrap_rebase_v24")&&
       original.at("camera_bootstrap_rebase_v24").value("all_shared_parameters_must_be_released_after_initialization",false)),
       "camera freedom requires explicit temporary-initialization provenance");
   require(!release_alignment||(print_affine&&block.name.rfind("print/group/",0)==0&&shape_support[index].chart_alignment),"only unmeasured chart alignment may replace inherited affine zeros");
   for(const auto& name:cfg.value("temporary_runtime_fixed_physical_blocks",std::vector<std::string>{}))
     require(name!=block.name,"release contradicts exact freeze");
   for(const auto& prefix:cfg.value("temporary_runtime_fixed_physical_prefixes",std::vector<std::string>{}))
     require(block.name.rfind(prefix,0)!=0,"release contradicts prefix freeze");
   auto free=rec.at("free_ambient_indices").get<std::vector<int>>();
   std::sort(free.begin(),free.end());
   require(!free.empty() && std::adjacent_find(free.begin(),free.end())==free.end(),"empty or duplicate free scalar list");
   for(int i:free)require(i>=0 && std::size_t(i)<block.values.size() &&
       (std::find(block.fixed_indices.begin(),block.fixed_indices.end(),i)==block.fixed_indices.end()||(release_alignment&&(i==1||i==3))||release_camera),
       "cannot release an original structural/gauge fixed scalar: "+block.name+"["+std::to_string(i)+"]");
   std::set<std::string> shots;std::set<std::pair<std::string,std::int64_t>> exposures;
   std::set<std::pair<double,double>> corners;std::size_t rows=0;
   if(print_affine||optics||head||trajectory||restored_print){
     const auto& s=(print_affine||restored_print)?shape_support[index]:camera_support[index];
     require(restored_print||s.rows>0,"no active source support for "+block.name);
     require(!(print_affine||restored_print)||s.unmeasured,"print shape has no explicit unmeasured initializer provenance: "+block.name);
     rows=s.rows;shots=s.shots;exposures=s.exposures;
   }else if(ice&&rec.value("release_unmeasured_ice_geometry",false)){
     require(request.value("release_unmeasured_ice_geometry",false),"unrecorded ice geometry release");const auto& s=ice_support[index];require(s.rows>0&&s.unmeasured,"ice primitive has no admitted real material support or unmeasured initialization provenance: "+block.name);require(!(block.values.size()==6&&block.name.rfind("ice/line/",0)==0)||collinear_carrier,"shared line caps require their original straight carrier");rows=s.rows;shots=s.shots;exposures=s.exposures;
   }else if(ice){
     require(rec.contains("source_support"),"ice release requires reviewed native pixels");
     const auto&ref=rec.at("source_support");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
     const auto support=eventio::read_json(ref.at("file").get<std::string>());
     require(support.at("schema")=="rinkfit.reviewed_ice_primitive_support.v1"&&support.at("source_packet_sha256")==b.core.packet.source_hash,"ice support source mismatch");
     std::set<int>radii;std::set<std::size_t>seen;
     for(const auto&v:support.at("rows")){
       if(v.at("block_index").get<int>()!=index)continue;
       const auto i=v.at("packet_row").get<std::size_t>();require(i<b.core.packet.rows.size()&&seen.insert(i).second,"invalid ice support row");
       const auto&r=b.core.packet.rows[i];const auto&path=b.core.model.materials.at(r.material).paths.at(r.path);
       require(r.active&&r.kind==0&&path.kind==1&&path.geometry_block==index&&r.source_row_id==v.at("source_row_id")&&r.pts==v.at("native_pts")&&r.uv==v.at("uv").get<Vec2<double>>()&&v.at("visible_static_ice_material")==true,"reviewed ice support differs from actual source");
       require(path.provenance.find("initializer")!=std::string::npos||path.provenance.find("legacy_static_seed_only_not_world_truth")!=std::string::npos,"ice geometry lacks explicit unmeasured initializer provenance");
       ++rows;shots.insert(r.shot_id);exposures.insert({r.shot_id,r.pts});radii.insert(path.radius_index);
     }
     require(rows>0,"no reviewed active ice support");
     for(int i:free)require(i<2||radii.count(i),"released circle radius has no reviewed source boundary");
   }else if(ad && (rec.value("known_local_board_material_support",false)||rec.value("coherent_print_material_support",false))){
     if(rec.value("coherent_print_material_support",false))require(request.contains("confirmed_print_template_restore")||request.contains("static_shared_print_vertex_stage_v1"),"coherent print placement requires source-bound static template support");
     const auto&support=advertisement_support[index];rows=support.rows;shots=support.shots;exposures=support.exposures;
     require(rows>0,"no active known-local board material support: "+block.name);
   }else{
   for(const auto& row:b.core.packet.rows){
     if(!row.active||row.kind!=1)continue;
     const auto& path=b.core.model.materials.at(row.material).paths.at(row.path);
     if(path.kind!=2||!path.physical_point||path.geometry_block>=0||path.points.size()!=1)continue;
     const auto uv=path.points.front();
     if(!((uv[0]==0||uv[0]==1)&&(uv[1]==0||uv[1]==1)))continue;
     const auto layout=row.layout(b.core.model);
     if(std::find(layout.block_ids.begin(),layout.block_ids.end(),index)==layout.block_ids.end())continue;
     ++rows;shots.insert(row.shot_id);exposures.insert({row.shot_id,row.pts});corners.insert({uv[0],uv[1]});
   }
   require(rows>0&&exposures.size()>1&&corners.size()>=3&&shots.size()>=2,
           "insufficient actual fixed-local-corner support across shots: "+block.name);
   }
   auto* x=block.values.data();require(problem.HasParameterBlock(x),"block absent from Ceres graph");
   std::vector<int> fixed;for(int i=0;i<int(block.values.size());++i)
     if(!std::binary_search(free.begin(),free.end(),i))fixed.push_back(i);
   install_joint_ceres_parameter_bounds(problem,block);
   if(trajectory){
    if(block.name.rfind("shot/s01/control/",0)==0){require(request.value("release_original_s01_end_controls",false),"unrecorded original endpoint release");problem.SetManifold(x,nullptr);if(!fixed.empty())problem.SetManifold(x,new ceres::SubsetManifold(int(block.values.size()),fixed));}
    const auto* manifold=problem.GetManifold(x);const int tangent=s01_dimensions.at(index);
    require(problem.ParameterBlockTangentSize(x)==tangent&&rec.at("expected_tangent_size")==tangent,"S01 motion tangent differs from requested original subspace");
    std::vector<int> actual;if(manifold){std::vector<double> J(block.values.size()*tangent);require(manifold->PlusJacobian(x,J.data()),"S01 affine Jacobian unavailable");for(int row=0;row<int(block.values.size());++row)for(int col=0;col<tangent;++col)if(J[row*tangent+col]!=0.){actual.push_back(row);break;}}else{require(request.value("restore_full_original_s01_subspace",false)&&tangent==int(block.values.size()),"unrecorded Euclidean S01 release");for(int row=0;row<tangent;++row)actual.push_back(row);}require(actual==free,"S01 free rows differ from actual original subspace");
   }else if(paired_ad){require(request.contains("paired_advertisement_templates")&&block.values.size()==8&&fixed.empty()&&free.size()==8,"paired advertisement must retain both independent locations and common dimensions");require(problem.GetManifold(x)&&problem.ParameterBlockTangentSize(x)==rec.at("expected_tangent_size").get<int>()&&rec.at("expected_tangent_size").get<int>()>=5,"paired advertisement shared shape manifold lost");
   }else if(collinear_carrier){require(free.size()==6&&fixed.empty(),"collinear carrier cannot inherit pinned coordinates");problem.SetManifold(x,new JointCollinearCarrierV1({},&block));for(int j=0;j<6;++j){problem.SetParameterLowerBound(x,j,-std::numeric_limits<double>::max());problem.SetParameterUpperBound(x,j,std::numeric_limits<double>::max());}problem.AddResidualBlock(new JointCollinearCarrierBoxV1(block),nullptr,x);}
   else{problem.SetManifold(x,nullptr);if(!fixed.empty())problem.SetManifold(x,new ceres::SubsetManifold(int(block.values.size()),fixed));}
   problem.SetParameterBlockVariable(x);
   require(problem.ParameterBlockTangentSize(x)==(trajectory?s01_dimensions.at(index):collinear_carrier?5:paired_ad?rec.at("expected_tangent_size").get<int>():int(free.size())),"actual tangent differs from policy");
   released.push_back({{"block_index",index},{"block_name",block.name},{"free_ambient_indices",free},
     {"original_straight_carrier_preserved",collinear_carrier},{"existing_S01_motion_manifold_preserved",trajectory},{"actual_tangent",problem.ParameterBlockTangentSize(x)},{"actual_corner_rows",rows},
     {"actual_corner_shots",shots},{"native_exposures",exposures.size()},
     {"serialized_constant_preserved",block.constant},{"released_initialization_camera_indices",release_camera?block.fixed_indices:std::vector<int>{}}});
 }
 const auto vertex_stage=install_joint_shared_print_vertex_stage_v1(b,problem,cfg);
 require(!requested.empty()&&b.capture_state().token()==before,"empty stage or mutated primal");
 return {{"schema","rinkfit.actual_shared_parameter_stage.v2"},{"enabled",true},{"released",released},{"static_print_vertices",vertex_stage},
   {"source_membership_and_UV_unchanged",true},{"historical_model_flags_unchanged",true},
   {"world_values_measured",false},{"support_is_not_rank_certificate",true},
   {"S01_in_objective",true},{"full_scalar_whitelist_check_follows",true}};
}
}
