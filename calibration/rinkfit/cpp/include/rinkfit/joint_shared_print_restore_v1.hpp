#pragma once
#include "rinkfit/joint_model.hpp"
#include <cmath>
namespace rinkfit {
// Undo the documented kind6 -> fixed-template conversion after the complete
// primal has been restored. The first transition carries today's unfolded
// curve exactly; subsequent restarts restore metadata only, retaining the
// independently fitted shared vertices from their checkpoint.
inline eventio::Json restore_joint_original_print_shapes_v1(
    JointEventModel& m,const eventio::Json& cfg,bool transform_values) {
 using Json=eventio::Json;
 const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.contains("restore_original_print_vertices"))return {{"enabled",false}};
 auto require=[](bool ok,const std::string& why){if(!ok)throw std::runtime_error("restore original print shapes: "+why);};
 const auto ref=stage.at("restore_original_print_vertices");
 eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
 const auto overlay=eventio::read_json(ref.at("file").get<std::string>());
 require(overlay.at("schema")=="rinkfit.prototype.shared_print_group_chart_reencoding.v1","unsupported converter provenance");
 std::set<int> restored;std::set<std::pair<int,int>> seen;
 std::size_t vertices=0,changed=0;
 for(const auto& u:overlay.at("material_path_updates")){
  const int mi=u.at("material_index"),pi=u.at("path_index"),bi=u.at("original_geometry_block");
  require(seen.insert({mi,pi}).second&&restored.insert(bi).second,"original shape must have its recorded unique owner");
  auto& path=m.materials.at(mi).paths.at(pi);const auto& original=u.at("expected_path");
  auto& shape=m.blocks.at(bi);const int n=u.at("original_vertices");
  require(original.at("kind")==6&&original.at("geometry_block")==bi&&shape.name.rfind("print_component/polyline/",0)==0&&shape.fixed_indices.empty(),"not an original free shared polyline");
  require(original.at("provenance").get<std::string>().find("numerical initialization")!=std::string::npos,"shape was not an unmeasured initializer");
  const auto provenance=Json{{"schema","rinkfit.restored_original_shared_print_shape.v1"},{"converter",ref},{"original_path",original},{"source_UV_ID_PTS_and_finite_station_unchanged",true},{"one_shape_shared_across_exposures",true}}.dump();
  if(path.kind==6){require(path.geometry_block==bi&&path.provenance==provenance,"different restoration already present");continue;}
  require(path.kind==2&&m.source_json.at("materials").at(mi).at("paths").at(pi)==u.at("path"),"current template no longer matches the recorded converter");
  require(path.geometry_block==u.at("path").at("geometry_block")&&m.blocks.at(path.geometry_block).name.rfind("print/group/",0)==0,"shared template group changed");
  require(path.points.size()==std::size_t(n+(path.closed?1:0))&&original.value("closed",false)==path.closed&&path.id==original.at("id")&&path.variant_id==original.at("variant_id"),"finite curve ordering/identity changed");
  const auto indices=original.value("geometry_indices",std::vector<int>{});const int offset=original.value("geometry_offset",0);
  require((indices.empty()?int(shape.values.size())-offset:int(indices.size()))==2*n,"original vertex coordinate count changed");
  if(path.closed)require(path.points.front()==path.points.back(),"explicit closing vertex changed");
  if(transform_values){
   const auto& a=m.blocks.at(path.advertisement_block).values;const auto& g=m.blocks.at(path.geometry_block).values;
   require(a.size()==4&&g.size()==6&&a[1]>0&&a[3]>a[2],"invalid current unfolded chart");
   for(int k=0;k<n;++k){const auto p=path.points[k];const double xy[2]={a[0]+a[1]*(g[0]*p[0]+g[1]*p[1]+g[2]),a[2]+(a[3]-a[2])*(g[3]*p[0]+g[4]*p[1]+g[5])};
    for(int j=0;j<2;++j){const int index=indices.empty()?offset+2*k+j:indices.at(2*k+j);const double v=xy[j];require(std::isfinite(v)&&v>=shape.lower.at(index)&&v<=shape.upper.at(index),"current curve leaves original scalar domain");shape.values.at(index)=v;}
   }
  }
  path.kind=6;path.geometry_block=bi;path.advertisement_block=-1;path.points.clear();path.provenance=provenance;
  m.path_geometry_offsets[{mi,pi}]=offset;m.path_geometry_indices[{mi,pi}]=indices;
  vertices+=n;++changed;
 }
 require(restored.size()==overlay.at("orphaned_vertex_block_updates").size(),"incomplete restoration");
 return {{"enabled",true},{"converter",ref},{"shared_shapes",restored.size()},{"changed_paths",changed},{"vertices",vertices},{"current_unfolded_shape_carried",transform_values},{"source_UV_ID_PTS_path_and_station_unchanged",true},{"shape_is_not_a_world_measurement",true}};
}

// Restore the already reviewed printed contours as a coherent shared template.
// The advertisement's placement and its shared chart remain fitted; individual
// ink vertices no longer become independent world points.
inline eventio::Json restore_joint_confirmed_print_templates_v1(
    JointEventModel& m,const eventio::Json& cfg,bool initialize_values=true) {
 using Json=eventio::Json;
 const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.contains("confirmed_print_template_restore"))return {{"enabled",false}};
 const auto ref=stage.at("confirmed_print_template_restore");
 eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
 const auto patch=eventio::read_json(ref.at("file").get<std::string>());
 auto require=[](bool yes,const std::string& why){if(!yes)throw std::runtime_error("confirmed print template: "+why);};
 require(patch.at("schema")=="rinkfit.event.confirmed_advertisement_shape_restore.v1","schema mismatch");
 if(m.runtime_model_metadata.contains("paired_advertisement_templates")||m.source_json.contains("paired_advertisement_templates")){
  require(m.runtime_model_metadata.value("confirmed_advertisement_shapes",m.source_json.value("confirmed_advertisement_shapes",Json::object()))==ref,"paired model requires its recorded base template");
  // The paired geometry is already serialized, but legacy source flags may
  // still describe its initialization. Replay the recorded chart freedom
  // without re-seeding values or replacing any paired material path.
  for(const auto& group:patch.at("groups")){auto& block=m.blocks.at(group.at("block_index").get<int>());require(block.name==group.at("block_name")&&block.values.size()==6,"shared template chart changed");block.constant=false;block.fixed_indices.clear();}
  return {{"enabled",true},{"already_encoded_in_paired_template",true},{"recorded_chart_freedom_restored",true}};
 }
 const bool already=m.runtime_model_metadata.value("confirmed_advertisement_shapes",m.source_json.value("confirmed_advertisement_shapes",Json::object()))==ref;
 auto restored_materials=m.runtime_model_metadata.value("materials",m.source_json.at("materials"));
 const auto previous_refinements=m.runtime_model_metadata.value("printed_material_path_refinements",m.source_json.value("printed_material_path_refinements",Json::object()));
 std::set<std::pair<int,int>> seen;std::size_t changed=0;
 for(const auto& item:patch.at("material_path_updates")){
  const int mi=item.at("material_index"),pi=item.at("path_index");
  require(seen.insert({mi,pi}).second,"duplicate material path");
  require(m.materials.at(mi).label==item.at("label"),"material identity changed");
  auto& path=m.materials.at(mi).paths.at(pi);const auto& q=item.at("path");
  require(q.at("kind")==2&&q.at("physical_point")==false,"only actual printed curves may be restored");
  bool same_identity=path.id==q.at("id")&&path.variant_id==q.at("variant_id")&&path.closed==q.at("closed");
  if(!same_identity&&pi==0&&previous_refinements.contains(std::to_string(mi))){const auto& old=previous_refinements.at(std::to_string(mi));if(old.contains("original_identity")){const auto& identity=old.at("original_identity");same_identity=old.at("label")==item.at("label")&&identity.at("id")==q.at("id")&&identity.at("variant_id")==q.at("variant_id")&&identity.at("closed")==q.at("closed");}}
  require(same_identity,"finite contour identity or closure changed without recorded native topology refinement");
  const int old=item.at("original_geometry_block"),gi=q.at("geometry_block"),ai=q.at("advertisement_block");
  require((path.kind==6&&path.geometry_block==old)||(path.kind==2&&(path.geometry_block==old||path.geometry_block==gi)),"unexpected predecessor path");
  require((m.blocks.at(old).name.rfind("print_component/polyline/",0)==0||m.blocks.at(old).name.rfind("print/",0)==0)&&m.blocks.at(gi).name.rfind("print/",0)==0,"unrelated shape block");
  require(m.blocks.at(gi).values.size()==6&&m.blocks.at(ai).values.size()==4,"invalid shared chart");
  changed+=path.kind!=2;
  path.kind=2;path.geometry_block=gi;path.advertisement_block=ai;path.points.clear();
  for(const auto& xy:q.at("points")){const double x=xy.at(0),y=xy.at(1);require(std::isfinite(x)&&std::isfinite(y),"nonfinite template coordinate");path.points.push_back({x,y});}
  require(path.points.size()>1,"invalid finite contour closure");
  path.provenance=q.at("provenance").get<std::string>();
  m.path_geometry_offsets[{mi,pi}]=0;m.path_geometry_indices[{mi,pi}]={};
  restored_materials[mi]["paths"][pi]=q;
  m.blocks.at(old).constant=true;
 }
 // A coarse, blurred connected component can represent several distinct ink
 // boundaries. Preserve its source identity while replacing that geometry by
 // the actual finite contours in the same printed entity and fitted chart.
 Json refinements=Json::object();
 for(const auto& item:patch.value("material_curve_sets",Json::array())){
  const int mi=item.at("material_index");auto& material=m.materials.at(mi);require(material.label==item.at("label")&&material.label.find(".print.")!=std::string::npos,"print refinement identity changed");
  require(item.at("original_path_count")==1&&item.contains("source_evidence"),"print refinement lacks original native topology");const auto prototype=material.paths.front();require(prototype.kind==2&&!prototype.physical_point,"print refinement must preserve a finite ink entity");material.paths.clear();restored_materials[mi]["paths"]=Json::array();Json children=Json::array();int pi=0;
  for(const auto& q:item.at("paths")){auto path=prototype;require(q.at("kind")==2&&q.at("physical_point")==false&&q.at("geometry_block")==prototype.geometry_block&&q.at("advertisement_block")==prototype.advertisement_block,"ink refinement left its shared advertisement");path.id=q.at("id");path.variant_id=q.at("variant_id");path.closed=q.at("closed");path.provenance=q.at("provenance");path.points.clear();for(const auto& xy:q.at("points")){double x=xy.at(0),y=xy.at(1);require(std::isfinite(x)&&std::isfinite(y),"nonfinite refined ink");path.points.push_back({x,y});}require(path.points.size()>1,"empty refined ink");material.paths.push_back(std::move(path));restored_materials[mi]["paths"].push_back(q);m.path_geometry_offsets[{mi,pi}]=0;m.path_geometry_indices[{mi,pi}]={};m.path_station_intervals[{mi,pi}]={{0.,1.}};children.push_back(pi++);}
  require(!children.empty(),"print refinement cannot erase an entity");refinements[std::to_string(mi)]={{"label",material.label},{"original_path_count",1},{"original_identity",{{"id",prototype.id},{"variant_id",prototype.variant_id},{"closed",prototype.closed}}},{"children",children},{"source_evidence",item.at("source_evidence")}};
 }
 // Idempotent restore retains the pre-refinement source identity. The current
 // first child is not the original connected component.
 for(auto& entry:refinements.items())if(previous_refinements.contains(entry.key())){const auto& prior=previous_refinements.at(entry.key());require(prior.at("label")==entry.value().at("label")&&prior.at("original_path_count")==1&&prior.contains("original_identity"),"previous native topology identity differs");entry.value()["original_identity"]=prior.at("original_identity");}
 if(patch.contains("material_curve_sets"))m.runtime_model_metadata["printed_material_path_refinements"]=std::move(refinements);
 // Additional observed ink contours share the same fitted whole-print chart.
 // They add no measurements, parameters, or connections between distinct strokes.
 for(const auto& item:patch.value("material_additions",Json::array())){
  const int mi=item.at("material_index"),owner=item.at("owner_material_index");
  require(owner>=0&&owner<int(m.materials.size()),"missing original printed owner");
  require(mi>=0&&mi<=int(m.materials.size()),"noncontiguous printed material addition");
  const std::string label=item.at("label");
  if(mi==int(m.materials.size())){auto material=m.materials.at(owner);material.label=label;material.paths.clear();m.materials.push_back(std::move(material));restored_materials.push_back(Json{{"label",label},{"paths",Json::array()}});}
  require(m.materials.at(mi).label==label,"additional print identity changed");
  auto& material=m.materials.at(mi);material.paths.clear();restored_materials[mi]["paths"]=Json::array();
  int pi=0;for(const auto& q:item.at("paths")){
   require(q.at("kind")==2&&q.at("physical_point")==false,"additional ink must be a finite printed contour");
   auto path=m.materials.at(owner).paths.front();path.kind=2;path.id=q.at("id");path.variant_id=q.at("variant_id");path.closed=q.at("closed");path.physical_point=false;path.geometry_block=q.at("geometry_block");path.advertisement_block=q.at("advertisement_block");path.provenance=q.at("provenance");path.points.clear();
   require(path.geometry_block==m.materials.at(owner).paths.front().geometry_block&&path.advertisement_block==m.materials.at(owner).paths.front().advertisement_block,"additional contour left its printed chart");
   for(const auto& xy:q.at("points")){const double x=xy.at(0),y=xy.at(1);require(std::isfinite(x)&&std::isfinite(y),"nonfinite added ink coordinate");path.points.push_back({x,y});}
   require(path.points.size()>1,"empty added ink contour");material.paths.push_back(std::move(path));restored_materials[mi]["paths"].push_back(q);m.path_geometry_offsets[{mi,pi}]=0;m.path_geometry_indices[{mi,pi}]={};m.path_station_intervals[{mi,pi}]={{0.,1.}};++pi;
  }
 }
 // A printed boundary and its border observation are the same material.
 // Bind their finite geometry to the common print chart instead of allowing
 // an independent border to drift away from the lettering and colored bars.
 for(const auto& item:patch.value("material_surface_alias_updates",Json::array())){
  const int mi=item.at("material_index"),pi=item.at("path_index"),owner=item.at("owner_material_index");
  auto& material=m.materials.at(mi);auto& path=material.paths.at(pi);const auto& q=item.at("path");const auto& parent=m.materials.at(owner).paths.front();
  require(material.label==item.at("label")&&material.label.rfind("Ad",0)==0&&material.label.find(".print.")==std::string::npos,"surface alias is not an advertisement border");
  require(item.contains("source_evidence"),"surface alias lacks native material evidence");for(const auto& source:item.at("source_evidence"))eventio::verify(source.at("file").get<std::string>(),source.at("sha256"));
  require(path.kind==2&&!path.physical_point&&q.at("kind")==2&&q.at("physical_point")==false&&path.id==q.at("id")&&path.variant_id==q.at("variant_id")&&path.closed==q.at("closed"),"surface alias changed a source identity");
  const int gi=q.at("geometry_block"),ai=q.at("advertisement_block");
  require(ai==path.advertisement_block&&ai==parent.advertisement_block&&gi==parent.geometry_block&&(path.geometry_block<0||path.geometry_block==gi),"surface alias left its existing advertisement");
  require(m.blocks.at(gi).values.size()==6&&m.blocks.at(gi).name.rfind("print/",0)==0,"surface alias lacks its shared print chart");
  path.geometry_block=gi;path.points.clear();for(const auto& xy:q.at("points")){double x=xy.at(0),y=xy.at(1);require(std::isfinite(x)&&std::isfinite(y),"invalid aliased border coordinate");path.points.push_back({x,y});}
  require(path.points.size()>1,"empty physical border");path.provenance=q.at("provenance");restored_materials[mi]["paths"][pi]=q;m.path_geometry_offsets[{mi,pi}]=0;m.path_geometry_indices[{mi,pi}]={};
 }
 for(const auto& group:patch.at("groups")){
  const int gi=group.at("block_index");auto& block=m.blocks.at(gi);
  require(block.name==group.at("block_name")&&block.values.size()==6,"shared template chart changed");
  if(initialize_values&&!already){const auto values=group.at("initial_values").get<std::vector<double>>();require(values.size()==block.values.size(),"shared chart seed size");for(std::size_t k=0;k<values.size();++k)require(std::isfinite(values[k])&&values[k]>=block.lower[k]&&values[k]<=block.upper[k],"shared chart seed outside its domain");block.values=values;}
  block.constant=false;block.fixed_indices.clear();
 }
 m.runtime_model_metadata["materials"]=std::move(restored_materials);
 m.runtime_model_metadata["confirmed_advertisement_shapes"]=ref;
 return {{"enabled",true},{"restoration",ref},{"material_paths",seen.size()},{"changed_paths",changed},{"individual_ink_vertex_motion_removed",true},{"advertisement_and_camera_placement_remain_fitted",true},{"source_UV_ID_PTS_unchanged",true},{"fit_quality_accepted",false}};
}
}
