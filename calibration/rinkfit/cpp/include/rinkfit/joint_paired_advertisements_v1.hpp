#pragma once
#include "rinkfit/joint_model.hpp"
namespace rinkfit {
// Two physical installations share width and height in a single parameter
// block. Its six-dimensional tangent leaves both board positions independent.
inline eventio::Json restore_joint_paired_advertisements_v1(JointEventModel& m,const eventio::Json& cfg,bool initialize_values=true){
 using Json=eventio::Json;const auto stage=cfg.value("runtime_shared_parameter_stage_v2",Json::object());
 if(!stage.contains("paired_advertisement_templates"))return {{"enabled",false}};
 const auto ref=stage.at("paired_advertisement_templates");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto patch=eventio::read_json(ref.at("file").get<std::string>());
 auto require=[](bool x,const std::string& why){if(!x)throw std::runtime_error("paired advertisement template: "+why);};
 require(patch.at("schema")=="rinkfit.confirmed_paired_advertisement_templates.v1","schema");
 const auto authority=patch.at("user_requirement");eventio::verify(authority.at("file").get<std::string>(),authority.at("sha256"));const auto rule=eventio::read_json(authority.at("file").get<std::string>()).at("constraint");require(rule.at("same_printed_template")&&rule.at("shared_printed_dimensions")&&rule.at("physical_locations_independent")&&rule.at("no_added_equality_observation_or_reweighted_source"),"missing actual paired-print authority");
 for(const auto& source:patch.at("source_evidence"))eventio::verify(source.at("file").get<std::string>(),source.at("sha256"));
 auto materials=m.runtime_model_metadata.value("materials",m.source_json.at("materials"));auto manifolds=m.runtime_model_metadata.value("linear_control_manifolds",m.source_json.at("linear_control_manifolds"));
 std::set<int> exterior_supported_charts;
 if(patch.contains("source_chart_freedom_v1")){
  const auto& r=patch.at("source_chart_freedom_v1");eventio::verify(r.at("file").get<std::string>(),r.at("sha256"));const auto proof=eventio::read_json(r.at("file").get<std::string>());
  require(proof.at("schema")=="rinkfit.exterior_supported_print_chart_freedom.v1"&&proof.at("source_UV_ID_PTS_admission_weights_changed")==false&&proof.at("existing_physical_values_and_bounds_changed")==false&&proof.at("new_parameter_blocks")==0&&proof.at("maximum_initial_unit_curve_change").get<double>()<1e-12,"invalid exterior-supported print chart transition");
  for(const char* key:{"parent_template","complete_saved_model","packet"}){const auto& source=proof.at(key);eventio::verify(source.at("file").get<std::string>(),source.at("sha256"));}
  for(const auto& item:proof.at("pairs")){require(item.at("active_exterior_rows").get<std::size_t>()>0&&item.at("active_print_rows").get<std::size_t>()>0,"print chart has no independent exterior support");exterior_supported_charts.insert(item.at("block").get<int>());}
 }
 std::set<int> shared_charts;for(const auto& item:patch.value("shared_chart_blocks",Json::array())){const int gi=item.at("block_index");const auto& q=item.at("block");const int freedom=item.at("physical_affine_freedoms_retained");const bool full=freedom==6&&exterior_supported_charts.count(gi)&&q.at("fixed_indices").empty();const bool gauge=freedom==2&&q.at("fixed_indices")==Json::array({0,2,4,5});require(gi>=0&&gi<int(m.blocks.size())&&shared_charts.insert(gi).second&&m.blocks[gi].name==q.at("name")&&m.blocks[gi].values.size()==6&&(full||gauge),"invalid paired print coordinate chart");m.blocks[gi].constant=false;m.blocks[gi].fixed_indices=full?std::vector<int>{}:std::vector<int>{0,2,4,5};}
 std::set<int> pair_blocks;for(const auto& item:patch.at("pairs")){
  const int bi=item.at("block_index");const auto& q=item.at("block");require(bi>=0&&bi<=int(m.blocks.size())&&pair_blocks.insert(bi).second,"pair block order/identity");
  require(q.at("values").size()==8&&q.at("fixed_indices").empty()&&!q.at("constant").get<bool>(),"pair must retain its physical freedoms");
  if(bi==int(m.blocks.size())){EventParameterBlock b;b.name=q.at("name");b.values=q.at("values").get<std::vector<double>>();b.lower=q.at("lower").get<std::vector<double>>();b.upper=q.at("upper").get<std::vector<double>>();b.scales=q.at("scales").get<std::vector<double>>();b.constant=false;m.blocks.push_back(std::move(b));}
  auto& b=m.blocks.at(bi);require(b.name==q.at("name")&&b.values.size()==8,"pair block changed");b.constant=false;b.fixed_indices.clear();
  require(b.lower.size()==8&&b.upper.size()==8&&b.scales.size()==8,"pair domain sizes");for(int k=0;k<8;++k)require(std::isfinite(b.values[k])&&b.values[k]>=b.lower[k]&&b.values[k]<=b.upper[k]&&b.scales[k]>0,"invalid pair coordinates");
  const auto& tangent=item.at("manifold");require(tangent.at("block")==bi&&tangent.at("ambient_size")==8&&tangent.at("tangent_size").get<int>()>=5,"pair tangent changed");bool found=false;for(const auto& d:manifolds)if(d.at("block")==bi){require(d==tangent,"different saved pair tangent");found=true;}if(!found)manifolds.push_back(tangent);
  require(std::abs(b.values[1]-b.values[5])<1e-12&&std::abs((b.values[3]-b.values[2])-(b.values[7]-b.values[6]))<1e-12,"paired dimensions disagree");
 }

 // Re-express the retained unknown local texture point and its finite chart.
 // This is an invertible coordinate change, not a new world measurement.
 if(patch.contains("texture_coordinate_rebind_v1")){
  const auto previous=m.runtime_model_metadata.value("paired_texture_coordinate_rebind_v1",m.source_json.value("paired_texture_coordinate_rebind_v1",Json()));
  if(previous.is_null()){
   Json records=Json::array();std::set<int>geometries;auto rebound_blocks=m.runtime_model_metadata.value("blocks",m.source_json.at("blocks"));
   for(const auto&item:patch.at("texture_coordinate_rebind_v1")){
    const int gi=item.at("geometry_block"),old_ad=item.at("previous_advertisement_block"),new_ad=item.at("paired_advertisement_block"),off=item.at("paired_offset");auto&g=m.blocks.at(gi);const auto&a=m.blocks.at(old_ad).values;const auto&n=m.blocks.at(new_ad).values;
    require(geometries.insert(gi).second&&g.name==item.at("geometry_name")&&g.name.rfind("board_texture_shared_uv/",0)==0&&g.values.size()==6&&g.fixed_indices==std::vector<int>({0,1,3,4})&&a.size()==4&&n.size()==8&&pair_blocks.count(new_ad)&&(off==0||off==4),"invalid retained texture coordinate binding");
    const double ratios[2]={a[1]/n[off+1],(a[3]-a[2])/(n[off+3]-n[off+2])};const double shifts[2]={(a[0]-n[off])/n[off+1],(a[2]-n[off+2])/(n[off+3]-n[off+2])};
    const auto values_before=g.values,lower_before=g.lower,upper_before=g.upper;double error=0.;
    for(int axis=0;axis<2;++axis){const int ix=axis?5:2;require(std::isfinite(ratios[axis])&&ratios[axis]>0&&std::isfinite(shifts[axis]),"noninvertible texture chart change");g.values[ix]=shifts[axis]+ratios[axis]*g.values[ix];g.lower[ix]=shifts[axis]+ratios[axis]*g.lower[ix];g.upper[ix]=shifts[axis]+ratios[axis]*g.upper[ix];g.scales[ix]*=ratios[axis];require(std::isfinite(g.values[ix])&&g.values[ix]>=g.lower[ix]&&g.values[ix]<=g.upper[ix],"texture chart seed outside preserved domain");for(const auto&pair:std::array<std::pair<double,double>,3>{{{values_before[ix],g.values[ix]},{lower_before[ix],g.lower[ix]},{upper_before[ix],g.upper[ix]}}}){const double before=axis?a[2]+(a[3]-a[2])*pair.first:a[0]+a[1]*pair.first;const double after=axis?n[off+2]+(n[off+3]-n[off+2])*pair.second:n[off]+n[off+1]*pair.second;error=std::max(error,std::abs(before-after));}}
    require(error<1e-12,"texture chart changed the physical point or domain");rebound_blocks[gi]["lower"]=g.lower;rebound_blocks[gi]["upper"]=g.upper;rebound_blocks[gi]["scales"]=g.scales;records.push_back({{"geometry_block",gi},{"previous_advertisement_block",old_ad},{"paired_advertisement_block",new_ad},{"paired_offset",off},{"world_point_and_bounds_max_error",error}});
   }
   m.runtime_model_metadata["blocks"]=std::move(rebound_blocks);
   m.runtime_model_metadata["paired_texture_coordinate_rebind_v1"]={{"source",ref},{"records",records},{"raw_source_UV_changed",false},{"local_coordinate_freedoms_preserved",true}};
  }else if(previous.at("source")!=ref){
   // A print-chart revision does not rebind texture points a second time.
   // Require its parent and every original texture transformation to agree.
   require(patch.contains("parent_template")&&patch.at("parent_template")==previous.at("source"),"different already-consumed texture coordinate rebind");const auto& parent_ref=patch.at("parent_template");eventio::verify(parent_ref.at("file").get<std::string>(),parent_ref.at("sha256"));const auto parent=eventio::read_json(parent_ref.at("file").get<std::string>());require(parent.at("texture_coordinate_rebind_v1")==patch.at("texture_coordinate_rebind_v1"),"print chart revision changed retained texture coordinates");
  }
 }
 std::size_t changed=0,added=0;std::set<int> seen;
 for(const auto& item:patch.at("materials")){
  const int mi=item.at("material_index");require(mi>=0&&mi<=int(m.materials.size())&&seen.insert(mi).second,"material append order");const auto& q=item.at("material");const std::string label=q.at("label");const bool texture=label.rfind("board.texture.",0)==0;require(label.rfind("Ad",0)==0||texture,"not a printed board instance");const bool is_new=mi==int(m.materials.size());
  if(is_new){m.materials.push_back(EventMaterial{});materials.push_back(q);++added;}else require(m.materials.at(mi).label==label,"existing source material renamed");
  auto& mat=m.materials.at(mi);require(!texture||!is_new,"texture identities cannot be invented by a shape transition");if(!is_new){require(mat.paths.size()==q.at("paths").size(),"existing source path partition changed");for(std::size_t k=0;k<mat.paths.size();++k){const auto& a=mat.paths[k];const auto& z=q.at("paths").at(k);require(a.id==z.at("id")&&a.variant_id==z.at("variant_id")&&a.closed==z.at("closed")&&a.physical_point==z.at("physical_point"),"existing source contour identity changed");if(texture)require(a.kind==2&&a.physical_point&&a.geometry_block==z.at("geometry_block")&&a.points==z.at("points").get<std::vector<Vec2<double>>>(),"paired texture must retain its original free local coordinates");}}
  mat.label=label;mat.paths.clear();int pi=0;for(const auto& z:q.at("paths")){
   EventPath p;p.id=z.at("id");p.variant_id=z.at("variant_id");p.kind=z.at("kind");p.geometry_block=z.value("geometry_block",-1);p.advertisement_block=z.at("advertisement_block");p.advertisement_offset=z.at("advertisement_offset");p.physical_point=z.at("physical_point");p.closed=z.at("closed");p.provenance=z.at("provenance");p.points=z.at("points").get<std::vector<Vec2<double>>>();
   require(p.kind==2&&(p.geometry_block<0||shared_charts.count(p.geometry_block)||(texture&&p.physical_point&&p.geometry_block>=0))&&pair_blocks.count(p.advertisement_block)&&(p.advertisement_offset==0||p.advertisement_offset==4)&&!p.points.empty(),"invalid shared printed instance");for(const auto& xy:p.points)require(std::isfinite(xy[0])&&std::isfinite(xy[1]),"nonfinite printed shape");
   mat.paths.push_back(std::move(p));m.path_geometry_offsets[{mi,pi}]=z.value("geometry_offset",0);m.path_geometry_indices[{mi,pi}]=z.value("geometry_indices",std::vector<int>{});if(is_new)m.path_station_intervals[{mi,pi}]={{0.,1.}};++pi;
  }materials[mi]=q;++changed;
 }

 // Correct reviewed native glyphs in their existing free print chart. Raw
 // observations and the installation/affine coordinates are separate state.
 std::size_t corrected_native=0;
 if(patch.contains("unpaired_native_print_materials")){
  const auto& proof=patch.at("native_badge_repair_proof");
  require(proof.at("schema")=="rinkfit.native_badge_shape_correction.v1"&&proof.at("native_RGB_review_completed")==true&&proof.at("physical_values_or_bounds_changed")==false&&proof.at("new_physical_parameters")==0&&proof.at("original_raw_UV_ID_PTS_retained")==true,"native glyph repair lacks source review");
  for(const char* key:{"parent_model","canonical_source","raw_source_review"}){const auto& r=proof.at(key);eventio::verify(r.at("file").get<std::string>(),r.at("sha256"));}
  for(const auto& item:patch.at("unpaired_native_print_materials")){
   const int mi=item.at("material_index"),g=item.at("expected_geometry_block"),ad=item.at("expected_advertisement_block");
   require(mi>=0&&mi<int(m.materials.size())&&seen.insert(mi).second&&g>=0&&ad>=0&&g<int(m.blocks.size())&&ad<int(m.blocks.size())&&m.blocks[g].values.size()==6&&m.blocks[ad].values.size()==4&&!pair_blocks.count(ad),"native glyph changed parameter topology");
   const auto& q=item.at("material");auto& mat=m.materials.at(mi);const std::string label=q.at("label");require(label==mat.label&&label.rfind("Ad",0)==0&&label.find(".print.")!=std::string::npos&&q.at("paths").size()==mat.paths.size(),"native glyph renamed an existing material/path");
   for(std::size_t k=0;k<mat.paths.size();++k){auto& p=mat.paths[k];const auto& z=q.at("paths").at(k);
    require(p.id==z.at("id")&&p.variant_id==z.at("variant_id")&&p.closed==z.at("closed")&&p.physical_point==z.at("physical_point")&&!p.physical_point&&p.kind==2&&z.at("kind")==2&&p.geometry_block==g&&z.at("geometry_block")==g&&p.advertisement_block==ad&&z.at("advertisement_block")==ad,"native glyph changed its source or physical identity");
    const auto points=z.at("points").get<std::vector<Vec2<double>>>();require(points.size()>=2,"empty repaired native glyph");for(const auto& xy:points)require(std::isfinite(xy[0])&&std::isfinite(xy[1]),"nonfinite native glyph");p.points=points;p.provenance=z.at("provenance");
   }
   materials[mi]=q;++corrected_native;
  }
 }

 // Unused historical coordinates remain in the saved prefix, with no residuals.
 std::set<int> consumed;for(const auto& mat:m.materials)for(const auto& p:mat.paths){if(p.geometry_block>=0)consumed.insert(p.geometry_block);if(p.advertisement_block>=0)consumed.insert(p.advertisement_block);}
 for(std::size_t i=0;i<m.blocks.size();++i){auto& b=m.blocks[i];if(!consumed.count(int(i))&&(b.name.rfind("ad/",0)==0||b.name.rfind("print/",0)==0))b.constant=true;}
 m.runtime_model_metadata["materials"]=std::move(materials);m.runtime_model_metadata["linear_control_manifolds"]=std::move(manifolds);m.runtime_model_metadata["paired_advertisement_templates"]=ref;
 return {{"enabled",true},{"source",ref},{"pairs",pair_blocks.size()},{"updated_materials",changed},{"new_materials",added},{"corrected_native_materials",corrected_native},{"same_width_height_by_shared_parameterization",true},{"both_locations_remain_free",true},{"source_pixels_weights_and_ids_changed",false},{"added_equality_observations",0},{"quality_accepted",false}};
}
}
