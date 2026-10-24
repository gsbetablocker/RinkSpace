#pragma once
#include <utility>
namespace rinkfit {
// Same ordered curve and same parameter owners, including the station domain.
// Labels and provenance do not participate in the mathematical parameterization.
template<class Model>
inline bool joint_same_station_parameterization_v1(const Model& model,int old_material,int old_path,int new_material,int new_path){
 const auto&a=model.materials.at(old_material).paths.at(old_path);
 const auto&b=model.materials.at(new_material).paths.at(new_path);
 if(a.kind!=b.kind||a.geometry_block!=b.geometry_block||a.advertisement_block!=b.advertisement_block||
    a.advertisement_offset!=b.advertisement_offset||a.radius_index!=b.radius_index||a.boundary_segment!=b.boundary_segment||
    a.physical_point!=b.physical_point||a.closed!=b.closed||a.points!=b.points||
    a.start_angle!=b.start_angle||a.end_angle!=b.end_angle||a.fixed_height_fraction!=b.fixed_height_fraction)return false;
 const auto old_key=std::make_pair(old_material,old_path),new_key=std::make_pair(new_material,new_path);
 auto same_map=[&](const auto& values){const auto i=values.find(old_key),j=values.find(new_key);
  return (i==values.end())==(j==values.end())&&(i==values.end()||i->second==j->second);};
 return same_map(model.path_geometry_indices)&&same_map(model.path_geometry_offsets)&&
        model.station_intervals(old_material,old_path)==model.station_intervals(new_material,new_path);
}
template<class Model>
inline double joint_reviewed_station_initializer_v1(const Model& model,int old_material,int old_path,int new_material,int new_path,double saved_station,double review_station){
 return joint_same_station_parameterization_v1(model,old_material,old_path,new_material,new_path)?saved_station:review_station;
}
}
