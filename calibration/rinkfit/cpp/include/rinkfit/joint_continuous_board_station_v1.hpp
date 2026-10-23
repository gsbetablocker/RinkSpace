#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_finite_source_paths_v1.hpp"
#include <set>
namespace rinkfit {
// A numerical coordinate on the existing continuous board: (segment+s)/8.
// The original material/path identity and all source permission records stay intact.
// Segment lengths remain physical variables; no rink dimension is fixed here.
inline eventio::Json restore_joint_continuous_board_stations_v1(JointALBuilder& b,const eventio::Json&cfg,bool convert_primal=true){
 const auto stage=cfg.value("runtime_shared_parameter_stage_v2",eventio::Json::object());
 if(!stage.value("continuous_board_station_v1",false))return {{"enabled",false}};
 auto&m=b.core.model;const auto&meta=m.runtime_model_metadata.contains("continuous_board_station_v1")?m.runtime_model_metadata:m.source_json;
 const bool already=meta.contains("continuous_board_station_v1");
 eventio::Json policy=already?meta.at("continuous_board_station_v1"):eventio::Json{{"schema","rinkfit.continuous_board_station.v1"},{"original_segments",eventio::Json::object()}};
 if(policy.at("schema")!="rinkfit.continuous_board_station.v1")throw std::runtime_error("continuous board coordinate version");
 std::map<std::pair<int,int>,int>segments;
 for(std::size_t mi=0;mi<m.materials.size();++mi){const auto&mat=m.materials[mi];for(std::size_t pi=0;pi<mat.paths.size();++pi){const auto&p=mat.paths[pi];if(p.kind!=3)continue;const auto key=std::to_string(mi)+"/"+std::to_string(pi);int part=p.boundary_segment;
  if(already){if(!policy.at("original_segments").contains(key))continue;part=policy.at("original_segments").at(key);if(p.boundary_segment!=-2&&p.boundary_segment!=part)throw std::runtime_error("saved continuous board segment identity");}
  if(part<0||part>=8)continue;segments[{int(mi),int(pi)}]=part;policy["original_segments"][key]=part;
 }}
 std::size_t rows=0,active=0,joined=0,restricted=0;eventio::Json byshot=eventio::Json::object();
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){auto&r=b.core.packet.rows[i];const auto found=segments.find({r.material,r.path});if(found==segments.end()||r.kind==1)continue;
  r.numerical_continuous_board_intervals.clear();const auto&current=m.materials.at(r.material).paths.at(r.path);std::vector<std::array<double,2>>intervals;
  for(int pi:joint_source_allowed_finite_paths_v1(b.core.packet,m,i)){const auto it=segments.find({r.material,pi});if(it==segments.end())throw std::runtime_error("continuous board source permission changes material type");const auto&p=m.materials.at(r.material).paths.at(pi);
   if(p.geometry_block!=current.geometry_block||p.fixed_height_fraction!=current.fixed_height_fraction)throw std::runtime_error("continuous board source permission changes edge role or height");
   for(const auto&v:r.station_intervals(m,pi))intervals.push_back({(it->second+v[0])/8.,(it->second+v[1])/8.});
  }
  std::sort(intervals.begin(),intervals.end());std::vector<std::array<double,2>>merged;for(auto v:intervals){if(!merged.empty()&&v[0]<=merged.back()[1])merged.back()[1]=std::max(merged.back()[1],v[1]);else merged.push_back(v);}
  if(merged.empty())throw std::runtime_error("continuous board has no source-admitted station interval");
  if(convert_primal&&!already){b.core.station[i]=(found->second+b.core.station[i])/8.;r.station=(found->second+r.station)/8.;}
  if(r.active){bool legal=false;for(auto v:merged)legal=legal||(b.core.station[i]>=v[0]&&b.core.station[i]<=v[1]);if(!legal)throw std::runtime_error("continuous board primal lies outside original source permission");++active;byshot[r.shot_id]=byshot.value(r.shot_id,std::size_t(0))+1;joined+=intervals.size()>merged.size();restricted+=merged!=std::vector<std::array<double,2>>{{0.,1.}};}
  r.numerical_continuous_board_intervals=std::move(merged);++rows;
 }
 auto materials=m.runtime_model_metadata.value("materials",m.source_json.at("materials"));
 for(const auto&x:segments){const int mi=x.first.first,pi=x.first.second;auto&p=m.materials.at(mi).paths.at(pi);p.boundary_segment=-2;p.closed=true;materials.at(mi).at("paths").at(pi)["boundary_segment"]=-2;materials.at(mi).at("paths").at(pi)["closed"]=true;}
 m.runtime_model_metadata["materials"]=std::move(materials);m.runtime_model_metadata["continuous_board_station_v1"]=policy;
 return {{"enabled",true},{"schema","rinkfit.continuous_board_station_application.v1"},{"rows",rows},{"active_rows",active},{"joined_source_admitted_seams",joined},{"restricted_rows",restricted},{"shots",byshot},{"primal_coordinate_converted",convert_primal&&!already},{"physical_values_source_UV_ID_PTS_E_W_changed",false},{"path_switch_or_GPU_rebuild_needed_at_seam",false}};
}
}
