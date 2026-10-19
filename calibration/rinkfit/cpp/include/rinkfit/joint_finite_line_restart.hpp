#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <set>

namespace rinkfit {
namespace joint_finite_line_detail {
struct Auxiliary {std::size_t i=0;double station=0,row=0,source_station=0,source_row=0,source_alpha=0,proposed_station=0;};
struct Plan {
  bool proposed=false;int block=-1,material=-1,path=-1,endpoint=-1;
  std::array<double,4> before{},after{};double required_extent=1,extent=1,max_preserved_world_error=0;
  std::vector<Auxiliary> auxiliaries;std::vector<std::size_t>witnesses;
  eventio::Json navigation=eventio::Json::array();std::string withheld_reason;
};
inline eventio::Json finite(double x){return std::isfinite(x)?eventio::Json(x):eventio::Json();}
inline eventio::Json ray_vector(const Vec3<double>&v){return {finite(v[0]),finite(v[1]),finite(v[2])};}
inline double max_difference(const Mat3<double>&a,const Mat3<double>&b){double value=0;for(int i=0;i<9;++i)value=std::max(value,std::abs(a[i]-b[i]));return value;}
// The legacy inverse is used only after comparing its actual row-time camera
// to the same runtime joint camera, including the separate reference angles.
inline bool inverse_navigation(const JointEventModel&m,const JointObservation&r,const EventModel&legacy,EventRayResult*ray,eventio::Json*record){
  const auto layout=r.layout(m);const auto pointers=m.pointers(layout);const auto g=m.global(layout,pointers.data());std::vector<const double*>p;std::vector<double>geometry;
  const auto e=m.legacy_layout(layout,p,g,pointers.data(),geometry);const auto&math=m.shots.at(layout.shot_index).math;
  const double time=row_time(layout.frame_time,g[31],r.uv[1]);auto tr=math.trajectory(time,p.data(),e);tr[0]+=pointers[layout.reference_angles_slot][0];tr[1]+=pointers[layout.reference_angles_slot][1];
  const auto optics=zoom_optics(math.optics(g.data()),tr[2]);const auto pose=camera_pose(math.head(g.data()),tr[0],tr[1],optics.z);
  const auto ll=legacy.layout(0,0,layout.frame_time);const auto lp=legacy.pointers(ll);const auto&lg=legacy.blocks.at(0).values;const double lt=row_time(layout.frame_time,lg[31],r.uv[1]);
  const auto ltr=legacy.trajectory(lt,lp.data(),ll);const auto lo=zoom_optics(legacy.optics(lg.data()),ltr[2]);const auto lpose=camera_pose(legacy.head(lg.data()),ltr[0],ltr[1],lo.z);
  double center_error=INFINITY,rotation_error=INFINITY,optics_error=INFINITY;
  if(pose.valid()&&lpose.valid()){center_error=std::sqrt(dot(sub(pose.C,lpose.C),sub(pose.C,lpose.C)));rotation_error=max_difference(pose.R,lpose.R);
    const double a[]={optics.intr.fx,optics.intr.fy,optics.intr.cx,optics.intr.cy,optics.brown.k1,optics.brown.k2,optics.brown.p1,optics.brown.p2,optics.brown.radius_max};
    const double b[]={lo.intr.fx,lo.intr.fy,lo.intr.cx,lo.intr.cy,lo.brown.k1,lo.brown.k2,lo.brown.p1,lo.brown.p2,lo.brown.radius_max};optics_error=0;for(int i=0;i<9;++i)optics_error=std::max(optics_error,std::abs(a[i]-b[i]));}
  *record={{"runtime_native_row_time",finite(time)},{"legacy_native_row_time",finite(lt)},{"runtime_legacy_center_error_L1",finite(center_error)},{"runtime_legacy_rotation_max_error",finite(rotation_error)},{"runtime_legacy_optics_max_error",finite(optics_error)}};
  if(!(std::isfinite(time)&&std::isfinite(lt)&&std::abs(time-lt)<=1e-12&&center_error<=2e-12&&rotation_error<=2e-12&&optics_error<=1e-8)){(*record)["navigation_failure"]="legacy_snapshot_runtime_camera_mismatch";return false;}
  *ray=legacy.inverse_ray(r.uv[0],r.uv[1],layout.frame_time);(*record)["inverse_ray_valid"]=ray->valid;(*record)["forward_ice_intersection"]=ray->forward_ice_intersection;(*record)["inside_current_rink"]=ray->inside_rink;(*record)["inverse_ray_reason"]=ray->reason;(*record)["inverse_distortion_error_px"]=finite(ray->inverse_distortion_error);
  if(!ray->valid||!ray->forward_ice_intersection||!finite_array(ray->ice))return false;
  const auto check=project(add(ray->origin,ray->direction),pose,optics.intr,optics.brown);
  const double error=check.valid()?std::hypot(check.uv[0]-r.uv[0],check.uv[1]-r.uv[1]):INFINITY;
  const double inverse_origin=std::sqrt(dot(sub(ray->origin,pose.C),sub(ray->origin,pose.C)));
  (*record)["runtime_native_UV_roundtrip_error_px"]=finite(error);(*record)["inverse_runtime_origin_error_L1"]=finite(inverse_origin);(*record)["ice_navigation_L1"]=ray_vector(ray->ice);
  if(!(error<=1e-6&&inverse_origin<=2e-12)){(*record)["navigation_failure"]="inverse_ray_runtime_UV_roundtrip_failed";return false;}return true;
}
inline int geometry_orientation(const JointEventModel&m,int material,int path){
  const auto&v=m.materials.at(material).paths.at(path);if(v.kind!=0||v.physical_point)return 0;
  const auto&indices=m.path_geometry_indices.at({material,path});const int off=m.path_geometry_offsets.at({material,path});if(off!=0)return 0;
  if(indices.empty()||indices==std::vector<int>{0,1,2,3})return 1;if(indices==std::vector<int>{2,3,0,1})return -1;return 0;
}
inline Plan propose(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&spec){
  Plan plan;try{
    if(spec.at("source_packet_sha256").get<std::string>()!=b.core.packet.source_hash)throw std::runtime_error("finite-line navigation packet binding mismatch");
    const auto&clock=b.core.model.source_json.at("source_clock");if(clock.at("source_video_sha256")!=spec.at("source_video_sha256")||clock.at("time_base")!=eventio::Json::array({1,12800})||clock.at("frame_step_pts")!=256)throw std::runtime_error("finite-line navigation native source clock mismatch");
    if(spec.at("source_evidence").empty())throw std::runtime_error("finite-line navigation lacks actual source report");for(const auto&ref:spec.at("source_evidence"))eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
    plan.block=spec.at("geometry_block");plan.material=spec.at("material_index");plan.path=spec.at("path");plan.endpoint=spec.at("endpoint");if(plan.endpoint!=0&&plan.endpoint!=1)throw std::runtime_error("finite-line navigation endpoint must be explicit");
    const auto&mat=b.core.model.materials.at(plan.material);const auto&path=mat.paths.at(plan.path);auto&block=b.core.model.blocks.at(plan.block);
    if(mat.label!=spec.at("material_label").get<std::string>()||mat.label.rfind("ice.",0)!=0||path.geometry_block!=plan.block||geometry_orientation(b.core.model,plan.material,plan.path)!=1||block.values.size()!=4||block.lower.size()!=4||block.upper.size()!=4||block.constant||!block.fixed_indices.empty()||!problem.HasParameterBlock(block.values.data())||problem.IsParameterBlockConstant(block.values.data())||problem.ParameterBlockTangentSize(block.values.data())!=4)throw std::runtime_error("finite-line navigation requires existing four freely variable ice-line coordinates");
    std::copy(block.values.begin(),block.values.end(),plan.before.begin());plan.after=plan.before;const int moving=plan.endpoint*2,fixed=(1-plan.endpoint)*2;
    const Vec2<double>A={plan.before[fixed],plan.before[fixed+1]},D={plan.before[moving]-A[0],plan.before[moving+1]-A[1]};const double length2=D[0]*D[0]+D[1]*D[1];if(!(length2>0&&std::isfinite(length2)))throw std::runtime_error("finite-line navigation degenerate existing segment");
    if(spec.at("source_rows").size()!=2)throw std::runtime_error("finite-line navigation requires the two explicitly reviewed source witnesses");
    std::map<int,EventModel>snapshots;std::set<std::size_t>unique;
    for(const auto&s:spec.at("source_rows")){
      const auto i=s.at("row_index").get<std::size_t>();const auto&r=b.core.packet.rows.at(i);
      if(!unique.insert(i).second||!r.active||r.kind==1||r.material!=plan.material||r.path!=plan.path||r.source_row_id!=s.at("source_row_id").get<std::string>()||r.source_dataset_id!=s.at("source_dataset_id").get<std::string>()||r.shot_id!=s.at("shot_id").get<std::string>()||r.pts!=s.at("native_pts").get<std::int64_t>()||r.uv!=s.at("source_uv").get<Vec2<double>>())throw std::runtime_error("finite-line navigation source witness identity/UV mismatch");
      for(const auto&name:{"native_png","source_report"}){const auto&ref=s.at(name);eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));}
      const int shot=b.core.model.shot_map.at(r.shot_id);auto it=snapshots.find(shot);if(it==snapshots.end())it=snapshots.emplace(shot,b.core.model.legacy_snapshot(shot)).first;
      EventRayResult ray;eventio::Json record;if(!inverse_navigation(b.core.model,r,it->second,&ray,&record)){record["row_index"]=i;record["source_row_id"]=r.source_row_id;plan.navigation.push_back(record);throw std::runtime_error("finite-line source ray navigation unavailable; no arithmetic fallback");}
      const double t=((ray.ice[0]-A[0])*D[0]+(ray.ice[1]-A[1])*D[1])/length2;
      const Vec2<double>projection={A[0]+t*D[0],A[1]+t*D[1]};const double transverse=std::hypot(projection[0]-ray.ice[0],projection[1]-ray.ice[1]);
      record["row_index"]=i;record["source_row_id"]=r.source_row_id;record["shot_id"]=r.shot_id;record["native_pts"]=r.pts;record["source_uv"]=r.uv;record["navigation_extent"]=finite(t);record["transverse_navigation_error_L1"]=finite(transverse);record["original_native_png"]=s.at("native_png");record["source_report"]=s.at("source_report");plan.navigation.push_back(record);
      if(!(std::isfinite(t)&&t>1))throw std::runtime_error("a reviewed source ray is not beyond the specified current endpoint");plan.required_extent=std::max(plan.required_extent,t);plan.witnesses.push_back(i);
    }
    auto allowed=[&](double extent){const double x=A[0]+extent*D[0],y=A[1]+extent*D[1];
      if(!std::isfinite(x)||!std::isfinite(y)||x<std::max(block.lower[moving],problem.GetParameterLowerBound(block.values.data(),moving))||x>std::min(block.upper[moving],problem.GetParameterUpperBound(block.values.data(),moving))||y<std::max(block.lower[moving+1],problem.GetParameterLowerBound(block.values.data(),moving+1))||y>std::min(block.upper[moving+1],problem.GetParameterUpperBound(block.values.data(),moving+1)))return false;
      return EventModel::rink_membership(x,y,b.core.model.blocks.at(b.core.model.scene_block).values.data())<=0.;};
    if(!allowed(1))throw std::runtime_error("current endpoint is outside its original coordinate/physical domain");
    plan.extent=plan.required_extent;if(!allowed(plan.extent)){double lo=1,hi=plan.extent;for(int k=0;k<64&&std::nextafter(lo,hi)<hi;++k){const double mid=lo+(hi-lo)*.5;if(allowed(mid))lo=mid;else hi=mid;}plan.extent=lo;}
    if(!(plan.extent>1))throw std::runtime_error("original coordinate/rink domain permits no endpoint extension");plan.after[moving]=A[0]+plan.extent*D[0];plan.after[moving+1]=A[1]+plan.extent*D[1];
    // Cover every live consumer of this geometry, including reverse path views
    // and finite paint-width instances. Original source station domains remain
    // untouched; any incompatible view or remap withholds the WHOLE proposal.
    for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;const auto&p=b.core.model.materials.at(r.material).paths.at(r.path);if(p.geometry_block!=plan.block)continue;
      const int orientation=geometry_orientation(b.core.model,r.material,r.path);if(!orientation||r.kind==1||!problem.HasParameterBlock(&b.core.station[i])||problem.IsParameterBlockConstant(&b.core.station[i]))throw std::runtime_error("a live shared-line consumer cannot be exactly reparameterized");
      const double old=b.core.station[i],canonical=orientation>0?old:1-old;
      const double mapped=plan.endpoint==1?canonical/plan.extent:1-(1-canonical)/plan.extent;double station=orientation>0?mapped:1-mapped;
      if(!r.station_allowed(b.core.model,r.path,old,0)||!r.station_allowed(b.core.model,r.path,station,0)||station<problem.GetParameterLowerBound(&b.core.station[i],0)||station>problem.GetParameterUpperBound(&b.core.station[i],0))throw std::runtime_error("an original source station domain forbids exact line reparameterization");
      const Vec2<double>old_point={plan.before[0]*(1-canonical)+plan.before[2]*canonical,plan.before[1]*(1-canonical)+plan.before[3]*canonical};const Vec2<double>new_point={plan.after[0]*(1-mapped)+plan.after[2]*mapped,plan.after[1]*(1-mapped)+plan.after[3]*mapped};
      plan.max_preserved_world_error=std::max(plan.max_preserved_world_error,std::hypot(old_point[0]-new_point[0],old_point[1]-new_point[1]));
      // Width offsets keep the same oriented normal because only line extent
      // changes. Check both normals as well, without constructing host copies.
      if(r.width_block>=0){const double odx=plan.before[2]-plan.before[0],ody=plan.before[3]-plan.before[1],ndx=plan.after[2]-plan.after[0],ndy=plan.after[3]-plan.after[1],ol=std::hypot(odx,ody),nl=std::hypot(ndx,ndy),width=b.core.model.blocks.at(r.width_block).values.at(0);const double error=std::hypot(width*(ody/ol-ndy/nl),width*(odx/ol-ndx/nl));plan.max_preserved_world_error=std::max(plan.max_preserved_world_error,error);}
      for(std::size_t k=0;k<plan.witnesses.size();++k)if(plan.witnesses[k]==i){const double t=plan.navigation[k].at("navigation_extent");const double fraction=std::max(0.,std::min(1.,t/plan.extent));const double canonical_witness=plan.endpoint==1?fraction:1-fraction;station=orientation>0?canonical_witness:1-canonical_witness;
        if(!r.station_allowed(b.core.model,r.path,station,0)||station<problem.GetParameterLowerBound(&b.core.station[i],0)||station>problem.GetParameterUpperBound(&b.core.station[i],0))throw std::runtime_error("reviewed witness navigation violates its unchanged finite source domain");
        if(!problem.HasParameterBlock(&b.core.row[i])||problem.IsParameterBlockConstant(&b.core.row[i]))throw std::runtime_error("reviewed witness row is not an existing free numerical variable");}
      plan.auxiliaries.push_back({i,old,b.core.row[i],r.station,r.row,r.alpha,station});
    }
    if(plan.max_preserved_world_error>2e-12)throw std::runtime_error("shared-line remap did not preserve original world positions");if(plan.auxiliaries.empty())throw std::runtime_error("finite-line navigation target has no live consumers");plan.proposed=true;
  }catch(const std::exception&e){plan.proposed=false;plan.withheld_reason=e.what();plan.auxiliaries.clear();}
  return plan;
}
inline eventio::Json source_status(const JointALBuilder&b,const Plan&plan){eventio::Json rows=eventio::Json::array();for(auto i:plan.witnesses){const auto&r=b.core.packet.rows.at(i);const auto l=r.layout(b.core.model);const auto p=b.core.model.pointers(l);const auto projection=b.core.model.predict_for_al(l,b.core.station[i],b.core.row[i],p.data());const auto root=b.core.model.project_rs(l,b.core.station[i],r.uv[1]);bool visible=false;if(root.valid)visible=b.core.model.visibility(l,b.core.station[i],root.row,p.data()).visible();rows.push_back({{"row_index",i},{"source_row_id",r.source_row_id},{"native_pts",r.pts},{"source_uv",r.uv},{"station",b.core.station[i]},{"stored_row",b.core.row[i]},{"CP_ray_domain_valid",projection.physical_valid()},{"native_RS_valid",root.valid},{"native_visibility_valid",visible},{"native_RS_reason",root.reason},{"native_RS_equality_px",finite(root.equality_px)},{"native_RS_error_px",root.valid?finite(std::hypot(root.uv[0]-r.uv[0],root.uv[1]-r.uv[1])):eventio::Json()}});}return rows;}
inline void apply(JointALBuilder&b,const Plan&plan){auto&v=b.core.model.blocks.at(plan.block).values;std::copy(plan.after.begin(),plan.after.end(),v.begin());for(const auto&a:plan.auxiliaries){b.core.station[a.i]=a.proposed_station;b.core.packet.rows[a.i].station=a.proposed_station;}
  for(auto i:plan.witnesses){auto&r=b.core.packet.rows.at(i);const auto root=b.core.model.project_rs(r.layout(b.core.model),b.core.station[i],r.uv[1]);if(root.valid)b.core.row[i]=r.row=root.row;}}
inline void restore_source_mirrors(JointALBuilder&b,const Plan&plan){for(const auto&a:plan.auxiliaries){auto&r=b.core.packet.rows[a.i];r.station=a.source_station;r.row=a.source_row;r.alpha=a.source_alpha;}}
} // namespace joint_finite_line_detail

// Optional explicit source-bound numerical seed. This adds no relation or DOF,
// fixes no coordinate and never exports an observation beyond station [0,1].
inline eventio::Json refine_joint_finite_line_navigation(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&spec,
    const std::filesystem::path&out,int threads,const std::function<eventio::Json()>&constraint_debt){
  using namespace joint_finite_line_detail;const auto started=std::chrono::steady_clock::now();std::filesystem::create_directories(out);const auto plan=propose(b,problem,spec);
  eventio::Json report={{"schema","rinkfit.source_evidenced_finite_line_endpoint_navigation.v1"},{"source_packet_sha256",b.core.packet.source_hash},{"source_evidence",spec.value("source_evidence",eventio::Json::array())},{"navigation",plan.navigation},{"proposed",plan.proposed},{"accepted",false},{"withheld_reason",plan.withheld_reason},{"geometry_block",plan.block},{"material_index",plan.material},{"path",plan.path},{"endpoint",plan.endpoint},{"all_four_geometry_coordinates_remain_free",plan.proposed?eventio::Json(true):eventio::Json()},{"new_geometry_relations_or_DOF_added",false},{"direction_used_only_for_seed_not_final_endpoint_truth",true},{"source_UV_ID_Query_W_E_or_station_support_changed",false},{"runtime_camera_equivalence_and_native_clock_required_before_proposal",true}};
  if(!plan.proposed){report["status"]="withheld";eventio::atomic_json(out/"receipt.json",report);return report;}
  JointCompactAcceptedState before(b);auto evaluate=[&](){if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))cache->PrepareForEvaluation(false,true);ceres::Problem::EvaluateOptions options;options.num_threads=threads;double cost=NAN;if(!problem.Evaluate(options,&cost,nullptr,nullptr,nullptr)||!std::isfinite(cost))throw std::runtime_error("finite-line navigation complete current PHR evaluation failed");return cost;};
  const double cost_before=evaluate();const auto debt_before=constraint_debt(),sources_before=source_status(b,plan);double cost_candidate=INFINITY;eventio::Json debt_candidate,sources_candidate;std::string error;
  try{apply(b,plan);cost_candidate=evaluate();debt_candidate=constraint_debt();sources_candidate=source_status(b,plan);}catch(const std::exception&e){error=e.what();}
  const bool accepted=error.empty()&&std::isfinite(cost_candidate)&&cost_candidate<cost_before;
  if(!accepted){before.restore_values();restore_source_mirrors(b,plan);evaluate();if(!before.matches_current_values())throw std::runtime_error("finite-line navigation rollback changed accepted values");}
  report["status"]=accepted?"accepted_numerical_seed":"rejected_full_PHR_restored";report["accepted"]=accepted;report["candidate_evaluation_error"]=error;report["endpoint_before"]=plan.before;report["endpoint_proposed"]=plan.after;report["required_extent"]=plan.required_extent;report["allowed_proposed_extent"]=plan.extent;report["remapped_live_shared_geometry_consumers"]=plan.auxiliaries.size();report["non_witness_original_world_preservation_max_error_L1"]=plan.max_preserved_world_error;report["full_current_PHR_cost_before"]=cost_before;report["full_current_PHR_cost_proposed"]=finite(cost_candidate);report["original_constraint_debt_before"]=debt_before;report["original_constraint_debt_proposed"]=debt_candidate;report["diagnostic_source_RS_visibility_before"]=sources_before;report["diagnostic_source_RS_visibility_proposed"]=sources_candidate;report["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();eventio::atomic_json(out/"receipt.json",report);return report;
}
} // namespace rinkfit
