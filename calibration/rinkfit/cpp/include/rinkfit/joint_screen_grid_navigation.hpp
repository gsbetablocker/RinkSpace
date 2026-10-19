#pragma once
#include "rinkfit/joint_al.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <cmath>
#include <set>

namespace rinkfit {
namespace joint_screen_grid_navigation_detail {
// Fixed virtual plane coordinates and direct screen-layout targets are only
// numerical navigation. They are never inserted into JointPacket or original W.
struct PlaneLayoutCost {
 const JointEventModel*model=nullptr;JointRowLayout layout;
 std::vector<std::array<double,2>>world_xy,target_uv;double pixel_scale=1;
 template<class T>bool operator()(T const*const*pars,T*residual)const{try{
  const auto g=model->global(layout,pars);std::vector<const T*>p;std::vector<T>geometry;
  const auto e=model->legacy_layout(layout,p,g,pars,geometry);const auto&math=model->shots.at(layout.shot_index).math;
  const auto h=math.head(g.data());const auto o=math.optics(g.data());
  for(std::size_t k=0;k<world_xy.size();++k){
   // Native target row gives the existing signed-RS clock, never a new row DOF.
   auto tr=math.trajectory(row_time(T(layout.frame_time),g[31],T(target_uv[k][1])),p.data(),e);
   tr[0]+=pars[layout.reference_angles_slot][0];tr[1]+=pars[layout.reference_angles_slot][1];
   const auto zoom=zoom_optics(o,tr[2]);const auto pose=camera_pose(h,tr[0],tr[1],zoom.z);
   if(!pose.valid()||!valid_rotation(pose.R))return false;
   const Vec3<T>world{{T(world_xy[k][0]),T(world_xy[k][1]),T(0)}};
   const auto xc=matvec(pose.R,sub(world,pose.C));
   if(!(xc[2]>T(0))||!finite_array(xc))return false;
   const Vec2<T>xy{{xc[0]/xc[2],xc[1]/xc[2]}};
   const auto d=brown_distort(xy,zoom.brown);
   const Vec2<T>uv{{zoom.intr.fx*d[0]+zoom.intr.cx,zoom.intr.fy*d[1]+zoom.intr.cy}};
   residual[2*k]=T(pixel_scale)*(uv[0]-T(target_uv[k][0]));
   residual[2*k+1]=T(pixel_scale)*(uv[1]-T(target_uv[k][1]));
   if(!finite_scalar(residual[2*k])||!finite_scalar(residual[2*k+1]))return false;
  }return true;
 }catch(const std::exception&){return false;}}
};
} // namespace joint_screen_grid_navigation_detail

struct JointScreenGridNavigation {
 std::vector<ceres::ResidualBlockId>residual_blocks;eventio::Json receipt;
 double half_cost(ceres::Problem&problem)const{if(residual_blocks.empty())return 0.;ceres::Problem::EvaluateOptions o;o.residual_blocks=residual_blocks;double value=NAN;if(!problem.Evaluate(o,&value,nullptr,nullptr,nullptr)||!std::isfinite(value))throw std::runtime_error("direct grid navigation cost evaluation failed");return value;}
 void remove_from(ceres::Problem&problem){for(auto id:residual_blocks)problem.RemoveResidualBlock(id);residual_blocks.clear();}
};

// Call only after build_engineering_problem + apply_recovery_stage: the original
// full native pixel/RS/physical graph and its GPU callback stay present from the
// first iteration. Remove these factors before reporting original-objective cost.
inline JointScreenGridNavigation add_joint_screen_grid_numerical_navigation(
 JointALBuilder&b,ceres::Problem&problem,const eventio::Json&spec){
 using namespace joint_screen_grid_navigation_detail;
 if(spec.at("schema")!="rinkfit.fixed_screen_plane_navigation.v1"||
    spec.at("numerical_initialization_only")!=true||spec.at("source_observation")!=false)
  throw std::runtime_error("direct grid navigation must be an explicit numerical-only layout stage");
 if(spec.at("source_packet_sha256").get<std::string>()!=b.core.packet.source_hash)
  throw std::runtime_error("direct grid navigation source packet mismatch");
 const auto&state=spec.at("initial_screen_grid_state");
 eventio::verify(state.at("file").get<std::string>(),state.at("sha256"));
 const auto&proof=spec.at("homography_layout_evidence");
 eventio::verify(proof.at("file").get<std::string>(),proof.at("sha256"));
 const auto&roles_ref=spec.at("frame_role_manifest");eventio::verify(roles_ref.at("file").get<std::string>(),roles_ref.at("sha256"));
 const auto roles=eventio::read_json(roles_ref.at("file").get<std::string>());
 if(roles.at("schema")!="rinkfit.joint_source_frame_protection.v1"||roles.at("source_video_sha256")!=b.core.model.source_json.at("source_clock").at("source_video_sha256"))throw std::runtime_error("direct grid navigation frame-role identity mismatch");
 if(spec.at("world_xy_gauge").get<std::string>().empty())throw std::runtime_error("direct grid virtual plane gauge is not declared");
 const double scale=spec.at("pixel_scale");if(!(scale>0&&std::isfinite(scale)))throw std::runtime_error("invalid numerical layout pixel scale");
 int material=-1,path=-1;
 for(std::size_t mi=0;mi<b.core.model.materials.size()&&material<0;++mi){const auto&m=b.core.model.materials[mi];
  if(m.label.rfind("ice.",0)!=0)continue;
  for(std::size_t pi=0;pi<m.paths.size();++pi)if(m.paths[pi].kind==0||m.paths[pi].kind==1){material=int(mi);path=int(pi);break;}
 }
 if(material<0)throw std::runtime_error("no original ice layout available for camera parameter connections");
 JointScreenGridNavigation out;std::set<std::int64_t>seen;std::size_t points=0;
 for(const auto&frame:spec.at("frames")){
  const auto sid=frame.at("shot_id").get<std::string>();if(sid=="s01")throw std::runtime_error("direct grid navigation cannot replace S01");
  const auto si=b.core.model.shot_map.at(sid);const auto&shot=b.core.model.shots.at(si);
  const auto pts=frame.at("native_pts").get<std::int64_t>();
  if(pts<shot.first_pts||pts>shot.last_pts||(pts-shot.first_pts)%256||!seen.insert(pts).second)
   throw std::runtime_error("direct grid navigation native scope/identity mismatch");
  if(frame.at("frame_role_allows_numerical_navigation")!=true)
   throw std::runtime_error("protected exposure cannot be a numerical layout fitting target");
  const auto&role=roles.at("frame_roles").at(std::to_string(pts));const auto role_name=role.at("role").get<std::string>();
  if(role.at("shot_id")!=sid||role_name=="reserved_pending_lineage"||role_name=="validation_guard"||role_name=="sealed_validation")throw std::runtime_error("direct grid target violates the original frame-role guard");
  auto*functor=new PlaneLayoutCost;functor->model=&b.core.model;functor->layout=b.core.model.layout(si,material,path,pts);
  functor->world_xy=frame.at("world_xy").get<std::vector<std::array<double,2>>>();
  functor->target_uv=frame.at("target_uv").get<std::vector<std::array<double,2>>>();functor->pixel_scale=scale;
  if(functor->world_xy.empty()||functor->world_xy.size()!=functor->target_uv.size()){delete functor;throw std::runtime_error("incomplete direct grid plane probe layout");}
  for(std::size_t k=0;k<functor->world_xy.size();++k)if(!std::isfinite(functor->world_xy[k][0])||!std::isfinite(functor->world_xy[k][1])||!std::isfinite(functor->target_uv[k][0])||!std::isfinite(functor->target_uv[k][1])||functor->target_uv[k][1]<0||functor->target_uv[k][1]>1079){delete functor;throw std::runtime_error("direct grid navigation requires finite, native-row probes");}
  auto*cost=new ceres::DynamicAutoDiffCostFunction<PlaneLayoutCost,4>(functor);
  std::vector<double*>parameters;
  for(int id:functor->layout.block_ids){auto&block=b.core.model.blocks.at(id);if(!problem.HasParameterBlock(block.values.data())){delete cost;throw std::runtime_error("navigation block missing from full original graph");}cost->AddParameterBlock(int(block.values.size()));parameters.push_back(block.values.data());}
  cost->SetNumResiduals(int(2*functor->world_xy.size()));points+=functor->world_xy.size();
  out.residual_blocks.push_back(problem.AddResidualBlock(cost,nullptr,parameters));
 }
 if(out.residual_blocks.empty())throw std::runtime_error("direct grid navigation has no permitted exposure targets");
 out.receipt={{"schema","rinkfit.direct_grid_numerical_navigation_factors.v1"},{"frames",seen.size()},{"plane_probe_points",points},{"pixel_scale",scale},
  {"initial_screen_grid_state",state},{"homography_layout_evidence",proof},{"frame_role_manifest",roles_ref},{"world_xy_gauge",spec.at("world_xy_gauge")},
  {"virtual_plane_probes_are_source_observations",false},{"original_G9_pixel_RS_physical_factors_removed",false},
  {"S01_direct_grid_factors",0},{"source_UV_W_Query_or_qualification_changed",false},{"new_physical_DOFs",0},
  {"original_bounds_and_manifolds_retained",true},{"original_objective_cost_requires_navigation_factors_removed",true},
  {"full_direct_grid_layout_acceptance_claimed",false},{"physical_candidate_requires_separate_generation_and_pixel_comparison",true}};
 return out;
}
} // namespace rinkfit
