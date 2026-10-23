#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_gpu_cache.hpp"
#include "rinkfit/joint_periodic_station.hpp"
#include <ceres/ceres.h>
#include <map>
#include <set>
namespace rinkfit {
// Conditional minimization of the existing circle geometry and its source
// stations. Every row retains its original identity, measured UV, time and W/E.
// The original live problem, including all domains, accepts or rolls back the
// proposed state. This does not fix any geometry in the joint solver.
class JointObservedIceCircleBlocksV1 {
 struct Pixel {
  Mat3<double>R;Vec3<double>C;double fx,fy,cx,cy,k1,k2,p1,p2,scale,start,sweep,shift;Vec2<double>uv;int ix,iy,ir;
  template<class T>bool operator()(T const*const*p,T*r)const{
   using std::cos;using std::sin;const T s=p[1][0],a=T(start)+T(sweep)*s,rad=p[0][ir]+T(shift);if(!(s>=T(0)&&s<=T(1)&&rad>T(0)))return false;
   const Vec3<T>X{p[0][ix]+rad*cos(a),p[0][iy]+rad*sin(a),T(0)};Vec3<T>q;for(int i=0;i<3;++i){q[i]=T(0);for(int j=0;j<3;++j)q[i]+=T(R[i*3+j])*(X[j]-T(C[j]));}if(!(q[2]>T(0)))return false;
   const T x=q[0]/q[2],y=q[1]/q[2],rr=x*x+y*y,dr=T(1)+T(k1)*rr+T(k2)*rr*rr;
   r[0]=T(scale)*(T(fx)*(x*dr+T(2*p1)*x*y+T(p2)*(rr+T(2)*x*x))+T(cx-uv[0]));r[1]=T(scale)*(T(fy)*(y*dr+T(p1)*(rr+T(2)*y*y)+T(2*p2)*x*y)+T(cy-uv[1]));return true;
  }
 };
 struct Group{int geometry=-1;std::vector<std::pair<std::size_t,double>>rows;};
 JointALBuilder&b_;ceres::Problem&p_;int threads_;std::vector<Group>groups_;
 Pixel pixel(std::size_t i,double weight)const{
  const auto&m=b_.core.model;const auto&r=b_.core.packet.rows[i];const auto l=r.layout(m);const auto ptr=m.pointers(l);const auto g=m.global(l,ptr.data());std::vector<const double*>frozen;std::vector<double>geometry;const auto e=m.legacy_layout(l,frozen,g,ptr.data(),geometry);const auto&math=m.shots[l.shot_index].math;auto tr=math.trajectory(row_time(l.frame_time,g[31],b_.core.row[i]),frozen.data(),e);tr[0]+=ptr[l.reference_angles_slot][0];tr[1]+=ptr[l.reference_angles_slot][1];const auto optical=zoom_optics(math.optics(g.data()),tr[2]);const auto pose=camera_pose(math.head(g.data()),tr[0],tr[1],optical.z);if(!pose.valid())throw std::runtime_error("ice circle block camera invalid");
  const auto&path=m.materials[r.material].paths[r.path];auto index=[&](int k){return l.geometry_indices.empty()?l.geometry_offset+k:l.geometry_indices.at(k);};Pixel v;v.R=pose.R;v.C=pose.C;v.fx=optical.intr.fx;v.fy=optical.intr.fy;v.cx=optical.intr.cx;v.cy=optical.intr.cy;v.k1=optical.brown.k1;v.k2=optical.brown.k2;v.p1=optical.brown.p1;v.p2=optical.brown.p2;v.scale=std::sqrt(weight)/r.E;v.uv=r.uv;v.start=path.start_angle;v.sweep=path.end_angle-path.start_angle;v.ix=index(0);v.iy=index(1);v.ir=index(path.radius_index);v.shift=l.width_slot<0?0.:l.width_sign*ptr[l.width_slot][0];return v;
 }
 public:
 JointObservedIceCircleBlocksV1(JointALBuilder&b,ceres::Problem&p,const eventio::Json&ledger,int threads):b_(b),p_(p),threads_(threads){
  const auto&m=b.core.model;std::map<int,Group>groups;std::set<int>unsupported;
  for(const auto&material:m.materials)for(const auto&path:material.paths)if(path.geometry_block>=0&&path.kind!=1)unsupported.insert(path.geometry_block);
  for(const auto&entry:ledger.at("groups")){const auto&ids=entry.at("actual_row_indices");const auto&w=entry.at("actual_per_row_weight");for(std::size_t k=0;k<ids.size();++k){const auto i=ids[k].get<std::size_t>();const auto&r=b.core.packet.rows[i];const auto&path=m.materials[r.material].paths[r.path];const int g=path.geometry_block;if(path.kind!=1||g<0||unsupported.count(g))continue;const auto&block=m.blocks[g];if(!p.HasParameterBlock(block.values.data())||p.IsParameterBlockConstant(block.values.data())||p.ParameterBlockTangentSize(block.values.data())!=int(block.values.size()))continue;auto&group=groups[g];group.geometry=g;group.rows.emplace_back(i,w[k].get<double>());}}
  for(auto&g:groups)groups_.push_back(std::move(g.second));
 }
 bool minimize(const std::filesystem::path&file){
  if(groups_.empty())return false;const auto started=std::chrono::steady_clock::now();auto&m=b_.core.model;auto objective=[&](){if(auto*c=b_.gpu_evaluation_callback())c->PrepareForEvaluation(false,true);ceres::Problem::EvaluateOptions o;o.num_threads=threads_;double v=INFINITY;if(!p_.Evaluate(o,&v,nullptr,nullptr,nullptr))return double(INFINITY);return v;};const double before=objective();std::map<int,std::vector<double>>backup;std::map<std::size_t,std::pair<double,double>>station_backup;eventio::Json reports=eventio::Json::array();bool changed=false;
  for(const auto&group:groups_){auto&block=m.blocks[group.geometry];const auto old=block.values;auto q=old;std::vector<double>stations(group.rows.size());ceres::Problem mini;auto ordering=std::make_shared<ceres::ParameterBlockOrdering>();std::size_t cursor=0;
   for(const auto&row:group.rows){double*st=&stations[cursor++];*st=b_.core.station[row.first];const auto f=pixel(row.first,row.second);const double*ptrs[]={q.data(),st};double rr[2];if(!f(ptrs,rr))throw std::runtime_error("ice circle block initial projection invalid");const auto&source=b_.core.packet.rows[row.first];const auto l=source.layout(m);const auto ptr=m.pointers(l);const auto uv=m.predict_for_al(l,*st,b_.core.row[row.first],ptr.data());if(!uv.arithmetic_valid||std::max(std::abs(rr[0]/f.scale+source.uv[0]-uv.uv[0]),std::abs(rr[1]/f.scale+source.uv[1]-uv.uv[1]))>1e-7)throw std::runtime_error("ice circle block differs from original native projection");auto*cost=new ceres::DynamicAutoDiffCostFunction<Pixel,8>(new Pixel(f));cost->AddParameterBlock(int(q.size()));cost->AddParameterBlock(1);cost->SetNumResiduals(2);mini.AddResidualBlock(cost,nullptr,q.data(),st);
    if(source.kind==1||!p_.HasParameterBlock(&b_.core.station[row.first])||p_.IsParameterBlockConstant(&b_.core.station[row.first]))mini.SetParameterBlockConstant(st);
    else{if(joint_full_circle_station(m,source))mini.SetManifold(st,new JointPeriodicStationManifold);else{double lo=p_.GetParameterLowerBound(&b_.core.station[row.first],0),hi=p_.GetParameterUpperBound(&b_.core.station[row.first],0);bool found=false;for(auto interval:source.station_intervals(m,source.path))if(*st>=interval[0]&&*st<=interval[1]){lo=std::max(lo,interval[0]);hi=std::min(hi,interval[1]);found=true;break;}if(!found)throw std::runtime_error("ice circle station outside original source interval");mini.SetParameterLowerBound(st,0,lo);mini.SetParameterUpperBound(st,0,hi);}ordering->AddElementToGroup(st,0);}
   }
   for(int j=0;j<int(q.size());++j){mini.SetParameterLowerBound(q.data(),j,p_.GetParameterLowerBound(block.values.data(),j));mini.SetParameterUpperBound(q.data(),j,p_.GetParameterUpperBound(block.values.data(),j));}ordering->AddElementToGroup(q.data(),1);ceres::Solver::Options options;options.linear_solver_type=ceres::DENSE_SCHUR;options.linear_solver_ordering=ordering;options.num_threads=threads_;options.function_tolerance=1e-8;options.gradient_tolerance=1e-10;options.parameter_tolerance=1e-10;ceres::Solver::Summary summary;ceres::Solve(options,&mini,&summary);const bool usable=summary.IsSolutionUsable()&&summary.final_cost<summary.initial_cost;reports.push_back({{"block",group.geometry},{"name",block.name},{"rows",group.rows.size()},{"before",summary.initial_cost},{"after",summary.final_cost},{"usable",usable},{"old_geometry",old},{"proposed_geometry",q},{"seconds",summary.total_time_in_seconds}});if(!usable)continue;
   backup.emplace(group.geometry,old);std::copy(q.begin(),q.end(),block.values.begin());for(std::size_t k=0;k<group.rows.size();++k){const auto i=group.rows[k].first;station_backup.emplace(i,std::make_pair(b_.core.station[i],b_.core.packet.rows[i].station));b_.core.station[i]=b_.core.packet.rows[i].station=stations[k];}changed=true;
  }
  const double after=changed?objective():before;const bool accepted=changed&&std::isfinite(after)&&after<before-64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(before));if(!accepted&&changed){for(const auto&x:backup)std::copy(x.second.begin(),x.second.end(),m.blocks[x.first].values.begin());for(const auto&x:station_backup){b_.core.station[x.first]=x.second.first;b_.core.packet.rows[x.first].station=x.second.second;}objective();}
  eventio::atomic_json(file,{{"schema","rinkfit.actual_ice_circle_block_minimization.v1"},{"accepted",accepted},{"before",before},{"proposed",std::isfinite(after)?eventio::Json(after):eventio::Json()},{"groups",reports},{"same_original_live_objective",true},{"source_UV_ID_PTS_E_weights_changed",false},{"new_parameters_or_fixed_physical_coordinates",false},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}});std::cout<<"Ice circle geometry/stations: actual objective "<<before<<" -> "<<after<<(accepted?" accepted":" unchanged/restored")<<std::endl;return accepted;
 }
};
}
