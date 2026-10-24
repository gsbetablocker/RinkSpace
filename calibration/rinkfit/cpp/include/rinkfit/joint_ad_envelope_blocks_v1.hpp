#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_gpu_cache.hpp"
#include <ceres/ceres.h>
#include <map>
#include <set>
#include <iostream>
namespace rinkfit {
// Four numerical directions separate an observed advertisement envelope from
// the coherent print it contains. A compensated update keeps EVERY printed
// world point unchanged, including independently located texture landmarks.
// All source pixels, weights, identities and physical freedoms are retained.
class JointAdEnvelopeBlocksV1 {
 struct Pixel {
  std::array<double,5> scene;Mat3<double> R;Vec3<double>C;double fx,fy,cx,cy,k1,k2,p1,p2;
  double origin,width,bottom,height,u0,v0,du,dv,scale;Vec2<double>observed;
  template<class T>bool operator()(const T*x,const T*station,T*r)const{
   T g[5];for(int i=0;i<5;++i)g[i]=T(scene[i]);
   const T w=T(width)*(T(1)+x[1]-x[0]),h=T(height)*(T(1)+x[3]-x[2]),s=T(origin)+T(width)*x[0],b=T(bottom)+T(height)*x[2];
   if(!(w>T(0)&&h>T(0)&&b>=T(0)&&b+h<=g[4]))return false;
   const T u=T(u0)+station[0]*T(du),v=T(v0)+station[0]*T(dv);const auto X=EventModel::board_point(s+w*u,b+h*v,g);Vec3<T>q;
   for(int i=0;i<3;++i){q[i]=T(0);for(int j=0;j<3;++j)q[i]+=T(R[i*3+j])*(X[j]-T(C[j]));}
   if(!(q[2]>T(0)))return false;const T a=q[0]/q[2],z=q[1]/q[2],rr=a*a+z*z,rad=T(1)+T(k1)*rr+T(k2)*rr*rr;
   const T dx=a*rad+T(2*p1)*a*z+T(p2)*(rr+T(2)*a*a),dy=z*rad+T(p1)*(rr+T(2)*z*z)+T(2*p2)*a*z;
   r[0]=T(scale)*(T(fx)*dx+T(cx-observed[0]));r[1]=T(scale)*(T(fy)*dy+T(cy-observed[1]));return true;
  }
 };
 struct Geometry{int block=-1;bool affine=false;Vec2<double>point{};};
 struct Group{int ad=-1;std::vector<std::pair<std::size_t,double>>rows;std::vector<Geometry>geometry;};
 JointALBuilder&b_;ceres::Problem&p_;int threads_;std::vector<Group>groups_;
 Pixel pixel(std::size_t i,double weight)const{
  const auto&m=b_.core.model;const auto&r=b_.core.packet.rows[i];const auto l=r.layout(m);const auto ptr=m.pointers(l);const auto g=m.global(l,ptr.data());std::vector<const double*>frozen;std::vector<double>geometry;
  const auto e=m.legacy_layout(l,frozen,g,ptr.data(),geometry);const auto&math=m.shots[l.shot_index].math;auto tr=math.trajectory(row_time(l.frame_time,g[31],b_.core.row[i]),frozen.data(),e);tr[0]+=ptr[l.reference_angles_slot][0];tr[1]+=ptr[l.reference_angles_slot][1];
  const auto optical=zoom_optics(math.optics(g.data()),tr[2]);const auto pose=camera_pose(math.head(g.data()),tr[0],tr[1],optical.z);if(!pose.valid())throw std::runtime_error("ad envelope camera invalid");
  const auto&path=m.materials[r.material].paths[r.path];const double s=b_.core.station[i];Vec2<double>uv=path.points.front();if(path.points.size()>1){const double z=s*(path.points.size()-1);const int k=std::min(int(path.points.size())-2,int(std::floor(z)));for(int axis=0;axis<2;++axis)uv[axis]=path.points[k][axis]*(1-(z-k))+path.points[k+1][axis]*(z-k);}
  const auto&a=m.blocks[path.advertisement_block].values;const int o=path.advertisement_offset;Pixel out;std::copy(g.begin(),g.begin()+5,out.scene.begin());out.R=pose.R;out.C=pose.C;out.fx=optical.intr.fx;out.fy=optical.intr.fy;out.cx=optical.intr.cx;out.cy=optical.intr.cy;out.k1=optical.brown.k1;out.k2=optical.brown.k2;out.p1=optical.brown.p1;out.p2=optical.brown.p2;out.origin=a[o];out.width=a[o+1];out.bottom=a[o+2];out.height=a[o+3]-a[o+2];out.u0=path.points.front()[0];out.v0=path.points.front()[1];out.du=path.points.back()[0]-out.u0;out.dv=path.points.back()[1]-out.v0;out.scale=std::sqrt(weight)/r.E;out.observed=r.uv;return out;
 }
 public:
 JointAdEnvelopeBlocksV1(JointALBuilder&b,ceres::Problem&p,const eventio::Json&ledger,int threads):b_(b),p_(p),threads_(threads){
  auto&m=b.core.model;std::map<int,Group>candidates;
  for(std::size_t i=0;i<m.materials.size();++i)for(const auto&path:m.materials[i].paths){if(path.kind!=2||path.advertisement_block<0||path.geometry_block<0)continue;auto&ab=m.blocks[path.advertisement_block];auto&gb=m.blocks[path.geometry_block];if(ab.values.size()!=8||gb.values.size()!=6||!p.HasParameterBlock(gb.values.data())||p.ParameterBlockTangentSize(gb.values.data())!=6||p.IsParameterBlockConstant(gb.values.data())||m.materials[i].label.find(".print.")==std::string::npos)continue;candidates[path.advertisement_block].ad=path.advertisement_block;}
  std::map<int,std::map<int,Geometry>>geometries;std::set<int>invalid;for(const auto&material:m.materials)for(const auto&path:material.paths)if(path.kind==6&&!path.vertex_blocks.empty()&&path.advertisement_block>=0)invalid.insert(path.advertisement_block);if(!invalid.empty())std::cout<<"Legacy compensated affine advertisement envelope blocks skipped for "<<invalid.size()<<" static vertex installations; full original camera/ad/vertex residuals and physical domains remain active"<<std::endl;
  for(std::size_t mi=0;mi<m.materials.size();++mi)for(std::size_t pi=0;pi<m.materials[mi].paths.size();++pi){const auto&path=m.materials[mi].paths[pi];const int ad=path.advertisement_block;if(!candidates.count(ad)||path.geometry_block<0)continue;auto&gb=m.blocks[path.geometry_block];if(path.kind!=2||gb.values.size()!=6||m.path_geometry_offsets.at({int(mi),int(pi)})||!m.path_geometry_indices.at({int(mi),int(pi)}).empty()||!p.HasParameterBlock(gb.values.data())||p.IsParameterBlockConstant(gb.values.data())){invalid.insert(ad);continue;}
   const bool full=p.ParameterBlockTangentSize(gb.values.data())==6;if(!full&&path.points.size()!=1){invalid.insert(ad);continue;}Geometry entry{path.geometry_block,full,path.points.front()};auto&seen=geometries[ad];auto it=seen.find(entry.block);if(it!=seen.end()&&(it->second.affine!=full||(!full&&it->second.point!=entry.point)))invalid.insert(ad);seen[entry.block]=entry;
  }
  // A compensated chart must not serve any other physical advertisement pair.
  for(const auto&material:m.materials)for(const auto&path:material.paths)if(path.geometry_block>=0)for(const auto&gg:geometries)if(gg.second.count(path.geometry_block)&&path.advertisement_block!=gg.first)invalid.insert(gg.first);
  for(const auto&entry:ledger.at("groups")){if(!entry.contains("actual_row_indices"))throw std::runtime_error("ad envelope requires actual row weights");const auto&ids=entry.at("actual_row_indices");const auto&weights=entry.at("actual_per_row_weight");for(std::size_t k=0;k<ids.size();++k){const auto i=ids[k].get<std::size_t>();const auto&r=b.core.packet.rows[i];const auto&path=m.materials[r.material].paths[r.path];auto it=candidates.find(path.advertisement_block);if(it!=candidates.end()&&path.geometry_block<0&&path.kind==2)it->second.rows.emplace_back(i,weights[k].get<double>());}}
  for(auto&entry:candidates){if(invalid.count(entry.first)||entry.second.rows.empty())continue;for(const auto&g:geometries[entry.first])entry.second.geometry.push_back(g.second);groups_.push_back(std::move(entry.second));}
 }
 bool minimize(const std::filesystem::path&file){
  if(groups_.empty())return false;const auto started=std::chrono::steady_clock::now();auto&m=b_.core.model;
  auto objective=[&](){if(auto*c=b_.gpu_evaluation_callback())c->PrepareForEvaluation(false,true);ceres::Problem::EvaluateOptions o;o.num_threads=threads_;double value=INFINITY;if(!p_.Evaluate(o,&value,nullptr,nullptr,nullptr))return double(INFINITY);return value;};
  const double before=objective();std::map<int,std::vector<double>>backup;std::map<std::size_t,std::pair<double,double>>station_backup;eventio::Json reports=eventio::Json::array();bool changed=false;double preservation=0;
  for(const auto&group:groups_){auto&ad=m.blocks[group.ad];const auto old=ad.values;if(std::abs(old[1]-old[5])>1e-12||std::abs((old[3]-old[2])-(old[7]-old[6]))>1e-12)throw std::runtime_error("paired envelope dimensions differ");
   ceres::Problem mini;double x[4]={0,0,0,0};std::vector<double>stations(group.rows.size());auto ordering=std::make_shared<ceres::ParameterBlockOrdering>();std::size_t cursor=0;for(const auto&r:group.rows){double*st=&stations[cursor++];*st=b_.core.station[r.first];const auto f=pixel(r.first,r.second);double rr[2];if(!f(x,st,rr))throw std::runtime_error("ad envelope initial projection invalid");const auto&source=b_.core.packet.rows[r.first];const auto l=source.layout(m);const auto ptr=m.pointers(l);const auto q=m.predict_for_al(l,b_.core.station[r.first],b_.core.row[r.first],ptr.data());if(!q.arithmetic_valid||std::max(std::abs(rr[0]/f.scale+source.uv[0]-q.uv[0]),std::abs(rr[1]/f.scale+source.uv[1]-q.uv[1]))>1e-7)throw std::runtime_error("ad envelope differs from actual native projection");mini.AddResidualBlock(new ceres::AutoDiffCostFunction<Pixel,2,4,1>(new Pixel(f)),nullptr,x,st);if(source.kind==1||!p_.HasParameterBlock(&b_.core.station[r.first])||p_.IsParameterBlockConstant(&b_.core.station[r.first]))mini.SetParameterBlockConstant(st);else{double lo=p_.GetParameterLowerBound(&b_.core.station[r.first],0),hi=p_.GetParameterUpperBound(&b_.core.station[r.first],0);bool interval_found=false;for(const auto&interval:source.station_intervals(m,source.path))if(*st>=interval[0]&&*st<=interval[1]){lo=std::max(lo,interval[0]);hi=std::min(hi,interval[1]);interval_found=true;break;}if(!interval_found)throw std::runtime_error("ad envelope station outside existing source support");mini.SetParameterLowerBound(st,0,lo);mini.SetParameterUpperBound(st,0,hi);ordering->AddElementToGroup(st,0);}}
   // Independent edge displacements make the existing point-inside-panel
   // constraints scalar bounds, so a blocked left edge does not discard a
   // valid right-edge improvement. No internal point or physical bound moves.
   std::array<double,4>lo{-INFINITY,-INFINITY,-INFINITY,-INFINITY},hi{INFINITY,INFINITY,INFINITY,INFINITY};
   const double W=old[1],H=old[3]-old[2];
   for(int o:{0,4}){lo[0]=std::max(lo[0],(p_.GetParameterLowerBound(ad.values.data(),o)-old[o])/W);hi[0]=std::min(hi[0],(p_.GetParameterUpperBound(ad.values.data(),o)-old[o])/W);lo[2]=std::max(lo[2],(p_.GetParameterLowerBound(ad.values.data(),o+2)-old[o+2])/H);hi[2]=std::min(hi[2],(p_.GetParameterUpperBound(ad.values.data(),o+2)-old[o+2])/H);lo[3]=std::max(lo[3],(p_.GetParameterLowerBound(ad.values.data(),o+3)-old[o+3])/H);hi[3]=std::min(hi[3],(std::min(p_.GetParameterUpperBound(ad.values.data(),o+3),m.blocks[m.scene_block].values[4])-old[o+3])/H);}
   for(const auto&geometry:group.geometry)if(!geometry.affine&&geometry.point==Vec2<double>{0.,0.}){const auto&q=m.blocks[geometry.block].values;for(int axis=0;axis<2;++axis){const int c=axis?5:2,L=axis?2:0,R=L+1;if(p_.GetParameterLowerBound(q.data(),c)==0.&&p_.GetParameterUpperBound(q.data(),c)==1.){hi[L]=std::min(hi[L],q[c]);lo[R]=std::max(lo[R],q[c]-1.);}}}
   for(int j=0;j<4;++j){if(lo[j]>0.||hi[j]<0.)throw std::runtime_error("ad envelope initial edge coordinates outside original bounds");if(std::isfinite(lo[j]))mini.SetParameterLowerBound(x,j,lo[j]);if(std::isfinite(hi[j]))mini.SetParameterUpperBound(x,j,hi[j]);}
   ceres::Solver::Options options;options.linear_solver_type=ceres::DENSE_SCHUR;ordering->AddElementToGroup(x,1);options.linear_solver_ordering=ordering;options.num_threads=threads_;options.function_tolerance=1e-10;options.gradient_tolerance=1e-10;options.parameter_tolerance=1e-10;ceres::Solver::Summary summary;ceres::Solve(options,&mini,&summary);if(!summary.IsSolutionUsable()||!(summary.final_cost<summary.initial_cost)){reports.push_back({{"ad_block",group.ad},{"name",ad.name},{"updated",false},{"reason","mini solve has no usable decrease"},{"summary",summary.BriefReport()}});continue;}
   auto proposed=old;const double width=old[1]*(1.+x[1]-x[0]),height=(old[3]-old[2])*(1.+x[3]-x[2]),du=old[1]*x[0],dv=(old[3]-old[2])*x[2];for(int o:{0,4}){proposed[o]=old[o]+du;proposed[o+1]=width;proposed[o+2]=old[o+2]+dv;proposed[o+3]=proposed[o+2]+height;}
   std::map<int,std::vector<double>>updates;updates[group.ad]=proposed;
   for(const auto&geometry:group.geometry){const auto&q=m.blocks[geometry.block].values;auto next=q;const double sx=old[1]/width,sy=(old[3]-old[2])/height;if(geometry.affine){next[0]=sx*q[0];next[1]=sx*q[1];next[2]=sx*q[2]-du/width;next[3]=sy*q[3];next[4]=sy*q[4];next[5]=sy*q[5]-dv/height;}else{const double u=q[0]*geometry.point[0]+q[1]*geometry.point[1]+q[2],v=q[3]*geometry.point[0]+q[4]*geometry.point[1]+q[5];next[2]=sx*u-du/width-q[0]*geometry.point[0]-q[1]*geometry.point[1];next[5]=sy*v-dv/height-q[3]*geometry.point[0]-q[4]*geometry.point[1];}
    for(Vec2<double>uv:{geometry.point,Vec2<double>{0,0},Vec2<double>{1,1}}){if(!geometry.affine&&uv!=geometry.point)continue;const double a=old[0]+old[1]*(q[0]*uv[0]+q[1]*uv[1]+q[2]),b=proposed[0]+width*(next[0]*uv[0]+next[1]*uv[1]+next[2]),c=old[2]+(old[3]-old[2])*(q[3]*uv[0]+q[4]*uv[1]+q[5]),d=proposed[2]+height*(next[3]*uv[0]+next[4]*uv[1]+next[5]);preservation=std::max({preservation,std::abs(a-b),std::abs(c-d)});}updates[geometry.block]=std::move(next);
   }
   bool legal=true;for(const auto&u:updates){const auto&block=m.blocks[u.first];for(std::size_t j=0;j<u.second.size();++j)legal&=std::isfinite(u.second[j])&&u.second[j]>=p_.GetParameterLowerBound(block.values.data(),int(j))&&u.second[j]<=p_.GetParameterUpperBound(block.values.data(),int(j));}if(!legal){reports.push_back({{"ad_block",group.ad},{"name",ad.name},{"updated",false},{"reason","compensated coordinate would leave original scalar bounds"},{"coordinate_update",{x[0],x[1],x[2],x[3]}}});continue;}if(preservation>2e-12)throw std::runtime_error("ad envelope compensation moved the existing print");
   for(const auto&u:updates){auto&v=m.blocks[u.first].values;backup.emplace(u.first,v);std::copy(u.second.begin(),u.second.end(),v.begin());}for(std::size_t k=0;k<group.rows.size();++k){const auto i=group.rows[k].first;station_backup.emplace(i,std::make_pair(b_.core.station[i],b_.core.packet.rows[i].station));b_.core.station[i]=b_.core.packet.rows[i].station=stations[k];}changed=true;reports.push_back({{"ad_block",group.ad},{"name",ad.name},{"exterior_rows",group.rows.size()},{"compensated_geometry_blocks",group.geometry.size()},{"edge_cost_before",summary.initial_cost},{"edge_cost_after",summary.final_cost},{"coordinate_update",{x[0],x[1],x[2],x[3]}}});
  }
  const double after=changed?objective():before;const double tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(before));const bool accepted=changed&&std::isfinite(after)&&after<before-tol;if(!accepted&&changed){for(const auto&u:backup)std::copy(u.second.begin(),u.second.end(),m.blocks[u.first].values.begin());for(const auto&s:station_backup){b_.core.station[s.first]=s.second.first;b_.core.packet.rows[s.first].station=s.second.second;}objective();}
  eventio::atomic_json(file,{{"schema","rinkfit.actual_ad_envelope_block_minimization.v1"},{"accepted",accepted},{"groups",reports},{"before",before},{"proposed",std::isfinite(after)?eventio::Json(after):eventio::Json()},{"maximum_print_unfolded_world_change",preservation},{"source_pixels_weights_ID_PTS_changed",false},{"full_original_problem_acceptance",true},{"new_parameters_or_fixed_physical_coordinates",false},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}});
  std::cout<<"Advertisement envelopes: actual objective "<<before<<" -> "<<after<<(accepted?" accepted, all existing print world positions preserved":" unchanged/restored")<<std::endl;return accepted;
 }
};
}
