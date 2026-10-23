#pragma once
#include "rinkfit/joint_model.hpp"
#include <memory>
#include <set>
#include <utility>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <mutex>
#include <cstring>
namespace rinkfit {
struct JointMotionObservationV1 {int shot=0;std::int64_t pts0=0,pts1=0;Vec2<double> uv0{},uv1{};std::array<double,2> latent{};std::size_t pair=0,index=0;double scale0=1,scale1=1;};
struct JointMotionStoreV1 {
 eventio::Json manifest;std::string hash;std::vector<JointMotionObservationV1> rows;
 struct Dependency {std::filesystem::path path;std::string sha;std::uintmax_t size;std::filesystem::file_time_type time;};
 std::map<std::string,Dependency> dependencies;
 void assert_dependencies()const{for(const auto& item:dependencies){const auto& d=item.second;if(!std::filesystem::exists(d.path)||d.size!=std::filesystem::file_size(d.path)||d.time!=std::filesystem::last_write_time(d.path))throw std::runtime_error("motion source dependency changed: "+d.path.u8string());}}
 using Exposure=std::pair<std::string,std::int64_t>;
 std::map<Exposure,std::size_t> incidence(const JointEventModel&m)const{std::map<Exposure,std::size_t> a;for(const auto&r:rows){++a[{m.shots.at(r.shot).id,r.pts0}];++a[{m.shots.at(r.shot).id,r.pts1}];}return a;}
 static std::shared_ptr<JointMotionStoreV1> load(const eventio::Json&ref,const JointEventModel&m){
  auto require=[](bool c,const std::string&msg){if(!c)throw std::runtime_error("static motion: "+msg);};
  auto s=std::make_shared<JointMotionStoreV1>();
  auto pin=[&](const eventio::Json&r){auto p=std::filesystem::u8path(r.at("file").get<std::string>());if(!p.is_absolute())throw std::runtime_error("motion references must be absolute");const auto sha=r.at("sha256").get<std::string>();const auto size=std::filesystem::file_size(p);const auto time=std::filesystem::last_write_time(p);const auto key=p.lexically_normal().u8string();auto found=s->dependencies.find(key);if(found!=s->dependencies.end()){if(found->second.sha!=sha||found->second.size!=size||found->second.time!=time)throw std::runtime_error("motion conflicting or modified source");return p;}eventio::verify(p,sha);if(size!=std::filesystem::file_size(p)||time!=std::filesystem::last_write_time(p))throw std::runtime_error("motion source changed during SHA validation");s->dependencies.emplace(key,Dependency{p,sha,size,time});return p;};
  s->manifest=eventio::read_json(pin(ref));s->hash=ref.at("sha256");const auto&j=s->manifest;
  require(j.at("schema")=="rinkfit.static_surface_motion_packet.v2"&&j.at("latent_coordinates")=="board_arclength_and_height_fraction"&&j.at("surface")=="board_inner"&&j.at("world_truth_claimed")==false&&j.at("all_selected_sources_retained")==true,"unsupported source/surface contract");
  const auto roles=eventio::read_json(pin(j.at("frame_role_authority")));require(roles.at("schema")=="rinkfit.joint_source_frame_protection.v1"&&roles.at("source_video_sha256")==m.source_json.at("source_clock").at("source_video_sha256")&&j.at("all_fitted_motion_exposures_pass_original_role_guard")==true,"original exposure-role authority required");
  for(const auto&r:j.at("evidence_files"))pin(r);
  const auto bytes=eventio::bytes(pin(j.at("table")));const std::size_t n=j.at("selected_matches");require(bytes.size()==n*12*sizeof(double),"complete table length mismatch");
  s->rows.resize(n);std::set<std::pair<std::size_t,std::size_t>>seen;std::size_t cursor=0;
  for(const auto&pair:j.at("pairs")){auto raw=eventio::read_json(pin(pair.at("source")));const auto selected=pair.at("selected_indices").get<std::vector<std::size_t>>();
   require(raw.at("source_world_projection_used")==false&&raw.at("source_or_current_UV_not_moved")==true&&raw.at("cross_cut")==false&&raw.at("complete_21x21_patch_required_in_both_real_exposure_domains")==true,"source pair not actual static pixels");
   const int shot=m.shot_map.at(raw.at("shot_id").get<std::string>());const auto p0=raw.at("previous_native_pts").get<std::int64_t>(),p1=raw.at("native_pts").get<std::int64_t>();
   for(auto pts:{p0,p1}){const auto& role=roles.at("frame_roles").at(std::to_string(pts));const auto name=role.at("role").get<std::string>();require(role.at("shot_id")==m.shots.at(shot).id&&name!="reserved_pending_lineage"&&name!="validation_guard"&&name!="sealed_validation","protected/unresolved or cross-shot exposure cannot be motion training data");}
   require(p1-p0==256&&p0>=m.shots[shot].first_pts&&p1<=m.shots[shot].last_pts,"clock/cut mismatch");
   for(const auto ix:selected){require(cursor<n&&seen.emplace(pair.at("pair_index").get<std::size_t>(),ix).second,"duplicate or extra pair index");double v[12];std::memcpy(v,bytes.data()+cursor*12*sizeof(double),sizeof(v));for(double x:v)require(std::isfinite(x),"nonfinite motion row");
    auto&r=s->rows[cursor++];r.shot=shot;r.pts0=p0;r.pts1=p1;r.pair=pair.at("pair_index");r.index=ix;r.uv0=raw.at("source_uv").at(ix).get<Vec2<double>>();r.uv1=raw.at("current_uv").at(ix).get<Vec2<double>>();r.latent={v[7],v[8]};require(v[8]>=0.&&v[8]<=1.,"height fraction outside unit interval");
    require(v[0]==shot&&v[1]==double(p0)&&v[2]==double(p1)&&v[3]==r.uv0[0]&&v[4]==r.uv0[1]&&v[5]==r.uv1[0]&&v[6]==r.uv1[1]&&v[9]==double(r.pair)&&v[10]==double(ix)&&v[11]==1.,"raw UV/PTS/index/table mismatch");
   }
  }require(cursor==n,"missing selected motion rows");return s;
 }
};
struct JointMotionFactorV1 {
 const JointEventModel*model;JointRowLayout a,b;std::vector<int>ma,mb,sizes;Vec2<double> u,v;int latent_slot=0;double scale0=1,scale1=1;bool true_rs=false;
 struct Cache {std::mutex mutex;std::vector<double>key;std::array<double,2>row{},den{};bool valid=false,has=false;};std::shared_ptr<Cache>cache=std::make_shared<Cache>();
 template<class T>bool raw(const JointRowLayout&l,const std::vector<int>&map,T const*const*pars,T row,Vec2<T>*uv)const{
  std::vector<const T*>p(map.size(),nullptr);for(std::size_t k=0;k<map.size();++k)if(map[k]>=0)p[k]=pars[map[k]];
  const auto g=model->global(l,p.data());const auto*z=pars[latent_slot];if(event_scalar(z[1])<0||event_scalar(z[1])>1.)return false;
  const auto X=EventModel::board_point(z[0],z[1]*g[4],g.data());std::vector<const T*>ep;std::vector<T>geometry;const auto e=model->legacy_layout(l,ep,g,p.data(),geometry);const auto&math=model->shots.at(l.shot_index).math;
  auto q=math.trajectory(row_time(T(l.frame_time),g[31],row),ep.data(),e);q[0]+=p[l.reference_angles_slot][0];q[1]+=p[l.reference_angles_slot][1];const auto o=zoom_optics(math.optics(g.data()),q[2]);const auto pose=camera_pose(math.head(g.data()),q[0],q[1],o.z);
  if(!pose.valid()||!valid_rotation(pose.R))return false;const auto zproj=project(X,pose,o.intr,o.brown);if(!zproj.valid()||!joint_visibility(g.data(),pose.C,X,true).visible())return false;*uv=zproj.uv;return finite_array(*uv);
 }
 bool roots(const double*const*pars,std::array<double,2>&roots,std::array<double,2>&den)const{
  const JointRowLayout*ls[2]={&a,&b};const std::vector<int>*maps[2]={&ma,&mb};const Vec2<double>obs[2]={u,v};
  for(int side=0;side<2;++side){const auto&l=*ls[side];const auto&map=*maps[side];std::vector<const double*>p(map.size(),nullptr);for(std::size_t k=0;k<map.size();++k)if(map[k]>=0)p[k]=pars[map[k]];auto g=model->global(l,p.data());EventModel e=model->shots[l.shot_index].math;
   const auto h=e.head(g.data());e.R0_seed=matmul(matmul(rodrigues(h.b,p[l.reference_angles_slot][1]),h.R0),rodrigues(h.n,p[l.reference_angles_slot][0]));e.n_seed=h.n;e.b_seed=h.b;const Vec3<double>axis=std::abs(h.n[0])<.8?Vec3<double>{1,0,0}:Vec3<double>{0,1,0};e.n_tangent1=event_normalized(cross(h.n,axis));e.n_tangent2=cross(h.n,e.n_tangent1);for(int k=5;k<=10;++k)g[k]=0;
   e.blocks.clear();e.blocks.resize(e.n_controls+1);e.control_block_begin=1;e.blocks[0].values.assign(g.begin(),g.end());for(int k=0;k<e.n_controls;++k)e.blocks[k+1].values={NAN,NAN,NAN};
   for(std::size_t k=0;k<l.control_ids.size();++k){const auto*x=p[l.control_slots[k]]+l.control_offsets[k];e.blocks[1+l.control_ids[k]].values={x[0],x[1],x[2]};}
   const auto*lat=pars[latent_slot];const auto X=EventModel::board_point(lat[0],lat[1]*g[4],g.data());const auto root=e.project_world_rs(X,l.frame_time,obs[side][1]);if(!root.valid||!root.unique)return false;roots[side]=root.row;
   using K=ceres::Jet<double,1>;std::vector<std::vector<K>>store(sizes.size());std::vector<const K*>kp;for(std::size_t i=0;i<sizes.size();++i){for(int j=0;j<sizes[i];++j)store[i].emplace_back(pars[i][j]);kp.push_back(store[i].data());}K rw(root.row);rw.v[0]=1;Vec2<K>q;if(!raw(l,map,kp.data(),rw,&q))return false;den[side]=1-q[1].v[0];if(!std::isfinite(den[side])||std::abs(den[side])<1e-6||std::abs(q[1].a-root.row)>1e-7)return false;
  }return true;
 }
 template<class T>bool operator()(T const*const*p,T*r)const{
  try{Vec2<T>x,y;if(!true_rs){if(!raw(a,ma,p,T(u[1]),&x)||!raw(b,mb,p,T(v[1]),&y))return false;}else{
   std::vector<double>key;std::vector<std::vector<double>>store(sizes.size());std::vector<const double*>ptr;for(std::size_t i=0;i<sizes.size();++i){for(int j=0;j<sizes[i];++j){const double q=event_scalar(p[i][j]);key.push_back(q);store[i].push_back(q);}ptr.push_back(store[i].data());}
   std::array<double,2>rw,den;{std::lock_guard<std::mutex>lock(cache->mutex);if(!cache->has||cache->key!=key){cache->valid=roots(ptr.data(),cache->row,cache->den);cache->key=key;cache->has=true;}if(!cache->valid)return false;rw=cache->row;den=cache->den;}
   Vec2<T>xa,ya;if(!raw(a,ma,p,T(rw[0]),&xa)||!raw(b,mb,p,T(rw[1]),&ya))return false;
   const T ra=T(rw[0])+(xa[1]-T(event_scalar(xa[1])))/T(den[0]),rb=T(rw[1])+(ya[1]-T(event_scalar(ya[1])))/T(den[1]);
   if(!raw(a,ma,p,ra,&x)||!raw(b,mb,p,rb,&y))return false;
  }for(int k=0;k<2;++k){r[k]=T(scale0)*(x[k]-T(u[k]));r[k+2]=T(scale1)*(y[k]-T(v[k]));}return true;}catch(...){return false;}
 }
};
inline std::pair<ceres::CostFunction*,std::vector<double*>> make_joint_motion_factor_v1(JointEventModel&m,JointMotionObservationV1&r,bool true_rs){
 auto a=m.layout(r.shot,0,0,r.pts0),b=m.layout(r.shot,0,0,r.pts1);a.geometry_slot=a.advertisement_slot=a.width_slot=-1;b.geometry_slot=b.advertisement_slot=b.width_slot=-1;
 auto*f=new JointMotionFactorV1;f->model=&m;f->a=a;f->b=b;f->u=r.uv0;f->v=r.uv1;f->scale0=r.scale0;f->scale1=r.scale1;f->true_rs=true_rs;std::map<int,int>slots;std::vector<double*>p;
 auto bind=[&](const JointRowLayout&l,std::vector<int>&map){map.assign(l.block_ids.size(),-1);std::vector<int>required={l.scene_slot,l.rig_slot,l.optics_slot,l.reference_angles_slot};required.insert(required.end(),l.control_slots.begin(),l.control_slots.end());for(int k:required){int id=l.block_ids.at(k);if(!slots.count(id)){slots[id]=int(p.size());p.push_back(m.blocks[id].values.data());f->sizes.push_back(int(m.blocks[id].values.size()));}map[k]=slots.at(id);}};bind(a,f->ma);bind(b,f->mb);f->latent_slot=int(p.size());p.push_back(r.latent.data());f->sizes.push_back(2);
 auto*cost=new ceres::DynamicAutoDiffCostFunction<JointMotionFactorV1,64>(f);for(int size:f->sizes)cost->AddParameterBlock(size);cost->SetNumResiduals(4);return {cost,std::move(p)};
}
inline eventio::Json initialize_joint_motion_domain_v1(JointMotionStoreV1&s,JointEventModel&m,bool true_rs){
 eventio::Json changed=eventio::Json::array();
 for(std::size_t i=0;i<s.rows.size();++i){auto&r=s.rows[i];auto f=make_joint_motion_factor_v1(m,r,true_rs);std::unique_ptr<ceres::CostFunction>cost(f.first);std::vector<const double*>p(f.second.begin(),f.second.end());double residual[4];if(cost->Evaluate(p.data(),residual,nullptr))continue;
  const auto old=r.latent;bool valid=false;for(double fraction:{.25,.5,.75,1.}){r.latent[1]=old[1]+fraction*(1.-old[1]);if(cost->Evaluate(p.data(),residual,nullptr)){valid=true;break;}}
  if(!valid){r.latent=old;throw std::runtime_error("no feasible initial shared surface latent at motion row "+std::to_string(i)+"; all source pairs retained");}
  changed.push_back({{"motion_row",i},{"old_surface_seed",old},{"new_surface_seed",r.latent}});
 }
 return {{"schema","rinkfit.motion_numerical_surface_domain_initialization.v1"},{"rows",s.rows.size()},{"changed_latents",changed},{"changed_count",changed.size()},{"all_source_UV_and_admission_unchanged",true},{"source_world_coordinates_issued",false},{"true_RS",true_rs}};
}

}
