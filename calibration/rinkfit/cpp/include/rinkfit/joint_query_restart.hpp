#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_path_generation_v1.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

namespace rinkfit {
// Numerical proposals only. No live Ceres graph or GPU layout may survive an
// apply/rollback: its captured JointRowLayout can contain the old finite chart.
struct JointQueryRestartRow {
  std::size_t row_index=0;
  int old_path=0,path=0;
  double old_station=0,old_row=0,old_alpha=0;
  double old_source_station=0,old_source_row=0,old_source_alpha=0;
  double station=0,row=0,alpha=0;
  double old_navigation_score=INFINITY,navigation_score=INFINITY;
  double query_error_px=INFINITY,pixel_error_px=INFINITY,RS_equality_px=INFINITY;
};
struct JointQueryRestartPlan {
  std::string source_packet_sha256,physical_values_token;
  std::size_t raw_rows=0,eligible_queries=0,searched_queries=0,protected_queries=0;
  std::vector<std::string> protected_shots;
  std::vector<JointQueryRestartRow> rows;
  double elapsed_seconds=0;
  bool applied=false;
  std::size_t path_changes()const {std::size_t n=0;for(const auto&r:rows)n+=r.path!=r.old_path;return n;}
};
namespace joint_query_restart_detail {
inline bool same(double a,double b){return std::memcmp(&a,&b,sizeof(double))==0;}
// A stale-plan guard, not a source-qualification certificate.
inline std::string physical_token(const JointEventModel&m){
  std::uint64_t h=1469598103934665603ull;
  auto add=[&](const void*p,std::size_t n){const auto*x=static_cast<const unsigned char*>(p);for(std::size_t i=0;i<n;++i){h^=x[i];h*=1099511628211ull;}};
  for(const auto&b:m.blocks){add(b.name.data(),b.name.size());const auto n=b.values.size();add(&n,sizeof(n));add(b.values.data(),n*sizeof(double));}
  std::ostringstream s;s<<std::hex<<h;return s.str();
}
inline int family(const std::string&label){return label.rfind("board.",0)==0?0:(label.rfind("Ad",0)==0||label.rfind("advertisement",0)==0?1:2);}
struct AlphaResult {double alpha=0,distance_sq=INFINITY;};
inline AlphaResult nearest_alpha(const JointQuery&q,const Vec2<double>&target,double old_alpha){
  if(q.alpha.size()<2||!finite_array(target))return {};
  const double lower=q.alpha.front(),upper=q.alpha.back();
  auto value=[&](double a){Vec2<double>p;if(!q.point(a,&p))return double(INFINITY);const double u=p[0]-target[0],v=p[1]-target[1];return u*u+v*v;};
  AlphaResult best;best.alpha=std::max(lower,std::min(upper,old_alpha));best.distance_sq=value(best.alpha);
  if(q.den.empty()){
    // Exact closest point on every segment of an original piecewise Query.
    for(std::size_t i=1;i<q.alpha.size();++i){const auto&a=q.uv[i-1];const auto&b=q.uv[i];const double u=b[0]-a[0],v=b[1]-a[1],d=u*u+v*v;
      const double t=d>0?std::max(0.,std::min(1.,((target[0]-a[0])*u+(target[1]-a[1])*v)/d)):0.;
      const double alpha=q.alpha[i-1]+t*(q.alpha[i]-q.alpha[i-1]),score=value(alpha);if(score<best.distance_sq)best={alpha,score};
    }
    return best;
  }
  std::vector<double>grid=q.alpha;for(int k=0;k<=16;++k)grid.push_back(lower+(upper-lower)*k/16.);grid.push_back(best.alpha);
  std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
  std::vector<double>scores; scores.reserve(grid.size());for(double a:grid)scores.push_back(value(a));
  for(std::size_t i=0;i<grid.size();++i){
    if(scores[i]<best.distance_sq)best={grid[i],scores[i]};
    if((i&&scores[i]>scores[i-1])||(i+1<grid.size()&&scores[i]>scores[i+1]))continue;
    double lo=grid[i?i-1:0],hi=grid[std::min(i+1,grid.size()-1)];
    for(int k=0;k<24&&hi>lo;++k){const double a=lo+(hi-lo)*.3819660112501051,b=lo+(hi-lo)*.6180339887498949;if(value(a)<value(b))hi=b;else lo=a;}
    const double a=(lo+hi)*.5,s=value(a);if(s<best.distance_sq)best={a,s};
  }
  return best;
}
inline std::vector<double> station_grid(const JointEventModel&m,const JointRowLayout&layout,const EventPath&path,const std::array<double,2>&interval,double old_station){
  std::vector<double>grid{interval[0],interval[1]};for(int k=0;k<=32;++k)grid.push_back(interval[0]+(interval[1]-interval[0])*k/32.);
  if(old_station>=interval[0]&&old_station<=interval[1])grid.push_back(old_station);
  std::size_t segments=0;if(path.kind==2&&path.points.size()>1)segments=path.closed?path.points.size():path.points.size()-1;
  if(path.kind==6){const auto n=layout.geometry_indices.empty()?m.blocks.at(path.geometry_block).values.size()-layout.geometry_offset:layout.geometry_indices.size();const auto vertices=n/2;if(vertices>1)segments=path.closed?vertices:vertices-1;}
  if(segments)for(std::size_t k=0;k<=segments;++k){const double s=double(k)/segments;if(s>=interval[0]&&s<=interval[1])grid.push_back(s);}
  std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());return grid;
}
inline void check_plan(const JointALBuilder&b,const JointQueryRestartPlan&p,bool proposed){
  if(b.core.packet.source_hash!=p.source_packet_sha256||b.core.packet.rows.size()!=p.raw_rows||physical_token(b.core.model)!=p.physical_values_token)throw std::runtime_error("Query restart plan source/model changed");
  for(const auto&c:p.rows){const auto&r=b.core.packet.rows.at(c.row_index);
    if(std::binary_search(p.protected_shots.begin(),p.protected_shots.end(),r.shot_id))throw std::runtime_error("Query restart would change a protected shot");
    if(!r.active||!r.query||r.path!=(proposed?c.path:c.old_path)||
       !same(b.core.station[c.row_index],proposed?c.station:c.old_station)||!same(b.core.row[c.row_index],proposed?c.row:c.old_row)||!same(b.core.alpha[c.row_index],proposed?c.alpha:c.old_alpha)||
       !same(r.station,proposed?(r.kind==1?c.old_source_station:c.station):c.old_source_station)||!same(r.row,proposed?c.row:c.old_source_row)||!same(r.alpha,proposed?c.alpha:c.old_source_alpha))throw std::runtime_error("Query restart row changed after proposal");
  }
}
} // namespace joint_query_restart_detail

// cfg must contain the CURRENT outer rho in physical_constraint_penalty, and
// the same pixel_total_weight as build_engineering_problem. The navigation
// score includes original per-row engineering pixel and current Query dual/PHR
// terms. It is not a replacement for evaluating the complete current graph.
inline JointQueryRestartPlan propose_joint_query_restart(
    JointALBuilder&b,const eventio::Json&cfg,const std::vector<double>&equality_nu,
    const std::filesystem::path&out,int threads){
  using namespace joint_query_restart_detail;
  const auto started=std::chrono::steady_clock::now();JointQueryRestartPlan plan;
  plan.source_packet_sha256=b.core.packet.source_hash;plan.raw_rows=b.core.packet.rows.size();plan.physical_values_token=physical_token(b.core.model);
  if(!equality_nu.empty()&&equality_nu.size()!=b.schema().equality_ids.size())throw std::runtime_error("Query restart equality dual schema mismatch");
  const double rho=cfg.value("physical_constraint_penalty",10.);if(!(rho>0&&std::isfinite(rho)))throw std::runtime_error("invalid current Query restart rho");
  using Key=std::tuple<int,int,std::int64_t>;std::map<Key,std::size_t>counts;std::map<std::pair<int,int>,std::size_t>exposures;std::map<int,std::set<int>>families;
  std::size_t active=0;for(const auto&r:b.core.packet.rows)if(r.active){++active;const int shot=b.core.model.shot_map.at(r.shot_id),f=family(b.core.model.materials.at(r.material).label);if(++counts[{shot,f,r.pts}]==1)++exposures[{shot,f}];families[shot].insert(f);}
  const double total_weight=cfg.value("pixel_total_weight",double(active));if(!(total_weight>0&&std::isfinite(total_weight)))throw std::runtime_error("invalid Query restart pixel weight");
  // A restart changes auxiliaries outside Ceres. Parameter-block constants alone
  // cannot protect a shot here, so honor the same explicit production scope.
  const auto protected_list=cfg.value("temporary_runtime_fixed_nuisance_shots",std::vector<std::string>{});
  const std::set<std::string>protected_shots(protected_list.begin(),protected_list.end());
  for(const auto&sid:protected_shots)if(!b.core.model.shot_map.count(sid))throw std::runtime_error("Query restart names an unknown protected shot");
  plan.protected_shots.assign(protected_shots.begin(),protected_shots.end());
  struct Work{std::size_t i;double weight,nu_RS,nu_u,nu_v;};std::vector<Work>work;std::size_t equality=0;
  for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;const auto RS=equality++;
    if(r.query){const int shot=b.core.model.shot_map.at(r.shot_id),f=family(b.core.model.materials.at(r.material).label);
      const double weight=total_weight*(1./b.core.model.shots.size())*(1./families.at(shot).size())*(1./exposures.at({shot,f}))/counts.at({shot,f,r.pts});
      if(protected_shots.count(r.shot_id))++plan.protected_queries;
      else work.push_back({i,weight,equality_nu.empty()?0:equality_nu.at(RS),equality_nu.empty()?0:equality_nu.at(equality),equality_nu.empty()?0:equality_nu.at(equality+1)});
      equality+=2;
    }
  }
  if(equality!=b.schema().equality_ids.size())throw std::runtime_error("Query restart original equality indexing differs");
  plan.eligible_queries=work.size();std::atomic<std::size_t>next{0},searched{0};std::atomic<bool>failed{false};std::mutex mutex;std::string error;
  auto worker=[&](){try{for(;;){const auto wi=next.fetch_add(1);if(wi>=work.size())return;const auto&w=work[wi];const auto i=w.i;const auto&r=b.core.packet.rows[i];
    if(!(r.E>0)||r.query->alpha.size()<2)throw std::runtime_error("invalid original Query support");
    const double old_station=b.core.station[i],old_row=b.core.row[i],old_alpha=b.core.alpha[i];const auto old_layout=r.layout(b.core.model);const auto old_pointers=b.core.model.pointers(old_layout);
    const double source_station=r.kind==1?r.station:old_station;const auto current=b.core.model.predict_for_al(old_layout,source_station,old_row,old_pointers.data());
    Vec2<double>old_q;const bool old_q_valid=r.query->point(old_alpha,&old_q);const double old_query=current.arithmetic_valid&&old_q_valid?std::hypot(current.uv[0]-old_q[0],current.uv[1]-old_q[1]):INFINITY;
    const double old_pixel=current.arithmetic_valid?std::abs(r.normal[0]*(current.uv[0]-r.uv[0])+r.normal[1]*(current.uv[1]-r.uv[1])):INFINITY;
    const double old_RS=current.arithmetic_valid?old_row-current.uv[1]:INFINITY;
    bool at_endpoint=false;for(const auto&v:r.station_intervals(b.core.model,r.path))at_endpoint=at_endpoint||source_station==v[0]||source_station==v[1];
    if(current.physical_valid()&&r.station_allowed(b.core.model,r.path,source_station,1e-12)&&old_q_valid&&old_query<=r.E&&old_pixel<=r.E&&std::abs(old_RS)<=r.E&&!at_endpoint)continue;
    ++searched;
    auto score=[&](const Vec2<double>&px,double alpha,double RS){Vec2<double>q;if(!r.query->point(alpha,&q)||!finite_array(px))return double(INFINITY);
      const double e=r.normal[0]*(px[0]-r.uv[0])+r.normal[1]*(px[1]-r.uv[1]),u=px[0]-q[0],v=px[1]-q[1];return .5*w.weight*(e/r.E)*(e/r.E)+w.nu_u*u+w.nu_v*v+.5*rho*(u*u+v*v)+w.nu_RS*RS+.5*rho*RS*RS;};
    const double old_score=current.arithmetic_valid&&old_q_valid?score(current.uv,old_alpha,old_RS):INFINITY;
    std::vector<int>paths{r.path};const auto&material=b.core.model.materials.at(r.material);
    if((material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty()){
      if(material.paths.size()!=8)throw std::runtime_error("whole-board Query restart requires original eight finite charts");
      const bool original_domain=b.core.model.station_intervals(r.material,r.path)!=std::vector<std::array<double,2>>{{0.,1.}};
      paths.clear();for(std::size_t k=0;k<material.paths.size();++k){const auto&p=material.paths[k];if(p.kind!=3||p.physical_point)throw std::runtime_error("whole-board Query restart chart kind changed");
        if((b.core.model.station_intervals(r.material,int(k))!=std::vector<std::array<double,2>>{{0.,1.}})!=original_domain)throw std::runtime_error("whole-board Query restart would change native-domain schema");paths.push_back(int(k));}
    }
    JointQueryRestartRow chosen;chosen.row_index=i;chosen.old_path=chosen.path=r.path;chosen.old_station=chosen.station=old_station;chosen.old_row=chosen.row=old_row;chosen.old_alpha=chosen.alpha=old_alpha;
    chosen.old_source_station=r.station;chosen.old_source_row=r.row;chosen.old_source_alpha=r.alpha;chosen.old_navigation_score=old_score;chosen.navigation_score=old_score;
    for(int path_index:paths){const auto&path=material.paths.at(path_index);const auto layout=b.core.model.layout(b.core.model.shot_map.at(r.shot_id),r.material,path_index,r.pts,r.width_block,r.width_sign);const auto pointers=b.core.model.pointers(layout);
      auto nav=[&](double s){const auto p=b.core.model.predict_for_al(layout,s,r.uv[1],pointers.data());if(!p.physical_valid()||!finite_array(p.uv))return double(INFINITY);const auto a=nearest_alpha(*r.query,{p.uv[0]+w.nu_u/rho,p.uv[1]+w.nu_v/rho},old_alpha);return score(p.uv,a.alpha,0);};
      auto native=[&](double s,JointQueryRestartRow*result){
        if(!r.station_allowed(b.core.model,path_index,s,1e-12))return double(INFINITY);const auto root=b.core.model.project_rs(layout,s,r.uv[1]);
        if(!root.valid||!finite_array(root.uv)||root.row<0||root.row>1079||!b.core.model.visibility(layout,s,root.row,pointers.data()).visible())return double(INFINITY);
        if(r.width_block>=0){const double g=b.core.model.measurement_domain(layout,s,pointers.data(),5);if(!std::isfinite(g)||g>1e-8)return double(INFINITY);if(path.kind==1||path.kind==4){const double g=b.core.model.measurement_domain(layout,s,pointers.data(),6);if(!std::isfinite(g)||g>1e-8)return double(INFINITY);}}
        const auto a=nearest_alpha(*r.query,{root.uv[0]+w.nu_u/rho,root.uv[1]+w.nu_v/rho},old_alpha);const double value=score(root.uv,a.alpha,root.row-root.uv[1]);
        if(result&&std::isfinite(value)){result->path=path_index;result->station=r.kind==1?old_station:s;result->row=root.row;result->alpha=a.alpha;result->navigation_score=value;Vec2<double>q;r.query->point(a.alpha,&q);result->query_error_px=std::hypot(root.uv[0]-q[0],root.uv[1]-q[1]);result->pixel_error_px=std::abs(r.normal[0]*(root.uv[0]-r.uv[0])+r.normal[1]*(root.uv[1]-r.uv[1]));result->RS_equality_px=root.equality_px;}
        return value;
      };
      if(r.kind==1||path.physical_point){JointQueryRestartRow candidate=chosen;if(native(source_station,&candidate)<chosen.navigation_score)chosen=candidate;continue;}
      for(const auto&interval:r.station_intervals(b.core.model,path_index)){
        const auto grid=station_grid(b.core.model,layout,path,interval,old_station);std::vector<double>values;values.reserve(grid.size());for(double s:grid)values.push_back(nav(s));
        for(std::size_t k=0;k<grid.size();++k){if(!std::isfinite(values[k])||(k&&values[k]>values[k-1])||(k+1<grid.size()&&values[k]>values[k+1]))continue;
          double lo=grid[k?k-1:0],hi=grid[std::min(k+1,grid.size()-1)];const double original_lo=lo,original_hi=hi;
          for(int step=0;step<28&&hi>lo;++step){const double a=lo+(hi-lo)*.3819660112501051,z=lo+(hi-lo)*.6180339887498949;if(nav(a)<nav(z))hi=z;else lo=a;}
          const double refined=(lo+hi)*.5;JointQueryRestartRow candidate=chosen;const double native_grid=native(grid[k],&candidate);if(native_grid<chosen.navigation_score)chosen=candidate;
          candidate=chosen;const double native_refined=native(refined,&candidate);if(native_refined<chosen.navigation_score)chosen=candidate;
          // Refine the SAME finite bracket using completed native RS projections.
          lo=original_lo;hi=original_hi;for(int step=0;step<20&&hi>lo;++step){const double a=lo+(hi-lo)*.3819660112501051,z=lo+(hi-lo)*.6180339887498949;if(native(a,nullptr)<native(z,nullptr))hi=z;else lo=a;}
          candidate=chosen;if(native((lo+hi)*.5,&candidate)<chosen.navigation_score)chosen=candidate;
        }
      }
    }
    if(std::isfinite(chosen.navigation_score)&&chosen.navigation_score<old_score&&
       (chosen.path!=r.path||!same(chosen.station,old_station)||!same(chosen.row,old_row)||!same(chosen.alpha,old_alpha))){std::lock_guard<std::mutex>lock(mutex);plan.rows.push_back(chosen);}
  }}catch(const std::exception&e){failed=true;std::lock_guard<std::mutex>lock(mutex);error=e.what();}};
  std::vector<std::thread>pool;for(int k=0;k<std::max(1,threads);++k)pool.emplace_back(worker);for(auto&t:pool)t.join();if(failed)throw std::runtime_error("Query restart proposal failed: "+error);
  plan.searched_queries=searched.load();std::sort(plan.rows.begin(),plan.rows.end(),[](const auto&a,const auto&b){return a.row_index<b.row_index;});check_plan(b,plan,false);
  plan.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::filesystem::create_directories(out);std::ofstream stream(out/"proposed_rows.jsonl");
  for(const auto&c:plan.rows){const auto&r=b.core.packet.rows[c.row_index];stream<<eventio::Json({{"row_index",c.row_index},{"source_row_id",r.source_row_id},{"source_dataset_id",r.source_dataset_id},{"shot_id",r.shot_id},{"native_pts",r.pts},{"material_index",r.material},{"source_uv",r.uv},{"old_path",c.old_path},{"path",c.path},{"old_station",c.old_station},{"station",c.station},{"old_row",c.old_row},{"row",c.row},{"old_alpha",c.old_alpha},{"alpha",c.alpha},{"original_alpha_bounds",{r.query->alpha.front(),r.query->alpha.back()}},{"old_navigation_score",std::isfinite(c.old_navigation_score)?eventio::Json(c.old_navigation_score):eventio::Json()},{"navigation_score",c.navigation_score},{"native_Query_error_px",c.query_error_px},{"native_pixel_error_px",c.pixel_error_px},{"native_RS_equality_px",c.RS_equality_px}}).dump()<<'\n';}
  stream.close();if(!stream)throw std::runtime_error("Query restart proposal ledger write failed");return plan;
}

inline void apply_joint_query_restart_plan(JointALBuilder&b,JointQueryRestartPlan&plan,bool no_live_problem){
  if(!no_live_problem||plan.applied)throw std::runtime_error("Query restart apply requires destroyed graph and unapplied plan");joint_query_restart_detail::check_plan(b,plan,false);
  if(!plan.rows.empty())advance_joint_path_generation_v1(b.core.packet);
  for(const auto&c:plan.rows){auto&r=b.core.packet.rows[c.row_index];r.path=c.path;b.core.station[c.row_index]=c.station;b.core.row[c.row_index]=r.row=c.row;b.core.alpha[c.row_index]=r.alpha=c.alpha;if(r.kind!=1)r.station=c.station;}
  plan.applied=true;
}
inline void rollback_joint_query_restart_plan(JointALBuilder&b,JointQueryRestartPlan&plan,bool no_live_problem){
  if(!no_live_problem||!plan.applied)throw std::runtime_error("Query restart rollback requires destroyed graph and applied plan");joint_query_restart_detail::check_plan(b,plan,true);
  if(!plan.rows.empty())advance_joint_path_generation_v1(b.core.packet);
  for(const auto&c:plan.rows){auto&r=b.core.packet.rows[c.row_index];r.path=c.old_path;r.station=c.old_source_station;r.row=c.old_source_row;r.alpha=c.old_source_alpha;b.core.station[c.row_index]=c.old_station;b.core.row[c.row_index]=c.old_row;b.core.alpha[c.row_index]=c.old_alpha;}
  plan.applied=false;joint_query_restart_detail::check_plan(b,plan,false);
}
// Invoke AFTER keeping the proposal or performing its exact rollback. The
// caller must have evaluated both complete graphs in the SAME outer context.
inline eventio::Json write_joint_query_restart_result(JointALBuilder&b,const JointQueryRestartPlan&plan,bool accepted,
    double full_cost_before,double full_cost_candidate,const eventio::Json&debt_before,const eventio::Json&debt_candidate,const std::filesystem::path&out){
  if(accepted&&(!plan.applied||plan.rows.empty()||!std::isfinite(full_cost_before)||!std::isfinite(full_cost_candidate)||!(full_cost_candidate<full_cost_before)))throw std::runtime_error("Query restart acceptance lacks strict complete current PHR decrease");
  if(!accepted&&plan.applied)throw std::runtime_error("rejected Query restart was not rolled back");joint_query_restart_detail::check_plan(b,plan,accepted);
  eventio::Json receipt={{"schema","rinkfit.same_finite_Query_auxiliary_restart.v1"},{"source_packet_sha256",plan.source_packet_sha256},{"raw_rows",plan.raw_rows},{"eligible_active_Query_rows",plan.eligible_queries},{"searched_rows",plan.searched_queries},{"proposed_rows",plan.rows.size()},{"numerical_chart_changes",plan.path_changes()},{"applied_rows",accepted?plan.rows.size():0},{"accepted",accepted},{"full_current_PHR_cost_before",std::isfinite(full_cost_before)?eventio::Json(full_cost_before):eventio::Json()},{"full_current_PHR_cost_candidate",std::isfinite(full_cost_candidate)?eventio::Json(full_cost_candidate):eventio::Json()},{"original_constraint_debt_before",debt_before},{"original_constraint_debt_candidate",debt_candidate},{"navigation_score_is_not_full_problem_acceptance",true},{"source_UV_ID_material_Query_operator_normal_alpha_support_W_E_or_physical_parameters_changed",false},{"chart_policy","Only original eight kind3 whole board.top/contact charts without source station intervals; same native-domain schema. All other materials, including closed C0, stay on their original finite path."},{"graph_and_GPU_layout_must_be_rebuilt_after_apply_or_rollback",true},{"discrete_path_checkpoint_required","finite_material_paths_RLE_u32_v1"},{"no_source_qualification_or_statistics_granted",true},{"proposed_rows_file_sha256",eventio::file_sha256(out/"proposed_rows.jsonl")},{"proposal_elapsed_seconds",plan.elapsed_seconds}};
  receipt["protected_nuisance_shots"]=plan.protected_shots;
  receipt["protected_active_Query_rows"]=plan.protected_queries;
  receipt["all_active_Query_rows"]=plan.eligible_queries+plan.protected_queries;
  receipt["complete_graph_and_equality_denominator_retained"]=true;
  eventio::atomic_json(out/"receipt.json",receipt);return receipt;
}
} // namespace rinkfit
