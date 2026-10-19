#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <fstream>
#include <functional>
#include <chrono>
namespace rinkfit {

// A block-coordinate numerical restart on the SAME finite material path.
// Original source UV, identities, Query, W and physical parameter values never
// change. Proposals survive only if the complete current PHR objective falls.
inline eventio::Json refine_joint_ordinary_stations(
    JointALBuilder& b,ceres::Problem& problem,const std::filesystem::path& out,
    int threads,const std::function<eventio::Json()>& constraint_debt) {
  std::filesystem::create_directories(out);
  const auto start=std::chrono::steady_clock::now();
  JointCompactAcceptedState before(b);
  auto evaluate=[&](){
    if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))
      cache->PrepareForEvaluation(false,true);
    ceres::Problem::EvaluateOptions options;options.num_threads=threads;
    double cost=NAN;
    if(!problem.Evaluate(options,&cost,nullptr,nullptr,nullptr)||!std::isfinite(cost))
      throw std::runtime_error("ordinary-station restart cannot evaluate the complete objective");
    return cost;
  };
  const auto debt_before=constraint_debt();
  const double cost_before=evaluate();
  std::ofstream changes(out/"proposed_rows.jsonl");
  std::mutex output_mutex;
  std::atomic<std::size_t> next{0},eligible{0},searched{0},proposed{0};
  std::atomic<bool> error{false};std::string error_text;
  auto worker=[&](){try{for(;;){
    const auto i=next.fetch_add(1);if(i>=b.core.packet.rows.size())return;
    const auto&r=b.core.packet.rows[i];
    if(!r.active||r.kind==1||r.query||
       !problem.HasParameterBlock(&b.core.station[i])||
       problem.IsParameterBlockConstant(&b.core.station[i])||
       !problem.HasParameterBlock(&b.core.row[i])||problem.IsParameterBlockConstant(&b.core.row[i]))continue;
    ++eligible;
    const auto layout=r.layout(b.core.model);const auto pointers=b.core.model.pointers(layout);
    const auto&path=b.core.model.materials.at(r.material).paths.at(r.path);
    if(path.physical_point)continue;
    const double old_station=b.core.station[i],old_row=b.core.row[i];
    const auto current=b.core.model.predict_for_al(layout,old_station,old_row,pointers.data());
    const double old_error=current.arithmetic_valid?
      std::hypot(current.uv[0]-r.uv[0],current.uv[1]-r.uv[1]):INFINITY;
    if(current.physical_valid()&&old_error<=r.E&&
       r.station_allowed(b.core.model,r.path,old_station,1e-12))continue;
    ++searched;
    auto observed_error=[&](double s){
      const auto p=b.core.model.predict_for_al(layout,s,r.uv[1],pointers.data());
      return p.physical_valid()&&finite_array(p.uv)?
        std::hypot(p.uv[0]-r.uv[0],p.uv[1]-r.uv[1]):INFINITY;
    };
    auto native_error=[&](double s,EventRootResult* result=nullptr){
      if(!r.station_allowed(b.core.model,r.path,s,1e-12))return double(INFINITY);
      auto root=b.core.model.project_rs(layout,s,r.uv[1]);
      if(!root.valid||!finite_array(root.uv)||
         !b.core.model.visibility(layout,s,root.row,pointers.data()).visible())return double(INFINITY);
      if(result)*result=root;
      return std::hypot(root.uv[0]-r.uv[0],root.uv[1]-r.uv[1]);
    };
    double best=INFINITY,chosen=old_station,bracket_lo=0,bracket_hi=1;
    std::size_t segments=0;
    if(path.kind==2&&path.points.size()>1)segments=path.closed?path.points.size():path.points.size()-1;
    if(path.kind==6){
      const std::size_t coordinates=layout.geometry_indices.empty()?
        b.core.model.blocks.at(path.geometry_block).values.size()-layout.geometry_offset:
        layout.geometry_indices.size();
      const auto vertices=coordinates/2;if(vertices>1)segments=path.closed?vertices:vertices-1;
    }
    for(const auto& interval:r.station_intervals(b.core.model,r.path)){
      std::vector<double> grid{interval[0],interval[1]};
      for(int k=0;k<=32;++k)grid.push_back(interval[0]+(interval[1]-interval[0])*k/32.);
      if(old_station>=interval[0]&&old_station<=interval[1])grid.push_back(old_station);
      if(segments)for(std::size_t k=0;k<=segments;++k){
        const double s=double(k)/segments;if(s>=interval[0]&&s<=interval[1])grid.push_back(s);
      }
      std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
      std::vector<Vec2<double>> pixels(grid.size());std::vector<double> errors(grid.size(),INFINITY);
      for(std::size_t k=0;k<grid.size();++k){
        auto p=b.core.model.predict_for_al(layout,grid[k],r.uv[1],pointers.data());
        pixels[k]=p.uv;
        if(p.physical_valid()&&finite_array(p.uv))errors[k]=std::hypot(p.uv[0]-r.uv[0],p.uv[1]-r.uv[1]);
        if(errors[k]<best){best=errors[k];chosen=grid[k];bracket_lo=grid[k?k-1:0];bracket_hi=grid[std::min(k+1,grid.size()-1)];}
      }
      // Include each finite polyline edge, including very short source edges.
      for(std::size_t k=1;k<grid.size();++k)if(std::isfinite(errors[k-1])&&std::isfinite(errors[k])){
        const auto a=pixels[k-1],z=pixels[k];const double du=z[0]-a[0],dv=z[1]-a[1],length=du*du+dv*dv;
        if(!(length>0))continue;
        const double t=std::max(0.,std::min(1.,((r.uv[0]-a[0])*du+(r.uv[1]-a[1])*dv)/length));
        const double s=grid[k-1]+t*(grid[k]-grid[k-1]),e=observed_error(s);
        if(e<best){best=e;chosen=s;bracket_lo=grid[k-1];bracket_hi=grid[k];}
      }
    }
    if(!std::isfinite(best))continue;
    double lo=bracket_lo,hi=bracket_hi;
    for(int k=0;k<28&&hi>lo;++k){
      const double a=lo+(hi-lo)*.3819660112501051,z=lo+(hi-lo)*.6180339887498949;
      if(observed_error(a)<observed_error(z))hi=z;else lo=a;
    }
    const double local=(lo+hi)*.5;
    if(observed_error(local)<best)chosen=local;
    EventRootResult root;double candidate_error=native_error(chosen,&root);
    if(!std::isfinite(candidate_error)||!(candidate_error<old_error))continue;
    b.core.station[i]=chosen;b.core.row[i]=root.row;++proposed;
    std::lock_guard<std::mutex>lock(output_mutex);
    changes<<eventio::Json({{"row_index",i},{"source_row_id",r.source_row_id},{"shot_id",r.shot_id},
      {"native_pts",r.pts},{"material_index",r.material},{"path",r.path},{"source_uv",r.uv},
      {"old_station",old_station},{"proposed_station",chosen},{"old_row",old_row},{"proposed_row",root.row},
      {"old_explicit_pixel_error_px",std::isfinite(old_error)?eventio::Json(old_error):eventio::Json()},
      {"proposed_independent_RS_pixel_error_px",candidate_error},{"native_RS_equality_px",root.equality_px}}).dump()<<'\n';
  }}catch(const std::exception&e){error=true;std::lock_guard<std::mutex>lock(output_mutex);error_text=e.what();}};
  std::vector<std::thread> pool;for(int i=0;i<std::max(1,threads);++i)pool.emplace_back(worker);
  for(auto&t:pool)t.join();changes.close();
  if(error){before.restore(b);throw std::runtime_error("ordinary-station proposal failed: "+error_text);}
  const double candidate_cost=evaluate();
  const auto debt_candidate=constraint_debt();
  const bool accepted=proposed>0&&candidate_cost<cost_before;
  if(!accepted){before.restore(b);evaluate();}
  const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  eventio::Json report={{"schema","rinkfit.same_finite_path_ordinary_station_restart.v1"},
    {"source_packet_sha256",b.core.packet.source_hash},{"eligible_ordinary_nonQuery_rows",eligible.load()},
    {"searched_rows",searched.load()},{"proposed_rows",proposed.load()},{"applied_rows",accepted?proposed.load():0},
    {"full_current_PHR_cost_before",cost_before},{"full_current_PHR_cost_proposed",candidate_cost},{"accepted",accepted},
    {"original_constraint_debt_before",debt_before},{"original_constraint_debt_proposed",debt_candidate},
    {"source_UV_ID_material_path_Query_W_E_or_physical_parameters_changed",false},
    {"no_source_qualification_or_statistics_granted",true},{"same_complete_problem_retained",true},
    {"proposed_rows_file_sha256",eventio::file_sha256(out/"proposed_rows.jsonl")},{"elapsed_seconds",elapsed}};
  eventio::atomic_json(out/"receipt.json",report);return report;
}
}
