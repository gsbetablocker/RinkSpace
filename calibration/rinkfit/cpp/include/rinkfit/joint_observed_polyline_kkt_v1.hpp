#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_shape_cells_v1.hpp"
#include <map>
#include <atomic>
#include <thread>
namespace rinkfit {
// An internal polyline vertex has TWO one-sided derivatives. A vertex whose
// two admissible sides both increase its pixel merit belongs to the current
// numerical active set. Release invalidated vertices after accepted physical updates; enter new vertices after active-face convergence;
// never convert it into a source restriction or a world measurement.
class JointObservedPolylineKktV1 {
 int threads_=1;std::vector<std::size_t> rows_;std::map<std::size_t,double> active_,pending_;double maximum_violation_=0,check_seconds_=0;mutable double maximum_snap_pixel_delta_=0;mutable std::size_t roundoff_equivalent_entries_=0;std::size_t invalidated_=0;bool accepted_step_check_=false;std::vector<double>pixel_scales2_;double invalidated_pixel_cost_bound_=0,recent_joint_decrease_=0;bool restart_deferred_=false;
 using J=ceres::Jet<double,1>;
 struct JetParameters {
  std::vector<std::vector<J>> values;
  explicit JetParameters(const JointEventModel& model){values.reserve(model.blocks.size());for(const auto& block:model.blocks){values.emplace_back();auto& v=values.back();v.reserve(block.values.size());for(double x:block.values)v.emplace_back(x);}}
  void pointers(const JointRowLayout& layout,std::vector<const J*>& out)const{out.clear();out.reserve(layout.block_ids.size());for(int id:layout.block_ids)out.push_back(values.at(id).data());}
 };
 struct Slope{bool valid=false;double gradient=0,tolerance=0,squared_error=0;};
 static Slope slope(JointALBuilder&b,std::size_t i,double s,const JetParameters&parameters){
  const auto&r=b.core.packet.rows.at(i);if(!r.station_allowed(b.core.model,r.path,s))return {};
  // Each one-sided derivative uses its own legal local segment. Reusing the
  // live two-vertex layout would freeze the opposite side at a polyline knot.
  const auto l=b.core.model.layout(b.core.model.shot_map.at(r.shot_id),r.material,r.path,r.pts,r.width_block,r.width_sign,s,-1);const auto dp=b.core.model.pointers(l);std::vector<const J*>ptr;parameters.pointers(l,ptr);
  J station(s);station.v[0]=1;const auto q=b.core.model.predict_for_al(l,station,J(b.core.row.at(i)),ptr.data());if(!q.arithmetic_valid||q.physical_status!=DomainStatus::kValid)return {};
  const auto visible=b.core.model.visibility(l,s,b.core.row.at(i),dp.data());if(!visible.arithmetic_valid||visible.front>0||visible.occlusion>0)return {};
  double g=0,j2=0,r2=0;for(int k=0;k<2;++k){const double e=q.uv[k].a-r.uv[k];g+=e*q.uv[k].v[0];j2+=q.uv[k].v[0]*q.uv[k].v[0];r2+=e*e;}
  return {true,g,128*std::numeric_limits<double>::epsilon()*std::max(1.,std::sqrt(r2*j2)),r2};
 }
 std::map<std::size_t,double> analyze(JointALBuilder&b)const{
  if(rows_.empty())return {};
  JetParameters parameters(b.core.model);std::vector<double> best_stations(rows_.size(),NAN),snap_pixels(rows_.size(),0.);std::vector<unsigned char>roundoff_entries(rows_.size(),0);std::atomic<std::size_t> index{0};
  auto work=[&](){std::vector<const J*> jets;for(;;){const auto row_index=index.fetch_add(1);if(row_index>=rows_.size())break;const auto i=rows_[row_index];const auto&r=b.core.packet.rows.at(i);const auto&p=b.core.model.materials.at(r.material).paths.at(r.path);const auto l=r.layout(b.core.model);
   const int n=b.core.model.polyline_vertex_count(r.material,r.path);
   // Legacy kind 2 explicitly repeats its closing point. Its actual CPU/GPU
   // projection uses N-1 edges even when closed=true. Kind 6 closes N vertices.
   const int den=p.kind==2?n-1:(p.closed?n:n-1);if(den<1)continue;
   const bool closed=p.kind==2?(p.points.front()==p.points.back()):p.closed;
   const double s=b.core.station.at(i);const int cell=!p.vertex_blocks.empty()?r.numerical_shape_segment:std::min(den-1,int(std::floor(s*den)));const auto ptr=b.core.model.pointers(l);parameters.pointers(l,jets);
   const auto before=b.core.model.predict_for_al(l,s,b.core.row.at(i),ptr.data());if(!before.arithmetic_valid)continue;
   const double beforeCost=std::pow(before.uv[0]-r.uv[0],2)+std::pow(before.uv[1]-r.uv[1],2),tol=128*std::numeric_limits<double>::epsilon()*std::max(1.,beforeCost);double bestCost=INFINITY,best=NAN,bestPixels=0.;bool bestRoundoff=false;
   for(int v:{cell,cell+1}){if(v<0||v>den)continue;const double knot=double(v)/den;if(!r.station_allowed(b.core.model,r.path,knot))continue;
    const auto q=b.core.model.predict_for_al(l,knot,b.core.row.at(i),ptr.data());if(!q.arithmetic_valid||q.physical_status!=DomainStatus::kValid)continue;
    const double cost=std::pow(q.uv[0]-r.uv[0],2)+std::pow(q.uv[1]-r.uv[1],2);const double pixelDelta=std::hypot(q.uv[0]-before.uv[0],q.uv[1]-before.uv[1]);const bool roundoffEquivalent=pixelDelta<=1e-8;/* The global nearest-point profile can land a few ulps beside a true vertex. A sub-1e-8-pixel difference is the existing CPU/GPU projection-equivalence tolerance, not evidence for a smooth side of that corner. Retain both one-sided KKT tests and every original source value. */if(cost>beforeCost+tol&&!roundoffEquivalent)continue;
    const auto visible=b.core.model.visibility(l,knot,b.core.row.at(i),ptr.data());if(!visible.arithmetic_valid||visible.front>0||visible.occlusion>0)continue;
    bool hasL=v>0||closed,hasR=v<den||closed;double left=std::nextafter(v==0?1.:knot,-INFINITY),right=std::nextafter(v==den?0.:knot,INFINITY);const int lc=v==0?den-1:v-1,rc=v==den?0:v;
    for(int k=0;k<16&&hasL&&int(std::floor(left*den))!=lc;++k)left=std::nextafter(left,-INFINITY);for(int k=0;k<16&&hasR&&int(std::floor(right*den))!=rc;++k)right=std::nextafter(right,INFINITY);
    hasL=hasL&&r.station_allowed(b.core.model,r.path,left);hasR=hasR&&r.station_allowed(b.core.model,r.path,right);if(!hasL&&!hasR)continue;
    const auto L=hasL?slope(b,i,left,parameters):Slope{},R=hasR?slope(b,i,right,parameters):Slope{};if((hasL&&!L.valid)||(hasR&&!R.valid))continue;
    if((!hasL||L.gradient<=L.tolerance)&&(!hasR||R.gradient>=-R.tolerance)&&(cost<bestCost||(cost==bestCost&&std::abs(knot-s)<std::abs(best-s)))){best=knot;bestCost=cost;bestPixels=pixelDelta;bestRoundoff=cost>beforeCost+tol;}
   }if(std::isfinite(best)){best_stations[row_index]=best;snap_pixels[row_index]=bestPixels;roundoff_entries[row_index]=bestRoundoff;}
  }};std::vector<std::thread> pool;for(int q=1;q<threads_;++q)pool.emplace_back(work);work();for(auto&t:pool)t.join();
  std::map<std::size_t,double> next;maximum_snap_pixel_delta_=0.;roundoff_equivalent_entries_=0;for(std::size_t k=0;k<rows_.size();++k)if(std::isfinite(best_stations[k])){next.emplace(rows_[k],best_stations[k]);maximum_snap_pixel_delta_=std::max(maximum_snap_pixel_delta_,snap_pixels[k]);roundoff_equivalent_entries_+=roundoff_entries[k];}return next;
 }

 public:
 JointObservedPolylineKktV1(JointALBuilder&b,ceres::Problem&p,int threads=1,const eventio::Json*ledger=nullptr):threads_(std::max(1,threads)){if(!b.source_uv_observed_row_initialization())throw std::runtime_error("observed polyline working set used outside measured-row mode");for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active||r.kind==1)continue;const auto&shape=b.core.model.materials.at(r.material).paths.at(r.path);if(joint_shape_vertex_path_v1(shape))continue;/* Local vertex rows use their actual one-sided Ceres cell box; accepted cell navigation owns cross-vertex release. */if((shape.kind==2||shape.kind==6)&&p.HasParameterBlock(&b.core.station[i])&&!p.IsParameterBlockConstant(&b.core.station[i]))rows_.push_back(i);}if(ledger){pixel_scales2_.assign(b.core.packet.rows.size(),0.);for(const auto&g:ledger->at("groups")){const auto&ids=g.at("actual_row_indices");const auto&weights=g.at("actual_per_row_weight");if(ids.size()!=weights.size())throw std::runtime_error("polyline progress restart weight size mismatch");for(std::size_t k=0;k<ids.size();++k){const auto i=ids.at(k).get<std::size_t>();const double E=b.core.packet.rows.at(i).E,w=weights.at(k).get<double>();pixel_scales2_.at(i)=w/(E*E);}}for(const auto i:rows_)if(!(pixel_scales2_.at(i)>0)||!std::isfinite(pixel_scales2_.at(i)))throw std::runtime_error("polyline progress restart requires original actual pixel weights");}pending_=analyze(b);}
 bool holds(std::size_t row)const{return active_.count(row)!=0;}
 bool changed(JointALBuilder&b){accepted_step_check_=false;pending_=analyze(b);return pending_!=active_;}
 bool active_invalidated(JointALBuilder&b,int threads,double successful_joint_decrease=0){
  if(active_.empty()){pending_.clear();invalidated_=0;maximum_violation_=0;check_seconds_=0;invalidated_pixel_cost_bound_=0;restart_deferred_=false;accepted_step_check_=true;return false;}const auto start=std::chrono::steady_clock::now();std::vector<std::pair<std::size_t,double>>items(active_.begin(),active_.end());std::vector<double>violations(items.size(),0),pixel_cost_bounds(items.size(),0);std::atomic<std::size_t>next{0};
  JetParameters parameters(b.core.model);auto work=[&](){std::vector<const J*> jets;for(;;){const auto k=next.fetch_add(1);if(k>=items.size())break;const auto i=items[k].first;const auto&r=b.core.packet.rows.at(i);const auto&p=b.core.model.materials.at(r.material).paths.at(r.path);const auto layout=r.layout(b.core.model);parameters.pointers(layout,jets);const auto ptr=b.core.model.pointers(layout);
   const int n=b.core.model.polyline_vertex_count(r.material,r.path);const int den=p.kind==2?n-1:(p.closed?n:n-1);if(den<1)continue;
   const bool closed=p.kind==2?(p.closed&&p.points.front()==p.points.back()):p.closed;const double knot=items[k].second;const int v=int(std::llround(knot*den));bool hasL=v>0||closed,hasR=v<den||closed;double left=std::nextafter(v==0?1.:knot,-INFINITY),right=std::nextafter(v==den?0.:knot,INFINITY);const int lc=v==0?den-1:v-1,rc=v==den?0:v;
   for(int q=0;q<16&&hasL&&int(std::floor(left*den))!=lc;++q)left=std::nextafter(left,-INFINITY);for(int q=0;q<16&&hasR&&int(std::floor(right*den))!=rc;++q)right=std::nextafter(right,INFINITY);
   const auto L=hasL?slope(b,i,left,parameters):Slope{},R=hasR?slope(b,i,right,parameters):Slope{};double violation=0;
   if(L.valid&&L.gradient>L.tolerance)violation=std::max(violation,L.gradient/std::max(L.tolerance,std::numeric_limits<double>::min()));
   if(R.valid&&R.gradient<-R.tolerance)violation=std::max(violation,-R.gradient/std::max(R.tolerance,std::numeric_limits<double>::min()));violations[k]=violation;if(violation>1.&&!pixel_scales2_.empty())pixel_cost_bounds[k]=.5*pixel_scales2_.at(i)*std::max(L.squared_error,R.squared_error);
  }};
  std::vector<std::thread>pool;for(int q=1;q<std::max(1,threads);++q)pool.emplace_back(work);work();for(auto&t:pool)t.join();
  pending_=active_;maximum_violation_=0;invalidated_=0;invalidated_pixel_cost_bound_=0;for(std::size_t k=0;k<items.size();++k)if(violations[k]>1.){pending_.erase(items[k].first);maximum_violation_=std::max(maximum_violation_,violations[k]);++invalidated_;invalidated_pixel_cost_bound_+=pixel_cost_bounds[k];}
  // A stationary, admissible source row cannot reduce its pixel half-cost by
  // more than its whole current pixel half-cost. Continue the active face while
  // the accepted joint step is worth more than the sum of these conservative
  // bounds. This is scheduling, not a KKT tolerance: changed() still reconsiders
  // every vertex at face convergence, using the original one-sided roundoff tests.
  recent_joint_decrease_=successful_joint_decrease;restart_deferred_=invalidated_!=0&&!pixel_scales2_.empty()&&std::isfinite(successful_joint_decrease)&&successful_joint_decrease>invalidated_pixel_cost_bound_;
  accepted_step_check_=true;check_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return invalidated_!=0&&!restart_deferred_;
 }
 eventio::Json progress_report()const{return {{"invalidated_vertices",invalidated_},{"invalidated_pixel_half_cost_upper_bound",invalidated_pixel_cost_bound_},{"accepted_joint_half_cost_decrease",recent_joint_decrease_},{"restart_deferred_while_joint_step_is_larger",restart_deferred_},{"check_seconds",check_seconds_},{"exact_KKT_reconsidered_at_face_convergence",true},{"source_or_physical_freedoms_changed",false}};}
 eventio::Json install(JointALBuilder&b,ceres::Problem&p){eventio::Json changed=eventio::Json::array();std::size_t released=0,entered=0;for(const auto&x:active_)if(!pending_.count(x.first)){p.SetParameterBlockVariable(&b.core.station[x.first]);changed.push_back(x.first);++released;}for(const auto&x:pending_){b.core.station[x.first]=x.second;p.SetParameterBlockConstant(&b.core.station[x.first]);if(!active_.count(x.first)){++entered;changed.push_back(x.first);}}active_=pending_;eventio::Json ids=eventio::Json::array();for(const auto&x:active_)ids.push_back({{"row",x.first},{"station",x.second}});return {{"schema","rinkfit.observed_polyline_vertex_working_set.v1"},{"eligible_curve_rows",rows_.size()},{"active_vertices",active_.size()},{"roundoff_equivalent_vertex_entries",roundoff_equivalent_entries_},{"maximum_vertex_snap_pixel_delta",maximum_snap_pixel_delta_},{"entered",entered},{"released",released},{"vertices",ids},{"changed_station_rows",changed},{"source_domain_or_physical_parameter_freedoms_changed",false},{"reconsidered_after_active_face_convergence",!accepted_step_check_},{"accepted_physical_step_invalidation_check",accepted_step_check_},{"invalidated",invalidated_},{"maximum_violation_over_roundoff_tolerance",maximum_violation_},{"check_seconds",check_seconds_},{"restarted_after_every_physical_step",false}};}
};
}
