#pragma once
#include "rinkfit/joint_coupled_motion_restart.hpp"
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <array>
#include <map>
#include <tuple>

namespace rinkfit {
namespace joint_source_camera_mode_detail {
struct Mode {
 int shot=-1;
 std::vector<int>blocks;
 std::vector<std::array<double,3>>coefficients;
 std::array<double,3>lower{{-INFINITY,-INFINITY,-INFINITY}},upper{{INFINITY,INFINITY,INFINITY}};
 eventio::Json anchor_responses=eventio::Json::array();
};
inline std::vector<double> trajectory_basis(const JointShot&shot,double time,int axis){
 const auto&math=shot.math;const double x=(time-math.first_control_time)/math.dt;
 const int cell=int(std::floor(x));if(cell<1||cell+2>=math.n_controls)throw std::runtime_error("source camera mode outside original cubic domain");
 const auto w=cubic_weights(x-cell);std::vector<double>basis(math.n_controls,0);
 for(int k=0;k<4;++k)basis[cell-1+k]+=w[k];
 if(axis<2){const int dependent=math.dependent_control;const double influence=basis.at(dependent);basis[dependent]=0;double pivot=0;
  for(int k=0;k<4;++k)if(math.reference_indices[k]==dependent)pivot=math.reference_weights[k];
  if(pivot==0)throw std::runtime_error("source camera mode has an invalid dependent reference");
  for(int k=0;k<4;++k)if(math.reference_indices[k]!=dependent)basis.at(math.reference_indices[k])-=influence*math.reference_weights[k]/pivot;
 }return basis;
}
inline Mode make_mode(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&spec){
 Mode mode;mode.shot=b.core.model.shot_map.at(spec.at("shot_id").get<std::string>());const auto&shot=b.core.model.shots.at(mode.shot);mode.blocks=shot.control_blocks;
 if(shot.control_sources!=shot.control_blocks||std::any_of(shot.control_offsets.begin(),shot.control_offsets.end(),[](int x){return x!=0;}))throw std::runtime_error("source camera mode requires this shot's existing direct three-coordinate controls");
 std::vector<double>times{0.};for(const auto&v:spec.at("preserved_anchor_native_pts")){
  const auto pts=v.get<std::int64_t>();if(pts<shot.first_pts||pts>shot.last_pts||(pts-shot.first_pts)%256)throw std::runtime_error("source camera mode anchor clock mismatch");
  const double t=(pts-shot.time_origin_pts)/12800.;if(std::find(times.begin(),times.end(),t)==times.end())times.push_back(t);
 }
 const auto target=spec.at("target_native_pts").get<std::int64_t>();if(target<shot.first_pts||target>shot.last_pts||(target-shot.first_pts)%256)throw std::runtime_error("source camera mode target clock mismatch");
 const double target_time=(target-shot.time_origin_pts)/12800.;if(std::find(times.begin(),times.end(),target_time)!=times.end())throw std::runtime_error("source camera mode target is also a preserved anchor");times.push_back(target_time);
 mode.coefficients.resize(mode.blocks.size(),{{0,0,0}});
 for(int axis=0;axis<3;++axis){std::vector<int>free;
  for(std::size_t k=0;k<mode.blocks.size();++k){const auto&block=b.core.model.blocks.at(mode.blocks[k]);if(block.values.size()!=3)throw std::runtime_error("source camera mode original control shape changed");
   if(axis<2&&int(k)==shot.math.dependent_control)continue;
   if(!block.constant&&std::find(block.fixed_indices.begin(),block.fixed_indices.end(),axis)==block.fixed_indices.end()&&problem.HasParameterBlock(block.values.data())&&!problem.IsParameterBlockConstant(block.values.data()))free.push_back(int(k));
  }
  if(free.empty())throw std::runtime_error("source camera mode has no original free coordinates");
  Eigen::MatrixXd A(times.size(),free.size());Eigen::VectorXd rhs=Eigen::VectorXd::Zero(times.size());rhs[rhs.size()-1]=1;
  for(std::size_t i=0;i<times.size();++i){const auto basis=trajectory_basis(shot,times[i],axis);for(std::size_t j=0;j<free.size();++j)A(int(i),int(j))=basis.at(free[j]);}
  Eigen::JacobiSVD<Eigen::MatrixXd>svd(A,Eigen::ComputeThinU|Eigen::ComputeThinV);Eigen::VectorXd solution=svd.solve(rhs);
  // An exactly unused allocated control has exactly zero minimum-norm change,
  // including when its present value lies on an original box bound.
  for(int j=0;j<A.cols();++j)if(A.col(j).squaredNorm()==0)solution[j]=0;
  if(!solution.allFinite()||(A*solution-rhs).lpNorm<Eigen::Infinity>()>1e-10)throw std::runtime_error("source camera preserve-anchor mode is infeasible in original free coordinates");
  for(std::size_t j=0;j<free.size();++j){const int k=free[j],id=mode.blocks[k];const auto&block=b.core.model.blocks[id];const double c=solution[int(j)];mode.coefficients[k][axis]=c;if(c==0)continue;
   const double lower=std::max(block.lower.at(axis),problem.GetParameterLowerBound(block.values.data(),axis)),upper=std::min(block.upper.at(axis),problem.GetParameterUpperBound(block.values.data(),axis));
   const double a=(lower-block.values[axis])/c,z=(upper-block.values[axis])/c;
   mode.lower[axis]=std::max(mode.lower[axis],std::min(a,z));mode.upper[axis]=std::min(mode.upper[axis],std::max(a,z));
  }
  if(!(mode.lower[axis]<=0&&mode.upper[axis]>=0&&mode.lower[axis]<=mode.upper[axis]))throw std::runtime_error("source camera mode original coefficient bounds are infeasible");
  eventio::Json responses=eventio::Json::array();for(std::size_t i=0;i<times.size();++i)responses.push_back({{"frame_time",times[i]},{"required",rhs[int(i)]},{"actual",(A*solution)[int(i)]}});
  mode.anchor_responses.push_back({{"axis",axis},{"free_original_coordinates",free.size()},{"mode_norm",solution.norm()},{"constraint_rank",svd.rank()},{"responses",responses},
   {"delta_lower",std::isfinite(mode.lower[axis])?eventio::Json(mode.lower[axis]):eventio::Json()},{"delta_upper",std::isfinite(mode.upper[axis])?eventio::Json(mode.upper[axis]):eventio::Json()}});
 }return mode;
}
struct RowCost {
 const JointEventModel*model=nullptr;const JointObservation*source=nullptr;JointRowLayout layout;
 std::vector<std::vector<double>>values;std::vector<std::array<double,3>>coefficients;
 double pixel_scale=1,sqrt_rho=1,RS_shift=0;
 template<class T>bool operator()(T const*const*p,T*residual)const{try{
  if(!source->station_allowed(*model,layout.path,event_scalar(p[1][0]),1e-12))return false;
  std::vector<std::vector<T>>storage(values.size());std::vector<const T*>ptr;ptr.reserve(values.size());
  for(std::size_t i=0;i<values.size();++i){storage[i].reserve(values[i].size());for(double v:values[i])storage[i].push_back(T(v));
   if(coefficients[i]!=std::array<double,3>{{0,0,0}}){if(storage[i].size()!=3)return false;for(int axis=0;axis<3;++axis)storage[i][axis]+=T(coefficients[i][axis])*p[0][axis];}
   ptr.push_back(storage[i].data());
  }
  const auto q=model->predict_for_al(layout,p[1][0],p[2][0],ptr.data());if(!q.arithmetic_valid)return false;
  residual[0]=T(pixel_scale)*(q.uv[0]-T(source->uv[0]));residual[1]=T(pixel_scale)*(q.uv[1]-T(source->uv[1]));
  residual[2]=T(sqrt_rho)*(p[2][0]-q.uv[1]+T(RS_shift));return finite_scalar(residual[0])&&finite_scalar(residual[1])&&finite_scalar(residual[2]);
 }catch(const std::exception&){return false;}}
};
inline int family(const std::string&label){return label.rfind("board.",0)==0?0:(label.rfind("Ad",0)==0||label.rfind("advertisement",0)==0?1:2);}
inline double diagnostic_rms(const eventio::Json&rows){long double s=0;for(const auto&r:rows){if(!r.at("native_RS_valid").get<bool>()||r.at("native_RS_error_px").is_null())return INFINITY;const double v=r.at("native_RS_error_px");s+=v*v;}return rows.empty()?INFINITY:std::sqrt(double(s/rows.size()));}
} // namespace joint_source_camera_mode_detail

// Production numerical seed only: original trajectory + a source-supported
// correction mode, followed by the existing unrestricted full-scope solve.
// The tiny solve never writes b. The transaction below applies ONLY accepted
// control proposals, reinitializes all actual layout consumers, and evaluates
// the complete current engineering PHR graph before keeping any state.
inline eventio::Json refine_joint_source_camera_mode_navigation(
 JointALBuilder&b,std::unique_ptr<ceres::Problem>&problem,const eventio::Json&spec,
 const eventio::Json&current_engineering_cfg,const std::vector<double>&current_equality_nu,
 const ceres::Solver::Options&current_solver_options,const std::filesystem::path&out,int threads,
 const std::function<void()>&destroy_graph,
 const std::function<void(const std::filesystem::path&)>&rebuild_graph,
 const std::function<eventio::Json(const std::vector<std::uint8_t>&,const std::filesystem::path&)>&initialize,
 const std::function<eventio::Json()>&constraint_debt){
 using namespace joint_source_camera_mode_detail;namespace compact=joint_coupled_motion_detail;
 if(!problem||spec.at("source_packet_sha256").get<std::string>()!=b.core.packet.source_hash)throw std::runtime_error("source camera mode requires complete current source-bound graph");
 if(spec.at("source_evidence").empty()||spec.at("source_rows").empty()||spec.at("diagnostic_source_rows").empty())throw std::runtime_error("source camera mode lacks actual source witnesses");
 for(const auto&ref:spec.at("source_evidence"))eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
 const auto started=std::chrono::steady_clock::now();std::filesystem::create_directories(out);
 const auto mode=make_mode(b,*problem,spec);const auto&shot=b.core.model.shots.at(mode.shot);
 const double rho=current_engineering_cfg.value("physical_constraint_penalty",10.);if(!(rho>0&&std::isfinite(rho)))throw std::runtime_error("source camera mode current rho invalid");
 if(!current_equality_nu.empty()&&current_equality_nu.size()!=b.schema().equality_ids.size())throw std::runtime_error("source camera mode current equality dual schema mismatch");
 std::vector<std::size_t>fit_rows;std::set<std::size_t>unique;std::set<std::int64_t>source_exposures;
 for(const auto&s:spec.at("source_rows")){const auto i=s.at("row_index").get<std::size_t>();const auto&r=b.core.packet.rows.at(i);
  if(!unique.insert(i).second||!r.active||r.kind!=0||r.query||r.shot_id!=shot.id||r.source_row_id!=s.at("source_row_id").get<std::string>()||r.pts!=s.at("native_pts").get<std::int64_t>()||r.material!=s.at("material_index").get<int>()||r.path!=s.at("path").get<int>()||r.uv!=s.at("source_uv").get<Vec2<double>>()||r.E!=s.at("E").get<double>()||r.source_dataset_id!=s.at("source_dataset_id").get<std::string>()||!(r.E>0))throw std::runtime_error("source camera mode original qualified ordinary witness changed");
  const auto&path=b.core.model.materials.at(r.material).paths.at(r.path);if(path.kind!=1||path.physical_point||path.geometry_block!=spec.at("geometry_block").get<int>())throw std::runtime_error("source camera mode witness original finite material changed");
  fit_rows.push_back(i);source_exposures.insert(r.pts);
 }
 for(const auto&v:spec.at("preserved_anchor_native_pts"))if(!source_exposures.count(v.get<std::int64_t>()))throw std::runtime_error("source camera mode preserved anchor has no actual source witness");
 if(!source_exposures.count(spec.at("target_native_pts").get<std::int64_t>()))throw std::runtime_error("source camera mode target has no actual source witness");
 for(const auto&v:spec.at("diagnostic_source_rows")){const auto i=v.get<std::size_t>();if(!unique.count(i)||b.core.packet.rows[i].pts!=spec.at("target_native_pts").get<std::int64_t>())throw std::runtime_error("source camera mode diagnostic must be an actual target source witness");}
 using Key=std::tuple<int,int,std::int64_t>;std::map<Key,std::size_t>counts;std::map<std::pair<int,int>,std::size_t>exposures;std::map<int,std::set<int>>families;std::size_t active=0;
 for(const auto&r:b.core.packet.rows)if(r.active){++active;const int shot_index=b.core.model.shot_map.at(r.shot_id),f=family(b.core.model.materials.at(r.material).label);if(++counts[{shot_index,f,r.pts}]==1)++exposures[{shot_index,f}];families[shot_index].insert(f);}
 const double total_weight=current_engineering_cfg.value("pixel_total_weight",double(active));if(!(total_weight>0&&std::isfinite(total_weight)))throw std::runtime_error("source camera mode original engineering total weight invalid");
 std::map<std::size_t,double>RS_nu;std::size_t equality=0;for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;if(unique.count(i))RS_nu[i]=current_equality_nu.empty()?0:current_equality_nu.at(equality);equality+=1+(r.query?2:0);}
 std::map<int,std::array<double,3>>by_block;std::set<int>changed_blocks;
 for(std::size_t k=0;k<mode.blocks.size();++k){by_block[mode.blocks[k]]=mode.coefficients[k];if(mode.coefficients[k]!=std::array<double,3>{{0,0,0}})changed_blocks.insert(mode.blocks[k]);}
 std::vector<std::uint8_t>preserved(b.core.packet.rows.size(),1);std::vector<compact::Row>selected;
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;const auto layout=r.layout(b.core.model);bool consumer=false;for(int id:layout.block_ids)consumer=consumer||changed_blocks.count(id);if(!consumer)continue;
  const auto&mat=b.core.model.materials.at(r.material);if((mat.label=="board.top"||mat.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty()){
   if(mat.paths.size()!=8)throw std::runtime_error("source camera mode requires original eight whole-board charts");
   const bool domain=b.core.model.station_intervals(r.material,r.path)!=std::vector<std::array<double,2>>{{0.,1.}};for(std::size_t k=0;k<mat.paths.size();++k)if(mat.paths[k].kind!=3||mat.paths[k].physical_point||(b.core.model.station_intervals(r.material,int(k))!=std::vector<std::array<double,2>>{{0.,1.}})!=domain)throw std::runtime_error("source camera mode changed whole-board finite native-domain schema");
  }
  preserved[i]=0;selected.push_back({i,r.path,r.station,r.row,r.alpha,b.core.station[i],b.core.row[i],b.core.alpha[i]});
 }
 const auto tokens_before=compact::row_tokens(b,preserved);compact::Primal before(b);
 const int residual_count=problem->NumResiduals(),parameter_count=problem->NumParameters(),effective_count=compact::effective_parameters(*problem);
 auto evaluate=[&](){if(!problem)throw std::runtime_error("source camera mode complete graph missing");if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))cache->PrepareForEvaluation(false,true);ceres::Problem::EvaluateOptions options;options.num_threads=threads;double value=NAN;if(!problem->Evaluate(options,&value,nullptr,nullptr,nullptr)||!std::isfinite(value))throw std::runtime_error("source camera mode full current PHR evaluation failed");return value;};
 const double cost_before=evaluate();const auto debt_before=constraint_debt(),roots_before=compact::roots(b,spec.at("diagnostic_source_rows"));const double rms_before=diagnostic_rms(roots_before);
 std::array<double,3>delta{{0,0,0}};std::vector<double>stations,rows;stations.reserve(fit_rows.size());rows.reserve(fit_rows.size());for(auto i:fit_rows){stations.push_back(b.core.station[i]);rows.push_back(b.core.row[i]);}
 ceres::Problem tiny;for(std::size_t j=0;j<fit_rows.size();++j){const auto i=fit_rows[j];const auto&r=b.core.packet.rows[i];RowCost functor;functor.model=&b.core.model;functor.source=&r;functor.layout=r.layout(b.core.model);functor.sqrt_rho=std::sqrt(rho);functor.RS_shift=RS_nu.at(i)/rho;
  const int f=family(b.core.model.materials.at(r.material).label);const double weight=total_weight*(1./b.core.model.shots.size())*(1./families.at(mode.shot).size())*(1./exposures.at({mode.shot,f}))/counts.at({mode.shot,f,r.pts});functor.pixel_scale=std::sqrt(weight)/r.E;
  for(int id:functor.layout.block_ids){functor.values.push_back(b.core.model.blocks[id].values);const auto it=by_block.find(id);functor.coefficients.push_back(it==by_block.end()?std::array<double,3>{{0,0,0}}:it->second);}
  auto cost=event_autodiff_cost(std::move(functor),{3,1,1},3);tiny.AddResidualBlock(cost.release(),nullptr,delta.data(),&stations[j],&rows[j]);
  const auto intervals=r.station_intervals(b.core.model,r.path);if(intervals.empty())throw std::runtime_error("source camera mode original station domain empty");tiny.SetParameterLowerBound(&stations[j],0,intervals.front()[0]);tiny.SetParameterUpperBound(&stations[j],0,intervals.back()[1]);tiny.SetParameterLowerBound(&rows[j],0,0);tiny.SetParameterUpperBound(&rows[j],0,1079);
 }
 std::vector<int>fixed_delta;for(int axis=0;axis<3;++axis){if(mode.lower[axis]==mode.upper[axis]){if(mode.lower[axis]!=0)throw std::runtime_error("source camera mode fixed delta is not current state");fixed_delta.push_back(axis);}else{if(std::isfinite(mode.lower[axis]))tiny.SetParameterLowerBound(delta.data(),axis,mode.lower[axis]);if(std::isfinite(mode.upper[axis]))tiny.SetParameterUpperBound(delta.data(),axis,mode.upper[axis]);}}
 if(!fixed_delta.empty())tiny.SetManifold(delta.data(),new ceres::SubsetManifold(3,fixed_delta));
 auto tiny_options=current_solver_options;tiny_options.linear_solver_type=ceres::DENSE_QR;tiny_options.dense_linear_algebra_library_type=ceres::EIGEN;tiny_options.num_threads=1;tiny_options.use_inner_iterations=false;tiny_options.callbacks.clear();tiny_options.update_state_every_iteration=false;tiny_options.minimizer_progress_to_stdout=false;tiny_options.logging_type=ceres::SILENT;tiny_options.linear_solver_ordering.reset();tiny_options.inner_iteration_ordering.reset();
 ceres::Solver::Summary tiny_summary;ceres::Solve(tiny_options,&tiny,&tiny_summary);
 eventio::Json proposals=eventio::Json::array();for(std::size_t k=0;k<mode.blocks.size();++k){const auto&block=b.core.model.blocks[mode.blocks[k]];for(int axis=0;axis<3;++axis)if(mode.coefficients[k][axis]!=0){const double proposed=block.values[axis]+mode.coefficients[k][axis]*delta[axis];if(!std::isfinite(proposed)||proposed<block.lower[axis]-1e-12||proposed>block.upper[axis]+1e-12)throw std::runtime_error("source camera tiny solution exceeded original coefficient bounds");proposals.push_back({{"block_index",mode.blocks[k]},{"block",block.name},{"axis",axis},{"mode_coefficient",mode.coefficients[k][axis]},{"before",block.values[axis]},{"proposed",std::max(block.lower[axis],std::min(block.upper[axis],proposed))}});}}
 std::ofstream tiny_ledger(out/"tiny_source_rows.jsonl");for(std::size_t j=0;j<fit_rows.size();++j){const auto&r=b.core.packet.rows[fit_rows[j]];tiny_ledger<<eventio::Json({{"row_index",fit_rows[j]},{"source_row_id",r.source_row_id},{"native_pts",r.pts},{"source_uv",r.uv},{"old_station",b.core.station[fit_rows[j]]},{"tiny_station",stations[j]},{"old_row",b.core.row[fit_rows[j]]},{"tiny_row",rows[j]}}).dump()<<'\n';}tiny_ledger.close();if(!tiny_ledger)throw std::runtime_error("source camera tiny ledger write failed");
 if(!before.matches()||compact::row_tokens(b,preserved)!=tokens_before)throw std::runtime_error("source camera tiny solve mutated production state");
 auto destroy=[&](){destroy_graph();if(problem)throw std::runtime_error("source camera mode retained live graph before mutation");};
 auto rebuild=[&](const char*name){if(problem)throw std::runtime_error("source camera rebuild requires destroyed graph");rebuild_graph(out/name);if(!problem||problem->NumResiduals()!=residual_count||problem->NumParameters()!=parameter_count||compact::effective_parameters(*problem)!=effective_count)throw std::runtime_error("source camera changed complete residual/free-parameter denominator or stage");};
 bool accepted=false,mutated=false,rolled_back=false;double candidate_cost=NAN,restored_cost=NAN,rms_candidate=INFINITY;eventio::Json candidate_debt,roots_candidate,initialization;std::string candidate_error;
 auto rollback=[&](){if(problem)destroy();before.restore();for(const auto&s:selected){auto&r=b.core.packet.rows[s.i];r.path=s.path;r.station=s.source_station;r.row=s.source_row;r.alpha=s.source_alpha;b.core.station[s.i]=s.station;b.core.row[s.i]=s.row;b.core.alpha[s.i]=s.alpha;}
  if(!before.matches()||compact::row_tokens(b,preserved)!=tokens_before)throw std::runtime_error("source camera exact rollback failed");rebuild("rollback_gpu_raw_uv_cache.jsonl");restored_cost=evaluate();if(std::abs(restored_cost-cost_before)>1e-8*std::max(1.,std::abs(cost_before)))throw std::runtime_error("source camera rollback complete objective mismatch");rolled_back=true;mutated=false;};
 if(tiny_summary.IsSolutionUsable()&&delta!=std::array<double,3>{{0,0,0}}){try{
  mutated=true;destroy();for(const auto&p:proposals)b.core.model.blocks.at(p.at("block_index").get<int>()).values.at(p.at("axis").get<int>())=p.at("proposed").get<double>();
  initialization=initialize(preserved,out/"native_auxiliary_navigation");if(compact::row_tokens(b,preserved)!=tokens_before)throw std::runtime_error("source camera initializer changed immutable source or unselected auxiliaries");
  for(const auto&s:selected){const auto&r=b.core.packet.rows[s.i];const auto&material=b.core.model.materials.at(r.material);const bool whole_board=(material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty();
   if((r.path!=s.path&&!whole_board)||!r.station_allowed(b.core.model,r.path,b.core.station[s.i],1e-12)||!std::isfinite(b.core.row[s.i])||b.core.row[s.i]<0||b.core.row[s.i]>1079||(r.kind==1&&std::memcmp(&r.station,&s.source_station,sizeof(double)))||(r.query&&(!std::isfinite(b.core.alpha[s.i])||b.core.alpha[s.i]<r.query->alpha.front()||b.core.alpha[s.i]>r.query->alpha.back())))throw std::runtime_error("source camera initializer changed original finite source/Query support");
  }
  rebuild("candidate_gpu_raw_uv_cache.jsonl");candidate_cost=evaluate();candidate_debt=constraint_debt();roots_candidate=compact::roots(b,spec.at("diagnostic_source_rows"));rms_candidate=diagnostic_rms(roots_candidate);
  accepted=candidate_cost<cost_before&&compact::valid_roots(roots_candidate)>=compact::valid_roots(roots_before)&&rms_candidate<rms_before;
 }catch(const std::exception&e){candidate_error=e.what();accepted=false;}}
 if(!accepted&&mutated)rollback();
 const eventio::Json receipt={{"schema","rinkfit.source_supported_preserve_anchor_camera_mode_navigation.v1"},{"source_packet_sha256",b.core.packet.source_hash},{"source_evidence",spec.at("source_evidence")},{"shot_id",shot.id},{"preserved_anchor_native_pts",spec.at("preserved_anchor_native_pts")},{"target_native_pts",spec.at("target_native_pts")},{"mode_anchor_responses",mode.anchor_responses},{"delta_yaw_pitch_logf",delta},{"control_proposals",proposals},{"original_qualified_pixel_rows",fit_rows.size()},{"all_selected_layout_consumers",selected.size()},{"tiny_residuals",tiny.NumResiduals()},{"tiny_parameters",tiny.NumParameters()},{"tiny_linear_solver","DENSE_QR/EIGEN float64"},{"tiny_summary",tiny_summary.BriefReport()},{"tiny_solution_usable",tiny_summary.IsSolutionUsable()},{"tiny_initial_cost",tiny_summary.initial_cost},{"tiny_final_cost",tiny_summary.final_cost},{"tiny_constant_dual_square_term_not_used_for_full_acceptance",true},{"current_rho",rho},{"tiny_uses_original_RS_dual_shift",true},{"full_current_PHR_cost_before",cost_before},{"full_current_PHR_cost_candidate",std::isfinite(candidate_cost)?eventio::Json(candidate_cost):eventio::Json()},{"original_constraint_debt_before",debt_before},{"original_constraint_debt_candidate",candidate_debt},{"diagnostic_source_roots_before",roots_before},{"diagnostic_source_roots_candidate",roots_candidate},{"diagnostic_native_pixel_RMS_before",std::isfinite(rms_before)?eventio::Json(rms_before):eventio::Json()},{"diagnostic_native_pixel_RMS_candidate",std::isfinite(rms_candidate)?eventio::Json(rms_candidate):eventio::Json()},{"accepted",accepted},{"exact_rollback_performed",rolled_back},{"rollback_complete_cost",std::isfinite(restored_cost)?eventio::Json(restored_cost):eventio::Json()},{"candidate_error",candidate_error},{"auxiliary_initializer_receipt",initialization},{"source_UV_ID_material_Query_W_E_geometry_rig_optics_or_precision_changed",false},{"all_original_production_free_coordinates_retained",true},{"trajectory_or_geometry_prior_or_smoothing_added",false},{"tiny_row_states_are_navigation_only_not_published",true},{"whole_problem_acceptance_required",true},{"complete_checkpoint_required_from_caller",true},{"tiny_source_rows_sha256",eventio::file_sha256(out/"tiny_source_rows.jsonl")},{"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
 eventio::atomic_json(out/"receipt.json",receipt);return receipt;
}
} // namespace rinkfit
