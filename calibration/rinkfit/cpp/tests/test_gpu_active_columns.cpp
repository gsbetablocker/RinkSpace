#include "rinkfit/joint_model.hpp"
#include <iostream>
#include <iomanip>

using namespace rinkfit;
EventParameterBlock block(std::string name, std::vector<double> values) {
    EventParameterBlock b; b.name=name; b.values=values;
    b.lower.assign(values.size(),-100); b.upper.assign(values.size(),100);
    b.scales.assign(values.size(),1); return b;
}
JointEventModel fixture(int kind) {
    JointEventModel m; m.scene_block=0;
    m.blocks.push_back(block("scene/rink5",{.5,.1,.1,2.,.02}));
    std::vector<double> head(15,0); head[6]=0;head[7]=-1;head[8]=.04;
    m.blocks.push_back(block("rig/main/head15",head));
    std::vector<double> optics(12,0); optics[0]=959.5;optics[1]=539.5;
    m.blocks.push_back(block("rig/main/optics12",optics));
    m.blocks.push_back(block("shot/s01/reference_angles",{0,0}));
    m.blocks.push_back(block("print_component/test", kind==6?
        std::vector<double>{.35,.015,.45,.015}:
        std::vector<double>{.4,.012,.04,0,.005}));
    JointRig r; r.id="main";r.head_block=1;r.optics_block=2;
    r.math.R0_seed={1,0,0,0,0,-1,0,1,0};
    r.math.n_seed={0,0,1};r.math.b_seed={1,0,0};
    r.math.n_tangent1={1,0,0};r.math.n_tangent2={0,1,0};
    r.math.f_ref=1000;r.math.radius_max=2;r.math.tau_bound=.02;
    m.rigs.push_back(r);m.rig_map["main"]=0;
    JointShot s; s.id="s01";s.rig_id="main";s.first_pts=0;s.last_pts=2560;s.time_origin_pts=0;
    s.reference_angles_block=3;s.math=r.math;s.math.dt=.1;s.math.first_control_time=-.2;
    s.math.n_controls=8;s.math.dependent_control=2;
    s.math.reference_indices={1,2,3,4};s.math.reference_weights={1./6,4./6,1./6,0};
    for(int i=0;i<8;++i){int j=m.blocks.size();m.blocks.push_back(block("control/"+std::to_string(i),{0,0,std::log(1000.)}));s.control_blocks.push_back(j);s.control_sources.push_back(j);s.control_offsets.push_back(0);}
    m.shots.push_back(s);m.shot_map["s01"]=0;
    EventPath p;p.id="print";p.variant_id="print";p.kind=kind;p.geometry_block=4;
    EventMaterial material;material.label="print";material.paths.push_back(p);m.materials.push_back(material);
    m.path_geometry_offsets[{0,0}]=0;m.path_geometry_indices[{0,0}]={};m.path_station_intervals[{0,0}]={{0,1}};
    return m;
}

#include "rinkfit/joint_gpu_cache.hpp"
#include "rinkfit/joint_gpu_batch_reuse.hpp"
#include "rinkfit/joint_source_uv_capture_v1.hpp"
#include "rinkfit/joint_gpu_active_columns.hpp"
#include <chrono>
#include <iomanip>

namespace {
void check_active(bool pass,const std::string&message){if(!pass)throw std::runtime_error(message);}
std::size_t gpu_jobs(const eventio::Json&r){std::size_t n=0;for(const auto&b:r.at("batch_state"))n+=b.at("jobs").get<std::size_t>();return n;}
}
int main(int argc,char**argv){try{
 check_active(argc==3,"usage PTX NEW_OUTPUT");const std::filesystem::path out=argv[2];
 check_active(!std::filesystem::exists(out),"immutable regression output exists");std::filesystem::create_directories(out);
 double max_delta=0,primal_delta=0,full_kernel=0,pruned_kernel=0;std::size_t comparisons=0,before_jobs=0,after_jobs=0,before_bytes=0,after_bytes=0;
 eventio::Json cases=eventio::Json::array();
 {double quaternion[4]={1,0,0,0};ceres::Problem nonlinear;nonlinear.AddParameterBlock(quaternion,4);nonlinear.SetManifold(quaternion,new ceres::QuaternionManifold);auto active=joint_gpu_ambient_activity(nonlinear,quaternion,4);check_active(std::all_of(active.begin(),active.end(),[](auto v){return v==1;}),"pointwise-zero nonlinear manifold row was incorrectly pruned");}
 for(int path_kind:{0,1,2,3})for(bool implicit:{false,true}){
  auto model=fixture(6);auto&path=model.materials[0].paths[0];path.kind=path_kind;
  if(path_kind==0)model.blocks[4].values={-.08,0,.08,0};
  if(path_kind==1)model.blocks[4].values={0,0,.08};
  if(path_kind==2){model.blocks[4].values={1,0,0,0,1,0};path.advertisement_block=int(model.blocks.size());model.blocks.push_back(block("ad/test",{.35,.1,.002,.018}));path.points={{0,0},{1,0},{1,1},{0,1},{0,0}};}
  if(path_kind==3){model.blocks[4].values={.65};path.fixed_height_fraction=.65;path.boundary_segment=0;}
  model.blocks[2].values[11]=.012;
  for(int k=0;k<8;++k)model.blocks[model.shots[0].control_blocks[k]].values={.003*(k-2),-.002*(k-2),std::log(1000.)+.005*(k-2)};
  JointPacket packet;packet.source_hash=std::string(64,'a');
  for(auto pts:{std::int64_t(0),std::int64_t(1280),std::int64_t(2560)})for(double station:{.07,.25,.5,.83}){
   JointObservation row;row.active=true;row.shot_id="s01";row.pts=pts;row.material=0;row.path=0;row.kind=0;row.station=station;row.row=539.5;row.source_row_id="static/"+std::to_string(packet.rows.size());packet.rows.push_back(row);
  }
  JointProblemBuilder core(std::move(model),std::move(packet));double budget=1;
  const auto prefix=std::to_string(path_kind)+(implicit?"_RS":"_explicit");
  auto cache=implicit?make_joint_source_uv_capture_gpu_cache_v1(core,&budget,argv[1],1,out/(prefix+".jsonl"),"ice_line_width_v1"):make_joint_gpu_batch_reuse_uv_cache(core,&budget,argv[1],1,out/(prefix+".jsonl"),"ice_line_width_v1");
  cache->PrepareForEvaluation(true,true);auto full_report=cache->report();check_active(!full_report.at("disabled_to_CPU").get<bool>(),"full GPU cache disabled");
  std::vector<JointUVCacheView>before;std::vector<std::vector<JointUVPartial>>saved(core.packet.rows.size());std::vector<std::vector<int>>starts;
  for(std::size_t raw=0;raw<core.packet.rows.size();++raw){JointUVCacheView v;check_active(cache->lookup(raw,&v)&&v.jacobians_ready,"full derivative row unavailable");auto l=core.packet.rows[raw].layout(core.model);for(std::size_t k=0;k<l.block_ids.size();++k)for(std::size_t c=0;c<core.model.blocks[l.block_ids[k]].values.size();++c)saved[raw].push_back(v.partial(v.block_column_starts->at(k)+int(c)));saved[raw].push_back(v.partial(v.station_column));saved[raw].push_back(v.partial(v.row_column));before.push_back(v);starts.push_back(*v.block_column_starts);}
  ceres::Problem problem;for(auto&b:core.model.blocks)problem.AddParameterBlock(b.values.data(),int(b.values.size()));
  for(std::size_t k=0;k<core.model.blocks.size();++k){auto&b=core.model.blocks[k];if(k<5||k>=13)problem.SetParameterBlockConstant(b.values.data());else problem.SetManifold(b.values.data(),new ceres::SubsetManifold(int(b.values.size()),{1}));}
  // One affine control manifold has a truly fixed ambient row.
  auto&vblock=core.model.blocks[5];problem.SetManifold(vblock.values.data(),new JointLinearControlManifold({{"ambient_size",3},{"tangent_size",1},{"basis",{{1.},{0.},{0.}}}}));
  cache->configure_for_problem(problem);cache->PrepareForEvaluation(true,true);auto compact_report=cache->report();check_active(compact_report.at("actual_problem_activity_bound").get<bool>()&&!compact_report.at("disabled_to_CPU").get<bool>(),"actual active layout not used");
  for(std::size_t raw=0;raw<core.packet.rows.size();++raw){JointUVCacheView now;check_active(cache->lookup(raw,&now)&&now.jacobians_ready,"compact derivative row unavailable");auto l=core.packet.rows[raw].layout(core.model);std::size_t at=0;
   for(std::size_t k=0;k<l.block_ids.size();++k){auto&b=core.model.blocks[l.block_ids[k]];auto active=joint_gpu_ambient_activity(problem,b.values.data(),int(b.values.size()));for(std::size_t c=0;c<b.values.size();++c,++at){const int column=now.block_column_starts->at(k)+int(c);if(!active[c]){check_active(now.partial_columns->at(column)==-1,"fixed ambient coordinate still has a GPU job");continue;}for(int d=0;d<2;++d){const auto&a=saved[raw][at];const auto&b=now.partial(column);const double expected=d?a.v:a.u,value=d?b.v:b.u;max_delta=std::max(max_delta,std::abs(value-expected)/(1+std::abs(expected)));++comparisons;}}}
   for(int c:{now.station_column,now.row_column}){const auto&a=saved[raw][at++];const auto&b=now.partial(c);for(int d=0;d<2;++d){const double expected=d?a.v:a.u,value=d?b.v:b.u;max_delta=std::max(max_delta,std::abs(value-expected)/(1+std::abs(expected)));++comparisons;}}
   for(int d=0;d<2;++d)primal_delta=std::max(primal_delta,std::abs(before[raw].canonical_uv[d]-now.canonical_uv[d]));
   // Exercise the production implicit derivative lift against uncached CPU AD.
   if(implicit){auto source=std::make_shared<JointObservation>(core.packet.rows[raw]);JointSourceUVCaptureRowV1 functor;functor.source=source;functor.implicit.model=&core.model;functor.implicit.observation=source.get();functor.implicit.layout=l;functor.cache=cache.get();functor.cache_row=raw;
    using Jet=ceres::Jet<double,1>;std::vector<std::vector<Jet>>storage;std::vector<const Jet*>p;
    for(std::size_t k=0;k<l.block_ids.size();++k){auto&b=core.model.blocks[l.block_ids[k]];auto active=joint_gpu_ambient_activity(problem,b.values.data(),int(b.values.size()));storage.emplace_back();for(std::size_t c=0;c<b.values.size();++c){storage.back().emplace_back(b.values[c]);storage.back().back().v[0]=active[c]?.0001*(1+c):0.;}functor.implicit.parameter_sizes[int(k)]=int(b.values.size());p.push_back(storage.back().data());}
    Jet station(core.station[raw]);station.v[0]=.0001;functor.implicit.station_slot=int(p.size());functor.implicit.parameter_sizes[int(p.size())]=1;p.push_back(&station);
    Vec2<Jet>actual,expected;auto plain=functor;plain.cache=nullptr;check_active(functor.raw_uv(p.data(),&actual)&&plain.raw_uv(p.data(),&expected),"implicit lift invalid");for(int d=0;d<2;++d){max_delta=std::max(max_delta,std::abs(actual[d].v[0]-expected[d].v[0])/(1+std::abs(expected[d].v[0])));primal_delta=std::max(primal_delta,std::abs(actual[d].a-expected[d].a));++comparisons;}
   }
  }
  const auto first_generation=compact_report.at("derivative_layout_generation").get<std::size_t>();cache->configure_for_problem(problem);check_active(cache->report().at("derivative_layout_generation")==first_generation,"unchanged stage rebuilt layout");
  problem.SetParameterBlockVariable(core.model.blocks[2].values.data());cache->configure_for_problem(problem);cache->PrepareForEvaluation(true,true);const auto expanded=cache->report();check_active(expanded.at("derivative_layout_generation").get<std::size_t>()==first_generation+1&&gpu_jobs(expanded)>gpu_jobs(compact_report),"unlock did not rebuild and invalidate derivative layout");
  before_jobs+=gpu_jobs(full_report);after_jobs+=gpu_jobs(compact_report);
  before_bytes+=full_report.at("driver_total_metrics").at("D2H_bytes").get<std::size_t>();after_bytes+=compact_report.at("driver_total_metrics").at("D2H_bytes").get<std::size_t>()-full_report.at("driver_total_metrics").at("D2H_bytes").get<std::size_t>();
  full_kernel+=full_report.at("GPU_kernel_seconds").get<double>();pruned_kernel+=compact_report.at("GPU_kernel_seconds").get<double>()-full_report.at("GPU_kernel_seconds").get<double>();
  cases.push_back({{"path_kind",path_kind},{"implicit_RS",implicit},{"rows",core.packet.rows.size()},{"unpruned_jobs",gpu_jobs(full_report)},{"active_jobs",gpu_jobs(compact_report)},{"unpruned_device_bytes",full_report.at("driver_total_metrics").at("owned_device_current_bytes")},{"active_device_bytes",compact_report.at("driver_total_metrics").at("owned_device_current_bytes")},{"fixed_coordinates_have_no_jobs",true},{"stage_unlock_rebuilt_layout",true}});
 }
 check_active(max_delta<2e-8&&primal_delta<1e-8&&after_jobs<before_jobs&&after_bytes<before_bytes,"active derivative equivalence failed");
 eventio::Json report={{"schema","rinkfit.actual_GPU_active_column_regression.v1"},{"passed",true},{"cases",cases},{"comparisons",comparisons},{"max_scaled_derivative_delta",max_delta},{"max_primal_delta_px",primal_delta},{"unpruned_jobs",before_jobs},{"active_jobs",after_jobs},{"unpruned_D2H_bytes",before_bytes},{"active_D2H_bytes",after_bytes},{"small_fixture_unpruned_kernel_seconds",full_kernel},{"small_fixture_active_kernel_seconds",pruned_kernel},{"full_six_shot_speedup_measured",false},{"Solve_started",false},{"source_files_or_checkpoints_modified",false}};
 eventio::atomic_json(out/"report.json",report);std::cout<<report.dump(2);return 0;
 }catch(const std::exception&e){std::cerr<<e.what();return 1;}}
