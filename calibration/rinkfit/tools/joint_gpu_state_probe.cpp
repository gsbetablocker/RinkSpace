// Actual-state raw-UV CPU/GPU and CPU fallback derivative checks. Never solves.
#include "rinkfit/joint_gpu_batch_reuse.hpp"
#include "rinkfit/joint_shared_print_restore_v1.hpp"
#include "rinkfit/joint_representative_revision.hpp"
#include "rinkfit/joint_dense_capture_weights_v1.hpp"
#include <ceres/jet.h>
#include <ceres/ceres.h>
#include <fstream>
#include <iostream>
#include <cstring>
using namespace rinkfit;
using J=ceres::Jet<double,1>;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--withdrawal-ancestry"){
  const auto fixture=eventio::read_json(argv[2]);const auto rows=reviewed_joint_dense_withdrawals_v1(fixture.at("packet"),fixture.at("qualification"),fixture.at("raw_rows"));eventio::Json result={{"passed",true},{"Solve_started",false},{"withheld_rows",std::vector<std::size_t>(rows.begin(),rows.end())}};eventio::atomic_json(argv[3],result);std::cout<<result.dump()<<std::endl;return 0;
 }
 if(argc==5&&std::string(argv[1])=="--identity-cost"){
  auto model=JointEventModel::load(argv[2]);const auto fixture=eventio::read_json(argv[3]);JointPacket packet;packet.source_hash=fixture.at("source_packet_sha256");packet.source_json={{"probe_only",true}};JointProblemBuilder core(std::move(model),std::move(packet));
  eventio::Json rows=eventio::Json::array();long double before=0,after=0;
  for(const auto&v:fixture.at("rows")){eventio::Json item={{"raw_row",v.at("raw_row")},{"weight",v.at("weight")},{"E",v.at("E")}};
   for(const auto&phase:{"before","after"}){const auto&z=v.at(phase);JointObservation r;r.shot_id=v.at("shot");r.pts=v.at("pts");r.material=z.at("material");r.path=z.at("path");r.kind=v.at("kind");r.uv=v.at("uv").get<Vec2<double>>();r.E=v.at("E");r.station=z.at("station");r.row=v.at("row");r.width_block=v.value("width_block",-1);r.width_sign=v.value("width_sign",0.);const auto l=r.layout(core.model);auto pointers=core.model.pointers(l);const auto q=core.model.predict_for_al(l,r.station,r.row,pointers.data());if(!q.arithmetic_valid||!finite_array(q.uv))throw std::runtime_error("nonfinite identity cost witness");const double error2=(q.uv[0]-r.uv[0])*(q.uv[0]-r.uv[0])+(q.uv[1]-r.uv[1])*(q.uv[1]-r.uv[1]);const double cost=.5*v.at("weight").get<double>()*error2/(r.E*r.E);item[phase]={{"material",r.material},{"path",r.path},{"station",r.station},{"predicted_UV",q.uv},{"raw_error_px",std::sqrt(error2)},{"weighted_half_cost",cost}};if(std::string(phase)=="before")before+=cost;else after+=cost;}
   rows.push_back(std::move(item));}
  eventio::Json result={{"schema","rinkfit.same_frozen_model_identity_cost.v1"},{"Solve_started",false},{"physical_model_unchanged",true},{"source_UV_E_W_unchanged",true},{"weighted_half_cost_before",double(before)},{"weighted_half_cost_after",double(after)},{"rows",std::move(rows)}};eventio::atomic_json(argv[4],result);std::cout<<"identity-cost rows="<<fixture.at("rows").size()<<" before="<<double(before)<<" after="<<double(after)<<std::endl;return 0;
 }
 if(argc==5&&std::string(argv[1])=="--identity-ancestry"){
  const auto cfg=eventio::read_json(argv[3]);auto pin=[](const eventio::Json&r){const auto file=r.at("file").get<std::string>();eventio::verify(file,r.at("sha256"));return file;};
  const auto qualification=eventio::read_json(pin(cfg.at("dense_source_qualification")));
  auto result=representative_identity_ancestry_v1(argv[2],qualification.at("source_packet"),cfg.at("representative_material_rebind_v1"),cfg.at("dense_source_qualification"),pin);
  result["reviewed_rows"]=result.at("rows").size();result.erase("rows");result["passed"]=true;result["Solve_started"]=false;eventio::atomic_json(argv[4],result);std::cout<<result.dump()<<std::endl;return 0;
 }
 if(argc!=6)throw std::runtime_error("usage MODEL FIXTURE PTX CONFIG OUT");
 const auto out=std::filesystem::path(argv[5]);std::filesystem::create_directories(out);
 auto m=JointEventModel::load(argv[1]);const auto fixture=eventio::read_json(argv[2]),cfg=eventio::read_json(argv[4]);
 const auto restoration=restore_joint_original_print_shapes_v1(m,cfg,false);
 JointPacket p;p.source_hash=fixture.at("source_packet_sha256");p.source_json={{"probe_only",true}};
 for(const auto&v:fixture.at("rows")){JointObservation r;r.source_row_id=std::to_string(v.at("raw_row").get<std::size_t>());r.source_dataset_id="actual-state-probe-subset";r.shot_id=v.at("shot");r.pts=v.at("pts");r.material=v.at("material");r.path=v.at("path");r.kind=v.at("kind");r.uv=v.at("uv").get<Vec2<double>>();r.E=v.at("E");r.station=v.at("station");r.row=v.at("row");r.active=true;r.fit_role="fit";p.rows.push_back(r);}
 JointProblemBuilder core(std::move(m),std::move(p));
 auto token=[&](){std::uint64_t h=1469598103934665603ull;for(const auto&b:core.model.blocks)for(double v:b.values){std::uint64_t x;std::memcpy(&x,&v,8);h=(h^x)*1099511628211ull;}return h;};const auto initial=token();
 ceres::Problem activity;std::map<int,std::vector<int>>released;
 if(cfg.contains("runtime_shared_parameter_stage_v2"))for(const auto&r:cfg.at("runtime_shared_parameter_stage_v2").at("blocks"))released[r.at("block_index")]=r.at("free_ambient_indices").get<std::vector<int>>();
 for(std::size_t bi=0;bi<core.model.blocks.size();++bi){auto&b=core.model.blocks[bi];auto*x=b.values.data();activity.AddParameterBlock(x,int(b.values.size()));auto found=released.find(int(bi));bool constant=b.constant&&found==released.end();for(const auto&name:cfg.value("temporary_runtime_fixed_physical_blocks",std::vector<std::string>{}))constant=constant||b.name==name;for(const auto&prefix:cfg.value("temporary_runtime_fixed_physical_prefixes",std::vector<std::string>{}))constant=constant||b.name.rfind(prefix,0)==0;
  if(constant){activity.SetParameterBlockConstant(x);continue;}std::vector<int>fixed=b.fixed_indices;if(found!=released.end()){fixed.clear();for(int k=0;k<int(b.values.size());++k)if(std::find(found->second.begin(),found->second.end(),k)==found->second.end())fixed.push_back(k);}if(!fixed.empty())activity.SetManifold(x,new ceres::SubsetManifold(int(b.values.size()),fixed));}
 double budget=0;auto cache=make_joint_gpu_batch_reuse_uv_cache(core,&budget,argv[3],4,out/"cache.jsonl","ice_line_width_v1");cache->configure_for_problem(activity);cache->PrepareForEvaluation(true,true);
 std::size_t compared=0,fallback=0,fd_columns=0,fd_failed=0;double max_J=0,max_primal=0,max_fd=0;eventio::Json failures=eventio::Json::array(),coverage=eventio::Json::object(),optical_coverage=eventio::Json::object();
 for(int pass=0;pass<2;++pass){
  if(pass){for(auto&b:core.model.blocks){activity.SetParameterBlockVariable(b.values.data());activity.SetManifold(b.values.data(),nullptr);}cache->configure_for_problem(activity);cache->PrepareForEvaluation(true,true);}
  std::size_t pass_gpu=0,pass_pruned=0;
  for(std::size_t i=0;i<core.packet.rows.size();++i){const auto&r=core.packet.rows[i];const auto l=r.layout(core.model);const auto&shape=core.model.materials[r.material].paths[r.path];
   std::vector<std::vector<J>>store;std::vector<const J*>jp;std::vector<const double*>dp;
   for(int id:l.block_ids){store.emplace_back();for(double x:core.model.blocks[id].values)store.back().emplace_back(x);dp.push_back(core.model.blocks[id].values.data());}for(const auto&v:store)jp.push_back(v.data());
   JointUVCacheView view;const bool gpu=cache->lookup(i,&view)&&view.jacobians_ready;double row_worst=0;std::string bad;
   auto ad=[&](double ds=0.,double dr=0.){J s(r.kind==1?r.station:core.station[i]),row(core.row[i]);s.v[0]=ds;row.v[0]=dr;return core.model.predict_for_al(l,s,row,jp.data());};
   if(gpu){++pass_gpu;auto primal=ad();for(int k=0;k<2;++k)max_primal=std::max(max_primal,std::abs(primal.uv[k].a-view.canonical_AD_uv[k]));
    auto compare=[&](int col,double ds=0.,double dr=0.){if(view.partial_columns->at(col)<0){++pass_pruned;return;}const auto q=ad(ds,dr);const auto&v=view.partial(col);if(!q.arithmetic_valid||!v.arithmetic||v.status!=int(q.physical_status)){bad="arithmetic/status mismatch";return;}const double e=std::max(std::abs(v.u-q.uv[0].v[0])/(1+std::abs(q.uv[0].v[0])),std::abs(v.v-q.uv[1].v[0])/(1+std::abs(q.uv[1].v[0])));row_worst=std::max(row_worst,e);max_J=std::max(max_J,e);++compared;};
    for(std::size_t k=0;k<store.size();++k)for(std::size_t c=0;c<store[k].size();++c){store[k][c].v[0]=1.;compare(view.block_column_starts->at(k)+int(c));store[k][c].v[0]=0.;}compare(view.station_column,1.,0.);compare(view.row_column,0.,1.);
    if(row_worst>2e-8)bad="CPU/GPU ambient derivative disagreement";
   }else if(!pass){++fallback;
    // Independent central differences for each local physical dependency family.
    std::set<int>slots;for(int k:{l.scene_slot,l.rig_slot,l.optics_slot,l.reference_angles_slot,l.geometry_slot,l.advertisement_slot,l.width_slot})if(k>=0)slots.insert(k);if(!l.control_slots.empty())slots.insert(l.control_slots.front());
    for(int k:slots){const int id=l.block_ids[k];int c=0;if(k==l.geometry_slot)c=l.geometry_indices.empty()?l.geometry_offset:l.geometry_indices.front();auto&x=core.model.blocks[id].values.at(c);const double old=x,h=1e-7*std::max(1.,std::abs(old));store[k][c].v[0]=1.;auto exact=ad();store[k][c].v[0]=0.;double best=INFINITY;
     for(double step:{h,h*.25}){x=old+step;auto plus=core.model.predict_for_al(l,core.station[i],core.row[i],dp.data());x=old-step;auto minus=core.model.predict_for_al(l,core.station[i],core.row[i],dp.data());x=old;if(exact.arithmetic_valid&&plus.arithmetic_valid&&minus.arithmetic_valid){double e=0;for(int z=0;z<2;++z)e=std::max(e,std::abs((plus.uv[z]-minus.uv[z])/(2*step)-exact.uv[z].v[0])/(1+std::abs(exact.uv[z].v[0])));best=std::min(best,e);}}
     ++fd_columns;max_fd=std::max(max_fd,best);if(!(best<=1e-5)){++fd_failed;bad="CPU fallback finite difference disagreement";row_worst=std::max(row_worst,best);}
    }
   }
   if(!pass){
    const auto g=core.model.global(l,dp.data());std::vector<const double*>frozen;std::vector<double>geometry;
    const auto legacy=core.model.legacy_layout(l,frozen,g,dp.data(),geometry);const auto&math=core.model.shots.at(l.shot_index).math;
    const auto trajectory=math.trajectory(row_time(l.frame_time,g[31],core.row[i]),frozen.data(),legacy);
    const bool wide=std::any_of(math.wide_coefficients.begin(),math.wide_coefficients.end(),[](double x){return x!=0.;})&&trajectory[2]<math.wide_log_f_boundary;
    std::string optical=core.model.shots.at(l.shot_index).rig_id+(wide?"/wide_active":"/affine_optics");auto&ov=optical_coverage[optical];if(ov.is_null())ov=0;ov=ov.get<int>()+1;
    std::string key=r.shot_id+"/kind"+std::to_string(shape.kind);auto&v=coverage[key];if(v.is_null())v={{"rows",0},{"GPU_rows",0},{"CPU_fallback_rows",0}};v["rows"]=v.at("rows").get<int>()+1;const char*which=gpu?"GPU_rows":"CPU_fallback_rows";v[which]=v.at(which).get<int>()+1;}
   if(!bad.empty())failures.push_back({{"raw_row",fixture.at("rows").at(i).at("raw_row")},{"shot",r.shot_id},{"native_pts",r.pts},{"material",r.material},{"path",r.path},{"primitive",shape.kind},{"pass",pass},{"reason",bad},{"max_scaled_error",row_worst}});
  }
  std::cout<<"pass="<<pass<<" rows="<<core.packet.rows.size()<<" GPU="<<pass_gpu<<" pruned_columns="<<pass_pruned<<" compared="<<compared<<std::endl;
 }
 const bool restored=initial==token();const bool passed=failures.empty()&&compared>0&&max_primal<=1e-8&&restored;eventio::Json result={{"schema","rinkfit.actual_state_raw_UV_derivative_probe.v1"},{"passed",passed},{"model_sha256",eventio::file_sha256(argv[1])},{"fixture_sha256",eventio::file_sha256(argv[2])},{"ptx_sha256",eventio::file_sha256(argv[3])},{"selected_rows",core.packet.rows.size()},{"CPU_GPU_columns_compared",compared},{"CPU_fallback_rows",fallback},{"CPU_fallback_finite_difference_columns",fd_columns},{"CPU_fallback_finite_difference_failures",fd_failed},{"max_scaled_CPU_GPU_J_error",max_J},{"max_CPU_GPU_primal_error",max_primal},{"max_scaled_finite_difference_error",max_fd},{"actual_stage_then_full_ambient_activity",true},{"parameters_restored_bitwise",restored},{"runtime_shape_restoration",restoration},{"coverage",coverage},{"optical_coverage",optical_coverage},{"failures",failures},{"cache",cache->report()},{"Solve_started",false},{"quality_accepted",false},{"source_E_W_or_physical_domain_changed",false}};eventio::atomic_json(out/"receipt.json",result);std::cout<<result.dump()<<std::endl;return passed?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
