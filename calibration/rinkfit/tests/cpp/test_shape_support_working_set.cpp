#include "rinkfit/joint_shape_support_working_set_v1.hpp"
#include "rinkfit/joint_solver_metric_v1.hpp"
#include "rinkfit/joint_gpu_active_columns.hpp"
#include <cassert>
#include <iostream>
using namespace rinkfit;
struct ScalarHeight final:ceres::SizedCostFunction<1,2>{
 double limit=10.;
 bool Evaluate(double const*const*p,double*r,double**j)const override{r[0]=p[0][1]-limit;if(j&&j[0]){j[0][0]=0;j[0][1]=1;}return true;}
};
struct RotatingShapeEvidence final:ceres::SizedCostFunction<3,2,1>{bool Evaluate(double const*const*p,double*r,double**j)const override{const double a=p[1][0];r[0]=p[0][1]-1.;r[1]=p[0][1]+a*p[0][0]-2.;r[2]=10.*(a-1.);if(j){if(j[0]){j[0][0]=0.;j[0][1]=1.;j[0][2]=a;j[0][3]=1.;j[0][4]=j[0][5]=0.;}if(j[1]){j[1][0]=0.;j[1][1]=p[0][0];j[1][2]=10.;}}return true;}};
struct MiniBuilder {
 struct Core {JointEventModel model;JointPacket packet;std::vector<double>station,row;}core;
 struct Cache final:JointUVCache{
  MiniBuilder*b;mutable JointUVPartial partials[5];std::vector<int>starts{0,2};mutable std::vector<int>column_map{0,1,2,3,4};mutable std::vector<std::pair<int,int>>sparse_map;int missing_mode=0;
  explicit Cache(MiniBuilder*x):b(x){}void PrepareForEvaluation(bool,bool)override{}
  bool lookup(std::size_t,JointUVCacheView*v)const override{
   const auto&a=b->core.model.blocks[0].values;const auto&c=b->core.model.blocks[1].values;const double s=b->core.station[0];
   v->canonical_uv=v->canonical_AD_uv={(1-s)*a[0]+s*c[0],(1-s)*a[1]+s*c[1]};
   partials[0]={1-s,0,1,0};partials[1]={0,1-s,1,0};partials[2]={s,0,1,0};partials[3]={0,s,1,0};partials[4]={c[0]-a[0],c[1]-a[1],1,0};
   column_map={0,1,2,3,4};if(missing_mode==1)column_map[1]=-1;if(missing_mode==2)column_map[4]=-1;v->partial_columns=&column_map;if(missing_mode==5){sparse_map={{0,0},{2,2},{3,3},{4,4}};v->partial_columns=nullptr;v->sparse_partial_columns=&sparse_map;}if(missing_mode==6)v->partial_columns=nullptr;v->partials=partials;v->block_column_starts=&starts;v->station_column=4;v->jacobians_ready=true;for(int k=0;k<5;++k){v->native_domain_available[k]=true;v->canonical_AD_native_domains[k]=-1;}return true;
  }
  eventio::Json report()const override{return eventio::Json::object();}
 }cache{this};
 bool source_uv_observed_row_initialization()const{return true;}ceres::EvaluationCallback*gpu_evaluation_callback(){return &cache;}
};
int main(int argc,char**argv){
 using Json=eventio::Json;
 // A line's tangential vertex displacement is exactly absorbed by its legal
 // station. Actual anisotropic per-row W is applied before elimination.
 JointShapeConditionalSupportV1 a;Eigen::Matrix2d J=Eigen::Matrix2d::Identity();Eigen::Vector2d js(1,0),e(0,-1),units(.02,.02);a.add(2.*J,2.*js,2.*e,2,true);auto d=a.decision(units);assert(d.rank==1&&std::abs(d.direction[0])<1e-14);assert(std::abs(a.H(1,1)-4.)<1e-14);
 JointShapeConditionalSupportV1 at_endpoint;at_endpoint.add(J,js,e,2,false);assert(at_endpoint.decision(units).rank==2);
 JointShapeConditionalSupportV1 fixed_station;fixed_station.add(3.*J,3.*js,3.*e,2,false);assert(fixed_station.decision(units).rank==2);
 JointShapeConditionalSupportV1 two_edges=a;two_edges.add(J,Eigen::Vector2d(0,1),e,2,true);assert(two_edges.decision(units).rank==2);
 JointShapeConditionalSupportV1 weak;J<<1e-8,0,0,1e-8;weak.add(J,js*1e-8,e*1e-8,2,true);assert(weak.decision(units).rank==1);
 JointShapeConditionalSupportV1 nonzero_second;J<<1,0,0,1e-5;nonzero_second.add(J,Eigen::Vector2d::Zero(),e,2,false);assert(nonzero_second.decision(Eigen::Vector2d::Ones()).rank==2);
 JointShapeConditionalSupportV1 zero;assert(zero.decision(units).rank==0);zero.protected_direction=true;assert(zero.decision(units).rank==2);
 MiniBuilder b;b.core.model.blocks.resize(3);for(int k=0;k<3;++k){b.core.model.blocks[k].name="vertex/"+std::to_string(k);b.core.model.blocks[k].values={double(k),0};b.core.model.blocks[k].scales={.02,.02};}
 b.core.model.materials.resize(1);b.core.model.materials[0].paths.resize(1);auto&path=b.core.model.materials[0].paths[0];path.kind=6;path.vertex_blocks={0,1,2};
 b.core.packet.rows.resize(1);auto&source=b.core.packet.rows[0];source.active=true;source.material=0;source.path=0;source.kind=0;source.E=2;source.uv={.4,1};b.core.station={.4};
 ceres::Problem::Options po;po.evaluation_callback=&b.cache;po.enable_fast_removal=true;ceres::Problem p(po);auto*x=b.core.model.blocks[0].values.data();auto*y=b.core.model.blocks[1].values.data();auto*s=b.core.station.data();
 JointExplicitRow row;row.model=&b.core.model;row.source=&source;row.layout.block_ids={0,1};row.layout.shape_vertex_count=3;row.layout.shape_vertex_slots={0,1};row.physical_slots={0,1};row.station_slot=2;row.fixed_row_value=0;row.uv_cache=&b.cache;
 BuildInventory inv;inv.statistical_blocks.push_back(p.AddResidualBlock(new JointEngineeringUVChainCost(JointEngineeringPixel{row,2.},{2,2,1}),nullptr,x,y,s));p.SetParameterLowerBound(s,0,0.);p.SetParameterUpperBound(s,0,1.);
 JointGlobalPHRStateEpochV1 state;
 Json policy={{"schema","rinkfit.engineering_global_phr_policy.v1"},{"initialization","explicit_new_objective_zero_duals"},{"initial_rho",10.},{"rho_growth",2.},{"required_progress_ratio",.5},{"primal_tolerance",1e-8},{"complementarity_tolerance",1e-8}};
 auto physics=std::make_shared<JointPhysicalFeasibilityV1>(state,policy,"fixture");auto*height=new ScalarHeight;physics->add_original_final_factor(p,"height",false,std::unique_ptr<ceres::CostFunction>(height),{x});physics->finish_direct_build(1,0);
 Json ledger={{"groups",Json::array({Json{{"actual_row_indices",Json::array({0})},{"actual_per_row_weight",Json::array({16.})}}})}};
 JointShapeSupportWorkingSetV1 working(b,inv,ledger,physics);
 // Both real observation chains initially see only the vertical vertex
 // direction. Once the camera moves, the horizontal direction is observable.
 // An initial conditional-nullspace hard gauge would keep x[0]=0 and leave
 // a positive residual even after the same reduced Solve converges.
 auto initial_support=working.prepare(b,p);double camera=0.;auto rotating=p.AddResidualBlock(new RotatingShapeEvidence,nullptr,x,&camera);ceres::Solver::Options solve;solve.linear_solver_type=ceres::DENSE_QR;solve.max_num_iterations=100;solve.function_tolerance=1e-13;solve.parameter_tolerance=0.;solve.gradient_tolerance=1e-12;ceres::Solver::Summary result;ceres::Solve(solve,&p,&result);std::cout<<"rotating actual shape evidence: same Solve final cost="<<result.final_cost<<"; horizontal vertex="<<x[0]<<std::endl;assert(result.IsSolutionUsable()&&result.final_cost<1e-16&&std::abs(x[0]-1.)<1e-7);p.RemoveResidualBlock(rotating);p.RemoveParameterBlock(&camera);working.restore(p);x[0]=x[1]=y[1]=0.;y[0]=1.;*s=.4;
 double before=0;assert(p.Evaluate({},&before,nullptr,nullptr,nullptr));auto report=working.prepare(b,p);assert(report.at("rank1_numerical_gauges")==0&&report.at("conditional_rank1_vertices")==2);assert(p.ParameterBlockTangentSize(x)==2&&p.ParameterBlockTangentSize(y)==2);const auto*chart=p.GetManifold(x);assert(joint_gpu_ambient_activity(p,x,2)==std::vector<unsigned char>({1,1}));double after=0;assert(p.Evaluate({},&after,nullptr,nullptr,nullptr)&&before==after);
 // Existing model units apply to both original physical directions.
 JointSolverMetricV1 metric;auto mr=metric.prepare(b,p);assert(p.ParameterBlockTangentSize(x)==2);assert(joint_gpu_ambient_activity(p,x,2)==std::vector<unsigned char>({1,1}));metric.restore(p);
 working.restore(p);assert(p.ParameterBlockTangentSize(x)==2);
 for(int repeat=0;repeat<10;++repeat){working.prepare(b,p);assert(p.GetManifold(x)==chart);working.restore(p);}
 p.SetParameterBlockConstant(y);working.prepare(b,p);working.restore(p);assert(p.IsParameterBlockConstant(y));p.SetParameterBlockVariable(y);
 // Whole-contour cardinality is three while every live row has only two
 // vertex slots. Missing/pruned ambient columns must not become fake zeros.
 for(int mode:{1,2,3,4,5,6}){b.cache.missing_mode=mode;b.cache.starts={0,2};if(mode==3)b.cache.starts[0]=-1;if(mode==4)b.cache.starts.resize(1);report=working.prepare(b,p);assert(p.ParameterBlockTangentSize(x)==2);if(mode==2||mode==6)assert(p.ParameterBlockTangentSize(y)==2);working.restore(p);}
 b.cache.missing_mode=0;b.cache.starts={0,2};
 // A changing correspondence/station corner restores the second data direction.
 *s=0.;report=working.prepare(b,p);assert(report.at("rank1_numerical_gauges")==0);working.restore(p);
 *s=.4;working.prepare(b,p);working.restore(p);source.uv={1.4,1};auto release=working.reconsider(b,p);assert(!release.at("restart_required").get<bool>());report=working.prepare(b,p);assert(report.at("rank1_numerical_gauges")==0);working.restore(p);
 // A path/cell generation releases old numerical history, then an active real
 // physical factor independently protects the affected vertex.
 ++b.core.packet.numerical_path_generation_v1;source.uv={.4,1};height->limit=-1.;report=working.prepare(b,p);assert(p.ParameterBlockTangentSize(x)==2&&p.ParameterBlockTangentSize(y)==2);working.restore(p);
 // No parameter value or real residual has been changed by gauge preparation.
 assert(x[0]==0&&x[1]==0&&y[0]==1&&y[1]==0&&*s==.4);
 if(argc>1){auto actual=eventio::read_json(argv[1]);int checked=0;for(const auto&v:actual.at("vertices")){auto scales=v.at("unit_scales").get<std::array<double,2>>();JointShapeConditionalSupportV1 support;for(int i=0;i<2;++i)for(int j=0;j<2;++j)support.H(i,j)=v.at("scaled_station_eliminated_gram").at(i).at(j).get<double>()/(scales[i]*scales[j]);auto decision=support.decision(Eigen::Vector2d(scales[0],scales[1]));assert(decision.rank==v.at("numerical_rank").get<int>());++checked;}assert(checked==5);std::cout<<"five previously measured actual-W vertex conditional Grams recognized; physical activity remains independently protected\n";}
 std::cout<<"conditional shape support: local ranks diagnosed; full original freedoms retained through actual camera changes; W/domains/cache unchanged\n";
}
