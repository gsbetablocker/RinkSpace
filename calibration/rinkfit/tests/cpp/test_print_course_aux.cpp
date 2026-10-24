#include "rinkfit/joint_print_course_aux_v1.hpp"
#include "rinkfit/augmented_lagrangian.hpp"
#include <fstream>
#include <iostream>
using namespace rinkfit;
using J=eventio::Json;
static void check(bool x,const char*why){if(!x)throw std::runtime_error(why);}
static void check_close(double x,double y,double e=1e-10){if(std::abs(x-y)>e)throw std::runtime_error("analytic result differs");}
class RawLinear final:public ceres::SizedCostFunction<1,2>{
 double a_,b_,c_;
 public:RawLinear(double a,double b,double c):a_(a),b_(b),c_(c){}
 bool Evaluate(double const*const*p,double*r,double**j)const override{r[0]=a_*p[0][0]+b_*p[0][1]+c_;if(j&&j[0]){j[0][0]=a_;j[0][1]=b_;}return true;}
};
static JointPrintCourseProblemV1 example(){JointPrintCourseProblemV1 p;p.rho=1;p.start={{0,2}};p.members={{0,0,0},{2,0,0}};p.spans={{1,1,0}};return p;}
static void tiny_checks(){
 auto p=example();auto s=solve_joint_print_course_aux_v1(p);check_close(s.values[0],1./3);check_close(s.values[1],5./3);check_close(s.after.half_cost,1./6);check(s.certified&&s.values!=p.start,"span conflict must not use minmax");
 p=example();p.members[1].u=1;p.spans[0].perimeter=2;p.start={{0,1}};s=solve_joint_print_course_aux_v1(p);check(s.certified&&s.unchanged&&s.values==p.start,"zero-cost state moved");
 p=example();p.rho=2;p.members[0].lambda_lower=2;p.members[1].lambda_upper=2;p.spans[0].lambda=2;s=solve_joint_print_course_aux_v1(p);check_close(s.values[0],1./3);check_close(s.values[1],5./3);check_close(s.after.half_cost,16./3);
 p=example();p.start[0]=.5;p.lower[0]=.5;s=solve_joint_print_course_aux_v1(p);check_close(s.values[0],.5);check_close(s.values[1],1.75);check_close(s.after.half_cost,.1875);
 // Exercise the actual PHR factory and the commit/rollback boundary.
 JointEventModel m;m.scene_block=0;m.blocks.resize(5);m.blocks[0].values={.5,.1,.1,1,.02};m.blocks[1].values={0,0};m.blocks[2].values={2,0};m.blocks[3].values={0,1,0,.02};m.blocks[4].values={0,2};
 EventPath path;path.kind=6;path.vertex_blocks={1,2};path.course_envelope_block=4;path.advertisement_block=3;m.materials.resize(1);m.materials[0].paths.push_back(path);
 const double zero[2]={0,0};const double per=-joint_normalized_print_envelope_span_v1(m.blocks[0].values.data(),zero,m.blocks[3].values.data(),0);m.blocks[3].values[1]=per;
 std::vector<std::string>ids={"normalized_print/v1/e4/u0","normalized_print/v1/e4/u1","normalized_print/v2/e4/u0","normalized_print/v2/e4/u1","normalized_print/e4/a3/o0/span"};
 J state={{"schema","rinkfit.engineering_global_phr_state.v1"},{"rho",3.},{"inequality_ids",ids},{"lambda",std::vector<double>(5,0.)},{"unchanged_sentinel",17}};const std::string state_before=state.dump();const auto blocks_before=m.blocks;
 ceres::Problem live;double*env=m.blocks[4].values.data();
 auto add=[&](double a,double b,double c){live.AddResidualBlock(make_phr_cost(std::make_unique<RawLinear>(a,b,c),0,3,false,true).release(),nullptr,env);};
 add(1,0,0);add(0,-1,0);add(1,0,-2);add(0,-1,2);add(-per,per,-per);
 auto plans=plan_joint_print_course_aux_v1(m,state,&live);check(plans.size()==1,"one envelope plan");apply_joint_print_course_aux_v1(m,state,live,plans);
 check(state.dump()==state_before,"PHR state changed");for(int k=0;k<4;++k)check(m.blocks[k].values==blocks_before[k].values,"physical input changed");check_close(m.blocks[4].values[0],plans[0].solution.values[0]);check_close(m.blocks[4].values[1],plans[0].solution.values[1]);
 // Tampering with the original multiplier must reject before any commit.
 auto second=plan_joint_print_course_aux_v1(m,state,&live);state["lambda"][0]=1;bool rejected=false;try{apply_joint_print_course_aux_v1(m,state,live,second);}catch(const std::exception&){rejected=true;}check(rejected,"changed dual accepted");
 std::cout<<"tiny_checks passed\n";
}
int main(int argc,char**argv){try{
 tiny_checks();if(argc<3)return 0;
 std::ifstream input(argv[1]);J request;input>>request;J report={{"schema","rinkfit.print_course_aux_offline_result.v1"},{"source_state_token",request.at("source_state_token")},{"production_state_modified",false},{"raw_constraints_removed",0},{"duals_modified",false},{"physical_coordinates_modified",false},{"formal_quality_accepted",false},{"groups",J::array()}};
 const auto scene=request.at("scene").get<std::vector<double>>();const double zero[2]={0,0};
 for(const auto&group:request.at("groups")){
  JointPrintCourseProblemV1 p;p.rho=request.at("rho");p.start=group.at("start").get<std::array<double,2>>();p.lower=group.at("lower").get<std::array<double,2>>();p.upper=group.at("upper").get<std::array<double,2>>();
  for(const auto&x:group.at("members"))p.members.push_back({x.at("u"),x.at("lambda_lower"),x.at("lambda_upper")});
  J height_g=J::array();for(const auto&x:group.at("spans")){const auto ad=x.at("ad_values").get<std::vector<double>>();const int off=x.at("offset");const double per=-joint_normalized_print_envelope_span_v1(scene.data(),zero,ad.data(),off);p.spans.push_back({x.at("width"),per,x.at("lambda")});double height_max=0;for(const auto&v:group.at("members")){const double height=ad.at(off+2)+(ad.at(off+3)-ad.at(off+2))*v.at("values").at(1).get<double>();height_max=std::max({height_max,-height,height-scene.at(4)});}height_g.push_back({{"ad",x.at("advertisement_block")},{"offset",off},{"unchanged_height_g_max_positive",height_max}});}
  const auto s=solve_joint_print_course_aux_v1(p);
  // Same existing PHR residual factory, evaluated locally without Solve.
  ceres::Problem original;double q[2]={p.start[0],p.start[1]};
  auto add=[&](double a,double b,double c,double lambda){original.AddResidualBlock(make_phr_cost(std::make_unique<RawLinear>(a,b,c),lambda,p.rho,false,true).release(),nullptr,q);};
  for(const auto&x:p.members){add(1,0,-x.u,x.lambda_lower);add(0,-1,x.u,x.lambda_upper);}
  for(const auto&x:p.spans)add(-x.width,x.width,-x.perimeter,x.lambda);
  double before=0,after=0;check(original.Evaluate(ceres::Problem::EvaluateOptions{},&before,nullptr,nullptr,nullptr),"original PHR before failed");q[0]=s.values[0];q[1]=s.values[1];check(original.Evaluate(ceres::Problem::EvaluateOptions{},&after,nullptr,nullptr,nullptr),"original PHR after failed");check_close(before,s.before.half_cost,std::max(1e-9,s.cost_tolerance));check_close(after,s.after.half_cost,std::max(1e-9,s.cost_tolerance));
  J raw_after=J::array();for(const auto&x:p.spans)raw_after.push_back(x.width*(s.values[1]-s.values[0])-x.perimeter);
  report["groups"].push_back({{"target_vertex",group.at("target_vertex")},{"envelope_block",group.at("envelope_block")},{"members",p.members.size()},{"span_terms",p.spans.size()},{"before",p.start},{"proposal",s.values},{"original_PHR_half_cost_before",before},{"original_PHR_half_cost_after",after},{"original_raw_g_max_before",s.before.raw_g_max_positive},{"original_raw_g_max_after",s.after.raw_g_max_positive},{"original_span_g_after",raw_after},{"projected_gradient_inf",s.after.projected_gradient_inf},{"gradient_tolerance",s.gradient_tolerance},{"pressure",s.pressure},{"pressure_intervals",s.pressure_intervals},{"KKT_certified",s.certified},{"unchanged_physical_height_constraints",height_g}});
 }
 std::ofstream output(argv[2]);output<<report.dump(2)<<"\n";std::cout<<report.dump(2)<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
