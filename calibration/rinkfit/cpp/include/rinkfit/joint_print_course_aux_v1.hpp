#pragma once
#include "rinkfit/joint_normalized_print_domain_v1.hpp"
#include <ceres/ceres.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace rinkfit {

// Exact active-set solution of the existing two-coordinate PHR subproblem.
// These are numerical envelope coordinates, not measured contour vertices.
struct JointPrintCourseMemberV1 {double u=0,lambda_lower=0,lambda_upper=0;};
struct JointPrintCourseSpanV1 {double width=0,perimeter=0,lambda=0;};
struct JointPrintCourseProblemV1 {
 double rho=0;std::array<double,2>start{{0,0}};
 std::array<double,2>lower{{-std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()}};
 std::array<double,2>upper{{std::numeric_limits<double>::max(),std::numeric_limits<double>::max()}};
 std::vector<JointPrintCourseMemberV1>members;std::vector<JointPrintCourseSpanV1>spans;
};
struct JointPrintCourseMeasureV1 {
 double half_cost=0,projected_gradient_inf=0,gradient_scale=0,raw_g_max_positive=0;
 std::array<double,2>gradient{{0,0}};
};
struct JointPrintCourseSolutionV1 {
 std::array<double,2>values{{0,0}};JointPrintCourseMeasureV1 before,after;
 double pressure=0,gradient_tolerance=0,cost_tolerance=0;
 std::size_t pressure_intervals=0;bool unchanged=false,certified=false;
};
namespace joint_print_course_aux_detail {
inline void require(bool ok,const std::string&why){if(!ok)throw std::runtime_error("print course auxiliary: "+why);}
inline double finite(long double x,const char*why){const double d=static_cast<double>(x);require(std::isfinite(d),why);return d;}
inline double clipped(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
inline double positive(double x){return std::max(0.,x);}
inline void validate(const JointPrintCourseProblemV1&p){
 require(std::isfinite(p.rho)&&p.rho>0&&!p.members.empty()&&!p.spans.empty(),"invalid rho/empty original terms");
 for(int k=0;k<2;++k)require(std::isfinite(p.start[k])&&std::isfinite(p.lower[k])&&std::isfinite(p.upper[k])&&p.lower[k]<=p.start[k]&&p.start[k]<=p.upper[k],"invalid initial value or original bounds");
 for(const auto&x:p.members)require(std::isfinite(x.u)&&std::isfinite(x.lambda_lower)&&x.lambda_lower>=0&&std::isfinite(x.lambda_upper)&&x.lambda_upper>=0,"invalid original member/dual");
 for(const auto&x:p.spans)require(std::isfinite(x.width)&&x.width>0&&std::isfinite(x.perimeter)&&x.perimeter>0&&std::isfinite(x.lambda)&&x.lambda>=0,"invalid original span/dual");
}
inline JointPrintCourseMeasureV1 measure(const JointPrintCourseProblemV1&p,const std::array<double,2>&q){
 JointPrintCourseMeasureV1 out;long double cost=0,gl=0,gh=0,scale=0;
 auto term=[&](double g,double lambda,double dl,double dh){require(std::isfinite(g),"nonfinite original raw g");const double z=finite(static_cast<long double>(lambda)+static_cast<long double>(p.rho)*g,"PHR hinge overflow"),r=positive(z);cost+=static_cast<long double>(r)*r/(2*p.rho);gl+=static_cast<long double>(dl)*r;gh+=static_cast<long double>(dh)*r;scale+=std::abs(static_cast<long double>(dl)*r)+std::abs(static_cast<long double>(dh)*r);out.raw_g_max_positive=std::max(out.raw_g_max_positive,positive(g));};
 for(const auto&x:p.members){term(q[0]-x.u,x.lambda_lower,1,0);term(x.u-q[1],x.lambda_upper,0,-1);}
 for(const auto&x:p.spans)term(finite(static_cast<long double>(x.width)*(static_cast<long double>(q[1])-q[0])-x.perimeter,"span overflow"),x.lambda,-x.width,x.width);
 out.half_cost=finite(cost,"PHR cost overflow");out.gradient={{finite(gl,"gradient overflow"),finite(gh,"gradient overflow")}};out.gradient_scale=finite(scale,"gradient scale overflow");
 for(int k=0;k<2;++k){double v=out.gradient[k];if(p.lower[k]==p.upper[k]||(q[k]<=p.lower[k]&&v>=0)||(q[k]>=p.upper[k]&&v<=0))v=0;out.projected_gradient_inf=std::max(out.projected_gradient_inf,std::abs(v));}return out;
}
// Write a_i=u_i-lambda_lower_i/rho,
// b_i=u_i+lambda_upper_i/rho, c_j=P_j/W_j-lambda_span_j/(rho*W_j).
// Apart from the common rho/2 the original cost is
// sum(l-a)_+^2 + sum(b-h)_+^2 + sum W^2(h-l-c)_+^2.
// At a solution the lower/upper membership pressures equal the span
// pressure t, except for the original box normal cones. Thus the two
// waterlines and t=sum W^2(h(t)-l(t)-c)_+ give the convex KKT solution.
// All waterline and span activation events are finite and scanned once;
// no iterative nonlinear solver, min/max substitution or new constraint.
// Inverse of sum_i (y-v_i)_+ = t, with the original independent box.
struct Waterline {
 std::vector<double>v,prefix,threshold;double lower,upper;
 Waterline(std::vector<double>x,double lo,double hi):v(std::move(x)),lower(lo),upper(hi){
  require(!v.empty(),"empty waterline");std::sort(v.begin(),v.end());prefix.push_back(0);long double sum=0;
  for(std::size_t k=0;k<v.size();++k){const long double t=static_cast<long double>(k)*v[k]-sum;threshold.push_back(t>std::numeric_limits<double>::max()?INFINITY:finite(std::max(0.L,t),"waterline threshold overflow"));sum+=v[k];prefix.push_back(finite(sum,"waterline prefix overflow"));}
 }
 std::size_t count(double t)const{const auto k=static_cast<std::size_t>(std::upper_bound(threshold.begin(),threshold.end(),t)-threshold.begin());return std::max<std::size_t>(1,k);}
 std::pair<double,double>value_slope(double t)const{const auto k=count(t);const double y=finite((static_cast<long double>(t)+prefix[k])/k,"waterline value overflow");return {clipped(y,lower,upper),(y<lower||y>=upper)?0.:1./k};}
 void events(std::vector<double>&out)const{
  for(double t:threshold)if(std::isfinite(t)&&t>0)out.push_back(t);
  for(double y:{lower,upper}){long double t=0;for(double x:v)if(y>x)t+=static_cast<long double>(y)-x;if(t>0&&t<std::numeric_limits<double>::max())out.push_back(static_cast<double>(t));}
 }
};
inline double gradient_tolerance(const JointPrintCourseProblemV1&p,const JointPrintCourseMeasureV1&m){return 256*std::numeric_limits<double>::epsilon()*(2*p.members.size()+p.spans.size()+1)*std::max(1.,m.gradient_scale);}
}
inline JointPrintCourseSolutionV1 solve_joint_print_course_aux_v1(const JointPrintCourseProblemV1&p){
 using namespace joint_print_course_aux_detail;validate(p);JointPrintCourseSolutionV1 out;out.values=p.start;out.before=measure(p,p.start);out.gradient_tolerance=gradient_tolerance(p,out.before);
 if(out.before.projected_gradient_inf<=out.gradient_tolerance){out.after=out.before;out.unchanged=true;out.certified=true;return out;}
 std::vector<double>a,b,c;for(const auto&x:p.members){a.push_back(finite(static_cast<long double>(x.u)-x.lambda_lower/p.rho,"lower dual shift overflow"));b.push_back(finite(-static_cast<long double>(x.u)-x.lambda_upper/p.rho,"upper dual shift overflow"));}
 for(const auto&x:p.spans)c.push_back(finite(static_cast<long double>(x.perimeter)/x.width-static_cast<long double>(x.lambda)/(p.rho*x.width),"span dual shift overflow"));
 const Waterline lo(std::move(a),p.lower[0],p.upper[0]),minus_hi(std::move(b),-p.upper[1],-p.lower[1]);
 std::vector<double>base{0};lo.events(base);minus_hi.events(base);std::sort(base.begin(),base.end());base.erase(std::unique(base.begin(),base.end()),base.end());
 std::vector<double>events=base;
 for(std::size_t k=0;k<base.size();++k){const double t=base[k],end=k+1<base.size()?base[k+1]:INFINITY;const auto l=lo.value_slope(t),y=minus_hi.value_slope(t);const double d=-y.first-l.first,s=l.second+y.second;if(s<=0)continue;for(double cutoff:c){const long double cross=static_cast<long double>(t)+(static_cast<long double>(d)-cutoff)/s;if(cross>t&&cross<end&&cross<std::numeric_limits<double>::max())events.push_back(static_cast<double>(cross));}}
 std::sort(events.begin(),events.end());events.erase(std::unique(events.begin(),events.end()),events.end());bool found=false;
 for(std::size_t k=0;k<events.size();++k){
  ++out.pressure_intervals;const double t=events[k],end=k+1<events.size()?events[k+1]:INFINITY;const auto l=lo.value_slope(t),y=minus_hi.value_slope(t);const long double d=-static_cast<long double>(y.first)-l.first,ds=l.second+y.second;long double pressure=0,slope=1;
  for(std::size_t j=0;j<p.spans.size();++j){const long double w=p.spans[j].width,excess=d-c[j];if(excess>0){pressure+=w*w*excess;slope+=w*w*ds;}}
  const long double f=static_cast<long double>(t)-pressure,proposal=static_cast<long double>(t)-f/slope;
  if(f>=0||proposal<=end){out.pressure=finite(std::max(static_cast<long double>(t),proposal),"pressure solution overflow");found=true;break;}
 }
 require(found,"finite piecewise-linear pressure root missing");out.values={{lo.value_slope(out.pressure).first,-minus_hi.value_slope(out.pressure).first}};out.after=measure(p,out.values);
 out.gradient_tolerance=gradient_tolerance(p,out.after);out.cost_tolerance=256*std::numeric_limits<double>::epsilon()*(2*p.members.size()+p.spans.size()+1)*std::max({1.,out.before.half_cost,out.after.half_cost});
 out.certified=out.after.projected_gradient_inf<=out.gradient_tolerance&&out.after.half_cost<=out.before.half_cost+out.cost_tolerance;
 require(out.certified,"original PHR objective/KKT certificate failed");out.unchanged=out.values==p.start;return out;
}

struct JointPrintCourseAuxPlanV1 {
 int envelope_block=-1;JointPrintCourseProblemV1 problem;JointPrintCourseSolutionV1 solution;
 std::vector<std::pair<std::size_t,std::string>>dual_ids;std::vector<double>duals;
 std::map<int,std::vector<double>>fixed_inputs;
};
inline std::vector<JointPrintCourseAuxPlanV1>plan_joint_print_course_aux_v1(
 const JointEventModel&m,const eventio::Json&state,const ceres::Problem*live=nullptr,
 const std::set<int>&selected={}){
 using namespace joint_print_course_aux_detail;
 require(state.at("schema")=="rinkfit.engineering_global_phr_state.v1","wrong original PHR state");
 const double rho=state.at("rho");require(std::isfinite(rho)&&rho>0,"invalid original rho");
 std::map<int,std::set<int>>vertices;std::map<int,std::set<std::pair<int,int>>>spans;
 for(const auto&mat:m.materials)for(const auto&p:mat.paths)if(p.kind==6&&p.advertisement_block>=0&&!p.vertex_blocks.empty()&&(selected.empty()||selected.count(p.course_envelope_block))){vertices[p.course_envelope_block].insert(p.vertex_blocks.begin(),p.vertex_blocks.end());spans[p.course_envelope_block].insert({p.advertisement_block,p.advertisement_offset});}
 if(!selected.empty())for(int id:selected)require(vertices.count(id),"selected envelope has no original split-print terms");
 std::map<std::string,std::size_t>needed;
 for(const auto&entry:vertices){for(int v:entry.second)for(int k:{0,1})needed.emplace("normalized_print/v"+std::to_string(v)+"/e"+std::to_string(entry.first)+"/u"+std::to_string(k),std::numeric_limits<std::size_t>::max());for(const auto&s:spans.at(entry.first))needed.emplace("normalized_print/e"+std::to_string(entry.first)+"/a"+std::to_string(s.first)+"/o"+std::to_string(s.second)+"/span",std::numeric_limits<std::size_t>::max());}
 const auto&ids=state.at("inequality_ids");const auto&lambda=state.at("lambda");require(ids.size()==lambda.size(),"original dual shape");
 for(std::size_t k=0;k<ids.size();++k){const auto it=needed.find(ids[k].get_ref<const std::string&>());if(it!=needed.end()){require(it->second==std::numeric_limits<std::size_t>::max(),"duplicate original ID");it->second=k;}}
 for(const auto&x:needed)require(x.second!=std::numeric_limits<std::size_t>::max(),"missing original constraint ID: "+x.first);
 std::vector<JointPrintCourseAuxPlanV1>out;
 for(const auto&entry:vertices){
  JointPrintCourseAuxPlanV1 plan;plan.envelope_block=entry.first;plan.problem.rho=rho;const auto&block=m.blocks.at(entry.first);require(block.values.size()==2,"envelope shape");if(block.constant)continue;
  plan.problem.start={{block.values[0],block.values[1]}};
  for(int k=0;k<2;++k){plan.problem.lower[k]=block.lower.empty()?-std::numeric_limits<double>::max():block.lower.at(k);plan.problem.upper[k]=block.upper.empty()?std::numeric_limits<double>::max():block.upper.at(k);if(std::find(block.fixed_indices.begin(),block.fixed_indices.end(),k)!=block.fixed_indices.end())plan.problem.lower[k]=plan.problem.upper[k]=block.values[k];}
  if(live){const double*p=block.values.data();require(live->HasParameterBlock(p)&&live->ParameterBlockSize(p)==2,"envelope absent from live graph");if(live->IsParameterBlockConstant(p))continue;require(live->GetManifold(p)==nullptr,"non-Euclidean envelope manifold");for(int k=0;k<2;++k){plan.problem.lower[k]=live->GetParameterLowerBound(p,k);plan.problem.upper[k]=live->GetParameterUpperBound(p,k);}}
  auto dual=[&](const std::string&id){const auto index=needed.at(id);const double value=lambda.at(index);require(std::isfinite(value)&&value>=0,"invalid original lambda");plan.dual_ids.push_back({index,id});plan.duals.push_back(value);return value;};
  plan.fixed_inputs.emplace(m.scene_block,m.blocks.at(m.scene_block).values);plan.fixed_inputs.emplace(entry.first,block.values);
  for(int v:entry.second){const auto&q=m.blocks.at(v).values;require(q.size()==2,"vertex shape");plan.fixed_inputs.emplace(v,q);const std::string prefix="normalized_print/v"+std::to_string(v)+"/e"+std::to_string(entry.first)+"/u";plan.problem.members.push_back({q[0],dual(prefix+"0"),dual(prefix+"1")});}
  for(const auto&s:spans.at(entry.first)){const auto&a=m.blocks.at(s.first).values;require(s.second>=0&&s.second+3<int(a.size()),"advertisement slice");plan.fixed_inputs.emplace(s.first,a);const double zero[2]={0,0};const double per=-joint_normalized_print_envelope_span_v1(m.blocks.at(m.scene_block).values.data(),zero,a.data(),s.second);plan.problem.spans.push_back({a.at(s.second+1),per,dual("normalized_print/e"+std::to_string(entry.first)+"/a"+std::to_string(s.first)+"/o"+std::to_string(s.second)+"/span")});}
  plan.solution=solve_joint_print_course_aux_v1(plan.problem);out.push_back(std::move(plan));
 }return out;
}

// Apply only the two numerical envelope coordinates. The actual incident
// residuals are checked locally; no pixel graph rebuild or full-raw scan.
inline void apply_joint_print_course_aux_v1(JointEventModel&m,const eventio::Json&state,
 ceres::Problem&live,const std::vector<JointPrintCourseAuxPlanV1>&plans){
 using namespace joint_print_course_aux_detail;
 std::set<int>seen;std::vector<std::vector<ceres::ResidualBlockId>>incident;
 auto local_cost=[&](const std::vector<ceres::ResidualBlockId>&ids){long double total=0;for(auto id:ids){require(live.GetCostFunctionForResidualBlock(id)->num_residuals()==1&&live.GetLossFunctionForResidualBlock(id)==nullptr,"unexpected envelope factor/loss");double cost=0;require(live.EvaluateResidualBlock(id,false,&cost,nullptr,nullptr)&&std::isfinite(cost),"original incident factor evaluation failed");total+=cost;}return finite(total,"incident cost overflow");};
 for(const auto&plan:plans){
  require(seen.insert(plan.envelope_block).second&&plan.solution.certified,"duplicate/uncertified plan");require(state.at("rho").get<double>()==plan.problem.rho,"rho changed since proposal");
  for(std::size_t k=0;k<plan.dual_ids.size();++k){const auto&id=plan.dual_ids[k];require(state.at("inequality_ids").at(id.first)==id.second&&state.at("lambda").at(id.first).get<double>()==plan.duals[k],"original ID/dual changed since proposal");}
  for(const auto&input:plan.fixed_inputs)require(m.blocks.at(input.first).values==input.second,"local model changed since proposal");
  auto&block=m.blocks.at(plan.envelope_block);double*p=block.values.data();require(live.HasParameterBlock(p)&&live.ParameterBlockSize(p)==2&&!live.IsParameterBlockConstant(p)&&live.GetManifold(p)==nullptr,"live envelope freedom changed");
  for(int k=0;k<2;++k)require(live.GetParameterLowerBound(p,k)==plan.problem.lower[k]&&live.GetParameterUpperBound(p,k)==plan.problem.upper[k],"live envelope bounds changed");
  std::vector<ceres::ResidualBlockId>ids;live.GetResidualBlocksForParameterBlock(p,&ids);require(ids.size()==2*plan.problem.members.size()+plan.problem.spans.size(),"unexpected incident envelope factors");
  const double actual=local_cost(ids),tolerance=std::max(plan.solution.cost_tolerance,256*std::numeric_limits<double>::epsilon()*ids.size()*std::max(1.,actual));require(std::abs(actual-plan.solution.before.half_cost)<=tolerance,"actual incident cost differs from original PHR subproblem");incident.push_back(std::move(ids));
 }
 // All local proposals are checked before any write. Restore all envelopes if
 // a real original-factor check fails after writing a proposal.
 try{for(std::size_t k=0;k<plans.size();++k){const auto&plan=plans[k];auto&v=m.blocks.at(plan.envelope_block).values;v[0]=plan.solution.values[0];v[1]=plan.solution.values[1];const double actual=local_cost(incident[k]),tolerance=std::max(plan.solution.cost_tolerance,256*std::numeric_limits<double>::epsilon()*incident[k].size()*std::max(1.,actual));require(std::abs(actual-plan.solution.after.half_cost)<=tolerance&&actual<=plan.solution.before.half_cost+tolerance,"actual original PHR decrease differs from proposal");}}
 catch(...){for(const auto&plan:plans){auto&v=m.blocks.at(plan.envelope_block).values;v[0]=plan.problem.start[0];v[1]=plan.problem.start[1];}throw;}
}
}
