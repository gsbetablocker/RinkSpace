#pragma once
#include "rinkfit/joint_al.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace rinkfit { namespace print_gauge_v1 {
using V2=std::array<double,2>;
using V6=std::array<double,6>;
struct Halfplane { double a,b,c; };
inline bool feasible(const V2& x,const std::vector<Halfplane>& hs) {
 for(const auto& h:hs)if(h.a*x[0]+h.b*x[1]>h.c+2e-12*std::max(1.,std::abs(h.c)))return false;
 return std::isfinite(x[0])&&std::isfinite(x[1]);
}
inline std::vector<V2> vertices(const std::vector<Halfplane>& h) {
 std::vector<V2> out;
 for(std::size_t i=0;i<h.size();++i)for(std::size_t j=i+1;j<h.size();++j){
  const double d=h[i].a*h[j].b-h[j].a*h[i].b;
  if(std::abs(d)<1e-18)continue;
  V2 x={(h[i].c*h[j].b-h[j].c*h[i].b)/d,(h[i].a*h[j].c-h[j].a*h[i].c)/d};
  if(feasible(x,h))out.push_back(x);
 }
 return out;
}
inline V6 composite(const std::vector<double>& a,const std::vector<double>& q) {
 const double h=a.at(3)-a.at(2);
 return {a[1]*q[0],a[1]*q[1],a[0]+a[1]*q[2],h*q[3],h*q[4],a[2]+h*q[5]};
}
inline std::vector<double> decode(const V6& p,const std::vector<double>& a) {
 const double h=a[3]-a[2];
 return {p[0]/a[1],p[1]/a[1],(p[2]-a[0])/a[1],p[3]/h,p[4]/h,(p[5]-a[2])/h};
}
struct Witness {bool valid=false;double margin=0;std::vector<double> ad,ink;};
inline Witness witness(const EventParameterBlock& a,const EventParameterBlock& q,
                       const V6& p,double height,double perimeter,double margin) {
 std::array<std::vector<Halfplane>,2> hs;
 for(int axis=0;axis<2;++axis){
  auto& h=hs[axis];
  for(int j=0;j<3;++j){
   const int k=axis*3+j;const double range=q.upper[k]-q.lower[k];
   const double lo=q.lower[k]+margin*range,hi=q.upper[k]-margin*range;
   if(j<2){h.push_back({0,lo,p[k]});h.push_back({0,-hi,-p[k]});}
   else{h.push_back({1,lo,p[k]});h.push_back({-1,-hi,-p[k]});}
  }
  if(axis==0){
   const double lo=a.lower[0]+margin*(a.upper[0]-a.lower[0]),hi=a.upper[0]-margin*(a.upper[0]-a.lower[0]);
   const double wmax=std::min(a.upper[1],perimeter),wlo=a.lower[1]+margin*(wmax-a.lower[1]),whi=wmax-margin*(wmax-a.lower[1]);
   h.insert(h.end(),{{-1,0,-lo},{1,0,hi},{0,-1,-wlo},{0,1,whi}});
  }else{
   const double lo=std::max(0.,a.lower[2]),hi=std::min(height,a.upper[2]);
   const double tlo=std::max(0.,a.lower[3]),thi=std::min(height,a.upper[3]);
   h.insert(h.end(),{{-1,0,-(lo+margin*(hi-lo))},{1,0,hi-margin*(hi-lo)},
       {-1,-1,-(tlo+margin*(thi-tlo))},{1,1,thi-margin*(thi-tlo)},
       {0,-1,-(1e-6+margin*height)}});
  }
 }
 auto x=vertices(hs[0]),z=vertices(hs[1]);Witness out;
 if(x.empty()||z.empty())return out;
 const auto minscale=[](const V2& a,const V2& b){return a[1]<b[1];};
 const auto xx=*std::min_element(x.begin(),x.end(),minscale),zz=*std::min_element(z.begin(),z.end(),minscale);
 if(!(xx[1]>0&&zz[1]>0))return out;
 if((p[0]*p[4]-p[1]*p[3])/(xx[1]*zz[1])<1e-6*(1+margin))return out;
 out.ad={xx[0],xx[1],zz[0],zz[0]+zz[1]};out.ink=decode(p,out.ad);out.margin=margin;out.valid=true;
 for(int k=0;k<6;++k)if(out.ink[k]<q.lower[k]-2e-9||out.ink[k]>q.upper[k]+2e-9)out.valid=false;
 return out;
}
inline Witness interior_witness(const EventParameterBlock& a,const EventParameterBlock& q,
                                const V6& p,double h,double perimeter) {
 auto out=witness(a,q,p,h,perimeter,0);if(!out.valid)return out;
 double lo=0,hi=.499999;
 for(int k=0;k<36;++k){double m=(lo+hi)*.5;if(witness(a,q,p,h,perimeter,m).valid)lo=m;else hi=m;}
 if(lo>1e-10)out=witness(a,q,p,h,perimeter,lo*.5);
 return out;
}
inline double margin_at(const EventParameterBlock& a,const EventParameterBlock& q,double h,double per) {
 double value=INFINITY;
 auto bound=[&](double x,double lo,double hi){if(hi>lo)value=std::min(value,std::min(x-lo,hi-x)/(hi-lo));};
 for(int i=0;i<6;++i)bound(q.values[i],q.lower[i],q.upper[i]);
 bound(a.values[0],a.lower[0],a.upper[0]);bound(a.values[1],a.lower[1],std::min(a.upper[1],per));
 bound(a.values[2],std::max(0.,a.lower[2]),std::min(h,a.upper[2]));
 bound(a.values[3],std::max(0.,a.lower[3]),std::min(h,a.upper[3]));
 value=std::min(value,(a.values[3]-a.values[2]-1e-6)/h);
 return value;
}
inline const eventio::Json* policy(const eventio::Json& cfg) {
 if(!cfg.contains("runtime_shared_parameter_stage_v2"))return nullptr;
 const auto& s=cfg.at("runtime_shared_parameter_stage_v2");
 return s.contains("print_only_coordinate_gauge")?&s.at("print_only_coordinate_gauge"):nullptr;
}
inline double perimeter(const JointEventModel& m) {
 std::array<double,32> g{};const auto& v=m.blocks.at(m.scene_block).values;std::copy(v.begin(),v.end(),g.begin());
 return 2*(1-2*g[1]+g[0]-2*g[2])+4*EventModel::quarter_arclength(1.5707963267948966,g.data());
}
inline eventio::Json install(JointALBuilder& b,ceres::Problem& problem,const eventio::Json& cfg) {
 const auto* pp=policy(cfg);if(!pp)return {{"enabled",false}};
 if(pp->at("schema")!="rinkfit.print_only_coordinate_gauge.v1"||pp->at("physical_dimensions_measured")!=false||pp->at("adaptive_original_domain_rechart")!=true)throw std::runtime_error("invalid print-only coordinate policy");
 eventio::Json groups=eventio::Json::array();std::set<int> used;
 for(const auto& item:pp->at("groups")){
  const int ai=item.at("advertisement_block"),qi=item.at("print_block");
  auto& a=b.core.model.blocks.at(ai);auto& q=b.core.model.blocks.at(qi);
  if(a.name!=item.at("advertisement_name")||q.name!=item.at("print_name")||a.values.size()!=4||q.values.size()!=6||!used.insert(qi).second)throw std::runtime_error("print-only chart shape/identity");
  if(problem.IsParameterBlockConstant(a.values.data())||problem.IsParameterBlockConstant(q.values.data())||problem.ParameterBlockTangentSize(a.values.data())!=4||problem.ParameterBlockTangentSize(q.values.data())!=6)throw std::runtime_error("print-only gauge requires actual original 4+6 free stage");
  if(std::abs(q.values[0])<1e-10||std::abs(q.values[4])<1e-10)throw std::runtime_error("print-only diagonal gauge chart is singular");
  std::size_t rows=0;
  for(const auto& r:b.core.packet.rows)if(r.active){const auto& p=b.core.model.materials.at(r.material).paths.at(r.path);
   if(p.geometry_block==qi&&p.advertisement_block!=ai)throw std::runtime_error("print-only affine shared across installations");
   if(p.advertisement_block==ai){if(p.kind!=2||p.geometry_block!=qi||p.advertisement_offset!=0 )throw std::runtime_error("independent exterior evidence invalidates print-only coordinate gauge");++rows;}
  }
  if(!rows)throw std::runtime_error("empty print-only source support");
  problem.SetManifold(q.values.data(),new ceres::SubsetManifold(6,{0,2,4,5}));
  groups.push_back({{"advertisement_block",ai},{"print_block",qi},{"active_rows",rows},{"physical_tangent_before",10},{"physical_tangent_after",6},{"coordinate_directions_removed",4},{"actual_installation_translation_and_scales_free",true},{"shear_free_indices",{1,3}},{"historical_flags_and_bounds_changed",false}});
 }
 return {{"enabled",true},{"schema","rinkfit.actual_print_only_coordinate_gauge.v1"},{"groups",groups},{"physical_values_changed",false},{"fixed_witness_is_entire_domain",false},{"original_domain_decoder_enabled",true}};
}
inline bool needs_rechart(const JointALBuilder& b,const eventio::Json& cfg) {
 const auto* pp=policy(cfg);if(!pp)return false;
 const double h=b.core.model.blocks.at(b.core.model.scene_block).values[4],per=perimeter(b.core.model);
 for(const auto& item:pp->at("groups")){
  const auto& a=b.core.model.blocks.at(item.at("advertisement_block").get<int>());const auto& q=b.core.model.blocks.at(item.at("print_block").get<int>());
  const auto w=interior_witness(a,q,composite(a.values,q.values),h,per);
  if(w.valid&&w.margin>1e-9&&margin_at(a,q,h,per)<.1*w.margin)return true;
 }
 return false;
}
inline eventio::Json rechart(JointALBuilder& b,const eventio::Json& cfg,bool initialize) {
 const auto* pp=policy(cfg);if(!pp)return {{"enabled",false}};
 const double h=b.core.model.blocks.at(b.core.model.scene_block).values[4],per=perimeter(b.core.model);
 eventio::Json report=eventio::Json::array();
 for(const auto& item:pp->at("groups")){
  auto& a=b.core.model.blocks.at(item.at("advertisement_block").get<int>());auto& q=b.core.model.blocks.at(item.at("print_block").get<int>());
  const auto p=composite(a.values,q.values);const auto w=interior_witness(a,q,p,h,per);
  if(!w.valid)throw std::runtime_error("print-only common original-domain witness is infeasible");
  const double old_margin=margin_at(a,q,h,per);const bool change=(initialize||old_margin<.1*w.margin)&&w.margin>old_margin+1e-9;
  double maximum=0;
  if(change){const auto after=composite(w.ad,w.ink);for(int k=0;k<6;++k)maximum=std::max(maximum,std::abs(p[k]-after[k]));if(maximum>1e-12)throw std::runtime_error("print-only rechart changed physical affine");std::copy(w.ad.begin(),w.ad.end(),a.values.begin());std::copy(w.ink.begin(),w.ink.end(),q.values.begin());}
  report.push_back({{"advertisement_block",item.at("advertisement_block")},{"changed",change},{"previous_relative_margin",old_margin},{"common_witness_relative_margin",w.margin},{"max_physical_coefficient_change",maximum}});
 }
 return {{"schema","rinkfit.print_only_original_domain_rechart.v1"},{"enabled",true},{"groups",report},{"raw_UV_PTS_membership_and_physical_curves_changed",false},{"original_bounds_preserved",true}};
}
}} // rinkfit::print_gauge_v1
