#include "rinkfit/event_model.hpp"
#include "rinkfit/event_packet.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/jet.h>
#include <Eigen/LU>
#include <chrono>
#include <iostream>
using namespace rinkfit;
using Json=eventio::Json;
struct ProjectionAD{const EventModel* model;EventRowLayout layout;std::size_t physical_slots;
 template<class T>bool operator()(T const*const* p,T* y)const{auto q=model->predict_for_al(layout,p[physical_slots][0],p[physical_slots][1],p);if(!q.arithmetic_valid)return false;y[0]=q.uv[0];y[1]=q.uv[1];return finite_array(q.uv);}};
struct SceneAD{const EventModel* model;EventPhysicalConstraint spec;
 template<class T>bool operator()(T const*const* p,T* y)const{try{y[0]=model->evaluate_physical(spec,p);return finite_scalar(y[0]);}catch(...){return false;}}};
struct UVJac{Vec2<double> uv;std::vector<std::vector<double>> jac;std::vector<int> ids;std::array<double,2> nuisance;};
UVJac projection_ad(const EventModel& m,const EventRowLayout& l,double s,double v){
 UVJac out;out.ids=l.block_ids;out.nuisance={s,v};ceres::DynamicAutoDiffCostFunction<ProjectionAD,64> cost(new ProjectionAD{&m,l,l.block_ids.size()});
 std::vector<const double*> p;std::vector<double*> j;
 for(auto id:l.block_ids){auto n=m.blocks[id].values.size();cost.AddParameterBlock(int(n));p.push_back(m.blocks[id].values.data());out.jac.emplace_back(2*n);}
 cost.AddParameterBlock(2);p.push_back(out.nuisance.data());out.jac.emplace_back(4);for(auto& a:out.jac)j.push_back(a.data());cost.SetNumResiduals(2);
 if(!cost.Evaluate(p.data(),out.uv.data(),j.data()))throw std::runtime_error("projection AD arithmetic failure");return out;
}
double station_gradient(const EventModel& model,const EventRowLayout& layout,double s,double frame_time,const Vec2<double>& observed,double E){
 const auto root=model.project_rs(layout.material,layout.path,s,frame_time);if(!root.valid||!root.unique)return NAN;
 using J=ceres::Jet<double,2>;std::vector<std::vector<J>> storage;std::vector<const J*> p;
 for(auto id:layout.block_ids){storage.emplace_back();for(double x:model.blocks[id].values){J v;v.a=x;v.v.setZero();storage.back().push_back(v);}}for(auto& a:storage)p.push_back(a.data());
 J sj,vj;sj.a=s;sj.v<<1,0;vj.a=root.row;vj.v<<0,1;auto z=model.predict_for_al(layout,sj,vj,p.data());if(!z.arithmetic_valid)return NAN;
 const J eu=z.uv[0]-J(observed[0]),ev=z.uv[1]-J(observed[1]),q=(eu*eu+ev*ev)/J(E);
 return q.v[0]+q.v[1]*z.uv[1].v[0]/(1-z.uv[1].v[1]);
}
// Only refines a nuisance coordinate in its current finite smooth segment.
// The derivative is an exact Jet derivative; no finite-difference q_s, new
// observation, geometry parameter, enlarged support, or tolerance change.
double refine_station_legacy(const EventModel& model,const EventRowLayout& layout,double initial,const Vec2<double>& observed,double E,std::uint8_t* mode){
 const auto& curve=model.materials[layout.material].paths[layout.path];const bool periodic=curve.kind==1&&curve.closed;
 auto wrap=[&](double x){if(periodic){x=std::fmod(x,1.);if(x<0)x+=1.;}return x;};
 double lower=periodic?initial-.25:0.,upper=periodic?initial+.25:1.;
 if(curve.kind==2){const double segments=double(curve.points.size()-1);const auto cell=std::min(curve.points.size()-2,std::size_t(std::floor(initial*segments)));lower=cell/segments;upper=(cell+1)/segments;}
 auto gradient=[&](double x){return station_gradient(model,layout,wrap(x),layout.frame_time,observed,E);};
 auto qvalue=[&](double x){const auto root=model.project_rs(layout.material,layout.path,wrap(x),layout.frame_time);if(!root.valid||!root.unique)return std::numeric_limits<double>::infinity();const double eu=root.uv[0]-observed[0],ev=root.uv[1]-observed[1];return(eu*eu+ev*ev)/E;};
 const double original_q=qvalue(initial),tolerance=1e-5*std::max(1.,original_q);double g0=gradient(initial);if(!std::isfinite(g0)||std::abs(g0)<=tolerance)return initial;
 const double direction=g0>0?-1.:1.,limit=direction<0?lower:upper,room=std::abs(limit-initial);if(!(room>0))return initial;
 double h=std::min(1e-6/std::max(1.,curve.kind==2?double(curve.points.size()-1):1.),room);double a=initial,b=initial,ga=g0,gb=g0;
 for(int k=0;k<24;++k){b=initial+direction*std::min(h,room);gb=gradient(b);if(!std::isfinite(gb)){h*=.5;if(h<32*std::numeric_limits<double>::epsilon())return initial;continue;}if(g0*gb<=0)break;if(std::abs(b-limit)<=32*std::numeric_limits<double>::epsilon()){
    // A finite endpoint/vertex can be the minimum, but verify both adjacent
    // physical sides before declaring a vertex response stationary.
    if(curve.kind==2){const double eps=1e-7/(curve.points.size()-1);double l=b-eps,r=b+eps;if(curve.closed){if(l<0)l+=1;if(r>1)r-=1;}if(l>=0&&r<=1){const double gl=gradient(l),gr=gradient(r);if(std::isfinite(gl)&&std::isfinite(gr)&&gl<=tolerance&&gr>=-tolerance&&qvalue(b)<=original_q+1e-10*std::max(1.,original_q)){*mode=1;return wrap(b);}}}
    if(!curve.closed&&((b==0&&gb>=-tolerance)||(b==1&&gb<=tolerance))&&qvalue(b)<=original_q+1e-10*std::max(1.,original_q)){*mode=2;return b;}return initial;
   }h=std::min(2*h,room);}
 if(!std::isfinite(gb)||g0*gb>0)return initial;if(a>b){std::swap(a,b);std::swap(ga,gb);}double best=std::abs(ga)<std::abs(gb)?a:b,best_gradient=std::min(std::abs(ga),std::abs(gb));
 for(int k=0;k<24;++k){double x=(a*gb-b*ga)/(gb-ga);if(!std::isfinite(x)||x<=a||x>=b)x=(a+b)/2;const double gx=gradient(x);if(!std::isfinite(gx))return initial;if(std::abs(gx)<best_gradient){best=x;best_gradient=std::abs(gx);}if(std::abs(gx)<=tolerance)break;if(ga*gx>0){a=x;ga=gx;}else{b=x;gb=gx;}if(b-a<=16*std::numeric_limits<double>::epsilon())break;}
 return qvalue(best)<=original_q+1e-10*std::max(1.,original_q)?wrap(best):initial;
}

// Preserve an actually stationary legacy return exactly. A legacy move
// is not a stationarity certificate; test its returned foot before fallback. Keep a legal bracket when an expanding probe
// leaves the original sensor domain; never bounce between the same two probes.
double refine_station(const EventModel& model,const EventRowLayout& layout,double initial,const Vec2<double>& observed,double E,std::uint8_t* mode,bool allow_vertex_canonical=true){
 const double legacy=refine_station_legacy(model,layout,initial,observed,E,mode);if(*mode!=0)return legacy;initial=legacy;
 const auto& curve=model.materials.at(layout.material).paths.at(layout.path);const bool periodic=curve.kind==1&&curve.closed;
 auto wrap=[&](double x){if(periodic){x=std::fmod(x,1.);if(x<0)x+=1.;}return x;};
 auto root_at=[&](double x){return model.project_rs(layout.material,layout.path,wrap(x),layout.frame_time);};
 auto qvalue=[&](double x){const auto z=root_at(x);if(!z.valid||!z.unique)return std::numeric_limits<double>::infinity();const double eu=z.uv[0]-observed[0],ev=z.uv[1]-observed[1];return(eu*eu+ev*ev)/E;};
 auto gradient=[&](double x){return station_gradient(model,layout,wrap(x),layout.frame_time,observed,E);};
 const double q0=qvalue(initial),tol=1e-5*std::max(1.,q0),g0=gradient(initial);if(!std::isfinite(q0))return initial;
 double lower=periodic?initial-.25:0.,upper=periodic?initial+.25:1.;
 if(curve.kind==2){const double edges=double(curve.points.size()-1),coordinate=initial*edges;const auto cell=std::min(curve.points.size()-2,std::size_t(std::floor(coordinate)));lower=cell/edges;upper=(cell+1)/edges;
  const double knot=std::round(coordinate)/edges;if((allow_vertex_canonical||initial==knot)&&std::abs(coordinate-std::round(coordinate))<=1e-7&&(curve.closed||(knot>0&&knot<1))){if(qvalue(knot)<=q0+1e-10*std::max(1.,q0)){*mode=1;return knot;}}
 }
 if(!std::isfinite(g0))return initial;
 if(!curve.closed&&((initial<=1e-10&&g0>=-tol)||(initial>=1-1e-10&&g0<=tol))){const double endpoint=initial<=1e-10?0.:1.;if(qvalue(endpoint)<=q0+1e-10*std::max(1.,q0)){*mode=2;return endpoint;}}
 if(std::abs(g0)<=tol)return initial;
 const double direction=g0>0?-1.:1.,limit=direction<0?lower:upper,room=std::abs(limit-initial);if(!(room>0))return initial;
 double a=initial,ga=g0,b=initial,gb=g0,h=std::min(1e-6/std::max(1.,curve.kind==2?double(curve.points.size()-1):1.),room);bool bracket=false;
 for(int k=0;k<64;++k){b=initial+direction*std::min(h,room);gb=gradient(b);
  if(!std::isfinite(gb)){
   double legal=a,illegal=b;for(int j=0;j<36;++j){const double mid=(legal+illegal)/2;const auto z=root_at(mid);if(z.valid&&z.unique)legal=mid;else illegal=mid;}
   b=legal;gb=gradient(b);if(!std::isfinite(gb)||b==a)return initial;
   if(g0*gb<=0){bracket=true;break;}
   const auto z=root_at(b);const double q=qvalue(b);std::uint8_t boundary=255;if(std::abs(z.row-1079.)<=1e-5)boundary=3;else if(std::abs(z.row)<=1e-5)boundary=4;
   if(boundary!=255&&q<q0){const auto jac=projection_ad(model,layout,wrap(b),z.row);const auto& jsv=jac.jac.back();const double dv=jsv[2]/(1-jsv[3]);if(std::abs(dv)>1e-12){const double mu=(boundary==3?-gb:gb)/dv;if(std::isfinite(mu)&&mu>=-1e-5*std::max(1.,q)){*mode=boundary;return wrap(b);}}}
   return initial;
  }
  // A derivative sign jump at the exact finite-cell endpoint is a
  // material vertex, not a smooth zero. Keep its original station exactly;
  // the existing adjacent-vertex refiner and downstream KKT decide validity.
  if(curve.kind==2&&std::abs(b-limit)<=32*std::numeric_limits<double>::epsilon()){
   const double q=qvalue(limit);if(std::isfinite(q)&&q<=q0+1e-10*std::max(1.,q0)){
    if(curve.closed||(limit>0&&limit<1)){*mode=1;return limit;}
    if((limit==0&&gb>=-1e-5*std::max(1.,q))||(limit==1&&gb<=1e-5*std::max(1.,q))){*mode=2;return limit;}
   }
  }
  if(g0*gb<=0){bracket=true;break;}
  if(std::abs(b-limit)<=32*std::numeric_limits<double>::epsilon())return initial;
  a=b;ga=gb;h=std::min(2*h,room);
 }
 if(!bracket)return initial;if(a>b){std::swap(a,b);std::swap(ga,gb);}double best=std::abs(ga)<std::abs(gb)?a:b,best_gradient=std::min(std::abs(ga),std::abs(gb));
 for(int k=0;k<48;++k){double x=(a*gb-b*ga)/(gb-ga);if(!std::isfinite(x)||x<=a||x>=b)x=(a+b)/2;const double gx=gradient(x);if(!std::isfinite(gx))return initial;if(std::abs(gx)<best_gradient){best=x;best_gradient=std::abs(gx);}if(std::abs(gx)<=1e-5*std::max(1.,qvalue(x)))break;if(ga*gx>0){a=x;ga=gx;}else{b=x;gb=gx;}if(b-a<=16*std::numeric_limits<double>::epsilon())break;}
 return qvalue(best)<=q0+1e-10*std::max(1.,q0)?wrap(best):initial;
}

// Nonstationary vertex: enter only a true descent side on the same finite path.
double refine_vertex_station(const EventModel& model,const EventRowLayout& layout,double initial,const Vec2<double>& observed,double E,std::uint8_t* mode,Json* evidence){
 const auto& curve=model.materials.at(layout.material).paths.at(layout.path);if(curve.kind!=2||curve.points.size()<2)return initial;
 const double edges=double(curve.points.size()-1),eps=1e-7/edges;
 auto wrap=[&](double x){if(curve.closed){x=std::fmod(x,1.);if(x<0)x+=1.;}return x;};
 auto qvalue=[&](double x){const auto root=model.project_rs(layout.material,layout.path,x,layout.frame_time);if(!root.valid||!root.unique)return std::numeric_limits<double>::infinity();const double eu=root.uv[0]-observed[0],ev=root.uv[1]-observed[1];return(eu*eu+ev*ev)/E;};
 auto gradient=[&](double x){return station_gradient(model,layout,x,layout.frame_time,observed,E);};
 auto verified=[&](double x,std::uint8_t m,double q){
  if(!std::isfinite(q)||x<0||x>1)return false;const auto root=model.project_rs(layout.material,layout.path,x,layout.frame_time);if(!root.valid||!root.unique)return false;const double tol=1e-5*std::max(1.,q),qs=gradient(x);
  if(m==0)return std::isfinite(qs)&&std::abs(qs)<=tol;
  if(m==1){if(std::abs(x*edges-std::round(x*edges))>1e-7)return false;double l=x-eps,r=x+eps;if(curve.closed){l=wrap(l);r=wrap(r);}if(l<0||r>1)return false;const double dl=gradient(l),dr=gradient(r);return std::isfinite(dl)&&std::isfinite(dr)&&dl<=tol&&dr>=-tol;}
  if(m==2)return !curve.closed&&((x<=1e-10&&qs>=-tol)||(x>=1-1e-10&&qs<=tol));
  if(m==3||m==4){if(std::abs(root.row-(m==3?1079.:0.))>1e-5)return false;const auto z=projection_ad(model,layout,x,root.row);const auto& j=z.jac.back();if(!(1-j[3]>0)||std::abs(j[2])<1e-12)return false;const double mu=(m==3?-qs:qs)/(j[2]/(1-j[3]));return std::isfinite(mu)&&mu>=-tol;}
  return false;
 };
 const double q0=qvalue(initial);if(!std::isfinite(q0))return initial;
 double best=initial,bestq=q0,lowest=initial,lowestq=q0;std::uint8_t bestmode=*mode,lowestmode=*mode;bool have_verified=verified(initial,*mode,q0);
 std::vector<double> pending{initial};std::set<long long> visited;Json steps=Json::array();
 // Every visited item is an actual vertex of this finite material. A lower
 // unverified vertex never discards an already verified smooth/endpoint foot.
 while(!pending.empty()){
  const double x=pending.back();pending.pop_back();long long vertex=std::llround(x*edges);if(curve.closed)vertex%=static_cast<long long>(curve.points.size()-1);if(!visited.insert(vertex).second)continue;
  const double qx=qvalue(x);if(!std::isfinite(qx))continue;const double tol=1e-5*std::max(1.,qx);double dl=NAN,dr=NAN;if(curve.closed||x>0)dl=gradient(wrap(x-eps));if(curve.closed||x<1)dr=gradient(wrap(x+eps));
  if(verified(x,1,qx)&&(!have_verified||qx<bestq)){best=x;bestq=qx;bestmode=1;have_verified=true;}
  for(int direction:{-1,1}){
   const bool descent=direction<0?std::isfinite(dl)&&dl>tol:std::isfinite(dr)&&dr<-tol;if(!descent)continue;const double seed=wrap(x+direction*eps);if(seed<0||seed>1)continue;
   std::uint8_t candidate_mode=0;const double candidate=refine_station(model,layout,seed,observed,E,&candidate_mode,false),q=qvalue(candidate);if(!std::isfinite(q)||!(q<qx))continue;
   if(q<lowestq){lowest=candidate;lowestq=q;lowestmode=candidate_mode;}
   const bool valid=verified(candidate,candidate_mode,q);steps.push_back({{"from_s",x},{"from_q",qx},{"direction",direction},{"candidate_s",candidate},{"candidate_q",q},{"candidate_mode",candidate_mode},{"original_KKT_verified",valid}});
   if(valid&&(!have_verified||q<bestq)){best=candidate;bestq=q;bestmode=candidate_mode;have_verified=true;}
   if(candidate_mode==1)pending.push_back(candidate);
  }
 }
 if(!have_verified){best=lowest;bestq=lowestq;bestmode=lowestmode;}*mode=bestmode;
 *evidence={{"initial_s",initial},{"initial_q",q0},{"refined_s",best},{"refined_q",bestq},{"refined_mode",bestmode},{"refined_q_s",gradient(best)},{"same_finite_path",true},{"physical_theta_changed",false},{"original_KKT_verified",have_verified},{"visited_finite_vertices",visited.size()},{"vertex_descent_steps",steps}};return best;
}

// Release a stale readout-boundary active set only when its original
// inequality multiplier has the wrong sign. The original exact-Jet smooth
// refiner and all downstream KKT/RS tolerances are unchanged.
double release_visibility_station(const EventModel& model,const EventRowLayout& layout,double initial,const Vec2<double>& observed,double E,std::uint8_t* mode,Json* evidence){
 const auto old_mode=*mode;if(old_mode!=3&&old_mode!=4)return initial;
 const auto root=model.project_rs(layout.material,layout.path,initial,layout.frame_time);if(!root.valid||!root.unique)return initial;
 const auto z=projection_ad(model,layout,initial,root.row);const auto& j=z.jac.back();const double den=1-j[3],vs=j[2];
 if(!(den>0)||std::abs(vs)<1e-12||std::abs(root.row-(old_mode==3?1079.:0.))>1e-5)return initial;
 const double eu=z.uv[0]-observed[0],ev=z.uv[1]-observed[1],q0=(eu*eu+ev*ev)/E,qs=(2*eu/E)*(j[0]+j[1]*vs/den)+(2*ev/E)*vs/den,tol=1e-5*std::max(1.,q0),mu=(old_mode==3?-qs:qs)/(vs/den);
 if(!std::isfinite(mu)||mu>=-tol)return initial;
 *evidence={{"initial_s",initial},{"initial_mode",old_mode},{"initial_q",q0},{"initial_row",root.row},{"initial_q_s",qs},{"initial_visibility_multiplier",mu},{"original_stationarity_tolerance",tol},{"same_finite_path",true},{"physical_theta_changed",false},{"released",false}};
 std::uint8_t candidate_mode=0;const double candidate=refine_station(model,layout,initial,observed,E,&candidate_mode);if(candidate==initial)return initial;
 const auto next=model.project_rs(layout.material,layout.path,candidate,layout.frame_time);if(!next.valid||!next.unique)return initial;
 const double q=((next.uv[0]-observed[0])*(next.uv[0]-observed[0])+(next.uv[1]-observed[1])*(next.uv[1]-observed[1]))/E;
 if(!std::isfinite(q)||!(q<q0))return initial;
 *mode=candidate_mode;(*evidence)["released"]=true;(*evidence)["refined_s"]=candidate;(*evidence)["refined_mode"]=candidate_mode;(*evidence)["refined_q"]=q;(*evidence)["refined_row"]=next.row;(*evidence)["refined_q_s"]=station_gradient(model,layout,candidate,layout.frame_time,observed,E);return candidate;
}
std::array<double,2> rational(const std::vector<double>& a,double x){double y=0,d=0;for(auto i=a.rbegin();i!=a.rend();++i){d=d*x+y;y=y*x+*i;}return {y,d};}
struct CSR{std::vector<std::uint64_t> offsets{0};std::vector<std::uint32_t> blocks,components;std::vector<double> values;void end(){offsets.push_back(values.size());}void add(int b,int c,double value){if(value!=0){blocks.push_back(std::uint32_t(b));components.push_back(std::uint32_t(c));values.push_back(value);}}};
int main(int argc,char**argv){try{
 const auto started=std::chrono::steady_clock::now();const auto input_path=std::filesystem::u8path(argv[1]),out=std::filesystem::u8path(argv[2]);std::filesystem::create_directories(out);if(std::filesystem::exists(out/"manifest.json"))throw std::runtime_error("refuse to overwrite a frozen qbar result");
 const auto input_hash=eventio::file_sha256(input_path);const auto input=eventio::read_json(input_path);const auto dir=input_path.parent_path();
 if(input.at("schema")!="rinkfit.qbar_sparse_linearization_input.v1")throw std::runtime_error("qbar input schema");
 auto ref=[&](const Json& r){const auto p=dir/std::filesystem::u8path(r.at(r.contains("path")?"path":"file").get<std::string>());eventio::verify(p,r.at("sha256"));return p;};
 const auto packet_path=ref(input.at("packet")),model_path=ref(input.at("model")),sem_path=ref(input.at("semantic"));EventPacket packet(packet_path);packet.load_semantics(sem_path);packet.load_query(std::filesystem::u8path(packet.manifest.at("query_packet").at("manifest").get<std::string>()),packet.root.parent_path()/"query_initialization.json");const auto model=EventModel::load(model_path.u8string());
 const auto& a=input.at("arrays");auto s=eventio::read_array<double>(dir,a.at("station"),"<f8");const auto v=eventio::read_array<double>(dir,a.at("row"),"<f8"),alpha=eventio::read_array<double>(dir,a.at("alpha"),"<f8");const auto path=eventio::read_array<std::uint32_t>(dir,a.at("path"),"<u4");auto mode=eventio::read_array<std::uint8_t>(dir,a.at("mode"),"|u1");const auto n=packet.rows.size();if(s.size()!=n||v.size()!=n||alpha.size()!=n||path.size()!=n||mode.size()!=n)throw std::runtime_error("qbar full source arrays required");
 std::vector<std::uint64_t> selected=packet.active_rows;if(input.contains("source_rows")){if(input.at("source_rows").is_array())selected=input.at("source_rows").get<std::vector<std::uint64_t>>();else selected=eventio::read_array<std::uint64_t>(dir,input.at("source_rows"),"<u8");}if(std::set<std::uint64_t>(selected.begin(),selected.end()).size()!=selected.size())throw std::runtime_error("duplicate selected source row");
 CSR qcsr;std::vector<double> qvalues,stationarity,maxF,implicit_rcond,linearization_s,linearization_v;std::vector<std::uint8_t> status,modes;Json diagnoses=Json::array(),vertex_refinements=Json::array(),visibility_releases=Json::array();std::map<int,std::size_t> status_counts;std::size_t refined_stations=0;const auto projection_started=std::chrono::steady_clock::now();
 for(std::size_t ordinal=0;ordinal<selected.size();++ordinal){const auto row=selected[ordinal];if(row>=n)throw std::runtime_error("qbar source row range");std::uint8_t st=0;double q=NAN,qs=NAN,err=INFINITY,rcond=NAN;std::vector<std::tuple<int,int,double>> derivative;
  try{
   if(!packet.is_active(row)){st=8;throw std::runtime_error("excluded ledger entry");}
   const int mi=int(packet.model_material(row)),pi=int(path[row]);const auto& material=model.materials.at(mi);const auto& curve=material.paths.at(pi);if(curve.variant_id!=packet.model_variant_id(row))throw std::runtime_error("resolved finite variant mismatch");const auto layout=model.layout(mi,pi,(packet.pts[row]-256256)/12800.);
   if(packet.kind[row]==0&&(mode[row]==3||mode[row]==4)){Json evidence;const double old=s[row];s[row]=release_visibility_station(model,layout,old,{packet.uv[2*row],packet.uv[2*row+1]},packet.budget[row]*packet.budget[row],&mode[row],&evidence);if(s[row]!=old)++refined_stations;if(!evidence.is_null()){evidence["source_row"]=row;visibility_releases.push_back(evidence);}}
   if(packet.kind[row]!=2&&mode[row]==0){const Vec2<double> observed{packet.uv[2*row],packet.uv[2*row+1]};const double refined=refine_station(model,layout,s[row],observed,packet.budget[row]*packet.budget[row],&mode[row]);if(refined!=s[row]){s[row]=refined;++refined_stations;}}
   if(packet.kind[row]==0&&mode[row]==1){Json evidence;const double old=s[row];s[row]=refine_vertex_station(model,layout,old,{packet.uv[2*row],packet.uv[2*row+1]},packet.budget[row]*packet.budget[row],&mode[row],&evidence);if(s[row]!=old)++refined_stations;if(vertex_refinements.size()<100){evidence["source_row"]=row;vertex_refinements.push_back(evidence);}}
   const auto root=model.project_rs(mi,pi,s[row],layout.frame_time,v[row]);
   if(!root.valid||!root.unique){st=2;throw std::runtime_error("no strict unique RS root");}const auto z=projection_ad(model,layout,s[row],root.row);const auto& jsv=z.jac.back();const double us=jsv[0],ur=jsv[1],vs=jsv[2],vr=jsv[3],den=1-vr;if(!(den>0)||!std::isfinite(den)){st=3;throw std::runtime_error("singular RS implicit denominator");}
   const double eu=z.uv[0]-packet.uv[2*row],ev=z.uv[1]-packet.uv[2*row+1],E=packet.budget[row]*packet.budget[row];double qu=2*eu/E,qv=2*ev/E;
   if(packet.kind[row]==2){const double en=packet.normal[2*row]*eu+packet.normal[2*row+1]*ev;q=en*en/E;qu=2*en*packet.normal[2*row]/E;qv=2*en*packet.normal[2*row+1]/E;}else q=(eu*eu+ev*ev)/E;
   qs=qu*(us+ur*vs/den)+qv*vs/den;err=std::abs(root.row-z.uv[1]);const double tol=1e-5*std::max(1.,q);const auto m=mode[row];Eigen::Matrix3d profile_inv;Vec2<double> Q{},Qprime{};
   if(packet.kind[row]==2){
    if(m!=6){st=5;throw std::runtime_error("profile mode mismatch");}const auto& qr=packet.queries.at(std::size_t(packet.query_index[row]));auto U=rational(qr.numerator_u,alpha[row]),V=rational(qr.numerator_v,alpha[row]),D=rational(qr.denominator,alpha[row]);if(std::abs(D[0])<1e-14){st=3;throw std::runtime_error("query denominator singular");}Q={U[0]/D[0],V[0]/D[0]};Qprime={(U[1]*D[0]-U[0]*D[1])/(D[0]*D[0]),(V[1]*D[0]-V[0]*D[1])/(D[0]*D[0])};err=std::max(err,std::max(std::abs(z.uv[0]-Q[0]),std::abs(z.uv[1]-Q[1])));
    Eigen::Matrix3d A;A<<-vs,den,0,us,ur,-Qprime[0],vs,vr,-Qprime[1];Eigen::FullPivLU<Eigen::Matrix3d> lu(A);rcond=lu.rcond();if(!lu.isInvertible()||!std::isfinite(rcond)||rcond<128*std::numeric_limits<double>::epsilon()){st=3;throw std::runtime_error("profile implicit Jacobian singular/ill-conditioned");}profile_inv=lu.inverse();
    if(alpha[row]<=qr.lower+1e-10||alpha[row]>=qr.upper-1e-10||s[row]<=1e-10||s[row]>=1-1e-10){st=4;throw std::runtime_error("finite profile root on an active support bound");}
    if(curve.kind==2&&std::abs(s[row]*(curve.points.size()-1)-std::round(s[row]*(curve.points.size()-1)))<1e-7){st=4;throw std::runtime_error("profile root at a nonsmooth polyline vertex");}
   }else if(m==0){if(std::abs(qs)>tol){st=1;throw std::runtime_error("smooth foot not stationary; no envelope claim");}}
   else if(m==1){
    if(curve.kind!=2||curve.points.size()<2||std::abs(s[row]*(curve.points.size()-1)-std::round(s[row]*(curve.points.size()-1)))>1e-7){st=5;throw std::runtime_error("polyline vertex mode mismatch");}const double h=1e-7/(curve.points.size()-1);double left=s[row]-h,right=s[row]+h;if(curve.closed){if(left<0)left+=1;if(right>1)right-=1;}if(left<0||right>1){st=4;throw std::runtime_error("vertex is also an open support endpoint");}const Vec2<double> observed{packet.uv[2*row],packet.uv[2*row+1]};const double dl=station_gradient(model,layout,left,layout.frame_time,observed,E),dr=station_gradient(model,layout,right,layout.frame_time,observed,E);if(!std::isfinite(dl)||!std::isfinite(dr)||dl>tol||dr<-tol){st=1;throw std::runtime_error("vertex not a one-sided stationary foot");}
   }else if(m==2){if(curve.closed||!(s[row]<=1e-10||s[row]>=1-1e-10)||((s[row]<=1e-10?qs:-qs)<-tol)){st=1;throw std::runtime_error("open endpoint foot KKT not verified");}}
   else if(m==3||m==4){if(std::abs(root.row-(m==3?1079.:0.))>1e-5||std::abs(vs)<1e-12){st=3;throw std::runtime_error("visibility boundary mode/singularity");}const double mu=(m==3?-qs:qs)/(vs/den);if(mu<-tol){st=1;throw std::runtime_error("visibility boundary multiplier has wrong sign");}}
   else if(m==5){if(!curve.physical_point){st=5;throw std::runtime_error("entity mode does not name a physical point");}}
   else{st=4;throw std::runtime_error("multiple active conditions or unknown mode");}
   if(err>1e-6){st=2;throw std::runtime_error("input root residual too large for an implicit derivative QA");}
   for(std::size_t b=0;b<z.ids.size();++b){const auto id=z.ids[b];const auto& spec=model.blocks[id];for(std::size_t c=0;c<spec.values.size();++c){if(spec.constant||std::find(spec.fixed_indices.begin(),spec.fixed_indices.end(),int(c))!=spec.fixed_indices.end())continue;const auto size=spec.values.size();const double ut=z.jac[b][c],vt=z.jac[b][size+c];double du=ut,dv=vt;
     if(m==6){const Eigen::Vector3d dz=-profile_inv*Eigen::Vector3d(-vt,ut,vt);du+=us*dz[0]+ur*dz[1];dv+=vs*dz[0]+vr*dz[1];}
     else if(m==3||m==4){const double ds=-vt/vs;du+=us*ds;dv+=vs*ds;}
     else{const double dr=vt/den;du+=ur*dr;dv+=vr*dr;}
     const double gradient=qu*du+qv*dv;if(!std::isfinite(gradient)){st=3;throw std::runtime_error("nonfinite qbar physical derivative");}derivative.emplace_back(id,int(c),gradient);
   }}
  }catch(const std::exception& e){if(st==0)st=2;if(diagnoses.size()<100)diagnoses.push_back({{"source_row",row},{"status",st},{"reason",e.what()},{"mode",mode[row]},{"stationarity_q_s",std::isfinite(qs)?Json(qs):Json(nullptr)}});derivative.clear();}
  for(const auto& [b,c,g]:derivative)qcsr.add(b,c,g);qcsr.end();qvalues.push_back(q);stationarity.push_back(qs);maxF.push_back(err);implicit_rcond.push_back(rcond);status.push_back(st);modes.push_back(mode[row]);linearization_s.push_back(s[row]);linearization_v.push_back(std::isfinite(q)?model.project_rs(int(packet.model_material(row)),int(path[row]),s[row],(packet.pts[row]-256256)/12800.,v[row]).row:v[row]);++status_counts[int(st)];if((ordinal+1)%16384==0)std::cout<<Json{{"kind","qbar_sparse_progress"},{"accounted_rows",ordinal+1},{"required_rows",selected.size()}}.dump()<<std::endl;
 }
 const double q_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-projection_started).count();CSR scene_csr;std::vector<double> scene_values;Json scene_ids=Json::array();const auto specs=model.physical_specs();const auto scene_started=std::chrono::steady_clock::now();
 for(const auto& c:specs){ceres::DynamicAutoDiffCostFunction<SceneAD,64> cost(new SceneAD{&model,c});std::vector<const double*> p;std::vector<std::vector<double>> j;std::vector<double*> jp;for(auto id:c.block_ids){cost.AddParameterBlock(int(model.blocks[id].values.size()));p.push_back(model.blocks[id].values.data());j.emplace_back(model.blocks[id].values.size());}for(auto& a:j)jp.push_back(a.data());cost.SetNumResiduals(1);double value;if(!cost.Evaluate(p.data(),&value,jp.data())||!std::isfinite(value))throw std::runtime_error("scene normalized constraint arithmetic failure");scene_values.push_back(value);scene_ids.push_back({{"id",c.id},{"scale",c.scale},{"type",c.type},{"block_ids",c.block_ids}});for(std::size_t b=0;b<c.block_ids.size();++b){const auto id=c.block_ids[b];const auto& spec=model.blocks[id];for(std::size_t i=0;i<spec.values.size();++i)if(!spec.constant&&std::find(spec.fixed_indices.begin(),spec.fixed_indices.end(),int(i))==spec.fixed_indices.end()){if(!std::isfinite(j[b][i]))throw std::runtime_error("scene derivative nonfinite");scene_csr.add(id,int(i),j[b][i]);}}scene_csr.end();}
 auto write=[&](const std::string& filename,const auto& data,const std::string& dtype){using T=typename std::decay_t<decltype(data)>::value_type;std::ofstream f(out/filename,std::ios::binary);if(!data.empty())f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(T)));f.close();if(!f)throw std::runtime_error("qbar binary write failed");return Json{{"file",filename},{"dtype",dtype},{"shape",Json::array({data.size()})},{"sha256",eventio::file_sha256(out/filename)}};};
 auto write_csr=[&](const std::string& prefix,const CSR& c){return Json{{"row_offsets",write(prefix+"row_offsets.u64",c.offsets,"<u8")},{"block_id",write(prefix+"block_id.u32",c.blocks,"<u4")},{"component",write(prefix+"component.u32",c.components,"<u4")},{"values",write(prefix+"values.f64",c.values,"<f8")}};};
 Json report{{"schema","rinkfit.qbar_sparse_physical_AD.v1"},{"no_Solve",true},{"physical_parameters_changed",false},{"production_model_templates_used",true},{"phase_scope","PhaseI independent engineering q only; PhaseII original grouped W still requires joint station optimization"},{"input",{{"path",input_path.u8string()},{"sha256",input_hash}}},{"packet",input.at("packet")},{"model",input.at("model")},{"semantic",input.at("semantic")},{"raw_source_rows",n},{"active_source_rows",packet.active_rows.size()},{"selected_rows",selected.size()},{"qbar_gradient_complete_for_selected_rows",status_counts.size()==1&&status_counts.count(0)},{"status_counts",status_counts},{"status_codes",{{"0","stationary active-chart implicit gradient verified algebraically"},{"1","foot stationarity/KKT unverified"},{"2","physical/root/arithmetic invalid"},{"3","singular/ill-conditioned implicit derivative"},{"4","multiple active/nonsmooth conditions unverified"},{"5","mode disagrees with physical material"},{"8","excluded source retained"}}},{"unverified_gradient_is_not_a_zero_derivative",true},{"stationarity_test_relative_to_max_1_q",1e-5},{"implicit_root_arithmetic_QA_tolerance_px",1e-6},{"source_rows",write("source_rows.u64",selected,"<u8")},{"q",write("q.f64",qvalues,"<f8")},{"status",write("status.u8",status,"|u1")},{"mode",write("mode.u8",modes,"|u1")},{"stationarity_q_s",write("stationarity_q_s.f64",stationarity,"<f8")},{"max_F",write("max_F.f64",maxF,"<f8")},{"profile_implicit_rcond",write("profile_implicit_rcond.f64",implicit_rcond,"<f8")},{"qbar_CSR",write_csr("qbar_",qcsr)},{"scene_CSR",write_csr("scene_",scene_csr)},{"scene_values",write("scene_g.f64",scene_values,"<f8")},{"scene_specs",scene_ids},{"scene_derivative_scope","same active-max-branch AD as production; not a KKT certificate"},{"diagnoses_first_100",diagnoses},{"timing_seconds",{{"qbar",q_seconds},{"scene",std::chrono::duration<double>(std::chrono::steady_clock::now()-scene_started).count()},{"total",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}}},{"final_fit_or_KKT_success",false}};
 report["visibility_active_set_releases"]=visibility_releases;report["visibility_release_version"]="finite_domain_safeguarded_active_set_refiner.v7";report["vertex_refinements_first_100"]=vertex_refinements;report["refiner_version"]="same_finite_path_adjacent_vertex_descent_exactJet.v1";report["refined_station_count"]=refined_stations;report["linearization_station"]=write("linearization_station.f64",linearization_s,"<f8");report["linearization_row"]=write("linearization_row.f64",linearization_v,"<f8");report["nuisance_refinement"]="exact Jet q_s bracketed secant within the same finite segment, strict RS at every trial, no tolerance relaxation";
 if(eventio::file_sha256(input_path)!=input_hash)throw std::runtime_error("immutable input changed during qbar linearization");eventio::atomic_json(out/"manifest.json",report);std::cout<<Json{{"output",out.u8string()},{"selected_rows",selected.size()},{"qbar_nnz",qcsr.values.size()},{"scene_rows",specs.size()},{"scene_nnz",scene_csr.values.size()},{"status_counts",status_counts},{"refined_stations",refined_stations},{"qbar_seconds",q_seconds}}.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
