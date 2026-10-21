#ifndef RINKFIT_EVENT_MODEL_HPP
#define RINKFIT_EVENT_MODEL_HPP

#include "rinkfit/physical.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <type_traits>

namespace rinkfit {

// All lengths are in the exact L=1 world gauge. No frame pose is a parameter.
// global: W,rx,ry,p,H; R0 rotation chart(3); n tangent chart(2),b phase;
// P(3),d(3),axis spacing,pupil linear/quadratic; cx,cy,eta,cx_z,cy_z;
// k1,k1_z,k2,k2_z,p1,p2,tau. f_ref is immutable.
constexpr int kEventGlobalSize=32;
struct EventParameterBlock {
  std::string name;
  std::vector<double> values, lower, upper, scales;
  std::vector<int> fixed_indices;
  bool constant=false;
};
struct EventPath {
  // 0=ice segment,1=ice circle,2=board path in normalized ad coordinates,
  // 3=whole board boundary. Each disconnected path remains independent.
  int kind=0, geometry_block=-1, advertisement_block=-1,radius_index=2,boundary_segment=-1;
  bool physical_point=false, closed=false;
  std::string id, variant_id, provenance;
  std::vector<Vec2<double>> points;
  double start_angle=0, end_angle=6.2831853071795864769;
  double fixed_height_fraction=0;
};
struct EventMaterial {
  std::string label;
  std::vector<EventPath> paths;
};
struct EventRowLayout {
  std::vector<int> block_ids;
  int global_slot=0, geometry_slot=-1, advertisement_slot=-1;
  std::vector<int> control_ids, control_slots;
  int station_slot=-1, row_slot=-1;
  double frame_time=0;
  int material=0,path=0;
};
struct EventRootResult {
  Vec2<double> uv{NAN,NAN};
  double row=NAN, depth=NAN, equality_px=INFINITY, implicit_denominator=NAN;
  int roots=0;
  bool valid=false, unique=false, sampled_monotonic=false;
  std::string reason;
};
struct EventPhysicalConstraint {
  std::string id;
  std::vector<int> block_ids;
  int type=0,index=0,axis=0,direction=1;
  // Nonempty only for a static polygon's exact affine canvas support test.
  std::vector<Vec2<double>> vertices;
  double scale=1;
};
struct EventRayResult {
  Vec3<double> origin{NAN,NAN,NAN},direction{NAN,NAN,NAN},ice{NAN,NAN,NAN};
  double native_time=NAN,inverse_distortion_error=INFINITY;
  bool valid=false,forward_ice_intersection=false,inside_rink=false;
  std::string reason;
};
// Finite arithmetic continuation used ONLY by the AL objective. A finite
// value beyond the frozen optical domain is not a physical projection.
template<class T> struct EventALProjection {
  Vec2<T> uv{T(std::numeric_limits<double>::quiet_NaN()),T(std::numeric_limits<double>::quiet_NaN())};
  T depth=T(std::numeric_limits<double>::quiet_NaN());
  T ray_radius_sq=T(std::numeric_limits<double>::quiet_NaN());
  T brown_lower=T(std::numeric_limits<double>::quiet_NaN());
  DomainStatus physical_status=DomainStatus::kNonFinite;
  bool arithmetic_valid=false;
  bool physical_valid() const {return arithmetic_valid&&physical_status==DomainStatus::kValid;}
};

template<class T> inline double event_scalar(const T& x) {
  if constexpr(std::is_arithmetic<T>::value) return double(x);
  else return event_scalar(x.a);
}
template<class T> inline T event_pow_nonnegative(const T& x,const T& p) {
  using std::pow;
  // x==0 is an analytic endpoint; d(0^p)/dp=0 for all admitted p>1.
  return event_scalar(x)<=0 ? T(0) : pow(x,p);
}
template<class T> inline Vec3<T> event_normalized(const Vec3<T>& v) {
  using std::sqrt; return scale(v,T(1)/sqrt(dot(v,v)));
}
template<class T> inline Mat3<T> event_exp_rotation(const Vec3<T>& w) {
  using std::sqrt;using std::sin;using std::cos;
  const T q=dot(w,w); T A,B;
  if(event_scalar(q)<1e-12){ A=T(1)-q/T(6)+q*q/T(120); B=T(.5)-q/T(24)+q*q/T(720); }
  else {const T t=sqrt(q);A=sin(t)/t;B=(T(1)-cos(t))/q;}
  const T x=w[0],y=w[1],z=w[2];
  return {T(1)-B*(y*y+z*z),B*x*y-A*z,B*x*z+A*y,
          B*x*y+A*z,T(1)-B*(x*x+z*z),B*y*z-A*x,
          B*x*z-A*y,B*y*z+A*x,T(1)-B*(x*x+y*y)};
}

class EventModel {
 public:
  std::string schema="rinkfit.event_model.v1.1",model_id,source_identity;
  std::vector<EventParameterBlock> blocks;
  std::vector<EventMaterial> materials;
  Mat3<double> R0_seed=identity3<double>();
  Vec3<double> n_seed{0,0,1},b_seed{1,0,0},n_tangent1{1,0,0},n_tangent2{0,1,0};
  double f_ref=1, radius_max=.5,dt=.02,first_control_time=-.04;
  double wide_log_f_boundary=0;
  std::array<double,4> wide_coefficients{};
  int n_controls=0,control_block_begin=1,dependent_control=2;
  std::array<int,4> reference_indices{1,2,3,4};
  std::array<double,4> reference_weights{1./6,4./6,1./6,0};
  double tau_bound=.02;
  static EventModel load(const std::string& json_path);
  EventRowLayout layout(int material,int path,double frame_time) const;
  std::vector<const double*> pointers(const EventRowLayout& layout) const;
  std::array<double,3> restricted_trajectory_derivative_bounds(double lo,double hi) const;
  EventRootResult project_rs(int material,int path,double station,double frame_time,
                            double initial_row=539.5) const;
  EventRayResult inverse_ray(double native_u,double native_v,double frame_time) const;
  std::vector<double> physical_constraints() const;
  std::vector<EventPhysicalConstraint> physical_specs() const;
  double corner_length() const;
  template<class T> static T rink_membership(const T& x,const T& y,const T* g) {
    using std::abs;const T xx=(abs(x)-(T(.5)-g[1]))/g[1],yy=(abs(y)-(g[0]/T(2)-g[2]))/g[2];
    return event_pow_nonnegative(xx,g[3])+event_pow_nonnegative(yy,g[3])-T(1);
  }
  template<class T> T evaluate_physical(const EventPhysicalConstraint& c,T const*const* p) const {
    const T* g=p[0];const T* q=c.block_ids.size()>1?p[1]:g;T value=T(0);
    if(c.type==0)value=T(2)*g[2]-g[0]+T(2e-5);
    else if(c.type==1)value=-q[2];
    else if(c.type==2)value=q[2]-q[3]+T(1e-6);
    else if(c.type==3)value=q[3]-g[4];
    else if(c.type==4){const T per=T(2)*(T(1)-T(2)*g[1]+g[0]-T(2)*g[2])+T(4)*quarter_arclength(T(1.5707963267948966),g);value=q[1]-per;}
    else if(c.type==5)value=T(1e-6)-q[c.index];
    else if(c.type==6)value=q[c.index]-q[c.index+1]+T(1e-6);
    else if(c.type==7)value=rink_membership(q[c.index],q[c.index+1],g);
    else if(c.type==8){const T dx=q[0]-q[2],dy=q[1]-q[3];value=T(1e-10)-dx*dx-dy*dy;}
    else if(c.type==9){const T r=q[c.index];value=rink_membership(q[0]+T(c.axis)*r,q[1]+T(c.direction)*r,g);}
    else if(c.type==10){bool first=true;for(const auto& xy:c.vertices){const T x=q[c.axis*3]*T(xy[0])+q[c.axis*3+1]*T(xy[1])+q[c.axis*3+2];
        const T v=c.direction>0?x-T(1):-x;if(first||event_scalar(v)>event_scalar(value)){value=v;first=false;}}}
    else if(c.type==11)value=T(1e-6)-(q[0]*q[4]-q[1]*q[3]);
    else if(c.type==12){const T* a=p[2];bool first=true;for(const auto& xy:c.vertices){const T vv=q[3]*T(xy[0])+q[4]*T(xy[1])+q[5];
        const T h=a[2]+(a[3]-a[2])*vv,v=c.direction>0?h-g[4]:-h;
        if(first||event_scalar(v)>event_scalar(value)){value=v;first=false;}}}
    else throw std::domain_error("unknown physical constraint");
    return value/T(c.scale);
  }

  template<class T> T control(int i,int channel,T const*const* pars,const EventRowLayout& l) const {
    auto raw=[&](int j)->T {for(std::size_t k=0;k<l.control_ids.size();++k)
      if(l.control_ids[k]==j)return pars[l.control_slots[k]][channel];
      throw std::out_of_range("event cubic dependency missing");};
    if(channel<2 && i==dependent_control){T sum=T(0);double pivot=0;
      for(int k=0;k<4;++k){if(reference_indices[k]==i)pivot=reference_weights[k];
        else if(reference_weights[k]!=0)sum+=T(reference_weights[k])*raw(reference_indices[k]);}
      return -sum/T(pivot);}
    return raw(i);
  }
  template<class T> std::array<T,3> trajectory(const T& time,T const*const* pars,const EventRowLayout& l) const {
    const T x=(time-T(first_control_time))/T(dt);
    const int cell=int(std::floor(event_scalar(x)));
    if(cell<1||cell+2>=n_controls)throw std::domain_error("outside allocated native cubic domain");
    const T u=x-T(cell); const auto w=cubic_weights(u);std::array<T,3> out{};
    for(int k=0;k<4;++k)for(int c=0;c<3;++c)out[c]+=w[k]*control(cell-1+k,c,pars,l);
    return out;
  }
  template<class T> FixedHead<T> head(const T* g) const {
    FixedHead<T> h;
    Mat3<T> rseed;Vec3<T> ns,bs,u,v;for(int i=0;i<9;++i)rseed[i]=T(R0_seed[i]);
    for(int i=0;i<3;++i){ns[i]=T(n_seed[i]);bs[i]=T(b_seed[i]);u[i]=T(n_tangent1[i]);v[i]=T(n_tangent2[i]);}
    h.R0=matmul(event_exp_rotation(Vec3<T>{g[5],g[6],g[7]}),rseed);
    h.n=event_normalized(add(ns,add(scale(u,g[8]),scale(v,g[9]))));
    const auto q=matvec(h.R0,h.n);
    const auto e=event_normalized(sub(bs,scale(q,dot(bs,q))));const auto f=cross(q,e);
    using std::sin;using std::cos;h.b=add(scale(e,cos(g[10])),scale(f,sin(g[10])));
    h.P={g[11],g[12],g[13]};h.d={g[14],g[15],g[16]};h.a=g[17];
    h.pupil_linear=g[18];h.pupil_quadratic=g[19];return h;
  }
  template<class T> ZoomOptics<T> optics(const T* g) const {
    ZoomOptics<T> o;o.f_ref=f_ref;o.radius_max=T(radius_max);o.wide_log_f_boundary=wide_log_f_boundary;o.wide_coefficients=wide_coefficients;
    o.cx=g[20];o.cy=g[21];o.eta=g[22];o.cx_linear=g[23];o.cy_linear=g[24];
    o.k1=g[25];o.k1_linear=g[26];o.k2=g[27];o.k2_linear=g[28];o.p1=g[29];o.p2=g[30];return o;
  }
  // Polar superellipse quarter: regular finite first derivatives at endpoints
  // for all p>1. There is no silent ellipse replacement of the old p candidate.
  template<class T> static Vec2<T> quarter(const T& angle,const T* g) {
    using std::sin;using std::cos;using std::pow;
    const T c=cos(angle),s=sin(angle),d=pow(event_pow_nonnegative(c,g[3])+event_pow_nonnegative(s,g[3]),-T(1)/g[3]);
    return {g[1]*c*d,g[2]*s*d};
  }
  template<class T> static T quarter_speed(const T& a,const T* g) {
    using std::sin;using std::cos;using std::pow;using std::sqrt;
    const T c=cos(a),s=sin(a),cp=event_pow_nonnegative(c,g[3]),sp=event_pow_nonnegative(s,g[3]);
    const T d=pow(cp+sp,-T(1)/g[3]);
    const T v=(event_pow_nonnegative(c,g[3]-T(1))*s-event_pow_nonnegative(s,g[3]-T(1))*c)/(cp+sp);
    const T x=g[1]*d*(-s+c*v),y=g[2]*d*(c+s*v);return sqrt(x*x+y*y);
  }
  template<class T> static T quarter_arclength(const T& angle,const T* g) {
    // 32-point composite Gauss quadrature, endpoints never evaluated.
    static constexpr double x[8]={.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499};
    static constexpr double w[8]={.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541};
    T total=T(0);for(int part=0;part<2;++part){const T mid=angle*T((part+.5)/2),half=angle/T(4);
      for(int i=0;i<8;++i)total+=half*T(w[i])*(quarter_speed(mid-half*T(x[i]),g)+quarter_speed(mid+half*T(x[i]),g));}return total;
  }
  template<class T> static Vec3<T> board_point(T s,const T& h,const T* g) {
    constexpr double pi=3.14159265358979323846;
    const T longlen=T(1)-T(2)*g[1],shortlen=g[0]-T(2)*g[2],arc=quarter_arclength(T(pi/2),g);
    const T perimeter=T(2)*(longlen+shortlen)+T(4)*arc;
    s-=T(std::floor(event_scalar(s/perimeter)))*perimeter;
    const T lengths[8]={longlen,arc,shortlen,arc,longlen,arc,shortlen,arc};
    int seg=0;while(seg<7 && event_scalar(s)>event_scalar(lengths[seg]))s-=lengths[seg++];
    const T hx=T(.5),hy=g[0]/T(2),rx=g[1],ry=g[2];
    if(seg==0)return {-hx+rx+s,-hy,h};if(seg==2)return {hx,-hy+ry+s,h};
    if(seg==4)return {hx-rx-s,hy,h};if(seg==6)return {-hx,hy-ry-s,h};
    // Invert arclength with safeguarded scalar Newton then implicit derivative.
    double gd[kEventGlobalSize];for(int j=0;j<kEventGlobalSize;++j)gd[j]=event_scalar(g[j]);
    const int k=(seg-1)/2;const T target=(k==0||k==2)?arc-s:s;
    double lo=0,hi=pi/2,a=std::max(0.,std::min(pi/2,event_scalar(target/arc)*pi/2));
    for(int it=0;it<48;++it){const double f=quarter_arclength(a,gd)-event_scalar(target);if(std::abs(f)<1e-13)break;
      if(f>0)hi=a;else lo=a;const double next=a-f/quarter_speed(a,gd);a=(next>lo&&next<hi)?next:(lo+hi)/2;}
    if(std::abs(quarter_arclength(a,gd)-event_scalar(target))>1e-11)throw std::domain_error("superellipse arclength inversion failed");
    const T theta=T(a)-(quarter_arclength(T(a),g)-target)/quarter_speed(T(a),g);
    auto q=quarter(theta,g);
    // arc0 runs bottom-right from bottom to right; signs reverse polar chart.
    if(k==0)return {hx-rx+q[0],-hy+ry-q[1],h};
    if(k==1)return {hx-rx+q[0],hy-ry+q[1],h};
    if(k==2)return {-hx+rx-q[0],hy-ry+q[1],h};
    return {-hx+rx-q[0],-hy+ry-q[1],h};
  }
  template<class T> Vec3<T> world_point(const EventPath& p,const T& station,T const*const* pars,const EventRowLayout& l) const {
    if(event_scalar(station)<0||event_scalar(station)>1)throw std::domain_error("finite material station outside [0,1]");
    const T* g=pars[l.global_slot];const T* q=l.geometry_slot<0?nullptr:pars[l.geometry_slot];
    using std::sin;using std::cos;
    if(p.kind==0)return {q[0]*(T(1)-station)+q[2]*station,q[1]*(T(1)-station)+q[3]*station,T(0)};
    if(p.kind==1){const T a=T(p.start_angle)+station*T(p.end_angle-p.start_angle);return {q[0]+q[p.radius_index]*cos(a),q[1]+q[p.radius_index]*sin(a),T(0)};}
    if(p.kind==3){const T arc=quarter_arclength(T(1.5707963267948966),g),ll=T(1)-T(2)*g[1],sl=g[0]-T(2)*g[2];
      const T perimeter=T(2)*(ll+sl)+T(4)*arc;T position=station*perimeter;
      if(p.boundary_segment>=0){const T lengths[8]={ll,arc,sl,arc,ll,arc,sl,arc};position=station*lengths[p.boundary_segment];for(int j=0;j<p.boundary_segment;++j)position+=lengths[j];}
      return board_point(position,g[4]*T(p.fixed_height_fraction),g);}
    if(p.points.empty())throw std::domain_error("empty board path");
    Vec2<T> uv;
    if(p.points.size()==1)uv={T(p.points[0][0]),T(p.points[0][1])};
    else {const T x=station*T(p.points.size()-1);const int i=std::min(int(p.points.size())-2,int(std::floor(event_scalar(x))));
      const T u=x-T(i);uv={T(p.points[i][0])*(T(1)-u)+T(p.points[i+1][0])*u,T(p.points[i][1])*(T(1)-u)+T(p.points[i+1][1])*u};}
    if(q){const T x=uv[0],y=uv[1];uv={q[0]*x+q[1]*y+q[2],q[3]*x+q[4]*y+q[5]};}
    const T* ad=pars[l.advertisement_slot];
    return board_point(ad[0]+ad[1]*uv[0],ad[2]+(ad[3]-ad[2])*uv[1],g);
  }
  template<class T> Projection<T> predict_explicit(const EventRowLayout& l,const T& station,const T& row,T const*const* pars) const {
    const T* g=pars[l.global_slot];const T time=row_time(T(l.frame_time),g[31],row);
    const auto tr=trajectory(time,pars,l);const auto o=zoom_optics(optics(g),tr[2]);
    return project(world_point(materials.at(l.material).paths.at(l.path),station,pars,l),camera_pose(head(g),tr[0],tr[1],o.z),o.intr,o.brown);
  }
  template<class T> EventALProjection<T> predict_for_al(const EventRowLayout& l,const T& station,const T& row,T const*const* pars) const {
    EventALProjection<T> out;const T* g=pars[l.global_slot];const T time=row_time(T(l.frame_time),g[31],row);
    const auto tr=trajectory(time,pars,l);const auto o=zoom_optics(optics(g),tr[2]);const auto pose=camera_pose(head(g),tr[0],tr[1],o.z);
    if(!pose.valid()){out.physical_status=pose.status;return out;}
    if(!valid_rotation(pose.R)){out.physical_status=DomainStatus::kInvalidRotation;return out;}
    const auto world=world_point(materials.at(l.material).paths.at(l.path),station,pars,l);
    const std::array<T,9> p={o.intr.fx,o.intr.fy,o.intr.cx,o.intr.cy,o.brown.k1,o.brown.k2,o.brown.p1,o.brown.p2,o.brown.radius_max};
    if(!finite_array(world)||!finite_array(pose.C)||!finite_array(p)||!(o.intr.fx>T(0))||!(o.intr.fy>T(0))||!(o.brown.radius_max>T(0)))return out;
    const auto xc=matvec(pose.R,sub(world,pose.C));if(!finite_array(xc))return out;out.depth=xc[2];
    // The perspective singularity still has no finite arithmetic extension.
    // Its explicit g_depth normal is registered, but zero depth cannot pass
    // the strict native replay or be called a physical solution.
    if(!(out.depth>T(0))){out.physical_status=DomainStatus::kNonPositiveDepth;return out;}
    const Vec2<T> xy={xc[0]/out.depth,xc[1]/out.depth};if(!finite_array(xy))return out;
    out.ray_radius_sq=xy[0]*xy[0]+xy[1]*xy[1];out.brown_lower=brown_no_fold_lower_bound(o.brown);
    if(!finite_scalar(out.ray_radius_sq)||!finite_scalar(out.brown_lower))return out;
    const auto distorted=brown_distort(xy,o.brown);out.uv={o.intr.fx*distorted[0]+o.intr.cx,o.intr.fy*distorted[1]+o.intr.cy};
    if(!finite_array(out.uv))return out;
    out.physical_status=out.ray_radius_sq>o.brown.radius_max*o.brown.radius_max?DomainStatus::kOutsideDistortionDomain:
      (!(out.brown_lower>T(0))?DomainStatus::kDistortionUncertified:DomainStatus::kValid);
    out.arithmetic_valid=true;return out;
  }
};
} // namespace rinkfit
#endif
