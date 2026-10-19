#pragma once
#include "abi.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#define HD __host__ __device__
namespace gpu_exact {
HD inline bool scalar_finite(double x){
#ifdef __CUDA_ARCH__
return (static_cast<unsigned long long>(__double_as_longlong(x)) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
#else
return std::isfinite(x);
#endif
}

__device__ int gpu_first_math_failure=0;
[[noreturn]] HD inline void gpu_abort(int line){
#ifdef __CUDA_ARCH__
  if(atomicCAS(&gpu_first_math_failure,0,line)==0)printf("GPU math guard line %d\n",line);
  asm("trap;");
#else
  throw std::domain_error("GPU experiment unsupported arithmetic-domain exception");
#endif
}
struct Jet {double a,v;HD Jet(double x=0):a(x),v(0){}HD Jet(double x,double d):a(x),v(d){}
HD Jet& operator+=(Jet b){a+=b.a;v+=b.v;return *this;}
HD Jet& operator-=(Jet b){a-=b.a;v-=b.v;return *this;}
HD Jet& operator*=(Jet b){v=v*b.a+a*b.v;a*=b.a;return *this;}
HD Jet& operator/=(Jet b){const double inverse=1./b.a,ratio=a*inverse;v=(v-ratio*b.v)*inverse;a=ratio;return *this;}};
HD inline Jet operator+(Jet a,Jet b){return a+=b;}HD inline Jet operator-(Jet a,Jet b){return a-=b;}
HD inline Jet operator*(Jet a,Jet b){return a*=b;}HD inline Jet operator/(Jet a,Jet b){return a/=b;}
HD inline Jet operator-(Jet a){return Jet(-a.a,-a.v);}
HD inline bool operator<(Jet a,Jet b){return a.a<b.a;}HD inline bool operator>(Jet a,Jet b){return a.a>b.a;}
HD inline bool operator<=(Jet a,Jet b){return a.a<=b.a;}HD inline bool operator>=(Jet a,Jet b){return a.a>=b.a;}
HD inline Jet sin(Jet x){return Jet(::sin(x.a),::cos(x.a)*x.v);}HD inline Jet cos(Jet x){return Jet(::cos(x.a),-::sin(x.a)*x.v);}
HD inline Jet sqrt(Jet x){double y=::sqrt(x.a),scale=.5/y;return Jet(y,x.v*scale);}HD inline Jet exp(Jet x){double y=::exp(x.a);return Jet(y,y*x.v);}
HD inline Jet pow(Jet x,Jet p){double y=::pow(x.a,p.a),dx=p.a*::pow(x.a,p.a-1.),dp=y*::log(x.a);return Jet(y,dx*x.v+dp*p.v);}
HD inline Jet abs(Jet x){return x.a<0?-x:x;}HD inline bool isfinite(Jet x){return scalar_finite(x.a)&&scalar_finite(x.v);}
template<class T>struct Span {const T* data=nullptr;int count=0;HD const T& operator[](int i)const{return data[i];}HD const T& at(int i)const{return data[i];}HD int size()const{return count;}HD bool empty()const{return count==0;}};
} // namespace gpu_exact
namespace gpu_exact {

template <class T> using Vec2 = std::array<T, 2>;
template <class T> using Vec3 = std::array<T, 3>;
template <class T> using Mat3 = std::array<T, 9>;

enum class DomainStatus {
  kValid,
  kNonPositiveDepth,
  kOutsideDistortionDomain,
  kDistortionUncertified,
  kNonFinite,
  kInvalidRotation,
  kSingularAxis,
  kOutsideFiniteMaterial
};

template <class T> HD inline bool finite_scalar(const T& x) {
  if constexpr(std::is_arithmetic<T>::value)return scalar_finite(double(x));
  else return isfinite(x);
}
template <class T, std::size_t N>
HD inline bool finite_array(const std::array<T, N>& x) {
  for (const auto& v : x) if (!finite_scalar(v)) return false;
  return true;
}
template <class T> HD inline T abs_value(const T& x) {
  using std::abs;
  return abs(x);
}
template <class T> HD inline Vec3<T> add(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
template <class T> HD inline Vec3<T> sub(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
template <class T> HD inline Vec3<T> scale(const Vec3<T>& a, const T& s) {
  return {s*a[0], s*a[1], s*a[2]};
}
template <class T> HD inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
template <class T> HD inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
template <class T> HD inline Mat3<T> identity3() {
  return {T(1),T(0),T(0), T(0),T(1),T(0), T(0),T(0),T(1)};
}
template <class T> HD inline Mat3<T> transpose(const Mat3<T>& A) {
  return {A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8]};
}
template <class T> HD inline Mat3<T> matmul(const Mat3<T>& A, const Mat3<T>& B) {
  Mat3<T> C{};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int k=0;k<3;++k) C[3*i+j] += A[3*i+k]*B[3*k+j];
  return C;
}
template <class T> HD inline Vec3<T> matvec(const Mat3<T>& A, const Vec3<T>& v) {
  return {A[0]*v[0]+A[1]*v[1]+A[2]*v[2],
          A[3]*v[0]+A[4]*v[1]+A[5]*v[2],
          A[6]*v[0]+A[7]*v[1]+A[8]*v[2]};
}
template <class T> HD inline T determinant(const Mat3<T>& A) {
  return A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])
       + A[2]*(A[3]*A[7]-A[4]*A[6]);
}
template <class T> HD inline bool valid_rotation(const Mat3<T>& R) {
  if (!finite_array(R)) return false;
  const auto RtR = matmul(transpose(R), R);
  for (int i=0;i<9;++i)
    if (abs_value(RtR[i]-T(i%4==0 ? 1 : 0)) > T(1e-9)) return false;
  return abs_value(determinant(R)-T(1)) <= T(1e-9);
}

// The axis must already be unit length.  No normalization hides an invalid
// mechanism state; optimizer-side manifolds/charts must enforce that property.
template <class T> HD inline Mat3<T> rodrigues(const Vec3<T>& axis, const T& angle) {
  if (!finite_array(axis) || !finite_scalar(angle) || abs_value(dot(axis,axis)-T(1)) > T(1e-9))
    gpu_abort(__LINE__);
  using std::sin; using std::cos;
  const T s=sin(angle), q=T(1)-cos(angle);
  const T x=axis[0], y=axis[1], z=axis[2];
  return {T(1)-q*(y*y+z*z), q*x*y-s*z, q*x*z+s*y,
          q*x*y+s*z, T(1)-q*(x*x+z*z), q*y*z-s*x,
          q*x*z-s*y, q*y*z+s*x, T(1)-q*(x*x+y*y)};
}

template <class T> struct Pose {
  Mat3<T> R=identity3<T>();
  Vec3<T> C{T(0),T(0),T(0)};
  DomainStatus status=DomainStatus::kValid;
  HD bool valid() const { return status==DomainStatus::kValid; }
};

template <class T> struct FixedHead {
  Mat3<T> R0=identity3<T>();
  Vec3<T> n{T(0),T(1),T(0)};  // fixed world pan axis
  Vec3<T> b{T(1),T(0),T(0)};  // camera-local tilt axis
  Vec3<T> P{T(0),T(0),T(0)};
  Vec3<T> d{T(0),T(0),T(0)};
  T a=T(0);
  T pupil_linear=T(0), pupil_quadratic=T(0);  // ell(0)=0 exactly
};

template <class T>
HD inline Pose<T> camera_pose(const FixedHead<T>& head, const T& alpha, const T& beta, const T& z) {
  Pose<T> result;
  if (!valid_rotation(head.R0)) { result.status=DomainStatus::kInvalidRotation; return result; }
  if (!finite_array(head.n) || !finite_array(head.b) || !finite_array(head.P) ||
      !finite_array(head.d) || !finite_scalar(head.a) || !finite_scalar(head.pupil_linear) ||
      !finite_scalar(head.pupil_quadratic) || !finite_scalar(alpha) || !finite_scalar(beta) ||
      !finite_scalar(z)) { result.status=DomainStatus::kNonFinite; return result; }
  if (abs_value(dot(head.n,head.n)-T(1)) > T(1e-9) ||
      abs_value(dot(head.b,head.b)-T(1)) > T(1e-9) ||
      abs_value(dot(head.b,matvec(head.R0,head.n))) > T(1e-9)) {
    result.status=DomainStatus::kSingularAxis; return result;
  }
  result.R=matmul(matmul(rodrigues(head.b,beta),head.R0),rodrigues(head.n,alpha));
  const auto Rt=transpose(result.R);
  const auto normal=cross(head.n,matvec(Rt,head.b));
  using std::sqrt;
  const T norm=sqrt(dot(normal,normal));
  if (!(norm>T(1e-12))) { result.status=DomainStatus::kSingularAxis; return result; }
  const T ell=head.pupil_linear*z+head.pupil_quadratic*z*z;
  auto lever=head.d; lever[2]+=ell;
  result.C=add(add(head.P,matvec(Rt,lever)),scale(normal,head.a/norm));
  if (!finite_array(result.C) || !finite_array(result.R)) result.status=DomainStatus::kNonFinite;
  return result;
}

template <class T> struct Intrinsics {
  T fx=T(1), fy=T(1), cx=T(0), cy=T(0);
};
template <class T> struct Brown {
  T k1=T(0), k2=T(0), p1=T(0), p2=T(0), radius_max=T(2);
};
template <class T> struct ZoomOptics {
  // f_ref is a fixed double gauge, never an optimized second focal origin.
  double f_ref=1.0;
  T cx=T(959.5), cy=T(539.5), eta=T(0);
  T cx_linear=T(0), cy_linear=T(0);
  T k1=T(0), k1_linear=T(0), k2=T(0), k2_linear=T(0);
  T p1=T(0), p2=T(0), radius_max=T(2);
};
template <class T> struct OpticalState {
  Intrinsics<T> intr;
  Brown<T> brown;
  T z=T(0);
};
template <class T> HD inline OpticalState<T> zoom_optics(const ZoomOptics<T>& optics, const T& log_f) {
  if (!scalar_finite(optics.f_ref) || !(optics.f_ref>0))
    gpu_abort(__LINE__);
  using std::exp;
  OpticalState<T> out;
  out.z=log_f-T(std::log(optics.f_ref));
  out.intr={exp(log_f+optics.eta/T(2)),exp(log_f-optics.eta/T(2)),
            optics.cx+optics.cx_linear*out.z,optics.cy+optics.cy_linear*out.z};
  out.brown={optics.k1+optics.k1_linear*out.z,optics.k2+optics.k2_linear*out.z,
             optics.p1,optics.p2,optics.radius_max};
  return out;
}
template <class T> HD inline T quadratic_minimum(const T& c0,const T& c1,const T& c2,const T& xmax) {
  T low=c0;
  const T end=c0+c1*xmax+c2*xmax*xmax;
  if (end<low) low=end;
  if (c2>T(0)) {
    const T x=-c1/(T(2)*c2);
    if (x>T(0) && x<xmax) { const T val=c0+c1*x+c2*x*x; if (val<low) low=val; }
  }
  return low;
}
template <class T> HD inline T brown_no_fold_lower_bound(const Brown<T>& d) {
  const T r2=d.radius_max*d.radius_max;
  const T tangential=quadratic_minimum(T(1),d.k1,d.k2,r2);
  const T radial=quadratic_minimum(T(1),T(3)*d.k1,T(5)*d.k2,r2);
  using std::sqrt;
  // Avoid the undefined Jet derivative of sqrt(0) at p1=p2=0: the certificate
  // is a domain predicate, not a fitted residual; its selected zero branch is
  // sufficient to test the undistorted candidate exactly.
  const T pnorm2=d.p1*d.p1+d.p2*d.p2;
  const T pnorm=pnorm2>T(0) ? sqrt(pnorm2) : T(0);
  return (tangential<radial ? tangential : radial)-T(6)*pnorm*d.radius_max;
}
template <class T> HD inline Vec2<T> brown_distort(const Vec2<T>& xy,const Brown<T>& d) {
  const T x=xy[0],y=xy[1],r2=x*x+y*y,L=T(1)+d.k1*r2+d.k2*r2*r2;
  return {x*L+T(2)*d.p1*x*y+d.p2*(r2+T(2)*x*x),
          y*L+d.p1*(r2+T(2)*y*y)+T(2)*d.p2*x*y};
}
template <class T> struct Projection {
  Vec2<T> uv{T(std::numeric_limits<double>::quiet_NaN()),T(std::numeric_limits<double>::quiet_NaN())};
  T depth=T(std::numeric_limits<double>::quiet_NaN());
  DomainStatus status=DomainStatus::kNonFinite;
  HD bool valid() const { return status==DomainStatus::kValid; }
};
template <class T> HD inline Projection<T> project(const Vec3<T>& world,const Pose<T>& pose,
                                                const Intrinsics<T>& intr,const Brown<T>& brown) {
  Projection<T> out;
  if (!pose.valid()) { out.status=pose.status; return out; }
  if (!valid_rotation(pose.R)) { out.status=DomainStatus::kInvalidRotation; return out; }
  const std::array<T,9> params={intr.fx,intr.fy,intr.cx,intr.cy,brown.k1,brown.k2,brown.p1,brown.p2,brown.radius_max};
  if (!finite_array(world) || !finite_array(pose.C) || !finite_array(params) ||
      !(intr.fx>T(0)) || !(intr.fy>T(0)) || !(brown.radius_max>T(0))) return out;
  const auto xc=matvec(pose.R,sub(world,pose.C));
  if (!finite_array(xc)) return out;
  out.depth=xc[2];
  if (!(xc[2]>T(0))) { out.status=DomainStatus::kNonPositiveDepth; return out; }
  const Vec2<T> xy={xc[0]/xc[2],xc[1]/xc[2]};
  if (!finite_array(xy)) return out;
  if (xy[0]*xy[0]+xy[1]*xy[1]>brown.radius_max*brown.radius_max) {
    out.status=DomainStatus::kOutsideDistortionDomain; return out;
  }
  if (!(brown_no_fold_lower_bound(brown)>T(0))) {
    out.status=DomainStatus::kDistortionUncertified; return out;
  }
  const auto distorted=brown_distort(xy,brown);
  const Vec2<T> pixels={intr.fx*distorted[0]+intr.cx,intr.fy*distorted[1]+intr.cy};
  if (!finite_array(pixels)) return out;
  out.uv=pixels; out.status=DomainStatus::kValid; return out;
}

// This is the explicit row-time *equation core*, not an implicit-root solver.
// The full Python reference returns root multiplicity, domain and derivative
// statuses; production C++ final replay still needs an equivalent root solver.
template <class T> HD inline T row_time(const T& frame_time,const T& signed_readout,
                                     const T& predicted_row,int image_height=1080) {
  if (image_height<2 || !finite_scalar(frame_time) || !finite_scalar(signed_readout) || !finite_scalar(predicted_row))
    gpu_abort(__LINE__);
  return frame_time+signed_readout*(predicted_row/T(image_height-1)-T(0.5));
}
template <class T> HD inline T explicit_rs_equality(const T& row,const Projection<T>& instant) {
  if (!instant.valid()) gpu_abort(__LINE__);
  return row-instant.uv[1];
}

template <class T> HD inline bool valid_station(const T& s,const T& lo,const T& hi) {
  return finite_scalar(s) && finite_scalar(lo) && finite_scalar(hi) && lo<=hi && s>=lo && s<=hi;
}
template <class T> HD inline Vec3<T> finite_line_point(const Vec3<T>& A,const Vec3<T>& B,const T& s) {
  if (!finite_array(A) || !finite_array(B) || !valid_station(s,T(0),T(1)) || !(dot(sub(B,A),sub(B,A))>T(0)))
    gpu_abort(__LINE__);
  return add(scale(A,T(1)-s),scale(B,s));
}
template <class T> HD inline Vec3<T> circle_arc_point(const Vec2<T>& center,const T& radius,
                                                   const T& angle,const T& start,const T& end) {
  constexpr double two_pi=6.283185307179586476925286766559;
  if (!finite_array(center) || !finite_scalar(radius) || !(radius>T(0)) ||
      !valid_station(angle,start,end) || !(end>start) || end-start>T(two_pi+1e-12))
    gpu_abort(__LINE__);
  using std::sin; using std::cos;
  return {center[0]+radius*cos(angle),center[1]+radius*sin(angle),T(0)};
}
template <class T> HD inline Vec3<T> ellipse_corner_point(const Vec2<T>& center,const T& rx,const T& ry,
                                                      const T& angle,const T& start,const T& end) {
  if (!finite_array(center) || !finite_scalar(rx) || !finite_scalar(ry) || !(rx>T(0)) || !(ry>T(0)) ||
      !valid_station(angle,start,end) || !(end>start))
    gpu_abort(__LINE__);
  using std::sin; using std::cos;
  return {center[0]+rx*cos(angle),center[1]+ry*sin(angle),T(0)};
}

// Local cardinal cubic values; knot allocation/time-domain checking lives in
// the trajectory layer.  Exactly four controls enter each local evaluation.
template <class T> HD inline std::array<T,4> cubic_weights(const T& u) {
  if (!valid_station(u,T(0),T(1))) gpu_abort(__LINE__);
  const T u2=u*u,u3=u2*u,v=T(1)-u;
  return {v*v*v/T(6),(T(3)*u3-T(6)*u2+T(4))/T(6),
          (-T(3)*u3+T(3)*u2+T(3)*u+T(1))/T(6),u3/T(6)};
}
template <class T> HD inline T cubic_value(const std::array<T,4>& controls,const T& u) {
  if (!finite_array(controls)) gpu_abort(__LINE__);
  const auto w=cubic_weights(u);
  T out=T(0); for (int i=0;i<4;++i) out+=w[i]*controls[i]; return out;
}
// Eliminate the selected reference-support coefficient exactly.  fixed_index
// is the largest reference weight, selected once outside AutoDiff.  The other
// three controls are ordered by ascending reference-support index.
template <class T> HD inline T gauge_dependent_control(const std::array<double,4>& reference_weights,
                                                    int fixed_index,const std::array<T,3>& free_controls) {
  if (fixed_index<0 || fixed_index>3 || !finite_array(reference_weights) ||
      std::abs(reference_weights[fixed_index])<1e-12 || !finite_array(free_controls))
    gpu_abort(__LINE__);
  T sum=T(0); int k=0;
  for (int i=0;i<4;++i) if (i!=fixed_index) sum+=T(reference_weights[i])*free_controls[k++];
  return -sum/T(reference_weights[fixed_index]);
}

// IMPLEMENTED: templated exact orthogonal head, Brown/zoom projection, finite
// line/circle/ellipse primitives, local cubic basis and reference-value gauge,
// explicit signed-row timing/equality.  Compilable with double without Ceres.
// NOT IMPLEMENTED: C++ implicit root isolation/certificates/Jacobian wrapper,
// full spline allocation, board/logo object graph, p!=2 superellipse corners,
// optimizer/manifolds, visibility, OIS/exposure/output alternatives, data fit.
// Domain errors/statuses are explicit.  No per-frame camera/image warp exists.


template<class T> HD inline double event_scalar(const T& x) {
  if constexpr(std::is_arithmetic<T>::value) return double(x);
  else return event_scalar(x.a);
}
template<class T> HD inline T event_pow_nonnegative(const T& x,const T& p) {
  using std::pow;
  // x==0 is an analytic endpoint; d(0^p)/dp=0 for all admitted p>1.
  return event_scalar(x)<=0 ? T(0) : pow(x,p);
}
template<class T> HD inline Vec3<T> event_normalized(const Vec3<T>& v) {
  using std::sqrt; return scale(v,T(1)/sqrt(dot(v,v)));
}
template<class T> HD inline Mat3<T> event_exp_rotation(const Vec3<T>& w) {
  using std::sqrt;using std::sin;using std::cos;
  const T q=dot(w,w); T A,B;
  if(event_scalar(q)<1e-12){ A=T(1)-q/T(6)+q*q/T(120); B=T(.5)-q/T(24)+q*q/T(720); }
  else {const T t=sqrt(q);A=sin(t)/t;B=(T(1)-cos(t))/q;}
  const T x=w[0],y=w[1],z=w[2];
  return {T(1)-B*(y*y+z*z),B*x*y-A*z,B*x*z+A*y,
          B*x*y+A*z,T(1)-B*(x*x+z*z),B*y*z-A*x,
          B*x*z-A*y,B*y*z+A*x,T(1)-B*(x*x+y*y)};
}
template<class T> struct EventALProjection {
  Vec2<T> uv{T(std::numeric_limits<double>::quiet_NaN()),T(std::numeric_limits<double>::quiet_NaN())};
  T depth=T(std::numeric_limits<double>::quiet_NaN());
  T ray_radius_sq=T(std::numeric_limits<double>::quiet_NaN());
  T brown_lower=T(std::numeric_limits<double>::quiet_NaN());
  DomainStatus physical_status=DomainStatus::kNonFinite;
  bool arithmetic_valid=false;
  HD bool physical_valid() const {return arithmetic_valid&&physical_status==DomainStatus::kValid;}
};



constexpr int kEventGlobalSize=32;
struct Path {int kind,radius_index,boundary_segment;double start_angle,end_angle,fixed_height_fraction;Span<Vec2<double>> points;};
struct Layout {int global_slot=0,geometry_slot=-1,advertisement_slot=-1,flatpath=0;double frame_time=0;Span<int>control_ids,control_slots;};
struct Model {
Mat3<double> R0_seed;Vec3<double> n_seed,b_seed,n_tangent1,n_tangent2;
double f_ref,radius_max,dt,first_control_time;int n_controls,control_block_begin,dependent_control;
std::array<int,4>reference_indices;std::array<double,4>reference_weights;const Path* paths;
template<class T> HD T control(int i,int channel,T const*const* pars,const Layout& l) const {
    auto raw=[&](int j)->T {for(std::size_t k=0;k<l.control_ids.size();++k)
      if(l.control_ids[k]==j)return pars[l.control_slots[k]][channel];
      gpu_abort(__LINE__);};
    if(channel<2 && i==dependent_control){T sum=T(0);double pivot=0;
      for(int k=0;k<4;++k){if(reference_indices[k]==i)pivot=reference_weights[k];
        else if(reference_weights[k]!=0)sum+=T(reference_weights[k])*raw(reference_indices[k]);}
      return -sum/T(pivot);}
    return raw(i);
  }
template<class T> HD std::array<T,3> trajectory(const T& time,T const*const* pars,const Layout& l) const {
    const T x=(time-T(first_control_time))/T(dt);
    const int cell=int(std::floor(event_scalar(x)));
    if(cell<1||cell+2>=n_controls)gpu_abort(__LINE__);
    const T u=x-T(cell); const auto w=cubic_weights(u);std::array<T,3> out{};
    for(int k=0;k<4;++k)for(int c=0;c<3;++c)out[c]+=w[k]*control(cell-1+k,c,pars,l);
    return out;
  }
template<class T> HD FixedHead<T> head(const T* g) const {
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
template<class T> HD ZoomOptics<T> optics(const T* g) const {
    ZoomOptics<T> o;o.f_ref=f_ref;o.radius_max=T(radius_max);
    o.cx=g[20];o.cy=g[21];o.eta=g[22];o.cx_linear=g[23];o.cy_linear=g[24];
    o.k1=g[25];o.k1_linear=g[26];o.k2=g[27];o.k2_linear=g[28];o.p1=g[29];o.p2=g[30];return o;
  }
template<class T> HD static Vec2<T> quarter(const T& angle,const T* g) {
    using std::sin;using std::cos;using std::pow;
    const T c=cos(angle),s=sin(angle),d=pow(event_pow_nonnegative(c,g[3])+event_pow_nonnegative(s,g[3]),-T(1)/g[3]);
    return {g[1]*c*d,g[2]*s*d};
  }
template<class T> HD static T quarter_speed(const T& a,const T* g) {
    using std::sin;using std::cos;using std::pow;using std::sqrt;
    const T c=cos(a),s=sin(a),cp=event_pow_nonnegative(c,g[3]),sp=event_pow_nonnegative(s,g[3]);
    const T d=pow(cp+sp,-T(1)/g[3]);
    const T v=(event_pow_nonnegative(c,g[3]-T(1))*s-event_pow_nonnegative(s,g[3]-T(1))*c)/(cp+sp);
    const T x=g[1]*d*(-s+c*v),y=g[2]*d*(c+s*v);return sqrt(x*x+y*y);
  }
template<class T> HD static T quarter_arclength(const T& angle,const T* g) {
    // 32-point composite Gauss quadrature, endpoints never evaluated.
    static constexpr double x[8]={.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499};
    static constexpr double w[8]={.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541};
    T total=T(0);for(int part=0;part<2;++part){const T mid=angle*T((part+.5)/2),half=angle/T(4);
      for(int i=0;i<8;++i)total+=half*T(w[i])*(quarter_speed(mid-half*T(x[i]),g)+quarter_speed(mid+half*T(x[i]),g));}return total;
  }
template<class T> HD static Vec3<T> board_point(T s,const T& h,const T* g) {
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
    if(std::abs(quarter_arclength(a,gd)-event_scalar(target))>1e-11)gpu_abort(__LINE__);
    const T theta=T(a)-(quarter_arclength(T(a),g)-target)/quarter_speed(T(a),g);
    auto q=quarter(theta,g);
    // arc0 runs bottom-right from bottom to right; signs reverse polar chart.
    if(k==0)return {hx-rx+q[0],-hy+ry-q[1],h};
    if(k==1)return {hx-rx+q[0],hy-ry+q[1],h};
    if(k==2)return {-hx+rx-q[0],hy-ry+q[1],h};
    return {-hx+rx-q[0],-hy+ry-q[1],h};
  }
template<class T> HD Vec3<T> world_point(const Path& p,const T& station,T const*const* pars,const Layout& l) const {
    if(event_scalar(station)<0||event_scalar(station)>1)gpu_abort(__LINE__);
    const T* g=pars[l.global_slot];const T* q=l.geometry_slot<0?nullptr:pars[l.geometry_slot];
    using std::sin;using std::cos;
    if(p.kind==0)return {q[0]*(T(1)-station)+q[2]*station,q[1]*(T(1)-station)+q[3]*station,T(0)};
    if(p.kind==1){const T a=T(p.start_angle)+station*T(p.end_angle-p.start_angle);return {q[0]+q[p.radius_index]*cos(a),q[1]+q[p.radius_index]*sin(a),T(0)};}
    if(p.kind==3){const T arc=quarter_arclength(T(1.5707963267948966),g),ll=T(1)-T(2)*g[1],sl=g[0]-T(2)*g[2];
      const T perimeter=T(2)*(ll+sl)+T(4)*arc;T position=station*perimeter;
      if(p.boundary_segment>=0){const T lengths[8]={ll,arc,sl,arc,ll,arc,sl,arc};position=station*lengths[p.boundary_segment];for(int j=0;j<p.boundary_segment;++j)position+=lengths[j];}
      return board_point(position,g[4]*T(p.fixed_height_fraction),g);}
    if(p.points.empty())gpu_abort(__LINE__);
    Vec2<T> uv;
    if(p.points.size()==1)uv={T(p.points[0][0]),T(p.points[0][1])};
    else {const T x=station*T(p.points.size()-1);const int i=std::min(int(p.points.size())-2,int(std::floor(event_scalar(x))));
      const T u=x-T(i);uv={T(p.points[i][0])*(T(1)-u)+T(p.points[i+1][0])*u,T(p.points[i][1])*(T(1)-u)+T(p.points[i+1][1])*u};}
    if(q){const T x=uv[0],y=uv[1];uv={q[0]*x+q[1]*y+q[2],q[3]*x+q[4]*y+q[5]};}
    const T* ad=pars[l.advertisement_slot];
    return board_point(ad[0]+ad[1]*uv[0],ad[2]+(ad[3]-ad[2])*uv[1],g);
  }
template<class T> HD EventALProjection<T> predict_for_al(const Layout& l,const T& station,const T& row,T const*const* pars) const {
    EventALProjection<T> out;const T* g=pars[l.global_slot];const T time=row_time(T(l.frame_time),g[31],row);
    const auto tr=trajectory(time,pars,l);const auto o=zoom_optics(optics(g),tr[2]);const auto pose=camera_pose(head(g),tr[0],tr[1],o.z);
    if(!pose.valid()){out.physical_status=pose.status;return out;}
    if(!valid_rotation(pose.R)){out.physical_status=DomainStatus::kInvalidRotation;return out;}
    const auto world=world_point(paths[l.flatpath],station,pars,l);
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
} // namespace gpu_exact
