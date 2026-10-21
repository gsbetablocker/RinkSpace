#ifndef RINKFIT_PHYSICAL_HPP
#define RINKFIT_PHYSICAL_HPP

// v1.1 equation core.  Header-only C++17; std::array; double or Ceres Jet via
// argument-dependent sin/cos/sqrt/exp/isfinite.  No Ceres dependency is needed
// to compile the double reference.  R is row-major and maps world to camera.
// Invalid projections contain NaN pixels and an explicit status.  Factors
// must check valid() and reject arithmetic-domain failures, not make residuals
// zero.  A legal but nonzero support/equality residual belongs to external AL.

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rinkfit {

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

template <class T> inline bool finite_scalar(const T& x) {
  using std::isfinite;
  return isfinite(x);
}
template <class T, std::size_t N>
inline bool finite_array(const std::array<T, N>& x) {
  for (const auto& v : x) if (!finite_scalar(v)) return false;
  return true;
}
template <class T> inline T abs_value(const T& x) {
  using std::abs;
  return abs(x);
}
template <class T> inline Vec3<T> add(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
template <class T> inline Vec3<T> sub(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
template <class T> inline Vec3<T> scale(const Vec3<T>& a, const T& s) {
  return {s*a[0], s*a[1], s*a[2]};
}
template <class T> inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
template <class T> inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
template <class T> inline Mat3<T> identity3() {
  return {T(1),T(0),T(0), T(0),T(1),T(0), T(0),T(0),T(1)};
}
template <class T> inline Mat3<T> transpose(const Mat3<T>& A) {
  return {A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8]};
}
template <class T> inline Mat3<T> matmul(const Mat3<T>& A, const Mat3<T>& B) {
  Mat3<T> C{};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    for (int k=0;k<3;++k) C[3*i+j] += A[3*i+k]*B[3*k+j];
  return C;
}
template <class T> inline Vec3<T> matvec(const Mat3<T>& A, const Vec3<T>& v) {
  return {A[0]*v[0]+A[1]*v[1]+A[2]*v[2],
          A[3]*v[0]+A[4]*v[1]+A[5]*v[2],
          A[6]*v[0]+A[7]*v[1]+A[8]*v[2]};
}
template <class T> inline T determinant(const Mat3<T>& A) {
  return A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])
       + A[2]*(A[3]*A[7]-A[4]*A[6]);
}
template <class T> inline bool valid_rotation(const Mat3<T>& R) {
  if (!finite_array(R)) return false;
  const auto RtR = matmul(transpose(R), R);
  for (int i=0;i<9;++i)
    if (abs_value(RtR[i]-T(i%4==0 ? 1 : 0)) > T(1e-9)) return false;
  return abs_value(determinant(R)-T(1)) <= T(1e-9);
}

// The axis must already be unit length.  No normalization hides an invalid
// mechanism state; optimizer-side manifolds/charts must enforce that property.
template <class T> inline Mat3<T> rodrigues(const Vec3<T>& axis, const T& angle) {
  if (!finite_array(axis) || !finite_scalar(angle) || abs_value(dot(axis,axis)-T(1)) > T(1e-9))
    throw std::domain_error("Rodrigues requires a finite unit axis");
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
  bool valid() const { return status==DomainStatus::kValid; }
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
inline Pose<T> camera_pose(const FixedHead<T>& head, const T& alpha, const T& beta, const T& z) {
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
  // Source-estimated local wide candidate; exactly zero on the preserved
  // narrow interval. Coefficients are immutable calibration metadata here.
  double wide_log_f_boundary=0;
  std::array<double,4> wide_coefficients{};
};
template <class T> struct OpticalState {
  Intrinsics<T> intr;
  Brown<T> brown;
  T z=T(0);
};
template <class T> inline OpticalState<T> zoom_optics(const ZoomOptics<T>& optics, const T& log_f) {
  if (!std::isfinite(optics.f_ref) || !(optics.f_ref>0))
    throw std::domain_error("fixed f_ref must be finite and positive");
  using std::exp;
  OpticalState<T> out;
  out.z=log_f-T(std::log(optics.f_ref));
  out.intr={exp(log_f+optics.eta/T(2)),exp(log_f-optics.eta/T(2)),
            optics.cx+optics.cx_linear*out.z,optics.cy+optics.cy_linear*out.z};
  const T hinge=log_f<T(optics.wide_log_f_boundary)?T(optics.wide_log_f_boundary)-log_f:T(0);
  const T basis=hinge*hinge;
  out.brown={optics.k1+optics.k1_linear*out.z+T(optics.wide_coefficients[0])*basis,
             optics.k2+optics.k2_linear*out.z+T(optics.wide_coefficients[1])*basis,
             optics.p1+T(optics.wide_coefficients[2])*basis,
             optics.p2+T(optics.wide_coefficients[3])*basis,optics.radius_max};
  return out;
}
template <class T> inline T quadratic_minimum(const T& c0,const T& c1,const T& c2,const T& xmax) {
  T low=c0;
  const T end=c0+c1*xmax+c2*xmax*xmax;
  if (end<low) low=end;
  if (c2>T(0)) {
    const T x=-c1/(T(2)*c2);
    if (x>T(0) && x<xmax) { const T val=c0+c1*x+c2*x*x; if (val<low) low=val; }
  }
  return low;
}
template <class T> inline T brown_no_fold_lower_bound(const Brown<T>& d) {
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
template <class T> inline Vec2<T> brown_distort(const Vec2<T>& xy,const Brown<T>& d) {
  const T x=xy[0],y=xy[1],r2=x*x+y*y,L=T(1)+d.k1*r2+d.k2*r2*r2;
  return {x*L+T(2)*d.p1*x*y+d.p2*(r2+T(2)*x*x),
          y*L+d.p1*(r2+T(2)*y*y)+T(2)*d.p2*x*y};
}
template <class T> struct Projection {
  Vec2<T> uv{T(std::numeric_limits<double>::quiet_NaN()),T(std::numeric_limits<double>::quiet_NaN())};
  T depth=T(std::numeric_limits<double>::quiet_NaN());
  DomainStatus status=DomainStatus::kNonFinite;
  bool valid() const { return status==DomainStatus::kValid; }
};
template <class T> inline Projection<T> project(const Vec3<T>& world,const Pose<T>& pose,
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
template <class T> inline T row_time(const T& frame_time,const T& signed_readout,
                                     const T& predicted_row,int image_height=1080) {
  if (image_height<2 || !finite_scalar(frame_time) || !finite_scalar(signed_readout) || !finite_scalar(predicted_row))
    throw std::domain_error("invalid native row-time arguments");
  return frame_time+signed_readout*(predicted_row/T(image_height-1)-T(0.5));
}
template <class T> inline T explicit_rs_equality(const T& row,const Projection<T>& instant) {
  if (!instant.valid()) throw std::domain_error("RS equality cannot use an invalid instantaneous projection");
  return row-instant.uv[1];
}

template <class T> inline bool valid_station(const T& s,const T& lo,const T& hi) {
  return finite_scalar(s) && finite_scalar(lo) && finite_scalar(hi) && lo<=hi && s>=lo && s<=hi;
}
template <class T> inline Vec3<T> finite_line_point(const Vec3<T>& A,const Vec3<T>& B,const T& s) {
  if (!finite_array(A) || !finite_array(B) || !valid_station(s,T(0),T(1)) || !(dot(sub(B,A),sub(B,A))>T(0)))
    throw std::domain_error("finite line requires distinct endpoints and legal station in [0,1]");
  return add(scale(A,T(1)-s),scale(B,s));
}
template <class T> inline Vec3<T> circle_arc_point(const Vec2<T>& center,const T& radius,
                                                   const T& angle,const T& start,const T& end) {
  constexpr double two_pi=6.283185307179586476925286766559;
  if (!finite_array(center) || !finite_scalar(radius) || !(radius>T(0)) ||
      !valid_station(angle,start,end) || !(end>start) || end-start>T(two_pi+1e-12))
    throw std::domain_error("circle arc requires positive radius and finite legal unwrapped angle");
  using std::sin; using std::cos;
  return {center[0]+radius*cos(angle),center[1]+radius*sin(angle),T(0)};
}
template <class T> inline Vec3<T> ellipse_corner_point(const Vec2<T>& center,const T& rx,const T& ry,
                                                      const T& angle,const T& start,const T& end) {
  if (!finite_array(center) || !finite_scalar(rx) || !finite_scalar(ry) || !(rx>T(0)) || !(ry>T(0)) ||
      !valid_station(angle,start,end) || !(end>start))
    throw std::domain_error("ellipse corner requires positive radii and legal angle");
  using std::sin; using std::cos;
  return {center[0]+rx*cos(angle),center[1]+ry*sin(angle),T(0)};
}

// Local cardinal cubic values; knot allocation/time-domain checking lives in
// the trajectory layer.  Exactly four controls enter each local evaluation.
template <class T> inline std::array<T,4> cubic_weights(const T& u) {
  if (!valid_station(u,T(0),T(1))) throw std::domain_error("cubic cell u outside [0,1]");
  const T u2=u*u,u3=u2*u,v=T(1)-u;
  return {v*v*v/T(6),(T(3)*u3-T(6)*u2+T(4))/T(6),
          (-T(3)*u3+T(3)*u2+T(3)*u+T(1))/T(6),u3/T(6)};
}
template <class T> inline T cubic_value(const std::array<T,4>& controls,const T& u) {
  if (!finite_array(controls)) throw std::domain_error("nonfinite cubic controls");
  const auto w=cubic_weights(u);
  T out=T(0); for (int i=0;i<4;++i) out+=w[i]*controls[i]; return out;
}
// Eliminate the selected reference-support coefficient exactly.  fixed_index
// is the largest reference weight, selected once outside AutoDiff.  The other
// three controls are ordered by ascending reference-support index.
template <class T> inline T gauge_dependent_control(const std::array<double,4>& reference_weights,
                                                    int fixed_index,const std::array<T,3>& free_controls) {
  if (fixed_index<0 || fixed_index>3 || !finite_array(reference_weights) ||
      std::abs(reference_weights[fixed_index])<1e-12 || !finite_array(free_controls))
    throw std::domain_error("invalid exact spline function-value gauge");
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

}  // namespace rinkfit
#endif  // RINKFIT_PHYSICAL_HPP
