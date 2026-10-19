#include "source_edge_native.hpp"
#include "brown15_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {
using rinkspace_native::Brown15Projection;

struct Point {
    double u, v, z, j[30];
};
struct Segment {
    double ax, ay, dx, dy, length2;
    int64_t first;
    bool valid;
};

// One workspace per native worker. Each residual evaluation overwrites every
// used point/segment/mask entry; the central-status arrays are reset below.
// Keeping capacity across the nine evaluations of one Jacobian avoids repeated
// shared-heap allocation in GTSAM's TBB factor loop.
struct Scratch {
    std::vector<Point> points;
    std::vector<Segment> segments;
    std::vector<unsigned char> crossing, available;
    std::vector<int8_t> endpoint_central, midpoint_central;
    std::vector<double> plus, minus;
};
thread_local Scratch scratch;

bool descriptor_valid(const EdgeFactorDesc& f) noexcept {
    if (!f.world_xyz || !f.observed_uv || !f.quality_scales ||
        f.world_points<2 || f.observations<1 ||
        f.world_points>1000000 || f.observations>1000000 ||
        !std::isfinite(f.sigma) || f.sigma<=0.0 ||
        (f.dynamic_candidates!=0 && f.dynamic_candidates!=1) ||
        (f.fixed_mount!=0 && f.fixed_mount!=1) ||
        (f.central_domain!=0 && f.central_domain!=1) ||
        f.frozen_count<0 || f.frozen_count>=f.world_points ||
        (f.frozen_count>0 && !f.frozen_segments)) return false;
    for (int64_t i=0;i<f.world_points*3;++i)
        if (!std::isfinite(f.world_xyz[i])) return false;
    for (int64_t i=0;i<f.observations;++i) {
        if (!std::isfinite(f.observed_uv[2*i]) ||
            !std::isfinite(f.observed_uv[2*i+1]) ||
            !std::isfinite(f.quality_scales[i]) ||
            f.quality_scales[i]<1.0) return false;
    }
    int64_t previous=-1;
    for (int64_t i=0;i<f.frozen_count;++i) {
        const int64_t index=f.frozen_segments[i];
        if (index<=previous || index>=f.world_points-1) return false;
        previous=index;
    }
    if (f.fixed_mount)
        for (double value:f.fixed_center_xyz)
            if (!std::isfinite(value)) return false;
    return true;
}

void robust_point(const double* raw, const double quality, const double delta,
                  double* out, const double* raw_j, double* out_j,
                  bool with_j) noexcept {
    const double wx=raw[0]/quality, wy=raw[1]/quality;
    const double norm=std::sqrt(wx*wx+wy*wy);
    double scale=1.0, derivative=0.0;
    if (norm>delta) {
        const double rho_root=std::sqrt(2.0*delta*norm-delta*delta);
        scale=rho_root/norm;
        derivative=delta/(rho_root*norm*norm)-rho_root/(norm*norm*norm);
    }
    out[0]=wx*scale;
    out[1]=wy*scale;
    if (with_j) {
        for (int col=0;col<11;++col) {
            const double jx=raw_j[col]/quality;
            const double jy=raw_j[15+col]/quality;
            const double radial=wx*jx+wy*jy;
            out_j[col]=scale*jx+derivative*wx*radial;
            out_j[15+col]=scale*jy+derivative*wy*radial;
        }
        for (int col=11;col<15;++col) {
            out_j[col]=0.0;
            out_j[15+col]=0.0;
        }
    }
}

// Returns 0 success, 2 for a numeric state that needs the original Python
// implementation, and 3 for allocation failure (handled by outer catch).
int residual_once(const EdgeFactorDesc& f, const double input_q[15],
                  double* residual, double* jacobian, bool with_j) {
    double q[15];
    std::memcpy(q,input_q,sizeof(q));
    if (f.fixed_mount)
        for (int i=0;i<3;++i) q[i]=f.fixed_center_xyz[i]/100.0;
    // OpenCV Rodrigues' analytic Jacobian is numerically irregular at tiny
    // nonzero rotations. Keep the production derivative via explicit fallback.
    const double rotation_norm2=q[3]*q[3]+q[4]*q[4]+q[5]*q[5];
    if (with_j && rotation_norm2>0.0 && rotation_norm2<1e-8) return 2;
    Brown15Projection camera;
    if (!camera.prepare(q)) return 2;
    if (!camera.numerically_supported_camera(q)) return 2;
    const int64_t n=f.world_points, m=f.observations;
    auto& points = scratch.points;
    points.resize(static_cast<size_t>(n));
    for (int64_t i=0;i<n;++i) {
        Point& point=points[static_cast<size_t>(i)];
        double uv[2];
        if (!camera.project(f.world_xyz+3*i,uv,&point.z,
                            with_j?point.j:nullptr)) return 2;
        point.u=uv[0]; point.v=uv[1];
    }
    auto& segments = scratch.segments;
    segments.resize(static_cast<size_t>(n-1));
    auto& crossing = scratch.crossing;
    crossing.resize(static_cast<size_t>(n-1));
    auto& available = scratch.available;
    available.resize(static_cast<size_t>(n-1));
    auto& endpoint_central = scratch.endpoint_central;
    endpoint_central.assign(static_cast<size_t>(n),-2);
    auto& midpoint_central = scratch.midpoint_central;
    midpoint_central.assign(static_cast<size_t>(n-1),-2);
    double lowx=std::numeric_limits<double>::infinity();
    double lowy=lowx, highx=-lowx, highy=-lowx;
    for (int64_t i=0;i<m;++i) {
        const double x=f.observed_uv[2*i], y=f.observed_uv[2*i+1];
        lowx=std::min(lowx,x); lowy=std::min(lowy,y);
        highx=std::max(highx,x); highy=std::max(highy,y);
    }
    lowx-=350.0; lowy-=350.0; highx+=350.0; highy+=350.0;
    int64_t crossing_count=0;
    for (int64_t i=0;i<n-1;++i) {
        const Point& a=points[static_cast<size_t>(i)];
        const Point& b=points[static_cast<size_t>(i+1)];
        Segment& seg=segments[static_cast<size_t>(i)];
        seg.first=i; seg.ax=a.u; seg.ay=a.v;
        seg.dx=b.u-a.u; seg.dy=b.v-a.v;
        seg.length2=seg.dx*seg.dx+seg.dy*seg.dy;
        const bool endpoint_base=(a.z>0.0 && b.z>0.0);
        available[static_cast<size_t>(i)]=static_cast<unsigned char>(endpoint_base);
        const bool cross=(std::min(a.u,b.u)<=highx && std::max(a.u,b.u)>=lowx &&
                          std::min(a.v,b.v)<=highy && std::max(a.v,b.v)>=lowy);
        crossing[static_cast<size_t>(i)]=static_cast<unsigned char>(endpoint_base && cross);
        crossing_count+=endpoint_base && cross;
        seg.valid=false;
    }
    auto central_segment=[&](int64_t index) -> int {
        if (!f.central_domain) return 1;
        const int64_t ids[2]={index,index+1};
        bool ambiguous=false, invalid=false;
        for (int local=0;local<2;++local) {
            int8_t& cached=endpoint_central[static_cast<size_t>(ids[local])];
            if (cached==-2) {
                const Point& point=points[static_cast<size_t>(ids[local])];
                const double uv[2]={point.u,point.v};
                cached=static_cast<int8_t>(camera.central_status_from_projection(
                    f.world_xyz+3*ids[local],uv,point.z));
            }
            ambiguous|=(cached==2);
            invalid|=(cached==0);
        }
        int8_t& middle=midpoint_central[static_cast<size_t>(index)];
        if (middle==-2) {
            const double midpoint[3]={
                .5*(f.world_xyz[3*index]+f.world_xyz[3*index+3]),
                .5*(f.world_xyz[3*index+1]+f.world_xyz[3*index+4]),
                .5*(f.world_xyz[3*index+2]+f.world_xyz[3*index+5])};
            middle=static_cast<int8_t>(camera.central_status(midpoint));
        }
        ambiguous|=(middle==2);
        invalid|=(middle==0);
        return invalid ? 0 : (ambiguous ? 2 : 1);
    };
    int64_t selected_count=0;
    if (f.dynamic_candidates) {
        // Production first tries the generous observed-pixel box, then falls
        // back to every front-facing world segment if central filtering leaves
        // that first set empty. This cannot be reduced to a single mask.
        if (crossing_count) {
            for (int64_t i=0;i<n-1;++i) if (crossing[static_cast<size_t>(i)]) {
                const int central=central_segment(i);
                if (central==2) return 2;
                if (central==1) {
                    segments[static_cast<size_t>(i)].valid=true;
                    ++selected_count;
                }
            }
        }
        if (!selected_count) {
            for (int64_t i=0;i<n-1;++i) if (available[static_cast<size_t>(i)]) {
                const int central=central_segment(i);
                if (central==2) return 2;
                if (central==1) segments[static_cast<size_t>(i)].valid=true;
            }
        }
    } else {
        for (int64_t i=0;i<f.frozen_count;++i) {
            Segment& seg=segments[static_cast<size_t>(f.frozen_segments[i])];
            if (!available[static_cast<size_t>(seg.first)]) continue;
            const int central=central_segment(seg.first);
            if (central==2) return 2;
            seg.valid=(central==1);
        }
    }
    for (Segment& seg:segments) seg.valid=seg.valid && seg.length2>1e-4;
    bool any_valid=false;
    for (const Segment& seg:segments) if (seg.valid) {any_valid=true;break;}
    if (!any_valid) {
        if (f.central_domain) return 4;
        const double raw[2]={1000.0,0.0};
        for (int64_t i=0;i<m;++i) {
            robust_point(raw,f.quality_scales[i],2.0*f.sigma,residual+2*i,
                         nullptr,jacobian?jacobian+30*i:nullptr,false);
            if (with_j) std::memset(jacobian+30*i,0,30*sizeof(double));
        }
        return 0;
    }
    for (int64_t i=0;i<m;++i) {
        const double ox=f.observed_uv[2*i], oy=f.observed_uv[2*i+1];
        double best=std::numeric_limits<double>::infinity();
        double best_t=0.0, best_raw_t=0.0, nearest[2]={0.0,0.0};
        int64_t winner=-1;
        for (int64_t k=0;k<n-1;++k) {
            const Segment& seg=segments[static_cast<size_t>(k)];
            if (!seg.valid) continue;
            const double ex=ox-seg.ax, ey=oy-seg.ay;
            double raw_t=ex*seg.dx;
            raw_t+=ey*seg.dy;
            raw_t/=std::max(seg.length2,1e-9);
            const double t=std::max(0.0,std::min(1.0,raw_t));
            const double nx=seg.ax+t*seg.dx, ny=seg.ay+t*seg.dy;
            const double rx=nx-ox, ry=ny-oy;
            double sq=rx*rx;
            sq+=ry*ry;
            if (!std::isfinite(sq)) return 2;
            if (sq<best) {
                best=sq; winner=k; best_t=t; best_raw_t=raw_t;
                nearest[0]=nx; nearest[1]=ny;
            }
        }
        if (winner<0) return 2;
        const double raw[2]={nearest[0]-ox,nearest[1]-oy};
        double raw_j[30]={};
        if (with_j) {
            const Segment& seg=segments[static_cast<size_t>(winner)];
            const Point& a=points[static_cast<size_t>(winner)];
            const Point& b=points[static_cast<size_t>(winner+1)];
            const double offx=ox-seg.ax, offy=oy-seg.ay;
            for (int col=0;col<11;++col) {
                const double j0x=a.j[col], j0y=a.j[15+col];
                const double jdx=b.j[col]-j0x, jdy=b.j[15+col]-j0y;
                const double numerator_j=-(j0x*seg.dx+j0y*seg.dy)+
                    offx*jdx+offy*jdy;
                const double length2_j=2.0*(seg.dx*jdx+seg.dy*jdy);
                double tj=(numerator_j-best_raw_t*length2_j)/seg.length2;
                if (best_raw_t<=0.0 || best_raw_t>=1.0) tj=0.0;
                raw_j[col]=j0x+best_t*jdx+seg.dx*tj;
                raw_j[15+col]=j0y+best_t*jdy+seg.dy*tj;
                if (f.fixed_mount && col<3) {
                    raw_j[col]=0.0; raw_j[15+col]=0.0;
                }
            }
        }
        robust_point(raw,f.quality_scales[i],2.0*f.sigma,residual+2*i,
                     raw_j,jacobian?jacobian+30*i:nullptr,with_j);
        if (!std::isfinite(residual[2*i]) || !std::isfinite(residual[2*i+1])) return 2;
    }
    return 0;
}
}  // namespace

extern "C" RS_EDGE_API int rs_edge_eval(
    const EdgeFactorDesc* factor, const double q15[15],
    double* residual_2N, double* jacobian_2Nx15, int mode) noexcept {
    if (!factor || !q15 || !residual_2N || (mode!=0 && mode!=1) ||
        (mode==0 && !jacobian_2Nx15) || !descriptor_valid(*factor)) return 1;
    try {
        const int64_t rows=2*factor->observations;
        if (mode==1) return residual_once(*factor,q15,residual_2N,nullptr,false);
        int status=residual_once(*factor,q15,residual_2N,jacobian_2Nx15,true);
        if (status) return status;
        auto& plus = scratch.plus;
        auto& minus = scratch.minus;
        plus.resize(static_cast<size_t>(rows));
        minus.resize(static_cast<size_t>(rows));
        for (int axis=11;axis<15;++axis) {
            double qplus[15],qminus[15];
            std::memcpy(qplus,q15,sizeof(qplus));
            std::memcpy(qminus,q15,sizeof(qminus));
            const double h=1e-6*std::max(1.0,std::abs(q15[axis]));
            qplus[axis]+=h; qminus[axis]-=h;
            status=residual_once(*factor,qplus,plus.data(),nullptr,false);
            if (status) return status;
            status=residual_once(*factor,qminus,minus.data(),nullptr,false);
            if (status) return status;
            for (int64_t row=0;row<rows;++row)
                jacobian_2Nx15[15*row+axis]=(plus[static_cast<size_t>(row)]-
                    minus[static_cast<size_t>(row)])/(2.0*h);
        }
        return 0;
    } catch (const std::bad_alloc&) {return 3;}
      catch (...) {return 3;}
}

extern "C" RS_EDGE_API int rs_edge_eval_batch(
    const EdgeFactorDesc* factors, const double* q15_list,
    double* const* residuals, double* const* jacobians,
    int64_t count, int threads, int mode, int* statuses) noexcept {
    if (!factors || !q15_list || !residuals || !statuses || count<0 ||
        count>1000000 || (threads!=1 && threads!=2) ||
        (mode!=0 && mode!=1) || (mode==0 && !jacobians)) return 1;
    for (int64_t i=0;i<count;++i) statuses[i]=3;
    auto worker=[&](int lane) noexcept {
        for (int64_t i=lane;i<count;i+=threads)
            statuses[i]=rs_edge_eval(factors+i,q15_list+15*i,residuals[i],
                                     mode==0?jacobians[i]:nullptr,mode);
    };
    try {
        if (threads==1 || count<2) worker(0);
        else {
            std::thread other(worker,1);
            worker(0);
            other.join();
        }
    } catch (...) {return 3;}
    for (int64_t i=0;i<count;++i) if (statuses[i]) return statuses[i];
    return 0;
}
