#include <stdint.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* Exact original scalar ordering, first argmin, clamped finite endpoints.
 * No fast-math/FMA. Every observation and valid segment remains present.
 * OpenMP owns only disjoint output rows; all inputs are immutable doubles. */
EXPORT int normal_scanline_human_strict(
    const double *uv, const double *normal, int64_t n,
    const double *a, const double *d, const double *l2, int64_t m,
    const double *threshold, int threads,
    uint8_t *covered, int64_t *ray_index, double *ray_station, double *ray_signed,
    int64_t *foot_index, double *foot_station, double *foot_xy)
{
    if (!uv || !normal || !a || !d || !l2 || !threshold || !covered ||
        !ray_index || !ray_station || !ray_signed || !foot_index || !foot_station || !foot_xy || n<0 || m<1) return 1;
    if(n>INT_MAX)return 1;
    int error=0, ni=(int)n, i;
    #pragma omp parallel for schedule(static) num_threads(threads) if(n*m>=262144 && threads>1) reduction(|:error)
    for (i=0;i<ni;++i) {
        double ox=uv[2*i], oy=uv[2*i+1], nx=normal[2*i], ny=normal[2*i+1];
        double best_ray=INFINITY, best_foot=INFINITY;
        int64_t ri=0, fi=0; double rs=NAN, rv=NAN, fs=0., fx=0., fy=0.; int hit=0;
        for (int64_t j=0;j<m;++j) {
            double ax=a[2*j], ay=a[2*j+1], dx=d[2*j], dy=d[2*j+1];
            double x=ox-ax, y=oy-ay;
            double determinant=dx*ny-dy*nx;
            if (fabs(determinant)>threshold[j]) {
                double station=(x*ny-y*nx)/determinant;
                double signed_value=(x*dy-y*dx)/determinant;
                if (!isfinite(station) || !isfinite(signed_value)) error|=2;
                if (station>=0. && station<=1.) {
                    double absolute=fabs(signed_value);
                    if (absolute<best_ray) {best_ray=absolute;ri=j;rs=station;rv=signed_value;}
                    hit=1;
                }
            }
            double raw=x*dx;raw+=y*dy;raw/=l2[j];
            if (!isfinite(raw)) error|=2;
            double t=raw;if(t<0.)t=0.;else if(t>1.)t=1.;
            double px=ax+t*dx, py=ay+t*dy;
            double vx=px-ox, vy=py-oy;
            double distance=vx*vx;distance+=vy*vy;
            if (!isfinite(distance)) error|=2;
            if (distance<best_foot) {best_foot=distance;fi=j;fs=t;fx=px;fy=py;}
        }
        covered[i]=(uint8_t)hit;ray_index[i]=ri;ray_station[i]=rs;ray_signed[i]=rv;
        foot_index[i]=fi;foot_station[i]=fs;foot_xy[2*i]=fx;foot_xy[2*i+1]=fy;
    }
    return error;
}

/* Exact minimum L-infinity outside the unchanged source support rectangle.
 * For a segment, the expanded normal/tangent slabs must share one t in [0,1].
 * Endpoint inequalities and the two cross-slab inequalities give its minimum
 * expansion directly, without iterative root searches or support padding. */
EXPORT int finite_source_rectangle_strict(
    const double *uv, const double *normal, const double *bounds, const double *tb, int64_t n,
    const double *a, const double *d, const double *l2, int64_t m,
    const int64_t *forced, int threads, int64_t *index, double *station, double *foot)
{
    if(!uv||!normal||!bounds||!tb||!a||!d||!l2||!forced||!index||!station||!foot||n<0||n>INT_MAX||m<1)return 1;
    int error=0, ni=(int)n, i;
    #pragma omp parallel for schedule(static) num_threads(threads) if(n*m>=262144 && threads>1) reduction(|:error)
    for(i=0;i<ni;++i){
        double best=INFINITY, bt=0.; int64_t bj=0;
        int64_t first=forced[i]<0?0:forced[i],last=forced[i]<0?m:first+1;
        if(first<0||last>m){error|=1;continue;}
        double nx=normal[2*i],ny=normal[2*i+1],ox=uv[2*i],oy=uv[2*i+1];
        double ln=bounds[2*i],un=bounds[2*i+1],lt=tb[2*i],ut=tb[2*i+1];
        for(int64_t j=first;j<last;++j){
            double x=a[2*j]-ox,y=a[2*j+1]-oy,dx=d[2*j],dy=d[2*j+1];
            double an=x*nx+y*ny,at=-x*ny+y*nx,dn=dx*nx+dy*ny,dt=-dx*ny+dy*nx;
            double cn=fabs(dn),ct=fabs(dt),u=0.,v;
            v=ln-fmax(an,an+dn);if(v>u)u=v;
            v=fmin(an,an+dn)-un;if(v>u)u=v;
            v=lt-fmax(at,at+dt);if(v>u)u=v;
            v=fmin(at,at+dt)-ut;if(v>u)u=v;
            double lnv=dn>=0.?ln-an:an-un,unv=dn>=0.?un-an:an-ln;
            double ltv=dt>=0.?lt-at:at-ut,utv=dt>=0.?ut-at:at-lt;
            v=(lnv*ct-utv*cn)/(cn+ct);if(v>u)u=v;
            v=(ltv*cn-unv*ct)/(cn+ct);if(v>u)u=v;
            double lo=0.,hi=1.,loi,hii;
            if(dn!=0.){
                loi=(dn>0.?ln-an-u:un-an+u)/dn;
                hii=(dn>0.?un-an+u:ln-an-u)/dn;
                if(loi>lo)lo=loi;if(hii<hi)hi=hii;
            }
            if(dt!=0.){
                loi=(dt>0.?lt-at-u:ut-at+u)/dt;
                hii=(dt>0.?ut-at+u:lt-at-u)/dt;
                if(loi>lo)lo=loi;if(hii<hi)hi=hii;
            }
            double t=-(an*dn+at*dt)/l2[j];
            if(lo>hi)t=.5*(lo+hi);else{if(t<lo)t=lo;if(t>hi)t=hi;}
            if(t<0.)t=0.;if(t>1.)t=1.;
            if(!isfinite(u)||!isfinite(t)){error|=2;continue;}
            if(u<best){best=u;bj=j;bt=t;}
        }
        index[i]=bj;station[i]=bt;foot[2*i]=a[2*bj]+bt*d[2*bj];foot[2*i+1]=a[2*bj+1]+bt*d[2*bj+1];
    }
    return error;
}
