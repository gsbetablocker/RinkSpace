#include <stdint.h>
#include <math.h>

/* Source UV and Jacobians stay OpenCV results. This fuses only the same
   Brown4 ray/domain/endpoint-agreement checks. Ambiguous cases return 2 for
   the unchanged original Python backend. No forward ray is discarded. */
__declspec(dllexport) int central_brown4_guard(const double *c, const double *camera_xyz,
    const double *uv, const double *depth, int64_t n,
    unsigned char *status) {
    const double k1=c[11],k2=c[12],p1=c[13],p2=c[14];
    const double fx=exp(c[6]),fy=exp(c[7]);
    int64_t i;
    if (!isfinite(fx)||!isfinite(fy)) return 2;
    for (i=0;i<n;++i) {
        double x,y,minimum=INFINITY;
        int station;
        status[i]=0;
        if (!(isfinite(uv[2*i])&&isfinite(uv[2*i+1])&&isfinite(depth[i])&&depth[i]>.05)) continue;
        if (fabs(camera_xyz[3*i+2]-.05)<1e-9) {status[i]=2;continue;}
        if (!(camera_xyz[3*i+2]>.05)) continue;
        /* The preceding matrix product is still the original NumPy BLAS
           arithmetic. These strict IEEE divisions match its normalization. */
        x=camera_xyz[3*i]/camera_xyz[3*i+2];
        y=camera_xyz[3*i+1]/camera_xyz[3*i+2];
        if (!(isfinite(x)&&isfinite(y))||fabs(x)>1000.||fabs(y)>1000.) {status[i]=2;continue;}
        for (station=0;station<9;++station) {
            const double alpha=(double)station/8.;
            const double xs=x*alpha,ys=y*alpha;
            const double r2=xs*xs+ys*ys;
            const double radial=1.+k1*r2+k2*r2*r2;
            const double slope=k1+2.*k2*r2;
            const double j11=radial+2.*xs*xs*slope+2.*p1*ys+6.*p2*xs;
            const double j12=2.*xs*ys*slope+2.*p1*xs+2.*p2*ys;
            const double j21=2.*xs*ys*slope+2.*p1*xs+2.*p2*ys;
            const double j22=radial+2.*ys*ys*slope+6.*p1*ys+2.*p2*xs;
            const double gap=j11-j22,off=j12+j21;
            const double eigen=.5*(j11+j22-sqrt(gap*gap+off*off));
            if (!isfinite(eigen)) {minimum=NAN;break;}
            if (eigen<minimum) minimum=eigen;
        }
        if (!isfinite(minimum)||fabs(minimum-1e-8)<=1e-7) {status[i]=2;continue;}
        if (!(minimum>1e-8)) continue;
        {
            const double r2=x*x+y*y;
            const double radial=1.+k1*r2+k2*r2*r2+0.;
            const double mx=x*radial+2.*p1*x*y+p2*(r2+2.*x*x)+0.+0.;
            const double my=y*radial+p1*(r2+2.*y*y)+2.*p2*x*y+0.+0.;
            const double px=fx*mx+c[10]*my+c[8],py=fy*my+c[9];
            const double dx=px-uv[2*i],dy=py-uv[2*i+1];
            const double disagreement=sqrt(dx*dx+dy*dy);
            if (!isfinite(px)||!isfinite(py)) continue;
            if (!isfinite(disagreement)||fabs(disagreement-1e-5)<=1e-9) {status[i]=2;continue;}
            status[i]=(unsigned char)(disagreement<=1e-5);
        }
    }
    return 0;
}
