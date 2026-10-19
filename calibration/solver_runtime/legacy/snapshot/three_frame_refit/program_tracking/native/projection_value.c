#include <stdint.h>
#include <math.h>

/* OpenCV's Brown4 value order with the original OpenCV R, t and K inputs.
   Jacobians are always evaluated by the original OpenCV path. */
__declspec(dllexport) int project_brown4_value(const double*c,const double*r,
    const double*t,const double*k,const double*world,int64_t n,double*uv) {
    int64_t i;
    const double fx=k[0],fy=k[4],cx=k[2],cy=k[5];
    for(i=0;i<n;++i) {
        const double X=world[3*i],Y=world[3*i+1],Z=world[3*i+2];
        double x=r[0]*X+r[1]*Y+r[2]*Z+t[0];
        double y=r[3]*X+r[4]*Y+r[5]*Z+t[1];
        double z=r[6]*X+r[7]*Y+r[8]*Z+t[2];
        const double iz=z!=0.?1./z:1.;
        double r2,r4,r6,a1,a2,a3,cdist,xd,yd,px,py;
        x*=iz;y*=iz;
        r2=x*x+y*y;r4=r2*r2;r6=r4*r2;
        a1=2.*x*y;a2=r2+2.*x*x;a3=r2+2.*y*y;
        cdist=1.+c[11]*r2+c[12]*r4+0.*r6;
        xd=x*cdist+ c[13]*a1+c[14]*a2+0.*r2+0.*r4;
        yd=y*cdist+ c[13]*a3+c[14]*a1+0.*r2+0.*r4;
        px=xd*fx+cx;py=yd*fy+cy;
        px+=c[10]*((py-c[9])/fy);
        if(!isfinite(px)||!isfinite(py)||fabs(px)>1e5||fabs(py)>1e5)return 2;
        uv[2*i]=px;uv[2*i+1]=py;
    }
    return 0;
}
