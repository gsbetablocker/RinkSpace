#pragma once
#include "geometry.hpp"
#include <memory>

namespace rinkspace {
/* Shared installed material chart, before wrapping onto the board course.
 * p=[unwrapped_s,z,rotation,width,height]; dimensions stay positive.
 * Template chart orientation is explicit in the supplied material vertices. */
inline Vec2 installed_print_chart(const double* p,double x,double y){
    const double ca=std::cos(p[2]),sa=std::sin(p[2]);
    return {p[0]+ca*x*p[3]-sa*y*p[4],p[1]+sa*x*p[3]+ca*y*p[4]};
}
/* A regular square in the Euclidean unfolded (s,z) plane. Material UV axes
 * have different physical scales; correcting by height/width keeps the
 * square exact when the enclosing print dimensions or rotation change.
 * The result contains four corners followed by the first corner. */
inline Status print_square_vertices(const double* p,size_t np,const double* centre,size_t nc,
                                    double side_over_height,double* vertices){
    if(np!=5||nc!=2||!vertices||!finite_array(p,np)||!finite_array(centre,nc)||
       p[3]<=0||p[4]<=0||!std::isfinite(side_over_height)||side_over_height<=0)
        return Status::invalid_argument;
    const double hv=side_over_height/2,hu=hv*p[4]/p[3];
    const double left=centre[0]-hu,right=centre[0]+hu;
    const double top=centre[1]-hv,bottom=centre[1]+hv;
    const double values[]={left,top,right,top,right,bottom,left,bottom,left,top};
    if(!finite_array(values,10))return Status::nonfinite;
    if(!(left<right&&top<bottom))return Status::degenerate;
    for(size_t i=0;i<10;++i)vertices[i]=values[i];
    return Status::ok;
}
inline Status sample_print_chart(const double* p,size_t np,const double* vertices,size_t nv,
                                const double* stations,size_t n,double* sz,int* statuses){
    if(np!=5||!finite_array(p,np)||p[3]<=0||p[4]<=0||nv<2||nv%2||
       !finite_array(vertices,nv)||(n&&(!stations||!sz||!statuses)))return Status::invalid_argument;
    const size_t count=nv/2;
    for(size_t i=0;i<n;++i){
        const double s=stations[i];Vec2 q{};Status status=Status::ok;
        if(!std::isfinite(s))status=Status::invalid_argument;
        else if(s<0||s>1)status=Status::outside_support;
        else {
            const double u=s*(count-1);
            const size_t j=count==1?0:std::min(static_cast<size_t>(std::floor(u)),count-2);
            const size_t k=count==1?0:j+1;const double t=count==1?0:u-j;
            q=installed_print_chart(p,(1-t)*vertices[2*j]+t*vertices[2*k],
                                      (1-t)*vertices[2*j+1]+t*vertices[2*k+1]);
            if(!std::isfinite(q.x)||!std::isfinite(q.y))status=Status::nonfinite;
        }
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok){sz[2*i]=q.x;sz[2*i+1]=q.y;}
        else {sz[2*i]=sz[2*i+1]=std::numeric_limits<double>::quiet_NaN();}
    }
    return Status::ok;
}
/* One finite-path evaluator used by replay, reviews and future factors.
 * kind: 0 segment XYZ, 1 planar circle/arc, 2 board chart polyline,
 * 3 board boundary, 4 XYZ polyline, 5 installed board polyline,
 * 6 rotated planar ellipse/arc [cx,cy,z,ax,ay,yaw,a0,a1],
 * 7 continuous board height course [height_fraction], with UNWRAPPED global
 * arc-length station (not local 0..1). Geometric piece joins are not bounds.
 * Legacy parameter coordinates preserve the
 * original vertex-index chart, including single-vertex material points.
 */
inline Status sample_entity_path(const double* bp,size_t nb,int kind,
    const double* p,size_t np,const double* vertices,size_t nv,
    const double* stations,size_t n,double* xyz,int* statuses) {
    if((n&&(!stations||!xyz||!statuses))||kind<0||kind>7)return Status::invalid_argument;
    const bool is_board=(kind==2||kind==3||kind==5||kind==7);
    std::unique_ptr<BoardCourse> course;
    double height=0;
    if(is_board){
        if(!bp||nb!=6)return Status::invalid_argument;
        BoardShape b{bp[0],bp[1],bp[2],bp[3],bp[4],bp[5]};
        if(!b.valid())return Status::invalid_argument;
        course.reset(new BoardCourse(b));height=b.height;
        if(!std::isfinite(course->perimeter()))return Status::nonfinite;
    }
    if(kind==0&&(np!=6||!finite_array(p,np)))return Status::invalid_argument;
    if(kind==1&&(np!=6||!finite_array(p,np)||p[3]<=0||p[5]<=p[4]))return Status::invalid_argument;
    if(kind==6&&(np!=8||!finite_array(p,np)||p[3]<=0||p[4]<=0||p[7]<=p[6]))return Status::invalid_argument;
    if(kind==3&&(np!=2||!finite_array(p,np)||p[0]!=std::floor(p[0])||p[0]<-2||p[0]>7||p[1]<0||p[1]>1))return Status::invalid_argument;
    if(kind==5&&(np!=5||!finite_array(p,np)||p[3]<=0||p[4]<=0))return Status::invalid_argument;
    if(kind==7&&(np!=1||!finite_array(p,np)||p[0]<0||p[0]>1))return Status::invalid_argument;
    const size_t dim=(kind==2||kind==5)?2:3;
    if((kind==2||kind==4||kind==5)&&(nv<dim||nv%dim||!finite_array(vertices,nv)))return Status::invalid_argument;
    for(size_t i=0;i<n;++i){
        const double s=stations[i];Vec3 q{};Status status=Status::ok;
        if(!std::isfinite(s))status=Status::invalid_argument;
        else if(kind!=7&&(s<0||s>1))status=Status::outside_support;
        else if(kind==7){Vec3 tangent{};status=course->point(s,p[0]*height,q,tangent);}
        else if(kind==0)q=add(mul(1-s,load3(p)),mul(s,load3(p+3)));
        else if(kind==1){const double a=p[4]+s*(p[5]-p[4]);q={p[0]+p[3]*std::cos(a),p[1]+p[3]*std::sin(a),p[2]};}
        else if(kind==6){const double a=p[6]+s*(p[7]-p[6]),x=p[3]*std::cos(a),y=p[4]*std::sin(a),ca=std::cos(p[5]),sa=std::sin(p[5]);q={p[0]+ca*x-sa*y,p[1]+sa*x+ca*y,p[2]};}
        else if(kind==2||kind==4||kind==5){
            const size_t count=nv/dim;const double u=s*(count-1);
            const size_t j=count==1?0:std::min(static_cast<size_t>(std::floor(u)),count-2);
            const size_t k=count==1?0:j+1;const double t=count==1?0:u-j;
            const double x=(1-t)*vertices[j*dim]+t*vertices[k*dim];
            const double y=(1-t)*vertices[j*dim+1]+t*vertices[k*dim+1];
            if(kind==2){Vec3 tangent{};status=course->point(x,y,q,tangent);}
            else if(kind==5){const Vec2 chart=installed_print_chart(p,x,y);Vec3 tangent{};status=course->point(chart.x,chart.y,q,tangent);}
            else q={x,y,(1-t)*vertices[j*dim+2]+t*vertices[k*dim+2]};
        } else {
            const int segment=static_cast<int>(p[0]);const auto& breaks=course->breaks();double position;
            if(segment==-2){const double u=s*8;const int j=std::min(static_cast<int>(std::floor(u)),7);position=breaks[j]+(u-j)*(breaks[j+1]-breaks[j]);}
            else if(segment>=0)position=breaks[segment]+s*(breaks[segment+1]-breaks[segment]);
            else position=s*course->perimeter();
            Vec3 tangent{};status=course->point(position,p[1]*height,q,tangent);
        }
        if(status==Status::ok&&!finite(q))status=Status::nonfinite;
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok)store3(xyz+3*i,q);
        else {const double nan=std::numeric_limits<double>::quiet_NaN();xyz[3*i]=xyz[3*i+1]=xyz[3*i+2]=nan;}
    }
    return Status::ok;
}

/* Native construction for the confirmed shared scene family v2.
 * legacy layout33: curling[8], lane[10], hockey[5], faceoff[6], ends[4].
 * layout35: same first29, ends[6]={axL,ayL,axR,ayR,trap_goal_half,trap_board_half}.
 * The two semiaxes of each actual blue crease are independent. Legacy33 keeps
 * its original circular replay interpretation; no old state is rewritten.
 * These are estimable geometry parameters, never observations or metre facts.
 */
inline Status shared_entity_definition(const double* bp,size_t nb,const double* q,size_t nq,
                                      int code,int& kind,double* out,size_t no,unsigned options=0){
    if(!bp||nb!=6||!q||(nq!=33&&nq!=35)||!out||!finite_array(q,nq)||(options&~15u))return Status::invalid_argument;
    if(((options&4u)||(options&8u))&&!(options&2u))return Status::invalid_argument;
    const bool ellipse=(nq==35&&(code==410||code==411));
    if(no!=(ellipse?8u:6u))return Status::invalid_argument;
    const BoardShape board{bp[0],bp[1],bp[2],bp[3],bp[4],bp[5]};
    if(!board.valid())return Status::invalid_argument;
    const double* c=q;const double* lane=q+8;const double* h=q+18;const double* f=q+23;const double* e=q+29;
    // Four equally spaced rows centred between y=+-board.half_width.
    // c[2] remains a legacy slot, not an independent DOF in this option.
    const double y0=(options&1u)?-1.5*c[3]:c[2];
    if(!(c[0]<c[1]&&c[3]>0&&c[4]>0&&0<c[5]&&c[5]<c[6]&&c[6]<c[7]&&c[7]<1&&lane[1]>0&&h[1]>0&&h[2]>0&&((options&2u)||h[3]<h[4])&&f[0]<0&&f[1]>0))return Status::invalid_argument;
    // A1/A4 and A2/A3 share rows and mirror about the transverse x=0 axis.
    // In this declared chart f[2]=common half longitudinal separation and
    // f[4]=common half transverse separation. f[3]/f[5] are inactive storage.
    // Goal lines retain their independent positions; no goal mirror is added.
    if(options&8u){
        if(!(f[2]>0&&f[4]>0&&f[2]<std::min(-f[0],f[1])))return Status::invalid_argument;
    }
    // Current Lane drawing support cannot extend past the left goal line.
    // Bounds are checked, never clamped, and each row retains its own support.
    if(options&4u){
        if(code>=200&&code<204){
            const double end=lane[6+code-200];
            // Each connection passes R and ends strictly before N. Its
            // support stays independent of both R and the Lane right edge.
            if(c[0]<f[0]||!(c[1]<end&&end<lane[0]))return Status::outside_support;
        }
        if(code>=300&&code<312){
            for(int row=0;row<4;++row)
                if(lane[2+row]<f[0]||lane[2+row]>lane[0])return Status::outside_support;
        }
    }
    const double yaw=(options&2u)?0.:h[0];
    const double ca=std::cos(yaw),sa=std::sin(yaw);
    auto point=[&](double x,double y){return Vec3{ca*x-sa*y,sa*x+ca*y,0};};
    auto line=[&](Vec3 a,Vec3 b){kind=0;store3(out,a);store3(out+3,b);return Status::ok;};
    auto circle=[&](Vec3 center,double radius,double a0,double a1){kind=1;store3(out,center);out[3]=radius;out[4]=a0;out[5]=a1;return radius>0?Status::ok:Status::invalid_argument;};
    if(code>=100&&code<132){const int i=(code-100)/4,ring=(code-100)%4;const double ratios[]={c[5],c[6],c[7],1.};return circle({i<4?c[0]:c[1],y0+(i%4)*c[3],0},c[4]*ratios[ring],0,2*pi);}
    if(code>=200&&code<204){const int row=code-200;return line({c[0],y0+row*c[3],0},{lane[6+row],y0+row*c[3],0});}
    if(code>=300&&code<312){const int row=(code-300)/3,edge=(code-300)%3;const double y=y0+row*c[3],lo=y-lane[1]/2,hi=y+lane[1]/2;
        if(edge==0)return line({lane[2+row],hi,0},{lane[0],hi,0});
        if(edge==1)return line({lane[0],hi,0},{lane[0],lo,0});
        return line({lane[0],lo,0},{lane[2+row],lo,0});}
    if(code>=400&&code<=404){
        const double xs[]={0,-h[1],h[1],f[0],f[1]};const double x=xs[code-400];
        if(options&2u){Segment carrier{};const auto status=board.vertical_section(x,carrier);
            if(status!=Status::ok)return status;return line(carrier.a,carrier.b);}
        return line(point(x,h[4]),point(x,h[3]));
    }
    if(code==405)return circle({0,0,0},h[2],0,2*pi);
    if(code>=406&&code<=409){
        if(options&8u){const double x[]={f[2],f[2],-f[2],-f[2]},y[]={-f[4],f[4],f[4],-f[4]};
            return circle(point(x[code-406],y[code-406]),h[2],0,2*pi);}
        const double x[]={f[1]-f[3],f[1]-f[3],f[0]+f[2],f[0]+f[2]},y[]={-f[5],f[5],f[4],-f[4]};
        return circle(point(x[code-406],y[code-406]),h[2],0,2*pi);}

    if(code==410||code==411){const int side=code-410;
        if(nq==35){const double ax=e[side*2],ay=e[side*2+1],a0=side?pi/2:-pi/2;
            if(!(ax>0&&ay>0))return Status::invalid_argument;
            kind=6;store3(out,point(f[side],0));out[3]=ax;out[4]=ay;out[5]=yaw;out[6]=a0;out[7]=a0+pi;return Status::ok;}
        const double a0=(side?pi/2:-pi/2)+yaw;return circle(point(f[side],0),e[side],a0,a0+pi);}

    if(code>=420&&code<=423){
        const int side=(code-420)/2;const double sign=side?1.:-1.,ysign=(code%2)?1.:-1.;
        const double goal_half=e[nq==35?4:2],board_half=e[nq==35?5:3];
        if(!(goal_half>0&&board_half>0))return Status::invalid_argument;
        const double y=ysign*board_half;
        auto membership=[&](double x){const Vec3 p=point(x,y);const double u=std::fmax(0.,(std::abs(p.x)-(board.half_length-board.rx))/board.rx),v=std::fmax(0.,(std::abs(p.y)-(board.half_width-board.ry))/board.ry);return std::pow(u,board.exponent)+std::pow(v,board.exponent)-1.;};
        if(membership(0)>=0)return Status::outside_support;
        double lo=0,hi=4*std::max(board.half_length,board.half_width);
        if(membership(sign*hi)<=0)return Status::no_intersection;
        for(int i=0;i<60;++i){const double mid=(lo+hi)/2;if(membership(sign*mid)>0)hi=mid;else lo=mid;}
        return line(point(f[side],ysign*goal_half),point(sign*(lo+hi)/2,y));
    }
    return Status::invalid_argument;
}
} // namespace rinkspace
