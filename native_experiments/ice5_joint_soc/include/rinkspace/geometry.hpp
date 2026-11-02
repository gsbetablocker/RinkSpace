#pragma once
#include "core_types.hpp"
#include <algorithm>
#include <array>
#include <limits>

namespace rinkspace {
struct CurlingLayout {
    double xF,xR,y0,pitch,radius,middle_ratio,inner_ratio,button_ratio,lane_width,right_x;
    RS_HD inline bool valid()const{
        const double p[]={xF,xR,y0,pitch,radius,middle_ratio,inner_ratio,button_ratio,lane_width,right_x};
        return finite_array(p,10)&&xF<xR&&pitch>0&&radius>0&&lane_width>0&&
            0<=button_ratio&&button_ratio<inner_ratio&&inner_ratio<middle_ratio&&middle_ratio<1;
    }
    RS_HD Vec3 centre(int house,int rows)const{return {house<rows?xF:xR,y0+(house%rows)*pitch,0};}
    RS_HD Vec3 ring(int house,int rows,int which,double theta)const{
        const double ratios[]={1.,middle_ratio,inner_ratio,button_ratio};
        Vec3 c=centre(house,rows);return {c.x+radius*ratios[which]*std::cos(theta),c.y+radius*ratios[which]*std::sin(theta),0};
    }
    RS_HD Vec3 lane_vertex(int row,int edge,int endpoint,double left_x)const{
        const double low=y0+row*pitch-lane_width/2,high=low+lane_width;
        if(edge==0)return {endpoint?right_x:left_x,low,0};
        if(edge==1)return {right_x,endpoint?high:low,0};
        return {endpoint?left_x:right_x,high,0};
    }
};
template<class T> struct SegmentValue { Vec3Value<T> a,b; };
using Segment=SegmentValue<double>;
struct Circle { Vec3 centre; double radius; };
RS_HD inline double cross2(Vec3 a,Vec3 b){return a.x*b.y-a.y*b.x;}
RS_HD inline bool same_plane(Vec3 a,Vec3 b){return std::abs(a.z-b.z)<=1e-11*(1+std::abs(a.z)+std::abs(b.z));}
RS_HD inline Status line_line(Segment a,Segment b,Vec3& out){
    if(!finite(a.a)||!finite(a.b)||!finite(b.a)||!finite(b.b))return Status::invalid_argument;
    if(!same_plane(a.a,a.b)||!same_plane(a.a,b.a)||!same_plane(a.a,b.b))return Status::not_coplanar;
    const Vec3 da=sub(a.b,a.a),db=sub(b.b,b.a),q=sub(b.a,a.a);
    const double na=da.x*da.x+da.y*da.y,nb=db.x*db.x+db.y*db.y;
    if(na<=0||nb<=0)return Status::degenerate;
    const double den=cross2(da,db);
    if(std::abs(den)<=1e-12*std::sqrt(na*nb))return Status::degenerate;
    const double t=cross2(q,db)/den,u=cross2(q,da)/den;
    if(t<0||t>1||u<0||u>1)return Status::outside_support;
    out=add(a.a,mul(t,da));return finite(out)?Status::ok:Status::nonfinite;
}
RS_HD inline Status line_circle(Segment line,Circle circle,int branch,Vec3& out){
    if(branch<0||branch>1||!finite(line.a)||!finite(line.b)||!finite(circle.centre)||
       !finite_scalar(circle.radius)||circle.radius<=0)return Status::invalid_argument;
    if(!same_plane(line.a,line.b)||!same_plane(line.a,circle.centre))return Status::not_coplanar;
    const Vec3 d=sub(line.b,line.a),f=sub(line.a,circle.centre);
    const double a=d.x*d.x+d.y*d.y;
    if(a<=0)return Status::degenerate;
    const double distance2=cross2(f,d)*cross2(f,d)/a;
    const double h2=circle.radius*circle.radius-distance2;
    const double tol=1e-12*(circle.radius*circle.radius+distance2);
    if(h2 < -tol)return Status::no_intersection;
    if(std::abs(h2)<=tol)return Status::degenerate;
    const double tc=-(f.x*d.x+f.y*d.y)/a,offset=std::sqrt(h2/a);
    const double t=tc+(branch?offset:-offset);
    if(t<0||t>1)return Status::outside_support;
    out=add(line.a,mul(t,d));return finite(out)?Status::ok:Status::nonfinite;
}

template<class T> struct BoardShapeValue {
    T half_length,half_width,rx,ry,exponent,height;
    RS_HD inline bool valid()const{
        const T p[]={half_length,half_width,rx,ry,exponent,height};
        return finite_array(p,6)&&half_length>0&&half_width>0&&rx>0&&ry>0&&
            rx<half_length&&ry<half_width&&exponent>1&&height>0;
    }
    RS_HD Vec2Value<T> polar(T theta)const{
        using std::cos;using std::sin;using std::pow;
        const T c=std::max<T>(cos(theta),T(0.)),s=std::max<T>(sin(theta),T(0.));
        const T rho=pow(pow(c,exponent)+pow(s,exponent),-1/exponent);
        return {rx*c*rho,ry*s*rho};
    }
    /* A world-Y carrier intersects this same continuous course. The corner
     * intersection is solved on polar(), not on a rectangular clipping box.
     * A carrier coincident with an end straight is degenerate, not invented.
     */
    RS_HD inline Status vertical_section(T x,SegmentValue<T>& out)const{
        using std::abs;using std::pow;
        if(!valid()||!finite_scalar(x))return Status::invalid_argument;
        const T ax=abs(x);
        if(ax>half_length)return Status::outside_support;
        if(ax==half_length)return Status::degenerate;
        T y=half_width;
        if(ax>half_length-rx){
            const T u=(ax-(half_length-rx))/rx;
            y=half_width-ry+ry*pow(1-pow(u,exponent),1/exponent);
        }
        out={{x,y,0.},{x,-y,0.}};
        return finite(out.a)&&finite(out.b)?Status::ok:Status::nonfinite;
    }
    RS_HD Vec2Value<T> derivative(T theta)const{
        using std::cos;using std::sin;using std::pow;
        const T c=std::max<T>(cos(theta),T(0.)),s=std::max<T>(sin(theta),T(0.));
        const T norm=pow(c,exponent)+pow(s,exponent),rho=pow(norm,-1/exponent);
        const T dr=(pow(c,exponent-1)*s-pow(s,exponent-1)*c)/norm;
        return {rx*rho*(-s+c*dr),ry*rho*(c+s*dr)};
    }
    RS_HD T speed(T theta)const{using std::sqrt;const Vec2Value<T> v=derivative(theta);return sqrt(v.x*v.x+v.y*v.y);}
};

using BoardShape=BoardShapeValue<double>;

/* Fixed ice/board-bottom intersection. Branch 0/1 orders the two roots along
 * the directed line BEFORE testing its finite support; a missing branch never
 * switches to the other root. Uses the same rounded-superellipse boundary as
 * BoardShape::polar. Tangencies and coincident boundary straights are not an
 * isolated, stable junction. There is no observation/station input. */
inline Status ice_board_junction(BoardShape board,Segment line,double height_fraction,
                                 int branch,Vec3& out){
    if(!board.valid()||!finite(line.a)||!finite(line.b)||
       !finite_scalar(height_fraction)||branch<0||branch>1)return Status::invalid_argument;
    if(height_fraction!=0||line.a.z!=0||line.b.z!=0)return Status::not_coplanar;
    const Vec3 d=sub(line.b,line.a);
    if(!finite(d))return Status::nonfinite;
    if(d.x==0&&d.y==0)return Status::degenerate;
    double lo=-std::numeric_limits<double>::infinity(),hi=-lo;
    const double origin[]={line.a.x,line.a.y},direction[]={d.x,d.y};
    const double extent[]={board.half_length,board.half_width};
    for(int axis=0;axis<2;++axis){
        if(direction[axis]==0){
            if(std::abs(origin[axis])>extent[axis])return Status::no_intersection;
        }else{
            const double a=(-extent[axis]-origin[axis])/direction[axis];
            const double b=(extent[axis]-origin[axis])/direction[axis];
            lo=std::max(lo,std::min(a,b));hi=std::min(hi,std::max(a,b));
        }
    }
    if(!finite_scalar(lo)||!finite_scalar(hi))return Status::nonfinite;
    if(lo>hi)return Status::no_intersection;
    if(lo==hi)return Status::degenerate;
    const auto level=[&](double t){
        const Vec3 p=add(line.a,mul(t,d));
        const double u=std::max(0.,(std::abs(p.x)-(board.half_length-board.rx))/board.rx);
        const double v=std::max(0.,(std::abs(p.y)-(board.half_width-board.ry))/board.ry);
        return std::pow(u,board.exponent)+std::pow(v,board.exponent)-1.;
    };
    // This level function is convex because exponent>1. An interior minimum
    // brackets both crossings even when neither finite endpoint is inside.
    double ml=lo,mh=hi;
    for(int k=0;k<80;++k){
        const double a=ml+(mh-ml)/3,b=mh-(mh-ml)/3;
        if(level(a)<level(b))mh=b;else ml=a;
    }
    const double middle=(ml+mh)/2,minimum=level(middle);
    const double tolerance=64*std::numeric_limits<double>::epsilon();
    if(!finite_scalar(minimum))return Status::nonfinite;
    if(minimum>tolerance)return Status::no_intersection;
    if(minimum>=-tolerance)return Status::degenerate;
    double a=branch?middle:lo,b=branch?hi:middle;
    double t=branch?hi:lo;
    if(std::abs(level(t))>tolerance){
        for(int k=0;k<80;++k){
            const double m=a+(b-a)/2;
            if((level(m)>0)==(branch==0))a=m;else b=m;
        }
        t=a+(b-a)/2;
    }
    // Only reconcile root-solver roundoff at an endpoint that itself lies on
    // this boundary. A real extension-only intersection remains unsupported.
    const double slack=tolerance*(1+std::abs(t));
    if(t<0&&t>=-slack&&std::abs(level(0))<=tolerance)t=0;
    if(t>1&&t<=1+slack&&std::abs(level(1))<=tolerance)t=1;
    if(t<0||t>1)return Status::outside_support;
    out=add(line.a,mul(t,d));
    return finite(out)?Status::ok:Status::nonfinite;
}

/* Immutable station cache, constructed once on host and copied unchanged
 * to CUDA. CPU and GPU use the same point/inverse-arclength implementation.
 * Integration occurs once per scene shape per batch. Each inverse only
 * integrates within one small cached interval, never from zero per point.
 */
template<class T> class BoardCourseValue {
public:
    static constexpr int cells=512;
    RS_HD explicit BoardCourseValue(BoardShapeValue<T> b):shape_(b){
        arc_[0]=0;
        for(int i=0;i<cells;++i)arc_[i+1]=arc_[i]+integrate(i*step(),(i+1)*step());
        const T x=2*(b.half_length-b.rx),y=2*(b.half_width-b.ry),c=arc_[cells];
        const T lengths[]={x,c,y,c,x,c,y,c};breaks_[0]=0;
        for(int i=0;i<8;++i)breaks_[i+1]=breaks_[i]+lengths[i];
    }
    RS_HD T perimeter()const{return breaks_[8];}
    RS_HD T corner_length()const{return arc_[cells];}
    RS_HD const std::array<T,9>& breaks()const{return breaks_;}
    RS_HD Status point(T station,T z,Vec3Value<T>& xyz,Vec3Value<T>& tangent)const{
        using std::fmod;using std::sqrt;
        if(!finite_scalar(station)||!finite_scalar(z))return Status::invalid_argument;
        if(z<0||z>shape_.height)return Status::outside_support;
        T s=fmod(station,perimeter());if(s<0)s+=perimeter();
        int part=0;while(part<7&&s>breaks_[part+1])++part;
        const T along=s-breaks_[part],a=shape_.half_length,b=shape_.half_width;
        if(part==0){xyz={-a+shape_.rx+along,-b,z};tangent={1,0,0};}
        else if(part==2){xyz={a,-b+shape_.ry+along,z};tangent={0,1,0};}
        else if(part==4){xyz={a-shape_.rx-along,b,z};tangent={-1,0,0};}
        else if(part==6){xyz={-a,b-shape_.ry-along,z};tangent={0,-1,0};}
        else{
            const bool reverse=(part==1||part==5);
            const T target=reverse?corner_length()-along:along;
            const T theta=angle(target);const Vec2Value<T> q=shape_.polar(theta),dq=shape_.derivative(theta);
            const T sx=(part==1||part==3)?1.:-1.,sy=(part==3||part==5)?1.:-1.;
            xyz={(a-shape_.rx)*sx+q.x*sx,(b-shape_.ry)*sy+q.y*sy,z};
            const T speed=sqrt(dq.x*dq.x+dq.y*dq.y),direction=reverse?-1.:1.;
            tangent={direction*sx*dq.x/speed,direction*sy*dq.y/speed,0};
        }
        // The first position derivative is finite at C1 superellipse joins;
        // derivative of the tangent (curvature) can diverge for 1<n<2 and is
        // not a positional Jacobian. Consumers use the primal unit tangent.
        return finite(xyz)&&finite(Vec3{primal(tangent.x),primal(tangent.y),primal(tangent.z)})?Status::ok:Status::nonfinite;
    }
private:
    RS_HD static constexpr double step(){return (pi/2)/cells;}
    RS_HD T integrate(T lo,T hi)const{
        // Positive half of the standard 16-point Gauss-Legendre rule.
        static constexpr double nodes[]={.095012509837637440185,.28160355077925891323,.45801677765722738634,.61787624440264374845,.75540440835500303390,.86563120238783174388,.94457502307323257608,.98940093499164993260};
        static constexpr double weights[]={.18945061045506849629,.18260341504492358887,.16915651939500253819,.14959598881657673208,.12462897125553387205,.09515851168249278481,.062253523938647892863,.027152459411754094852};
        const T mid=(lo+hi)/2,half=(hi-lo)/2;T sum=0;
        for(int i=0;i<8;++i)sum+=weights[i]*(shape_.speed(mid-half*nodes[i])+shape_.speed(mid+half*nodes[i]));
        return half*sum;
    }
    RS_HD T angle(T target)const{
        using std::abs;
        if(target<=0)return target/shape_.speed(T(0));
        if(target>=corner_length())return pi/2+(target-corner_length())/shape_.speed(T(pi/2));
        int begin=0,end=cells+1;
        while(begin<end){const int mid=begin+(end-begin)/2;
            if(target<arc_[mid])end=mid;else begin=mid+1;}
        const int cell=begin-1;
        const T origin=cell*step();T lo=origin,hi=origin+step();
        T theta=lo+(hi-lo)*(target-arc_[cell])/(arc_[cell+1]-arc_[cell]);
        for(int i=0;i<20;++i){
            const T error=arc_[cell]+integrate(origin,theta)-target;
            if(abs(error)<=1e-14*perimeter())return theta-error/shape_.speed(theta);
            if(error<0)lo=theta;else hi=theta;
            const T candidate=theta-error/shape_.speed(theta);
            theta=(candidate>lo&&candidate<hi)?candidate:(lo+hi)/2;
        }
        const T residual=arc_[cell]+integrate(origin,theta)-target;
        return abs(residual)<=1e-14*perimeter()?theta:T(std::numeric_limits<double>::quiet_NaN());
    }
    BoardShapeValue<T> shape_;std::array<T,cells+1> arc_{};std::array<T,9> breaks_{};
};
using BoardCourse=BoardCourseValue<double>;

} // namespace rinkspace
