#pragma once
#include "geometry.hpp"
#include <memory>

namespace rinkspace {
/* Floating implementation of a legal convex interpolation. The station
 * caller has already established 0<=t<=1, so the exact result MUST be in the
 * endpoint interval. Preserve the historical arithmetic whenever it obeys
 * that interval; repair only its provable floating rounding residue. This
 * never widens the shape box or clamps a caller-supplied material point.
 */
RS_HD inline double print_convex_interpolate(double a,double b,double t){
    if(a==b||t==0)return a;
    if(t==1)return b;
    const double value=(1-t)*a+t*b;
    return std::max(std::min(a,b),std::min(std::max(a,b),value));
}
/* Shared installed material chart, before wrapping onto the board course.
 * p=[unwrapped_s,z,rotation,width,height]; dimensions stay positive.
 * Template chart orientation is explicit in the supplied material vertices. */
template<class T> RS_HD inline Vec2Value<T> installed_print_chart(const T* p,T x,T y){
    using std::cos;using std::sin;const T ca=cos(p[2]),sa=sin(p[2]);
    return {p[0]+ca*x*p[3]-sa*y*p[4],p[1]+sa*x*p[3]+ca*y*p[4]};
}
/* A regular square in the Euclidean unfolded (s,z) plane. Material UV axes
 * have different physical scales; correcting by height/width keeps the
 * square exact when the enclosing print dimensions or rotation change.
 * The result contains four corners followed by the first corner. */
inline Status print_square_vertices(const double* p,size_t np,const double* centre,size_t nc,
                                    double side_over_height,double* vertices){
    if(np!=5||nc!=2||!vertices||!finite_array(p,np)||!finite_array(centre,nc)||
       p[3]<=0||p[4]<=0||!finite_scalar(side_over_height)||side_over_height<=0)
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
        if(!finite_scalar(s))status=Status::invalid_argument;
        else if(s<0||s>1)status=Status::outside_support;
        else {
            const double u=s*(count-1);
            const size_t j=count==1?0:std::min(static_cast<size_t>(std::floor(u)),count-2);
            const size_t k=count==1?0:j+1;const double t=count==1?0:u-j;
            q=installed_print_chart(p,print_convex_interpolate(vertices[2*j],vertices[2*k],t),
                                      print_convex_interpolate(vertices[2*j+1],vertices[2*k+1],t));
            if(!finite_scalar(q.x)||!finite_scalar(q.y))status=Status::nonfinite;
        }
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok){sz[2*i]=q.x;sz[2*i+1]=q.y;}
        else {sz[2*i]=sz[2*i+1]=std::numeric_limits<double>::quiet_NaN();}
    }
    return Status::ok;
}
/* One bounded, orientation-preserving material-chart deformation. The four
 * positive axis intervals are a softmax of three free logits and one fixed
 * zero logit. All paths of a material group, including its holes, use the same
 * map. This is not an independent displacement for drawing samples.
 * shape10=[tx,ty,log_sx,log_sy,x_logw0..2,y_logw0..2]. */
struct PrintShape10 {
    double lo[2]{},span[2]{},centre[2]{},translation[2]{},scale[2]{},scale_delta[2]{};
    double weights[2][4]{};
    RS_HD Status initialize(const double* bounds,size_t nb,const double* values,size_t nv){
        if(nb!=4||nv!=10||!finite_array(bounds,nb)||!finite_array(values,nv))
            return Status::invalid_argument;
        for(int axis=0;axis<2;++axis){
            lo[axis]=bounds[axis];span[axis]=bounds[axis+2]-bounds[axis];
            if(!(span[axis]>0)||!finite_scalar(span[axis]))return Status::invalid_argument;
            centre[axis]=lo[axis]+span[axis]/2;translation[axis]=values[axis];
            scale[axis]=std::exp(values[axis+2]);scale_delta[axis]=std::expm1(values[axis+2]);
            if(!finite_scalar(scale[axis])||!finite_scalar(scale_delta[axis]))return Status::nonfinite;
            if(!(scale[axis]>0))return Status::degenerate;
            const double* logits=values+4+3*axis;
            const double maximum=std::max(0.,std::max(logits[0],std::max(logits[1],logits[2])));
            double sum=0;
            for(int k=0;k<4;++k){
                weights[axis][k]=std::exp((k<3?logits[k]:0.)-maximum);
                if(!(weights[axis][k]>0)||!finite_scalar(weights[axis][k]))return Status::degenerate;
                sum+=weights[axis][k];
            }
            const double extent=scale[axis]*span[axis];
            if(!finite_scalar(extent))return Status::nonfinite;
            if(!(extent>0))return Status::degenerate;
            for(int k=0;k<4;++k){
                weights[axis][k]/=sum;
                const double derivative=(scale[axis]*weights[axis][k])*4;
                // Domain validity covers the whole group and cannot depend
                // on whether a caller requests a Jacobian or samples this cell.
                if(!(weights[axis][k]>0&&derivative>0))return Status::degenerate;
                if(!finite_scalar(derivative))return Status::nonfinite;
            }
        }
        return Status::ok;
    }
    RS_HD Status point(double x,double y,Vec2& output,double* jacobian=nullptr,
                 double* seed_derivative=nullptr)const{
        const double seed[]={x,y};double result[2]{};
        if(!finite_array(seed,2))return Status::invalid_argument;
        if(jacobian)for(int k=0;k<20;++k)jacobian[k]=0;
        for(int axis=0;axis<2;++axis){
            const double q=(seed[axis]-lo[axis])/span[axis];
            if(!finite_scalar(q)||q<0||q>1)return Status::outside_support;
            double coverage[4]{},delta=0;
            for(int k=0;k<4;++k){
                coverage[k]=std::min(1.,std::max(0.,4*q-k));
                // At zero state each term is exactly zero. Do not form f(q)-q
                // by subtracting independently accumulated identity positions.
                delta+=(weights[axis][k]-.25)*coverage[k];
            }
            // The normalized interval lengths sum to one mathematically.
            // Preserve that exact endpoint identity instead of exposing the
            // floating summation residue as a moving group boundary.
            const bool endpoint=(q==0||q==1);
            if(endpoint)delta=0;
            result[axis]=seed[axis]+translation[axis]+
                scale_delta[axis]*(seed[axis]-centre[axis])+scale[axis]*span[axis]*delta;
            if(!finite_scalar(result[axis]))return Status::nonfinite;
            if(jacobian){
                double* row=jacobian+10*axis;
                row[axis]=1;
                row[axis+2]=scale[axis]*(seed[axis]-centre[axis]+span[axis]*delta);
                for(int k=0;k<3;++k)
                    row[4+3*axis+k]=endpoint?0.:scale[axis]*span[axis]*weights[axis][k]*(coverage[k]-(q+delta));
                if(!finite_array(row,10))return Status::nonfinite;
            }
            if(seed_derivative){
                const int cell=std::min(static_cast<int>(std::floor(4*q)),3);
                seed_derivative[axis]=scale[axis]*4*weights[axis][cell];
                if(!(seed_derivative[axis]>0)||!finite_scalar(seed_derivative[axis]))return Status::degenerate;
            }
        }
        output={result[0],result[1]};return Status::ok;
    }
};
inline Status deform_print_points(const double* vertices,size_t nv,const double* bounds,size_t nb,
    const double* values,size_t nvalues,double* uv,double* jacobian,int* statuses){
    if(nv%2||(nv&&(!vertices||!uv||!statuses)))return Status::invalid_argument;
    PrintShape10 shape;const Status valid=shape.initialize(bounds,nb,values,nvalues);
    if(valid!=Status::ok)return valid;
    for(size_t i=0;i<nv/2;++i){
        Vec2 point{};double jac[20]{};
        const auto status=shape.point(vertices[2*i],vertices[2*i+1],point,jacobian?jac:nullptr);
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok){
            uv[2*i]=point.x;uv[2*i+1]=point.y;
            if(jacobian)for(int j=0;j<20;++j)jacobian[20*i+j]=jac[j];
        }
    }
    return Status::ok;
}
/* Evaluate source-index station BEFORE deformation. Warping only original
 * vertices and subsequently interpolating them would use a different curve
 * between vertices and can break the shared-map hole guarantee. */
inline Status sample_print_deformed_path(const double* bp,size_t nb,const double* p,size_t np,
    const double* vertices,size_t nv,const double* stations,size_t n,
    const double* bounds,size_t nbound,const double* values,size_t nvalues,
    double* uv,double* sz,double* xyz,double* juv,double* jsz,double* jxyz,
    double* jinstallation,double* jstation,int* statuses){
    const bool material_only=!bp&&nb==0;
    const bool derivatives=juv||jsz||jxyz||jinstallation||jstation;
    if((!material_only&&(nb!=6||!bp))||(material_only&&(xyz||derivatives))||
       np!=5||!finite_array(p,np)||!(p[3]>0&&p[4]>0)||
       nv<2||nv%2||!finite_array(vertices,nv)||(n&&(!stations||!uv||!sz||(!material_only&&!xyz)||!statuses)))
        return Status::invalid_argument;
    std::unique_ptr<BoardCourse> course;
    if(!material_only){
        const BoardShape board{bp[0],bp[1],bp[2],bp[3],bp[4],bp[5]};
        if(!board.valid())return Status::invalid_argument;
        course.reset(new BoardCourse(board));
        if(!finite_scalar(course->perimeter()))return Status::nonfinite;
    }
    PrintShape10 shape;const Status valid=shape.initialize(bounds,nbound,values,nvalues);
    if(valid!=Status::ok)return valid;
    const size_t count=nv/2;const double ca=std::cos(p[2]),sa=std::sin(p[2]);
    for(size_t i=0;i<n;++i){
        Status status=Status::ok;Vec2 material{},chart{};Vec3 point{},tangent{};
        double local[20]{},seed_derivative[2]{},chart_jac[20]{},world_jac[30]{},installation[15]{},station_jac[3]{};
        if(!finite_scalar(stations[i]))status=Status::invalid_argument;
        else if(stations[i]<0||stations[i]>1)status=Status::outside_support;
        else {
            const double u=stations[i]*(count-1);
            const size_t j=count==1?0:std::min(static_cast<size_t>(std::floor(u)),count-2);
            const size_t k=count==1?0:j+1;const double t=count==1?0:u-j;
            const double x=print_convex_interpolate(vertices[2*j],vertices[2*k],t);
            const double y=print_convex_interpolate(vertices[2*j+1],vertices[2*k+1],t);
            status=shape.point(x,y,material,derivatives?local:nullptr,jstation?seed_derivative:nullptr);
            if(status==Status::ok){
                chart=installed_print_chart(p,material.x,material.y);
                status=material_only?((finite_scalar(chart.x)&&finite_scalar(chart.y))?Status::ok:Status::nonfinite):
                    course->point(chart.x,chart.y,point,tangent);
            }
            if(status==Status::ok&&derivatives){
                for(int col=0;col<10;++col){
                    const double ds=ca*p[3]*local[col]-sa*p[4]*local[10+col];
                    const double dz=sa*p[3]*local[col]+ca*p[4]*local[10+col];
                    chart_jac[col]=ds;chart_jac[10+col]=dz;
                    world_jac[col]=tangent.x*ds;
                    world_jac[10+col]=tangent.y*ds;world_jac[20+col]=dz;
                }
                const double ds[]={1,0,-sa*material.x*p[3]-ca*material.y*p[4],ca*material.x,-sa*material.y};
                const double dz[]={0,1,ca*material.x*p[3]-sa*material.y*p[4],sa*material.x,ca*material.y};
                for(int col=0;col<5;++col){
                    installation[col]=tangent.x*ds[col];installation[5+col]=tangent.y*ds[col];installation[10+col]=dz[col];
                }
                if(jstation){
                    const double dx=seed_derivative[0]*(count-1)*(vertices[2*k]-vertices[2*j]);
                    const double dy=seed_derivative[1]*(count-1)*(vertices[2*k+1]-vertices[2*j+1]);
                    const double dsdt=ca*p[3]*dx-sa*p[4]*dy,dzdt=sa*p[3]*dx+ca*p[4]*dy;
                    station_jac[0]=tangent.x*dsdt;station_jac[1]=tangent.y*dsdt;station_jac[2]=dzdt;
                }
                if(!finite_array(chart_jac,20)||!finite_array(world_jac,30)||
                   !finite_array(installation,15)||!finite_array(station_jac,3))status=Status::nonfinite;
            }
        }
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok){
            uv[2*i]=material.x;uv[2*i+1]=material.y;sz[2*i]=chart.x;sz[2*i+1]=chart.y;
            if(xyz)store3(xyz+3*i,point);
            if(juv)for(int col=0;col<20;++col)juv[20*i+col]=local[col];
            if(jsz)for(int col=0;col<20;++col)jsz[20*i+col]=chart_jac[col];
            if(jxyz)for(int col=0;col<30;++col)jxyz[30*i+col]=world_jac[col];
            if(jinstallation)for(int col=0;col<15;++col)jinstallation[15*i+col]=installation[col];
            if(jstation)store3(jstation+3*i,{station_jac[0],station_jac[1],station_jac[2]});
        }
    }
    return Status::ok;
}

/* Typed analytic material structures. family=0 is one unfolded-metric square
 * with values=[centre_u,centre_v,log_side_over_panel_height], station [0,1]
 * traverses its four edges. family=1 is a material rectangle with values=
 * [centre_u,centre_v,log_width_uv,log_height_uv], side=top/right/bottom/left,
 * and station traverses that edge in clockwise order (or its explicit reverse).
 * No independent corners can break square/rectangle relationships. */
template<class T> RS_HD inline Status print_analytic_vertices(const T* p,const T* values,int family,T* vertices){
    using std::exp;T width,height;
    if(family==0){height=exp(values[2]);width=height*p[4]/p[3];}
    else{width=exp(values[2]);height=exp(values[3]);}
    if(!finite_scalar(width)||!finite_scalar(height)||!(width>0&&height>0))return Status::degenerate;
    T left=values[0]-width/2,right=values[0]+width/2,top=values[1]-height/2,bottom=values[1]+height/2;
    if(!(left<right&&top<bottom))return Status::degenerate;
    const T v[]={left,top,right,top,right,bottom,left,bottom,left,top};
    for(int k=0;k<10;++k)vertices[k]=v[k];return Status::ok;
}
inline Status sample_print_analytic_path(const double* bp,size_t nb,const double* p,size_t np,
    int family,int side,bool reverse,const double* values,size_t nvalues,
    const double* stations,size_t n,double* uv,double* sz,double* xyz,
    double* juv,double* jsz,double* jxyz,double* jinstallation,double* jstation,int* statuses){
    const bool material_only=!bp&&nb==0;
    const bool derivatives=juv||jsz||jxyz||jinstallation||jstation;
    if((!material_only&&(nb!=6||!bp))||(material_only&&(xyz||derivatives))||
       np!=5||!finite_array(p,np)||!(p[3]>0&&p[4]>0)||
       (family!=0&&family!=1)||nvalues!=(family==0?3u:4u)||!finite_array(values,nvalues)||
       side<0||side>3||(n&&(!stations||!uv||!sz||(!material_only&&!xyz)||!statuses)))return Status::invalid_argument;
    std::unique_ptr<BoardCourse> course;
    if(!material_only){
        const BoardShape board{bp[0],bp[1],bp[2],bp[3],bp[4],bp[5]};
        if(!board.valid())return Status::invalid_argument;
        course.reset(new BoardCourse(board));
        if(!finite_scalar(course->perimeter()))return Status::nonfinite;
    }
    double vertices[10]{};
    auto vertex_status=print_analytic_vertices(p,values,family,vertices);
    if(vertex_status!=Status::ok)return vertex_status;
    const double ca=std::cos(p[2]),sa=std::sin(p[2]);
    for(size_t i=0;i<n;++i){
        Status status=Status::ok;Vec2 material{},chart{};Vec3 point{},tangent{};
        double local[8]{},chart_jac[8]{},world_jac[12]{},installation[15]{},station_jac[3]{};
        if(!finite_scalar(stations[i]))status=Status::invalid_argument;
        else if(stations[i]<0||stations[i]>1)status=Status::outside_support;
        else {
            const double coordinate=reverse?1-stations[i]:stations[i];
            const double u=family==0?4*coordinate:coordinate;
            const int j=family==0?std::min(static_cast<int>(std::floor(u)),3):side;
            const double t=family==0?u-j:u;
            material={print_convex_interpolate(vertices[2*j],vertices[2*j+2],t),
                      print_convex_interpolate(vertices[2*j+1],vertices[2*j+3],t)};
            chart=installed_print_chart(p,material.x,material.y);
            status=material_only?((finite_scalar(chart.x)&&finite_scalar(chart.y))?Status::ok:Status::nonfinite):
                course->point(chart.x,chart.y,point,tangent);
            if(status==Status::ok&&derivatives){
                const int nlocal=static_cast<int>(nvalues);
                local[0]=1;local[nlocal+1]=1;
                local[2]=material.x-values[0];
                local[nlocal+(family==0?2:3)]=material.y-values[1];
                for(int col=0;col<nlocal;++col){
                    const double ds=ca*p[3]*local[col]-sa*p[4]*local[nlocal+col];
                    const double dz=sa*p[3]*local[col]+ca*p[4]*local[nlocal+col];
                    chart_jac[col]=ds;chart_jac[nlocal+col]=dz;
                    world_jac[col]=tangent.x*ds;world_jac[nlocal+col]=tangent.y*ds;
                    world_jac[2*nlocal+col]=dz;
                }
                double ds[]={1,0,-sa*material.x*p[3]-ca*material.y*p[4],ca*material.x,-sa*material.y};
                double dz[]={0,1,ca*material.x*p[3]-sa*material.y*p[4],sa*material.x,ca*material.y};
                if(family==0){
                    // Material square width depends inversely on panel width.
                    // Its physical side therefore stays equal in both axes.
                    const double offset_x=material.x-values[0];
                    const double du_dw=-offset_x/p[3],du_dh=offset_x/p[4];
                    ds[3]+=ca*p[3]*du_dw;dz[3]+=sa*p[3]*du_dw;
                    ds[4]+=ca*p[3]*du_dh;dz[4]+=sa*p[3]*du_dh;
                }
                for(int col=0;col<5;++col){
                    installation[col]=tangent.x*ds[col];installation[5+col]=tangent.y*ds[col];installation[10+col]=dz[col];
                }
                if(jstation){
                    const double factor=(family==0?4.:1.)*(reverse?-1.:1.);
                    const double dx=factor*(vertices[2*j+2]-vertices[2*j]);
                    const double dy=factor*(vertices[2*j+3]-vertices[2*j+1]);
                    const double dsdt=ca*p[3]*dx-sa*p[4]*dy,dzdt=sa*p[3]*dx+ca*p[4]*dy;
                    station_jac[0]=tangent.x*dsdt;station_jac[1]=tangent.y*dsdt;station_jac[2]=dzdt;
                }
                if(!finite_array(local,2*nvalues)||!finite_array(chart_jac,2*nvalues)||
                   !finite_array(world_jac,3*nvalues)||!finite_array(installation,15)||
                   !finite_array(station_jac,3))status=Status::nonfinite;
            }
        }
        statuses[i]=static_cast<int>(status);
        if(status==Status::ok){
            uv[2*i]=material.x;uv[2*i+1]=material.y;sz[2*i]=chart.x;sz[2*i+1]=chart.y;
            if(xyz)store3(xyz+3*i,point);
            if(juv)for(size_t col=0;col<2*nvalues;++col)juv[2*nvalues*i+col]=local[col];
            if(jsz)for(size_t col=0;col<2*nvalues;++col)jsz[2*nvalues*i+col]=chart_jac[col];
            if(jxyz)for(size_t col=0;col<3*nvalues;++col)jxyz[3*nvalues*i+col]=world_jac[col];
            if(jinstallation)for(int col=0;col<15;++col)jinstallation[15*i+col]=installation[col];
            if(jstation)store3(jstation+3*i,{station_jac[0],station_jac[1],station_jac[2]});
        }
    }
    return Status::ok;
}

/* Shared scalar primitive sampler. Parameters and finite support are validated
 * by the caller. CPU review and CUDA factors use identical arithmetic. */
template<class T,class U> RS_HD inline Vec3Value<T> sample_planar_primitive(int kind,const U* p,T s){
    using std::sin;using std::cos;
    Vec3Value<T> q{};
    if(kind==0)q=add(mul(1-s,scalar_cast<T>(load3(p))),mul(s,scalar_cast<T>(load3(p+3))));
    else if(kind==1){const T a=p[4]+s*(p[5]-p[4]);q={p[0]+p[3]*cos(a),p[1]+p[3]*sin(a),p[2]};}
    else if(kind==6){const T a=p[6]+s*(p[7]-p[6]),x=p[3]*cos(a),y=p[4]*sin(a),ca=cos(p[5]),sa=sin(p[5]);q={p[0]+ca*x-sa*y,p[1]+sa*x+ca*y,p[2]};}
    return q;
}

/* Shared one-point primitive forward, with its station tangent. Geometry and
 * placement are frozen inputs; arbitrary display samples are never landmarks. */
RS_HD inline Status sample_entity_point(const BoardCourse* course,double height,int kind,
    const double* p,const double* vertices,size_t nv,double s,Vec3& q,Vec3& ds){
    ds={0,0,0};
    if(!finite_scalar(s))return Status::invalid_argument;
    if(kind!=7&&(s<0||s>1))return Status::outside_support;
    if(kind==7){return course->point(s,p[0]*height,q,ds);}
    if(kind==0||kind==1||kind==6){
        q=sample_planar_primitive(kind,p,s);
        if(kind==0)ds=sub(load3(p+3),load3(p));
        else if(kind==1){double a=p[4]+s*(p[5]-p[4]),rate=p[5]-p[4];
            ds={-p[3]*std::sin(a)*rate,p[3]*std::cos(a)*rate,0};}
        else{double a=p[6]+s*(p[7]-p[6]),rate=p[7]-p[6],ca=std::cos(p[5]),sa=std::sin(p[5]);
            double x=-p[3]*std::sin(a)*rate,y=p[4]*std::cos(a)*rate;
            ds={ca*x-sa*y,sa*x+ca*y,0};}
    }else if(kind==2||kind==4||kind==5){
        const size_t dim=kind==4?3:2,count=nv/dim;const double u=s*(count-1);
        const size_t j=count==1?0:std::min(static_cast<size_t>(std::floor(u)),count-2);
        const size_t k=count==1?0:j+1;const double t=count==1?0:u-j;
        const double x=kind==5?print_convex_interpolate(vertices[j*dim],vertices[k*dim],t):
            (1-t)*vertices[j*dim]+t*vertices[k*dim];
        const double y=kind==5?print_convex_interpolate(vertices[j*dim+1],vertices[k*dim+1],t):
            (1-t)*vertices[j*dim+1]+t*vertices[k*dim+1];
        double dx=(vertices[k*dim]-vertices[j*dim])*(count-1),dy=(vertices[k*dim+1]-vertices[j*dim+1])*(count-1);
        if(kind==4){q={x,y,(1-t)*vertices[j*dim+2]+t*vertices[k*dim+2]};
            ds={dx,dy,(vertices[k*dim+2]-vertices[j*dim+2])*(count-1)};}
        else{
            Vec2 chart{x,y};
            if(kind==5){chart=installed_print_chart(p,x,y);
                const double ca=std::cos(p[2]),sa=std::sin(p[2]);
                double tx=ca*dx*p[3]-sa*dy*p[4],ty=sa*dx*p[3]+ca*dy*p[4];dx=tx;dy=ty;}
            Vec3 tangent{};auto status=course->point(chart.x,chart.y,q,tangent);
            if(status!=Status::ok)return status;ds={tangent.x*dx,tangent.y*dx,dy};
        }
    }else if(kind==3){
        const int segment=static_cast<int>(p[0]);const auto& breaks=course->breaks();double position,rate;
        if(segment==-2){double u=s*8;int j=std::min(static_cast<int>(std::floor(u)),7);
            rate=8*(breaks[j+1]-breaks[j]);position=breaks[j]+(u-j)*(breaks[j+1]-breaks[j]);}
        else if(segment>=0){rate=breaks[segment+1]-breaks[segment];position=breaks[segment]+s*rate;}
        else{rate=course->perimeter();position=s*rate;}
        Vec3 tangent{};auto status=course->point(position,p[1]*height,q,tangent);
        if(status!=Status::ok)return status;ds=mul(rate,tangent);
    }else return Status::invalid_argument;
    return finite(q)&&finite(ds)?Status::ok:Status::nonfinite;
}

/* Exact piecewise-linear unfolded chart of a frozen native print. Knots are
 * original seed/warp break stations, never adaptive drawing samples. */
RS_HD inline Status sample_knot_board(const BoardCourse& course,const double* knots,const double* sz,
    size_t count,double station,Vec3& xyz,Vec3& derivative){
    if(!finite_scalar(station)||count<2)return Status::invalid_argument;
    if(station<knots[0]||station>knots[count-1])return Status::outside_support;
    size_t lo=0,hi=count;
    while(lo<hi){size_t mid=lo+(hi-lo)/2;if(station<knots[mid])hi=mid;else lo=mid+1;}
    size_t j=std::min(lo?lo-1:0,count-2);double width=knots[j+1]-knots[j];
    double t=(station-knots[j])/width;
    double x=print_convex_interpolate(sz[2*j],sz[2*j+2],t),y=print_convex_interpolate(sz[2*j+1],sz[2*j+3],t);
    Vec3 tangent{};auto status=course.point(x,y,xyz,tangent);
    double dx=(sz[2*j+2]-sz[2*j])/width,dy=(sz[2*j+3]-sz[2*j+1])/width;
    derivative={tangent.x*dx,tangent.y*dx,dy};
    return status;
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
        if(!finite_scalar(course->perimeter()))return Status::nonfinite;
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
        Vec3 derivative{};status=sample_entity_point(course.get(),height,kind,p,vertices,nv,s,q,derivative);
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
 * Without option16 the two ends have independent semiaxes. Option16 shares
 * ax/ay at e[0:2] and mirrors goal stations from positive f[1]; f[0],e[2:4]
 * are then inactive storage. Legacy33 keeps its circular replay interpretation.
 * These are estimable geometry parameters, never observations or metre facts.
 */
template<class T> RS_HD inline Status shared_entity_definition(const T* bp,size_t nb,const T* q,size_t nq,
                                      int code,int& kind,T* out,size_t no,unsigned options=0){
    using std::sin;using std::cos;using std::abs;using std::pow;
    if(!bp||nb!=6||!q||(nq!=33&&nq!=35)||!out||!finite_array(q,nq)||(options&~31u))return Status::invalid_argument;
    if(((options&4u)||(options&8u))&&!(options&2u))return Status::invalid_argument;
    if((options&16u)&&(!(options&2u)||nq!=35))return Status::invalid_argument;
    const bool ellipse=(nq==35&&(code==410||code==411));
    if(no!=(ellipse?8u:6u))return Status::invalid_argument;
    const BoardShapeValue<T> board{bp[0],bp[1],bp[2],bp[3],bp[4],bp[5]};
    if(!board.valid())return Status::invalid_argument;
    const T* c=q;const T* lane=q+8;const T* h=q+18;const T* f=q+23;const T* e=q+29;
    const T goals[]={(options&16u)?-f[1]:f[0],f[1]};
    // Four equally spaced rows centred between y=+-board.half_width.
    // c[2] remains a legacy slot, not an independent DOF in this option.
    const T y0=(options&1u)?-1.5*c[3]:c[2];
    if(!(c[0]<c[1]&&c[3]>0&&c[4]>0&&0<c[5]&&c[5]<c[6]&&c[6]<c[7]&&c[7]<1&&lane[1]>0&&h[1]>0&&h[2]>0&&((options&2u)||h[3]<h[4])&&goals[0]<0&&goals[1]>0))return Status::invalid_argument;
    // A1/A4 and A2/A3 share rows and mirror about the transverse x=0 axis.
    // In this declared chart f[2]=common half longitudinal separation and
    // f[4]=common half transverse separation. f[3]/f[5] are inactive storage.
    // Option16 additionally mirrors both goal lines and their attached ends.
    if(options&8u){
        if(!(f[2]>0&&f[4]>0&&f[2]<std::min(-goals[0],goals[1])))return Status::invalid_argument;
    }
    // Current Lane drawing support cannot extend past the left goal line.
    // Bounds are checked, never clamped, and each row retains its own support.
    if(options&4u){
        if(code>=200&&code<204){
            const T end=lane[6+code-200];
            // Each connection passes R and ends strictly before N. Its
            // support stays independent of both R and the Lane right edge.
            if(c[0]<goals[0]||!(c[1]<end&&end<lane[0]))return Status::outside_support;
        }
        if(code>=300&&code<312){
            for(int row=0;row<4;++row)
                if(lane[2+row]<goals[0]||lane[2+row]>lane[0])return Status::outside_support;
        }
    }
    const T yaw=(options&2u)?0.:h[0];
    const T ca=cos(yaw),sa=sin(yaw);
    auto point=[&](T x,T y){return Vec3Value<T>{ca*x-sa*y,sa*x+ca*y,0};};
    auto line=[&](Vec3Value<T> a,Vec3Value<T> b){kind=0;store3(out,a);store3(out+3,b);return Status::ok;};
    auto circle=[&](Vec3Value<T> center,T radius,T a0,T a1){kind=1;store3(out,center);out[3]=radius;out[4]=a0;out[5]=a1;return radius>0?Status::ok:Status::invalid_argument;};
    if(code>=100&&code<132){const int i=(code-100)/4,ring=(code-100)%4;const T ratios[]={c[5],c[6],c[7],1.};return circle({i<4?c[0]:c[1],y0+(i%4)*c[3],0},c[4]*ratios[ring],0,2*pi);}
    if(code>=200&&code<204){const int row=code-200;return line({c[0],y0+row*c[3],0},{lane[6+row],y0+row*c[3],0});}
    if(code>=300&&code<312){const int row=(code-300)/3,edge=(code-300)%3;const T y=y0+row*c[3],lo=y-lane[1]/2,hi=y+lane[1]/2;
        if(edge==0)return line({lane[2+row],hi,0},{lane[0],hi,0});
        if(edge==1)return line({lane[0],hi,0},{lane[0],lo,0});
        return line({lane[0],lo,0},{lane[2+row],lo,0});}
    if(code>=400&&code<=404){
        const T xs[]={0,-h[1],h[1],goals[0],goals[1]};const T x=xs[code-400];
        if(options&2u){SegmentValue<T> carrier{};const auto status=board.vertical_section(x,carrier);
            if(status!=Status::ok)return status;return line(carrier.a,carrier.b);}
        return line(point(x,h[4]),point(x,h[3]));
    }
    if(code==405)return circle({0,0,0},h[2],0,2*pi);
    if(code>=406&&code<=409){
        if(options&8u){const T x[]={f[2],f[2],-f[2],-f[2]},y[]={-f[4],f[4],f[4],-f[4]};
            return circle(point(x[code-406],y[code-406]),h[2],0,2*pi);}
        const T x[]={goals[1]-f[3],goals[1]-f[3],goals[0]+f[2],goals[0]+f[2]},y[]={-f[5],f[5],f[4],-f[4]};
        return circle(point(x[code-406],y[code-406]),h[2],0,2*pi);}

    if(code==410||code==411){const int side=code-410;
        if(nq==35){const int axes=(options&16u)?0:side*2;
            const T ax=e[axes],ay=e[axes+1],a0=side?pi/2:-pi/2;
            if(!(ax>0&&ay>0))return Status::invalid_argument;
            kind=6;store3(out,point(goals[side],0));out[3]=ax;out[4]=ay;out[5]=yaw;out[6]=a0;out[7]=a0+pi;return Status::ok;}
        const T a0=(side?pi/2:-pi/2)+yaw;return circle(point(goals[side],0),e[side],a0,a0+pi);}

    if(code>=420&&code<=423){
        const int side=(code-420)/2;const T sign=side?1.:-1.,ysign=(code%2)?1.:-1.;
        const T goal_half=e[nq==35?4:2],board_half=e[nq==35?5:3];
        if(!(goal_half>0&&board_half>0))return Status::invalid_argument;
        const T y=ysign*board_half;
        auto membership=[&](T x){const Vec3Value<T> p=point(x,y);const T u=std::max(T(0.),(abs(p.x)-(board.half_length-board.rx))/board.rx),v=std::max(T(0.),(abs(p.y)-(board.half_width-board.ry))/board.ry);return pow(u,board.exponent)+pow(v,board.exponent)-1.;};
        if(membership(0)>=0)return Status::outside_support;
        T lo=0,hi=4*std::max(board.half_length,board.half_width);
        if(membership(sign*hi)<=0)return Status::no_intersection;
        for(int i=0;i<60;++i){const T mid=(lo+hi)/2;if(membership(sign*mid)>0)hi=mid;else lo=mid;}
        T x=sign*(lo+hi)/2;const auto hit=point(x,y);
        const T u=std::max(T(0.),(abs(hit.x)-(board.half_length-board.rx))/board.rx);
        const T v=std::max(T(0.),(abs(hit.y)-(board.half_width-board.ry))/board.ry);
        const T dx=board.exponent*(pow(u,board.exponent-1)*(hit.x<0?-1.:1.)*ca/board.rx+pow(v,board.exponent-1)*(hit.y<0?-1.:1.)*sa/board.ry);
        if(abs(dx)<1e-12)return Status::degenerate;
        x-=membership(x)/dx;
        return line(point(goals[side],ysign*goal_half),point(x,y));
    }
    return Status::invalid_argument;
}
} // namespace rinkspace
