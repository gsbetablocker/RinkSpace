#include "rinkspace/core_api.h"
#include "rinkspace/geometry.hpp"
#include "rinkspace/projection.hpp"
#include "rinkspace/camera_state.hpp"
#include "rinkspace/entities.hpp"
#include <limits>

using namespace rinkspace;
namespace {
int code(Status s){return static_cast<int>(s);}
void nan_out(double* p,size_t n){if(p)for(size_t i=0;i<n;++i)p[i]=std::numeric_limits<double>::quiet_NaN();}
CurlingLayout curling(const double* p){return {p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]};}
BoardShape board(const double* p){return {p[0],p[1],p[2],p[3],p[4],p[5]};}
Camera camera(const double* p){
    Camera c{};for(int i=0;i<9;++i)c.rotation.a[i]=p[i];c.centre=load3(p+9);
    c.fx=p[12];c.fy=p[13];c.cx=p[14];c.cy=p[15];c.skew=p[16];
    c.k1=p[17];c.k2=p[18];c.p1=p[19];c.p2=p[20];c.k3=p[21];return c;
}
bool counts(size_t n,size_t k,size_t supplied){return n<=std::numeric_limits<size_t>::max()/k&&supplied==n*k;}
bool valid_board(const double* p,size_t n){return p&&n==6&&board(p).valid();}
bool valid_curling(const double* p,size_t n){return p&&n==10&&curling(p).valid();}
bool valid_rows(size_t rows){return rows>0&&rows<=static_cast<size_t>(std::numeric_limits<int>::max()/2);}
}
extern "C" {
int rs_print_square_vertices(const double* p,size_t np,const double* centre,size_t nc,
    double side_over_height,double* vertices,size_t nv){
    if(!vertices||nv!=10)return code(Status::invalid_argument);
    nan_out(vertices,nv);
    try{return code(print_square_vertices(p,np,centre,nc,side_over_height,vertices));}
    catch(...){return code(Status::invalid_argument);}
}
int rs_sample_print_chart(const double* p,size_t np,const double* vertices,size_t nv,
    const double* stations,size_t n,double* sz,size_t nsz,int* status,size_t ns){
    if(!counts(n,2,nsz)||ns!=n)return code(Status::invalid_argument);
    nan_out(sz,nsz);
    try{return code(sample_print_chart(p,np,vertices,nv,stations,n,sz,status));}
    catch(...){return code(Status::invalid_argument);}
}
int rs_sample_entity_path(const double* bp,size_t nb,int kind,
    const double* p,size_t np,const double* vertices,size_t nv,
    const double* stations,size_t n,double* xyz,size_t nxyz,int* status,size_t ns){
    if(!counts(n,3,nxyz)||ns!=n)return code(Status::invalid_argument);
    nan_out(xyz,nxyz);
    try {return code(sample_entity_path(bp,nb,kind,p,np,vertices,nv,stations,n,xyz,status));}
    catch(...){return code(Status::invalid_argument);}
}
int rs_shared_entity_definition(const double* bp,size_t nb,const double* layout,size_t nl,int feature,int* kind,double* parameters,size_t np){
    if(!kind)return code(Status::invalid_argument);
    try{return code(shared_entity_definition(bp,nb,layout,nl,feature,*kind,parameters,np));}
    catch(...){return code(Status::invalid_argument);}
}
int rs_shared_entity_definition_with_options(const double* bp,size_t nb,const double* layout,size_t nl,int feature,unsigned options,int* kind,double* parameters,size_t np){
    if(!kind)return code(Status::invalid_argument);
    try{return code(shared_entity_definition(bp,nb,layout,nl,feature,*kind,parameters,np,options));}
    catch(...){return code(Status::invalid_argument);}
}
int rs_board_course_chart(const double* bp,size_t nb,double* breaks,size_t no){
    if(!valid_board(bp,nb)||!breaks||no!=9)return code(Status::invalid_argument);
    nan_out(breaks,no);
    try{const BoardCourse course(board(bp));for(size_t i=0;i<9;++i)breaks[i]=course.breaks()[i];return 0;}
    catch(...){return code(Status::invalid_argument);}
}
int rs_board_stations_from_legacy(const double* bp,size_t nb,int segment,const double* local,size_t n,double* station,size_t no){
    if(!valid_board(bp,nb)||no!=n||segment< -2||segment>7||(n&&(!local||!station)))return code(Status::invalid_argument);
    nan_out(station,no);
    try{const BoardCourse course(board(bp));const auto& breaks=course.breaks();
        // Validate the whole request before producing partially valid import.
        for(size_t i=0;i<n;++i)if(!std::isfinite(local[i])||local[i]<0||local[i]>1)return code(Status::outside_support);
        for(size_t i=0;i<n;++i){const double u=local[i];
            if(segment==-1)station[i]=u*course.perimeter();
            else if(segment==-2){const double v=8*u;const int j=std::min(int(std::floor(v)),7);station[i]=breaks[j]+(v-j)*(breaks[j+1]-breaks[j]);}
            else station[i]=breaks[segment]+u*(breaks[segment+1]-breaks[segment]);
        }return 0;
    }catch(...){nan_out(station,no);return code(Status::invalid_argument);}
}
int rs_core_abi_version(void){return 2;}
const char* rs_status_message(int s){
    switch(static_cast<Status>(s)){
        case Status::ok:return "ok";case Status::invalid_argument:return "invalid argument";
        case Status::degenerate:return "degenerate geometry";case Status::outside_support:return "outside finite support";
        case Status::no_intersection:return "no intersection";case Status::behind_camera:return "nonpositive camera depth";
        case Status::nonfinite:return "nonfinite result";case Status::not_coplanar:return "geometry is not coplanar";
    }if(s==8)return "outside optical domain";if(s==9)return "row root is not certified unique over full readout";if(s==10)return "predicted row root failed";return "unknown status";
}
int rs_decode_d78_camera(const double* chart,size_t nc,double* rig,size_t nr){
    if(nr==38)nan_out(rig,nr);
    if(nc!=55||nr!=38||!rig||!finite_array(chart,nc))return code(Status::invalid_argument);
    try{return code(decode_d78_camera(chart,rig));}catch(...){nan_out(rig,nr);return code(Status::invalid_argument);}
}
int rs_project_motion(const double* rig,size_t nr,const double* controls,size_t n_controls,
    double first,double dt,int height,const double* times,size_t nt,const double* xyz,size_t count,
    double* output,size_t no,int* status,size_t ns){
    if(nr!=38||!rig||!motion_rig(rig).valid()||height<2||!counts(count,6,no)||nt!=count||ns!=count||
        count>std::numeric_limits<size_t>::max()/3||(count&&(!times||!xyz||!output||!status)))return code(Status::invalid_argument);
    const MotionSpline spline{controls,n_controls,first,dt};if(!spline.valid())return code(Status::invalid_argument);
    nan_out(output,no);const auto physical=motion_rig(rig);
    try{for(size_t i=0;i<count;++i){MotionProjection p{};status[i]=code(project_motion(physical,spline,height,load3(xyz+3*i),times[i],p));
        if(!status[i]){output[6*i]=p.uv.x;output[6*i+1]=p.uv.y;output[6*i+2]=p.depth;output[6*i+3]=p.row;output[6*i+4]=p.equality;output[6*i+5]=p.contraction;}}
    }catch(...){nan_out(output,no);return code(Status::invalid_argument);}return 0;
}
int rs_curling_centres(const double* p,size_t np,size_t rows,double* xyz,size_t nxyz){
    if(!valid_curling(p,np)||!valid_rows(rows)||!xyz||!counts(rows,6,nxyz))return code(Status::invalid_argument);
    nan_out(xyz,nxyz);const auto c=curling(p);
    for(int i=0;i<2*static_cast<int>(rows);++i){
        const Vec3 v=c.centre(i,static_cast<int>(rows));
        if(!finite(v))return code(Status::nonfinite);
        store3(xyz+3*i,v);
    }
    return 0;
}
int rs_curling_ring(const double* p,size_t np,size_t rows,int house,int ring,double theta,double* xyz,size_t nxyz){
    if(nxyz==3)nan_out(xyz,3);
    if(!valid_curling(p,np)||!valid_rows(rows)||!xyz||nxyz!=3||house<0||static_cast<size_t>(house)>=2*rows||ring<0||ring>=4||!std::isfinite(theta))return code(Status::invalid_argument);
    if(ring==3&&p[7]==0)return code(Status::invalid_argument);
    const Vec3 v=curling(p).ring(house,static_cast<int>(rows),ring,theta);if(!finite(v))return code(Status::nonfinite);store3(xyz,v);return 0;
}
int rs_curling_lanes(const double* p,size_t np,size_t rows,const double* left,size_t nl,double* xyz,size_t nxyz){
    if(!valid_curling(p,np)||!valid_rows(rows)||nl!=rows||!finite_array(left,rows)||!xyz||!counts(rows,18,nxyz))return code(Status::invalid_argument);
    nan_out(xyz,nxyz);const auto c=curling(p);for(size_t i=0;i<rows;++i)if(left[i]>=c.right_x)return code(Status::invalid_argument);
    for(size_t i=0;i<rows;++i)for(int e=0;e<3;++e)for(int v=0;v<2;++v){
        const Vec3 point=c.lane_vertex(static_cast<int>(i),e,v,left[i]);
        if(!finite(point))return code(Status::nonfinite);
        store3(xyz+3*(i*6+e*2+v),point);
    }
    return 0;
}
int rs_centre_circle(double radius,double theta,double* xyz,size_t nxyz){
    if(nxyz==3)nan_out(xyz,3);
    if(!xyz||nxyz!=3||!std::isfinite(radius)||radius<=0||!std::isfinite(theta))return code(Status::invalid_argument);
    store3(xyz,{radius*std::cos(theta),radius*std::sin(theta),0});return 0;
}
int rs_line_line(const double* a,size_t na,const double* b,size_t nb,double* xyz,size_t nxyz){
    if(nxyz==3)nan_out(xyz,3);
    if(!a||!b||na!=6||nb!=6||!xyz||nxyz!=3)return code(Status::invalid_argument);
    Vec3 q{};const Status s=line_line({load3(a),load3(a+3)},{load3(b),load3(b+3)},q);if(s==Status::ok)store3(xyz,q);return code(s);
}
int rs_line_circle(const double* line,size_t nl,const double* circle,size_t nc,int branch,double* xyz,size_t nxyz){
    if(nxyz==3)nan_out(xyz,3);
    if(!line||!circle||nl!=6||nc!=4||!xyz||nxyz!=3)return code(Status::invalid_argument);
    Vec3 q{};const Status s=line_circle({load3(line),load3(line+3)},{load3(circle),circle[3]},branch,q);if(s==Status::ok)store3(xyz,q);return code(s);
}
int rs_project_points(const double* p,size_t nc,const double* xyz,size_t n,double* uv,size_t nuv,int* status,size_t ns){
    if(!p||nc!=22||!camera(p).valid()||!counts(n,2,nuv)||ns!=n||n>std::numeric_limits<size_t>::max()/3||
       (n&&(!xyz||!uv||!status)))return code(Status::invalid_argument);
    const auto c=camera(p);nan_out(uv,nuv);
    for(size_t i=0;i<n;++i){Vec2 v{};const auto s=project(c,load3(xyz+3*i),v);status[i]=code(s);if(s==Status::ok){uv[2*i]=v.x;uv[2*i+1]=v.y;}}
    return 0;
}
int rs_project_jacobian(const double* p,size_t nc,const double* xyz,size_t nxyz,double* out,size_t no){
    if(no==8)nan_out(out,8);
    if(!p||nc!=22||!camera(p).valid()||!xyz||nxyz!=3||!out||no!=8)return code(Status::invalid_argument);
    Vec2 v{};double j[6];const auto s=project(camera(p),load3(xyz),v,j);
    if(s==Status::ok){out[0]=v.x;out[1]=v.y;for(int i=0;i<6;++i)out[2+i]=j[i];}return code(s);
}
int rs_pivot_pose(const double* r,size_t nr,const double* p,size_t np,const double* lever,size_t nl,const double* axis,size_t na,double spacing,double* pose,size_t no){
    if(no==12)nan_out(pose,12);
    if(!r||nr!=9||np!=3||nl!=3||na!=3||!finite_array(p,3)||!finite_array(lever,3)||!finite_array(axis,3)||!std::isfinite(spacing)||!pose||no!=12)return code(Status::invalid_argument);
    Mat3 rotation{};for(int i=0;i<9;++i)rotation.a[i]=r[i];if(!valid_rotation(rotation))return code(Status::invalid_argument);
    if(spacing!=0&&std::abs(dot(load3(axis),load3(axis))-1)>1e-6)return code(Status::invalid_argument);
    const Vec3 centre=pivot_centre(rotation,load3(p),load3(lever),load3(axis),spacing);
    if(!finite(centre))return code(Status::nonfinite);
    for(int i=0;i<9;++i)pose[i]=r[i];
    store3(pose+9,centre);return 0;
}
int rs_board_perimeter(const double* p,size_t nb,double* out,size_t no){
    if(no==1)nan_out(out,1);
    if(!valid_board(p,nb)||!out||no!=1)return code(Status::invalid_argument);
    const BoardCourse course(board(p));if(!std::isfinite(course.perimeter()))return code(Status::nonfinite);out[0]=course.perimeter();return 0;
}
int rs_board_point(const double* p,size_t nb,double station,double z,double* out,size_t no){
    if(no==6)nan_out(out,6);
    if(!valid_board(p,nb)||!out||no!=6)return code(Status::invalid_argument);
    const BoardCourse course(board(p));if(!std::isfinite(course.perimeter()))return code(Status::nonfinite);
    Vec3 v{},t{};const auto s=course.point(station,z,v,t);if(s==Status::ok){store3(out,v);store3(out+3,t);}return code(s);
}
int rs_board_points(const double* p,size_t nb,const double* stations,size_t n,const double* z,size_t nz,double* xyz,size_t nxyz,int* status,size_t ns){
    if(!valid_board(p,nb)||nz!=n||ns!=n||!counts(n,3,nxyz)||(n&&(!stations||!z||!xyz||!status)))return code(Status::invalid_argument);
    const BoardCourse course(board(p));if(!std::isfinite(course.perimeter()))return code(Status::nonfinite);nan_out(xyz,nxyz);
    for(size_t i=0;i<n;++i){Vec3 v{},t{};const auto s=course.point(stations[i],z[i],v,t);status[i]=code(s);if(s==Status::ok)store3(xyz+3*i,v);}return 0;
}
int rs_board_ad_polyline(const double* p,size_t nb,double s0,double width,double z0,double height,size_t samples,double* xyz,size_t nxyz){
    if(!valid_board(p,nb)||!xyz||samples<2||!counts(samples,12,nxyz)||!std::isfinite(s0)||!std::isfinite(width)||width<=0||
        !std::isfinite(z0)||!std::isfinite(height)||height<=0)return code(Status::invalid_argument);
    nan_out(xyz,nxyz);const auto shape=board(p);const BoardCourse course(shape);
    if(!std::isfinite(course.perimeter()))return code(Status::nonfinite);
    if(z0<0||z0+height>shape.height||width>course.perimeter())return code(Status::outside_support);
    for(int edge=0;edge<4;++edge)for(size_t i=0;i<samples;++i){
        const double t=static_cast<double>(i)/(samples-1);double s=s0,z=z0;
        if(edge==0)s+=width*t;else if(edge==1){s+=width;z+=height*t;}
        else if(edge==2){s+=width*(1-t);z+=height;}else z+=height*(1-t);
        Vec3 v{},tan{};const auto state=course.point(s,z,v,tan);if(state!=Status::ok)return code(state);
        store3(xyz+3*(static_cast<size_t>(edge)*samples+i),v);
    }return 0;
}
} // extern C
