#include "rinkspace/core_api.h"
#include "rinkspace/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void close(double x,double y,double tol,const char* why){require(std::abs(x-y)<tol,why);}
int main(){try{
    const double layout[]={-.31,.29,-.18,.12,.031,2./3.,1./3.,.08,.083,.41};
    const double left[]={-.43,-.41,-.44,-.42};double centres[24],lanes[72];
    require(rs_curling_centres(layout,10,4,centres,24)==0,"layout construction");
    require(rs_curling_lanes(layout,10,4,left,4,lanes,72)==0,"lane construction");
    for(int row=0;row<4;++row){
        close(centres[3*row+1],centres[3*(row+4)+1],1e-15,"F/R paired row");
        close(centres[3*row+1],(lanes[18*row+1]+lanes[18*row+10])/2,1e-15,"house/lane same centreline");
        close(lanes[18*row+3],layout[9],1e-15,"shared right edge");
    }
    double three_rows[18];require(rs_curling_centres(layout,10,3,three_rows,18)==0,"row count is event data");
    close(three_rows[10],three_rows[1],1e-15,"generic row grouping");
    double ring[3];for(int house=0;house<8;++house){
        require(rs_curling_ring(layout,10,4,house,0,.43,ring,3)==0,"ring construction");
        close(std::hypot(ring[0]-centres[3*house],ring[1]-centres[3*house+1]),layout[4],1e-15,"shared radius");
    }
    double no_button[10];std::copy(layout,layout+10,no_button);no_button[7]=0;
    require(rs_curling_ring(no_button,10,4,0,0,0,ring,3)==0,"outer circle with absent button");
    require(rs_curling_ring(no_button,10,4,0,3,0,ring,3)==1,"absent button is not invented");
    const double line[]={-2,0,0,2,0,0},circle[]={0,0,0,1},partial[]={0,0,0,2,0,0},tangent[]={-2,1,0,2,1,0};double point[3];
    require(rs_line_circle(line,6,circle,4,0,point,3)==0,"negative branch");close(point[0],-1,1e-14,"negative branch position");
    require(rs_line_circle(line,6,circle,4,1,point,3)==0,"positive branch");close(point[0],1,1e-14,"positive branch position");
    require(rs_line_circle(partial,6,circle,4,0,point,3)==3,"unsupported branch must not rebind");
    require(std::isnan(point[0]),"invalid intersection must not become a point");
    require(rs_line_circle(partial,6,circle,4,1,point,3)==0,"supported branch identity survives");
    require(rs_line_circle(tangent,6,circle,4,0,point,3)==2,"tangent branch is degenerate");
    const double cross[]={0,-2,0,0,2,0};require(rs_line_line(line,6,cross,6,point,3)==0,"finite intersection");
    const double parallel[]={-2,1,0,2,1,0};require(rs_line_line(line,6,parallel,6,point,3)==2,"parallel reject");
    double cam[]={1,0,0,0,1,0,0,0,1,.1,-.2,-2,1200,1180,960,540,.4,-.07,.012,.0007,-.0004,.001};
    double x[]={.23,-.12,.01},j[8];require(rs_project_jacobian(cam,22,x,3,j,8)==0,"Brown projection");
    double maxerr=0;for(int k=0;k<3;++k){
        double lo[8],hi[8];const double old=x[k],eps=1e-6;x[k]=old-eps;require(rs_project_jacobian(cam,22,x,3,lo,8)==0,"difference lo");
        x[k]=old+eps;require(rs_project_jacobian(cam,22,x,3,hi,8)==0,"difference hi");x[k]=old;
        for(int row=0;row<2;++row)maxerr=std::max(maxerr,std::abs((hi[row]-lo[row])/(2*eps)-j[2+3*row+k]));
    }require(maxerr<1e-6,"analytic point Jacobian");
    const double behind[]={0,0,-3};int stat[1];double uv[2];require(rs_project_points(cam,22,behind,1,uv,2,stat,1)==0,"batch structure");
    require(stat[0]==5&&std::isnan(uv[0]),"behind-camera point must be invalid");
    const double pivot[]={1,2,3},lever[]={.1,.2,.3},axis[]={0,0,1};double pose[12];
    require(rs_pivot_pose(cam,9,pivot,3,lever,3,axis,3,.05,pose,12)==0,"nonzero lever pose");close(pose[11],3.35,1e-15,"lever preserved");
    const double board[]={.5,.25,.108506,.123032,1.8439189,.02};
    const rinkspace::BoardCourse course({board[0],board[1],board[2],board[3],board[4],board[5]});
    double a[6],b[6],seam_error=0;
    for(double s:course.breaks()){
        require(rs_board_point(board,6,s-1e-9,.01,a,6)==0,"left seam sample");
        require(rs_board_point(board,6,s+1e-9,.01,b,6)==0,"right seam sample");
        seam_error=std::max(seam_error,std::hypot(a[0]-b[0],a[1]-b[1]));
        require(std::hypot(a[3]-b[3],a[4]-b[4])<1e-5,"board tangent continuity");
    }require(seam_error<2.1e-9,"board position continuity");
    double ad[12*11];require(rs_board_ad_polyline(board,6,course.perimeter()-.04,.08,.002,.015,11,ad,132)==0,"ad crosses closed seam");
    close(ad[0],ad[3*(4*11-1)],1e-14,"ad closes x");close(ad[1],ad[3*(4*11-1)+1],1e-14,"ad closes y");
    require(rs_board_point(board,6,.1,.03,a,6)==3,"board height outside support");
    std::cout<<"{\"abi\":1,\"geometry_invariants\":\"ok\",\"fixed_branch_identity\":\"ok\",\"point_jacobian_max_abs_error\":"<<maxerr
        <<",\"board_seam_distance\":"<<seam_error<<",\"gpu_or_real_video_fit\":false}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
