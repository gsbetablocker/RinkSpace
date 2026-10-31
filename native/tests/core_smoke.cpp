#include "rinkspace/core_api.h"
#include "rinkspace/geometry.hpp"
#include "rinkspace/camera_envelope.hpp"
#include "rinkspace/motion_derivatives.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void close(double x,double y,double tol,const char* why){require(std::abs(x-y)<tol,why);}

// A large lens basis is not a large actual lens coefficient. This deliberately
// exercises the same cancellation that an absolute-basis certificate misses,
// while an independent pinhole/rotation oracle has an ordinary unique RS root.
static void cancelled_zoom_rs_certificate(){
    using namespace rinkspace;
    constexpr int height=1080;constexpr double last=height-1,pan_rate=.05,tilt_rate=.08;
    const double logf=std::log(1000.)-1.,focal=std::exp(logf);
    const Vec3 xyz{2.4,.6,0.},tangent{.3,-.2,.1};
    auto rotate=[=](Vec3 q,double t){
        const double a=pan_rate*t,b=tilt_rate*t,px=std::cos(a)*q.x-std::sin(a)*q.y;
        const double py=std::sin(a)*q.x+std::cos(a)*q.y;
        return Vec3{px,std::cos(b)*py-std::sin(b)*q.z,std::sin(b)*py+std::cos(b)*q.z};
    };
    auto pinhole=[&](double time){
        const Vec3 ray=rotate({xyz.x,xyz.y,xyz.z+3.},time);
        return Vec2{960.+focal*ray.x/ray.z,539.5+focal*ray.y/ray.z};
    };
    for(int wide=0;wide<2;++wide){
        MotionRig rig{};rig.R0={{1,0,0,0,1,0,0,0,1}};
        rig.pan={0,0,1};rig.tilt={1,0,0};rig.pivot={0,0,-3};
        rig.fref=1000.;rig.radius=2.;rig.cx=960.;rig.cy=539.5;
        if(!wide){rig.k10=rig.k11=1000.;rig.k20=rig.k21=-1000.;}
        else{
            rig.wide_boundary=std::log(1000.);rig.k10=-1000.;rig.k20=1000.;
            rig.wide_coefficients[0]=1000.;rig.wide_coefficients[1]=-1000.;
            rig.p1=-25.;rig.p2=15.;rig.wide_coefficients[2]=25.;rig.wide_coefficients[3]=-15.;
        }
        double controls[24];
        for(int i=0;i<8;++i){const double t=-2.+i;
            controls[3*i]=pan_rate*t;controls[3*i+1]=tilt_rate*t;controls[3*i+2]=logf;}
        MotionSpline spline{controls,8,-2.,1.};MotionCameraState centre{};
        require(evaluate_motion_camera(rig,spline,0.,centre)==Status::ok,"cancelled lens snapshot");
        require(std::abs(centre.camera.k1)+std::abs(centre.camera.k2)+
            std::abs(centre.camera.p1)+std::abs(centre.camera.p2)<1e-8,"large bases cancel in actual lens");
        require(std::abs(rig.k10)+std::abs(rig.k11)+std::abs(rig.wide_coefficients[0])>1000.,"regression retains large lens bases");
        for(double tau:{-.02,.02}){
            rig.tau=tau;double tiny=0.,whole=0.;
            require(contraction_bound_window(rig,spline,xyz,0.,0.,height,tiny,true)==Status::ok,"zero-window actual-coefficient certificate");
            require(contraction_bound_window(rig,spline,xyz,0.,std::abs(tau)/2,height,whole,true)==Status::ok,"whole-window cancelled-lens certificate");
            require(tiny<=whole+1e-12&&whole<.01,"cancelled certificate tightens without waiving uniqueness");
            MotionProjection root{};
            require(project_motion(rig,spline,height,xyz,0.,root)==Status::ok,"cancelled lens has certified RS root");
            double lo=0.,hi=last;
            for(int i=0;i<80;++i){const double row=(lo+hi)/2;
                if(row-pinhole(tau*(row/last-.5)).y>0)hi=row;else lo=row;}
            const double row=(lo+hi)/2,time=tau*(row/last-.5);const Vec2 expected=pinhole(time);
            close(root.row,row,1e-8,"independent cancelled-lens row root");
            close(root.uv.x,expected.x,1e-8,"cancelled lens pinhole u");
            close(root.uv.y,expected.y,1e-8,"cancelled lens pinhole v");
            require(std::abs(root.equality)<1e-8,"cancelled lens predicted-row equality");
            const Vec3 q{xyz.x,xyz.y,xyz.z+3.},ray=rotate(q,time),dx=rotate(tangent,time);
            const double a=pan_rate*time,b=tilt_rate*time;
            const double px=std::cos(a)*q.x-std::sin(a)*q.y,py=std::sin(a)*q.x+std::cos(a)*q.y;
            const Vec3 velocity{-pan_rate*py,std::cos(b)*pan_rate*px-tilt_rate*ray.z,
                std::sin(b)*pan_rate*px+tilt_rate*ray.y};
            const double du_dt=focal*(velocity.x*ray.z-ray.x*velocity.z)/(ray.z*ray.z);
            const double dv_dt=focal*(velocity.y*ray.z-ray.y*velocity.z)/(ray.z*ray.z);
            const double du_ds=focal*(dx.x*ray.z-ray.x*dx.z)/(ray.z*ray.z);
            const double dv_ds=focal*(dx.y*ray.z-ray.y*dx.z)/(ray.z*ray.z);
            const double k=tau/last,den=1-k*dv_dt;
            require(den>.99&&std::abs(k*dv_dt)<root.contraction,"independent implicit row derivative stays contractive");
            Vec2 fast{};double full[8],weights[4];int cell=0;
            require(motion_station_partial(rig,spline,height,xyz,tangent,0.,root,fast)==Status::ok,"cancelled lens station derivative");
            require(motion_point_partials(rig,spline,height,xyz,tangent,0.,root,full,cell,weights)==Status::ok,"cancelled lens full derivative");
            close(fast.x,du_ds+du_dt*k*dv_ds/den,1e-8,"independent implicit station u");
            close(fast.y,dv_ds/den,1e-8,"independent implicit station v");
            close(fast.x,full[3],1e-10,"cancelled full/contracted station u");
            close(fast.y,full[7],1e-10,"cancelled full/contracted station v");
            const double eps=1e-4;MotionProjection low{},high{};
            require(project_motion(rig,spline,height,sub(xyz,mul(eps,tangent)),0.,low)==Status::ok,"cancelled station difference low");
            require(project_motion(rig,spline,height,add(xyz,mul(eps,tangent)),0.,high)==Status::ok,"cancelled station difference high");
            close(fast.x,(high.uv.x-low.uv.x)/(2*eps),2e-5,"cancelled station finite difference u");
            close(fast.y,(high.uv.y-low.uv.y)/(2*eps),2e-5,"cancelled station finite difference v");
            auto folded=rig;folded.k10-=4.;MotionInstant invalid{};Vec2 uv{};MotionProjection rejected{};
            require(motion_instant(folded,spline,xyz,0.,invalid,uv)==static_cast<Status>(8),"actual nonmonotone lens rejected instantaneously");
            require(project_motion(folded,spline,height,xyz,0.,rejected)==static_cast<Status>(8),"cancellation does not waive true optical fold");
            for(double drift:{-1e-6,1e-6}){
                double moving[24];std::copy(controls,controls+24,moving);
                for(int i=0;i<8;++i)moving[3*i+2]+=drift*(-2.+i);
                MotionSpline zoom{moving,8,-2.,1.};MotionProjection plain{},cached{};MotionProjectionContext context{};
                prepare_motion_projection_context(rig,zoom,0.,context);
                require(project_motion(rig,zoom,height,xyz,0.,plain)==Status::ok,"small signed zoom drift stays certified");
                require(project_motion(rig,zoom,height,xyz,0.,cached,-1.,&context)==Status::ok,"cached signed zoom drift certificate");
                close(cached.uv.x,plain.uv.x,1e-12,"drift cache u");close(cached.uv.y,plain.uv.y,1e-12,"drift cache v");
                close(cached.row,plain.row,1e-12,"drift cache predicted row");
                close(cached.contraction,plain.contraction,1e-12,"drift cache bound");
                require(std::abs(plain.equality)<1e-8,"drift predicted-row equality");
                for(int i=0;i<=16;++i){const double t=-.01+.02*i/16.;double kine[12];
                    require(zoom.kinematics(t,kine)==Status::ok,"drift readout kinematics including endpoints");
                    LocalJet tr[3];for(int c=0;c<3;++c)tr[c]=LocalJet(kine[c],kine[3+c]);
                    MotionCameraStateValue<LocalJet> camera{};Vec2Value<LocalJet> value{};
                    require(motion_camera_from_values(rig,tr,camera)==Status::ok,"drift actual signed optics");
                    require(project_uv(camera.camera,Vec3Value<LocalJet>{xyz.x,xyz.y,xyz.z},value)==Status::ok,"drift analytic time derivative");
                    require(std::abs(tau*value.y.v/last)<=plain.contraction,"window certificate covers actual endpoint/interior slope");
                    MotionInstant before{},after{};Vec2 a_uv{},b_uv{};
                    require(motion_instant(rig,zoom,xyz,t-eps,before,a_uv)==Status::ok,"drift time difference low");
                    require(motion_instant(rig,zoom,xyz,t+eps,after,b_uv)==Status::ok,"drift time difference high");
                    close(value.x.v,(b_uv.x-a_uv.x)/(2*eps),2e-5,"drift time finite difference u");
                    close(value.y.v,(b_uv.y-a_uv.y)/(2*eps),2e-5,"drift time finite difference v");
                }
            }
        }
    }
}

int main(){try{
    cancelled_zoom_rs_certificate();
    {
        rinkspace::MotionRig rig{};rig.R0={{1,0,0,0,1,0,0,0,1}};rig.pan={0,0,1};rig.tilt={1,0,0};rig.pivot={0,0,-3};
        rig.fref=100;rig.radius=2;rig.cx=rig.cy=50;rig.k10=.02;rig.k11=-.001;rig.p1=.0003;
        double controls[24];for(int i=0;i<8;++i){controls[3*i]=-.01+.003*i;controls[3*i+1]=-.02+.006*i;controls[3*i+2]=std::log(100.)+.001*i;}
        rinkspace::MotionSpline spline{controls,8,-2.,1.};rinkspace::Vec3 tangent{.3,.7,.01};
        for(double tau:{0.,.02,-.02})for(double y:{.1,4.}){
            rig.tau=tau;rinkspace::Vec3 xyz{.2,y,0.};rinkspace::MotionProjection root{};
            require(rinkspace::project_motion(rig,spline,101,xyz,.1,root)==rinkspace::Status::ok,"station tangent root");
            rinkspace::MotionProjectionContext context{};rinkspace::prepare_motion_projection_context(rig,spline,.1,context);
            rinkspace::MotionProjection cached{};
            require(rinkspace::project_motion(rig,spline,101,xyz,.1,cached,-1.,&context)==rinkspace::Status::ok,"cached projection root");
            close(cached.uv.x,root.uv.x,1e-12,"cache preserves u");close(cached.uv.y,root.uv.y,1e-12,"cache preserves v");
            close(cached.row,root.row,1e-12,"cache preserves predicted row");close(cached.contraction,root.contraction,1e-12,"cache preserves uniqueness bound");
            for(rinkspace::Vec3 invalid: {rinkspace::Vec3{0,0,-4},rinkspace::Vec3{20,20,0}}){
                rinkspace::MotionProjection a{},b{};
                require(rinkspace::project_motion(rig,spline,101,invalid,.1,a)==rinkspace::project_motion(rig,spline,101,invalid,.1,b,-1.,&context),"cache preserves optical/depth rejection");
            }
            double full[8],weights[4];int cell;rinkspace::Vec2 fast{};
            require(rinkspace::motion_point_partials(rig,spline,101,xyz,tangent,.1,root,full,cell,weights)==rinkspace::Status::ok,"full motion partials");
            require(rinkspace::motion_station_partial(rig,spline,101,xyz,tangent,.1,root,fast)==rinkspace::Status::ok,"contracted station partial");
            close(fast.x,full[3],1e-10,"station partial preserves u");close(fast.y,full[7],1e-10,"station partial preserves v");
            const double eps=1e-4;rinkspace::MotionProjection low{},high{};
            require(rinkspace::project_motion(rig,spline,101,rinkspace::sub(xyz,rinkspace::mul(eps,tangent)),.1,low)==rinkspace::Status::ok,"station difference lower");
            require(rinkspace::project_motion(rig,spline,101,rinkspace::add(xyz,rinkspace::mul(eps,tangent)),.1,high)==rinkspace::Status::ok,"station difference upper");
            close(fast.x,(high.uv.x-low.uv.x)/(2*eps),2e-5,"station derivative finite difference u");
            close(fast.y,(high.uv.y-low.uv.y)/(2*eps),2e-5,"station derivative finite difference v");
        }
    }
    {
        // z(u)=u^3-1.5*u^2+.5625*u has a maximum at .25 and minimum
        // at .75. Moving the cubic must update those two physical faces,
        // without accumulating an obsolete tangent at every old root time.
        rinkspace::MotionRig rig{};rig.fref=1.;rig.pupil1=1.;rig.pan={0,0,1};rig.tilt={1,0,0};
        double controls[12]={0,0,-1.5625,0,0,.5,0,0,-.4375,0,0,1.625};
        rinkspace::MotionSpline spline{controls,4,0.,1.};rinkspace::CameraEnvelope envelope{};
        rinkspace::CameraConstraint first[2],moved[2];
        auto collect=[&](rinkspace::CameraConstraint* faces){
            int count=0;
            rinkspace::camera_cell_constraints(rig,spline,envelope,1,[&](rinkspace::CameraConstraint a){
                if(a.kind==3&&a.sign==1&&a.zoom_branch){require(count<2,"two cubic stationary roots");faces[count++]=a;}
            });
            require(count==2,"both interior zoom extrema retained");
            if(faces[0].u>faces[1].u)std::swap(faces[0],faces[1]);
        };
        collect(first);close(first[0].u,.25,1e-14,"independent zoom maximum");close(first[1].u,.75,1e-14,"independent zoom minimum");
        controls[8]+=.01;collect(moved);
        require(first[0].u!=moved[0].u&&first[1].u!=moved[1].u,"control update moves extrema");
        require(rinkspace::same_camera_constraint_face(first[0],moved[0]),"moving maximum replaces its old tangent");
        require(rinkspace::same_camera_constraint_face(first[1],moved[1]),"moving minimum replaces its old tangent");
        require(!rinkspace::same_camera_constraint_face(first[0],first[1]),"minimum and maximum remain separate");
        auto endpoint=first[0];endpoint.zoom_branch=0;endpoint.u=0.;
        require(!rinkspace::same_camera_constraint_face(first[0],endpoint),"fixed endpoint remains separate");
        auto other=first[0];other.camera=1;
        require(!rinkspace::same_camera_constraint_face(first[0],other),"two physical rigs retain separate faces");
    }
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
    std::cout<<"{\"abi\":"<<rs_core_abi_version()<<",\"geometry_invariants\":\"ok\",\"fixed_branch_identity\":\"ok\",\"point_jacobian_max_abs_error\":"<<maxerr
        <<",\"board_seam_distance\":"<<seam_error<<",\"gpu_or_real_video_fit\":false}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
