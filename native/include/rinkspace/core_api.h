#pragma once
#include <stddef.h>

#if defined(_WIN32)
# if defined(RINKSPACE_CORE_BUILD)
#  define RS_API __declspec(dllexport)
# else
#  define RS_API __declspec(dllimport)
# endif
#else
# define RS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1: doubles are float64, all matrices row-major. No implicit resizing.
 * 0=OK,1=invalid argument,2=degenerate,3=outside finite support,
 * 4=no intersection,5=behind camera,6=nonfinite output,7=not coplanar.
 * A successful batch return validates its inputs; inspect every point_status.
 * Failed point outputs are NaN, never silently clamped into validity.
 */
/* ABI 2 adds physical time-varying projection; existing ABI 1 functions keep
 * their signatures. Added statuses: 8 optical domain, 9 uncertified row root,
 * 10 row root numerical failure. No nonlinear solver is exposed here. */
RS_API int rs_decode_d78_camera(const double* chart,size_t nc,double* rig,size_t nr);
/* rig[38]: R0(9),pan(3),tilt(3),pivot(3),lever(3),spacing,pupil1,pupil2,
 * fref,radius,cx,cy,eta,cxz,cyz,k10,k11,k20,k21,p1,p2,signed_readout.
 * controls is n_controls*3, fully expanded cardinal cubic (pan,tilt,log_f).
 * times[count] are source PTS differences multiplied by rational time_base.
 * outputs per row: u,v,depth,predicted_row,row_equality,contraction_bound.
 * Inspect point_status; failures retain NaN outputs and are never clamped. */
RS_API int rs_project_motion(const double* rig,size_t nr,const double* controls,size_t n_controls,
    double first_control_time,double control_dt,int image_height,
    const double* times,size_t nt,const double* xyz,size_t count,
    double* output,size_t no,int* point_status,size_t ns);
RS_API int rs_sample_entity_path(const double* board,size_t nboard,int kind,
    const double* parameters,size_t nparameters,const double* vertices,size_t nvertices,
    const double* stations,size_t n,double* xyz,size_t nxyz,int* status,size_t nstatus);
/* Installed print path in the UNWRAPPED board (s,z) chart, using the exact
 * kind5 transform before BoardCourse evaluation. Parameters are
 * [s,z,rotation,width,height]; vertices contain material XY pairs and stations
 * use the finite vertex-index chart [0,1]. Does not wrap or clamp chart values.
 * Inspect each status; failed rows contain NaN pairs. Additive ABI2 capability. */
RS_API int rs_sample_print_chart(const double* parameters,size_t nparameters,
    const double* vertices,size_t nvertices,const double* stations,size_t n,
    double* sz,size_t nsz,int* status,size_t nstatus);
/* Shared monotone material deformation, shape[10] = tx,ty,log_sx,log_sy,
 * three x interval logits, three y interval logits. Four positive normalized
 * intervals per axis; the final logit is fixed zero. bounds[4]=min_u,min_v,
 * max_u,max_v is the group's seed box, including its holes. Values zero replay
 * seed coordinates exactly. Outputs/Jacobians are row-major; Jacobian columns
 * are shape10 or installation5 [s,z,rotation,width,height]. Optional derivative
 * pointers must be null with count zero, or have the exact documented size.
 * This is geometry/linearization only, not a nonlinear or GPU solver.
 * rs_sample_print_deformed_path first samples the ORIGINAL vertex-index station,
 * then deforms material UV, then uses the same installed chart and BoardCourse.
 * Derivative counts: uv/shape and sz/shape=20*n, XYZ/shape=30*n,
 * XYZ/installation=15*n, XYZ/station=3*n. Invalid point outputs remain NaN.
 */
RS_API int rs_deform_print_points(const double* vertices,size_t nvertices,
    const double* bounds,size_t nbounds,const double* shape,size_t nshape,
    double* uv,size_t nuv,double* d_uv_d_shape,size_t njac,int* status,size_t nstatus);
RS_API int rs_sample_print_deformed_path(const double* board,size_t nboard,
    const double* parameters,size_t nparameters,const double* vertices,size_t nvertices,
    const double* stations,size_t n,const double* bounds,size_t nbounds,
    const double* shape,size_t nshape,double* uv,size_t nuv,double* sz,size_t nsz,
    double* xyz,size_t nxyz,double* d_uv_d_shape,size_t n_uv_shape,
    double* d_sz_d_shape,size_t n_sz_shape,double* d_xyz_d_shape,size_t n_xyz_shape,
    double* d_xyz_d_installation,size_t n_xyz_installation,
    double* d_xyz_d_station,size_t n_xyz_station,int* status,size_t nstatus);
/* Typed analytic print structure: family=0 square (values3 centre_u,
 * centre_v,log_side_over_panel_height), family=1 rectangle (values4 centre_u,
 * centre_v,log_width_uv,log_height_uv). Rectangle side=0 top/1 right/2 bottom/
 * 3 left; traversal is clockwise, reverse=1 reverses the source station. Square
 * station traverses all four edges. Derivative layouts match the deformed-path
 * entry above with nshape replaced by 3 or 4. Geometry still uses the same
 * installed chart/BoardCourse, including size dependence of a metric square. */
RS_API int rs_sample_print_analytic_path(const double* board,size_t nboard,
    const double* parameters,size_t nparameters,int family,int side,int reverse,
    const double* values,size_t nvalues,const double* stations,size_t n,
    double* uv,size_t nuv,double* sz,size_t nsz,double* xyz,size_t nxyz,
    double* d_uv_d_values,size_t n_uv_values,double* d_sz_d_values,size_t n_sz_values,
    double* d_xyz_d_values,size_t n_xyz_values,
    double* d_xyz_d_installation,size_t n_xyz_installation,
    double* d_xyz_d_station,size_t n_xyz_station,int* status,size_t nstatus);
/* Standard square in the unfolded board metric, expressed in material UV.
 * centre_uv[2] and positive side_over_panel_height are template parameters.
 * Parameters use the same [s,z,rotation,width,height] chart as kind5.
 * Returns five XY pairs (four exact corners plus repeated first), no source
 * observations and no padding/clamping to [0,1]. Additive ABI2 capability. */
RS_API int rs_print_square_vertices(const double* parameters,size_t nparameters,
    const double* centre_uv,size_t ncentre,double side_over_panel_height,
    double* vertices,size_t nvertices);
/* Shared layout33 remains legacy v2. Layout35 supplies independent blue-area
 * semiaxes: first29 unchanged, then axL,ayL,axR,ayR,trap_goal_half,trap_board_half.
 * Feature410/411 on layout35 returns kind6 and eight ellipse parameters
 * [cx,cy,z,ax,ay,yaw,a0,a1]; other entities return their original six values.
 * rs_sample_entity_path kind6 evaluates that same finite ellipse/arc.
 */
RS_API int rs_shared_entity_definition(const double* board,size_t nboard,const double* layout,size_t nlayout,int feature_code,int* kind,double* parameters,size_t nparameters);
/* Additive options; bit0 derives four-row y0=-1.5*pitch. Unknown bits fail.
 * The legacy entry above always means options=0. */
/* options bit 1: four centred curling rows; bit 2: rink-aligned hockey frame,
 * carriers400..404 terminate at the actual continuous BoardShape intersections.
 * In bit2, old yaw and y-support slots do not control geometry.
 * bit4 requires bit2: Lane left support and connection start may not pass
 * the left goal line. Connection ends must lie strictly between R and N;
 * each support is independent, and invalid support fails without clamping.
 * bit8 requires bit2: A1=(a,-b),A2=(a,b),A3=(-a,b),A4=(-a,-b),
 * where faceoff[2]=a and faceoff[4]=b are independent positive half-spans.
 * faceoff[3]/[5] are inactive storage; goal positions remain independent.
 * a must be inward of both goal lines. */
RS_API int rs_shared_entity_definition_with_options(const double* board,size_t nboard,const double* layout,size_t nlayout,int feature_code,unsigned options,int* kind,double* parameters,size_t nparameters);
/* Continuous board-course chart. Output the nine arc-length segment joins;
 * these joins are topology/display metadata, never bounds on fit stations. */
RS_API int rs_board_course_chart(const double* board,size_t nboard,double* breaks,size_t nbreaks);
/* Fixed relation ice_board_bottom: line[6] is a directed finite segment in
 * z=0; board_height_fraction must be zero. Branch 0/1 orders the full carrier's
 * roots before finite-support admission. Top/unknown depth, missing support,
 * tangent/coincident roots and invalid branches fail. No per-row station. */
RS_API int rs_ice_board_junction(const double* board,size_t nboard,
    const double* line,size_t nline,double board_height_fraction,int branch,
    double* xyz,size_t nxyz);
/* One-time import of valid legacy local 0..1 coordinates into global station.
 * segment=-1: full arc-length course; -2: historical equal-eighth chart;
 * 0..7: legacy finite segment. Optimisation then uses the returned unwrapped
 * global arc-length station directly through entity kind7 or board_points.
 * Source IDs/local values remain provenance; do not retain segment bounds. */
RS_API int rs_board_stations_from_legacy(const double* board,size_t nboard,int segment,const double* local,size_t count,double* global_station,size_t nstation);

RS_API int rs_core_abi_version(void);
RS_API const char* rs_status_message(int status);

/* p[10]: xF,xR,y0,pitch,outer_radius,middle_ratio,inner_ratio,
 * button_ratio,lane_width,lane_right_x. House order F1..Frows,R1..Rrows; row count is event data.
 * y_i=y0+i*pitch is used by BOTH houses and lanes. C0 is (0,0,0).
 * button_ratio=0 means absent/unmodelled: ring index 3 then returns invalid.
 */
RS_API int rs_curling_centres(const double* p,size_t np,size_t rows,double* xyz,size_t nxyz);
RS_API int rs_curling_ring(const double* p,size_t np,size_t rows,int house,int ring,double theta,double* xyz,size_t nxyz);
/* Only three painted edges per lane, ordered lower/right/upper.
 * left_x[rows] bounds visible/rendered support. It is not an observed endpoint.
 * Output: rows lanes * 3 segments * 2 vertices * XYZ = 18*rows doubles.
 */
RS_API int rs_curling_lanes(const double* p,size_t np,size_t rows,const double* left_x,size_t nl,double* xyz,size_t nxyz);
RS_API int rs_centre_circle(double radius,double theta,double* xyz,size_t nxyz);

/* Finite segment = [x0,y0,z0,x1,y1,z1]; ice geometry must be coplanar.
 * circle=[cx,cy,z,r]. Line-circle branch 0/1 means the smaller/larger
 * directed segment parameter, including when the other root is unsupported.
 * Tangency returns DEGENERATE instead of changing branch identity.
 */
RS_API int rs_line_line(const double* a,size_t na,const double* b,size_t nb,double* xyz,size_t nxyz);
RS_API int rs_line_circle(const double* line,size_t nl,const double* circle,size_t nc,int branch,double* xyz,size_t nxyz);

/* camera[22]: R[9],C[3],fx,fy,cx,cy,skew,k1,k2,p1,p2,k3.
 * World-to-camera is R*(X-C). Brown distortion is in normalized coordinates.
 * Static exposure projection only: no motion spline or RS is claimed here.
 */
RS_API int rs_project_points(const double* camera,size_t nc,const double* xyz,size_t count,double* uv,size_t nuv,int* point_status,size_t ns);
/* uv[2] then d(uv)/d(worldXYZ), row-major 2x3. */
RS_API int rs_project_jacobian(const double* camera,size_t nc,const double* xyz,size_t nxyz,double* uv_jac,size_t nout);
/* centre = pivot + R^T*lever + axis_spacing*axis_world.
 * axis_world is supplied by the declared head kinematics, not inferred here.
 * Output R[9],C[3]. This preserves nonzero lever, but is not a full rig model.
 */
RS_API int rs_pivot_pose(const double* rotation,size_t nr,const double* pivot,size_t np,const double* lever,size_t nl,const double* axis_world,size_t na,double axis_spacing,double* pose,size_t no);

/* Continuous polar-superellipse board, same family as the imported d78 model.
 * b[6]: half_length,half_width,corner_rx,corner_ry,exponent,height.
 * Station zero: (-half_length+rx,-half_width), directed along the bottom side.
 * Stations wrap around the closed course. Height must lie in [0,board_height].
 * Batch calls build one immutable arc-length cache, then reuse it for all rows.
 */
RS_API int rs_board_perimeter(const double* b,size_t nb,double* perimeter,size_t no);
/* Output XYZ plus unit tangent dXYZ/dstation. */
RS_API int rs_board_point(const double* b,size_t nb,double station,double z,double* xyz_tangent,size_t no);
RS_API int rs_board_points(const double* b,size_t nb,const double* stations,size_t count,const double* z,size_t nz,double* xyz,size_t nxyz,int* point_status,size_t ns);
/* Unwrapped material rectangle (s0,width,z0,height), four sampled edges.
 * Output 4*samples_per_edge XYZ points; duplicate corners are intentional.
 * Edges remain curved in 3D when crossing a corner or the perimeter seam.
 */
RS_API int rs_board_ad_polyline(const double* b,size_t nb,double s0,double width,double z0,double height,size_t samples_per_edge,double* xyz,size_t nxyz);

#ifdef __cplusplus
}
#endif
