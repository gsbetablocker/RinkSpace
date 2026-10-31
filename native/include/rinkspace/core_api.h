#pragma once
#include <stddef.h>
#include <stdint.h>

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
 * 10 row root numerical failure. CUDA/LM entry points are additive below. */
RS_API int rs_decode_d78_camera(const double* chart,size_t nc,double* rig,size_t nr);
/* Additive ABI2 camera evaluation at explicit times (seconds in the spline
 * clock). output has 24 values per time: pan,tilt,log_f,zoom_log_ratio,
 * world-to-camera R(9 row-major), world camera centre C(3),
 * fx,fy,cx,cy,k1,k2,p1,p2. This is an instantaneous camera, not an RS frame;
 * callers must retain readout semantics and evaluate each actual row time.
 * Inspect time_status: failed rows remain NaN. No source pixels or fitting. */
/* Native cubic values and first/second/third time derivatives, N x 4 x 3.
 * Channels are pan, tilt, log(f). At knots jerk follows the selected cell.
 * Diagnostic only: neither parameter values nor solver activity are changed. */
RS_API int rs_evaluate_motion_kinematics(const double* controls,size_t n_controls,
    double first_control_time,double control_dt,const double* times,size_t nt,
    double* output,size_t no,int* status,size_t ns);
/* Exact continuous derivative extrema, output[3 orders,3 channels,3 fields].
 * Fields: signed absolute-extreme value, time, cubic cell (jerk side identity).
 * Supplied lo/hi must lie within the allocated native spline support. */
RS_API int rs_motion_derivative_extrema(const double* controls,size_t n_controls,
    double first_control_time,double control_dt,double lo,double hi,double* output,size_t no);
RS_API int rs_evaluate_motion_camera(const double* rig,size_t nr,
    const double* controls,size_t n_controls,double first_control_time,double control_dt,
    const double* times,size_t nt,double* output,size_t no,int* time_status,size_t ns);
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
/* Material/display evaluation of the SAME deformed/analytic curves, without
 * a scene board or a claim of installed XYZ. Additive ABI2 capabilities: callers
 * must require these symbols explicitly, rather than inventing a display board.
 * Chart parameters remain [s,z,rotation,width,height]; output uv and sz each
 * contain 2*n doubles. Material UV may legitimately lie outside the unit panel.
 * Stations, finite parameters, positive dimensions and shape domains retain the
 * same validation. No XYZ or geometry Jacobian is produced by these entries.
 * Existing installed-path entries continue to require a valid physical board. */
RS_API int rs_sample_print_deformed_material_path(
    const double* parameters,size_t nparameters,const double* vertices,size_t nvertices,
    const double* stations,size_t n,const double* bounds,size_t nbounds,
    const double* shape,size_t nshape,double* uv,size_t nuv,double* sz,size_t nsz,
    int* status,size_t nstatus);
RS_API int rs_sample_print_analytic_material_path(
    const double* parameters,size_t nparameters,int family,int side,int reverse,
    const double* values,size_t nvalues,const double* stations,size_t n,
    double* uv,size_t nuv,double* sz,size_t nsz,int* status,size_t nstatus);
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
 * Optional layout39 appends four independent transverse lane-mark stations.
 * Features320..323 span the corresponding lane's shared side boundaries;
 * omitted path records do not generate marks. All layout35 entities retain
 * exactly the same meaning, including the elliptical end geometry.
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
   * a must be inward of both goal lines.
   * bit16 requires bit2 and layout35: goal x = -/+faceoff[1]; both crease arcs
   * use ends[0:2] as common ax/ay. faceoff[0],ends[2:4] are inactive storage.
   * Both ends, including the upper/lower trapezoid sides, are exact mirrors.
   * Crease centres coincide with goal-line midpoints; kind6 contains only
   * the finite half-ellipse arc, never its closing diameter.
   * bit32 requires layout39: transverse marks320..323 share layout[35]
   * as their x coordinate. Slots36..38 are inactive legacy storage. Each
   * mark connects its own lane's longitudinal sides, parallel to rink Y.
   * bit64 requires ALL bits1..32 and layout146: first39 retain the nominal
   * layout39 meaning; next107 are dimensionless deviations in [-.01,+.01].
   * Dimensions multiply their common nominal value by (1+deviation);
   * positions use declared local nominal sizes, directions use drift/run.
   * Finite typed coordinate groups are declared in ice_tolerance.hpp and
   * domain/ice_tolerance.py. Line/circle/ellipse kinds, shared attachment nodes,
   * ring/row/carrier order and finite supports remain exact. No penalty terms.
   * Zero deviations reproduce option63; old options/layouts remain unchanged. */
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

/* Optional G1 CUDA capability (built with RINKSPACE_ENABLE_CUDA).
 * Inputs are immutable until the optional G3 solver is explicitly attached.
 * Per row: native generated feature code, finite station, relative frame time,
 * observed source UV. Same shared native geometry + implicit RS on CPU/GPU.
 * output[N,11]=XYZ,UV,depth,row,root_equality,contraction,residualUV.
 * All rows are retained; unsupported code is row status 11, invalid results NaN.
 * create metrics[10]=wall_ms,context_ms,h2d_event_ms,owned_device_bytes,
 * free_before_alloc,free_after_alloc,total_device_bytes,device,cc_major,cc_minor.
 * evaluate metrics[3]=wall_ms,kernel_event_ms,d2h_event_ms; backend 0 CPU,1 CUDA.
 */
RS_API int rs_g1_create(const double* board,size_t nb,const double* layout,size_t nl,unsigned options,
    const double* rig,size_t nr,const double* controls,size_t nc,double first,double dt,int height,
    const int* codes,const double* stations,const double* times,const double* observed,size_t count,
    void** handle,double* metrics,size_t nm);
/* Frozen current-native geometry plan, before attaching linearization:
 * paths[P,7]=kind,parameter offset/count,vertex offset/count,knot offset/count.
 * Values packed doubles; codes become row path indices. Kinds 0..7 are native
 * primitives; kind8 is an exact frozen native print chart with original knots.
 * No geometry is inferred from source pixels or independent drawing vertices. */
RS_API int rs_gpu_attach_paths(void* handle,const int* paths,size_t npaths,
    const double* values,size_t nvalues,double* owned_device_bytes);
RS_API int rs_g1_evaluate(void* handle,int backend,double* output,size_t no,int* statuses,size_t ns,
    double* metrics,size_t nm);
/* Compact CUDA diagnostics use the identical row evaluator; invalid rows are +inf. */
RS_API int rs_g1_evaluate_errors(void* handle,double* errors,size_t n);
RS_API int rs_g1_evaluate_rows(void* handle,const int* rows,size_t n,double* values,size_t nv,int* statuses,size_t ns);
RS_API int rs_g3_objective_timings(void* handle,double* values,size_t n);
RS_API void rs_g1_destroy(void* handle);
RS_API const char* rs_g1_last_error(void);

/* G2 attaches a frozen activity/noise plan to the same G1 resident problem.
 * dirs[ncontrols,3,q] maps scaled retained coordinates to existing controls.
 * row_local[N] is -1 for fixed/frozen points, otherwise a shared station ID.
 * Complete groups partition N rows; each original W is rank x (2*members).
 * Native compiles connected local components (never per-row Schur).
 * info[12]=q,nlocal,ng,ncomponent,white_values,H_values,g_values,device_bytes,
 * setup_wall_ms,h2d_event_ms,max_local_component,total_whitened_rank.
 */
RS_API int rs_g2_attach(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,const int* offsets,int ng,const int* members,size_t nmembers,
    const int* ranks,const int* weight_offsets,const double* weights,size_t nweights,double huber_delta,
    double* info,size_t ninfo);
// Raw per-point squared pixel distances. This entry point accepts no W,
// sigma, covariance, robust threshold, source query, or pixel error allowance.
RS_API int rs_g2_attach_pixels(void* handle,int q,const double* dirs,size_t ndirs,const int* row_local,size_t nrows,
    const double* local_scales,size_t nlocal,double* info,size_t ninfo);
// Native coordinate minimization of independent ordinary curve points.
// Fixed corners and genuinely shared material stations are not changed.
RS_API int rs_g3_refine_stations(void* handle,double* info,size_t ninfo);
RS_API int rs_g3_refine_stations_ex(void* handle,int global_search,double* info,size_t ninfo);
/* Metadata only: groups[G,6], components[C,7], packed local IDs[L].
 * Group: member begin/count,rank,W begin,component,white begin.
 * Component: local begin/count,group begin/count,dimension,H begin,g begin. */
RS_API int rs_g2_layout(void* handle,int* groups,size_t ng,int* components,size_t nc,int* locals,size_t nl);
/* Sparse components use their own retained-column list. Packed value offsets
 * are 64-bit; row/station/global parameter identities remain int32.
 * info[6]=CSR nnz,column-map entries,scatter entries,packed J values,
 * maximum row retained columns,maximum component retained columns. */
RS_API int rs_g2_layout64(void* handle,int64_t* groups,size_t ng,int64_t* components,size_t nc,int* locals,size_t nl);
RS_API int rs_g2_sparse_info(void* handle,int64_t* info,size_t ninfo);
RS_API int rs_g2_sparse_layout(void* handle,int64_t* component_columns,size_t nc,int* columns,size_t ncol,int* csr_offsets,size_t no,int* csr_columns,size_t ni);
RS_API int rs_g2_export_sparse_blocks(void* handle,int backend,double* h,size_t nh,double* g,size_t ng,double* shared,size_t nq);
/* Normal production call copies ONLY summary[4] and timings[8].
 * summary=failed rows,quadratic whitened cost,Huber cost,downweighted groups.
 * timings=wall,raw-J,white,loss,block+reduce,summary-D2H,device bytes,D2H bytes.
 * No parameter update, damping, elimination or solve. */
RS_API int rs_g2_linearize(void* handle,int backend,double* summary,size_t ns,double* timings,size_t nt);
/* Explicit, separately requested engineering diagnostics. Never automatic. */
RS_API int rs_g2_export_blocks(void* handle,int backend,double* h,size_t nh,double* g,size_t ng,double* shared,size_t nq);
RS_API int rs_g2_probe_rows(void* handle,int backend,const int* rows,size_t n,double* jac,size_t nj);
RS_API int rs_g2_probe_group(void* handle,int backend,int group,double* white,size_t nw,double* loss,size_t ns);

/* Optional G3: same resident problem and G2 factor plan.
 * options[6]=initial_lambda_relative,initial_lambda(0 auto),acceptance_eta,
 * reduced_relative_tolerance,available_device_budget,available_host_budget.
 * Optional seventh value: 0 identity damping (legacy), 1 diag(H),
 * 2 diag(H) with raw image station metric, 3 max(1,diag(H),diag(J_image^T J_image)).
 * These metrics change only damping; the original W/loss remain unchanged.
 * The latter uses a 64*epsilon*max(diag(H)) numerical floor and a dimensionless
 * initial_lambda_relative; all actual trial factors remain unchanged.
 * finite lower/upper are the intersection of original supports per station.
 * step executes exactly ONE LM candidate, not an iteration loop.
 * State stays native; rejected trials cannot replace accepted state.
 * A pending stop at a stream boundary returns 25 without committing a trial;
 * rs_g4_solve translates that interruption into stop code 3.
 * summary[32] and timings[12] fields documented by the Python thin binding.
 */
RS_API int rs_g3_attach(void* handle,const double* lower,const double* upper,size_t nlocal,const double* options,size_t noptions,double* info,size_t ninfo);
RS_API int rs_g3_step(void* handle,double* summary,size_t nsummary,double* timings,size_t ntimings);
/* G4 owns the continuous loop in native code; callback is telemetry only.
 * tolerances[3]=projected_gradient_inf,accepted_step_inf,relative_cost.
 * stop codes: 0 gradient,1 gradient AND accepted small step/cost,2 floating-point stagnation
 * (2 is NOT convergence), 3=explicit stop at an accepted-state boundary.
 * No implicit iteration/time budget.
 * summary[17] fields are named in the thin Python binding.
 */
typedef void (*rs_lm_progress)(const double* summary,const double* timings,double gradient,double numerical_rejections,void* context);
/* 0: legacy; 1: independent boxes; 2: declared physical tangent cone.
 * Additive diagnostics preserve the existing G4 summary/tolerance ABI. */
RS_API int rs_g4_stationarity_version(void);
// Current-state cache: valid, full vector, objective, exact norm, active faces.
RS_API int rs_g4_stationarity_cached_norm(void* handle,double* values,size_t count);
// As above plus kind: 1 full projection, 2 exact scalar norm, 3 strict lower bound.
// Lower bounds only prove nonconvergence; they never authorize convergence.
RS_API int rs_g4_stationarity_cached_certificate(void* handle,double* values,size_t count);
/* Optional performance telemetry: 19 values (16-field prefix accepted).
 * Last fields: latest damping energy, predicted decrease, damping recoveries.
 * Host counters/event spans; this getter performs no device synchronization. */
RS_API int rs_g4_performance_metrics(void* handle,double* values,size_t count);
/* Read-only current-state diagnostics: 12 doubles, no optimizer step. */
RS_API int rs_g4_stationarity_diagnostics(void* handle,double tolerance,double* summary,size_t nsummary);
/* Last diagnostic: 24 doubles/face, normalized Jacobian[count,q], [raw,p][2,q]. */
RS_API int rs_g4_stationarity_export(void* handle,double* faces,size_t nfaces,double* jacobian,size_t njacobian,double* gradient,size_t ngradient);
RS_API int rs_g4_solve(void* handle,const double* tolerances,size_t ntolerances,
    rs_lm_progress callback,void* context,double* summary,size_t nsummary);
/* Thread-safe stop request. Only this function may run concurrently with solve.
 * Keeps accepted state and damping; calling solve again continues the same handle. */
RS_API int rs_g4_request_stop(void* handle);
/* Safe-boundary numerical-retry telemetry; callback may request stop only. */
typedef void (*rs_lm_numeric_progress)(int rejection_count,double lambda,double nu,void* context);
RS_API int rs_g4_set_numeric_progress(void* handle,rs_lm_numeric_progress callback,void* context);
/* Explicit state/system snapshots only; never part of normal step D2H. */
/* Optional native physical chart: 27 rows (base rig38), or 31 rows
 * (wide rig43), by the SAME retained columns as
 * trajectory directions. Bounds are explicit physical component bounds. */
RS_API int rs_gpu_attach_profiles(void* handle,const int* ids,size_t ni,const double* data,size_t nd);
RS_API int rs_gpu_set_profile_mode(void* handle,int enabled);
RS_API int rs_gpu_profile_roots(void* handle,int backend,double* roots,size_t nr,int* status,size_t ns);
RS_API int rs_gpu_attach_scene(void* handle,const double* values,size_t nv,const int* paths,size_t np,const double* data,size_t nd,int q,const double* directions,size_t nj,const double* lower,const double* upper);
RS_API int rs_gpu_set_scene_chart(void* handle,const int* chart,size_t n);
RS_API int rs_scene_retract(const double* values,size_t n,const int* chart,const double* delta,double* out);
RS_API int rs_gpu_export_scene(void* handle,int trial,double* values,size_t n);
/* Attach multiple complete physical rigs and one camera index per original row,
 * before geometry/linearization. The shared scene and source pixels are unchanged. */
RS_API int rs_gpu_set_camera_groups(void* handle,const double* rigs,size_t nr,const int* ids,size_t count,double* bytes);
RS_API int rs_gpu_set_mount_constraints_for_camera(void* handle,int camera,int side);
RS_API int rs_gpu_set_camera_envelope_for_camera(void* handle,int camera,const double* limits,size_t n);
RS_API int rs_gpu_set_mount_constraints(void* handle,int side);
/* Full continuous camera envelope: forward lo/hi, pupil lo/hi, maximum pan
 * tilt from vertical and tilt-axis deviation from camera right, optical
 * elevation lo/hi, body roll lo/hi. Distances in scene units, angles radians.
 * The complete allocated spline interval includes the rolling-row halo.
 * n=10 preserves the static envelope. n=19 additionally supplies three maxima
 * each for absolute speed, acceleration and jerk, in pan/tilt/log(f) order.
 * Angular units are radians; time units are seconds. Positive infinity means
 * explicitly unset; zero/negative/NaN limits fail instead of freezing motion.
 * n=21 additionally selects exposure-midpoint bounds in spline-grid coordinates
 * (time-first)/dt. They must contain all resident exposures. Every candidate's
 * signed readout expands these bounds to the full sensor-row interval; complete
 * intersecting cells are checked, not unused allocated extrapolation cells.
 * n=19+2*K accepts K ordered, disjoint exposure intervals in spline-grid
 * coordinates. Their readout halos are certified independently; intervals
 * between shots do not impose an invented camera trajectory. */
RS_API int rs_gpu_set_camera_envelope(void* handle,const double* limits,size_t n);
/* Independent diagnostic using the same native envelope. out[4] = minimum
 * signed margin, constraint kind, cubic cell, local u. Optional out[8] appends
 * channel, derivative order, declared limit and signed derivative value.
 * Kinds 7/8/9 check speed/acceleration/jerk exactly over each cubic cell;
 * their margin is 1-abs(derivative)/limit. No numerical fitting. */
RS_API int rs_camera_envelope_evaluate(const double* rig,size_t nr,const double* controls,size_t nc,double dt,
    const double* limits,size_t n,double* out,size_t no);
RS_API int rs_gpu_configure_camera(void* handle,int q,const double* directions,size_t n,const double* lower,const double* upper,size_t nb,double max_readout);
/* Optional computational row-time support for a fixed-rig sparse plan. The
 * pixel plane remains unbounded; roots outside this time interval fail rather
 * than losing motion columns. Must contain the full admissible readout. */
RS_API int rs_gpu_set_projection_time_support(void* handle,double half_seconds);
RS_API int rs_gpu_export_model(void* handle,int trial,double* board,size_t nb,double* layout,size_t nl,double* rig,size_t nr);
RS_API int rs_camera_retract(const double* rig,size_t nr,const double* delta,size_t nd,double* out,size_t no);
RS_API int rs_g3_export_state(void* handle,int trial,double* controls,size_t ncontrols,double* stations,size_t nstations);
/* Immediate last-step diagnostic N x 6: before UV, trial UV, J*actual_step.
 * Call before any subsequent evaluate/linearize/step on the same handle. */
RS_API int rs_g3_probe_changes(void* handle,double* values,size_t n);
RS_API int rs_g3_export_system(void* handle,double* values,size_t nvalues,int* blocked,size_t nblocked);
/* Same snapshot with CSR values instead of dense q x q S. */
RS_API int rs_g3_export_sparse_system(void* handle,double* values,size_t nvalues,int* blocked,size_t nblocked);

#ifdef __cplusplus
}
#endif

/* Moving-camera rig lengths: 38 is the base physical model; 43 appends
 * [selected wide log-f knot, k1/k2/p1/p2 squared-hinge coefficients].
 * The wide knot is the selected model basis, coefficients are four additional
 * free chart coordinates (31 total). CPU projection, CUDA and review share
 * the same formula, implicit derivatives and complete row-time certificate. */
