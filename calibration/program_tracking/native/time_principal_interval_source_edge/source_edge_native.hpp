#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef RS_EDGE_BUILD
#define RS_EDGE_API __declspec(dllexport)
#else
#define RS_EDGE_API __declspec(dllimport)
#endif
#else
#define RS_EDGE_API
#endif

/* All arrays are immutable and owned by caller for the duration of evaluation.
   world_xyz: world_points x 3; observed_uv: observations x 2;
   quality_scales: observations; frozen_segments: ascending original segment ids.
   Camera input is the rig's physical15 vector: centre/100, Rodrigues,
   log fx/fy, principal point and skew/1000, and Brown k1,k2,p1,p2.
   Other Brown coefficients are fixed zero, as projection.from_state specifies. */
struct EdgeFactorDesc {
    const double* world_xyz;
    int64_t world_points;
    const double* observed_uv;
    const double* quality_scales;
    int64_t observations;
    const double* interval_normals;   /* observations x 2; unknown rows may be NaN */
    const double* interval_bounds;    /* observations x 2; unknown rows may be NaN */
    const uint8_t* interval_eligible; /* observations; 0 unknown, 1 source-qualified */
    const int64_t* frozen_segments;
    int64_t frozen_count;
    double sigma;
    int32_t dynamic_candidates;
    int32_t fixed_mount;
    int32_t central_domain;
    int32_t qualified_only;
    double fixed_center_xyz[3];
};

/* mode: 0 = complete residual + 2N x 15 Jacobian; 1 = residual only.
   Residual is quality-scaled, pointwise Huber transformed, before /sigma.
   Jacobian is row-major. Status: 0 success, 1 bad descriptor/argument,
   2 unsupported projection/domain numeric condition, 3 allocation/internal,
   4 source_edge_no_central_segment (only central_domain=1).
   An all-invalid segment set is a successful 1000,0 raw sentinel at every
   observed point, passed through quality and Huber, with zero Jacobian. */
extern "C" RS_EDGE_API int rs_edge_eval(
    const EdgeFactorDesc* factor, const double q15[15],
    double* residual_2N, double* jacobian_2Nx15, int mode) noexcept;

/* One descriptor and one physical15 input per factor; each output pointer
   addresses a caller-owned buffer of factor-specific length. threads=1 or 2.
   statuses[i] gets the same code as rs_edge_eval; return is 0 only if all pass. */
extern "C" RS_EDGE_API int rs_edge_eval_batch(
    const EdgeFactorDesc* factors, const double* q15_list,
    double* const* residuals, double* const* jacobians,
    int64_t count, int threads, int mode, int* statuses) noexcept;
