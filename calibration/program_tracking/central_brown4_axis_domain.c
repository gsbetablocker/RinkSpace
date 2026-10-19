#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/*
 * Brown4 specialization of physical_projection.safe_project's nine-station
 * symmetric-Jacobian eigenvalue test. The rational denominator is exactly 1
 * when k3..k6 and thin-prism terms are all fixed zero. Caller retains the
 * original OpenCV Rodrigues rays and project_fast UV/J. Compile /fp:strict.
 */
EXPORT int brown4_axis_minimum_eigenvalue(
    const double *normalized_xy, int64_t count,
    const double *brown4, double *minimum_out)
{
    if (!normalized_xy || !brown4 || !minimum_out || count < 0) return 1;
    const double k1 = brown4[0], k2 = brown4[1];
    const double p1 = brown4[2], p2 = brown4[3];
    for (int64_t i = 0; i < count; ++i) {
        const double endpoint_x = normalized_xy[2*i];
        const double endpoint_y = normalized_xy[2*i+1];
        double minimum = INFINITY;
        for (int station = 0; station < 9; ++station) {
            const double alpha = (double)station / 8.0;
            const double x = endpoint_x * alpha;
            const double y = endpoint_y * alpha;
            const double r2 = x*x + y*y;
            const double radial = 1.0 + k1*r2 + k2*r2*r2;
            const double slope = k1 + 2.0*k2*r2;
            const double j11 = radial + 2.0*x*x*slope + 2.0*p1*y + 6.0*p2*x;
            const double j12 = 2.0*x*y*slope + 2.0*p1*x + 2.0*p2*y;
            const double j21 = 2.0*x*y*slope + 2.0*p1*x + 2.0*p2*y;
            const double j22 = radial + 2.0*y*y*slope + 6.0*p1*y + 2.0*p2*x;
            const double gap = j11-j22;
            const double off = j12+j21;
            const double eigen = 0.5*(j11+j22-sqrt(gap*gap+off*off));
            if (eigen != eigen) {
                minimum = NAN;
                break;
            }
            if (eigen < minimum) minimum = eigen;
        }
        minimum_out[i] = minimum;
    }
    return 0;
}
