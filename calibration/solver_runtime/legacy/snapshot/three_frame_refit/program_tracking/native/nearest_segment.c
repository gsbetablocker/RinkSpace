#include <stdint.h>
#include <float.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/*
 * Search finite projected segments for each observed source pixel.
 * All inputs are contiguous doubles except valid (uint8_t). Arithmetic is
 * deliberately ordered like the NumPy implementation. Strict < preserves
 * np.argmin's first-index rule when distances tie exactly.
 */
EXPORT int nearest_segments_strict(
    const double *observed, int64_t observations,
    const double *start, const double *direction,
    const double *length2, const uint8_t *valid, int64_t segments,
    int64_t *selected_index, double *selected_raw_t,
    double *selected_t, double *selected_nearest)
{
    if (!observed || !start || !direction || !length2 || !valid ||
        !selected_index || !selected_raw_t || !selected_t || !selected_nearest ||
        observations < 0 || segments < 1) return 1;
    for (int64_t i = 0; i < observations; ++i) {
        const double ox = observed[2*i];
        const double oy = observed[2*i+1];
        double best = DBL_MAX;
        int64_t best_index = -1;
        double best_raw = 0.0;
        double best_t = 0.0;
        double best_x = 0.0;
        double best_y = 0.0;
        for (int64_t j = 0; j < segments; ++j) {
            if (!valid[j]) continue;
            const double sx = start[2*j];
            const double sy = start[2*j+1];
            const double dx = direction[2*j];
            const double dy = direction[2*j+1];
            const double offset_x = ox-sx;
            const double offset_y = oy-sy;
            double raw = offset_x*dx;
            raw += offset_y*dy;
            const double denom = length2[j] > 1e-9 ? length2[j] : 1e-9;
            raw /= denom;
            if (!isfinite(raw)) return 3;
            double t = raw;
            if (t < 0.0) t = 0.0;
            else if (t > 1.0) t = 1.0;
            const double nx = sx + t*dx;
            const double ny = sy + t*dy;
            const double ex = nx-ox;
            const double ey = ny-oy;
            double squared = ex*ex;
            squared += ey*ey;
            if (!isfinite(squared)) return 3;
            if (squared < best) {
                best = squared;
                best_index = j;
                best_raw = raw;
                best_t = t;
                best_x = nx;
                best_y = ny;
            }
        }
        if (best_index < 0) return 2;
        selected_index[i] = best_index;
        selected_raw_t[i] = best_raw;
        selected_t[i] = best_t;
        selected_nearest[2*i] = best_x;
        selected_nearest[2*i+1] = best_y;
    }
    return 0;
}
