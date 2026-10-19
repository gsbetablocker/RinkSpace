#include <stdint.h>
#include <stddef.h>
#include <omp.h>

/* Only row-local ownership is parallel. Input CSR arrays remain read-only.
   The caller validates the original source/shared-padding layout. */
#define MAKE_SCATTER(NAME, INDEX) \
__declspec(dllexport) int NAME(int64_t rows, int64_t source_rows, int64_t geo_start, \
    int64_t geo_rows, const INDEX *lp, const INDEX *li, const double *lv, \
    const INDEX *gp, const INDEX *gi, const double *gv, \
    const int64_t *geo_columns, const int64_t *xyz, int64_t view_start, \
    double scale, int threads, INDEX *op, INDEX *oi, double *ov, int *sorted_output) { \
    int64_t row; \
    int all_sorted = 1; \
    for (row = 0; row <= rows; ++row) { \
        int64_t removed = row < source_rows ? row : source_rows; \
        int64_t inserted = 0; \
        if (row > geo_start) { \
            int64_t gr = row - geo_start; \
            inserted = gp[gr < geo_rows ? gr : geo_rows]; \
        } \
        op[row] = (INDEX)((int64_t)lp[row] - removed + inserted); \
    } \
    _Pragma("omp parallel for schedule(static) num_threads(threads) reduction(&:all_sorted)") \
    for (row = 0; row < rows; ++row) { \
        int64_t start = lp[row], stop = lp[row + 1]; \
        int64_t georow = row - geo_start; \
        int has_geo = georow >= 0 && georow < geo_rows; \
        int64_t head_count = 0, k; \
        if (has_geo) { \
            for (k = gp[georow]; k < gp[georow + 1]; ++k) \
                if (gi[k] < 5) ++head_count; \
        } \
        for (k = start; k < stop; ++k) { \
            int64_t col = li[k]; \
            int64_t pos; \
            double value; \
            if (col == 3) continue; \
            pos = (int64_t)op[row] + (k - start) - (row < source_rows && col > 3) + head_count; \
            value = lv[k]; \
            if (col < 3) { value *= scale; col = xyz[col]; } \
            else col += view_start - 4; \
            oi[pos] = (INDEX)col; ov[pos] = value; \
        } \
        if (has_geo) { \
            int64_t gstart = gp[georow]; \
            for (k = gstart; k < gp[georow + 1]; ++k) { \
                int64_t pos = (int64_t)op[row] + (k - gstart); \
                if (gi[k] >= 5) pos += stop - start - 1; \
                oi[pos] = (INDEX)geo_columns[gi[k]]; ov[pos] = gv[k]; \
            } \
        } \
        for (k = op[row] + 1; k < op[row + 1]; ++k) \
            if (oi[k] < oi[k - 1]) { all_sorted = 0; break; } \
    } \
    *sorted_output = all_sorted; \
    return 0; \
}

MAKE_SCATTER(scatter_i32, int32_t)
MAKE_SCATTER(scatter_i64, int64_t)
