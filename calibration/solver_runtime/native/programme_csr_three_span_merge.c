#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <omp.h>

/* Every output row has one owner.  The original producers have already
   calculated/scaled every value: this kernel only copies double bytes. */
static int column_present(const int64_t *columns, int64_t count, int64_t value) {
    int64_t lo = 0, hi = count;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (columns[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo < count && columns[lo] == value;
}

#define MAKE_MERGE(SUFFIX, INDEX, INDEX_MAXIMUM) \
__declspec(dllexport) int merge_prepare_##SUFFIX( \
    int64_t rows, int64_t total_columns, int64_t old_nnz, \
    const INDEX *old_pointer, const INDEX *old_columns, \
    int64_t lens_rows, const double *lens, const int64_t *lens_columns, \
    int64_t mount_rows, const double *mount, const int64_t *mount_columns, \
    const int64_t *selected_at, int64_t selected_count, int64_t course_width, \
    const double *course, const int64_t *course_columns, int threads, \
    INDEX *pointer, int64_t *first_overlap) { \
    int64_t row, sum = 0; \
    int invalid = 0; \
    *first_overlap = -1; \
    if (rows < 0 || total_columns < 0 || old_nnz < 0 || threads < 1 || \
        old_pointer[0] != 0 || old_pointer[rows] != old_nnz) return 1; \
    _Pragma("omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)") \
    for (row = 0; row < rows; ++row) { \
        int64_t begin = old_pointer[row], end = old_pointer[row + 1], k; \
        int64_t selected = selected_at[row], count = 0; \
        int overlap = 0; \
        if (begin < 0 || end < begin || end > old_nnz || \
            selected < -1 || selected >= selected_count) { \
            invalid |= 1; pointer[row + 1] = 0; continue; \
        } \
        for (k = begin; k < end; ++k) { \
            int64_t col = old_columns[k]; \
            if (col < 0 || col >= total_columns || \
                (k > begin && col < old_columns[k - 1])) invalid |= 1; \
            if (selected >= 0 && column_present(course_columns, course_width, col)) overlap = 1; \
        } \
        count = end - begin; \
        for (k = 0; k < 3; ++k) { \
            if (row < lens_rows && lens[3 * row + k] != 0.0) { \
                ++count; \
                if (selected >= 0 && column_present(course_columns, course_width, lens_columns[k])) overlap = 1; \
            } \
            if (row < mount_rows && mount[3 * row + k] != 0.0) { \
                ++count; \
                if (selected >= 0 && column_present(course_columns, course_width, mount_columns[k])) overlap = 1; \
            } \
        } \
        if (selected >= 0) \
            for (k = 0; k < course_width; ++k) \
                if (course[selected * course_width + k] != 0.0) ++count; \
        if (count > INDEX_MAXIMUM) invalid |= 1; \
        pointer[row + 1] = (INDEX)count; \
        if (overlap) { \
            _Pragma("omp critical(rinkspace_csr_merge_overlap)") \
            { \
                if (*first_overlap < 0 || row < *first_overlap) *first_overlap = row; \
            } \
        } \
    } \
    if (invalid) return 1; \
    if (*first_overlap >= 0) return 2; \
    pointer[0] = 0; \
    for (row = 0; row < rows; ++row) { \
        int64_t count = pointer[row + 1]; \
        if (count > INDEX_MAXIMUM - sum) return 3; \
        sum += count; pointer[row + 1] = (INDEX)sum; \
    } \
    return 0; \
} \
__declspec(dllexport) int merge_fill_##SUFFIX( \
    int64_t rows, const INDEX *old_pointer, const INDEX *old_columns, const double *old_values, \
    int64_t lens_rows, const double *lens, const int64_t *lens_columns, \
    int64_t mount_rows, const double *mount, const int64_t *mount_columns, \
    const int64_t *selected_at, int64_t course_width, const double *course, const int64_t *course_columns, \
    int threads, const INDEX *pointer, INDEX *columns, double *values) { \
    int64_t row; \
    int invalid = 0; \
    _Pragma("omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)") \
    for (row = 0; row < rows; ++row) { \
        int64_t old = old_pointer[row], end = old_pointer[row + 1], dest = pointer[row]; \
        int64_t selected = selected_at[row], li = 0, mi = 0; \
        int64_t ci = selected >= 0 ? 0 : course_width; \
        const double *cv = selected >= 0 ? course + selected * course_width : NULL; \
        while (1) { \
            int64_t col = INT64_MAX; \
            const double *value = NULL; \
            int source = -1; \
            while (li < 3 && (row >= lens_rows || lens[3 * row + li] == 0.0)) ++li; \
            while (mi < 3 && (row >= mount_rows || mount[3 * row + mi] == 0.0)) ++mi; \
            while (ci < course_width && (!cv || cv[ci] == 0.0)) ++ci; \
            if (old < end) { col = old_columns[old]; value = old_values + old; source = 0; } \
            if (li < 3 && lens_columns[li] < col) { col = lens_columns[li]; value = lens + 3 * row + li; source = 1; } \
            if (mi < 3 && mount_columns[mi] < col) { col = mount_columns[mi]; value = mount + 3 * row + mi; source = 2; } \
            if (ci < course_width && course_columns[ci] < col) { col = course_columns[ci]; value = cv + ci; source = 3; } \
            if (source < 0) break; \
            if (dest >= pointer[row + 1]) { invalid |= 1; break; } \
            columns[dest] = (INDEX)col; \
            memcpy(values + dest, value, sizeof(double)); \
            ++dest; \
            if (source == 0) ++old; else if (source == 1) ++li; \
            else if (source == 2) ++mi; else ++ci; \
        } \
        if (dest != pointer[row + 1]) invalid |= 1; \
    } \
    return invalid; \
}

MAKE_MERGE(i32, int32_t, INT32_MAX)
MAKE_MERGE(i64, int64_t, INT64_MAX)
