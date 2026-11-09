#include "rinkspace/core_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

struct Bounds {
    std::array<double, 3> low;
    std::array<double, 3> high;

    static Bounds empty() {
        const double inf = std::numeric_limits<double>::infinity();
        return {{inf, inf, inf}, {-inf, -inf, -inf}};
    }

    void include(const Bounds& other) {
        for (int axis = 0; axis != 3; ++axis) {
            low[axis] = std::min(low[axis], other.low[axis]);
            high[axis] = std::max(high[axis], other.high[axis]);
        }
    }

    bool overlaps(const Bounds& other) const {
        return low[0] <= other.high[0] && other.low[0] <= high[0]
            && low[1] <= other.high[1] && other.low[1] <= high[1]
            && low[2] <= other.high[2] && other.low[2] <= high[2];
    }

    double center(int axis) const {
        return 0.5 * low[axis] + 0.5 * high[axis];
    }
};

uint32_t spread_bits(uint32_t value) {
    value = (value | (value << 16)) & 0x030000ffu;
    value = (value | (value << 8)) & 0x0300f00fu;
    value = (value | (value << 4)) & 0x030c30c3u;
    return (value | (value << 2)) & 0x09249249u;
}

struct FaceKey {
    uint32_t morton;
    int32_t face;
};

struct PairSearch {
    static constexpr size_t leaf_size = 8;
    const std::vector<Bounds>& faces;
    const std::vector<FaceKey>& ordered;
    const std::vector<Bounds>& tree;
    size_t leaf_base;
    const int32_t* excluded_rows;
    const int32_t* excluded_faces;
    int32_t* output;
    size_t capacity;
    int64_t count = 0;

    void record(int32_t a, int32_t b) {
        if (!faces[a].overlaps(faces[b])) return;
        if (a > b) std::swap(a, b);
        const auto* first = excluded_faces + excluded_rows[a];
        const auto* last = excluded_faces + excluded_rows[a + 1];
        if (std::binary_search(first, last, b)) return;
        const auto index = static_cast<size_t>(count);
        if (index < capacity) {
            output[2 * index] = a;
            output[2 * index + 1] = b;
        }
        ++count;
    }

    void visit(size_t a, size_t b) {
        if (!tree[a].overlaps(tree[b])) return;
        if (a >= leaf_base && b >= leaf_base) {
            const auto first_a = leaf_size * (a - leaf_base);
            const auto first_b = leaf_size * (b - leaf_base);
            const auto end_a = std::min(first_a + leaf_size, ordered.size());
            const auto end_b = std::min(first_b + leaf_size, ordered.size());
            for (size_t i = first_a; i < end_a; ++i) {
                for (size_t j = a == b ? i + 1 : first_b; j < end_b; ++j) {
                    record(ordered[i].face, ordered[j].face);
                }
            }
        } else if (a == b) {
            visit(2 * a, 2 * a);
            visit(2 * a, 2 * a + 1);
            visit(2 * a + 1, 2 * a + 1);
        } else if (a < leaf_base && (b >= leaf_base || a < b)) {
            visit(2 * a, b);
            visit(2 * a + 1, b);
        } else {
            visit(a, 2 * b);
            visit(a, 2 * b + 1);
        }
    }
};

}  // namespace

int64_t rs_surface_aabb_pairs(
    const double* vertices, size_t vertex_count, const int32_t* faces,
    size_t face_count, const int32_t* excluded_rows, const int32_t* excluded_faces,
    size_t exclusion_count, int32_t* output, size_t capacity) {
    if (face_count < 2) return 0;
    if (!vertices || !faces || !excluded_rows || !excluded_faces || (capacity && !output)
        || face_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || exclusion_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || excluded_rows[0] != 0 || excluded_rows[face_count] != static_cast<int32_t>(exclusion_count)) {
        return -1;
    }
    try {
        std::vector<Bounds> boxes(face_count);
        Bounds centers = Bounds::empty();
        for (size_t f = 0; f < face_count; ++f) {
            if (excluded_rows[f] < 0 || excluded_rows[f] > excluded_rows[f + 1]) return -1;
            auto box = Bounds::empty();
            for (size_t k = 0; k != 3; ++k) {
                const auto vertex = faces[3 * f + k];
                if (vertex < 0 || static_cast<size_t>(vertex) >= vertex_count) return -1;
                const double* point = vertices + 3 * static_cast<size_t>(vertex);
                for (int axis = 0; axis != 3; ++axis) {
                    if (!std::isfinite(point[axis])) return -1;
                    box.low[axis] = std::min(box.low[axis], point[axis]);
                    box.high[axis] = std::max(box.high[axis], point[axis]);
                }
            }
            boxes[f] = box;
            for (int axis = 0; axis != 3; ++axis) {
                const auto mid = box.center(axis);
                centers.low[axis] = std::min(centers.low[axis], mid);
                centers.high[axis] = std::max(centers.high[axis], mid);
            }
        }

        // The Morton key determines only visit order. Intersection tests retain
        // the full double bounds, including exact contact on a box boundary.
        std::array<double, 3> inv_extent{};
        for (int axis = 0; axis != 3; ++axis) {
            const double extent = centers.high[axis] - centers.low[axis];
            if (!std::isfinite(extent)) return -1;
            inv_extent[axis] = 1023.0 / std::max(extent, 1e-30);
        }
        std::vector<FaceKey> order(face_count);
        for (size_t f = 0; f < face_count; ++f) {
            uint32_t morton = 0;
            for (int axis = 0; axis != 3; ++axis) {
                const auto normalized = (boxes[f].center(axis) - centers.low[axis]) * inv_extent[axis];
                const auto cell = static_cast<uint32_t>(std::clamp(normalized, 0.0, 1023.0));
                morton |= spread_bits(cell) << axis;
            }
            order[f] = {morton, static_cast<int32_t>(f)};
        }
        std::sort(order.begin(), order.end(), [](const FaceKey& a, const FaceKey& b) {
            return a.morton < b.morton || (a.morton == b.morton && a.face < b.face);
        });
        size_t leaf_base = 1;
        while (leaf_base * PairSearch::leaf_size < face_count) leaf_base *= 2;
        std::vector<Bounds> tree(2 * leaf_base, Bounds::empty());
        for (size_t f = 0; f < face_count; ++f) {
            tree[leaf_base + f / PairSearch::leaf_size].include(boxes[order[f].face]);
        }
        for (size_t node = leaf_base - 1; node > 0; --node) {
            tree[node] = tree[2 * node];
            tree[node].include(tree[2 * node + 1]);
        }
        PairSearch search{boxes, order, tree, leaf_base, excluded_rows, excluded_faces, output, capacity};
        search.visit(1, 1);
        return search.count;
    } catch (...) {
        return -2;
    }
}
