#include "grid_search.hpp"

#include <limits>
#include <vector>

#include "point_in_cell.hpp"

namespace resfo {

struct Candidate {
    int i, j, k;
    float z_dist;
};

// Gather candidate cells from the given columns, filtering by z-range.
static std::vector<Candidate> gather_z_candidates(
    const std::vector<std::pair<int, int>>& columns,
    const float* zcorn, const GridDimensions& dims,
    float pz, float bound_tol)
{
    std::vector<Candidate> candidates;
    candidates.reserve(columns.size() * 2);

    for (const auto& [ci, cj] : columns) {
        for (int k = 0; k < dims.nk; ++k) {
            int zcorn_idx = (ci * dims.nj * dims.nk + cj * dims.nk + k) * NUM_CORNERS;
            auto [z_min_it, z_max_it] = std::minmax_element(
                zcorn + zcorn_idx, zcorn + zcorn_idx + NUM_CORNERS);
            if (pz >= *z_min_it - 2 * bound_tol && pz <= *z_max_it + 2 * bound_tol) {
                float z_center = (*z_min_it + *z_max_it) * 0.5f;
                candidates.emplace_back(Candidate{ci, cj, k, std::abs(z_center - pz)});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.z_dist < b.z_dist; });
    return candidates;
}

// Test candidates in z-distance order, returning the first match.
static std::optional<CellIndex> test_candidates(
    const std::vector<Candidate>& candidates,
    const Eigen::Vector3d& p, const float* coord, const float* zcorn,
    const GridDimensions& dims, float tolerance)
{
    for (const auto& c : candidates) {
        if (resfo::point_in_cell(p, c.i, c.j, c.k, coord, zcorn, dims, tolerance)) {
            return CellIndex{c.i, c.j, c.k};
        }
    }
    return std::nullopt;
}

// Filter candidate columns by computing tight (x,y) bounds at the query depth.
// Each column is defined by 4 pillars; we interpolate their (x,y) at query z
// and reject columns where the point is clearly outside.
static std::vector<std::pair<int, int>> filter_columns_at_depth(
    const std::vector<std::pair<int, int>>& columns,
    const float* coord, const GridDimensions& dims,
    float px, float py, float pz, float tol)
{
    std::vector<std::pair<int, int>> filtered;
    filtered.reserve(columns.size());

    for (const auto& [ci, cj] : columns) {
        const int pillar_idx[4] = {
            (ci * (dims.nj + 1) + cj) * 6,
            (ci * (dims.nj + 1) + (cj + 1)) * 6,
            ((ci + 1) * (dims.nj + 1) + cj) * 6,
            ((ci + 1) * (dims.nj + 1) + (cj + 1)) * 6
        };

        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        for (int v = 0; v < 4; ++v) {
            int base = pillar_idx[v];
            float z_top = coord[base + 2];
            float z_bot = coord[base + 5];

            float dz = z_bot - z_top;
            float x_at_z, y_at_z;
            if (std::abs(dz) <= std::numeric_limits<float>::epsilon() * std::max(std::abs(z_top), std::abs(z_bot))) {
                x_at_z = coord[base];
                y_at_z = coord[base + 1];
            } else {
                float t = (pz - z_top) / dz;
                x_at_z = coord[base]     + t * (coord[base + 3] - coord[base]);
                y_at_z = coord[base + 1] + t * (coord[base + 4] - coord[base + 1]);
            }

            min_x = std::min(min_x, x_at_z);
            max_x = std::max(max_x, x_at_z);
            min_y = std::min(min_y, y_at_z);
            max_y = std::max(max_y, y_at_z);
        }

        if (px >= min_x - tol && px <= max_x + tol &&
            py >= min_y - tol && py <= max_y + tol) {
            filtered.emplace_back(ci, cj);
        }
    }

    return filtered;
}

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const ColumnIntervalTree& tree, float tolerance) {

    float bound_tol = 20.0f * tolerance;

    if (dims.ni <= 0 || dims.nj <= 0 || dims.nk <= 0) {
        return std::nullopt;
    }

    auto columns = tree.query(static_cast<float>(p[0]), static_cast<float>(p[1]), bound_tol);
    columns = filter_columns_at_depth(
        columns, coord, dims,
        static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
        bound_tol);

    const float pz = static_cast<float>(p[2]);
    auto candidates = gather_z_candidates(columns, zcorn, dims, pz, bound_tol);
    return test_candidates(candidates, p, coord, zcorn, dims, tolerance);
}

}  // namespace resfo
