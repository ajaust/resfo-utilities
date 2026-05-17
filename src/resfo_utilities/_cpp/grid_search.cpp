#include "grid_search.hpp"

#include <unordered_set>
#include <queue>
#include <vector>
#include <functional>

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
    candidates.reserve(columns.size() * dims.nk);

    for (const auto& [ci, cj] : columns) {
        for (int k = 0; k < dims.nk; ++k) {
            int zcorn_idx = (ci * dims.nj * dims.nk + cj * dims.nk + k) * NUM_CORNERS;
            auto [z_min_it, z_max_it] = std::minmax_element(
                zcorn + zcorn_idx, zcorn + zcorn_idx + NUM_CORNERS);
            if (pz >= *z_min_it - 2 * bound_tol && pz <= *z_max_it + 2 * bound_tol) {
                float z_center = (*z_min_it + *z_max_it) * 0.5f;
                candidates.push_back({ci, cj, k, std::abs(z_center - pz)});
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

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const std::vector<float>& top, const std::vector<float>& bot, float tolerance,
    std::optional<std::pair<int, int>> prev_ij) {

    float bound_tol = 20.0*tolerance;

    if (dims.ni <= 0 || dims.nj <= 0) {
        return std::nullopt;
    }

    auto intersection = pillar_z_intersection(coord, dims, p[2]);

    std::priority_queue<QuadNode, std::vector<QuadNode>, std::greater<QuadNode>> queue;
    std::unordered_set<std::pair<int, int>, PairHash> visited;

    if (prev_ij.has_value()) {
        queue.emplace(prev_ij->first, prev_ij->second, p, 1, 1, intersection, dims);
    } else {
        queue.emplace(dims.ni / 2, dims.nj / 2, p, dims.ni / 2, dims.nj / 2, intersection,
                      dims);
    }

    visited.insert({queue.top().i, queue.top().j});

    while (!queue.empty()) {
        QuadNode node = queue.top();
        queue.pop();

        int i = node.i;
        int j = node.j;

        float dist_from_bounds = distance_from_bounds(p, top, bot, i, j, dims);
        if (dist_from_bounds <= 2 * bound_tol) {
            for (int k = 0; k < dims.nk; ++k) {
                int zcorn_idx = (i * dims.nj * dims.nk + j * dims.nk + k) * NUM_CORNERS;
                auto [z_min, z_max] = std::minmax_element(zcorn + zcorn_idx, zcorn + zcorn_idx + NUM_CORNERS);

                if (p[2] >= *z_min - 2 * bound_tol && p[2] <= *z_max + 2 * bound_tol) {
                    if (resfo::point_in_cell(p, i, j, k, coord, zcorn, dims, tolerance)) {
                        return CellIndex{i, j, k};
                    }
                }
            }
        }

        int size_i = node.i_neighbourhood;
        for (int di : {-1 * size_i, -1, 0, 1, size_i}) {
            int nbr_i = i + di;
            if (nbr_i < 0 || nbr_i >= dims.ni) continue;

            int size_j = node.j_neighbourhood;
            for (int dj : {-1 * size_j, -1, 0, 1, size_j}) {
                int nbr_j = j + dj;
                if (nbr_j < 0 || nbr_j >= dims.nj) continue;

                if (visited.find({nbr_i, nbr_j}) == visited.end()) {
                    queue.emplace(nbr_i, nbr_j, p, std::max(size_i / 2, 1), std::max(size_j / 2, 1),
                                  intersection, dims);
                    visited.insert({nbr_i, nbr_j});
                }
            }
        }
    }

    return std::nullopt;
}

std::optional<CellIndex> grid_search_pillar_interval_tree(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    float tolerance, const PillarIntervalTree& tree) {

    float bound_tol = 20.0f * tolerance;

    if (dims.ni <= 0 || dims.nj <= 0 || dims.nk <= 0) {
        return std::nullopt;
    }

    auto columns = tree.query(static_cast<float>(p[0]), static_cast<float>(p[1]), bound_tol);

    const float pz = static_cast<float>(p[2]);
    auto candidates = gather_z_candidates(columns, zcorn, dims, pz, bound_tol);
    return test_candidates(candidates, p, coord, zcorn, dims, tolerance);
}
std::optional<CellIndex> grid_search_hybrid(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const std::vector<float>& top, const std::vector<float>& bot,
    float tolerance, const PillarIntervalTree& tree,
    std::optional<std::pair<int, int>> prev_ij) {

    float bound_tol = 20.0f * tolerance;

    if (dims.ni <= 0 || dims.nj <= 0 || dims.nk <= 0) {
        return std::nullopt;
    }

    auto columns = tree.query(static_cast<float>(p[0]), static_cast<float>(p[1]), bound_tol);

    if ( columns.size() > 4) {
        return grid_search(p, coord, zcorn, dims, top, bot, tolerance, prev_ij);
    }

    const float pz = static_cast<float>(p[2]);
    auto candidates = gather_z_candidates(columns, zcorn, dims, pz, bound_tol);
    return test_candidates(candidates, p, coord, zcorn, dims, tolerance);
}

}  // namespace resfo
