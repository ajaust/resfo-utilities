#include <benchmark/benchmark.h>
#include <algorithm>
#include <optional>
#include <vector>

#include "grid_search.hpp"
#include "pillar_interval_tree.hpp"
#include "point_in_cell.hpp"
#include "bench_helpers.hpp"

// ---------------------------------------------------------------------------
// Point sets
// ---------------------------------------------------------------------------

// All points inside the grid.
static std::vector<Eigen::Vector3d> make_points_inside(int ni, int nj, int nk, int n_pts)
{
    return make_points(ni, nj, nk, n_pts);
}

// All points outside the grid (shifted far away).
static std::vector<Eigen::Vector3d> make_points_outside(int ni, int nj, int nk, int n_pts)
{
    auto pts = make_points(ni, nj, nk, n_pts);
    for (auto& p : pts) { p[0] += ni * 10; p[1] += nj * 10; }
    return pts;
}

// 5 % of points outside, the rest inside.
static std::vector<Eigen::Vector3d> make_points_mixed(int ni, int nj, int nk, int n_pts)
{
    auto pts = make_points(ni, nj, nk, n_pts);
    int n_outside = std::max(1, n_pts / 20);
    for (int i = 0; i < n_outside; ++i) {
        pts[i][0] += ni * 10;
        pts[i][1] += nj * 10;
    }
    return pts;
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

using PointSet = std::vector<Eigen::Vector3d>(*)(int, int, int, int);

template<PointSet MakePoints>
static void BM_GridSearch(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);
    const int n_pts = state.range(2);

    std::vector<float> coord, zcorn;
    make_grid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    auto [z_min_it, z_max_it] = std::minmax_element(zcorn.begin(), zcorn.end());
    auto top = resfo::pillar_z_intersection(coord.data(), dims, *z_min_it);
    auto bot = resfo::pillar_z_intersection(coord.data(), dims, *z_max_it);

    auto points = MakePoints(ni, nj, nk, n_pts);

    for (auto _ : state) {
        std::optional<std::pair<int,int>> prev_ij;
        for (const auto& p : points) {
            auto r = resfo::grid_search(p, coord.data(), zcorn.data(), dims, top, bot, 1e-6f, prev_ij);
            if (r) prev_ij = {r->i, r->j};
            else   prev_ij = std::nullopt;
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}

template<PointSet MakePoints>
static void BM_GridSearchPillarIntervalTree(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);
    const int n_pts = state.range(2);

    std::vector<float> coord, zcorn;
    make_grid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    auto pillar_bboxes = resfo::create_pillar_bounding_boxes(coord.data(), dims);
    resfo::PillarIntervalTree tree(std::move(pillar_bboxes));

    auto points = MakePoints(ni, nj, nk, n_pts);

    for (auto _ : state) {
        for (const auto& p : points) {
            auto r = resfo::grid_search_pillar_interval_tree(p, coord.data(), zcorn.data(), dims, 1e-6f, tree);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}



// ---------------------------------------------------------------------------
// Benchmarks for non-regular (tilted / faulted) grids
// ---------------------------------------------------------------------------

using GridFactory = void(*)(int, int, int, std::vector<float>&, std::vector<float>&);

template<GridFactory MakeGrid, PointSet MakePoints>
static void BM_GridSearch_Custom(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);
    const int n_pts = state.range(2);

    std::vector<float> coord, zcorn;
    MakeGrid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    auto [z_min_it, z_max_it] = std::minmax_element(zcorn.begin(), zcorn.end());
    auto top = resfo::pillar_z_intersection(coord.data(), dims, *z_min_it);
    auto bot = resfo::pillar_z_intersection(coord.data(), dims, *z_max_it);

    auto points = MakePoints(ni, nj, nk, n_pts);

    for (auto _ : state) {
        std::optional<std::pair<int,int>> prev_ij;
        for (const auto& p : points) {
            auto r = resfo::grid_search(p, coord.data(), zcorn.data(), dims, top, bot, 1e-6f, prev_ij);
            if (r) prev_ij = {r->i, r->j};
            else   prev_ij = std::nullopt;
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}

template<GridFactory MakeGrid, PointSet MakePoints>
static void BM_GridSearchPillarIntervalTree_Custom(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);
    const int n_pts = state.range(2);

    std::vector<float> coord, zcorn;
    MakeGrid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    auto pillar_bboxes = resfo::create_pillar_bounding_boxes(coord.data(), dims);
    resfo::PillarIntervalTree tree(std::move(pillar_bboxes));

    auto points = MakePoints(ni, nj, nk, n_pts);

    for (auto _ : state) {
        for (const auto& p : points) {
            auto r = resfo::grid_search_pillar_interval_tree(p, coord.data(), zcorn.data(), dims, 1e-6f, tree);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}

#define REGISTER_CUSTOM(BM, grid_fn, pts_fn, n_pts) \
    BENCHMARK_TEMPLATE(BM, grid_fn, pts_fn)->Args({50, 10, n_pts})->Args({50, 50, n_pts})

// Tilted grid (pillars lean 0.5 units per depth unit → heavy bbox overlap)
REGISTER_CUSTOM(BM_GridSearch_Custom,                   make_grid_tilted, make_points_tilted_inside,  100);
REGISTER_CUSTOM(BM_GridSearchPillarIntervalTree_Custom, make_grid_tilted, make_points_tilted_inside,  100);

REGISTER_CUSTOM(BM_GridSearch_Custom,                   make_grid_tilted, make_points_tilted_outside, 100);
REGISTER_CUSTOM(BM_GridSearchPillarIntervalTree_Custom, make_grid_tilted, make_points_tilted_outside, 100);

// Faulted grid (j >= nj/2 thrown down by 5 depth units)
REGISTER_CUSTOM(BM_GridSearch_Custom,                   make_grid_faulted, make_points_faulted_inside,  100);
REGISTER_CUSTOM(BM_GridSearchPillarIntervalTree_Custom, make_grid_faulted, make_points_faulted_inside,  100);

REGISTER_CUSTOM(BM_GridSearch_Custom,                   make_grid_faulted, make_points_faulted_outside, 100);
REGISTER_CUSTOM(BM_GridSearchPillarIntervalTree_Custom, make_grid_faulted, make_points_faulted_outside, 100);



#define REGISTER(BM, suffix, n_pts) \
    BENCHMARK_TEMPLATE(BM, make_points_##suffix)->Args({50, 10, n_pts})->Args({50, 50, n_pts})

// All inside
REGISTER(BM_GridSearch,                   inside,  100);
REGISTER(BM_GridSearchPillarIntervalTree, inside,  100);

// All outside
REGISTER(BM_GridSearch,                   outside, 100);
REGISTER(BM_GridSearchPillarIntervalTree, outside, 100);

// 5% outside
REGISTER(BM_GridSearch,                   mixed,   100);
REGISTER(BM_GridSearchPillarIntervalTree, mixed,   100);

BENCHMARK_MAIN();
