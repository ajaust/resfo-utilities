#include <benchmark/benchmark.h>
#include <vector>

#include "interval_tree.hpp"
#include "grid.hpp"
#include "bench_helpers.hpp"

static void BM_CreateBoundingBoxes(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);

    std::vector<float> coord, zcorn;
    make_grid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    for (auto _ : state) {
        auto boxes = resfo::create_bounding_boxes(coord.data(), zcorn.data(), dims);
        benchmark::DoNotOptimize(boxes);
    }

    state.SetItemsProcessed(state.iterations() * ni * nj * nk);
}

static void BM_CreatePillarBoundingBoxes(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);

    std::vector<float> coord, zcorn;
    make_grid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    for (auto _ : state) {
        auto boxes = resfo::create_pillar_bounding_boxes(coord.data(), dims);
        benchmark::DoNotOptimize(boxes);
    }

    state.SetItemsProcessed(state.iterations() * ni * nj);
}

// ni=nj, nk — vary grid size and layer count
BENCHMARK(BM_CreateBoundingBoxes)
    ->Args({50, 10})
    ->Args({50, 50})
    ->Args({100, 10})
    ->Args({100, 50});

BENCHMARK(BM_CreatePillarBoundingBoxes)
    ->Args({50, 10})
    ->Args({50, 50})
    ->Args({100, 10})
    ->Args({100, 50});

BENCHMARK_MAIN();
