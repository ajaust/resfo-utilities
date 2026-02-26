#include <benchmark/benchmark.h>
#include <vector>

#include "interval_tree.hpp"
#include "grid.hpp"

// Build a regular (ni+1) x (nj+1) pillar grid with nk layers.
// coord layout: [(ni+1)*(nj+1)] pillars, each 6 floats [x,y,z_top, x,y,z_bot]
// zcorn layout: [ni*nj*nk*8] corner depths
static void make_grid(int ni, int nj, int nk,
                      std::vector<float>& coord,
                      std::vector<float>& zcorn)
{
    coord.resize((ni + 1) * (nj + 1) * 6);
    for (int i = 0; i <= ni; ++i) {
        for (int j = 0; j <= nj; ++j) {
            int p = (i * (nj + 1) + j) * 6;
            coord[p + 0] = static_cast<float>(i); // x top
            coord[p + 1] = static_cast<float>(j); // y top
            coord[p + 2] = 0.0f;                  // z top
            coord[p + 3] = static_cast<float>(i); // x bot
            coord[p + 4] = static_cast<float>(j); // y bot
            coord[p + 5] = static_cast<float>(nk); // z bot
        }
    }

    zcorn.resize(ni * nj * nk * 8);
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            for (int k = 0; k < nk; ++k) {
                int base = (i * nj * nk + j * nk + k) * 8;
                float z_top = static_cast<float>(k);
                float z_bot = static_cast<float>(k + 1);
                for (int c = 0; c < 4; ++c) zcorn[base + c]     = z_top;
                for (int c = 0; c < 4; ++c) zcorn[base + 4 + c] = z_bot;
            }
        }
    }
}

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
