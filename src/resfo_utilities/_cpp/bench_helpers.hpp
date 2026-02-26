#pragma once

#include <vector>
#include <Eigen/Dense>

// Build a regular (ni+1) x (nj+1) pillar grid with nk layers.
// coord layout: [(ni+1)*(nj+1)] pillars, each 6 floats [x,y,z_top, x,y,z_bot]
// zcorn layout: [ni*nj*nk*8] corner depths, one layer per k
inline void make_grid(int ni, int nj, int nk,
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

// A set of query points scattered across the grid interior.
inline std::vector<Eigen::Vector3d> make_points(int ni, int nj, int nk, int n_pts)
{
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n_pts);
    for (int idx = 0; idx < n_pts; ++idx) {
        float fi = static_cast<float>(idx % ni) + 0.5f;
        float fj = static_cast<float>((idx / ni) % nj) + 0.5f;
        float fk = static_cast<float>((idx / (ni * nj)) % nk) + 0.5f;
        pts.push_back({fi, fj, fk});
    }
    return pts;
}
