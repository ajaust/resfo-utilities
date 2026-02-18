#include "interval_tree.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

#include "grid.hpp"
#include "point_in_cell.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace resfo {


std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
    os << "BoundingBox(min_x=" << box.min_x << ", min_y=" << box.min_y
       << ", max_x=" << box.max_x << ", max_y=" << box.max_y << ")";
    return os;
}


IntervalTree2D::IntervalTree2D(std::vector<BoundingBox> boxes) {
    if (boxes.empty()) return;

    // Choose cell size as the median bounding-box diagonal so that most boxes
    // span only a handful of hash cells while still providing good locality.
    std::vector<double> diagonals;
    diagonals.reserve(boxes.size());
    for (const auto& b : boxes) {
        double dx = b.max_x - b.min_x;
        double dy = b.max_y - b.min_y;
        diagonals.push_back(std::sqrt(dx * dx + dy * dy));
    }
    std::nth_element(diagonals.begin(), diagonals.begin() + diagonals.size() / 2, diagonals.end());
    double median_diag = diagonals[diagonals.size() / 2];
    cell_size = (median_diag > 0.0) ? median_diag : 1.0;

    for (const auto& box : boxes) {
        auto [ix0, iy0] = hash_cell(box.min_x, box.min_y);
        auto [ix1, iy1] = hash_cell(box.max_x, box.max_y);
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                table[{ix, iy}].push_back(box);
            }
        }
    }
}

std::vector<CellIndex> IntervalTree2D::query(double x0, double y0, double tolerance) const {
    auto [ix0, iy0] = hash_cell(x0 - tolerance, y0 - tolerance);
    auto [ix1, iy1] = hash_cell(x0 + tolerance, y0 + tolerance);

    // Collect unique candidates from all overlapping hash cells.
    std::vector<CellIndex> results;
    results.reserve(30);

    // Use a flat visited set keyed on linearised (i,j,k) to avoid duplicates.
    struct CellHash {
        std::size_t operator()(const std::tuple<int,int,int>& t) const noexcept {
            auto h = std::hash<int>{};
            return h(std::get<0>(t)) ^ (h(std::get<1>(t)) << 11) ^ (h(std::get<2>(t)) << 22);
        }
    };
    std::unordered_set<std::tuple<int,int,int>, CellHash> seen;

    for (int ix = ix0; ix <= ix1; ++ix) {
        for (int iy = iy0; iy <= iy1; ++iy) {
            auto it = table.find({ix, iy});
            if (it == table.end()) continue;
            for (const auto& box : it->second) {
                if (box.min_x - tolerance <= x0 && x0 <= box.max_x + tolerance &&
                    box.min_y - tolerance <= y0 && y0 <= box.max_y + tolerance) {
                    auto key = std::make_tuple(box.cell_index.i, box.cell_index.j, box.cell_index.k);
                    if (seen.insert(key).second) {
                        results.push_back(box.cell_index);
                    }
                }
            }
        }
    }

    return results;
}

std::vector<BoundingBox> create_bounding_boxes(const float* coord, const float* zcorn, const resfo::GridDimensions& dims) {
    std::vector<BoundingBox> boxes;
    boxes.reserve(dims.ni * dims.nj * dims.nk);

    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            for (int k = 0; k < dims.nk; ++k) {
                auto corners = resfo::cell_corners(i, j, k, coord, zcorn, dims);
                boxes.emplace_back(BoundingBox({i, j, k}, corners));
            }
        }
    }

    return boxes;
}

} /* namespace resfo */
