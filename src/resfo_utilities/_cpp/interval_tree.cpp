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

std::vector<PillarBoundingBox> create_pillar_bounding_boxes(const float* coord, const resfo::GridDimensions& dims) {
    std::vector<PillarBoundingBox> boxes;
    boxes.reserve(dims.ni * dims.nj);

    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            const int pillar_idx[4] = {
                (i * (dims.nj + 1) + j) *  6,
                (i * (dims.nj + 1) + (j + 1)) * 6,
                ((i + 1) * (dims.nj + 1) + j) * 6,
                ((i + 1) * (dims.nj + 1) + (j + 1)) * 6
            };

            PillarBoundingBox box;
            box.cell_index = {i, j, 0};
            for (int v = 0; v < 4; ++v) {
                int idx = pillar_idx[v];
                box.min_x = std::min(box.min_x, coord[idx]);
                box.max_x = std::max(box.max_x, coord[idx]);

                box.min_x = std::min(box.min_x, coord[idx + 3]);
                box.max_x = std::max(box.max_x, coord[idx + 3]);
            }
            for (int v = 0; v < 4; ++v) {
                int idx = pillar_idx[v];
                box.min_y = std::min(box.min_y, coord[idx + 1]);
                box.max_y = std::max(box.max_y, coord[idx + 1]);

                box.min_y = std::min(box.min_y, coord[idx + 3 + 1]);
                box.max_y = std::max(box.max_y, coord[idx + 3 + 1]);
            }

            boxes.push_back(std::move(box));
        }
    }


    return boxes;
}

PillarTree2D::PillarTree2D(std::vector<PillarBoundingBox> boxes) {
    if (boxes.empty()) return;

    std::vector<float> diagonals;
    diagonals.reserve(boxes.size());
    for (const auto& b : boxes) {
        float dx = b.max_x - b.min_x;
        float dy = b.max_y - b.min_y;
        diagonals.push_back(std::sqrt(dx * dx + dy * dy));
    }
    std::nth_element(diagonals.begin(), diagonals.begin() + diagonals.size() / 2, diagonals.end());
    float median_diag = diagonals[diagonals.size() / 2];
    cell_size = (median_diag > 0.0f) ? median_diag : 1.0f;

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

std::vector<std::pair<int,int>> PillarTree2D::query(float x, float y, float tolerance) const {
    auto [ix0, iy0] = hash_cell(x - tolerance, y - tolerance);
    auto [ix1, iy1] = hash_cell(x + tolerance, y + tolerance);

    std::vector<std::pair<int,int>> results;
    for (int ix = ix0; ix <= ix1; ++ix) {
        for (int iy = iy0; iy <= iy1; ++iy) {
            auto it = table.find({ix, iy});
            if (it == table.end()) continue;
            for (const auto& box : it->second) {
                if (box.min_x - tolerance <= x && x <= box.max_x + tolerance &&
                    box.min_y - tolerance <= y && y <= box.max_y + tolerance) {
                    results.emplace_back(box.cell_index.i, box.cell_index.j);
                }
            }
        }
    }
    return results;
}

// ---------------------------------------------------------------------------
// PillarIntervalTree — true 2D interval tree on PillarBoundingBox
// ---------------------------------------------------------------------------

int PillarIntervalTree::build(std::vector<PillarBoundingBox> boxes) {
    if (boxes.empty()) return -1;

    // Midpoint = median of all interval midpoints to keep the tree balanced.
    std::vector<float> mids;
    mids.reserve(boxes.size());
    for (const auto& b : boxes)
        mids.push_back((b.min_x + b.max_x) * 0.5f);
    std::nth_element(mids.begin(), mids.begin() + mids.size() / 2, mids.end());
    float mid = mids[mids.size() / 2];

    std::vector<PillarBoundingBox> left_boxes, right_boxes, spanning;
    for (auto& b : boxes) {
        if (b.max_x < mid)      left_boxes.push_back(std::move(b));
        else if (b.min_x > mid) right_boxes.push_back(std::move(b));
        else                    spanning.push_back(std::move(b));
    }

    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    nodes_[idx].mid    = mid;
    nodes_[idx].by_min = spanning;
    nodes_[idx].by_max = std::move(spanning);
    std::sort(nodes_[idx].by_min.begin(), nodes_[idx].by_min.end(),
              [](const auto& a, const auto& b) { return a.min_x < b.min_x; });
    std::sort(nodes_[idx].by_max.begin(), nodes_[idx].by_max.end(),
              [](const auto& a, const auto& b) { return a.max_x > b.max_x; });

    nodes_[idx].left  = build(std::move(left_boxes));
    nodes_[idx].right = build(std::move(right_boxes));
    return idx;
}

void PillarIntervalTree::query_node(int idx, float x, float y, float tol,
                                    std::vector<std::pair<int,int>>& results) const {
    if (idx == -1) return;
    const Node& node = nodes_[idx];

    if (x <= node.mid) {
        // All spanning intervals have max_x >= mid >= x, so containment in X reduces
        // to min_x <= x + tol. The list is sorted by min_x ascending: stop at first miss.
        for (const auto& b : node.by_min) {
            if (b.min_x > x + tol) break;
            if (b.min_y - tol <= y && y <= b.max_y + tol)
                results.emplace_back(b.cell_index.i, b.cell_index.j);
        }
        query_node(node.left, x, y, tol, results);
    } else {
        // All spanning intervals have min_x <= mid < x, so containment in X reduces
        // to max_x >= x - tol. The list is sorted by max_x descending: stop at first miss.
        for (const auto& b : node.by_max) {
            if (b.max_x < x - tol) break;
            if (b.min_y - tol <= y && y <= b.max_y + tol)
                results.emplace_back(b.cell_index.i, b.cell_index.j);
        }
        query_node(node.right, x, y, tol, results);
    }
}

PillarIntervalTree::PillarIntervalTree(std::vector<PillarBoundingBox> boxes) {
    if (boxes.empty()) return;

    // Choose the primary tree axis as whichever has the smaller total bbox
    // extent — this minimises the size of spanning sets and therefore the
    // number of candidates that reach the secondary-axis linear scan.
    // For a tilted grid (lean in x), x-extents grow with tilt*depth while
    // y-extents stay unit-width, so building on y is much more efficient.
    float sum_x = 0.f, sum_y = 0.f;
    for (const auto& b : boxes) {
        sum_x += b.max_x - b.min_x;
        sum_y += b.max_y - b.min_y;
    }
    transposed_ = (sum_y < sum_x);
    if (transposed_) {
        for (auto& b : boxes) {
            std::swap(b.min_x, b.min_y);
            std::swap(b.max_x, b.max_y);
        }
    }

    nodes_.reserve(boxes.size());
    build(std::move(boxes));
}

std::vector<std::pair<int,int>> PillarIntervalTree::query(float x, float y, float tolerance) const {
    std::vector<std::pair<int,int>> results;
    if (nodes_.empty()) return results;
    // Swap coordinates to match the axis the tree was built on.
    if (transposed_) std::swap(x, y);
    query_node(0, x, y, tolerance, results);
    return results;
}

} /* namespace resfo */
