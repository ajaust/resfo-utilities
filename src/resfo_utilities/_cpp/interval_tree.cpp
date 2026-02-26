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

    //Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<6>> xcoords(coord, dims.ni);
    //Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<6>> ycoords(coord + 1, dims.nj);
//
    //std::array<int, 4> pillar_indices = {0, 0, 0, 0};
    //for (int i = 0; i < dims.ni; ++i) {
    //    for (int j = 0; j < dims.nj; ++j) {
    //        int pillar_idx[4] = {
    //            (i * (dims.nj + 1) + j) ,
    //            (i * (dims.nj + 1) + (j + 1)) ,
    //            ((i + 1) * (dims.nj + 1) + j) ,
    //            ((i + 1) * (dims.nj + 1) + (j + 1))
    //        };
    //        //pillar_indices[0] = i + j * dims.nj;
    //        //pillar_indices[1] = pillar_indices[0] + 1;
    //        //pillar_indices[2] = pillar_indices[0] + dims.nj;
    //        //pillar_indices[3] = pillar_indices[2] + 1;
    //        //pillar_indices[1] = i + j * dims.nj + 1;
    //        //pillar_indices[2] = i + (j + 1) * dims.nj;
    //        //pillar_indices[3] = i + (j + 1) * dims.nj + 1;

    //        resfo::PillarBoundingBox box;
    //        for (const auto& index: pillar_idx) {
    //            box.min_x = std::min(box.min_x, xcoord[index]);
    //            box.max_x = std::max(box.max_x, xcoord[index]);
    //            box.min_y = std::min(box.min_y, ycoord[index+1]);
    //            box.max_y = std::max(box.max_y, ycoord[index+1]);
    //        }

    //        boxes.push_back(std::move(box));
    //    }
    //}

    //auto minmax_coord = [&coord](const int pillar_idx[4]) {
    //    float min_val = std::numeric_limits<float>::max();
    //    float max_val = std::numeric_limits<float>::lowest();

    //    for (int v = 0; v < 4; ++v) {
    //        const int& idx = pillar_idx[v];
    //        min_val = std::min(min_val, coord[idx]);
    //        max_val = std::max(max_val, coord[idx]);

    //        min_val = std::min(min_val, coord[idx + 3]);
    //        max_val = std::max(max_val, coord[idx + 3]);
    //    }

    //    return std::pair<float, float>{min_val, max_val};
    //};

    //auto xy_corners = [&xcoords, &ycoords](const int pillar_idx[4]) {
    //    std::array<float, resfo::NUM_CORNERS > xcorners;
    //    std::array<float, resfo::NUM_CORNERS > ycorners;
    //    for (int v = 0; v < 4; ++v) {
    //        int idx = pillar_idx[v];
    //        xcorners[v] = xcoords[idx];
    //        ycorners[v] = ycoords[idx];
    //    }
    //    for (int v = 0; v < 4; ++v) {
    //        int idx = pillar_idx[v];
    //        xcorners[v + 4] = xcoords[idx + 3];
    //        ycorners[v + 4] = ycoords[idx + 3];
    //    }
    //    return std::pair<
    //            std::array<float, resfo::NUM_CORNERS>,
    //            std::array<float, resfo::NUM_CORNERS>
    //        >{xcorners, ycorners};
    //};

    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            const int pillar_idx[4] = {
                (i * (dims.nj + 1) + j) *  6,
                (i * (dims.nj + 1) + (j + 1)) * 6,
                ((i + 1) * (dims.nj + 1) + j) * 6,
                ((i + 1) * (dims.nj + 1) + (j + 1)) * 6
            };

            //std::array<float, 3> top[4], bot[4];
            //for (int p = 0; p < 4; ++p) {
            //    top[p] = {coord[pillar_idx[p]], coord[pillar_idx[p] + 1], coord[pillar_idx[p] + 2]};
            //    bot[p] = {coord[pillar_idx[p] + 3], coord[pillar_idx[p] + 4], coord[pillar_idx[p] + 5]};
            //}

            //for (int v = 0; v < resfo::NUM_CORNERS; ++v) {
            //    int p_idx = pillar_idx[v];
            //    xcoords[v] = coord[p_idx];
            //    ycoords[v] = coord[p_idx + 1];
            //    xcoords[v + 3] = coord[p_idx + 3];
            //    xcoords[v + 3] = coord[p_idx + 3 + 1];
            //}

            //PillarBoundingBox box;
            //{
            //    auto [min_x_it, max_x_it] = std::minmax_element(xcoords.begin(), xcoords.end());
            //    box.min_x = *min_x_it;
            //    box.max_x = *max_x_it;
            //    auto [min_y_it, max_y_it] = std::minmax_element(ycoords.begin(), ycoords.end());
            //    box.min_y = *min_y_it;
            //    box.max_y = *max_y_it;
            //}


            PillarBoundingBox box;
            //auto [xcorners, ycorners] = xy_corners(pillar_idx);
            // for (int v = 0; v < resfo::NUM_CORNERS; ++v) {
            //    const float& x = xcorners[v];
            //    const float& y = ycorners[v];
            //    box.min_x = std::min(box.min_x, x);
            //    box.max_x = std::max(box.max_x, x);
            //    box.min_y = std::min(box.min_y, y);
            //    box.max_y = std::max(box.max_y, y);
            //}
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

} /* namespace resfo */
