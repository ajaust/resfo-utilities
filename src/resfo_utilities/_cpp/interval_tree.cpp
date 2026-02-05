#include "interval_tree.hpp"

#include <iostream>

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


double IntervalTree2D::median_of(std::vector<double>& values) {
    size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    return values[n / 2];
}

 // Build returns index of the created node in nodes vector
int IntervalTree2D::build(std::vector<BoundingBox>& boxes) {
    if (boxes.empty()) return -1;

    std::vector<double> midpoints;
    midpoints.reserve(boxes.size());
    for (const auto& box : boxes)
        midpoints.push_back((box.min_x + box.max_x) * 0.5);
    double median = median_of(midpoints);

    std::vector<BoundingBox> left_boxes, right_boxes, overlapping_boxes;
    for (const auto& box : boxes) {
        if (box.max_x < median)
            left_boxes.push_back(box);
        else if (box.min_x > median)
            right_boxes.push_back(box);
        else
            overlapping_boxes.push_back(box);
    }

    std::sort(overlapping_boxes.begin(), overlapping_boxes.end(),
              [](const BoundingBox& a, const BoundingBox& b) {
                  return a.min_y < b.min_y;
              });

    int node_index = (int)nodes.size();
    nodes.push_back(Node{median, std::move(overlapping_boxes), -1, -1});

    nodes[node_index].left = this->build(left_boxes);
    nodes[node_index].right = this->build(right_boxes);

    return node_index;
}


void IntervalTree2D::query(int node_index, double x0, double y0, std::vector<CellIndex>& results) const {
    if (node_index == -1) return;

    const Node& node = nodes[node_index];

    if (x0 < node.median_x) {
        this->query(node.left, x0, y0, results);
    } else if (x0 > node.median_x) {
        this->query(node.right, x0, y0, results);
    }

    // Linear scan overlapping intervals (could optimize with binary search)
    for (const auto& box : node.overlapping) {
        if (box.min_y <= y0 && y0 <= box.max_y && box.min_x <= x0 && x0 <= box.max_x) {
            results.push_back(box.cell_index);
        }
    }
}

std::vector<BoundingBox> create_bounding_boxes(const float* coord, const float* zcorn,const resfo::GridDimensions& dims) {
    std::vector<BoundingBox> boxes;
    boxes.reserve(
        dims.ni * dims.nj * dims.nk
    );

    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            for (int k = 0; k < dims.nk; ++k) {
                auto corners = resfo::cell_corners(i, j, k, coord, zcorn, dims);
                {
                    boxes.emplace_back(BoundingBox({i, j, k}, corners));
                }
            }
        }
    }

    return boxes;
}

} /* namespace resfo */
