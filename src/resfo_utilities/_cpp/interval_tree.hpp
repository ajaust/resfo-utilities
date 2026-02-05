#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "grid.hpp"

namespace resfo {

struct BoundingBox {
    resfo::CellIndex cell_index;

    double min_y = std::numeric_limits<double>::max();
    double min_x = std::numeric_limits<double>::max();

    double max_y = std::numeric_limits<double>::lowest();
    double max_x = std::numeric_limits<double>::lowest();

    BoundingBox(resfo::CellIndex cell_index_, const std::vector<double>& corners) : cell_index(std::move(cell_index_)) {
        for (int v = 0; v < resfo::NUM_CORNERS; v += 3) {
            const double& x = corners[v];
            const double& y = corners[v + 1];
            this->min_x = std::min(this->min_x, x);
            this->min_y = std::min(this->min_y, y);

            this->max_x = std::max(this->max_x, x);
            this->max_y = std::max(this->max_y, y);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& box);
};

std::vector<BoundingBox> create_bounding_boxes(
    const float* coord,
    const float* zcorn,
    const resfo::GridDimensions& dims
);

class IntervalTree2D {
private:
    struct Node {
        double median_x;
        std::vector<BoundingBox> overlapping_by_min_y; // sorted by min_y ascending
        std::vector<BoundingBox> overlapping_by_max_y; // sorted by max_y ascending
        int left = -1;
        int right = -1;
    };

    int root_index = -1;
    std::vector<Node> nodes;

    static double median_of(std::vector<double>& values);

    int build(std::vector<BoundingBox>& boxes);

    void query(int node_index, double x0, double y0, double tolerance, std::vector<CellIndex>& results) const;


public:
    IntervalTree2D() = default;

    explicit IntervalTree2D(std::vector<BoundingBox> boxes) {
        root_index = build(boxes);
    }

    std::vector<CellIndex> query(double x0, double y0, double tolerance = 1.e-6) const {
        std::vector<CellIndex> results;
        results.reserve(30);
        this->query(this->root_index, x0, y0, tolerance, results);
        return results;
    }
};

} /* namespace resfo */

std::ostream& operator<<(std::ostream& os, const resfo::BoundingBox& box);
