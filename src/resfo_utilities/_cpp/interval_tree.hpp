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
        std::vector<BoundingBox> overlapping; // sorted by min_y
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(double median, std::vector<BoundingBox>&& boxes)
            : median_x(median), overlapping(std::move(boxes)) {}
    };

    std::unique_ptr<Node> root;

    static double median_of(std::vector<double>& values);

    static std::unique_ptr<Node> build(std::vector<BoundingBox>& boxes);

    static void query(const Node* node, double x0, double y0, std::vector<BoundingBox>& results);

public:
    IntervalTree2D() = default;

    explicit IntervalTree2D(std::vector<BoundingBox> boxes) {
        root = build(boxes);
    }

    std::vector<BoundingBox> query(double x0, double y0) const {
        std::vector<BoundingBox> results;
        query(root.get(), x0, y0, results);
        return results;
    }

    std::vector<resfo::CellIndex> query_cells(double x0, double y0) const {
        std::vector<BoundingBox> tmp;
        query(root.get(), x0, y0, tmp);

        std::vector<resfo::CellIndex> results;
        results.reserve(tmp.size());
        for (const auto& box : tmp) {
            results.push_back(box.cell_index);
        }
        return results;
    }
};

} /* namespace resfo */

std::ostream& operator<<(std::ostream& os, const resfo::BoundingBox& box);
