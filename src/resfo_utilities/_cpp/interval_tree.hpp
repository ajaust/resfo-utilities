#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
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

// Spatial hash replacing the previous IntervalTree2D implementation.
// Bounding boxes are inserted into all hash cells they overlap; queries
// collect candidates from the relevant hash cells and filter by the exact bbox.
class IntervalTree2D {
private:
    struct PairHash {
        std::size_t operator()(const std::pair<int,int>& p) const noexcept {
            return std::hash<long long>()(
                (static_cast<long long>(p.first) << 32) ^ static_cast<unsigned int>(p.second));
        }
    };

    double cell_size = 1.0;
    std::unordered_map<std::pair<int,int>, std::vector<BoundingBox>, PairHash> table;

    std::pair<int,int> hash_cell(double x, double y) const {
        return {static_cast<int>(std::floor(x / cell_size)),
                static_cast<int>(std::floor(y / cell_size))};
    }

public:
    IntervalTree2D() = default;

    explicit IntervalTree2D(std::vector<BoundingBox> boxes);

    std::vector<CellIndex> query(double x0, double y0, double tolerance = 1.e-6) const;
};

} /* namespace resfo */

std::ostream& operator<<(std::ostream& os, const resfo::BoundingBox& box);
