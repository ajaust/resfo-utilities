#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "grid.hpp"

#include <Eigen/Dense>

namespace resfo {

struct PillarBoundingBox {
    resfo::CellIndex cell_index;

    float min_y = std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();

    float max_y = std::numeric_limits<float>::lowest();
    float max_x = std::numeric_limits<float>::lowest();

    //PillarBoundingBox(const Eigen::Ref<Eigen::VectorXd> xcoords, const Eigen::Ref<Eigen::VectorXd> ycoords) {
    //    //{
    //    //    auto [min_x_it, max_x_it] = std::minmax_element(xcoords.begin(), xcoords.begin() + 4);
    //    //    min_x = *min_x_it;
    //    //    max_x = *max_x_it;
    //    //}
    //    //{
    //    //    auto [min_y_it, max_y_it] = std::minmax_element(ycoords.begin(), ycoords.begin() + 4);
    //    //    min_y = *min_y_it;
    //    //    max_y = *max_y_it;
    //    //}
    //    //for (int v = 0; v < resfo::NUM_CORNERS; v += 3) {
    //    //    const double& x = corners[v];
    //    //    const double& y = corners[v + 1];
    //    //    this->min_x = std::min(this->min_x, x);
    //    //    this->min_y = std::min(this->min_y, y);

    //    //    this->max_x = std::max(this->max_x, x);
    //    //    this->max_y = std::max(this->max_y, y);
    //    //}
    //}

    friend std::ostream& operator<<(std::ostream& os, const PillarBoundingBox& box);
};

std::vector<PillarBoundingBox> create_pillar_bounding_boxes(
    const float* coord,
    const resfo::GridDimensions& dims
);


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

// Spatial hash over PillarBoundingBox (one per (i,j) column).
// query(x, y) returns all (i,j) columns whose bounding box contains the point.
class PillarTree2D {
private:
    struct PairHash {
        std::size_t operator()(const std::pair<int,int>& p) const noexcept {
            return std::hash<long long>()(
                (static_cast<long long>(p.first) << 32) ^ static_cast<unsigned int>(p.second));
        }
    };

    float cell_size = 1.0f;
    std::unordered_map<std::pair<int,int>, std::vector<PillarBoundingBox>, PairHash> table;

    std::pair<int,int> hash_cell(float x, float y) const {
        return {static_cast<int>(std::floor(x / cell_size)),
                static_cast<int>(std::floor(y / cell_size))};
    }

public:
    PillarTree2D() = default;
    explicit PillarTree2D(std::vector<PillarBoundingBox> boxes);

    // Returns (i,j) indices of all pillar columns whose bounding box contains (x,y).
    std::vector<std::pair<int,int>> query(float x, float y, float tolerance = 1.e-6f) const;
};

// True 2D interval tree over PillarBoundingBox.
// The X dimension is indexed with a classic interval tree (O(log n + k') stabbing query);
// each candidate is then filtered by Y containment. This avoids the spatial hash's
// worst-case behaviour when bounding boxes are large (e.g. heavily tilted pillars).
class PillarIntervalTree {
private:
    struct Node {
        float mid;
        // Spanning intervals sorted by primary-axis min ascending  (used when query <= mid).
        std::vector<PillarBoundingBox> by_min;
        // Spanning intervals sorted by primary-axis max descending (used when query >  mid).
        std::vector<PillarBoundingBox> by_max;
        int left  = -1;
        int right = -1;
    };

    std::vector<Node> nodes_;
    // When true the tree is built on the Y axis (x and y coords are swapped in
    // the stored bboxes).  The query() method swaps its x,y arguments before
    // descending the tree and the results are returned with (i,j) unmodified.
    bool transposed_ = false;

    // Returns index of the root node, or -1 for an empty tree.
    int build(std::vector<PillarBoundingBox> boxes);
    void query_node(int idx, float x, float y, float tol,
                    std::vector<std::pair<int,int>>& results) const;

public:
    PillarIntervalTree() = default;
    explicit PillarIntervalTree(std::vector<PillarBoundingBox> boxes);

    // Returns (i,j) indices of all pillar columns whose bounding box contains (x,y).
    std::vector<std::pair<int,int>> query(float x, float y, float tolerance = 1.e-6f) const;
};

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
