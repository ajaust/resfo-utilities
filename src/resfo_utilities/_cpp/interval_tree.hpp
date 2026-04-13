#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "grid.hpp"

#include <Eigen/Dense>

namespace resfo {

// Bounding box of a column of cells defined by for pillars of the corner-point
// grid.
struct PillarBoundingBox {
    resfo::CellIndex cell_index;

    float min_y = std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();

    float max_y = std::numeric_limits<float>::lowest();
    float max_x = std::numeric_limits<float>::lowest();

    bool contains_y(float y, float tol = 1e-6f) const {
        return (min_y - tol <= y && y <= max_y + tol);
    }

    friend std::ostream& operator<<(std::ostream& os, const PillarBoundingBox& box);
};

std::vector<PillarBoundingBox> create_pillar_bounding_boxes(
    const float* coord,
    const resfo::GridDimensions& dims
);

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

} /* namespace resfo */

