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

std::unique_ptr<IntervalTree2D::Node> IntervalTree2D::build(std::vector<BoundingBox> boxes) {
    if (boxes.empty()) return nullptr;

    // Compute median of interval midpoints on x
    std::vector<double> midpoints;
    midpoints.reserve(boxes.size());
    for (const auto& box : boxes)
        midpoints.push_back((box.min_x + box.max_x) * 0.5);
    double median = median_of(midpoints);

    std::vector<BoundingBox> left, right, overlapping;
    for (const auto& box : boxes) {
        if (box.max_x < median)
            left.push_back(box);
        else if (box.min_x > median)
            right.push_back(box);
        else
            overlapping.push_back(box);
    }

    // Sort overlapping by min_y for efficient y queries
    std::sort(overlapping.begin(), overlapping.end(),
              [](const BoundingBox& a, const BoundingBox& b) {
                  return a.min_y < b.min_y;
              });

    auto node = std::make_unique<IntervalTree2D::Node>(median, std::move(overlapping));
    node->left = build(std::move(left));
    node->right = build(std::move(right));
    return node;
}

void IntervalTree2D::query(const IntervalTree2D::Node* node, double x0, double y0, std::vector<CellIndex>& results) {
    if (!node) return;

    if (x0 < node->median_x) {
        query(node->left.get(), x0, y0, results);
    } else if (x0 > node->median_x) {
        query(node->right.get(), x0, y0, results);
    }

    // Since overlapping is sorted by min_y, we can do a binary search to limit candidates
    // but here we do a simple linear scan for clarity
    for (const auto& box : node->overlapping) {
        if (box.min_y <= y0 && y0 <= box.max_y && box.min_x <= x0 && x0 <= box.max_x) {
            results.push_back(box.cell_index);
        }
    }
}

/*
#include <algorithm>
#include <vector>

struct FlatNode{
    BoundingBox box;
    double max_x;

    double min_y;
    double max_y;
};

class FlatIntervalTree2D {
    std::vector<FlatNode> tree_;
    std::size_t size_;
    std::size_t nodes_inserted_;

public:
    FlatIntervalTree2D(
        std::vector<BoundingBox> boxes
    ) {
        this->size_ = boxes.size();
        if (this->size_ == 0) return;

        std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
            return a.min_x < b.min_x;
        });

        this->tree_.resize(this->size_);
        this->nodes_inserted_ = 0;
        build(boxes, 0, 0, static_cast<int>(this->size_) - 1);

    }

    std::vector<resfo::CellIndex> queryPoint(double x, double y) const {
        std::vector<resfo::CellIndex> result;
        if (this->size_ == 0) return result;
        result.reserve(20);
        queryUtil(0, 0, static_cast<int>(this->size_) - 1, x, y, result);
        return result;
    }

    std::size_t size() const {
        return this->size_;
    }

private:
    void build(const std::vector<BoundingBox>& boxes, int node, int start, int end) {
        int mid = start + (end - start) / 2;

        if (node >= this->tree_.size()) {
            std::cout << "Node " << node << ", tree size " << this->tree_.size() << std::endl;
            throw std::runtime_error("Node index out of bounds during tree build");
        }
        std::cout << "Inserting node:" << node << ", " << this->nodes_inserted_ + 1 << " total" << std::endl;
        ++this->nodes_inserted_;
        this->tree_[node] = {boxes[mid], boxes[mid].max_x, boxes[mid].min_y, boxes[mid].max_y};

        if (start == end) return;

        if (start <= mid - 1) {
            int left = leftChild(node);
            build(boxes, left, start, mid - 1);
            this->updateAugmentedData(node, left);
        }

        if (mid + 1 <= end) {
            int right = rightChild(node);
            build(boxes, right, mid + 1, end);
            this->updateAugmentedData(node, right);
        }
    }

    void updateAugmentedData(int parent, int child) {
        if (child >= this->tree_.size()) {
            throw std::runtime_error("Child index out of bounds during augmented data update");
        }
        if (parent >= this->tree_.size()) {
            throw std::runtime_error("Parent index out of bounds during augmented data update");
        }
        this->tree_[parent].max_x = std::max(this->tree_[parent].max_x, this->tree_[child].max_x);
        this->tree_[parent].min_y = std::min(this->tree_[parent].min_y, this->tree_[child].min_y);
        this->tree_[parent].max_y = std::max(this->tree_[parent].max_y, this->tree_[child].max_y);
    }

    void queryUtil(int node, int start, int end, double x, double y, std::vector<resfo::CellIndex>& results) const {
        const FlatNode& current = this->tree_[node];

        if (x > current.max_x || y < current.min_y || y > current.max_y) {
            return; // No need to search this subtree
        }

        if (x >= current.box.min_x && x <= current.box.max_x && y >= current.box.min_y && y <= current.box.max_y) {
            results.push_back(current.box.cell_index);
        }

        int mid = start + (end - start) / 2;
        if (start <= mid - 1) {
            queryUtil(leftChild(node), start, mid - 1, x, y, results);
        }

        if (mid + 1 <= end and x >= current.box.min_x) {
            queryUtil(rightChild(node), mid + 1, end, x, y, results);
        }
    }

    inline int leftChild(int node) const {
        return 2 * node + 1;
    }

    inline int rightChild(int node) const {
        return 2 * node + 2;
    }
};
*/

/*
// 1D centered interval tree used for Y indexing (stores pointers to BoundingBox)
class YIntervalTree {
public:
    struct Node {
        double center;
        std::vector<const BoundingBox*> intervals; // boxes whose y-interval contains center
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node(double c) : center(c) {}
    };

    YIntervalTree() = default;

    void build(std::vector<const BoundingBox*> boxes) {
        root = build_rec(std::move(boxes));
    }

    // Append boxes whose [min_y,max_y] contains y to out
    void query(double y, std::vector<const BoundingBox*>& out) const {
        query_rec(root.get(), y, out);
    }

private:
    std::unique_ptr<Node> root;

    static double median_of_endpoints(const std::vector<const BoundingBox*>& boxes) {
        std::vector<double> pts;
        pts.reserve(boxes.size() * 2);
        for (auto b : boxes) { pts.push_back(b->min_y); pts.push_back(b->max_y); }
        std::nth_element(pts.begin(), pts.begin() + pts.size()/2, pts.end());
        return pts[pts.size()/2];
    }

    static std::unique_ptr<Node> build_rec(std::vector<const BoundingBox*> boxes) {
        if (boxes.empty()) return nullptr;
        double center = median_of_endpoints(boxes);
        auto node = std::make_unique<Node>(center);

        std::vector<const BoundingBox*> left_list, right_list;
        for (auto b : boxes) {
            if (b->max_y < center) left_list.push_back(b);
            else if (b->min_y > center) right_list.push_back(b);
            else node->intervals.push_back(b);
        }
        node->left  = build_rec(std::move(left_list));
        node->right = build_rec(std::move(right_list));
        return node;
    }

    static void query_rec(const Node* node, double y, std::vector<const BoundingBox*>& out) {
        if (!node) return;
        for (const BoundingBox* b : node->intervals)
            if (b->min_y <= y && y <= b->max_y) out.push_back(b);

        if (y < node->center) query_rec(node->left.get(), y, out);
        else if (y > node->center) query_rec(node->right.get(), y, out);
    }
};


// Primary 2D interval tree on X, with per-node YIntervalTree
class IntervalTree2D {
public:
    struct Node {
        double center;
        std::vector<const BoundingBox*> x_intervals; // boxes whose x-interval contains center
        YIntervalTree ytree;                 // built from x_intervals
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node(double c) : center(c) {}
    };

    IntervalTree2D() = default;

    // Build from vector<BoundingBox>. Copies boxes into internal storage.
    void build(const std::vector<BoundingBox>& boxes) {
        b_storage.clear();
        b_storage.reserve(boxes.size());
        std::vector<const BoundingBox*> ptrs;
        ptrs.reserve(boxes.size());
        for (const auto& b : boxes) {
            b_storage.push_back(b);           // copy into internal storage
            ptrs.push_back(&b_storage.back()); // stable pointer to stored BoundingBox
        }
        root = build_rec(std::move(ptrs));
    }

    // Query point (x,y) -> returns vector of ids of boxes containing the point
    std::vector<int> query_point(double x, double y) const {
        std::vector<int> result;
        query_rec(root.get(), x, y, result);
        return result;
    }

private:
    std::unique_ptr<Node> root;
    std::vector<BoundingBox> b_storage; // internal copy so pointers remain valid

    static double median_of_endpoints(const std::vector<const BoundingBox*>& boxes) {
        std::vector<double> pts;
        pts.reserve(boxes.size() * 2);
        for (auto b : boxes) { pts.push_back(b->min_x); pts.push_back(b->max_x); }
        std::nth_element(pts.begin(), pts.begin() + pts.size()/2, pts.end());
        return pts[pts.size()/2];
    }

    static std::unique_ptr<Node> build_rec(std::vector<const BoundingBox*> boxes) {
        if (boxes.empty()) return nullptr;
        double center = median_of_endpoints(boxes);
        auto node = std::make_unique<Node>(center);

        std::vector<const BoundingBox*> left_list, right_list, center_list;
        for (auto b : boxes) {
            if (b->max_x < center) left_list.push_back(b);
            else if (b->min_x > center) right_list.push_back(b);
            else center_list.push_back(b);
        }

        node->x_intervals = std::move(center_list);
        node->ytree.build(node->x_intervals); // build Y tree from x-intervals

        node->left  = build_rec(std::move(left_list));
        node->right = build_rec(std::move(right_list));
        return node;
    }

    static void query_rec(const Node* node, double x, double y, std::vector<int>& out) {
        if (!node) return;
        // Query the per-node y-tree for candidates whose y-range contains y
        std::vector<const BoundingBox*> y_candidates;
        node->ytree.query(y, y_candidates);

        // Filter by x (some boxes in this node may not contain the queried x)
        for (const BoundingBox* b : y_candidates)
            if (b->min_x <= x && x <= b->max_x) out.push_back(b->id);

        if (x < node->center) query_rec(node->left.get(), x, y, out);
        else if (x > node->center) query_rec(node->right.get(), x, y, out);
    }
};
*/


//std::vector<std::vector<std::vector<BoundingBox>>> create_bounding_boxes(const float* coord, const float* zcorn,const resfo::GridDimensions& dims) {
std::vector<BoundingBox> create_bounding_boxes(const float* coord, const float* zcorn,const resfo::GridDimensions& dims) {
    //std::cout << "Grid dimensions: " << dims.ni << " " << dims.nj << " " << dims.nk << "\n";

    //std::vector<std::vector<std::vector<BoundingBox>>> boxes(
    //    dims.ni, std::vector<std::vector<BoundingBox>>(
    //        dims.nj, std::vector<BoundingBox>(dims.nk, BoundingBox())
    //    )
    //);
    std::vector<BoundingBox> boxes;
    boxes.reserve(
        dims.ni * dims.nj * dims.nk
    );
    //Eigen::TensorMap<Eigen::Tensor<BoundingBox, 3>> mapped_tensor(
    //    boxes.data(), dims.ni, dims.nj, dims.nk
    //);

    // Placeholder function to avoid linker errors if needed
    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            //auto corners_top = resfo::cell_corners(i, j, 0, coord, zcorn, dims);
            //auto corners_top = resfo::cell_corners(i, j, dims.nk-1, coord, zcorn, dims);
            //boxes[i][j] = BoundingBox({});
            for (int k = 0; k < dims.nk; ++k) {
                // Implementation would go here
                auto corners = resfo::cell_corners(i, j, k, coord, zcorn, dims);
                {
                    //mapped_tensor[i][j][k] = BoundingBox({i, j, k}, corners);
                    //mapped_tensor(i, j, k) = BoundingBox({i, j, k}, corners);
                    boxes.emplace_back(BoundingBox({i, j, k}, corners));
                }

                //if (i < 2 and j < 2 and k < 1) {
                //    std::cout << "Cell (" << i << ", " << j << ", " << k << ") corners:\n";
                //    for (int v = 0; v < resfo::NUM_CORNERS; ++v) {
                //        std::cout << "Corner " << v << ": ("
                //            << corners[v * 3] << ", "
                //            << corners[v * 3 + 1] << ", "
                //            << corners[v * 3 + 2] << ")\n";
                //    }
                //    std::cout << "Bounding box" << std::endl;
                //    std::cout << boxes[i][j][k] << std::endl;

                //    getchar();
                //}
            }
        }
    }

    return boxes;
}

} /* namespace resfo */
