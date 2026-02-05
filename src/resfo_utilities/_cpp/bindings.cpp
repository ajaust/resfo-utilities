#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "grid_search.hpp"
#include "point_in_cell.hpp"
#include <Eigen/Dense>
namespace py = pybind11;


struct BoundingBox {
    resfo::CellIndex cell_index;

    double min_x;
    double min_y;
    double max_x;
    double max_y;

    BoundingBox() : cell_index({-1, -1, -1}) {
        this->min_y = std::numeric_limits<double>::max();
        this->min_x = std::numeric_limits<double>::max();

        this->max_y = std::numeric_limits<double>::lowest();
        this->max_x = std::numeric_limits<double>::lowest();
    }

    BoundingBox(resfo::CellIndex cell_index_, std::vector<double> corners) : BoundingBox() {
        this->cell_index = cell_index_;
        for (int v = 0; v < resfo::NUM_CORNERS; v += 3) {
            const double& x = corners[v];
            const double& y = corners[v + 1];
            this->min_x = std::min(this->min_x, x);
            this->min_y = std::min(this->min_y, y);

            this->max_x = std::max(this->max_x, x);
            this->max_y = std::max(this->max_y, y);
            //std::cout << "Corner " << v << ": ("
            //    << corners[v * 3] << ", "
            //    << corners[v * 3 + 1] << ", "
            //    << corners[v * 3 + 2] << ")\n";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& box);
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
    os << "BoundingBox(min_x=" << box.min_x << ", min_y=" << box.min_y
       << ", max_x=" << box.max_x << ", max_y=" << box.max_y << ")";
    return os;
}


#include <algorithm>
#include <vector>

struct FlatNode{
    BoundingBox box;
    double max_x;

    double min_y;
    double max_y;
};

class FlatIntervalTree2D {
    std::vector<FlatNode> tree;
    std::size_t size;

public:
    FlatIntervalTree2D(
        std::vector<BoundingBox> boxes
    ) {
        this->size = boxes.size();
        if (size == 0) return;

        std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
            return a.min_x < b.min_x;
        });

        this->tree.resize(this->size);
        build(boxes, 0, 0, static_cast<int>(this->size) - 1);

    }

    std::vector<resfo::CellIndex> queryPoint(double x, double y) const {
        std::vector<resfo::CellIndex> result;
        if (this->size == 0) return result;
        result.reserve(20);
        queryUtil(0, 0, static_cast<int>(this->size) - 1, x, y, result);
        return result;
    }

private:
    void build(const std::vector<BoundingBox>& boxes, int node, int start, int end) {
        int mid = start + (end - start) / 2;

        tree[node] = {boxes[mid], boxes[mid].max_x, boxes[mid].min_y, boxes[mid].max_y};

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
        tree[parent].max_x = std::max(tree[parent].max_x, tree[child].max_x);
        tree[parent].min_y = std::min(tree[parent].min_y, tree[child].min_y);
        tree[parent].max_y = std::max(tree[parent].max_y, tree[child].max_y);
    }

    void queryUtil(int node, int start, int end, double x, double y, std::vector<resfo::CellIndex>& results) const {
        const FlatNode& current = tree[node];

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


std::vector<BoundingBox> create_bounding_boxes(const float* coord, const float* zcorn,const resfo::GridDimensions& dims) {
    std::cout << "Grid dimensions: " << dims.ni << " " << dims.nj << " " << dims.nk << "\n";

    std::vector<std::vector<std::vector<BoundingBox>>> boxes(
        dims.ni, std::vector<std::vector<BoundingBox>>(
            dims.nj, std::vector<BoundingBox>(dims.nk, BoundingBox())
        )
    );
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
                    boxes[i][j][k] = BoundingBox({i, j, k}, corners);
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

}

std::vector<std::optional<std::tuple<int, int, int>>> find_cells_containing_points(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> zcorn_array,
    float tolerance) {

    auto points_buf = points_array.request();
    auto coord_buf = coord_array.request();
    auto zcorn_buf = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("Points array must have shape (n, 3)");
    }

    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3) {
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    }

    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8) {
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");
    }

    const float* points = static_cast<const float*>(points_buf.ptr);
    const float* coord = static_cast<const float*>(coord_buf.ptr);
    const float* zcorn = static_cast<const float*>(zcorn_buf.ptr);

    auto zcorn_shape = zcorn_buf.shape;
    resfo::GridDimensions dims{
        static_cast<int>(zcorn_shape[0]),
        static_cast<int>(zcorn_shape[1]),
        static_cast<int>(zcorn_shape[2])
    };

    auto [z_min, z_max] = std::minmax_element(zcorn, zcorn + zcorn_buf.size);

    auto top_intersection = resfo::pillar_z_intersection(coord, dims, *z_min);
    auto bot_intersection = resfo::pillar_z_intersection(coord, dims, *z_max);

    size_t num_points = points_buf.shape[0];
    std::vector<std::optional<std::tuple<int, int, int>>> results;
    results.reserve(num_points);
    std::optional<std::pair<int, int>> prev_ij;

    for (size_t p_idx = 0; p_idx < num_points; ++p_idx) {
        Eigen::Vector3d p{
            points[p_idx * 3],
            points[p_idx * 3 + 1],
            points[p_idx * 3 + 2]
        };

        auto result = resfo::grid_search(
            p, coord, zcorn, dims, top_intersection, bot_intersection, tolerance,
            prev_ij);

        if (result.has_value()) {
            results.push_back(std::make_tuple(result->i, result->j, result->k));
            prev_ij = std::make_pair(result->i, result->j);
        } else {
            results.push_back(std::nullopt);
            prev_ij = std::nullopt;
        }
    }
    return results;
}

std::vector<std::optional<std::tuple<int, int, int>>> find_cells_containing_points_interval_tree(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> zcorn_array,
    float tolerance) {

    auto points_buf = points_array.request();
    auto coord_buf = coord_array.request();
    auto zcorn_buf = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("Points array must have shape (n, 3)");
    }

    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3) {
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    }

    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8) {
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");
    }

    const float* points = static_cast<const float*>(points_buf.ptr);
    const float* coord = static_cast<const float*>(coord_buf.ptr);
    const float* zcorn = static_cast<const float*>(zcorn_buf.ptr);

    auto zcorn_shape = zcorn_buf.shape;
    resfo::GridDimensions dims{
        static_cast<int>(zcorn_shape[0]),
        static_cast<int>(zcorn_shape[1]),
        static_cast<int>(zcorn_shape[2])
    };
    auto bboxes = create_bounding_boxes(coord, zcorn, dims);
    auto interval_tree = FlatIntervalTree2D(std::move(bboxes));

    auto [z_min, z_max] = std::minmax_element(zcorn, zcorn + zcorn_buf.size);

    auto top_intersection = resfo::pillar_z_intersection(coord, dims, *z_min);
    auto bot_intersection = resfo::pillar_z_intersection(coord, dims, *z_max);

    size_t num_points = points_buf.shape[0];
    std::vector<std::optional<std::tuple<int, int, int>>> results;
    results.reserve(num_points);
    std::optional<std::pair<int, int>> prev_ij;

    for (size_t p_idx = 0; p_idx < num_points; ++p_idx) {
        Eigen::Vector3d p{
            points[p_idx * 3],
            points[p_idx * 3 + 1],
            points[p_idx * 3 + 2]
        };

        auto result = resfo::grid_search(
            p, coord, zcorn, dims, top_intersection, bot_intersection, tolerance,
            prev_ij);

        if (result.has_value()) {
            results.push_back(std::make_tuple(result->i, result->j, result->k));
            prev_ij = std::make_pair(result->i, result->j);
        } else {
            results.push_back(std::nullopt);
            prev_ij = std::nullopt;
        }
    }
    return results;
}

py::array_t<bool> point_in_cell_wrapper(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_array,
    int i, int j, int k,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> zcorn_array,
    float tolerance) {

    auto points_buf = points_array.request();
    auto coord_buf = coord_array.request();
    auto zcorn_buf = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("Points array must have shape (n, 3)");
    }

    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3) {
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    }

    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8) {
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");
    }

    const float* points = static_cast<const float*>(points_buf.ptr);
    const float* coord = static_cast<const float*>(coord_buf.ptr);
    const float* zcorn = static_cast<const float*>(zcorn_buf.ptr);

    auto zcorn_shape = zcorn_buf.shape;
    resfo::GridDimensions dims{
        static_cast<int>(zcorn_shape[0]),
        static_cast<int>(zcorn_shape[1]),
        static_cast<int>(zcorn_shape[2])
    };
    create_bounding_boxes(coord, zcorn, dims);

    size_t num_points = points_buf.shape[0];
    auto result = py::array_t<bool>(num_points);
    auto result_buf = result.request();
    bool* result_ptr = static_cast<bool*>(result_buf.ptr);

    for (size_t p_idx = 0; p_idx < num_points; ++p_idx) {
        Eigen::Vector3d p{
            points[p_idx * 3],
            points[p_idx * 3 + 1],
            points[p_idx * 3 + 2]
        };

        result_ptr[p_idx] = resfo::point_in_cell(p, i, j, k, coord, zcorn, dims, tolerance);
    }

    return result;
}

PYBIND11_MODULE(_grid_cpp, m) {
    m.doc() = "Fast C++ implementation of grid search algorithms";

    m.def("find_cells_containing_points", &find_cells_containing_points,
          py::arg("points"),
          py::arg("coord"),
          py::arg("zcorn"),
          py::arg("tolerance") = 1e-6f,
          "Find cells containing given points");

    m.def("find_cells_containing_points_interval_tree", &find_cells_containing_points_interval_tree,
          py::arg("points"),
          py::arg("coord"),
          py::arg("zcorn"),
          py::arg("tolerance") = 1e-6f,
          "Find cells containing given points");

    m.def("point_in_cell", &point_in_cell_wrapper,
          py::arg("points"),
          py::arg("i"),
          py::arg("j"),
          py::arg("k"),
          py::arg("coord"),
          py::arg("zcorn"),
          py::arg("tolerance") = 1e-6f,
          "Check if points are in a specific cell");
}
