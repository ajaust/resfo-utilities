#include "point_in_cell.hpp"

#include <cmath>
#include <array>
#include <algorithm>

#include <Eigen/Dense>

namespace resfo {

/*
 * To determine whether the point is in a given cell, the inverse of a
 * trilinear hexahedral (Q1) finite-element mapping is solved.  If the
 * solution (ξ, η, ζ) lies in [-1, 1]³ the point is inside the cell.
 *
 * Reference: "The Finite Element Method: A Practical Course"
 *            G. R. Liu & S. S. Quek, chapter 9.3.
 *
 * The solver is a bounded Levenberg–Marquardt for the 3×3 nonlinear
 * least-squares problem  min ½‖F(x) − p‖²  with x ∈ [-1,1]³, using
 * the analytic Jacobian of the trilinear shape functions.
 */

// Reference-cube corner signs: (ξ, η, ζ) ∈ {-1, +1}³.
static constexpr std::array<std::array<int,3>, NUM_CORNERS> ref_corners = {{
    {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
    {-1, -1,  1}, {1, -1,  1}, {-1, 1,  1}, {1, 1,  1}
}};

// Evaluate the trilinear mapping residual r = F(x) − point and its
// Jacobian J = dF/dx.  Both are written in-place; J is only written
// when compute_jacobian is true.
static inline void evaluate_trilinear(
    const Eigen::Vector3d& x,
    const double* corners,
    const Eigen::Vector3d& point,
    Eigen::Vector3d& residual,
    Eigen::Matrix3d& J,
    bool compute_jacobian)
{
    const double xi   = x[0];
    const double eta  = x[1];
    const double zeta = x[2];

    Eigen::Vector3d mapped = Eigen::Vector3d::Zero();
    if (compute_jacobian) J.setZero();

    for (int v = 0; v < NUM_CORNERS; ++v) {
        const double sx = ref_corners[v][0];
        const double sy = ref_corners[v][1];
        const double sz = ref_corners[v][2];

        const double N = 0.125 * (1 + xi * sx) * (1 + eta * sy) * (1 + zeta * sz);

        for (int d = 0; d < 3; ++d)
            mapped[d] += N * corners[v * 3 + d];

        if (compute_jacobian) {
            const double dN_dxi   = 0.125 * sx * (1 + eta * sy) * (1 + zeta * sz);
            const double dN_deta  = 0.125 * sy * (1 + xi  * sx) * (1 + zeta * sz);
            const double dN_dzeta = 0.125 * sz * (1 + xi  * sx) * (1 + eta  * sy);

            for (int d = 0; d < 3; ++d) {
                J(d, 0) += dN_dxi   * corners[v * 3 + d];
                J(d, 1) += dN_deta  * corners[v * 3 + d];
                J(d, 2) += dN_dzeta * corners[v * 3 + d];
            }
        }
    }

    residual = mapped - point;
}

std::vector<double> cell_corners(int i, int j, int k, const float* coord, const float* zcorn,
                                 const GridDimensions& dims) {
    std::vector<double> vertices(24, 0.0);

    // Pillar indices for the four corners of the cell (i,j)
    // (i,j), (i,j+1), (i+1,j), (i+1,j+1)
    int pillar_idx[4] = {
        (i * (dims.nj + 1) + j) * 6,
        (i * (dims.nj + 1) + (j + 1)) * 6,
        ((i + 1) * (dims.nj + 1) + j) * 6,
        ((i + 1) * (dims.nj + 1) + (j + 1)) * 6
    };

    std::array<float, 3> top[4], bot[4];
    for (int p = 0; p < 4; ++p) {
        top[p] = {coord[pillar_idx[p]], coord[pillar_idx[p] + 1], coord[pillar_idx[p] + 2]};
        bot[p] = {coord[pillar_idx[p] + 3], coord[pillar_idx[p] + 4], coord[pillar_idx[p] + 5]};
    }

    int zcorn_idx = (i * dims.nj * dims.nk + j * dims.nk + k) * NUM_CORNERS;
    std::array<float, NUM_CORNERS> z_values;
    std::copy_n(zcorn + zcorn_idx, NUM_CORNERS, z_values.begin());

    // The zcorn ordering is: TSW, TSE, TNW, TNE, BSW, BSE, BNW, BNE
    // where SW = (i,j), SE = (i+1,j), NW = (i,j+1), NE = (i+1,j+1)
    // Map zcorn index to pillar index
    std::array<int, NUM_CORNERS> pillar_order = {0, 2, 1, 3, 0, 2, 1, 3};

    for (int v = 0; v < NUM_CORNERS; ++v) {
        int p_idx = pillar_order[v];
        const auto& p_top = top[p_idx];
        const auto& p_bot = bot[p_idx];

        float height_diff = p_bot[2] - p_top[2];
        float t = (z_values[v] - p_top[2]) / height_diff;

        vertices[v * 3] = p_top[0] + t * (p_bot[0] - p_top[0]);
        vertices[v * 3 + 1] = p_top[1] + t * (p_bot[1] - p_top[1]);
        vertices[v * 3 + 2] = z_values[v];
    }

    return vertices;
}

bool point_in_cell(const Eigen::Vector3d& point, int i, int j, int k, const float* coord,
                   const float* zcorn, const GridDimensions& dims, float tolerance) {

    auto corners_vec = cell_corners(i, j, k, coord, zcorn, dims);
    const double* corners = corners_vec.data();

    const double tol_sq = static_cast<double>(tolerance) * static_cast<double>(tolerance);
    constexpr int max_iterations = 20;

    // Levenberg–Marquardt state
    Eigen::Vector3d x(0.0, 0.0, 0.0);  // initial guess: cell center
    Eigen::Vector3d r;
    Eigen::Matrix3d J;

    evaluate_trilinear(x, corners, point, r, J, /*compute_jacobian=*/true);
    double cost = 0.5 * r.squaredNorm();

    if (cost < 0.5 * tol_sq)
        return true;

    double lambda = 1e-3 * J.diagonal().array().abs().maxCoeff();
    if (lambda < 1e-10) lambda = 1e-6;
    double nu = 2.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Solve (J^T J + λ I) δ = -J^T r
        Eigen::Matrix3d JtJ = J.transpose() * J;
        Eigen::Vector3d Jtr = J.transpose() * r;
        JtJ.diagonal().array() += lambda;

        Eigen::Vector3d delta = JtJ.ldlt().solve(-Jtr);

        // Box-project to [-1, 1]³
        Eigen::Vector3d x_new = (x + delta).cwiseMax(-1.0).cwiseMin(1.0);
        Eigen::Vector3d actual_delta = x_new - x;

        Eigen::Vector3d r_new;
        Eigen::Matrix3d J_new;
        evaluate_trilinear(x_new, corners, point, r_new, J_new, false);
        double cost_new = 0.5 * r_new.squaredNorm();

        // Gain ratio: actual reduction / predicted reduction
        double predicted = -actual_delta.dot(Jtr) - 0.5 * actual_delta.dot(JtJ * actual_delta);
        if (predicted < 1e-30) predicted = 1e-30;
        double gain_ratio = (cost - cost_new) / predicted;

        if (gain_ratio > 0) {
            // Accept step
            x = x_new;
            r = r_new;
            cost = cost_new;

            if (cost < 0.5 * tol_sq)
                return true;

            // Recompute Jacobian at the new point
            evaluate_trilinear(x, corners, point, r, J, true);

            // Adjust damping (Nielsen update)
            double factor = 1.0 - std::pow(2.0 * gain_ratio - 1.0, 3);
            lambda *= std::max(factor, 1.0 / 3.0);
            nu = 2.0;
        } else {
            // Reject step, increase damping
            lambda *= nu;
            nu *= 2.0;
        }

        // If lambda grows too large, the step is effectively zero — give up.
        if (lambda > 1e16)
            return false;
    }

    return cost < 0.5 * tol_sq;
}

}  // namespace resfo
