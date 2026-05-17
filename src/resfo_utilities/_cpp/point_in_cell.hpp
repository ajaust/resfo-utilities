#pragma once
#include "grid_search.hpp"

#include <array>

#include <Eigen/Dense>

namespace resfo {

std::array<double, NUM_CORNERS * 3> cell_corners(
        int i, int j, int k, const float* coord, const float* zcorn,
        const GridDimensions& dims);

bool point_in_cell(const Eigen::Vector3d& point, int i, int j, int k, const float* coord,
                   const float* zcorn, const GridDimensions& dims, float tolerance);

}  // namespace resfo
