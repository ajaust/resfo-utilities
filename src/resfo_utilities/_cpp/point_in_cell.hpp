#pragma once
#include "grid_search.hpp"

#include <Eigen/Dense>

namespace resfo {

std::vector<double> cell_corners(int i, int j, int k, const float* coord, const float* zcorn,
                                 const GridDimensions& dims);

bool point_in_cell(const Eigen::Vector3d& point, int i, int j, int k, const float* coord,
                   const float* zcorn, const GridDimensions& dims, float tolerance);

}  // namespace resfo
