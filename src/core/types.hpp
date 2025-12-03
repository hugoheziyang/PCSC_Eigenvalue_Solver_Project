#pragma once
#include <Eigen/Dense>
#include <concepts>
#include <type_traits>

template <typename Scalar>
concept ScalarConcept = std::is_floating_point_v<Scalar>;

template <ScalarConcept Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
