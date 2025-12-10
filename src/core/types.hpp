// This file was mostly written by an AI tool.
#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <type_traits>
#include <complex>

namespace EigSol{

/// Helper trait: false by default
template <typename T>
struct is_complex_of_floating : std::false_type {};

/// Specialization: std::complex<Inner> is valid if Inner is floating point.
template <typename Inner>
struct is_complex_of_floating<std::complex<Inner>>
    : std::bool_constant<std::is_floating_point_v<Inner>> {};

/// Concept for acceptable scalar types in the library.
///
/// Currently accepts:
///   - float, double, long double
///   - std::complex<float>, std::complex<double>, std::complex<long double>
template <typename Scalar>
concept ScalarConcept =
    std::is_floating_point_v<Scalar> || is_complex_of_floating<Scalar>::value;

/// Convenience alias for column vectors with valid scalar type.
template <ScalarConcept Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

} // end namespace