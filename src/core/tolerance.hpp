#pragma once
/**
 * @file tolerance.hpp
 * @brief Implementation of relative tolerance checks.
 */

#include <cmath>
#include "types.hpp"

/**
 * @brief Checks whether two floating-point values are close under a relative tolerance.
 *
 * This comparison follows a stable variant of relative error. The scale term
 * uses 1 + |a| to avoid issues when the reference value is small. The test is:
 *
 *    |a - b| ≤ tol × (1 + |a|)
 *
 * This behaves like a relative tolerance for large magnitudes yet remains
 * well-defined near zero.
 *
 * @param a Reference value.
 * @param b Value to compare with the reference.
 * @param tol Non-negative tolerance factor.
 * @return true if the values are considered close under this criterion.
 */
template <ScalarConcept Scalar>
inline bool is_close_relative(Scalar a, Scalar b, double tol) {
    const double diff  = std::abs(a - b);
    const double scale = 1.0 + std::abs(a);
    return diff <= tol * scale;
}
