#pragma once
/**
 * @file eigen_result.hpp
 * @brief Struct that stores the output of single-eigenpair solvers.
 */

#include "../core/types.hpp"
#include <Eigen/Dense>

/**
 * @struct EigenResult
 * @brief Stores the result of algorithms that compute one eigenvalue and eigenvector.
 *
 * @tparam Scalar Numeric type of the matrix.
 *
 * This struct is used by:
 *   - Power method
 *   - Shifted inverse power
 */
template <ScalarConcept Scalar>
struct EigenResult {

    /// Computed eigenvalue.
    Scalar eigenvalue;

    /// Corresponding normalized eigenvector.
    Vector<Scalar> eigenvector;

    /// Number of iterations performed.
    int iterations = 0;

    /// Whether the solver converged successfully.
    bool converged = false;

    /// Default constructor.
    EigenResult() = default;

    /// Full constructor.
    EigenResult(
        const Scalar& lambda,
        const Vector<Scalar>& vec,
        int iters,
        bool conv
    ): 
        eigenvalue(lambda),
        eigenvector(vec),
        iterations(iters),
        converged(conv)
    {}
};
