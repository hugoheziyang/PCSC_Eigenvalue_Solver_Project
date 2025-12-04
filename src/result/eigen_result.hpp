#pragma once
/**
 * @file EigenResult.hpp
 * @brief Struct that stores the output of single-eigenpair solvers.
 */

#include "../core/types.hpp"
#include <Eigen/Dense>

/**
 * @struct EigenResult
 * @brief Stores the result of algorithms that compute one eigenvalue and eigenvector.
 *
 * @tparam Scalar Numeric type of the matrix (e.g., double).
 *
 * This struct is used by:
 *   - Power method
 *   - Shifted power / shifted inverse power
 */
template <ScalarConcept Scalar>
struct EigenResult {

    /// Computed eigenvalue.
    Scalar eigenvalue;

    /// Corresponding normalized eigenvector.
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eigenvector;

    /// Number of iterations performed.
    int iterations = 0;

    /// Whether the solver converged successfully.
    bool converged = false;

    /// Default constructor.
    EigenResult() = default;

    /// Full constructor.
    EigenResult(const Scalar& lambda,
                const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& vec,
                int iters,
                bool conv)
        : eigenvalue(lambda)
        , eigenvector(vec)
        , iterations(iters)
        , converged(conv)
    {}
};
