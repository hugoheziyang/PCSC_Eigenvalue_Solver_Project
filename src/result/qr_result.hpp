#pragma once
/**
 * @file qr_result.hpp
 * @brief Result type for QR eigenvalue algorithms.
 */

#include "../core/types.hpp"
#include <Eigen/Dense>

/**
 * @struct QRResult
 * @brief Stores the output of QR-based eigenvalue solvers.
 *
 * @tparam Scalar Numeric type (e.g., double).
 *
 * This struct stores:
 *   - The vector of computed eigenvalues
 *   - Number of QR iterations performed
 *   - A flag indicating whether convergence occurred
 */
template <ScalarConcept Scalar>
struct QRResult {

    /// Computed eigenvalues (typically sorted, depending on implementation).
    Vector<Scalar> eigenvalues;

    /// Number of QR iterations taken.
    int iterations = 0;

    /// Whether the QR iteration converged.
    bool converged = false;

    /// Default constructor.
    QRResult() = default;

    /// Full constructor.
    QRResult(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& eigvals, int iters, bool conv)
        : eigenvalues(eigvals)
        , iterations(iters)
        , converged(conv)
    {}
};
