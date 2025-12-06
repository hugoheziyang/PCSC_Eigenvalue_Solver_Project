#pragma once
/**
 * @file power_method.hpp
 * @brief Implementation of the power method for the Matrix wrapper.
 */

#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>

#include "../matrix/matrix.hpp"
#include "../core/types.hpp"
#include "../option/solver_option.hpp"
#include "../result/eigen_result.hpp"
#include "../core/tolerance.hpp"

/**
 * @brief Internal implementation of the power method on a concrete Eigen matrix.
 *
 * This function implements the standard power iteration on a concrete matrix type
 * Mat that behaves like an Eigen dense or sparse matrix. The scalar type must
 * satisfy ScalarConcept.
 *
 * The method iterates
 *   x_{k+1} = A x_k / ||A x_k||
 * and approximates the dominant eigenvalue by the Rayleigh quotient
 *   lambda_k = x_k^T A x_k
 *
 * Convergence is detected using a relative change criterion on lambda_k
 *
 * The returned EigenResult contains:
 *   - eigenvalue  : last Rayleigh quotient
 *   - eigenvector : last iterate x_k (normalized)
 *   - iterations  : number of iterations performed
 *   - converged   : true if the relative criterion was satisfied
 *
 * @tparam Mat  Concrete Eigen matrix type (dense or sparse).
 * @param A     Input matrix.
 * @param opts  Solver options (maximum iterations and tolerance).
 *
 * @return EigenResult<Scalar> with eigenvalue, eigenvector, iteration count, and convergence flag.
 *
 * @throws std::runtime_error If the matrix is not square or has zero size.
 */
template <typename Mat>
    requires ScalarConcept<typename Mat::Scalar>
EigenResult<typename Mat::Scalar> powerMethodImpl(const Mat& A, const SolverOptions& opts) {
    using Scalar = typename Mat::Scalar;

    const auto n = A.rows();
    if (n != A.cols()) {
        throw std::runtime_error("powerMethod: matrix must be square");
    }
    if (n == 0) {
        throw std::runtime_error("powerMethod: matrix has zero size");
    }

    // The future output
    Scalar lambda      = Scalar(0);    
    Vector<Scalar> x = Vector<Scalar>::Random(n); x.normalize();
    int    usedIters   = 0;
    bool   converged   = false;

    bool   initialized = false;
    for (int k = 0; k < opts.maxIterations; ++k) {
        Vector<Scalar> y = A * x;

        // If y becomes 0, we cannot continue
        const auto normY = y.norm();
        if (normY == Scalar(0)) {
            usedIters = k + 1;
            break;
        }

        x = y / normY;

        // New eigenvalue estimate from the Rayleigh quotient
        Scalar lambdaNew = x.dot(A * x);

        if (initialized) {
            if (is_close_relative(lambdaNew, lambda, opts.tolerance)) {
                // The algorithm converged
                lambda    = lambdaNew;
                usedIters = k + 1;
                converged = true;
                break;
            }
        }

        lambda      = lambdaNew;
        initialized = true;
        usedIters   = k + 1;
    }

    return EigenResult<Scalar>(lambda, x, usedIters, converged);
}

/**
 * @brief Power method interface for the Matrix wrapper.
 *
 * This function applies the power method to the matrix stored inside a Matrix
 * wrapper. It dispatches based on the internal storage:
 *   - If M.isDense() is true, it calls powerMethodImpl on a Dense<Scalar> matrix.
 *   - Otherwise, it calls powerMethodImpl on a Sparse<Scalar> matrix.
 *
 * The template parameter Scalar must match the scalar type actually stored in M.
 * At runtime the function checks
 *   M.scalar_type() == typeid(Scalar)
 * and throws std::runtime_error if this is not satisfied.
 *
 * Example:
 * @code
 *   Eigen::MatrixXd A = Eigen::MatrixXd::Random(100, 100);
 *   Matrix M(A);
 *   SolverOptions opts;
 *   opts.maxIterations = 500;
 *   opts.tolerance     = 1e-8;
 *
 *   EigenResult<double> result = powerMethod<double>(M, opts);
 *   std::cout << "lambda = " << result.eigenvalue << std::endl;
 * @endcode
 *
 * @tparam Scalar Scalar type of the eigenpair.
 * @param M       Matrix wrapper storing a dense or sparse Eigen matrix.
 * @param opts    Solver options (maximum iterations and tolerance).
 *
 * @return EigenResult<Scalar> containing the dominant eigenvalue approximation
 *         and its associated eigenvector.
 *
 * @throws std::runtime_error If the scalar type stored in M does not match Scalar.
 */
template <ScalarConcept Scalar>
EigenResult<Scalar> powerMethod(const Matrix& M, const SolverOptions& opts = SolverOptions{}) {
    if (M.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("powerMethod: scalar type mismatch");
    }

    if (M.isDense()) {
        using Mat = Matrix::Dense<Scalar>;
        return powerMethodImpl<Mat>(M.cast<Mat>(), opts);
    } else {
        using Mat = Matrix::Sparse<Scalar>;
        return powerMethodImpl<Mat>(M.cast<Mat>(), opts);
    }
}
