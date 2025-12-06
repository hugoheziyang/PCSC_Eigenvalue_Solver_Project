#pragma once
/**
 * @file shifted_inverse_power_solver.hpp
 * @brief Implementation of the shifted inverse power method for the Matrix wrapper.
 */

#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>

#include "../matrix/matrix.hpp"
#include "../matrix/solve_shifted.hpp"
#include "../core/types.hpp"
#include "../option/solver_option.hpp"
#include "../option/shifted_option.hpp"
#include "../result/eigen_result.hpp"
#include "../core/tolerance.hpp"

/**
 * @brief Core implementation of the shifted inverse power method.
 *
 * This templated helper operates on a concrete Eigen matrix type Mat
 * yet uses the Matrix wrapper for solving the shifted system
 * via solve_shifted.
 *
 * It repeatedly solves
 *
 *    (A - σ I) y_k = x_k
 *    x_{k+1}       = y_k / ||y_k||
 *
 * and approximates an eigenvalue of A near σ with the Rayleigh quotient
 *
 *    λ_k = x_k^T A x_k
 *
 * The loop stops when the relative change in λ_k satisfies the tolerance
 * in SolverOptions or when the maximum number of iterations is reached.
 *
 * @tparam Mat Concrete Eigen matrix type
 *             (Matrix::Dense<Scalar> or Matrix::Sparse<Scalar>).
 *
 * @param A_wrapped Matrix wrapper holding A.
 * @param A         Concrete Eigen view of A inside the wrapper.
 * @param opts      Generic solver options (maxIterations, tolerance).
 * @param shiftedOpts Shift options containing the shift σ.
 *
 * @return EigenResult<Scalar> with eigenvalue, eigenvector, iteration count, convergence flag.
 *
 * @throws std::runtime_error If the matrix is not square or has zero size.
 */
template <typename Mat>
    requires ScalarConcept<typename Mat::Scalar>
EigenResult<typename Mat::Scalar> shiftedInversePowerImpl(
    const Matrix& A_wrapped,
    const Mat& A,
    const SolverOptions& opts,
    const ShiftedOptions<typename Mat::Scalar>& shiftedOpts
) {
    using Scalar = typename Mat::Scalar;

    const auto n = A.rows();
    if (n != A.cols()) {
        throw std::runtime_error("shiftedInversePowerMethod: matrix must be square");
    }
    if (n == 0) {
        throw std::runtime_error("shiftedInversePowerMethod: matrix has zero size");
    }

    // Future output
    Scalar lambda = Scalar(0);
    Vector<Scalar> x = Vector<Scalar>::Random(n);
    x.normalize();

    int  usedIters  = 0;
    bool converged  = false;
    bool initialized = false;

    for (int k = 0; k < opts.maxIterations; ++k) {
        // Solve (A - σ I) y = x with the generic shifted solver
        Vector<Scalar> y = solve_shifted<Scalar>(A_wrapped, shiftedOpts, x);

        const auto normY = y.norm();
        if (normY == Scalar(0)) {
            usedIters = k + 1;
            break;
        }

        x = y / normY;

        // New eigenvalue estimate from the Rayleigh quotient on A
        Scalar lambdaNew = x.dot(A * x);

        if (initialized) {
            if (is_close_relative(lambdaNew, lambda, opts.tolerance)) {
                lambda     = lambdaNew;
                usedIters  = k + 1;
                converged  = true;
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
 * @brief Shifted inverse power method interface for the Matrix wrapper.
 *
 * This function applies the shifted inverse power method to the matrix stored
 * inside a Matrix wrapper. It dispatches based on the internal storage:
 *
 *   - If M.isDense() is true it calls shiftedInversePowerImpl on a Dense<Scalar>.
 *   - Sinon il appelle shiftedInversePowerImpl sur un Sparse<Scalar>.
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
 *
 *   SolverOptions opts;
 *   opts.maxIterations = 500;
 *   opts.tolerance     = 1e-8;
 *
 *   ShiftedOptions<double> shift(1.0); // σ = 1.0
 *
 *   EigenResult<double> result =
 *       shiftedInversePowerMethod<double>(M, shift, opts);
 * @endcode
 *
 * @tparam Scalar Numeric type stored in the matrix.
 * @param M        Matrix wrapper holding A.
 * @param shiftOpt Shift options containing the shift σ.
 * @param opts     Generic solver options (with default values).
 *
 * @return EigenResult<Scalar> containing the computed eigenpair.
 */
template <ScalarConcept Scalar>
EigenResult<Scalar> shiftedInversePowerMethod(const Matrix& M, const ShiftedOptions<Scalar>& shiftedOpt, const SolverOptions& opts = SolverOptions{}) {
    if (M.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("shiftedInversePowerMethod: scalar type mismatch");
    }

    if (M.isDense()) {
        using Mat = Matrix::Dense<Scalar>;
        return shiftedInversePowerImpl<Mat>(M, M.cast<Mat>(), opts, shiftedOpt);
    } else {
        using Mat = Matrix::Sparse<Scalar>;
        return shiftedInversePowerImpl<Mat>(M, M.cast<Mat>(), opts, shiftedOpt);
    }
}
