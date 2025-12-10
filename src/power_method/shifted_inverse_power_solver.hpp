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
#include "../option/shifted_solver_option.hpp"
#include "../result/eigen_result.hpp"
#include "../core/tolerance.hpp"


template <typename Mat>
    requires ScalarConcept<typename Mat::Scalar>
EigenResult<typename Mat::Scalar> shiftedInversePowerImpl(
    const Matrix& A_wrapped,
    const Mat& A,
    const ShiftedSolverOptions<typename Mat::Scalar>& opts
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
    Scalar shift = opts.shift;
    Scalar lambda = Scalar(0);
    Vector<Scalar> x = Vector<Scalar>::Random(n);
    x.normalize();

    int  usedIters  = 0;
    bool converged  = false;
    bool initialized = false;

    using Real = typename Eigen::NumTraits<Scalar>::Real;  
    for (int k = 0; k < opts.maxIterations; ++k) {
        // Solve (A - shift I) y = x
        Vector<Scalar> y = solve_shifted<Scalar>(A_wrapped, shift, x);

        const Real normY = y.norm();
        if (normY == Real(0)) {
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
 * inside a Matrix wrapper. It dispatches based on the internal storage.
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
 *   ShiftedSolverOptions<double> opts;
 *   opts.shift         = 1.0;   // σ = 1.0
 *   opts.maxIterations = 500;
 *   opts.tolerance     = 1e-8;
 *
 *   EigenResult<double> result =
 *       shiftedInversePowerMethod<double>(M, opts);
 * @endcode
 *
 * @tparam Scalar Numeric type stored in the matrix.
 * @param M    Matrix wrapper holding A.
 * @param opts Full solver options including the shift σ.
 *
 * @return EigenResult<Scalar> containing the computed eigenpair.
 */
template <ScalarConcept Scalar>
EigenResult<Scalar> shiftedInversePowerMethod(const Matrix& M, const ShiftedSolverOptions<Scalar>& opts = ShiftedSolverOptions<Scalar>{}) {
    if (M.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("shiftedInversePowerMethod: scalar type mismatch");
    }

    if (M.isDense()) {
        using Mat = Matrix::Dense<Scalar>;
        return shiftedInversePowerImpl<Mat>(M, M.cast<Mat>(), opts);
    } else {
        using Mat = Matrix::Sparse<Scalar>;
        return shiftedInversePowerImpl<Mat>(M, M.cast<Mat>(), opts);
    }
}
