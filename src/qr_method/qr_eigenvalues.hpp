#pragma once
/**
 * @file qr_eigenvalues.hpp
 * @brief Implementation of the QR eigenvalue iteration for the Matrix wrapper.
 */

#include <Eigen/Dense>
#include <algorithm>   
#include <cmath>       
#include <typeinfo>

#include "../core/types.hpp"             
#include "../option/solver_option.hpp"  
#include "../result/qr_result.hpp"        
#include "../matrix/matrix.hpp"          
#include "to_hessenberg.hpp"       
#include "qr_decompose.hpp"        

namespace EigSol {

/**
 * @brief QR eigenvalue iteration on a dense matrix using Householder-QR.
 *
 * Algorithm (unshifted QR iteration):
 *   1. Reduce A to Hessenberg form H = Qᵀ A Q   (via to_hessenberg_dense).
 *   2. Repeat up to opts.maxIterations:
 *        (i)   H = Q_k R_k       (via qr_decompose_dense)
 *        (ii)  H = R_k Q_k
 *        (iii) check convergence by monitoring the subdiagonal entries.
 *   3. At convergence, the diagonal of H approximates the eigenvalues.
 *
 * This is a basic QR iteration without sophisticated shifts or deflation,
 * intended for reasonably small matrices.
 *
 * @tparam Scalar  Floating-point or complex type satisfying ScalarConcept.
 * @param A        Input square matrix A.
 * @param opts     Solver options (maxIterations, tolerance).
 * @return         QRResult<Scalar> containing eigenvalues and status.
 */
template <ScalarConcept Scalar>
QRResult<Scalar>
qr_eigenvalues_dense(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const SolverOptions& opts
) {
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Index = Eigen::Index;

    const Index n = A.rows();

    if (A.rows() != A.cols()) {
        throw std::runtime_error("qr_eigenvalues_dense: A must be square");
    }
    if (n == 0) {
        return QRResult<Scalar>{ Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(), 0, true };
    }

    // 1. Reduce to Hessenberg form to accelerate QR iteration
    DenseMat H = to_hessenberg_dense<Scalar>(A);

    int iter = 0;
    bool converged = false;

    // Temporary storage for QR factors
    DenseMat Q(n, n);
    DenseMat R(n, n);

    for (iter = 0; iter < opts.maxIterations; ++iter) {
        // QR decomposition: H = Q * R
        qr_decompose_dense<Scalar>(H, Q, R);

        // Form next iterate: H := R * Q
        H = R * Q;

        // Convergence check: monitor subdiagonal magnitudes
        Real maxSubdiag = Real(0);
        for (Index i = 1; i < n; ++i) {
            using std::abs;
            Real val = abs(H(i, i - 1));
            if (val > maxSubdiag) {
                maxSubdiag = val;
            }
        }

        // Scale using matrix norm to get a relative tolerance
        Real scale = H.norm();  // Frobenius norm (real): sqrt(sum of squares of abs entries)
        Real thresh = static_cast<Real>(opts.tolerance) * (Real(1) + scale); // convergence threshold relative to scale of H

        if (maxSubdiag <= thresh) {
            converged = true;
            break;
        }
    }

    // 3. Extract diagonal as approximate eigenvalues
    Vector<Scalar> eigvals(n);
    for (Index i = 0; i < n; ++i) {
        eigvals(i) = H(i, i);
    }

    QRResult<Scalar> result;
    result.eigenvalues = std::move(eigvals);
    result.iterations = iter + 1;  // number of iterations performed (1-based)
    result.converged = converged;

    return result;
}

/**
 * @brief QR eigenvalue iteration for the Matrix wrapper (dense matrices only).
 *
 * This function:
 *   1. Checks that A_wrapped is dense and has scalar type Scalar.
 *   2. Extracts the underlying Eigen dense matrix.
 *   3. Calls qr_eigenvalues_dense on that matrix.
 *
 * @tparam Scalar     Floating-point or complex type satisfying ScalarConcept.
 * @param A_wrapped   Matrix wrapper holding a dense matrix of type Matrix::Dense<Scalar>.
 * @param opts        Solver options (maxIterations, tolerance).
 * @return            QRResult<Scalar> with eigenvalues and convergence info.
 *
 * @throws std::runtime_error if the matrix is not dense or scalar type mismatches.
 * @throws std::bad_cast      if the stored type is not Matrix::Dense<Scalar>.
 */
template <ScalarConcept Scalar>
QRResult<Scalar>
qr_eigenvalues(const Matrix& A_wrapped, const SolverOptions& opts)
{
    // 1. Only dense supported here
    if (!A_wrapped.isDense()) {
        throw std::runtime_error("qr_eigenvalues(Matrix): only dense matrices are supported");
    }

    // 2. Scalar type must match
    if (A_wrapped.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("qr_eigenvalues(Matrix): scalar type mismatch");
    }

    using DenseMat = Matrix::Dense<Scalar>;

    // 3. Extract underlying Eigen dense matrix
    const DenseMat& A = A_wrapped.cast<DenseMat>();

    // 4. Delegate to the dense version
    return qr_eigenvalues_dense<Scalar>(A, opts);
}

} // end namespace