#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <complex>  
#include <cmath>
#include "../core/types.hpp" 
#include "../matrix/matrix.hpp"

namespace EigSol {

/**
 * @brief Reduce a dense matrix A to upper Hessenberg form H using Householder reflections.
 *
 * @tparam Scalar  Floating-point type satisfying ScalarConcept.
 * @param A        Input square matrix.
 * @return         Upper Hessenberg matrix H similar to A.
 */
template <ScalarConcept Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
to_hessenberg_dense(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
{
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    const Eigen::Index n = A.rows();

    if (A.rows() != A.cols()) {
        throw std::runtime_error("to_hessenberg_dense: A must be square");
    }

    MatrixType H = A;  // work on a copy

    // Classic Hessenberg reduction using Householder reflections
    for (Eigen::Index k = 0; k < n - 2; ++k) { // k: current column
        const Eigen::Index m = n - k - 1;  // length of column segment below diagonal

        // x = column k, rows k+1..n-1
        Vector<Scalar> x = H.block(k + 1, k, m, 1);

        // Build Householder vector v to zero out x(1..end)
        Real norm_x = x.norm();                  // real norm
        if (x.tail(m - 1).norm() == Real(0)) {
            continue;
        }

        // sign = x(0)/|x(0)| for complex, or ±1 for real
        Scalar sign = Scalar(1);
        if (x(0) != Scalar(0)) {
            using std::abs;
            sign = x(0) / Scalar(abs(x(0)));     // unit-modulus phase of x(0)
        }

        Scalar alpha = -sign * Scalar(norm_x);   

        Vector<Scalar> v = x;
        v(0) -= alpha;

        Real vnorm = v.norm();
        if (vnorm == Real(0)) {
            continue;                            // guard against numerical issues
        }
        v /= Scalar(vnorm);                      // normalize v 

        // Left multiplication: H(k+1:n-1, k:n-1) -= 2 v (v* H_sub)
        auto H_sub_left = H.block(k + 1, k, m, n - k); // subdiagonal block affected by left multiplication
        auto tempLeft = v * (v.adjoint() * H_sub_left);
        H_sub_left -= Scalar(2) * tempLeft; // all entries below H(k+1,k) are zero

        // Right multiplication: H(0:n-1, k+1:n-1) -= 2 (H_sub v) v*
        auto H_sub_right = H.block(0, k + 1, n, m); // rows 0..n-1, cols k+1..n-1
        auto tempRight = H_sub_right * v;
        H_sub_right -= Scalar(2) * (tempRight * v.adjoint()); // similarity transformation to preserve eigenvalues
    }

    return H;
}

/**
 * @brief Reduce a Matrix wrapper to Hessenberg form and return an Eigen matrix.
 *
 * This only supports dense matrices stored as Matrix::Dense<Scalar>.
 * Internally it:
 *   1. checks that A_wrapped is dense and has scalar type Scalar,
 *   2. extracts the underlying Eigen matrix,
 *   3. applies to_hessenberg_dense(),
 *   4. returns the resulting Eigen matrix H.
 *
 * @tparam Scalar     Floating-point or complex type satisfying ScalarConcept.
 * @param A_wrapped   Matrix wrapper holding a dense matrix.
 * @return            Eigen::Matrix<Scalar,Dynamic,Dynamic> containing the Hessenberg H.
 *
 * @throws std::runtime_error if the matrix is not dense or scalar type mismatches.
 * @throws std::bad_cast      if the stored type is not Matrix::Dense<Scalar>.
 */
template <ScalarConcept Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
to_hessenberg(const Matrix& A_wrapped)
{
    // 1. Dense-only check
    if (!A_wrapped.isDense()) {
        throw std::runtime_error("to_hessenberg(Matrix): only dense matrices are supported");
    }

    // 2. Scalar type check
    if (A_wrapped.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("to_hessenberg(Matrix): scalar type mismatch");
    }

    // 3. Extract the underlying Eigen matrix:
    using DenseMat = Matrix::Dense<Scalar>;
    const DenseMat& A = A_wrapped.cast<DenseMat>();

    // 4. Compute Hessenberg form on the Eigen matrix and return it:
    return to_hessenberg_dense<Scalar>(A);
}

} // end namespace