#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include "../core/types.hpp" 
#include "../matrix/matrix.hpp"


/**
 * @brief Compute QR decomposition of a dense matrix A using Householder reflections.
 *
 * A = Q * R, with Q unitary (orthogonal in the real case) and R upper-triangular.
 *
 * @tparam Scalar  Floating-point or complex type satisfying ScalarConcept.
 * @param A        Input m×n matrix.
 * @param Q        Output m×m unitary matrix.
 * @param R        Output m×n upper-triangular matrix.
 */
template <ScalarConcept Scalar>
void qr_decompose_dense(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R
) {
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    const Eigen::Index m = A.rows();
    const Eigen::Index n = A.cols();

    if (m == 0 || n == 0) {
        throw std::runtime_error("qr_decompose_dense: empty matrix");
    }

    R = A;
    Q.setIdentity(m, m);

    const Eigen::Index kmax = std::min(m, n);

    for (Eigen::Index k = 0; k < kmax; ++k) {
        const Eigen::Index rows_k = m - k;

        // x = column k starting at row k
        VectorType x = R.block(k, k, rows_k, 1);

        Real norm_x = x.norm();
        if (x.tail(rows_k - 1).norm() == Real(0)) {
            continue;
        }

        // sign = x(0)/|x(0)| for complex, ±1 for real
        Scalar sign = Scalar(1);
        if (x(0) != Scalar(0)) {
            using std::abs;
            sign = x(0) / Scalar(abs(x(0)));
        }

        Scalar alpha = -sign * Scalar(norm_x);

        VectorType v = x;
        v(0) -= alpha;

        Real vnorm = v.norm();
        if (vnorm == Real(0)) {
            continue;
        }
        v /= Scalar(vnorm); // normalize v

        // Apply reflector from the left to R: R_k := R_k - 2 v (v* R_k)
        auto R_sub = R.block(k, k, rows_k, n - k); // rows k..m-1, cols k..n-1
        MatrixType tempR = v * (v.adjoint() * R_sub);
        R_sub -= Scalar(2) * tempR;

        // Apply reflector from the right to Q: Q := Q - 2 (Q v) v*
        auto Q_sub = Q.block(0, k, m, rows_k); // rows 0..m-1, cols k..m-1
        MatrixType tempQ = Q_sub * v;
        Q_sub -= Scalar(2) * (tempQ * v.adjoint());
    }
}

/**
 * @brief QR decomposition for the Matrix wrapper (dense matrices only).
 *
 * Given a Matrix A_wrapped holding a dense matrix of scalar type Scalar,
 * this function computes Q, R such that:
 *
 *      A = Q * R
 *
 * and returns them as Eigen matrices.
 *
 * @tparam Scalar  Floating-point type satisfying ScalarConcept.
 * @param A_wrapped  Matrix wrapper holding a dense matrix.
 * @return std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> containing (Q, R).
 *
 * @throws std::runtime_error if the matrix is not dense or scalar type mismatches.
 * @throws std::bad_cast      if the stored type is not Matrix::Dense<Scalar>.
 */
template <ScalarConcept Scalar>
auto qr_decompose(const Matrix& A_wrapped)
    -> std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
{
    // 1. Only dense supported
    if (!A_wrapped.isDense()) {
        throw std::runtime_error("qr_decompose(Matrix): only dense matrices are supported");
    }

    // 2. Scalar type must match
    if (A_wrapped.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("qr_decompose(Matrix): scalar type mismatch");
    }

    using DenseMat = Matrix::Dense<Scalar>;

    // 3. Extract underlying Eigen dense matrix
    const DenseMat& A = A_wrapped.cast<DenseMat>();

    // 4. Allocate Q and R
    DenseMat Q(A.rows(), A.rows());
    DenseMat R(A.rows(), A.cols());

    // 5. Compute QR with householder routine
    qr_decompose_dense<Scalar>(A, Q, R);

    return { Q, R };
}