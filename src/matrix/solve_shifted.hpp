#pragma once
/**
 * @file solve_shifted.hpp
 * @brief Solve the shifted linear system (A - λ I) x = b using Eigen.
 */

#include "../matrix/matrix.hpp"  
#include "../core/types.hpp"   

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <stdexcept>
#include <typeinfo>

/**
 * @brief Solve the shifted system (A - λ I) x = b.
 *
 * This function works with both dense and sparse matrices stored inside the
 * Matrix wrapper:
 *
 * In both cases it forms
 *
 *      M = A - λ I
 *
 * and then solves M x = b.
 *
 * @tparam Scalar  Numeric type satisfying ScalarConcept.
 *
 * @param A_wrapped   Matrix wrapper holding the system matrix A.
 * @param shift       Scalar shift applied to the diagonal of A.
 * @param b           Right-hand side vector b (size n).
 *
 * @return Vector<Scalar>  Solution vector x.
 *
 * @throw std::runtime_error if:
 *   - A_wrapped is empty,
 *   - scalar type of A_wrapped does not match Scalar,
 *   - A is not square,
 *   - dimensions of A and b are incompatible,
 *   - or the sparse solver reports failure.
 * @throw std::bad_cast if the stored type inside Matrix is not the expected
 *         Eigen dense or sparse matrix type.
 */
template <ScalarConcept Scalar>
Vector<Scalar>
solve_shifted(
    const Matrix& A_wrapped,
    const Scalar shift,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b
) {
    // ---- Basic check common to both dense and sparse ----
    if (A_wrapped.scalar_type() != typeid(Scalar)) {
        throw std::runtime_error("solve_shifted: scalar type mismatch");
    }

    // ======================================================
    // Dense case
    // ======================================================
    if (A_wrapped.isDense()) {
        using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        const DenseMat& A = A_wrapped.cast<DenseMat>();

        if (A.rows() != A.cols()) {
            throw std::runtime_error("solve_shifted: A must be square (dense case)");
        }
        if (A.rows() != b.size()) {
            throw std::runtime_error("solve_shifted: size mismatch between A and b (dense case)");
        }

        // Form M = A - λ I
        DenseMat M = A;
        M.diagonal().array() -= shift;

        Eigen::PartialPivLU<DenseMat> lu(M);
        return lu.solve(b);
    }

    // ======================================================
    // Sparse case
    // ======================================================
    using SparseMat = Eigen::SparseMatrix<Scalar>;  // canonical sparse type
    const SparseMat& A = A_wrapped.cast<SparseMat>();

    if (A.rows() != A.cols()) {
        throw std::runtime_error("solve_shifted: A must be square (sparse case)");
    }
    if (A.rows() != b.size()) {
        throw std::runtime_error("solve_shifted: size mismatch between A and b (sparse case)");
    }

    // Form M = A - λ I
    SparseMat M = A;

    // Subtract λ from each diagonal entry: M(i,i) -= λ.
    // coeffRef inserts the entry if it does not exist yet.
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        M.coeffRef(i, i) -= shift;
    }

    Eigen::SparseLU<SparseMat> solver;
    solver.analyzePattern(M);
    solver.factorize(M);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_shifted: SparseLU factorization failed");
    }

    Vector<Scalar> x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_shifted: SparseLU solve failed");
    }

    return x;
}
