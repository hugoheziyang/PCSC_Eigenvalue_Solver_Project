#pragma once
/**
 * @file solve_shifted.hpp
 * @brief Solve the shifted linear system (A - λ I) x = b using Eigen.
 */

#include "../matrix/matrix.hpp"  
#include "../option/shifted_option.hpp" 
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
 *  - If A_wrapped.isDense() is true, it expects the underlying type to be
 *      Eigen::Matrix<Scalar, Dynamic, Dynamic>
 *    and uses Eigen::PartialPivLU for solving.
 *
 *  - If A_wrapped.isDense() is false, it expects the underlying type to be
 *      Eigen::SparseMatrix<Scalar>
 *    (the canonical sparse type used by Matrix) and uses Eigen::SparseLU.
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
 * @param opts        Shift options (contains the shift λ).
 * @param b           Right-hand side vector b (size n).
 *
 * @return Eigen::Matrix<Scalar, Dynamic, 1>  Solution vector x.
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
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
solve_shifted(
    const Matrix& A_wrapped,
    const ShiftedOptions<Scalar>& opts,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b
) {
    // ---- Basic checks common to both dense and sparse ----
    if (!A_wrapped.has_value()) {
        throw std::runtime_error("solve_shifted: A is empty");
    }

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
        M.diagonal().array() -= opts.shift;

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
        M.coeffRef(i, i) -= opts.shift;
    }

    Eigen::SparseLU<SparseMat> solver;
    solver.analyzePattern(M);
    solver.factorize(M);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_shifted: SparseLU factorization failed");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_shifted: SparseLU solve failed");
    }

    return x;
}
