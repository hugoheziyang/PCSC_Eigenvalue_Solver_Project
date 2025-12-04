// This test file was mostly written by an AI tool.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Adjust these include paths to your actual project structure.
#include "../src/matrix/matrix.hpp"
#include "../src/core/types.hpp"
#include "../src/power_method/solve_shifted.hpp"

// ---------------------------------------------------------------
// Helper: check that M x ≈ b in 2-norm, with a given tolerance.
// ---------------------------------------------------------------
template <typename MatrixType, typename VectorType>
void expect_linear_system_residual_small(
    const MatrixType& M,
    const VectorType& x,
    const VectorType& b,
    double tol = 1e-10
) {
    auto r = M * x - b;                   // residual r = Mx - b
    EXPECT_NEAR(r.norm(), 0.0, tol);      // ||r|| ≈ 0
}

// ---------------------------------------------------------------
// Dense test: A = I, so (A - λ I) = (1 - λ) I.
// For λ = 2, we have M = -I, so x should be -b.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, DenseIdentity) {
    using Scalar = double;
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Construct A = I_3
    DenseMat A = DenseMat::Identity(3, 3);

    // Wrap A in Matrix
    Matrix A_wrapped(A);

    // Shift λ = 2
    ShiftedOptions<Scalar> opts(Scalar(2));

    // Right-hand side b
    Vec b(3);
    b << 1.0, -2.0, 3.0;

    // Solve (A - λ I) x = b  =>  (-I) x = b  => x = -b
    Vec x = solve_shifted<Scalar>(A_wrapped, opts, b);

    // Expected solution is -b
    Vec x_expected = -b;

    for (int i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x(i), x_expected(i), 1e-12);
    }

    // Also check residual explicitly: (A - λ I) x ≈ b
    DenseMat M = A - Scalar(2) * DenseMat::Identity(3, 3);
    expect_linear_system_residual_small(M, x, b);
}

// ---------------------------------------------------------------
// Dense test: general 2x2 matrix, compare with Eigen's direct solve.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, DenseGeneral2x2) {
    using Scalar = double;
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Arbitrary 2x2 matrix
    DenseMat A(2, 2);
    A << 3.0, 1.0,
         0.0, 4.0;

    Matrix A_wrapped(A);

    // Choose a shift λ
    Scalar lambda = 1.5;
    ShiftedOptions<Scalar> opts(lambda);

    // Right-hand side
    Vec b(2);
    b << 2.0, -1.0;

    // Solve via our wrapper
    Vec x = solve_shifted<Scalar>(A_wrapped, opts, b);

    // Form M = A - λ I and solve with Eigen directly
    DenseMat M = A - lambda * DenseMat::Identity(2, 2);
    Vec x_ref = M.partialPivLu().solve(b);

    // Compare solutions
    for (int i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x(i), x_ref(i), 1e-12);
    }

    // Check residual
    expect_linear_system_residual_small(M, x, b);
}

// ---------------------------------------------------------------
// Sparse test: A = I (sparse), so (A - λ I) = (1 - λ) I.
// Again λ = 2 => M = -I, so x = -b.
// This also exercises the sparse branch + SparseLU.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, SparseIdentity) {
    using Scalar = double;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Construct sparse A = I_3
    SparseMat A(3, 3);
    A.setIdentity();  // fills diagonal with 1's

    // Wrap A in Matrix (will be canonicalized inside Matrix)
    Matrix A_wrapped(A);

    // Shift λ = 2
    ShiftedOptions<Scalar> opts(Scalar(2));

    // Right-hand side b
    Vec b(3);
    b << 1.0, 0.5, -4.0;

    // Solve (A - λ I) x = b  => (-I)x = b  => x = -b
    Vec x = solve_shifted<Scalar>(A_wrapped, opts, b);

    Vec x_expected = -b;
    for (int i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x(i), x_expected(i), 1e-12);
    }

    // For the residual check, build a dense version of M = A - λ I.
    DenseMat M_dense = DenseMat::Identity(3, 3);
    M_dense.diagonal().array() -= Scalar(2);  // now M_dense = -I
    expect_linear_system_residual_small(M_dense, x, b);
}
