#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>

#include "../src/matrix/matrix.hpp"
#include "../src/core/types.hpp"
#include "../src/matrix/solve_shifted.hpp"

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
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    // Construct A = I_3
    DenseMat A = DenseMat::Identity(3, 3);

    // Wrap A in Matrix
    EigSol::Matrix A_wrapped(A);

    // Shift λ = 2
    Scalar lambda = 2;

    // Right-hand side b
    Vec b(3);
    b << 1.0, -2.0, 3.0;

    // Solve (A - λ I) x = b  =>  (-I) x = b  => x = -b
    Vec x = EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b);

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
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    // Arbitrary 2x2 matrix
    DenseMat A(2, 2);
    A << 3.0, 1.0,
         0.0, 4.0;

    EigSol::Matrix A_wrapped(A);

    // Choose a shift λ
    Scalar lambda = 1.5;

    // Right-hand side
    Vec b(2);
    b << 2.0, -1.0;

    // Solve via our wrapper
    Vec x = EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b);

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
    using SparseMat = EigSol::Matrix::Sparse<Scalar>;
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    // Construct sparse A = I_3
    SparseMat A(3, 3);
    A.setIdentity();  // fills diagonal with 1's

    // Wrap A in Matrix (will be canonicalized inside Matrix)
    EigSol::Matrix A_wrapped(A);

    // Shift λ = 2
    Scalar lambda = 2;

    // Right-hand side b
    Vec b(3);
    b << 1.0, 0.5, -4.0;

    // Solve (A - λ I) x = b  => (-I)x = b  => x = -b
    Vec x = EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b);

    Vec x_expected = -b;
    for (int i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x(i), x_expected(i), 1e-12);
    }

    // For the residual check, build a dense version of M = A - λ I.
    DenseMat M_dense = DenseMat::Identity(3, 3);
    M_dense.diagonal().array() -= Scalar(2);  // now M_dense = -I
    expect_linear_system_residual_small(M_dense, x, b);
}

// ---------------------------------------------------------------
// Dense complex test: compare with Eigen direct solve.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, DenseComplex2x2) {
    using Scalar = std::complex<double>;
    using SparseMat = EigSol::Matrix::Sparse<Scalar>;
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    // A is arbitrary 2x2 complex matrix
    DenseMat A(2, 2);
    A << Scalar(1.0, 1.0), Scalar(2.0, -1.0),
         Scalar(0.5, 0.0), Scalar(3.0,  2.0);

    EigSol::Matrix A_wrapped(A);

    Scalar lambda(0.7, -0.3);

    Vec b(2);
    b << Scalar(1.0, 0.0), Scalar(-2.0, 1.0);

    // Solve via our wrapper
    Vec x = EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b);

    // Reference: form M = A - λ I and use Eigen directly
    DenseMat M = A - lambda * DenseMat::Identity(2, 2);
    Vec x_ref  = M.partialPivLu().solve(b);

    for (int i = 0; i < x.size(); ++i) {
        using std::abs;
        EXPECT_NEAR(abs(x(i) - x_ref(i)), 0.0, 1e-10);
    }

    // Residual check: ||Mx - b|| small
    auto r = M * x - b;
    double res_norm = r.norm();
    EXPECT_NEAR(res_norm, 0.0, 1e-10);
}

// ---------------------------------------------------------------
// Error path: non-square dense A should throw.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, ThrowsOnNonSquareDense) {
    using Scalar   = double;
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    DenseMat A(2, 3);
    A.setRandom();

    EigSol::Matrix A_wrapped(A);
    Scalar lambda = 1.0;

    Vec b(2);
    b.setOnes();

    EXPECT_THROW(
        EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b),
        std::runtime_error
    );
}

// ---------------------------------------------------------------
// Error path: non-square sparse A should throw.
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, ThrowsOnNonSquareSparse) {
    using Scalar   = double;
    using SparseMat = EigSol::Matrix::Sparse<Scalar>;
    using Vec       = EigSol::Vector<Scalar>;

    SparseMat A(2, 3);
    A.insert(0, 0) = 1.0;
    A.insert(1, 2) = 2.0;

    EigSol::Matrix A_wrapped(A);
    Scalar lambda = 0.5;

    Vec b(2);
    b << 1.0, -1.0;

    EXPECT_THROW(
        EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b),
        std::runtime_error
    );
}

// ---------------------------------------------------------------
// Error path: size mismatch between A and b (dense).
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, ThrowsOnSizeMismatchDense) {
    using Scalar   = double;
    using DenseMat = EigSol::Matrix::Dense<Scalar>;
    using Vec = EigSol::Vector<Scalar>;

    DenseMat A = DenseMat::Identity(3, 3);
    EigSol::Matrix A_wrapped(A);
    Scalar lambda = 1.0;

    Vec b(2);      // wrong size
    b.setOnes();

    EXPECT_THROW(
        EigSol::solve_shifted<Scalar>(A_wrapped, lambda, b),
        std::runtime_error
    );
}

// ---------------------------------------------------------------
// Error path: scalar type mismatch (Matrix<double> vs solve_shifted<complex>).
// ---------------------------------------------------------------
TEST(SolveShiftedLinearSystem, ThrowsOnScalarTypeMismatch) {
    using RealScalar    = double;
    using ComplexScalar = std::complex<double>;

    using DenseMatReal = EigSol::Matrix::Dense<RealScalar>;
    using VecComplex   = EigSol::Vector<ComplexScalar>;

    DenseMatReal A(2, 2);
    A << 1.0, 2.0,
         3.0, 4.0;

    EigSol::Matrix A_wrapped(A);  // scalar_type() == typeid(double)

    ComplexScalar lambda = ComplexScalar(1.0, 0.0);

    VecComplex b(2);
    b << ComplexScalar(1.0, 0.0),
         ComplexScalar(0.0, 1.0);

    EXPECT_THROW(
        EigSol::solve_shifted<ComplexScalar>(A_wrapped, lambda, b),
        std::runtime_error
    );
}
