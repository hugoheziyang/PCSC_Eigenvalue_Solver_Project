// This test file was mostly written by an AI tool.
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

#include "../src/matrix/matrix.hpp"
#include "../src/power_method/power_method.hpp"
#include "../src/option/solver_option.hpp"
#include "../src/core/tolerance.hpp"


using DenseMat  = Eigen::MatrixXd;
using SparseMat = Eigen::SparseMatrix<double>;

// Small helper for comparing doubles with a relative tolerance
inline void expectCloseRelative(double value, double expected, double relTol)
{
    EXPECT_TRUE(is_close_relative(expected, value, relTol));
}

// Convenience to check that A * x ≈ lambda * x
static void expectEigenpair(const DenseMat& A,
                            const Eigen::VectorXd& x,
                            double lambda,
                            double relTol)
{
    Eigen::VectorXd lhs = A * x;
    Eigen::VectorXd rhs = lambda * x;
    ASSERT_EQ(lhs.size(), rhs.size());

    for (int i = 0; i < lhs.size(); ++i) {
        expectCloseRelative(lhs[i], rhs[i], relTol);
    }
}

// -------------------------------------------------------------
// Test 1: simple dense 2x2 matrix
// -------------------------------------------------------------
TEST(PowerMethodTest, DenseSimpleMatrix)
{
    DenseMat A(2, 2);
    A << 2.0, 0.0,
         0.0, 1.0;

    Matrix M(A);

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-10;

    auto result = powerMethod<double>(M, opts);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0); // result.iterations > 0

    expectCloseRelative(result.eigenvalue, 2.0, 1e-5);
    expectEigenpair(A, result.eigenvector, result.eigenvalue, 1e-5);
}

// -------------------------------------------------------------
// Test 2: sparse matrix should give the same dominant eigenvalue
// -------------------------------------------------------------
TEST(PowerMethodTest, SparseMatrix)
{
    DenseMat A_dense(2, 2);
    A_dense << 3.0, 1.0,
               0.0, 2.0;

    SparseMat A = A_dense.sparseView();
    Matrix M(A);

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-8;

    auto result = powerMethod<double>(M, opts);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0); // result.iterations > 0

    // Dominant eigenvalue of that upper triangular matrix is 3
    expectCloseRelative(result.eigenvalue, 3.0, 1e-6);
    expectEigenpair(A_dense, result.eigenvector, result.eigenvalue, 1e-6);
}

// -------------------------------------------------------------
// Test 3: non square matrix should throw
// -------------------------------------------------------------
TEST(PowerMethodTest, NonSquareMatrixThrows)
{
    DenseMat A(2, 3);
    Matrix M(A);

    SolverOptions opts;
    opts.maxIterations = 100;
    opts.tolerance     = 1e-6;

    EXPECT_THROW(powerMethod<double>(M, opts), std::runtime_error);
}

// -------------------------------------------------------------
// Test 4: small maxIterations, check iteration count
// -------------------------------------------------------------
TEST(PowerMethodTest, FewIterationsCanFailToConverge)
{
    DenseMat A(2, 2);
    A << 5.0, 1.0,
         1.0, 4.0;

    Matrix M(A);

    SolverOptions opts;
    opts.maxIterations = 1;      // deliberately tiny
    opts.tolerance     = 1e-12;

    auto result = powerMethod<double>(M, opts);

    // At least verify that the algorithm reports the expected iteration count
    EXPECT_EQ(result.iterations, opts.maxIterations);
}

// -------------------------------------------------------------
// Test 5: zero size matrix should throw
// -------------------------------------------------------------
TEST(PowerMethodTest, ZeroSizeMatrixThrows)
{
    DenseMat A(0, 0);
    Matrix M(A);

    SolverOptions opts;
    opts.maxIterations = 100;
    opts.tolerance     = 1e-6;

    EXPECT_THROW(powerMethod<double>(M, opts), std::runtime_error);
}
