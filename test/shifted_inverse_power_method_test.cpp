// This test file was mostly written by an AI tool.
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

#include "../src/matrix/matrix.hpp"
#include "../src/power_method/shifted_inverse_power_solver.hpp"
#include "../src/option/shifted_solver_option.hpp"
#include "../src/core/tolerance.hpp"

using DenseMat  = EigSol::Matrix::Dense<double>;
using SparseMat = EigSol::Matrix::Sparse<double>;

// Small helper for comparing doubles with a relative tolerance
inline void expectCloseRelative(double value, double expected, double relTol)
{
    EXPECT_TRUE(EigSol::is_close_relative(expected, value, relTol));
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
// Test 1: simple dense 2x2 diagonal matrix with shift near 2
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, DenseShiftNearFirstEigenvalue)
{
    DenseMat A(2, 2);
    A << 2.0, 0.0,
         0.0, 5.0;

    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 1.9;      // shift close to 2
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-10;

    auto result = shiftedInversePowerMethod<double>(M, opts);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0); // Verifies that result.iterations >= 0

    expectCloseRelative(result.eigenvalue, 2.0, 1e-5);
    expectEigenpair(A, result.eigenvector, result.eigenvalue, 1e-5);
}

// -------------------------------------------------------------
// Test 2: same dense matrix with shift near 5
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, DenseShiftNearSecondEigenvalue)
{
    DenseMat A(2, 2);
    A << 2.0, 0.0,
         0.0, 5.0;

    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 4.9;      // shift close to 5
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-10;

    auto result = EigSol::shiftedInversePowerMethod<double>(M, opts);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0); // Verifies that result.iterations >= 0

    expectCloseRelative(result.eigenvalue, 5.0, 1e-5);
    expectEigenpair(A, result.eigenvector, result.eigenvalue, 1e-5);
}

// -------------------------------------------------------------
// Test 3: sparse diagonal matrix should behave similarly
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, SparseMatrix)
{
    DenseMat A_dense = DenseMat::Zero(3, 3);
    A_dense(0, 0) = 1.0;
    A_dense(1, 1) = 3.0;
    A_dense(2, 2) = 10.0;

    SparseMat A = A_dense.sparseView();
    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 2.9;      // shift close to eigenvalue 3
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-8;

    auto result = EigSol::shiftedInversePowerMethod<double>(M, opts);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0); // Verifies that result.iterations >= 0

    expectCloseRelative(result.eigenvalue, 3.0, 1e-5);
    expectEigenpair(A_dense, result.eigenvector, result.eigenvalue, 1e-5);
}

// -------------------------------------------------------------
// Test 4: non square matrix should throw
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, NonSquareMatrixThrows)
{
    DenseMat A(2, 3);
    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 1.0;
    opts.maxIterations = 100;
    opts.tolerance     = 1e-6;

    EXPECT_THROW(
        EigSol::shiftedInversePowerMethod<double>(M, opts),
        std::runtime_error
    );
}

// -------------------------------------------------------------
// Test 5: zero size matrix should throw
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, ZeroSizeMatrixThrows)
{
    DenseMat A(0, 0);
    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 0.0;
    opts.maxIterations = 100;
    opts.tolerance     = 1e-6;

    EXPECT_THROW(
        EigSol::shiftedInversePowerMethod<double>(M, opts),
        std::runtime_error
    );
}

// -------------------------------------------------------------
// Test 6: too few iterations can fail to converge but iteration count is reported
// -------------------------------------------------------------
TEST(ShiftedInversePowerMethodTest, FewIterationsCanFailToConverge)
{
    DenseMat A(2, 2);
    A << 5.0, 1.0,
         1.0, 4.0;

    EigSol::Matrix M(A);

    EigSol::ShiftedSolverOptions<double> opts;
    opts.shift         = 4.0;
    opts.maxIterations = 1;        // deliberately tiny
    opts.tolerance     = 1e-12;

    auto result = EigSol::shiftedInversePowerMethod<double>(M, opts);

    // At least verify that the algorithm reports the expected iteration count
    EXPECT_EQ(result.iterations, opts.maxIterations);
}
