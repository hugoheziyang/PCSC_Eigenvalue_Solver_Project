// This test file was mostly written by an AI tool.
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <type_traits>

#include "../src/matrix/matrix.hpp"

using DenseMat  = EigSol::Matrix::Dense<double>;
using SparseMat = EigSol::Matrix::Sparse<double>;

// -------------------------------------------------------------
// Static checks: Matrix invariants
// -------------------------------------------------------------
static_assert(!std::is_default_constructible_v<EigSol::Matrix>,
              "Matrix must not be default constructible");
static_assert(!std::is_copy_constructible_v<EigSol::Matrix>,
              "Matrix must not be copy constructible");
static_assert(!std::is_copy_assignable_v<EigSol::Matrix>,
              "Matrix must not be copy assignable");
static_assert(!std::is_move_constructible_v<EigSol::Matrix>,
              "Matrix must not be move constructible");
static_assert(!std::is_move_assignable_v<EigSol::Matrix>,
              "Matrix must not be move assignable");

// -------------------------------------------------------------
// Test 1: Construct from Eigen dense matrix
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromEigenDense)
{
    DenseMat A(2, 2);
    A << 1.0, 2.0,
         3.0, 4.0;

    EigSol::Matrix M(A);   // should store a canonical dense matrix

    // Must be classified as dense
    EXPECT_TRUE(M.isDense());

    // Correct scalar type
    EXPECT_EQ(M.scalar_type(), typeid(double));

    // Retrieve underlying matrix using cast
    DenseMat& B = M.cast<DenseMat>();

    // Values should match original
    EXPECT_DOUBLE_EQ(B(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(B(1, 1), 4.0);
}

// -------------------------------------------------------------
// Test 2: Construct from Eigen sparse matrix
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromEigenSparse)
{
    SparseMat S(3, 3);
    S.insert(0, 0) = 5.0;
    S.insert(1, 2) = 7.0;

    EigSol::Matrix M(S);    // should store a canonical sparse matrix

    EXPECT_FALSE(M.isDense());
    EXPECT_EQ(M.scalar_type(), typeid(double));

    // Extract sparse matrix
    SparseMat& T = M.cast<SparseMat>();
    EXPECT_EQ(T.coeff(0, 0), 5.0);
    EXPECT_EQ(T.coeff(1, 2), 7.0);
}

// -------------------------------------------------------------
// Test 3: Construct from std::vector
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromStdVector)
{
    std::vector<double> vec = {1, 2, 3, 4};
    EigSol::Matrix M(vec, 2, 2);     // interpreted as row-major 2×2

    EXPECT_TRUE(M.isDense());
    EXPECT_EQ(M.scalar_type(), typeid(double));

    DenseMat& A = M.cast<DenseMat>();
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
}

// -------------------------------------------------------------
// Test 4: cast<T> throws bad_cast when wrong type given
// -------------------------------------------------------------
TEST(MatrixWrapperTest, CastThrowsOnWrongType)
{
    DenseMat A = DenseMat::Identity(3, 3);
    EigSol::Matrix M(A);

    // Expect std::bad_cast if casting to sparse
    EXPECT_THROW(M.cast<SparseMat>(), std::bad_cast);

    // Casting to correct type must not throw
    EXPECT_NO_THROW(M.cast<DenseMat>());
}

// -------------------------------------------------------------
// Test 5: type() and scalar_type()
// -------------------------------------------------------------
TEST(MatrixWrapperTest, TypeQueries)
{
    DenseMat A = DenseMat::Zero(2, 2);
    EigSol::Matrix M(A);

    EXPECT_EQ(M.type(), typeid(DenseMat));
    EXPECT_EQ(M.scalar_type(), typeid(double));
}
