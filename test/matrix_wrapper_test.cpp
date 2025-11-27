// This test file was mostly written by an AI tool.
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../matrix/matrix.hpp"   

using DenseMat = Eigen::MatrixXd;
using SparseMat = Eigen::SparseMatrix<double>;


// -------------------------------------------------------------
// Test 1: Construct from Eigen dense matrix
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromEigenDense)
{
    DenseMat A(2,2);
    A << 1.0, 2.0,
         3.0, 4.0;

    Matrix M(A);   // should store a BoxTyped<Eigen::MatrixXd>

    // Matrix must contain something
    ASSERT_TRUE(M.has_value());

    // Must be classified as dense
    EXPECT_TRUE(M.isDense());

    // Correct scalar type
    EXPECT_EQ(M.scalar_type(), typeid(double));

    // Retrieve underlying matrix using cast
    DenseMat& B = M.cast<DenseMat>();

    // Values should match original
    EXPECT_DOUBLE_EQ(B(0,0), 1.0);
    EXPECT_DOUBLE_EQ(B(1,1), 4.0);
}


// -------------------------------------------------------------
// Test 2: Construct from Eigen sparse matrix
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromEigenSparse)
{
    SparseMat S(3,3);
    S.insert(0,0) = 5.0;
    S.insert(1,2) = 7.0;

    Matrix M(S);    // should store a BoxTyped<SparseMat>

    ASSERT_TRUE(M.has_value());
    EXPECT_FALSE(M.isDense());
    EXPECT_EQ(M.scalar_type(), typeid(double));

    // Extract sparse matrix
    SparseMat& T = M.cast<SparseMat>();
    EXPECT_EQ(T.coeff(0,0), 5.0);
    EXPECT_EQ(T.coeff(1,2), 7.0);
}


// -------------------------------------------------------------
// Test 3: Construct from std::vector
// -------------------------------------------------------------
TEST(MatrixWrapperTest, ConstructFromStdVector)
{
    std::vector<double> vec = {1,2,3,4};
    Matrix M(vec, 2, 2);     // interpreted as row-major 2×2

    ASSERT_TRUE(M.has_value());
    EXPECT_TRUE(M.isDense());
    EXPECT_EQ(M.scalar_type(), typeid(double));

    DenseMat& A = M.cast<DenseMat>();
    EXPECT_DOUBLE_EQ(A(0,0), 1.0);
    EXPECT_DOUBLE_EQ(A(0,1), 2.0);
    EXPECT_DOUBLE_EQ(A(1,0), 3.0);
    EXPECT_DOUBLE_EQ(A(1,1), 4.0);
}


// -------------------------------------------------------------
// Test 4: Copy constructor -> must deep copy
// -------------------------------------------------------------
TEST(MatrixWrapperTest, CopyConstructorDeepCopy)
{
    DenseMat A(2,2);
    A << 1,2,3,4;

    Matrix M1(A);       // original
    Matrix M2(M1);      // deep copy

    // Modify the copy
    DenseMat& C2 = M2.cast<DenseMat>();
    C2(0,0) = 99.0;

    // Original must remain unchanged
    DenseMat& C1 = M1.cast<DenseMat>();
    EXPECT_DOUBLE_EQ(C1(0,0), 1.0);
    EXPECT_DOUBLE_EQ(C2(0,0), 99.0);
}


// -------------------------------------------------------------
// Test 5: Copy assignment -> must deep copy
// -------------------------------------------------------------
TEST(MatrixWrapperTest, CopyAssignmentDeepCopy)
{
    DenseMat A(2,2);
    A << 10,20,30,40;

    Matrix M1(A);
    Matrix M2;

    M2 = M1;   // assign

    DenseMat& B2 = M2.cast<DenseMat>();
    DenseMat& B1 = M1.cast<DenseMat>();

    // Mutate one
    B1(1,1) = 777;

    // Other must not change
    EXPECT_DOUBLE_EQ(B2(1,1), 40.0);
    EXPECT_DOUBLE_EQ(B1(1,1), 777.0);
}


// -------------------------------------------------------------
// Test 6: cast<T> throws bad_cast when wrong type given
// -------------------------------------------------------------
TEST(MatrixWrapperTest, CastThrowsOnWrongType)
{
    DenseMat A = DenseMat::Identity(3,3);
    Matrix M(A);

    // Expect std::bad_cast if casting to sparse
    EXPECT_THROW(M.cast<SparseMat>(), std::bad_cast);

    // Casting to correct type must *not* throw
    EXPECT_NO_THROW(M.cast<DenseMat>());
}


// -------------------------------------------------------------
// Test 7: type() and scalar_type()
// -------------------------------------------------------------
TEST(MatrixWrapperTest, TypeQueries)
{
    DenseMat A = DenseMat::Zero(2,2);
    Matrix M(A);

    EXPECT_EQ(M.type(), typeid(DenseMat));
    EXPECT_EQ(M.scalar_type(), typeid(double));
}


// -------------------------------------------------------------
// Test 8: has_value() works
// -------------------------------------------------------------
TEST(MatrixWrapperTest, HasValue)
{
    Matrix M1;     // empty
    EXPECT_FALSE(M1.has_value());

    DenseMat A = DenseMat::Ones(1,1);
    Matrix M2(A);
    EXPECT_TRUE(M2.has_value());
}
