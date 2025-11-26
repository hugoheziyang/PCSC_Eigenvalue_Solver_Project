#include <gtest/gtest.h>
#include <Eigen/Core>

#include "core/types.hpp"
#include "matrix/matrix_base.hpp"
#include "matrix/dense_matrix_base.hpp"
#include "matrix/eigen_matrix_adapter.hpp"

using Scalar      = double;
using VectorType  = Vector<Scalar>;
using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

TEST(EigenMatrixAdapterTest, Dimensions) {
    DenseMatrix A(3, 2);
    A << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    EigenMatrixAdapter<Scalar> adapter(A);

    EXPECT_EQ(adapter.rows(), 3);
    EXPECT_EQ(adapter.cols(), 2);
}

TEST(EigenMatrixAdapterTest, MultiplySimpleVector) {
    DenseMatrix A(3, 3);
    A << 4.0, 1.0, 0.0,
         1.0, 3.0, 1.0,
         0.0, 1.0, 2.0;

    EigenMatrixAdapter<Scalar> adapter(A);
    MatrixBase<Scalar>& mat = adapter;

    VectorType x(3);
    x << 1.0, 2.0, 3.0;

    VectorType y = mat.multiply(x);

    ASSERT_EQ(y.size(), 3);

    EXPECT_DOUBLE_EQ(y(0), 4.0 * 1.0 + 1.0 * 2.0 + 0.0 * 3.0);
    EXPECT_DOUBLE_EQ(y(1), 1.0 * 1.0 + 3.0 * 2.0 + 1.0 * 3.0);
    EXPECT_DOUBLE_EQ(y(2), 0.0 * 1.0 + 1.0 * 2.0 + 2.0 * 3.0);
}

TEST(EigenMatrixAdapterTest, SolveShiftedSystem) {
    DenseMatrix A(2, 2);
    A << 3.0, 1.0,
         1.0, 2.0;

    EigenMatrixAdapter<Scalar> adapter(A);
    MatrixBase<Scalar>& mat = adapter;

    VectorType rhs(2);
    rhs << 1.0, 0.0;

    Scalar shift = 0.5;
    VectorType x = mat.solveShifted(rhs, shift);

    DenseMatrix shifted = A;
    shifted.diagonal().array() -= shift;

    VectorType check = shifted * x;

    ASSERT_EQ(check.size(), 2);
    EXPECT_NEAR(check(0), rhs(0), 1e-10);
    EXPECT_NEAR(check(1), rhs(1), 1e-10);
}

TEST(EigenMatrixAdapterTest, GetSetElements) {
    DenseMatrix A(2, 2);
    A << 1.0, 2.0,
         3.0, 4.0;

    EigenMatrixAdapter<Scalar> adapter(A);
    DenseMatrixBase<Scalar>& dmat = adapter;

    EXPECT_DOUBLE_EQ(dmat.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dmat.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(dmat.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(dmat.get(1, 1), 4.0);

    dmat.set(0, 1, 10.0);
    EXPECT_DOUBLE_EQ(dmat.get(0, 1), 10.0);
}
