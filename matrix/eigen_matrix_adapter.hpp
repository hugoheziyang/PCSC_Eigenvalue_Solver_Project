#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include "dense_matrix_base.hpp"

template<ScalarConcept Scalar>
class EigenMatrixAdapter : public DenseMatrixBase<Scalar> {
public:
    using ScalarType = typename DenseMatrixBase<Scalar>::ScalarType;
    using VectorType = typename DenseMatrixBase<Scalar>::VectorType;
    using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    EigenMatrixAdapter(DenseMatrix const& A): A_(A) {}

    int rows() const override {
        return static_cast<int>(A_.rows());
    }

    int cols() const override {
        return static_cast<int>(A_.cols());
    }

    VectorType multiply(VectorType const& x) const override {
        if (x.size() != cols()) {
            throw std::invalid_argument(
                "EigenMatrixAdapter::multiply dimension mismatch"
            );
        }
        return A_ * x;
    }

    VectorType solveShifted(VectorType const& rhs,
                            Scalar shift) const override {
        const int n = rows();
        const int m = cols();

        if (n != m) {
            throw std::invalid_argument(
                "EigenMatrixAdapter::solveShifted requires a square matrix"
            );
        }
        if (rhs.size() != n) {
            throw std::invalid_argument(
                "EigenMatrixAdapter::solveShifted dimension mismatch"
            );
        }

        DenseMatrix shifted = A_;
        shifted.diagonal().array() -= shift;

        return shifted.partialPivLu().solve(rhs);
    }

    Scalar get(int i, int j) const override {
        return A_(i, j);
    }

    void set(int i, int j, Scalar value) override {
        A_(i, j) = value;
    }

    // To test
    DenseMatrix const& eigenMatrix() const { return A_; }
    DenseMatrix&       eigenMatrix()       { return A_; }

protected:
    DenseMatrix A_;

};
