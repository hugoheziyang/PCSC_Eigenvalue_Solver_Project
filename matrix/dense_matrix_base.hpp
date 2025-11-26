#pragma once

#include "../core/types.hpp"
#include "matrix_base.hpp"

/** Base interface for dense matrices with element access */
template<ScalarConcept Scalar>
class DenseMatrixBase : public MatrixBase<Scalar> {
public:
    using ScalarType = typename MatrixBase<Scalar>::ScalarType;
    using VectorType = typename MatrixBase<Scalar>::VectorType;

    /// Get A(i, j)
    virtual Scalar get(int i, int j) const = 0;

    /// Set A(i, j) = value
    virtual void set(int i, int j, Scalar value) = 0;

    virtual ~DenseMatrixBase() = default;
};
