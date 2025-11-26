#pragma once

#include <stdexcept>
#include "../core/types.hpp"

/** Base interface for all matrix types (dense or sparse) */
template<ScalarConcept Scalar>
class MatrixBase {
public:
    using ScalarType = Scalar;
    using VectorType = Vector<Scalar>;

    virtual ~MatrixBase() = default;

    /// Number of rows
    virtual int rows() const = 0;

    /// Number of columns
    virtual int cols() const = 0;

    /// y = A * x
    virtual VectorType multiply(VectorType const& x) const = 0;

    /// Solve (A - shift * I) x = rhs
    virtual VectorType solveShifted(VectorType const& rhs,
                                    Scalar shift) const = 0;
};
