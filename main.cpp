#include <iostream>
#include <Eigen/Core>

#include "core/types.hpp"
#include "matrix/eigen_matrix_adapter.hpp"

int main() {
    using Scalar = double;
    using VectorType = Vector<Scalar>;
    using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    DenseMatrix A(3, 3);
    A << 4.0, 1.0, 0.0,
         1.0, 3.0, 1.0,
         0.0, 1.0, 2.0;

    EigenMatrixAdapter<Scalar> adapter(A);

    VectorType x(3);
    x << 1.0, 2.0, 3.0;

    VectorType y = adapter.multiply(x);
    std::cout << "A * x =\n" << y << "\n\n";

    VectorType rhs(3);
    rhs << 1.0, 0.0, 0.0;

    Scalar shift = 0.5;
    VectorType sol = adapter.solveShifted(rhs, shift);

    std::cout << "Solution of (A - shift I) x = rhs\n";
    std::cout << sol << "\n\n";

    std::cout << "A(1, 1) = " << adapter.get(1, 1) << "\n";

    adapter.set(1, 1, 10.0);
    std::cout << "A modified\n" << adapter.eigenMatrix() << "\n";

    return 0;
}
