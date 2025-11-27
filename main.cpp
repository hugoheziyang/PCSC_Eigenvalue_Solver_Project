/**
 * @file main.cpp
 * @brief Minimal demonstration of the Matrix wrapper and solver interface.
 *
 * NOTE:
 * This file exists only as an example entry point. In normal use,
 * end-users will write their own main() and simply link against
 * the PCSC_Eigenvalue_Solver_Project library.
 *
 * You may freely modify or remove this file depending on your needs.
 */

#include <iostream>
#include <Eigen/Dense>

#include "matrix/matrix.hpp"   // Wrapper class

int main()
{
    std::cout << "=== Example: Matrix Wrapper Demo ===\n";

    // Construct an Eigen dense matrix
    Eigen::MatrixXd A(2,2);
    A << 4.0, 1.0,
         2.0, 3.0;

    // Wrap into your Matrix class
    Matrix M(A);

    std::cout << "Stored matrix is dense? " << std::boolalpha << M.isDense() << "\n";
    std::cout << "Scalar type: " << M.scalar_type().name() << "\n";

    // Retrieve underlying matrix via cast<T>()
    Eigen::MatrixXd& R = M.cast<Eigen::MatrixXd>();

    std::cout << "Matrix contents:\n" << R << "\n";

    std::cout << "Demo completed.\n";
    return 0;
}
