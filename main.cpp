// main.cpp
// Demo: lire deux matrices depuis des fichiers
// puis appliquer power method et shifted inverse power method.

// This example file was mostly written by an AI tool.
/**
 * @file main.cpp
 * @brief Minimal demo for power and shifted inverse power methods.
 */

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include "src/core/types.hpp"
#include "src/core/tolerance.hpp"

#include "src/matrix/matrix.hpp"

#include "src/option/solver_option.hpp"
#include "src/option/shifted_solver_option.hpp"

#include "src/result/eigen_result.hpp"

#include "src/power_method/power_method.hpp"
#include "src/power_method/shifted_inverse_power_solver.hpp"

#include "src/qr_method/to_hessenberg.hpp"
#include "src/qr_method/qr_decompose.hpp"
#include "src/qr_method/qr_eigenvalues.hpp"

#include "src/reader/file_matrix_reader.hpp"


template <typename VectorType>
void printVector(const VectorType& v, const std::string& name) {
    std::cout << name << std::endl;
    std::cout << "(" << v.transpose() << ")" << std::endl << std::endl;
}

int main() {
    using Scalar = double;

    const std::string fileA   = "../data/A.txt";
    const std::string fileB = "../data/B.txt";

    Matrix A = readMatrixFromFile<Scalar>(fileA);
    Matrix B = readMatrixFromFile<Scalar>(fileB);

    std::cout << "===== Power method =====" << std::endl;

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance = 1e-10;

    auto powerResultA = powerMethod<Scalar>(A, opts);
    std::cout << "Matrix A" << std::endl;
    std::cout << "Converged : " << std::boolalpha << powerResultA.converged << std::endl;
    std::cout << "Iterations: " << powerResultA.iterations << std::endl;
    std::cout << "Eigenvalue: " << powerResultA.eigenvalue << std::endl;
    printVector(powerResultA.eigenvector, "Eigenvector:");

    auto powerResultB = powerMethod<Scalar>(B, opts);
    std::cout << "Matrix B" << std::endl;
    std::cout << "Converged : " << std::boolalpha << powerResultB.converged << std::endl;
    std::cout << "Iterations: " << powerResultB.iterations << std::endl;
    std::cout << "Eigenvalue: " << powerResultB.eigenvalue << std::endl;
    printVector(powerResultB.eigenvector, "Eigenvector:");


    std::cout << "===== Shifted inverse power method =====" << std::endl;

    ShiftedSolverOptions<Scalar> shiftedOptsA;
    shiftedOptsA.shift = 3.1;
    shiftedOptsA.maxIterations = 1000;
    shiftedOptsA.tolerance = 1e-12;

    auto shiftInvPowResultA = shiftedInversePowerMethod<Scalar>(A, shiftedOptsA);

    std::cout << "Matrix A" << std::endl;
    std::cout << "Converged                : " << std::boolalpha << shiftInvPowResultA.converged << std::endl;
    std::cout << "Iterations               : " << shiftInvPowResultA.iterations << std::endl;
    std::cout << "Eigenvalue near the shift: " << shiftInvPowResultA.eigenvalue << std::endl;
    printVector(shiftInvPowResultA.eigenvector, "Eigenvector:");

    ShiftedSolverOptions<Scalar> shiftedOptsB;
    shiftedOptsB.shift         = 2.3;
    shiftedOptsB.maxIterations = 1000;
    shiftedOptsB.tolerance     = 1e-12;

    auto shiftInvPowResultB = shiftedInversePowerMethod<Scalar>(B, shiftedOptsB);

    std::cout << "Matrix B" << std::endl;
    std::cout << "Converged                : " << std::boolalpha << shiftInvPowResultB.converged << std::endl;
    std::cout << "Iterations               : " << shiftInvPowResultB.iterations << std::endl;
    std::cout << "Eigenvalue near the shift: " << shiftInvPowResultB.eigenvalue << std::endl;
    printVector(shiftInvPowResultB.eigenvector, "Eigenvector:");

    
    std::cout << "===== QR eigenvalue method =====" << std::endl;

    // QR solver options
    SolverOptions qrOpts;
    qrOpts.maxIterations = 1000;
    qrOpts.tolerance     = 1e-10;

    try {
        std::cout << "Hessenberg reduction of Matrix A" << std::endl;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H_A = to_hessenberg<Scalar>(A);
        std::cout << "H(A) = " << std::endl << H_A << std::endl << std::endl;

        std::cout << "QR decomposition of Matrix A" << std::endl;
        auto [Q_A, R_A] = qr_decompose<Scalar>(A);
        std::cout << "Q_A = " << std::endl << Q_A << std::endl << std::endl;
        std::cout << "R_A = " << std::endl << R_A << std::endl << std::endl;
        std::cout << "Q_A * R_A (should approximate A) = " << std::endl
                  << Q_A * R_A << std::endl << std::endl;

        auto qrResultA = qr_eigenvalues<Scalar>(A, qrOpts);
        std::cout << "QR eigenvalues for Matrix A" << std::endl;
        std::cout << "Converged              : " << std::boolalpha << qrResultA.converged << std::endl;
        std::cout << "Iterations             : " << qrResultA.iterations << std::endl;
        std::cout << "Eigenvalues (diag of H): " << std::endl
                  << qrResultA.eigenvalues.transpose() << std::endl << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "QR-related computation failed for Matrix A: " << e.what() << std::endl;
    }

    try {
        auto qrResultB = qr_eigenvalues<Scalar>(B, qrOpts);
        std::cout << "QR eigenvalues for Matrix B" << std::endl;
        std::cout << "Converged              : " << std::boolalpha << qrResultB.converged << std::endl;
        std::cout << "Iterations             : " << qrResultB.iterations << std::endl;
        std::cout << "Eigenvalues (diag of H): " << std::endl
                  << qrResultB.eigenvalues.transpose() << std::endl << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "QR-related computation failed for Matrix B: " << e.what() << std::endl;
    }

    return 0;
}
