// This example file was mostly written by an AI tool.
/**
 * @file main.cpp
 * @brief Example program showing how to use all main features of our project.
 */

#include <iostream>
#include <iomanip>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "src/core/types.hpp"
#include "src/core/tolerance.hpp"

#include "src/box/box.hpp"
#include "src/box/box_typed.hpp"

#include "src/matrix/matrix.hpp"
#include "src/matrix/solve_shifted.hpp"

#include "src/option/solver_option.hpp"
#include "src/option/shifted_solver_option.hpp"

#include "src/result/eigen_result.hpp"
#include "src/result/qr_result.hpp"

#include "src/power_method/power_method.hpp"
#include "src/power_method/shifted_inverse_power_solver.hpp"

#include "src/qr_method/to_hessenberg.hpp"
#include "src/qr_method/qr_decompose.hpp"
#include "src/qr_method/qr_eigenvalues.hpp"

using Scalar     = double;
using Complex    = std::complex<double>;
using DenseMat   = Matrix::Dense<Scalar>;
using SparseMat  = Matrix::Sparse<Scalar>;
using Vec        = Vector<Scalar>;
using CplxDense  = Matrix::Dense<Complex>;
using CplxVec    = Vector<Complex>;

// Small helpers for printing
template <typename Derived>
void printDenseMatrix(
    const Eigen::MatrixBase<Derived>& M,
    const std::string& name
) {
    std::cout << name << " (" << M.rows() << "x" << M.cols() << ")\n";
    std::cout << M << "\n\n";
}

template <typename VectorType>
void printVector(
    const VectorType& v,
    const std::string& name
) {
    std::cout << name << " (size " << v.size() << ")\n";
    std::cout << v.transpose() << "\n\n";
}

void demoBox() {
    std::cout << "===== Box / BoxTyped demo =====\n";
    BoxTyped<int>   bi(42);
    BoxTyped<double> bd(3.14);

    Box& baseRef = bi;
    std::cout << "Stored type in BoxTyped<int>: "
              << baseRef.type().name() << "\n";

    auto clonePtr = baseRef.clone();
    std::cout << "Clone type: " << clonePtr->type().name() << "\n\n";

}

void demoMatrixWrapper() {
    std::cout << "===== Matrix wrapper demo (dense / sparse) =====\n";

    // Dense 3x3 symmetric tridiagonal matrix
    DenseMat A(3, 3);
    A << 2.0, -1.0,  0.0,
        -1.0,  2.0, -1.0,
         0.0, -1.0,  2.0;

    printDenseMatrix(A, "Dense A");

    Matrix Md(A);

    std::cout << "Md.isDense() = " << std::boolalpha << Md.isDense() << "\n";
    std::cout << "Md.scalar_type().name() = "
              << Md.scalar_type().name() << "\n\n";

    const DenseMat& A_back = Md.cast<DenseMat>();
    printDenseMatrix(A_back, "A recovered from Md");

    // Sparse version of the same matrix
    SparseMat S(3, 3);
    S.insert(0, 0) =  2.0;
    S.insert(0, 1) = -1.0;
    S.insert(1, 0) = -1.0;
    S.insert(1, 1) =  2.0;
    S.insert(1, 2) = -1.0;
    S.insert(2, 1) = -1.0;
    S.insert(2, 2) =  2.0;
    S.makeCompressed();

    Matrix Ms(S);
    std::cout << "Ms.isDense() = " << std::boolalpha << Ms.isDense() << "\n";
    std::cout << "Ms.scalar_type().name() = "
              << Ms.scalar_type().name() << "\n\n";

    const SparseMat& S_back = Ms.cast<SparseMat>();
    std::cout << "Sparse S has " << S_back.nonZeros()
              << " nonzero entries\n\n";
}

void demoSolveShiftedReal(const Matrix& M_dense, const Matrix& M_sparse) {
    std::cout << "===== solve_shifted demo (real) =====\n";

    const DenseMat& A = M_dense.cast<DenseMat>();

    Vec b(3);
    b << 1.0, 2.0, 3.0;

    double lambda = 1.0;

    Vec x_dense  = solve_shifted<Scalar>(M_dense,  lambda, b);
    Vec x_sparse = solve_shifted<Scalar>(M_sparse, lambda, b);

    printVector(b,        "Right-hand side b");
    printVector(x_dense,  "Solution x (dense backend)");
    printVector(x_sparse, "Solution x (sparse backend)");

    DenseMat Mmat = A - lambda * DenseMat::Identity(3, 3);
    Vec resid = Mmat * x_dense - b;
    std::cout << "Dense residual norm: " << resid.norm() << "\n\n";
}

void demoSolveShiftedComplex() {
    std::cout << "===== solve_shifted demo (complex) =====\n";

    CplxDense Ac(2, 2);
    Ac << Complex(1.0, 0.0), Complex(0.0, 1.0),
          Complex(0.0, -1.0), Complex(2.0, 0.0);

    Matrix Mc(Ac);

    CplxVec bc(2);
    bc << Complex(1.0, 0.0),
          Complex(0.0, 1.0);

    Complex lambda(0.5, 0.1);

    CplxVec xc = solve_shifted<Complex>(Mc, lambda, bc);

    printDenseMatrix(Ac, "Complex matrix Ac");
    printVector(bc,      "Right-hand side bc");
    printVector(xc,      "Solution xc");

    CplxDense M = Ac - lambda * CplxDense::Identity(2, 2);
    CplxVec rc = M * xc - bc;
    std::cout << "Complex residual norm: " << rc.norm() << "\n\n";
}

void demoPowerMethods(const Matrix& M_dense) {
    std::cout << "===== Power method and shifted inverse power method =====\n";

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-10;

    auto pmResult = powerMethod<Scalar>(M_dense, opts);

    std::cout << "Power method converged    : "
              << std::boolalpha << pmResult.converged << "\n";
    std::cout << "Power method iterations   : "
              << pmResult.iterations << "\n";
    std::cout << "Dominant eigenvalue (PM)  : "
              << pmResult.eigenvalue << "\n";
    printVector(pmResult.eigenvector, "Dominant eigenvector (PM)");

    ShiftedSolverOptions<Scalar> sinvOpts;
    sinvOpts.shift         = 2.9;   // close to a larger eigenvalue
    sinvOpts.maxIterations = 1000;
    sinvOpts.tolerance     = 1e-12;

    auto sipResult = shiftedInversePowerMethod<Scalar>(M_dense, sinvOpts);

    std::cout << "Shifted inverse PM converged : "
              << std::boolalpha << sipResult.converged << "\n";
    std::cout << "Shifted inverse PM iterations: "
              << sipResult.iterations << "\n";
    std::cout << "Eigenvalue near shift       : "
              << sipResult.eigenvalue << "\n\n";
}

void demoHessenberg(const DenseMat& A) {
    std::cout << "===== Hessenberg reduction =====\n";

    DenseMat H_dense = to_hessenberg_dense<Scalar>(A);

    printDenseMatrix(A,       "Original A");
    printDenseMatrix(H_dense, "Hessenberg H (dense API)");

    Matrix M(A);
    DenseMat H_wrap = to_hessenberg<Scalar>(M);
    printDenseMatrix(H_wrap,  "Hessenberg H (Matrix wrapper API)");
}

void demoQRDecompose(const DenseMat& A) {
    std::cout << "===== QR decomposition =====\n";

    DenseMat Q(A.rows(), A.rows());
    DenseMat R(A.rows(), A.cols());

    qr_decompose_dense<Scalar>(A, Q, R);

    printDenseMatrix(Q, "Q from qr_decompose_dense");
    printDenseMatrix(R, "R from qr_decompose_dense");

    DenseMat A_reconstructed = Q * R;
    printDenseMatrix(A_reconstructed, "Q * R");

    Matrix M(A);
    auto qr_pair = qr_decompose<Scalar>(M); // wrapper-based
    const DenseMat& Qw = qr_pair.first;
    const DenseMat& Rw = qr_pair.second;

    printDenseMatrix(Qw, "Q from qr_decompose(Matrix)");
    printDenseMatrix(Rw, "R from qr_decompose(Matrix)");
}

void demoQREigenvalues(const DenseMat& A) {
    std::cout << "===== QR eigenvalue iteration =====\n";

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-10;

    QRResult<Scalar> resDense = qr_eigenvalues_dense<Scalar>(A, opts);

    std::cout << "Dense QR converged : "
              << std::boolalpha << resDense.converged << "\n";
    std::cout << "Dense QR iterations: " << resDense.iterations << "\n";
    printVector(resDense.eigenvalues, "Eigenvalues from qr_eigenvalues_dense");

    Matrix M(A);
    QRResult<Scalar> resWrap = qr_eigenvalues<Scalar>(M, opts);

    std::cout << "Wrapper QR converged : "
              << std::boolalpha << resWrap.converged << "\n";
    std::cout << "Wrapper QR iterations: " << resWrap.iterations << "\n";
    printVector(resWrap.eigenvalues, "Eigenvalues from qr_eigenvalues(Matrix)");

    // Compare first eigenvalue from QR with power method using is_close_relative
    if (resWrap.eigenvalues.size() > 0) {
        double lambda_qr = resWrap.eigenvalues(resWrap.eigenvalues.size() - 1);
        double lambda_ref = resWrap.eigenvalues(0);

        bool close = is_close_relative(lambda_ref, lambda_qr, 1e-6);
        std::cout << "First and last eigenvalues close? "
                  << std::boolalpha << close << "\n\n";
    }
}

int main() {
    std::cout << std::setprecision(8);

    demoBox();

    // Base dense matrix for most demos
    DenseMat A(3, 3);
    A << 2.0, -1.0,  0.0,
        -1.0,  2.0, -1.0,
         0.0, -1.0,  2.0;

    // Wrapper instances used several times
    Matrix Md(A);

    SparseMat S(3, 3);
    S.insert(0, 0) =  2.0;
    S.insert(0, 1) = -1.0;
    S.insert(1, 0) = -1.0;
    S.insert(1, 1) =  2.0;
    S.insert(1, 2) = -1.0;
    S.insert(2, 1) = -1.0;
    S.insert(2, 2) =  2.0;
    S.makeCompressed();
    Matrix Ms(S);

    demoMatrixWrapper();
    demoSolveShiftedReal(Md, Ms);
    demoSolveShiftedComplex();
    demoPowerMethods(Md);
    demoHessenberg(A);
    demoQRDecompose(A);
    demoQREigenvalues(A);

    std::cout << "All demonstrations completed.\n";
    return 0;
}
