#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>

#include "../src/core/types.hpp"
#include "../src/matrix/matrix.hpp"
#include "../src/option/solver_option.hpp"
#include "../src/result/qr_result.hpp"
#include "../src/qr_method/to_hessenberg.hpp"
#include "../src/qr_method/qr_decompose.hpp"
#include "../src/qr_method/qr_eigenvalues.hpp"

// Small helper: compare two real matrices A,B with max |A-B| < tol
template <typename Scalar>
void expect_matrices_near(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
                          double tol = 1e-10)
{
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            using std::abs;
            EXPECT_NEAR(abs(A(i, j) - B(i, j)), 0.0, tol);
        }
    }
}

// --- 1. to_hessenberg_dense / to_hessenberg ---------------------------------

TEST(ToHessenbergDenseTest, Real3x3ProducesUpperHessenberg)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(3, 3);
    A << 4.0, 1.0, -2.0,
         1.0, 3.0,  0.0,
         2.0, 1.0,  1.0;

    Mat H = to_hessenberg_dense<Scalar>(A);

    // Check: Hessenberg -> zeros below first subdiagonal
    for (Eigen::Index i = 2; i < 3; ++i) {       // for 3x3, only i=2
        for (Eigen::Index j = 0; j < i - 1; ++j) {
            EXPECT_NEAR(H(i, j), 0.0, 1e-12);
        }
    }

    // Wrap A in Matrix and check wrapper-based version matches dense version
    Matrix A_wrapped(A);
    Mat H2 = to_hessenberg<Scalar>(A_wrapped);
    expect_matrices_near(H, H2);
}

TEST(ToHessenbergDenseTest, Complex3x3ProducesUpperHessenberg)
{
    using Scalar = std::complex<double>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(3, 3);
    A << Scalar(4.0, 1.0), Scalar(1.0, 0.0),  Scalar(-2.0, 2.0),
         Scalar(1.0, 0.0), Scalar(3.0, -1.0), Scalar(0.0, 1.0),
         Scalar(2.0, 0.0), Scalar(1.0, 2.0),  Scalar(1.0, 0.0);

    Mat H = to_hessenberg_dense<Scalar>(A);

    // Check Hessenberg structure
    for (Eigen::Index i = 2; i < 3; ++i) {
        for (Eigen::Index j = 0; j < i - 1; ++j) {
            using std::abs;
            EXPECT_NEAR(abs(H(i, j)), 0.0, 1e-12);
        }
    }

    // Wrapper-based version
    Matrix A_wrapped(A);
    Mat H2 = to_hessenberg<Scalar>(A_wrapped);
    expect_matrices_near(H, H2);
}

TEST(ToHessenbergDenseTest, ThrowsOnNonSquare)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(2, 3);
    A.setRandom();

    EXPECT_THROW(to_hessenberg_dense<Scalar>(A), std::runtime_error);
}

TEST(ToHessenbergDenseTest, EigenvaluesPreservedReal) {
    using Scalar = double;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(3, 3);
    A << 4.0, 1.0, -2.0,
         1.0, 3.0,  0.0,
         2.0, 1.0,  1.0;

    Mat H = to_hessenberg_dense<Scalar>(A);

    // Compute eigenvalues of A and H using EigenSolver
    Eigen::EigenSolver<Mat> eigA(A);
    Eigen::EigenSolver<Mat> eigH(H);

    auto evA = eigA.eigenvalues();
    auto evH = eigH.eigenvalues();

    ASSERT_EQ(evA.size(), evH.size());
    const int n = evA.size();

    // Copy to std::vector for sorting
    std::vector<std::complex<double>> vA(n), vH(n);
    for (int i = 0; i < n; ++i) {
        vA[i] = evA(i);
        vH[i] = evH(i);
    }

    auto cmp = [](const std::complex<double>& x,
                  const std::complex<double>& y) {
        if (x.real() < y.real()) return true;
        if (x.real() > y.real()) return false;
        return x.imag() < y.imag();
    };

    std::sort(vA.begin(), vA.end(), cmp);
    std::sort(vH.begin(), vH.end(), cmp);

    for (int i = 0; i < n; ++i) {
        using std::abs;
        EXPECT_NEAR(abs(vA[i] - vH[i]), 0.0, 1e-8);
    }
}

// --- 2. qr_decompose_dense / qr_decompose -----------------------------------

TEST(QRDecomposeDenseTest, RealRectangular3x2)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(3, 2);
    A << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    Mat Q, R;
    qr_decompose_dense<Scalar>(A, Q, R);

    // Check dimensions
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 2);

    // Check A ≈ Q*R
    Mat QR = Q * R;
    expect_matrices_near(A, QR, 1e-10);

    // Check R is (approximately) upper triangular
    for (Eigen::Index i = 0; i < R.rows(); ++i) {
        for (Eigen::Index j = 0; j < i && j < R.cols(); ++j) {
            EXPECT_NEAR(R(i, j), 0.0, 1e-10);
        }
    }

    // Check Q^T Q ≈ I
    Mat I = Mat::Identity(3, 3);
    Mat QtQ = Q.adjoint() * Q;
    expect_matrices_near(I, QtQ, 1e-10);

    // Wrapper version
    Matrix A_wrapped(A);
    auto [Q2, R2] = qr_decompose<Scalar>(A_wrapped);
    expect_matrices_near(Q, Q2, 1e-10);
    expect_matrices_near(R, R2, 1e-10);
}

TEST(QRDecomposeDenseTest, ComplexSquare2x2)
{
    using Scalar = std::complex<double>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(2, 2);
    A << Scalar(1.0, 1.0), Scalar(2.0, -1.0),
         Scalar(0.5, 0.0), Scalar(3.0, 2.0);

    Mat Q, R;
    qr_decompose_dense<Scalar>(A, Q, R);

    // Dimensions
    EXPECT_EQ(Q.rows(), 2);
    EXPECT_EQ(Q.cols(), 2);
    EXPECT_EQ(R.rows(), 2);
    EXPECT_EQ(R.cols(), 2);

    // A ≈ Q * R
    Mat QR = Q * R;
    expect_matrices_near(A, QR, 1e-10);

    // R approximately upper triangular (in magnitude)
    for (Eigen::Index i = 0; i < R.rows(); ++i) {
        for (Eigen::Index j = 0; j < i && j < R.cols(); ++j) {
            using std::abs;
            EXPECT_NEAR(abs(R(i, j)), 0.0, 1e-10);
        }
    }


    // Q* is unitary: Q^* Q ≈ I
    Mat I = Mat::Identity(2, 2);
    Mat QtQ = Q.adjoint() * Q;
    expect_matrices_near(I, QtQ, 1e-10);

    // Wrapper version
    Matrix A_wrapped(A);
    auto [Q2, R2] = qr_decompose<Scalar>(A_wrapped);
    expect_matrices_near(Q, Q2, 1e-10);
    expect_matrices_near(R, R2, 1e-10);
}

TEST(QRDecomposeDenseTest, ThrowsOnEmpty)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(0, 0);
    Mat Q, R;
    EXPECT_THROW(qr_decompose_dense<Scalar>(A, Q, R), std::runtime_error);
}

// --- 3. qr_eigenvalues_dense / qr_eigenvalues -------------------------------

TEST(QREigenvaluesDenseTest, Real2x2KnownEigenvalues)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Symmetric matrix with eigenvalues 3 and 1
    Mat A(2, 2);
    A << 2.0, 1.0,
         1.0, 2.0;

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-12;

    auto result = qr_eigenvalues_dense<Scalar>(A, opts);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.eigenvalues.size(), 2);

    EXPECT_GE(result.iterations, 1);
    EXPECT_LE(result.iterations, opts.maxIterations);

    double l1 = result.eigenvalues(0);
    double l2 = result.eigenvalues(1);

    double max_ev = std::max(l1, l2);
    double min_ev = std::min(l1, l2);

    EXPECT_NEAR(max_ev, 3.0, 1e-8);
    EXPECT_NEAR(min_ev, 1.0, 1e-8);

    // Wrapper version
    Matrix A_wrapped(A);
    auto result2 = qr_eigenvalues<Scalar>(A_wrapped, opts);

    EXPECT_EQ(result2.eigenvalues.size(), 2);

    EXPECT_GE(result2.iterations, 1);
    EXPECT_LE(result2.iterations, opts.maxIterations);

    // Sort both sets by value and compare
    std::array<double, 2> ev1{ result.eigenvalues(0), result.eigenvalues(1) };
    std::array<double, 2> ev2{ result2.eigenvalues(0), result2.eigenvalues(1) };
    std::sort(ev1.begin(), ev1.end());
    std::sort(ev2.begin(), ev2.end());

    EXPECT_NEAR(ev1[0], ev2[0], 1e-8);
    EXPECT_NEAR(ev1[1], ev2[1], 1e-8);
}

TEST(QREigenvaluesDenseTest, Complex2x2SameEigenvalues)
{
    using Scalar = std::complex<double>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Same real matrix, but stored as complex<double>
    Mat A(2, 2);
    A << Scalar(2.0, 0.0), Scalar(1.0, 0.0),
         Scalar(1.0, 0.0), Scalar(2.0, 0.0);

    SolverOptions opts;
    opts.maxIterations = 1000;
    opts.tolerance     = 1e-12;

    auto result = qr_eigenvalues_dense<Scalar>(A, opts);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.eigenvalues.size(), 2);

    EXPECT_GE(result.iterations, 1);
    EXPECT_LE(result.iterations, opts.maxIterations);

    // Eigenvalues should still be 3 and 1 (purely real).
    double l1 = result.eigenvalues(0).real();
    double l2 = result.eigenvalues(1).real();

    double max_ev = std::max(l1, l2);
    double min_ev = std::min(l1, l2);

    EXPECT_NEAR(max_ev, 3.0, 1e-8);
    EXPECT_NEAR(min_ev, 1.0, 1e-8);

    // Wrapper version
    Matrix A_wrapped(A);
    auto result2 = qr_eigenvalues<Scalar>(A_wrapped, opts);

    EXPECT_GE(result.iterations, 1);
    EXPECT_LE(result.iterations, opts.maxIterations);

    std::array<double, 2> ev1{ result.eigenvalues(0).real(), result.eigenvalues(1).real() };
    std::array<double, 2> ev2{ result2.eigenvalues(0).real(), result2.eigenvalues(1).real() };
    std::sort(ev1.begin(), ev1.end());
    std::sort(ev2.begin(), ev2.end());

    EXPECT_NEAR(ev1[0], ev2[0], 1e-8);
    EXPECT_NEAR(ev1[1], ev2[1], 1e-8);
}

TEST(QREigenvaluesDenseTest, ThrowsOnNonSquare)
{
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(2, 3);
    A.setRandom();

    SolverOptions opts;
    opts.maxIterations = 10;
    opts.tolerance     = 1e-6;

    EXPECT_THROW(qr_eigenvalues_dense<Scalar>(A, opts), std::runtime_error);
}
