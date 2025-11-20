#pragma once

#include <Eigen/Dense>

template <typename Scalar>
struct QRResult {

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eigenvalues;

    int  iterations = 0; 
    bool converged  = false;

    QREigenResult() = default;

    QREigenResult(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& eigvals, int iters, bool conv) :
        eigenvalues(eigvals),
        iterations(iters),
        converged(conv)
    {}
};