#pragma once

#include <Eigen/Dense>

template <typename Scalar>
struct EigenResult {

    Scalar eigenvalue;                        
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eigenvector; 

    int  iterations = 0;  
    bool converged  = false;

    EigenResult() = default;
    
    EigenResult(const Scalar& lambda, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& vec, int iters, bool conv) :
        eigenvalue(lambda),
        eigenvector(vec),
        iterations(iters),
        converged(conv)
    {}
};