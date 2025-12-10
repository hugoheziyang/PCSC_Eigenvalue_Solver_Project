#pragma once

namespace EigSol {

/**
 * @file solver_option.hpp
 * @brief Defines generic solver parameters shared by multiple algorithms.
 */

/**
 * @struct SolverOptions
 * @brief Basic configuration parameters for iterative eigenvalue algorithms.
 */
struct SolverOptions {
    /// Maximum number of allowed iterations.
    int maxIterations = 1000;

    /// Convergence relative tolerance for stopping criteria.
    double tolerance = 1e-10;
};

} // end namespace