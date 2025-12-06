#pragma once

/**
 * @file solver_option.hpp
 * @brief Defines generic solver parameters shared by multiple algorithms.
 */

/**
 * @struct SolverOptions
 * @brief Basic configuration parameters for iterative eigenvalue algorithms.
 *
 * These options control numerical convergence behavior for solvers such as:
 *   - Power method
 *   - Inverse power method
 *   - QR algorithm (iterative version)
 */
struct SolverOptions {
    /// Maximum number of allowed iterations (default: 1000).
    int maxIterations = 1000;

    /// Convergence tolerance for stopping criteria (default: 1e-10).
    double tolerance = 1e-10;
};
