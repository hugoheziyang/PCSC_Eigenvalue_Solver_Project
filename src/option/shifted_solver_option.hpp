/**
 * @file shifted_solver_option.hpp
 * @brief Defines ShiftedSolverOptions an extension of SolverOptions
 *        that adds a spectral shift used by shifted inverse power
 *        and shifted QR methods.
 */

#pragma once

#include "solver_option.hpp"
#include "../core/types.hpp"

/**
 * @brief Options for solvers that use a spectral shift.
 *
 * This structure extends SolverOptions and gathers in a single
 * object both the general solver settings and the spectral shift
 * used by algorithms that operate on (A − σ I) or that iterate
 * around a shifted value σ to approach an eigenvalue located
 * near that point.
 *
 * It is designed for shifted inverse power methods shifted QR
 * iterations and any routine that benefits from a controllable
 * shift in the spectrum.
 *
 * @tparam Scalar Numeric type of the shift. Must satisfy ScalarConcept.
 */
template <typename Scalar>
struct ShiftedSolverOptions : public SolverOptions {
    Scalar shift;

    /**
     * @brief Default constructor.
     *
     * Initializes SolverOptions with its default values
     * and sets the shift to zero.
     */
    ShiftedSolverOptions():
        SolverOptions(),
        shift(Scalar(0))
    {}

    /**
     * @brief Constructor with an explicit shift.
     *
     * Other parameters inherited from SolverOptions remain unchanged.
     *
     * @param s The shift.
     */
    ShiftedSolverOptions(Scalar s):
        SolverOptions(),
        shift(s)
    {}

    /**
     * @brief Full constructor with shift maximum iterations and tolerance.
     *
     * @param s The shift.
     * @param maxIter Maximum number of iterations.
     * @param tol Convergence tolerance.
     */
    ShiftedSolverOptions(Scalar s, int maxIter, double tol) {
        shift = s;
        maxIterations = maxIter;
        tolerance = tol;
    }
};