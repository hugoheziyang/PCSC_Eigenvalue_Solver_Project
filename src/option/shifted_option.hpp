#pragma once

#include "../core/types.hpp"

/**
 * @file ShiftedOptions.hpp
 * @brief Defines options for shifted power and inverse power methods.
 */

/**
 * @struct ShiftedOptions
 * @brief Options for solvers that use a spectral shift (e.g. shifted inverse power).
 *
 * @tparam Scalar  Numeric type (e.g. double, float, std::complex).
 *
 * This struct stores the value of the shift \f$\sigma\f$ used in:
 *   - Shifted power method:      \f$ A - \sigma I \f$
 *   - Shifted inverse iteration: \f$ (A - \sigma I)^{-1} \f$
 */
template <ScalarConcept Scalar>
struct ShiftedOptions {
    /// Value of the shift \f$\sigma\f$.
    Scalar shift;

    /// Constructor (defaults to shift = 0).
    ShiftedOptions(Scalar s = Scalar(0))
        : shift(s)
    {}
};