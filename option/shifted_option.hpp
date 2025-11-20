#pragma once

template <typename Scalar>
struct ShiftedOptions {
    Scalar shift;
    ShiftedOptions(Scalar s = Scalar(0)) : shift(s) {}  // Initialize shift with default value 0
};