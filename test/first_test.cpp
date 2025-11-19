# include <gtest/gtest.h>
#include "add.hpp"

TEST(addTest, positive_integers) {
    EXPECT_EQ(6, add(4, 2));
    EXPECT_EQ(3, add(1, 2));
}

TEST(addTest, zero_input) {
    EXPECT_THROW(add(0, 5), std::invalid_argument);
    EXPECT_THROW(add(0, -3), std::invalid_argument);
}


