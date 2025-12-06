find_package(GTest REQUIRED)

# Test for Matrix wrapper
add_executable(matrix_wrapper_test
    matrix_wrapper_test.cpp
)

target_link_libraries(matrix_wrapper_test
    PRIVATE
        PCSC_Eigenvalue_Solver_Project_Library
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME matrix_wrapper_test COMMAND matrix_wrapper_test)

add_executable(solve_shifted_test
    solve_shifted_test.cpp
)

target_link_libraries(solve_shifted_test
    PRIVATE
        PCSC_Eigenvalue_Solver_Project_Library
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME solve_shifted_test COMMAND solve_shifted_test)

add_executable(qr_algorithms_test
    qr_algorithms_test.cpp
)

target_link_libraries(qr_algorithms_test
    PRIVATE
        PCSC_Eigenvalue_Solver_Project_Library
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME qr_algorithms_test COMMAND qr_algorithms_test)
