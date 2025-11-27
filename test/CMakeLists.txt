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
