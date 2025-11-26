find_package(GTest REQUIRED)

# Test for EigenMatrixAdapter
add_executable(eigen_matrix_adapter_test
    eigen_matrix_adapter_test.cpp
)
target_link_libraries(eigen_matrix_adapter_test
    PRIVATE
        PCSC_Eigenvalue_Solver_Project_Library
        GTest::gtest
        GTest::gtest_main
)


