find_package(GTest REQUIRED)

# Small helper to avoid repetition
function(add_solver_test name)
    add_executable(${name}
        ${name}.cpp
    )
    target_link_libraries(${name}
        PRIVATE
            PCSC_Eigenvalue_Solver_Project_Library
            GTest::gtest_main
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_solver_test(matrix_wrapper_test)
add_solver_test(solve_shifted_test)
add_solver_test(power_method_test)
add_solver_test(shifted_inverse_power_method_test)
add_solver_test(qr_algorithms_test)
