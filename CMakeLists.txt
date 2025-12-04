cmake_minimum_required(VERSION 3.10)
project(PCSC_Eigenvalue_Solver_Project)

set(CMAKE_CXX_STANDARD 20)

enable_testing()   

# Add Eigen
add_library(eigen_interface INTERFACE)
target_include_directories(eigen_interface INTERFACE
    ${CMAKE_SOURCE_DIR}/eigen
)


add_library(PCSC_Eigenvalue_Solver_Project_Library
    src/power_method/shifted_inverse_power_solver.cpp
    src/qr_method/qr_factoriser.cpp
    src/qr_method/qr_method.cpp
)

target_include_directories(PCSC_Eigenvalue_Solver_Project_Library PUBLIC
    ${CMAKE_SOURCE_DIR}
)

# Link the library with Eigen
target_link_libraries(PCSC_Eigenvalue_Solver_Project_Library PUBLIC eigen_interface)
add_executable(main main.cpp)
# Give the executable access to Eigen
target_link_libraries(main PRIVATE eigen_interface)


add_subdirectory(test)