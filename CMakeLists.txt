cmake_minimum_required(VERSION 3.16)

project(PCSC_Eigenvalue_Solver_Project)

set(CMAKE_CXX_STANDARD 20)

include(CTest)

# INTERFACE target carries include dirs or dependencies no sources
add_library(eigen_interface INTERFACE)

target_include_directories(eigen_interface INTERFACE
    ${CMAKE_SOURCE_DIR}/eigen
)

# ----------------------------------------------------------------------
# Current state simple header-only project
# ----------------------------------------------------------------------
add_library(PCSC_Eigenvalue_Solver_Project_Library INTERFACE)

target_include_directories(PCSC_Eigenvalue_Solver_Project_Library INTERFACE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(PCSC_Eigenvalue_Solver_Project_Library INTERFACE
    eigen_interface
)

# ----------------------------------------------------------------------
# Future version if .cpp files must be compiled
# ----------------------------------------------------------------------
#
# add_library(PCSC_Eigenvalue_Solver_Project_Library
#     .cpp here
# )
#
# target_include_directories(PCSC_Eigenvalue_Solver_Project_Library
#     PUBLIC
#         ${CMAKE_SOURCE_DIR}
# )
#
# target_link_libraries(PCSC_Eigenvalue_Solver_Project_Library
#     PUBLIC
#         eigen_interface
# )
#
# ----------------------------------------------------------------------

add_executable(main main.cpp)

target_link_libraries(main PRIVATE
    PCSC_Eigenvalue_Solver_Project_Library
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
