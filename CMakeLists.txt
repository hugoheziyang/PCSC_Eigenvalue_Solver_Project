cmake_minimum_required(VERSION 3.10)
project(PCSC_Eigenvalue_Solver_Project)

set(CMAKE_CXX_STANDARD 20)


set(SOURCES
    main.cpp
)


add_executable(main ${SOURCES})