
find_package(GTest REQUIRED)

add_executable(first_test
    first_test.cpp
    add.cpp
)

target_link_libraries(first_test
    PRIVATE
        GTest::gtest
        GTest::gtest_main
)