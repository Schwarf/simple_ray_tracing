# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.20)
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(google_test_lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(testfiles test_light_source.cpp test_image_buffer.cpp test_c_vector.cpp)

target_link_libraries(testfiles c_vector)
target_link_libraries(testfiles gtest gtest_main)