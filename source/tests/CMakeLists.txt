# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.20)
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(google_test_lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})


# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(testfiles rays/test_light_source.cpp miscellaneous/test_image_buffer.cpp miscellaneous/test_c_vector.cpp
        rays/test_ray.cpp objects/test_sphere.cpp objects/mock_ray.h)

target_link_libraries(testfiles c_vector)
target_link_libraries(testfiles ray)
target_link_libraries(testfiles misc)
target_link_libraries(testfiles gtest gtest_main)
target_link_libraries(testfiles gmock gmock_main)