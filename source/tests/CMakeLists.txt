# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.20)
project(Google_tests)

# 'google_test_lib' is the folder with Google Test sources
add_subdirectory(google_test_lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})


# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(all_tests test_main.cpp rays/test_light_source.cpp miscellaneous/test_image_buffer.cpp miscellaneous/test_n_tuple.cpp
        rays/test_ray.cpp objects/test_sphere.cpp objects/mock_ray.h miscellaneous/test_validate.cpp
        materials/test_material.cpp materials/mock_material_builder.h miscellaneous/test_quadratic_equation.cpp
        miscellaneous/test_cubic_equation.cpp miscellaneous/test_quartic_equation.cpp)

target_link_libraries(all_tests objects_lib)
target_link_libraries(all_tests materials_lib)
target_link_libraries(all_tests ray_lib)
target_link_libraries(all_tests misc_lib)
target_link_libraries(all_tests gtest gtest_main)
target_link_libraries(all_tests gmock gmock_main)