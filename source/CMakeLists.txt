cmake_minimum_required(VERSION 3.20)
project(simple_ray_tracing)


set(CMAKE_CXX_STANDARD 20   )
set(CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)


function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CUDA_FLAGS}" "${flag}" flag_already_set)
    if (flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if (flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif ()
        unset(flag_supported CACHE)
    endif ()
endfunction()

enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
enable_cxx_compiler_flag_if_supported("-std=c++17")
enable_cxx_compiler_flag_if_supported("-O3")
enable_cxx_compiler_flag_if_supported("-fopenmp")


add_executable(ray_cast ray_casting.cpp)
include_directories(cpp_implementation)
add_subdirectory(cpp_implementation)
add_subdirectory(tests)

target_link_libraries(ray_cast misc_lib misc_interfaces scenes_lib camera_lib)

