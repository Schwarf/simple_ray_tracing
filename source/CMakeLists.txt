cmake_minimum_required(VERSION 3.20)
project(simple_ray_tracing LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)
#   include(CheckCXXCompilerFlag)


#function(enable_cxx_compiler_flag_if_supported flag)
#    string(FIND "${CMAKE_CUDA_FLAGS}" "${flag}" flag_already_set)
#    if(flag_already_set EQUAL -1)
#        check_cxx_compiler_flag("${flag}" flag_supported)
#        if(flag_supported)
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
#        endif()
#        unset(flag_supported CACHE)
#    endif()
#endfunction()

#enable_cxx_compiler_flag_if_supported("-Wall")
#enable_cxx_compiler_flag_if_supported("-Wextra")
#enable_cxx_compiler_flag_if_supported("-pedantic")
#enable_cxx_compiler_flag_if_supported("-std=c++17")
#enable_cxx_compiler_flag_if_supported("-O3")
#enable_cxx_compiler_flag_if_supported("-fopenmp")

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -rdc=true -lcudart -lcudadevrt -lcuda)



#add_executable(ray_cast ray_casting.cpp)
# add_executable(cuda_sample cuda_playground/cuda_samples.cu)
add_executable(cuda_cast render.cu
        cuda_implementation/miscellaneous/templates/c_vector.cuh
        cuda_implementation/rays/interfaces/i_ray.cuh
        cuda_implementation/rays/ray.cuh
        cuda_implementation/rays/ray.cu)

#include_directories(cpp_implementation)
#add_subdirectory(cpp_implementation)
#add_subdirectory(cuda_implementation)
#add_subdirectory(cuda_playground)
#add_subdirectory(tests)



set_target_properties(cuda_cast PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#target_link_libraries(ray_cast materials_lib materials_interfaces ray_lib ray_interfaces misc_lib misc_interfaces objects_lib objects_interfaces)

