cmake_minimum_required(VERSION 3.18)
project(lightweight_ciphers_present_optimized LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(bench
  src/main.cu
)

target_include_directories(bench PRIVATE src)

set_target_properties(bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
  target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

target_compile_options(bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -maes -msse4.1 -mavx2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda -Xcompiler -maes,-msse4.1,-mavx2>
)
