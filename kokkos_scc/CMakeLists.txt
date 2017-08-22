cmake_minimum_required(VERSION 3.1)
project(kokkos_scc CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${KOKKOS_PATH} ${CMAKE_BINARY_DIR}/kokkos)

message("CMAKE_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID}")
#the XL compiler does not support standard detection
if( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

include_directories(${Kokkos_INCLUDE_DIRS_RET})
if(KOKKOS_ENABLE_CUDA)
  set(scc_main_tgt scc_main.gpu)
  add_definitions(-DGPU=1)
else()
  set(scc_main_tgt scc_main.host)
endif()
add_executable(${scc_main_tgt} scc_main.cpp)
set_property(TARGET ${scc_main_tgt} PROPERTY CXX_STANDARD_REQUIRED 11)
target_link_libraries(${scc_main_tgt} kokkos)
