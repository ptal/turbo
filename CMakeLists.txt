cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(Turbo LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# A new configuration mode "LDebug" for light debug.

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES LDebug)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
    "Add a light debug configuration (LDebug)"
    FORCE)
endif()

# XCSP3 C++ Parser dependency

find_package(LibXml2 REQUIRED)

FetchContent_Declare(
    xcsp3parser
    GIT_REPOSITORY    "https://github.com/xcsp3team/XCSP3-CPP-Parser"
    GIT_TAG           45c3e8f2c3dac71261e3a7bc441b56715918ad56
)
FetchContent_MakeAvailable(xcsp3parser)

# Turbo executable

if(GPU)
    set_source_files_properties(
        src/solver.cpp src/memory.cpp src/propagators.cpp src/turbo.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(turbo src/solver.cpp src/memory.cpp src/propagators.cpp src/turbo.cpp)
target_include_directories(turbo PRIVATE include ${LIBXML2_INCLUDE_DIR})
if(GPU)
    set_property(TARGET turbo PROPERTY CUDA_ARCHITECTURES 75)
    set_property(TARGET turbo APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_compile_options(turbo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -rdc=true>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>
)

target_link_libraries(turbo PRIVATE xcsp3parser ${LIBXML2_LIBRARIES})
