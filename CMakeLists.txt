cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(Turbo LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
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
    GIT_TAG           50ee7bfa59327c31a0a66ca4072a23cd08d0b182
)
FetchContent_MakeAvailable(xcsp3parser)

# Lattice land dependencies

FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(cuda_battery)

FetchContent_Declare(
  lala_core
  GIT_REPOSITORY "https://github.com/lattice-land/lala-core.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(lala_core)

FetchContent_Declare(
  lala_parsing
  GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(lala_parsing)

FetchContent_Declare(
  lala_ipc
  GIT_REPOSITORY "https://github.com/lattice-land/lala-ipc.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(lala_ipc)

FetchContent_Declare(
  lala_search
  GIT_REPOSITORY "https://github.com/lattice-land/lala-search.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(lala_search)

# Thrust dependency

FetchContent_Declare(
  thrust
  GIT_REPOSITORY "https://github.com/NVIDIA/thrust.git"
  GIT_TAG        1.12.0
)
FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(thrust)
endif()

# Turbo executable

if(GPU)
    set_source_files_properties(
        src/turbo.cpp src/config.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(turbo src/turbo.cpp src/config.cpp)
target_include_directories(turbo PRIVATE include ${LIBXML2_INCLUDE_DIR} ${lala_parsing_SOURCE_DIR}/include ${lala_ipc_SOURCE_DIR}/include ${lala_search_SOURCE_DIR}/include ${xcsp3parser_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_core_SOURCE_DIR}/include ${thrust_SOURCE_DIR})
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
