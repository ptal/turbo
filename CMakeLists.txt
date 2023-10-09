cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(Turbo
  HOMEPAGE_URL "https://github.com/ptal/turbo"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" ON)

# Dependencies

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(lala_pc SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-pc/")
  FetchContent_Declare(lala_parsing SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-parsing/")
  FetchContent_Declare(lala_power SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-power/")
else()
  FetchContent_Declare(
    lala_pc
    GIT_REPOSITORY "https://github.com/lattice-land/lala-pc.git"
    GIT_TAG        v1.0.2
  )

  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.0.1
  )

  FetchContent_Declare(
    lala_power
    GIT_REPOSITORY "https://github.com/lattice-land/lala-power.git"
    GIT_TAG        v1.0.1
  )
endif()

FetchContent_MakeAvailable(lala_parsing lala_pc lala_power)

# Turbo executable

if(GPU)
  set_source_files_properties(src/turbo.cpp src/config.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(turbo src/turbo.cpp src/config.cpp)
target_include_directories(turbo PRIVATE include)
target_compile_options(turbo PRIVATE
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:-Xptxas --suppress-stack-size-warning>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--verbose;--keep>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Debug>>:-ftemplate-backtrace-limit=0>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:SHELL:-Xcuda-ptxas --suppress-stack-size-warning>"
)
target_link_libraries(turbo PRIVATE lala_parsing lala_pc lala_power)
