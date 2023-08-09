cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(Turbo
  VERSION 1.0.0
  HOMEPAGE_URL "https://github.com/ptal/turbo"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

# Dependencies

include(FetchContent)

FetchContent_Declare(
  lala_parsing
  GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
  GIT_TAG        nvidia_bug4220100
)

FetchContent_Declare(
  lala_search
  GIT_REPOSITORY "https://github.com/lattice-land/lala-search.git"
  GIT_TAG        nvidia_bug4220100
)
FetchContent_MakeAvailable(lala_parsing lala_search)

# Turbo executable

if(GPU)
  set_source_files_properties(src/turbo.cpp src/config.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(turbo src/turbo.cpp src/config.cpp)
target_include_directories(turbo PRIVATE include)
target_compile_options(turbo PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ptxas-options --suppress-stack-size-warning>"
  "$<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-ftemplate-backtrace-limit=0;--verbose>;--verbose;--keep>"
  # "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -O0>"
)
target_link_libraries(turbo PRIVATE lala_parsing lala_search)
