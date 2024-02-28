cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(Turbo
  HOMEPAGE_URL "https://github.com/ptal/turbo"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" OFF)
option(TURBO_PROFILE_MODE "TURBO_PROFILE_MODE" OFF)

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
    GIT_TAG        v1.1.3
  )

  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.1.3
  )

  FetchContent_Declare(
    lala_power
    GIT_REPOSITORY "https://github.com/lattice-land/lala-power.git"
    GIT_TAG        v1.1.3
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
if(TURBO_PROFILE_MODE)
  target_compile_options(turbo PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-lineinfo>")
  target_compile_definitions(turbo PRIVATE TURBO_PROFILE_MODE)
endif()

target_link_libraries(turbo PRIVATE lala_parsing lala_pc lala_power)

# Documentation

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "Turbo Constraint Solver")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)
set(DOXYGEN_EXCLUDE_SYMBOLS = impl,MAP_LIMITS,FLOAT_ARITHMETIC_CUDA_IMPL,FLOAT_ARITHMETIC_CPP_IMPL)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_turbo
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
