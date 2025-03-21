cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(MSVC)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Available configuration types to select")
endif()

option(GPU "GPU" ON)
if(GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Targeted CUDA architectures")
  endif()
  project(Turbo
    HOMEPAGE_URL "https://github.com/ptal/turbo"
    LANGUAGES CUDA CXX)
else()
  project(Turbo
    HOMEPAGE_URL "https://github.com/ptal/turbo"
    LANGUAGES CXX)
endif()
option(LOCAL_DEPS "LOCAL_DEPS" OFF)
option(TURBO_VERBOSE "Compile with verbose output" ON)
option(WITH_ASAN "Compile with the Address Sanitizer to check for memory corruption errors" OFF)
option(WITH_XCSP3PARSER "Add support for parsing XCSP3 .xml files" ON)
option(NO_CONCURRENT_MANAGED_MEMORY "Add support for platform not supporting concurrent managed access to memory on GPUs (use pinned memory instead)." OFF)
option(TURBO_BUILD_DOC "Build the documentation of Turbo." OFF)
option(TURBO_IPC_ABSTRACT_DOMAIN "Use the interval propagator completion (PC) abstract domain instead of PIR." OFF)

if(MSVC)
  set(WITH_XCSP3PARSER OFF)  # XCSP3-CPP-Parser has dependencies on gcc that break MSVC
  set(NO_CONCURRENT_MANAGED_MEMORY ON)  # Windows requires a special allocator to provide concurrent access to managed memory
endif()

set(TURBO_INITLIB "" CACHE STRING "Include an external initialization library to construct statics before main() for use in code instrumentation")

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
    GIT_TAG        v1.2.7
  )

  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.2.7
  )

  FetchContent_Declare(
    lala_power
    GIT_REPOSITORY "https://github.com/lattice-land/lala-power.git"
    GIT_TAG        v1.2.7
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
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--keep$<$<BOOL:${TURBO_VERBOSE}>:;--verbose>>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Debug>>:-ftemplate-backtrace-limit=0>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:SHELL:-Xcuda-ptxas --suppress-stack-size-warning>"
    # MSVC requires /bigobj. See https://learn.microsoft.com/en-us/cpp/build/reference/bigobj-increase-number-of-sections-in-dot-obj-file
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:SHELL:/bigobj /W3>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:SHELL:--compiler-options /bigobj>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:SHELL:--compiler-options /W3>"
    # Address Sanitizer
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:--compiler-options -fsanitize=address>"
)
target_compile_definitions(turbo PRIVATE DISABLE_FULL_GPU_SOLVING)
target_compile_options(turbo PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-lineinfo>")

target_compile_definitions(turbo PRIVATE CUDA_THREADS_PER_BLOCK=256)
target_compile_definitions(turbo PRIVATE TURBO_ITV_BITS=32)

if(TURBO_IPC_ABSTRACT_DOMAIN)
  target_compile_definitions(turbo PRIVATE TURBO_IPC_ABSTRACT_DOMAIN)
endif()

if(WITH_XCSP3PARSER)
  target_compile_definitions(turbo PRIVATE WITH_XCSP3PARSER)
endif()

if(NO_CONCURRENT_MANAGED_MEMORY)
  target_compile_definitions(turbo PRIVATE NO_CONCURRENT_MANAGED_MEMORY)
endif()

target_link_libraries(turbo PRIVATE lala_parsing lala_pc lala_power)
target_link_options(turbo PRIVATE
    $<$<AND:$<BOOL:${WITH_ASAN}>,$<CXX_COMPILER_ID:GNU>>:-fsanitize=address;-static-libasan>
    $<$<AND:$<BOOL:${WITH_ASAN}>,$<CXX_COMPILER_ID:Clang>>:-fsanitize=address>
    $<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/INCREMENTAL:NO>
    $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT /STACK:8388608>
    # Link an optional initialization library ${TURBO_INITLIB}
    $<$<AND:$<BOOL:${TURBO_INITLIB}>,$<CXX_COMPILER_ID:GNU>>:SHELL:--whole-archive -l${TURBO_INITLIB} --no-whole-archive>
    $<$<AND:$<BOOL:${TURBO_INITLIB}>,$<CXX_COMPILER_ID:Clang>>:SHELL:-force_load ${TURBO_INITLIB}>
    $<$<AND:$<BOOL:${TURBO_INITLIB}>,$<CXX_COMPILER_ID:MSVC>>:SHELL:/WHOLEARCHIVE:${TURBO_INITLIB}>
)

if(MSVC)
  set_target_properties(turbo PROPERTIES VS_USER_PROPS "${CMAKE_CURRENT_SOURCE_DIR}/vs.props")
endif()

# Documentation
if(TURBO_BUILD_DOC)

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
endif()
