cmake_minimum_required(VERSION 3.20)

project(NumSim_CodeGen
    VERSION 0.1.0
    DESCRIPTION "Code generation from numsim-cas symbolic expressions for MOOSE constitutive models"
    HOMEPAGE_URL "https://github.com/NumSim-Stack/numsim-codegen"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── Options ────────────────────────────────────────────────────────────────

option(NUMSIM_CODEGEN_BUILD_TESTS "Build numsim-codegen tests" ON)
option(NUMSIM_CODEGEN_BUILD_EXAMPLES "Build numsim-codegen examples" ON)
option(NUMSIM_CODEGEN_USE_LOCAL_CAS "Use a local numsim-cas checkout (set NUMSIM_CAS_SOURCE_DIR)" OFF)

# ─── numsim-cas dependency ──────────────────────────────────────────────────

include(FetchContent)

# Suppress numsim-cas's own tests/examples — we only need the library.
set(NUMSIM_CAS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NUMSIM_CAS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NUMSIM_CAS_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NUMSIM_CAS_INSTALL_LIBRARY OFF CACHE BOOL "" FORCE)

if(NUMSIM_CODEGEN_USE_LOCAL_CAS)
    if(NOT NUMSIM_CAS_SOURCE_DIR)
        message(FATAL_ERROR "NUMSIM_CODEGEN_USE_LOCAL_CAS=ON requires NUMSIM_CAS_SOURCE_DIR to be set")
    endif()
    message(STATUS "Using local numsim-cas at ${NUMSIM_CAS_SOURCE_DIR}")
    add_subdirectory(${NUMSIM_CAS_SOURCE_DIR} numsim_cas_build EXCLUDE_FROM_ALL)
else()
    FetchContent_Declare(
        numsim_cas
        GIT_REPOSITORY https://github.com/NumSim-Stack/numsim-cas.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(numsim_cas)
endif()

# ─── Library (header-only) ──────────────────────────────────────────────────

add_library(numsim_codegen INTERFACE)
add_library(numsim::codegen ALIAS numsim_codegen)

target_include_directories(numsim_codegen
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(numsim_codegen INTERFACE NumSim_CAS)
target_compile_features(numsim_codegen INTERFACE cxx_std_23)

# ─── Tests ──────────────────────────────────────────────────────────────────

if(NUMSIM_CODEGEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Examples ───────────────────────────────────────────────────────────────

if(NUMSIM_CODEGEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ─── Install ────────────────────────────────────────────────────────────────
# Header-only install. CMake package config / export targets are deferred
# until numsim-cas itself exports installable targets (currently it does not
# expose NumSim_CAS in an export set, which blocks install(EXPORT) here).

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
