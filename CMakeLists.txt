cmake_minimum_required(VERSION 3.25)
# 3.25 minimum so that CPM's `SYSTEM YES` parameter actually propagates
# SYSTEM-include semantics to numsim-cas's headers. CMake 3.25 added the
# `SYSTEM` argument to `add_subdirectory()` and `FetchContent_Declare()`;
# CPM forwards it through (see cmake/CPM.cmake around lines 974 and 1163).
# On CMake 3.20-3.24 it would simply not be appended — no warning, no
# effect. CI uses ubuntu-24.04 (CMake 3.28+), so no cost to bumping.

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

# ─── Dependency manager: CPM ───────────────────────────────────────────────
#
# Vendored single-file CMake package manager (cmake/CPM.cmake). Wraps
# FetchContent with explicit SHA pinning, a global source cache (set
# CPM_SOURCE_CACHE env or -DCPM_SOURCE_CACHE=<path> to share downloads
# across builds), and source-availability detection.
#
# To bump CPM itself: download a newer release asset from
#   https://github.com/cpm-cmake/CPM.cmake/releases
# and overwrite cmake/CPM.cmake.

include(cmake/CPM.cmake)

# ─── numsim-cas dependency ──────────────────────────────────────────────────

if(NUMSIM_CODEGEN_USE_LOCAL_CAS)
    # Suppress numsim-cas's own tests/examples in the local-checkout path.
    # CPM scopes these via OPTIONS in the else-branch below.
    set(NUMSIM_CAS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_INSTALL_LIBRARY OFF CACHE BOOL "" FORCE)

    if(NOT NUMSIM_CAS_SOURCE_DIR)
        message(FATAL_ERROR "NUMSIM_CODEGEN_USE_LOCAL_CAS=ON requires NUMSIM_CAS_SOURCE_DIR to be set")
    endif()
    message(STATUS "Using local numsim-cas at ${NUMSIM_CAS_SOURCE_DIR}")
    add_subdirectory(${NUMSIM_CAS_SOURCE_DIR} numsim_cas_build EXCLUDE_FROM_ALL)
else()
    # Pin: bump this SHA deliberately. The previous `GIT_TAG main` made CI
    # non-reproducible — a breaking upstream commit would have flipped CI
    # red with no commit here. See issues #13, #21.
    #
    # GIT_SHALLOW is intentionally absent. CPM auto-disables shallow clones
    # when GIT_TAG is a commit hash (a SHA pin cannot be fetched shallowly
    # without server-side uploadpack.allowReachableSHA1InWant, which GitHub
    # does not guarantee). See `cpm_is_git_tag_commit_hash` in cmake/CPM.cmake.
    #
    # SYSTEM YES marks numsim-cas's include directories as system-includes
    # so its warnings don't surface in our builds (requires CMake 3.25+).
    #
    # EXCLUDE_FROM_ALL keeps numsim-cas's *own internal* targets (its tests,
    # examples, benchmark) off the default build. The NumSim_CAS library
    # itself is still pulled in transitively via the link dependency below
    # (target_link_libraries(numsim_codegen INTERFACE NumSim_CAS)).
    #
    # OPTIONS forwards the four NUMSIM_CAS_* flags as plain (non-cache)
    # variables into the subproject's directory scope; CPM forces
    # CMP0077 NEW, so these override numsim-cas's `option()` defaults
    # without touching the parent project's cache.
    CPMAddPackage(
        NAME numsim_cas
        GITHUB_REPOSITORY NumSim-Stack/numsim-cas
        GIT_TAG d9a9e8db574d4e3e1f56537b7702bd4fbb1fad44
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "NUMSIM_CAS_BUILD_TESTS OFF"
            "NUMSIM_CAS_BUILD_EXAMPLES OFF"
            "NUMSIM_CAS_BUILD_BENCHMARK OFF"
            "NUMSIM_CAS_INSTALL_LIBRARY OFF"
    )
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
